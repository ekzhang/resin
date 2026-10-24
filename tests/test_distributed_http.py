"""Real Gloo ranks behind one HTTP listener, including empty participants."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import resin
from resin import Parallelism, RunnerConfig
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.metrics import RunnerMetrics
from resin.backend.sampling import format_output, prompt_scores
from resin.backend.topology import ParallelGroups, ParallelismCapabilities
from resin.server import HttpServer


class RankCodec:
    def __init__(self, rank, case):
        if case == "initialize" and rank == 1:
            raise RuntimeError("injected codec initialization")

    def render(self, messages, *, tools=(), options=None):
        content = messages[0].content[0].text
        if content == "bad-render":
            raise ValueError("injected request rendering")
        return resin.recipe.RenderedPrompt(
            [99]
            if content == "explode"
            else ([4] if content == "cancelme" else [1, 2, 3]),
            stop_token_ids=[],
        )

    def create_parser(self, prompt):
        class Parser:
            def feed(self, ids):
                return [
                    resin.recipe.TextDelta("".join(str(token - 10) for token in ids))
                ]

        return Parser()


class RankRunner:
    runner_config = RunnerConfig(continuation_prefill=True)
    parallelism_capabilities = ParallelismCapabilities(
        attn_dp=True, attn_tp=True, ep=True, expert_tp=True
    )

    def __init__(self, groups, directory, case):
        self.groups = groups
        self.directory = directory
        self.case = case
        self.metrics_enabled = False
        self.calls = []
        self.released = []
        self.cancel_requests = set()

    def configure_metrics(self, *, enabled=True):
        self.metrics_enabled = enabled

    def metrics_snapshot(self):
        assert self.metrics_enabled
        if self.case == "metrics_failure" and self.groups.rank == 2:
            raise RuntimeError("injected metrics snapshot failure")
        return RunnerMetrics(
            # Synthetic rank-specific observations prove that physical ranks
            # publish independently of logical request ownership.
            forward_execution_seconds={"decode": float(self.groups.rank + 1)},
            cuda_graph_passes={"decode_none": len(self.calls)},
        )

    def release(self, ids):
        self.released.extend(map(int, ids))
        if self.cancel_requests.intersection(map(int, ids)):
            Path(self.directory, f"canceled-{self.groups.rank}").touch()

    def forward_distributed(self, plan, context):
        # Every replica and follower participates, even when another replica
        # owns all requests. A rank-stamped output proves actual placement.
        marker = torch.tensor(1)
        dist.all_reduce(marker, group=self.groups.device_domain.handle)
        self.calls.append(plan is None)
        if plan is None:
            return None
        for row, (start, end) in enumerate(
            zip(plan.token_indptr[:-1], plan.token_indptr[1:], strict=True)
        ):
            if 4 in plan.token_ids[start:end] and plan.context_lengths[row] == 0:
                self.cancel_requests.add(int(plan.sequence_ids[row]))
        if 99 in plan.token_ids:
            raise RuntimeError("injected remote forward failure")
        replica = self.groups.topology.coordinates(self.groups.rank)[0]
        logits = torch.zeros(plan.num_sequences, 32)
        logits[:, 10 + replica] = 2
        all_logits = torch.zeros(len(plan.token_ids), 32)
        return format_output(
            logits,
            logits.argmax(-1),
            plan,
            prompt_logprobs=prompt_scores(plan, all_logits),
        )


def _serve_rank(rank, world_size, directory, port, case):
    patch = pytest.MonkeyPatch()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=25),
    )
    groups = ParallelGroups.create(
        Parallelism(world_size=world_size, attn_dp=2, attn_tp=world_size // 2, ep=2),
        device_backend="gloo",
        timeout=timedelta(seconds=25),
    )
    runner = RankRunner(groups, directory, case)
    worker = DistributedWorker(runner, groups)
    engine = None
    error = None
    try:
        if not groups.is_driver:
            worker.serve()
        else:
            assert groups.drivers is not None
            engine = DistributedEngine(
                worker,
                context_length=2048,
                enable_metrics=case in ("normal", "metrics_failure"),
            )
            codec = RankCodec(rank, "normal")
            if case == "initialize" and groups.drivers.local_rank == 1:
                # Native session initialization snapshots an optional tokenizer.
                class BrokenCodec(RankCodec):
                    @property
                    def tokenizer(self):
                        raise RuntimeError("injected codec initialization")

                codec = BrokenCodec(rank, "normal")
            model = "rank-test"
            if case == "whitespace" and groups.drivers.local_rank == 0:
                model = " "
            if case == "infer" and groups.drivers.local_rank == 1:

                class InferenceFailureCodec(RankCodec):
                    @property
                    def tokenizer(self):
                        raise RuntimeError("injected model inference")

                codec = InferenceFailureCodec(rank, "normal")
                model = None
            if case == "frontend" and groups.drivers.local_rank == 0:
                import resin.server as serving

                def fail(*args, **kwargs):
                    raise RuntimeError("injected frontend construction")

                setattr(serving, "_HttpServer", fail)
            server = HttpServer(
                engine,
                recipe=codec,
                model=model,
                enable_openai=not (case == "flags" and groups.drivers.local_rank == 0),
                request_queue_capacity=(
                    2
                    if case == "request_queue_capacity"
                    and groups.drivers.local_rank == 1
                    else 0
                    if case == "invalid_request_queue_capacity"
                    and groups.drivers.local_rank == 1
                    else 1024
                ),
                stream_interval=(
                    3
                    if case == "normal"
                    else 2
                    if case == "stream_interval" and groups.drivers.local_rank == 1
                    else 0
                    if case == "invalid_stream_interval"
                    and groups.drivers.local_rank == 1
                    else 1
                ),
            )
            if case == "normal":

                def unexpected_object_collective(*args, **kwargs):
                    raise AssertionError(
                        "serving rounds must use typed buffer collectives"
                    )

                patch.setattr(
                    dist, "broadcast_object_list", unexpected_object_collective
                )
                patch.setattr(dist, "all_gather_object", unexpected_object_collective)
            if (
                case in ("ingress_decode", "ingress_copy")
                and groups.drivers.local_rank == 1
            ):
                assert server._coordinator is not None
                wire = server._coordinator._wire
                broadcast = wire.broadcast

                def malformed_ingress(payload):
                    encoded = broadcast(payload)
                    if case == "ingress_copy" and encoded:
                        raise MemoryError("injected HTTP receive copy failure")
                    return b"invalid packet" if encoded else encoded

                patch.setattr(wire, "broadcast", malformed_ingress)
            if case == "event_encode" and groups.drivers.local_rank == 1:
                assert server._coordinator is not None
                session = server._coordinator.session
                assert session is not None

                def fail_encoding():
                    raise MemoryError("injected HTTP event encoding failure")

                def process(tokens):
                    packet = session.process(tokens)
                    return (
                        SimpleNamespace(to_bytes=fail_encoding)
                        if packet.to_bytes()
                        else packet
                    )

                patch.setattr(
                    server._coordinator,
                    "session",
                    SimpleNamespace(
                        admit=session.admit, close=session.close, process=process
                    ),
                )
            if groups.drivers.local_rank == 0:

                def stop():
                    while not Path(directory, "stop").exists():
                        time.sleep(0.01)
                    server.shutdown()

                threading.Thread(target=stop, daemon=True).start()
            server.serve(port=port)
    except BaseException as exc:
        error = str(exc)
    finally:
        patch.undo()
        Path(directory, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "error": error,
                    "closed": worker.closed,
                    "calls": runner.calls,
                    "released": runner.released,
                    "engine_closed": None if engine is None else engine.closed,
                }
            )
        )
        groups.close()
        dist.destroy_process_group()


def _post(base, rank=None, *, content="hello", headers=None, **options):
    body = {
        "model": "rank-test",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2,
        **options,
    }
    if rank is not None:
        body["routed_dp_rank"] = rank
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        return urllib.request.urlopen(request, timeout=15)
    except urllib.error.HTTPError as error:
        error.add_note(error.read().decode())
        raise


def _wait_ready(base, directory):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=0.2):
                return
        except (OSError, urllib.error.URLError):
            failed = list(Path(directory).glob("rank-*.json"))
            if failed:
                raise AssertionError([path.read_text() for path in failed])
            time.sleep(0.02)
    raise AssertionError("distributed HTTP listener did not start")


def _metric_rows(text, name):
    return [
        (dict(re.findall(r'(\w+)="([^"]*)"', line)), float(line.rsplit(" ", 1)[1]))
        for line in text.splitlines()
        if line.startswith(name + "{")
    ]


def _assert_distributed_metrics(base):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with urllib.request.urlopen(f"{base}/metrics", timeout=1) as response:
            text = response.read().decode()
        requests = _metric_rows(text, "resin:num_requests_total")
        if sum(value for _, value in requests) == 8:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("distributed request metrics did not publish")
    assert {int(labels["rank"]): value for labels, value in requests} == {0: 2, 2: 6}
    assert {
        int(labels["attn_dp_rank"]): value
        for labels, value in _metric_rows(text, "resin:generation_tokens_total")
    } == {0: 4, 1: 13}
    assert {
        int(labels["rank"]): value
        for labels, value in _metric_rows(text, "resin:forward_execution_seconds_total")
    } == {0: 1, 1: 2, 2: 3, 3: 4}
    assert len(_metric_rows(text, "resin:cuda_graph_passes_total")) == 4
    assert text.count("# HELP resin:forward_execution_seconds_total ") == 1
    assert text.count("# TYPE resin:num_requests_total ") == 1
    assert _metric_rows(text, "resin:http_time_to_first_token_seconds_count")[0][1] == 7
    assert _metric_rows(text, "resin:http_request_latency_seconds_count")[0][1] == 8


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize(
    "case,world_size",
    [
        ("normal", 4),
        ("metrics_failure", 4),
        ("fatal", 2),
        ("ingress_decode", 2),
        ("ingress_copy", 2),
        ("event_encode", 2),
        ("initialize", 2),
        ("bind", 2),
        ("whitespace", 2),
        ("flags", 2),
        ("infer", 2),
        ("frontend", 2),
        ("stream_interval", 2),
        ("invalid_stream_interval", 2),
        ("request_queue_capacity", 2),
        ("invalid_request_queue_capacity", 2),
    ],
)
def test_distributed_http_routes_real_replica_engines(tmp_path, case, world_size):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    if case != "bind":
        probe.close()
    else:
        probe.listen()
    context = mp.spawn(
        _serve_rank,
        args=(world_size, str(tmp_path), port, case),
        nprocs=world_size,
        join=False,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        if case in (
            "normal",
            "fatal",
            "ingress_decode",
            "ingress_copy",
            "event_encode",
        ):
            _wait_ready(base, tmp_path)
        if case == "fatal":
            with pytest.raises(urllib.error.HTTPError) as failure:
                _post(base, 1, content="explode")
            assert failure.value.code == 500
        elif case in ("ingress_decode", "ingress_copy", "event_encode"):
            with pytest.raises(urllib.error.HTTPError) as failure:
                _post(base, 1)
            assert failure.value.code == 500
        elif case == "normal":
            for rank in (0, 1):
                with _post(base) as response:
                    assert (
                        json.load(response)["choices"][0]["message"]["content"]
                        == str(rank) * 2
                    )
            with _post(base, 1, logprobs=True, top_logprobs=2) as response:
                output = json.load(response)
            assert output["choices"][0]["message"]["content"] == "11"
            # Custom parsers cannot prove a token's semantic channel; raw
            # scores still cross replicas, while content-only scores stay empty.
            assert output["choices"][0]["logprobs"]["content"] == []
            assert len(output["resin"]["generation_logprobs"]) == 2
            with _post(base, 1, stream=True, max_tokens=5, logprobs=True) as response:
                chunks = [
                    json.loads(line[6:])
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
            assert [
                content
                for chunk in chunks
                for choice in chunk["choices"]
                if (content := choice["delta"].get("content"))
            ] == ["1", "111", "1"]
            assert [
                score["token_index"]
                for chunk in chunks
                for score in chunk.get("resin", {}).get("generation_logprobs", [])
            ] == list(range(5))
            with _post(base, headers={"resin-routed-dp-rank": "1"}) as response:
                assert json.load(response)["choices"][0]["message"]["content"] == "11"
            with _post(base, 999, headers={"resin-routed-dp-rank": "1"}) as response:
                assert json.load(response)["choices"][0]["message"]["content"] == "11"
            with pytest.raises(urllib.error.HTTPError) as failure:
                _post(base, 0, headers={"resin-routed-dp-rank": "bad"})
            assert failure.value.code == 400
            with _post(base, 1, max_tokens=0, prompt_logprobs=True) as response:
                output = json.load(response)
            assert output["choices"][0]["message"]["content"] == ""
            for rank in (2, -1):
                with pytest.raises(urllib.error.HTTPError) as failure:
                    _post(base, rank)
                assert failure.value.code == 400
            with pytest.raises(urllib.error.HTTPError):
                _post(base, 1, content="bad-render")
            with _post(base, 0) as response:
                assert json.load(response)["choices"][0]["message"]["content"] == "00"
            _assert_distributed_metrics(base)
            # Drop a streaming client while rank one remains active.
            with _post(
                base, 1, content="cancelme", stream=True, max_tokens=1000
            ) as response:
                response.readline()
            deadline = time.monotonic() + 5
            while not Path(tmp_path, f"canceled-{world_size // 2}").exists():
                assert time.monotonic() < deadline, (
                    "remote cancellation did not release its lease"
                )
                time.sleep(0.01)
        elif case == "metrics_failure":
            deadline = time.monotonic() + 30
            while len(list(tmp_path.glob("rank-*.json"))) < world_size:
                assert time.monotonic() < deadline, "metrics failure stranded a rank"
                time.sleep(0.05)
    finally:
        Path(tmp_path, "stop").touch()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if context.join(timeout=0.2):
                break
        else:
            probe.close()
            for process in context.processes:
                process.kill()
            raise AssertionError(
                "distributed HTTP shutdown left a driver/follower running"
            )
        probe.close()
    records = [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text())
        for rank in range(world_size)
    ]
    assert all(record["closed"] for record in records)
    if case == "normal":
        assert all(record["error"] is None for record in records)
        assert all(record["calls"] for record in records)
        assert any(any(record["calls"]) for record in records)
        assert all(len(record["calls"]) < 1000 for record in records)
    elif case == "metrics_failure":
        assert all(
            "injected metrics snapshot failure" in records[rank]["error"]
            for rank in (0, 2)
        )
        assert all(records[rank]["error"] is None for rank in (1, 3))
        assert all(
            record["engine_closed"]
            for record in records
            if record["engine_closed"] is not None
        )
    elif case == "bind":
        assert "bind" in records[0]["error"]
        assert records[1]["error"] is None
    elif case in ("ingress_decode", "ingress_copy"):
        assert all("HTTP ingress failed" in record["error"] for record in records)
        assert all(not record["calls"] for record in records)
        assert all(record["engine_closed"] for record in records)
    elif case == "event_encode":
        assert all(
            "HTTP event encoding failure" in record["error"] for record in records
        )
        assert all(record["engine_closed"] for record in records)
    elif case in (
        "whitespace",
        "flags",
        "infer",
        "frontend",
        "initialize",
        "stream_interval",
        "invalid_stream_interval",
        "request_queue_capacity",
        "invalid_request_queue_capacity",
    ):
        assert len({record["error"] for record in records}) == 1
        assert records[0]["error"] is not None
        assert all(record["engine_closed"] for record in records)
        if case in ("stream_interval", "request_queue_capacity"):
            assert "configuration must agree" in records[0]["error"]
        elif case.startswith("invalid_"):
            setting = case.removeprefix("invalid_")
            assert f"{setting} must be a positive integer" in records[0]["error"]
    else:
        assert all("injected" in record["error"] for record in records)
