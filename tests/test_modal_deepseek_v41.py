"""The EP/ADP endpoint verifies single-replica overlap before shared HTTP serving."""

from inspect import signature
from types import SimpleNamespace

import pytest
import tokenizers
import torch
import torch.distributed as dist

import resin
from deploy import modal_deepseek_v41 as deployment
from resin import Engine
from resin.backend import distributed
from resin.backend.topology import ParallelGroups
from resin.recipe import DeepSeekV41Recipe
from resin.server import HttpServer
from resin_models_cuda.deepseek_v4.cuda import DeepSeekV4CudaRunner
from resin_models_cuda.deepseek_v41 import cuda as runner_module


@pytest.mark.parametrize("rank,fail", [(0, False), (3, False), (0, True)])
def test_rank_startup_warms_then_joins_distributed_http(monkeypatch, rank, fail):
    calls = []
    groups = SimpleNamespace(rank=rank)
    render = DeepSeekV41Recipe.render_ids
    stops = DeepSeekV41Recipe.get_stop_token_ids

    def initialize(backend, **options):
        assert backend == "nccl"
        assert options["rank"] == rank and options["world_size"] == 4
        assert options["device_id"] == torch.device("cuda", rank)
        calls.append("initialize")

    def make_groups(topology):
        assert (topology.world_size, topology.attn_dp, topology.ep) == (4, 4, 4)
        assert topology.attn_tp == topology.cp == topology.expert_tp == 1
        calls.append("groups")
        return groups

    class Runner:
        eos_token_id = 1
        cuda_graph_stats = {"prefill": {"capture_count": 1}}

        def __init__(self, model, **options):
            assert options.pop("expert_backend") == "deepgemm"
            # Validate actual runner options rather than having a permissive
            # mock conceal unsupported CUDA constructor parameters.
            signature(DeepSeekV4CudaRunner.__init__).bind(self, model, **options)
            assert model == deployment.CHECKPOINT
            assert options["parallel_groups"] is groups
            assert options["device"] == f"cuda:{rank}"
            assert options["speculative_tokens"] == 3
            assert options["max_batch_size"] == 32
            assert options["max_seq_len"] == 1_048_576
            assert options["prefill_chunk_size"] == 8192
            calls.append("runner")

        def capture_cuda_graphs(self):
            calls.append("capture")

    class Codec:
        def __init__(self, tokenizer):
            assert tokenizer == "tokenizer"

        def render_ids(self, messages):
            signature(render).bind(self, messages)
            assert messages[0].role == resin.recipe.Role.USER
            return [4, 5]

        def get_stop_token_ids(self):
            signature(stops).bind(self)
            return [7, 8]

    class Worker:
        def __init__(self, runner, rank_groups):
            assert isinstance(runner, Runner) and rank_groups is groups

    class RequestEngine:
        closed = False
        local = SimpleNamespace(overlap_stats=(8, 6))

        def __init__(self, worker, **options):
            signature(Engine).bind(worker, **options)
            assert isinstance(worker, Worker)
            assert options["enable_overlap"] and options["prefix_cache"]
            assert options["context_length"] == 1_048_576
            assert options["max_running_requests"] == 32
            scheduler = options["scheduler_config"]
            assert scheduler.prefill_chunk_size == 8192
            assert scheduler.max_batch_tokens == 32 * 8192
            self.remaining = 2
            calls.append("engine")

        def submit(self, prompt, **options):
            signature(Engine.submit).bind(self, prompt, **options)
            assert prompt == [4, 5] and options["sampling_params"].stop_token_ids == (
                7,
                8,
            )
            assert options["sampling_params"].temperature == 0
            assert options["sampling_params"].max_output_tokens == 64 and rank == 0
            calls.append("submit")
            return SimpleNamespace(take_result=lambda: calls.append("warmup_done"))

        def has_work(self):
            return self.remaining > 0

        def step(self):
            self.remaining -= 1
            calls.append("step")

        def close(self):
            assert not self.closed
            self.closed = True
            calls.append("close")

    class Server:
        def __init__(self, engine, *, recipe, model):
            signature(HttpServer.__init__).bind(
                self, engine, recipe=recipe, model=model
            )
            assert isinstance(engine, RequestEngine) and isinstance(recipe, Codec)
            assert model == deployment.MODEL
            assert engine.remaining == 0
            self.engine = engine
            calls.append("http")

        def serve(self, *, host, port):
            assert (host, port) == ("0.0.0.0", 8000)
            calls.append("serve")
            if fail:
                raise RuntimeError("injected listener failure")
            self.engine.close()

    def tokenizer(path):
        assert path == str(deployment.CHECKPOINT / "tokenizer.json")
        return "tokenizer"

    monkeypatch.setattr(
        torch.cuda, "set_device", lambda value: calls.append(("device", value))
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(dist, "init_process_group", initialize)
    monkeypatch.setattr(ParallelGroups, "create", make_groups)
    monkeypatch.setattr(runner_module, "DeepSeekV41CudaRunner", Runner)
    monkeypatch.setattr(deployment, "DeepSeekV41Recipe", Codec)
    monkeypatch.setattr(tokenizers, "Tokenizer", SimpleNamespace(from_file=tokenizer))
    monkeypatch.setattr(distributed, "DistributedEngine", RequestEngine)
    monkeypatch.setattr(distributed, "DistributedWorker", Worker)
    monkeypatch.setattr("resin.server.HttpServer", Server)
    monkeypatch.setattr(
        deployment,
        "kernel_volume",
        SimpleNamespace(commit=lambda: calls.append("commit")),
    )
    if fail:
        with pytest.raises(RuntimeError, match="injected listener failure"):
            deployment._serve_rank(rank, 12345)
    else:
        deployment._serve_rank(rank, 12345)
    assert calls[:5] == [("device", rank), "initialize", "groups", "runner", "capture"]
    assert calls.count("step") == 2 and calls.count("close") == 1
    assert calls.index("capture") < calls.index("engine") < calls.index("http")
    if rank == 0:
        assert calls.index("engine") < calls.index("submit")
        assert calls.index("warmup_done") < calls.index("http")
    else:
        assert "submit" not in calls and "warmup_done" not in calls
    assert calls.index("http") < calls.index("serve")
    assert calls.count("commit") == (1 if rank == 0 else 0)
