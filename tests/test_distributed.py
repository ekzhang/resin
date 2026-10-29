from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from output_helpers import forward_output

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    ForwardOutput,
    Parallelism,
    ResourceBinding,
    RunnerConfig,
    SamplingParams,
)
from resin.backend.distributed import (
    DistributedEngine,
    DistributedExecutionError,
    DistributedWorker,
)
from resin.backend.topology import ParallelGroups, ParallelismCapabilities


@dataclass
class LocalRunner:
    forward: Callable[[BatchPlan], ForwardOutput]
    runner_config: RunnerConfig = field(default_factory=RunnerConfig)
    bind_resources: Callable[[ResourceBinding], None] = lambda _: None
    release: Callable[[list[int]], None] = lambda _: None


class CollectiveRunner:
    """CPU reference execution: context shards → EP route/combine → TP/CP sums."""

    def __init__(self, groups, case):
        self.groups, self.case = groups, case
        self.calls, self.released, self.bindings = [], [], []
        self.runner_config = RunnerConfig(
            cache_layout=CacheLayout(parent_capacity=64, groups=[CacheGroupLayout(2)]),
            max_batch_tokens=4,
            prefill_chunk_size=2,
            continuation_prefill=case != "prepare_failure",
        )

    def bind_resources(self, binding):
        self.bindings.append(binding.cache_pool_id)

    def release(self, ids):
        if self.case == "release_failure" and self.groups.rank == 1:
            raise RuntimeError("injected release failure")
        self.released.extend(map(int, ids))

    def validate_distributed(self, plan, context):
        if self.case == "preflight_failure" and self.groups.rank == 1:
            raise ValueError("injected preflight failure")
        assert context.placeholder == (plan is None)
        if plan is not None:
            assert plan.phase == context.phase
            assert plan.num_sequences <= context.max_num_sequences
            assert len(plan.token_ids) <= context.max_num_tokens
        if self.case == "normal":
            return context.step_id, plan

    def forward_distributed(self, plan, context, *, prepared=None):
        if self.case == "normal":
            assert prepared is not None
            assert prepared[0] == context.step_id and prepared[1] is plan
        else:
            assert prepared is None
        groups = self.groups
        _, cp, _, tp = groups.topology.coordinates(groups.rank)
        prefill = context.phase == BatchPhase.PREFILL
        offset, stride = (cp, groups.topology.cp) if prefill else (0, 1)
        tokens, rows = [], []
        if plan is not None:
            for row, (start, end) in enumerate(
                zip(plan.token_indptr[:-1], plan.token_indptr[1:], strict=True)
            ):
                for position in range(int(start) + offset, int(end), stride):
                    tokens.append(int(plan.token_ids[position]))
                    rows.append(row)
        # Real variable-size expert exchange, including completely empty senders.
        order = sorted(
            range(len(tokens)), key=lambda index: tokens[index] % groups.ep.size
        )
        counts = torch.tensor(
            [
                sum(token % groups.ep.size == expert for token in tokens)
                for expert in range(groups.ep.size)
            ],
            dtype=torch.int64,
        )
        received_counts = torch.empty_like(counts)
        dist.all_to_all_single(received_counts, counts, group=groups.ep.handle)
        incoming = torch.empty(int(received_counts.sum()), dtype=torch.int64)
        outgoing = torch.tensor([tokens[index] for index in order], dtype=torch.int64)
        dist.all_to_all_single(
            incoming,
            outgoing,
            received_counts.tolist(),
            counts.tolist(),
            group=groups.ep.handle,
        )
        processed = incoming * 2 + groups.ep.local_rank + 1
        returned = torch.empty_like(outgoing)
        dist.all_to_all_single(
            returned,
            processed,
            counts.tolist(),
            received_counts.tolist(),
            group=groups.ep.handle,
        )
        contributions = torch.zeros(max(context.num_sequences, 1), dtype=torch.int64)
        for index, value in zip(order, returned.tolist(), strict=True):
            contributions[rows[index]] += value * (tp + 1)
        dist.all_reduce(contributions, group=groups.attn_tp.handle)
        if prefill:
            dist.all_reduce(contributions, group=groups.cp.handle)
        replicated = [None] * groups.attn_replication.size
        dist.all_gather_object(
            replicated, contributions.tolist(), group=groups.attn_replication.handle
        )
        assert all(value == replicated[0] for value in replicated)
        dist.barrier(group=groups.expert_tp.handle)
        self.calls.append(
            {
                "step": context.step_id,
                "phase": str(context.phase),
                "placeholder": context.placeholder,
                "rows": context.num_sequences,
                "max_rows": context.max_num_sequences,
            }
        )
        if self.case == "forward_failure" and groups.rank == 1:
            raise RuntimeError("injected forward failure")
        if plan is None:
            return None
        expected = [
            sum(
                2 * int(token) + int(token) % groups.ep.size + 1
                for token in plan.token_ids[start:end]
            )
            * (groups.attn_tp.size * (groups.attn_tp.size + 1) // 2)
            % 127
            for start, end in zip(
                plan.token_indptr[:-1], plan.token_indptr[1:], strict=True
            )
        ]
        output = (contributions[: plan.num_sequences] % 127).numpy().astype(np.uint32)
        np.testing.assert_array_equal(output, expected)
        if self.case == "malformed_output" and groups.rank == 0:
            # Deliberately violate the native-only completed-result contract.
            return [[] for _ in output]
        # Exercise the same array handoff while coordinator metadata remains separate.
        scores = np.full(len(output), -0.25, np.float32)
        return ForwardOutput(
            token_ids=output,
            token_indptr=np.arange(len(output) + 1, dtype=np.uint32),
            logprobs=scores,
            top_indptr=np.arange(0, 2 * len(output) + 1, 2, dtype=np.uint32),
            top_token_ids=np.repeat(output, 2),
            top_logprobs=np.tile(np.array([-0.25, -2.0], np.float32), len(output)),
        )


def _run_case(groups, case):
    runner = CollectiveRunner(groups, case)
    worker = DistributedWorker(runner, groups)
    request_results, failed, local_poisoned = [], False, False
    if not groups.is_driver:
        try:
            worker.serve()
        except DistributedExecutionError as exc:
            assert "injected" in str(exc) or case in (
                "prepare_failure",
                "malformed_output",
                "initialize_failure",
                "serialize_failure",
            )
            failed = True
    else:
        options = (
            {"max_running_requests": 0}
            if case == "initialize_failure" and groups.rank == 0
            else {"max_running_requests": 2}
        )
        try:
            engine = DistributedEngine(worker, **options)
        except DistributedExecutionError:
            assert case == "initialize_failure"
            failed = True
        else:
            dp = groups.topology.coordinates(groups.rank)[0]
            requests = []
            if case == "normal":
                if dp == 0:
                    requests = [
                        engine.submit(
                            [1, 2, 3, 4, 5],
                            sampling_params=SamplingParams(
                                max_output_tokens=3, top_logprobs=2
                            ),
                        ),
                        engine.submit(
                            [6], sampling_params=SamplingParams(max_output_tokens=2)
                        ),
                    ]
            else:
                requests = [
                    engine.submit(
                        [1, 2, 3] if case == "prepare_failure" and dp == 0 else [7],
                        sampling_params=SamplingParams(max_output_tokens=1),
                    )
                ]
            if case == "serialize_failure" and dp == 0:
                engine.cancel(requests.pop())
                requests.append(
                    engine.submit(
                        [7],
                        sampling_params=SamplingParams(max_output_tokens=1),
                        media=SimpleNamespace(
                            token_ids=(7,),
                            cache_key=(1,),
                            token_spans=(),
                            injected=lambda: None,
                        ),
                    )
                )
            steps = 0
            try:
                while engine.has_work():
                    events = engine.step()
                    steps += 1
                    assert steps < 30
                    for event in events:
                        if event.logprobs is not None:
                            assert event.logprobs.logprob == -0.25
                    if case == "normal" and steps == 1 and dp == 1:
                        requests.append(
                            engine.submit(
                                [11],
                                sampling_params=SamplingParams(max_output_tokens=3),
                            )
                        )
                    if case == "normal" and steps == 2 and dp == 0:
                        cancelled = engine.submit(
                            [21], sampling_params=SamplingParams(max_output_tokens=2)
                        )
                        assert (
                            engine.cancel(cancelled) and cancelled.take_result() == []
                        )
                assert engine.step() == []
                request_results = [request.take_result() for request in requests]
            except DistributedExecutionError:
                failed = True
                assert engine.poisoned
                local_poisoned = engine.local.poisoned
                if case in ("preflight_failure", "forward_failure"):
                    assert engine.local.cache_occupied_block_count is not None
                    assert engine.local.cache_occupied_block_count > 0
                with pytest.raises(DistributedExecutionError):
                    engine.step()
            finally:
                try:
                    engine.close()
                except DistributedExecutionError:
                    assert failed
                assert engine.closed
                if not failed:
                    assert engine.local.cache_free_parent_count == 64
    assert failed == (case != "normal")
    return {
        "calls": runner.calls,
        "bindings": runner.bindings,
        "released": runner.released,
        "failed": failed,
        "local_poisoned": local_poisoned,
        "results": request_results,
    }


def _shared_experts_case(groups):
    runner = CollectiveRunner(groups, "normal")
    worker = DistributedWorker(runner, groups)
    if groups.is_driver:
        engine = DistributedEngine(worker)
        dp = groups.topology.coordinates(groups.rank)[0]
        request = engine.submit(
            [7], sampling_params=SamplingParams(max_output_tokens=dp * 3 + 1)
        )
        while engine.has_work():
            engine.step()
        assert len(request.take_result()) == dp * 3 + 1
        engine.close()
    else:
        worker.serve()
    return runner.calls


def _tensor_only_case(groups):
    class ForwardOnly:
        calls = 0

        def forward(self, plan):
            self.calls += 1
            values = torch.tensor(
                [int(plan.token_ids[end - 1]) for end in plan.token_indptr[1:]],
                dtype=torch.int64,
            )
            values *= groups.attn_tp.local_rank + 1
            dist.all_reduce(values, group=groups.attn_tp.handle)
            return forward_output((values % 127).tolist())

    runner = ForwardOnly()
    worker = DistributedWorker(runner, groups)
    if groups.is_driver:
        engine = DistributedEngine(worker)
        request = engine.submit(
            [2], sampling_params=SamplingParams(max_output_tokens=3)
        )
        while engine.has_work():
            engine.step()
        expected, value = [], 2
        for _ in range(3):
            value = value * (groups.attn_tp.size * (groups.attn_tp.size + 1) // 2) % 127
            expected.append(value)
        assert request.take_result() == expected
        engine.close()
    else:
        worker.serve()
    return runner.calls


def _distributed_process(rank, world_size, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    with pytest.raises(ValueError, match="agree"):
        ParallelGroups.create(
            Parallelism(world_size=8, attn_tp=8)
            if rank == 0
            else Parallelism(world_size=8, cp=2, attn_dp=2, attn_tp=2, ep=4),
            device_backend="gloo",
        )
    with pytest.raises(ValueError, match="world size"):
        ParallelGroups.create(Parallelism(world_size=4), device_backend="gloo")
    groups = ParallelGroups.create(
        Parallelism(world_size=8, cp=2, attn_dp=2, attn_tp=2, ep=4),
        device_backend="gloo",
        timeout=timedelta(seconds=20),
    )
    with pytest.raises(DistributedExecutionError, match="forward_distributed"):
        DistributedWorker(LocalRunner(forward=lambda plan: forward_output([1])), groups)
    unsupported = CollectiveRunner(groups, "normal")
    setattr(unsupported, "parallelism_capabilities", ParallelismCapabilities())
    with pytest.raises(DistributedExecutionError, match="does not support attn_dp"):
        DistributedWorker(unsupported, groups)
    divergent = CollectiveRunner(groups, "normal")
    if rank == 1:
        divergent.runner_config = RunnerConfig(
            cache_layout=CacheLayout(parent_capacity=63, groups=[CacheGroupLayout(2)]),
            max_batch_tokens=4,
            prefill_chunk_size=2,
            continuation_prefill=True,
        )
    with pytest.raises(DistributedExecutionError, match="configurations must agree"):
        DistributedWorker(divergent, groups)
    for max_accepted in ((1, 3), (3, 6)):
        divergent = CollectiveRunner(groups, "normal")
        divergent.runner_config = RunnerConfig(
            cache_layout=CacheLayout(parent_capacity=64, groups=[CacheGroupLayout(2)]),
            max_accepted_per_step=max_accepted[
                groups.topology.coordinates(rank)[0]
            ],
        )
        with pytest.raises(DistributedExecutionError, match="max_accepted_per_step must agree"):
            DistributedWorker(divergent, groups)
        assert divergent.calls == []

    # Attention-DP replicas keep independent cache capacities and scheduling
    # limits; only the collective decode protocol must match across them.
    independent = CollectiveRunner(groups, "normal")
    replica = groups.topology.coordinates(rank)[0]
    independent.runner_config = RunnerConfig(
        cache_layout=CacheLayout(
            parent_capacity=64 + replica, groups=[CacheGroupLayout(2)]
        ),
        max_batch_tokens=4 + replica,
        prefill_chunk_size=2 + replica,
        continuation_prefill=bool(replica),
        max_accepted_per_step=3,
    )
    DistributedWorker(independent, groups)
    records = {}
    try:
        for case in (
            "normal",
            "preflight_failure",
            "forward_failure",
            "malformed_output",
            "release_failure",
            "prepare_failure",
            "initialize_failure",
            "serialize_failure",
        ):
            records[case] = _run_case(groups, case)
            dist.barrier()
    finally:
        groups.close()
    shared = ParallelGroups.create(
        Parallelism(world_size=8, cp=2, attn_dp=2, attn_tp=2),
        device_backend="gloo",
        timeout=timedelta(seconds=20),
    )
    try:
        records["shared_experts"] = _shared_experts_case(shared)
        dist.barrier()
    finally:
        shared.close()
    tensor_only = ParallelGroups.create(
        Parallelism(world_size=8, attn_tp=8),
        device_backend="gloo",
        timeout=timedelta(seconds=20),
    )
    try:
        records["tensor_only_steps"] = _tensor_only_case(tensor_only)
        dist.barrier()
    finally:
        tensor_only.close()
    overlap = ParallelGroups.create(
        Parallelism(world_size=8, ep=8, attn_dp=4, attn_tp=2),
        device_backend="gloo",
        timeout=timedelta(seconds=20),
    )
    try:
        records["overlap"] = _run_case(overlap, "normal")
        dist.barrier()
    finally:
        overlap.close()
    replicated = ParallelGroups.create(
        Parallelism(world_size=8, ep=4, attn_dp=2, cp=2),
        device_backend="gloo",
        timeout=timedelta(seconds=20),
    )
    try:
        records["replicated_attention"] = _run_case(replicated, "normal")
        dist.barrier()
    finally:
        replicated.close()
        dist.destroy_process_group()
    Path(directory, f"{rank}.json").write_text(json.dumps(records))


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required"
)
def test_real_tp_cp_ep_and_attn_dp_processes(tmp_path):
    world_size = 8
    process = mp.start_processes(
        _distributed_process,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=world_size,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + 110
    try:
        while not process.join(timeout=1):
            if time.monotonic() > deadline:
                # ty misbinds pytest's decorated helper and drops its reason parameter.
                pytest.fail(
                    "distributed prototype did not shut down within 110 seconds"  # ty: ignore[invalid-argument-type]
                )
    finally:
        for child in process.processes:
            if child.is_alive():
                child.terminate()
        for child in process.processes:
            child.join(timeout=5)
    records = [
        json.loads((tmp_path / f"{rank}.json").read_text())
        for rank in range(world_size)
    ]
    calls = [record["normal"]["calls"] for record in records]
    phases = [(call["step"], call["phase"]) for call in calls[0]]
    assert all(
        [(call["step"], call["phase"]) for call in rank_calls] == phases
        for rank_calls in calls
    )
    assert any(call["placeholder"] for call in calls[0])
    assert any(call["placeholder"] for call in calls[4])
    assert list(map(len, records[0]["normal"]["results"])) == [3, 2]
    assert list(map(len, records[4]["normal"]["results"])) == [3]
    for replica in ((0, 1, 2, 3), (4, 5, 6, 7)):
        assert all(
            records[rank]["normal"]["bindings"]
            == records[replica[0]]["normal"]["bindings"]
            for rank in replica
        )
        assert all(
            records[rank]["normal"]["released"]
            == records[replica[0]]["normal"]["released"]
            for rank in replica
        )
    assert all(not record["preflight_failure"]["calls"] for record in records)
    assert all(not record["prepare_failure"]["calls"] for record in records)
    assert records[0]["forward_failure"]["local_poisoned"]
    assert records[4]["forward_failure"]["local_poisoned"]
    assert all(len(record["shared_experts"]) == 4 for record in records)
    assert [call["placeholder"] for call in records[0]["shared_experts"]] == [
        False,
        True,
        True,
        True,
    ]
    assert all(not call["placeholder"] for call in records[4]["shared_experts"])
    assert all(record["tensor_only_steps"] == 3 for record in records)
    assert list(map(len, records[0]["overlap"]["results"])) == [3, 2]
    assert list(map(len, records[2]["overlap"]["results"])) == [3]
    assert list(map(len, records[0]["replicated_attention"]["results"])) == [3, 2]
    assert list(map(len, records[4]["replicated_attention"]["results"])) == [3]
    for replica in ((0, 1, 2, 3), (4, 5, 6, 7)):
        assert all(
            records[rank]["replicated_attention"]["calls"]
            == records[replica[0]]["replicated_attention"]["calls"]
            for rank in replica
        )
        assert all(
            records[rank]["replicated_attention"]["bindings"]
            == records[replica[0]]["replicated_attention"]["bindings"]
            for rank in replica
        )
    assert all(not record["serialize_failure"]["calls"] for record in records)


def test_external_failure_preserves_native_resources_and_disables_cleanup():
    releases = []
    runner = LocalRunner(
        runner_config=RunnerConfig(
            cache_layout=CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(2)])
        ),
        bind_resources=lambda binding: None,
        forward=lambda plan: forward_output([3]),
        release=lambda ids: releases.extend(ids),
    )
    from resin import Engine

    engine = Engine(runner)
    request = engine.submit([1, 2], sampling_params=SamplingParams(max_output_tokens=3))
    engine.step()
    before = engine.cache_occupied_block_count
    engine.poison("peer execution failed")
    assert engine.poisoned and not request.done
    assert engine.cache_occupied_block_count == before
    with pytest.raises(RuntimeError, match="peer execution failed"):
        engine.close()
    assert releases == []


def _single_rank_groups():
    from resin.backend.topology import RankGroup

    singleton = RankGroup((0,), None, 0)
    return ParallelGroups(
        Parallelism(),
        0,
        singleton,
        singleton,
        singleton,
        singleton,
        singleton,
        singleton,
        singleton,
        singleton,
        singleton,
        singleton,
        singleton,
        None,
        [],
    )


def test_single_rank_replica_keeps_preparation_local_and_skips_serialization(
    monkeypatch,
):
    import resin.backend.distributed as implementation

    def unexpected_serialization(*args, **kwargs):
        raise AssertionError("a single-rank replica has no plan consumer")

    monkeypatch.setattr(
        implementation.PlanTransport, "prepare", unexpected_serialization
    )

    class PreparedRunner:
        runner_config = RunnerConfig()
        calls = 0
        validations = 0

        def validate_distributed(self, plan, context):
            self.validations += 1
            return context.step_id, plan

        def forward_distributed(self, plan, context, *, prepared=None):
            assert prepared is not None
            assert prepared[0] == context.step_id and prepared[1] is plan
            assert plan.media[0].callback() == 19
            self.calls += 1
            return forward_output([3])

    runner = PreparedRunner()
    engine = DistributedEngine(DistributedWorker(runner, _single_rank_groups()))
    request = engine.submit(
        [1],
        sampling_params=SamplingParams(max_output_tokens=1),
        media=SimpleNamespace(
            token_ids=(1,), cache_key=(1,), token_spans=(), callback=lambda: 19
        ),
    )
    engine.step()
    assert request.take_result() == [3]
    assert runner.calls == runner.validations == 1
    engine.close()


def test_control_failure_after_model_work_keeps_the_accepted_batch(monkeypatch):
    groups = _single_rank_groups()
    releases = []
    calls = []
    runner = LocalRunner(
        runner_config=RunnerConfig(
            cache_layout=CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(2)])
        ),
        bind_resources=lambda binding: None,
        forward=lambda plan: calls.append(plan) or forward_output([3]),
        release=lambda ids: releases.extend(ids),
    )
    worker = DistributedWorker(runner, groups)
    engine = DistributedEngine(worker)
    request = engine.submit([1, 2], sampling_params=SamplingParams(max_output_tokens=2))
    consensus_calls = 0
    original = worker._consensus

    def consensus(group, error):
        nonlocal consensus_calls
        consensus_calls += 1
        if consensus_calls == 2:  # The completion vote after model.forward returned.
            raise RuntimeError("injected transport completion failure")
        return original(group, error)

    monkeypatch.setattr(worker, "_consensus", consensus)
    with pytest.raises(DistributedExecutionError, match="transport completion"):
        engine.step()
    assert len(calls) == 1
    assert engine.local.poisoned and not request.done
    assert engine.local.ready_request_count == 0  # Kept in quarantine, not requeued.
    assert engine.local.cache_occupied_block_count is not None
    assert engine.local.cache_occupied_block_count > 0
    assert releases == []
    with pytest.raises(DistributedExecutionError):
        engine.close()
