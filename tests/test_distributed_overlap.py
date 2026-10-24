from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from output_helpers import forward_output

from resin import (
    CacheGroupLayout,
    CacheLayout,
    Parallelism,
    RunnerConfig,
    SamplingParams,
)
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups, ParallelismCapabilities


class CpuCompletion:
    def __init__(self, output, feedback):
        self.output = output
        self.feedback = feedback

    def result(self):
        return self.output


class ParallelRunner:
    runner_config = RunnerConfig(
        cache_layout=CacheLayout(32, [CacheGroupLayout(128)]),
        cache_write_lookahead_tokens=2,
    )
    parallelism_capabilities = ParallelismCapabilities(
        cp=True, ep=True, expert_tp=True, attn_dp=True
    )

    def __init__(self, groups):
        self.groups = groups
        self.calls = []

    def bind_resources(self, binding):
        pass

    def release(self, ids):
        pass

    def _join(self, context, asynchronous):
        codes = [None] * self.groups.domain.size
        dist.all_gather_object(
            codes, (context.step_id, asynchronous), group=self.groups.domain.handle
        )
        assert all(code == codes[0] for code in codes)
        self.calls.append((context.step_id, asynchronous, context.placeholder))

    def forward_distributed(self, plan, context):
        self._join(context, False)
        if plan is None:
            return None
        return forward_output(
            [[int(plan.token_ids[int(end) - 1]) + 1] for end in plan.token_indptr[1:]]
        )

    def enqueue_distributed(self, plan, context, *, feedback=None):
        self._join(context, True)
        if plan is None:
            tokens, positions = [], []
        else:
            tokens = plan.token_ids.tolist() if feedback is None else feedback.tokens
            positions = (
                plan.positions.tolist() if feedback is None else feedback.positions
            )
        rows = [[token + 1, token + 2] for token in tokens]
        return CpuCompletion(
            forward_output(rows),
            SimpleNamespace(
                tokens=[row[-1] for row in rows],
                positions=[
                    position + len(row)
                    for position, row in zip(positions, rows, strict=True)
                ],
            ),
        )


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    topology = Parallelism(world_size=4, attn_dp=2, cp=2, ep=2)
    groups = ParallelGroups.create(topology, distributed=dist, device_backend="gloo")
    for overlap in (False, True):
        runner = ParallelRunner(groups)
        worker = DistributedWorker(runner, groups)
        if groups.is_driver:
            engine = DistributedEngine(worker, enable_overlap=overlap)
            start = 1 + groups.attn_dp.local_rank * 20
            request = engine.submit(
                [start], sampling_params=SamplingParams(max_output_tokens=34)
            )
            events = []
            steps = 0
            while engine.has_work():
                events.extend(engine.step())
                steps += 1
                assert steps < 50
            assert request.take_result() == list(range(start + 1, start + 35))
            assert [event.token_id for event in events] == list(
                range(start + 1, start + 35)
            )
            assert (
                engine.local.overlap_stats[1] > 0
                if overlap
                else engine.local.overlap_stats == (0, 0)
            )
            engine.close()
            assert engine.local.cache_free_parent_count == 32
        else:
            worker.serve()
        dist.barrier()

    # An arrival changes phase on one DP replica while another has queued GPU
    # feedback. All ranks drain together before choosing the mixed-phase work.
    runner = ParallelRunner(groups)
    worker = DistributedWorker(runner, groups)
    if groups.is_driver:
        engine = DistributedEngine(worker)
        request = engine.submit(
            [1], sampling_params=SamplingParams(max_output_tokens=40)
        )
        events = []
        for _ in range(3):
            events.extend(engine.step())
        assert engine.local.overlap_pending
        if groups.attn_dp.local_rank == 0:
            engine.cancel(request)
            added = engine.submit(
                [20] * 5, sampling_params=SamplingParams(max_output_tokens=9)
            )
        else:
            added = None
        steps = 0
        while engine.has_work():
            events.extend(engine.step())
            steps += 1
            assert steps < 60
        if added is not None:
            assert added.take_result() == list(range(21, 30))
            assert request.cancelled
        else:
            assert request.take_result() == list(range(2, 42))
        engine.close()
        assert engine.local.cache_free_parent_count == 32
    else:
        worker.serve()
    for scenario in ("idle_cancel", "partial_cancel", "unequal_budget"):
        runner = ParallelRunner(groups)
        worker = DistributedWorker(runner, groups)
        if groups.is_driver:
            engine = DistributedEngine(worker)
            dp = groups.attn_dp.local_rank
            limit = 12 if scenario == "unequal_budget" and dp == 0 else 34
            request = engine.submit(
                [1], sampling_params=SamplingParams(max_output_tokens=limit)
            )
            survivor = (
                engine.submit(
                    [20], sampling_params=SamplingParams(max_output_tokens=34)
                )
                if scenario == "partial_cancel"
                else None
            )
            for _ in range(3):
                engine.step()
            assert engine.local.overlap_pending
            if scenario != "unequal_budget" and dp == 0:
                assert engine.cancel(request)
            steps = 0
            while engine.has_work():
                engine.step()
                steps += 1
                assert steps < 60
            if scenario != "unequal_budget" and dp == 0:
                assert request.cancelled
            else:
                assert request.take_result() == list(range(2, limit + 2))
            if survivor is not None:
                assert survivor.take_result() == list(range(21, 55))
            engine.close()
            assert engine.local.cache_free_parent_count == 32
        else:
            worker.serve()
        dist.barrier()

    # A single chat must overlap even while its EP peers have no attention work.
    # A later admission on an idle replica drains before mixed-phase execution;
    # EOS then retires that replica while its peer continues decoding.
    runner = ParallelRunner(groups)
    worker = DistributedWorker(runner, groups)
    if groups.is_driver:
        engine = DistributedEngine(worker)
        dp = groups.attn_dp.local_rank
        request = (
            engine.submit(
                [1],
                sampling_params=SamplingParams(
                    max_output_tokens=60, stop_token_ids=[48]
                ),
            )
            if dp == 0
            else None
        )
        for _ in range(8):
            engine.step()
        if dp == 0:
            assert engine.local.overlap_stats[1] > 0
        else:
            assert any(async_ and idle for _, async_, idle in runner.calls)
            request = engine.submit(
                [20] * 5,
                sampling_params=SamplingParams(
                    max_output_tokens=30, stop_token_ids=[29]
                ),
            )
        steps = 0
        events = []
        while engine.has_work():
            events.extend(engine.step())
            steps += 1
            assert steps < 80
        assert request is not None
        expected = list(range(2, 49)) if dp == 0 else list(range(21, 30))
        assert request.take_result() == expected
        assert engine.local.overlap_stats[1] > 0
        if dp == 1:
            assert [event.token_id for event in events] == expected
        engine.close()
        assert worker._completion is None
        assert engine.local.cache_free_parent_count == 32
    else:
        worker.serve()
    dist.barrier()

    groups.close()
    dist.destroy_process_group()


def test_overlap_feedback_composes_with_attention_dp_cp_ep_and_followers(tmp_path):
    mp.spawn(_worker, args=(tmp_path / "rendezvous",), nprocs=4, join=True)
