"""Physical-rank observations use explicit snapshots without draining inference."""

import json
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from resin import Parallelism, RunnerConfig
from resin.backend.distributed import DistributedExecutionError, DistributedWorker
from resin.backend.metrics import RunnerMetrics
from resin.backend.topology import ParallelGroups


class _Pending:
    def __init__(self):
        self.harvests = 0

    def result(self):
        self.harvests += 1


class _Runner:
    runner_config = RunnerConfig()
    worker: DistributedWorker

    def __init__(self, rank, invalid=False):
        self.rank = rank
        self.invalid = invalid
        self.enabled = False
        self.pending = _Pending()

    def configure_metrics(self, *, enabled=True):
        self.enabled = enabled
        self.worker._completion = self.pending

    def metrics_snapshot(self):
        assert self.enabled
        assert self.pending.harvests == 0
        return RunnerMetrics(
            forward_seconds=(
                cast(dict[str, float], {"decode": object()})
                if self.invalid
                else {"decode": self.rank + 0.125}
            ),
            speculative_drafts=(1 << 54) + self.rank,
        )

    def forward(self, plan):
        raise AssertionError("snapshot must not execute the model")


def _rank(rank, rendezvous, output, invalid):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    groups = ParallelGroups.create(
        Parallelism(world_size=2, attn_tp=2), device_backend="gloo"
    )
    runner = _Runner(rank, invalid=invalid and rank == 1)
    worker = DistributedWorker(runner, groups)
    runner.worker = worker

    def no_objects(*args, **kwargs):
        raise AssertionError("metrics must not pickle Python objects")

    original = dist.all_gather_object
    dist.all_gather_object = no_objects
    try:
        if groups.is_driver:
            worker.configure_metrics()
            if invalid:
                with pytest.raises(
                    DistributedExecutionError, match="not JSON serializable"
                ):
                    worker.collect_metrics()
                assert runner.pending.harvests == 0
                worker._send({"operation": "close", "error": None})
                return
            snapshots = worker.collect_metrics()
            assert runner.pending.harvests == 0
            Path(output).write_text(
                json.dumps(
                    [
                        [
                            rank,
                            snapshot.forward_seconds["decode"],
                            snapshot.speculative_drafts,
                        ]
                        for rank, snapshot in snapshots
                    ]
                )
            )
            worker._send({"operation": "close", "error": None})
        else:
            worker.serve()
    finally:
        dist.all_gather_object = original
        groups.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("invalid", [False, True])
def test_physical_rank_snapshots_preserve_counters_and_pending_work(tmp_path, invalid):
    output = tmp_path / "metrics.json"
    mp.spawn(
        _rank,
        args=(f"file://{tmp_path / 'rendezvous'}", str(output), invalid),
        nprocs=2,
        join=True,
    )
    if invalid:
        return
    assert json.loads(output.read_text()) == [
        [0, 0.125, 1 << 54],
        [1, 1.125, (1 << 54) + 1],
    ]
