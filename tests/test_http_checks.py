"""The shared loaded-engine helper closes every driver/follower on either outcome."""

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from http_checks import check_loaded_server
from test_distributed_http import RankRecipe, RankRunner

from resin import Parallelism
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups


def _check_rank(rank, directory, fail_check):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=20),
    )
    groups = ParallelGroups.create(
        Parallelism(world_size=4, attn_dp=2, attn_tp=2, ep=2),
        device_backend="gloo",
        timeout=timedelta(seconds=20),
    )
    runner = RankRunner(groups, directory, "normal")
    worker = DistributedWorker(runner, groups)
    error, engine = None, None
    try:
        if groups.is_driver:
            engine = DistributedEngine(worker, context_length=256)
            if fail_check:
                from scripts import check_server

                async def fail(args):
                    del args
                    raise RuntimeError("injected checker failure")

                setattr(check_server, "main", fail)
            check_loaded_server(
                engine,
                recipe=RankRecipe(rank, "normal"),
                model="rank-test",
                output=Path(directory) / "http.json",
                startup_timeout=5,
                check_args=(
                    "--max-tokens",
                    "2",
                    "--concurrency",
                    "2",
                    "--timeout",
                    "5",
                    "--skip-prefix-cache",
                    "--context-lengths",
                ),
            )
        else:
            worker.serve()
    except AssertionError as exc:
        error = str(exc)
    finally:
        Path(directory, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "closed": worker.closed,
                    "engine_closed": engine.closed if engine else None,
                    "error": error,
                    "calls": len(runner.calls),
                }
            )
        )
        groups.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_check", [False, True])
def test_loaded_http_check_coordinates_shutdown(tmp_path, fail_check):
    processes = mp.spawn(
        _check_rank, args=(str(tmp_path), fail_check), nprocs=4, join=False
    )
    deadline = time.monotonic() + 45
    try:
        while not processes.join(timeout=1):
            assert time.monotonic() < deadline, "HTTP qualification ranks did not stop"
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    records = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(4)
    ]
    assert all(record["closed"] for record in records)
    for rank in (0, 2):
        assert records[rank]["engine_closed"]
        assert bool(records[rank]["error"]) == fail_check
    report = json.loads((tmp_path / "http.json").read_text())
    assert report["ok"] != fail_check
    if not fail_check:
        assert all(record["calls"] > 0 for record in records)
        assert {
            check["name"] for check in report["checks"] if check["status"] == "passed"
        } >= {"chat", "logprobs", "concurrent"}
