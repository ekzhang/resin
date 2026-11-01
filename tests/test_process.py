"""Real spawned workers preserve rank arguments, failures, and group cleanup."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import pytest

import resin


def _record_rank(rank: int, directory: str, value: str) -> None:
    Path(directory, f"{rank}.txt").write_text(value)


def _fail(rank: int, ready_path: str) -> None:
    if rank == 0:
        # Ensure the peer is running when the failure is reported.
        deadline = time.monotonic() + 10
        while not Path(ready_path).exists():
            if time.monotonic() > deadline:
                raise RuntimeError("peer did not start")
            time.sleep(0.01)
        raise ValueError("worker exception detail")
    Path(ready_path).touch()
    time.sleep(60)


def _exit(rank: int) -> None:
    os._exit(7)


def _sleep(rank: int) -> None:
    time.sleep(60)


def _large_failure(rank: int) -> None:
    raise ValueError("x" * 200_000)


def _system_exit(rank: int, code: int | None) -> None:
    raise SystemExit(code)


def test_spawn_waits_and_returns_reaped_context(tmp_path: Path) -> None:
    context = resin.process.spawn(
        _record_rank, (str(tmp_path), "hello"), devices=["cpu", "cpu"], numa="off"
    )
    assert isinstance(context, resin.process.ProcessContext)
    assert context.placements == [None, None]
    assert [process.exitcode for process in context.processes] == [0, 0]
    assert [Path(tmp_path, f"{rank}.txt").read_text() for rank in range(2)] == [
        "hello",
        "hello",
    ]
    assert context.join(timeout=0)
    context.close()


def test_worker_exception_stops_and_reaps_peers(tmp_path: Path) -> None:
    context = resin.process.spawn(
        _fail,
        (str(tmp_path / "ready"),),
        devices=["cpu", "cpu"],
        numa="off",
        join=False,
    )
    with pytest.raises(
        resin.process.ProcessError, match="worker exception detail"
    ) as raised:
        context.join(timeout=15)
    assert raised.value.rank == 0
    assert raised.value.exitcode == 1
    assert "ValueError" in raised.value.worker_traceback
    assert all(not process.is_alive() for process in context.processes)
    assert all(process.exitcode is not None for process in context.processes)
    with pytest.raises(resin.process.ProcessError):
        context.join()


def test_nonzero_exit_is_reported() -> None:
    with pytest.raises(resin.process.ProcessError, match="rank 0.*code 7") as raised:
        resin.process.spawn(_exit, devices=["cpu"], numa="off")
    assert raised.value.worker_traceback == ""


@pytest.mark.parametrize("code", [None, 0])
def test_successful_system_exit_is_preserved(code: int | None) -> None:
    context = resin.process.spawn(_system_exit, (code,), devices=["cpu"], numa="off")
    assert context.processes[0].exitcode == 0


def test_unsuccessful_system_exit_preserves_code() -> None:
    with pytest.raises(resin.process.ProcessError, match="code 9") as raised:
        resin.process.spawn(_system_exit, (9,), devices=["cpu"], numa="off")
    assert raised.value.exitcode == 9
    assert "SystemExit: 9" in raised.value.worker_traceback


def test_large_traceback_does_not_block_worker_exit() -> None:
    context = resin.process.spawn(
        _large_failure, devices=["cpu"], numa="off", join=False
    )
    try:
        with pytest.raises(resin.process.ProcessError) as raised:
            context.join(timeout=15)
        assert len(raised.value.worker_traceback) > 200_000
    finally:
        context.close()


def test_timeout_leaves_workers_running_until_close() -> None:
    context = resin.process.spawn(
        _sleep, devices=["cpu", "cpu"], numa="off", join=False
    )
    started = time.monotonic()
    assert not context.join(timeout=0.05)
    assert time.monotonic() - started < 1
    assert any(process.is_alive() for process in context.processes)
    context.close()
    context.close()
    assert all(not process.is_alive() for process in context.processes)
    assert all(process.exitcode is not None for process in context.processes)


def test_context_manager_closes_workers_when_body_raises() -> None:
    context = resin.process.spawn(_sleep, devices=["cpu"], numa="off", join=False)
    with pytest.raises(ValueError, match="caller failed"):
        with context:
            raise ValueError("caller failed")
    assert not context.processes[0].is_alive()


def test_partial_start_failure_reaps_started_worker(monkeypatch) -> None:
    real_context = resin.process.mp.get_context("spawn")
    created = []

    class BrokenContext:
        def Process(self, **kwargs):
            process = real_context.Process(**kwargs)
            created.append(process)
            if len(created) == 2:

                def fail_start():
                    raise RuntimeError("start failed")

                monkeypatch.setattr(process, "start", fail_start)
            return process

    monkeypatch.setattr(resin.process.mp, "get_context", lambda method: BrokenContext())
    with pytest.raises(RuntimeError, match="start failed"):
        resin.process.spawn(_sleep, devices=["cpu", "cpu"], numa="off", join=False)
    assert created[0].exitcode is not None
    assert not created[0].is_alive()


@pytest.mark.parametrize("numa", ["auto", "required"])
def test_cpu_placement_modes_are_available(
    numa: Literal["auto", "required"], tmp_path: Path
) -> None:
    context = resin.process.spawn(
        _record_rank, (str(tmp_path), "cpu"), devices=["cpu"], numa=numa
    )
    assert (tmp_path / "0.txt").read_text() == "cpu"
    assert len(context.placements) == 1
    assert context.placements[0] is not None
    assert context.placements[0].device == "cpu"
    assert context.placements[0].reason is None


def test_invalid_options_fail_before_start() -> None:
    with pytest.raises(ValueError, match="numa"):
        resin.process.spawn(_sleep, devices=["cpu"], numa="invalid")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="at least one"):
        resin.process.spawn(_sleep, devices=[])
    with pytest.raises(TypeError, match="sequence"):
        resin.process.spawn(_sleep, devices="cpu")
