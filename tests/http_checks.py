"""Run shared HTTP qualification against an already-loaded distributed engine."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from resin.server import HttpServer
from scripts import check_server

if TYPE_CHECKING:
    from resin.backend.distributed import DistributedEngine
    from resin.recipe import Recipe


def check_loaded_server(
    engine: DistributedEngine,
    *,
    recipe: Recipe,
    model: str,
    output: Path,
    check_args: Sequence[str] = (),
    startup_timeout: float = 60,
) -> dict[str, Any]:
    """Call on every DP driver, with followers already in ``worker.serve()``.

    The root driver checks the public protocol on loopback. All drivers serve
    on their normal inference thread and receive the same report. Shutdown closes
    the supplied engine and its worker, so this must be the engine's final check.
    ``check_args`` are ordinary ``scripts/check_server.py`` CLI options.
    """
    worker = engine.worker
    group = worker.groups.drivers
    assert group is not None
    root = group.local_rank == 0
    port = 0
    if root:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    port = worker._drivers("HTTP qualification port", port)[0]
    base = f"http://127.0.0.1:{port}"
    server = HttpServer(engine, recipe=recipe, model=model)
    stopped = threading.Event()
    report: dict[str, Any] = {}
    timeout = startup_timeout

    def check():
        nonlocal timeout
        try:
            options = check_server.parse_args(
                [
                    base,
                    "--model",
                    model,
                    "--output",
                    str(output),
                    "--dp-size",
                    str(group.size),
                    *check_args,
                ]
            )
            timeout = options.timeout
            deadline = time.monotonic() + startup_timeout
            with httpx.Client(timeout=0.2) as client:
                while not stopped.is_set():
                    try:
                        client.get(base + "/health").raise_for_status()
                        break
                    except httpx.HTTPError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "HTTP server did not become ready"
                            ) from None
                        stopped.wait(0.02)
                else:
                    raise RuntimeError("HTTP server stopped before becoming ready")
            asyncio.run(check_server.main(options))
            report.update(json.loads(output.read_text()))
        except BaseException as error:
            report.update(ok=False, error=f"{type(error).__name__}: {error}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            server.shutdown()

    thread = threading.Thread(target=check, daemon=True) if root else None
    if thread is not None:
        thread.start()
    error = None
    try:
        server.serve(host="127.0.0.1", port=port)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stopped.set()
        if thread is not None:
            thread.join(timeout=timeout + 5)
            if thread.is_alive():
                error = error or "HTTP checker did not stop after server shutdown"
    results = worker._drivers("HTTP qualification result", (error, report))
    failures = [message for message, _ in results if message is not None]
    report = results[0][1]
    assert not failures and report.get("ok"), (
        f"HTTP qualification failed: {failures or report}"
    )
    return report
