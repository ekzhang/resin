"""Checkpoint-free production scorer benchmark; see deploy/modal_bench_scoring.py."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _inputs(rows, vocab, dtype, k, mixed, *, width=1):
    generator = torch.Generator("cuda").manual_seed(131)
    logits = torch.randn(rows, vocab, device="cuda", dtype=dtype, generator=generator)
    ids = torch.arange(rows, device="cuda") * 17 % vocab
    mask = (
        torch.arange(rows, device="cuda") % 3 != 1
        if mixed
        else torch.ones_like(ids, dtype=torch.bool)
    )
    counts = (
        torch.where(torch.arange(rows, device="cuda") % 3 == 2, 0, k).to(torch.int32)
        if mixed
        else torch.full_like(ids, k, dtype=torch.int32)
    )
    prefix = (rows // width, width) if width > 1 else (rows,)
    return (
        logits.reshape(*prefix, vocab),
        ids.reshape(prefix),
        {
            "score_mask": mask.reshape(prefix),
            "top_counts": counts.reshape(prefix),
            "max_top_k": k,
        },
    )


def _check(result, logits, ids, options):
    scores, top_ids, top_scores, _ = result
    logits, ids, scores = (
        logits.reshape(-1, logits.shape[-1]),
        ids.reshape(-1),
        scores.reshape(-1),
    )
    top_ids = top_ids.reshape(ids.numel(), options["max_top_k"])
    top_scores = top_scores.reshape(ids.numel(), options["max_top_k"])
    reference = logits.double().log_softmax(-1)
    mask, counts = options["score_mask"].reshape(-1), options["top_counts"].reshape(-1)
    expected = reference.gather(1, ids[:, None]).squeeze(1)
    torch.testing.assert_close(
        scores, torch.where(mask, expected, 0).float(), atol=2e-5, rtol=2e-5
    )
    for row in range(logits.shape[0]):
        count = int(counts[row]) if mask[row] else 0
        actual_ids = top_ids[row, :count].long()
        assert actual_ids.unique().numel() == count
        if count:
            torch.testing.assert_close(
                top_scores[row, :count],
                reference[row].topk(count).values.float(),
                atol=2e-5,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                top_scores[row, :count],
                reference[row, actual_ids].float(),
                atol=2e-5,
                rtol=2e-5,
            )
        assert (top_ids[row, count:] == -1).all()
        assert (top_scores[row, count:] == 0).all()


def _capture(function, chain=32):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(chain):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def _timings(functions):
    graphs = {name: _capture(function) for name, function in functions.items()}
    samples = {name: [] for name in graphs}
    names = list(graphs)
    for round_index in range(5):
        for name in names if round_index % 2 == 0 else reversed(names):
            start, stop = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(20):
                graphs[name].replay()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) * 1000 / 20 / 32)
    return {
        name: {
            "median_us": statistics.median(values),
            "samples_us": values,
            "calls_per_graph": 32,
        }
        for name, values in samples.items()
    }


def _allocation(function):
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outputs = function()
    torch.cuda.synchronize()
    return {
        "peak_extra_bytes": torch.cuda.max_memory_allocated() - baseline,
        "output_bytes": sum(value.numel() * value.element_size() for value in outputs),
    }


def _audit(score_logits):
    import cutlass.cute as cute

    original, calls, results = cute.compile, [], []

    def compile_kernel(*args, **kwargs):
        calls.append(type(args[0]).__name__)
        return original(*args, **kwargs)

    with patch.object(cute, "compile", compile_kernel):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for k in (0, 1, 32):
                logits, ids, options = _inputs(1, 32768, dtype, k, False)
                score_logits(logits, ids, **options)
            torch.cuda.synchronize()
            warm = len(calls)
            for rows in (1, 2, 4, 8, 16, 32, 64, 128, 3, 7, 17, 33, 127):
                for k in (0, 1, 5, 20, 32):
                    logits, ids, options = _inputs(rows, 32768, dtype, k, True)
                    score_logits(logits, ids, **options)
            torch.cuda.synchronize()
            results.append(
                {
                    "dtype": str(dtype),
                    "warm_compile_calls": warm,
                    "new_compile_calls": len(calls) - warm,
                    "new_compiles": calls[warm:],
                }
            )
            assert len(calls) == warm, results[-1]
    return results


def _metadata():
    files = [Path(__file__), ROOT / "resin/backend/cuda/completion.py"]
    files += list((ROOT / "models/cuda/resin_models_cuda").glob("*scoring*.py"))
    return {
        "hardware": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "nvidia-cutlass-dsl",
                "nvidia-cutlass-dsl-libs-base",
                "quack-kernels",
                "flashinfer-python",
            )
        },
        "cuda": torch.version.cuda,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "timing_method": "32 direct invocations/CUDA graph;20 replays/sample;five interleaved forward/reverse rounds;GPU-event time per invocation",
    }


@torch.inference_mode()
def cache_probe():
    """Run in a fresh process; count real CuTe compilation before scorer import."""
    import cutlass.cute as cute

    original, calls = cute.compile, []
    load_module, loads = cute.runtime.load_module, []

    def compile_kernel(*args, **kwargs):
        calls.append(type(args[0]).__name__)
        return original(*args, **kwargs)

    def load_cached(*args, **kwargs):
        loads.append(str(args[0]))
        return load_module(*args, **kwargs)

    with (
        patch.object(cute, "compile", compile_kernel),
        patch.object(cute.runtime, "load_module", load_cached),
    ):
        from resin_models_cuda._scoring_cutedsl import score_logits

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for k in (0, 32):
                logits, ids, options = _inputs(2, 32768, dtype, k, True)
                _check(score_logits(logits, ids, **options), logits, ids, options)
        torch.cuda.synchronize()
    return {
        **_metadata(),
        "pid": os.getpid(),
        "quack_cache_dir": os.environ.get("QUACK_CACHE_DIR"),
        "quack_cache_enabled": os.environ.get("QUACK_CACHE_ENABLED"),
        "cute_compile_calls": calls,
        "cute_module_loads": loads,
    }


@torch.inference_mode()
def run(*, smoke=False, comparison: str | None = None):
    from resin_models_cuda._scoring_cutedsl import score_logits

    report = {
        **_metadata(),
        "compile_audit": [] if smoke else _audit(score_logits),
        "records": [],
    }
    backends = {"production": score_logits}
    if comparison:
        spec = importlib.util.spec_from_file_location("scoring_comparisons", comparison)
        if spec is None or spec.loader is None:
            raise ValueError("cannot import scoring comparison module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backends.update(module.backends())
        report["comparison_source_sha256"] = module.source_hashes()
    cases = (
        [(2, 32768, k, torch.bfloat16, True) for k in (0, 1, 5, 32)]
        if smoke
        else [
            (rows, vocab, k, torch.bfloat16, mixed)
            for rows, vocab in ((1, 128256), (8, 128256), (32, 128256), (128, 262144))
            for k in (0, 1, 5, 32)
            for mixed in (False, True)
        ]
        + [
            (1, 128256, k, dtype, False)
            for dtype in (torch.float16, torch.float32)
            for k in (0, 5, 32)
        ]
        + [
            (128, 262144, k, dtype, mixed)
            for dtype, k in ((torch.float32, 0), (torch.float16, 32))
            for mixed in (False, True)
        ]
        + [(1, 128256, 32, torch.bfloat16, False, requested) for requested in (1, 5)]
    )
    for rows, vocab, k, dtype, mixed, *request in cases:
        width = 4 if rows == 128 else 1
        logits, ids, options = _inputs(rows, vocab, dtype, k, mixed, width=width)
        requested = request[0] if request else k
        if requested != k:
            options["top_counts"].clamp_max_(requested)
        functions = {
            name: lambda function=function: function(logits, ids, **options)
            for name, function in backends.items()
        }
        for function in functions.values():
            _check(function(), logits, ids, options)
        timings = _timings(functions)
        for name, function in functions.items():
            record = {
                "backend": name,
                "rows": rows,
                "width": width,
                "vocab": vocab,
                "k": k,
                "requested_k": requested,
                "dtype": str(dtype),
                "mixed": mixed,
                "gpu_kernels": timings[name],
                **_allocation(function),
            }
            report["records"].append(record)
            print("[score] " + json.dumps(record), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache-probe", action="store_true")
    parser.add_argument(
        "--comparison", help="optional module exposing backends() and source_hashes()"
    )
    parser.add_argument("--output", default="profiles/scoring/production.json")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = (
        cache_probe()
        if args.cache_probe
        else run(smoke=args.smoke, comparison=args.comparison)
    )
    destination.write_text(json.dumps(report, indent=2) + "\n")
