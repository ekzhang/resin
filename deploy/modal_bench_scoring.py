"""Qualify and benchmark the production scorer without model checkpoints.

uv run modal run deploy/modal_bench_scoring.py --mode all --gpus A100-80GB,H200,B200
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import modal

from deploy.modal_image import build_modal_image

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/opt/resin")
if modal.is_local():
    _snapshot = tempfile.TemporaryDirectory(prefix="resin-scoring-")
    source_root = Path(_snapshot.name)
    for name in (
        "Cargo.toml",
        "Cargo.lock",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "NOTICE",
    ):
        shutil.copy2(ROOT / name, source_root / name)
    for name in ("deploy", "crates", "docs", "resin", "models", "tests"):
        shutil.copytree(
            ROOT / name,
            source_root / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
        )
    (source_root / "scripts").mkdir()
    shutil.copy2(
        ROOT / "scripts/bench_scoring.py", source_root / "scripts/bench_scoring.py"
    )
else:
    source_root = ROOT
image = (
    build_modal_image(source_root)
    .uv_pip_install("pytest>=8,<9")
    .add_local_dir(source_root / "tests", "/opt/resin/tests", ignore=["__pycache__"])
    .add_local_dir(
        source_root / "models/cuda/tests",
        "/opt/resin/models/cuda/tests",
        ignore=["__pycache__"],
    )
    .add_local_file(
        source_root / "scripts/bench_scoring.py", "/opt/resin/scripts/bench_scoring.py"
    )
)
# Optional, ignored experiment adapters never become a production dependency.
comparison_root = ROOT / "profiles/scoring/production-integration"
if modal.is_local() and comparison_root.is_dir():
    shutil.copytree(
        comparison_root,
        source_root / "comparisons",
        ignore=shutil.ignore_patterns("__pycache__", "results"),
    )
    image = image.add_local_dir(
        source_root / "comparisons", "/opt/scoring-comparisons", ignore=["__pycache__"]
    )
cache = modal.Volume.from_name(
    "resin-scoring-compiler-cache-v1", create_if_missing=True
)
app = modal.App("resin-scoring", image=image)


@app.function(cpu=16, memory=65536, timeout=30 * 60, volumes={"/cache/scoring": cache})
def execute(mode: str, compare: bool = False, test_filter: str = ""):
    import hashlib
    import os
    import subprocess
    import traceback

    import torch

    capability = torch.cuda.get_device_capability()
    compiler_root = Path("/cache/scoring") / f"sm{capability[0]}{capability[1]}"
    for variable, directory in (
        ("CUTE_DSL_CACHE_DIR", "cute"),
        ("QUACK_CACHE_DIR", "quack"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCH_EXTENSIONS_DIR", "torch"),
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
    ):
        destination = compiler_root / directory
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(destination)
    os.environ["QUACK_CACHE_ENABLED"] = "1"
    os.environ["PYTHONPATH"] = (
        "/opt/resin:/opt/resin/models/cuda:/opt/resin/models/torch"
    )
    extensions = list(
        Path("/opt/resin-deps/.venv/lib").glob("python*/site-packages/resin/_resin*.so")
    )
    if len(extensions) != 1:
        raise RuntimeError(
            f"expected one built Resin extension, found {len(extensions)}"
        )
    shutil.copy2(extensions[0], ROOT / "resin" / extensions[0].name)
    sources = [
        *ROOT.glob("models/cuda/resin_models_cuda/*scoring*.py"),
        *ROOT.glob("models/cuda/resin_models_cuda/*/cuda.py"),
        ROOT / "models/cuda/resin_models_cuda/gemma4/edge_cuda.py",
        ROOT / "resin/backend/cuda/completion.py",
        ROOT / "scripts/bench_scoring.py",
    ]
    tests = [
        "models/cuda/tests/test_logprobs_cuda.py",
        "models/cuda/tests/test_scoring_runtime_cuda.py",
        "models/cuda/tests/test_verified_sampling_blocks.py",
        "tests/test_sampling_completion_cuda.py",
        "tests/test_cuda_completion.py",
    ]
    sources += [ROOT / path for path in tests]
    sources += sorted(ROOT.glob("crates/**/*.rs"))
    report = {
        "mode": mode,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "exit_code": 0,
        "native_extension_sha256": hashlib.sha256(
            extensions[0].read_bytes()
        ).hexdigest(),
    }
    python = "/opt/resin-deps/.venv/bin/python"
    try:
        if mode in ("qualify", "all"):
            result = subprocess.run(
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    *tests,
                    *(["-k", test_filter] if test_filter else []),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            report["qualification"] = {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            report["exit_code"] = result.returncode
            print(result.stdout, flush=True)
            print(result.stderr, flush=True)
            if not result.returncode:
                probes = []
                for index in range(2):
                    destination = Path(f"/tmp/scoring-cache-probe-{index}.json")
                    subprocess.run(
                        [
                            python,
                            str(ROOT / "scripts/bench_scoring.py"),
                            "--cache-probe",
                            "--output",
                            str(destination),
                        ],
                        cwd=ROOT,
                        check=True,
                    )
                    probes.append(json.loads(destination.read_text()))
                report["fresh_process_cache"] = probes
                if probes[1]["cute_compile_calls"]:
                    raise RuntimeError(
                        "fresh worker recompiled CuTe scorer despite persistent QuACK cache"
                    )
                if not probes[1]["cute_module_loads"]:
                    raise RuntimeError("fresh worker did not load cached CuTe modules")
        if not report["exit_code"] and mode in ("smoke", "benchmark", "all"):
            destination = Path("/tmp/scoring-result.json")
            command = [
                python,
                str(ROOT / "scripts/bench_scoring.py"),
                "--output",
                str(destination),
            ]
            if mode == "smoke":
                command.append("--smoke")
            if compare:
                command.extend(
                    ("--comparison", "/opt/scoring-comparisons/comparisons.py")
                )
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            report["exit_code"] = result.returncode
            report["stdout"], report["stderr"] = result.stdout, result.stderr
            print(result.stdout, flush=True)
            print(result.stderr, flush=True)
            if result.returncode == 0:
                report["benchmark"] = json.loads(destination.read_text())
    except Exception:
        report["exit_code"], report["error"] = 1, traceback.format_exc()
    cache.commit()
    return report


@app.local_entrypoint()
def main(
    gpus: str = "A100-80GB,H200,B200",
    mode: str = "all",
    compare: bool = False,
    test_filter: str = "",
    output: str = "profiles/scoring/production-integration/results",
):
    if mode not in ("smoke", "qualify", "benchmark", "all"):
        raise ValueError("mode must be smoke, qualify, benchmark, or all")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    calls = [
        (gpu, execute.with_options(gpu=gpu).spawn(mode, compare, test_filter))
        for gpu in gpus.split(",")
    ]
    for gpu, call in calls:
        print(f"[scoring-call] {gpu} {call.object_id}", flush=True)
    failed = []
    for gpu, call in calls:
        report = {**call.get(), "modal_call_id": call.object_id}
        path = destination / f"{gpu.lower()}-{mode}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[scoring-saved] {path}", flush=True)
        if report["exit_code"]:
            failed.append(gpu)
    if failed:
        raise RuntimeError(
            f"scoring validation failed on {', '.join(failed)}; see saved reports"
        )
