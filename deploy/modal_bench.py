"""Modal transport for scripts/bench.py; measurement lives in resin.bench."""

from __future__ import annotations

import os
from pathlib import Path

import modal

from deploy.modal_image import build_modal_image
from deploy.models import MODEL_SPECS, get_model, image_environment

ROOT = Path(__file__).resolve().parents[1]


def _write_source(path: Path, source: str | bytes) -> None:
    """Replace a Python source and invalidate bytecode for fresh workers."""
    import importlib

    path.write_bytes(source.encode() if isinstance(source, str) else source)
    for bytecode in (path.parent / "__pycache__").glob(f"{path.stem}.*.pyc"):
        bytecode.unlink(missing_ok=True)
    importlib.invalidate_caches()


TRACE_VOLUME = modal.Volume.from_name(
    "resin-benchmark-traces-v1", create_if_missing=True
)
# Local-only presets (for example MLX) have no Modal volume.
MODEL_VOLUMES = {
    spec.model_mount: modal.Volume.from_name(
        spec.model_volume, version=spec.model_volume_version, create_if_missing=True
    )
    for spec in MODEL_SPECS.values()
    if spec.model_mount is not None and spec.model_volume is not None
}
KERNEL_VOLUMES = {
    spec.kernel_mount: modal.Volume.from_name(
        spec.kernel_volume, version=spec.kernel_volume_version, create_if_missing=True
    )
    for spec in MODEL_SPECS.values()
    if spec.kernel_mount is not None and spec.kernel_volume is not None
}
image = build_modal_image(ROOT).add_local_file(
    ROOT / "scripts/bench.py", "/root/scripts/bench.py", copy=True
)
if (runtime_sources := ROOT / "runtime_sources").is_dir():
    # Pure-Python runtime additions can extend a cached native image without
    # rebuilding the Rust extension or accelerator dependencies.
    image = image.add_local_dir(
        runtime_sources,
        "/opt/resin-deps/.venv/lib/python3.12/site-packages",
        copy=True,
    )
app = modal.App(
    "resin-bench",
    image=image,
    volumes={**MODEL_VOLUMES, **KERNEL_VOLUMES, "/bench-traces": TRACE_VOLUME},
)

qualification_image = (
    image.uv_pip_install("pytest>=8,<9")
    .add_local_dir(ROOT / "tests", "/opt/resin/tests", copy=True)
    .add_local_dir(
        ROOT / "models/cuda/tests", "/opt/resin/models/cuda/tests", copy=True
    )
)


@app.function(cpu=16, memory=32768, timeout=4 * 60 * 60)
def download(model: str, metadata_only: bool = False) -> dict:
    """Populate the same pinned checkpoint and marker used by serving."""
    from deploy.models import download_checkpoint

    spec = get_model(model)
    volume = MODEL_VOLUMES[str(spec.mount)]
    volume.reload()
    result = download_checkpoint(spec, metadata_only=metadata_only)
    volume.commit()
    return result


@app.function(cpu=32, memory=262144, timeout=4 * 60 * 60, single_use_containers=True)
def sweep(
    model: str,
    cases: list[dict],
    topology: dict,
    options: dict,
    *,
    module_source_overlays: dict[str, str] | None = None,
    required_hardware: str | None = None,
) -> list[dict]:
    return _run_sweep(
        model,
        cases,
        topology,
        options,
        module_source_overlays=module_source_overlays,
        required_hardware=required_hardware,
    )


def _run_sweep(
    model: str,
    cases: list[dict],
    topology: dict,
    options: dict,
    *,
    module_source_overlays: dict[str, str] | None = None,
    required_hardware: str | None = None,
) -> list[dict]:
    import hashlib
    import importlib.util
    import subprocess

    spec = get_model(model)
    # Set compiler namespaces before importing any concrete CUDA runner. Each
    # invocation gets a fresh container, so models cannot inherit another arch.
    os.environ.update(image_environment(spec))
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()
    print(f"[bench-hardware] {hardware}", flush=True)
    if required_hardware is not None and hardware != required_hardware:
        raise RuntimeError(
            f"hardware mismatch before model loading: required {required_hardware!r}, got {hardware!r}"
        )
    overlays = {}
    for module_name, source in (module_source_overlays or {}).items():
        if not module_name.startswith(("resin.", "resin_models_cuda.", "scripts.")):
            raise ValueError("benchmark overlays must belong to Resin modules")
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None or module_spec.origin is None:
            raise ValueError(f"unknown source module {module_name}")
        _write_source(Path(module_spec.origin), source)
        overlays[module_name] = hashlib.sha256(source.encode()).hexdigest()
    from deploy.models import prepare_runtime, require_checkpoint
    from resin import Parallelism
    from scripts.bench import run_benchmark

    MODEL_VOLUMES[str(spec.mount)].reload()
    kernel_volume = KERNEL_VOLUMES.get(spec.kernel_mount)
    if kernel_volume is not None:
        kernel_volume.reload()
    checkpoint = require_checkpoint(spec, draft_tokens=options["draft_tokens"])
    prepare_runtime(spec)
    options = dict(options)
    options["metadata"] = {
        **options.get("metadata", {}),
        "repo_id": spec.repo_id,
        "checkpoint_revision": spec.revision,
        "runtime_source_overlays": overlays,
    }
    try:
        return run_benchmark(
            model, checkpoint, cases, parallelism=Parallelism(**topology), **options
        )
    finally:
        if kernel_volume is not None:
            kernel_volume.commit()
        if options.get("trace_dir") or options.get("progress_path"):
            TRACE_VOLUME.commit()


@app.function(cpu=32, memory=262144, timeout=4 * 60 * 60, single_use_containers=True)
def sweep_sequence(
    model: str,
    cases: list[dict],
    topology: dict,
    options: dict,
    variants: list[dict],
    *,
    module_source_overlays: dict[str, str] | None = None,
    required_hardware: str | None = None,
) -> list[dict]:
    """Measure model variants sequentially on one physical GPU allocation.

    Each leg launches a fresh benchmark worker, load, and graph catalog. The
    common harness remains fixed; variant overlays may change accelerator model
    modules only, and every leg must explicitly supply the same module names.
    """
    import subprocess

    if not variants:
        raise ValueError("a benchmark sequence requires at least one variant")
    names = [variant["name"] for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError("benchmark sequence variant names must be unique")
    variant_modules = set(variants[0].get("module_source_overlays", {}))
    for variant in variants:
        overlays = variant.get("module_source_overlays", {})
        if set(overlays) != variant_modules:
            raise ValueError(
                "every sequence variant must supply the same source modules"
            )
        if any(not name.startswith("resin_models_cuda.") for name in overlays):
            raise ValueError(
                "sequence variants may overlay accelerator model modules only"
            )
    identity = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    results = []
    for index, variant in enumerate(variants):
        variant_options = {**options, **variant.get("options", {})}
        variant_options["metadata"] = {
            **options.get("metadata", {}),
            **variant.get("options", {}).get("metadata", {}),
            "benchmark_sequence": {
                "index": index,
                "name": variant["name"],
                "order": names,
                "gpu_uuid": identity,
            },
        }
        print(
            f"[bench-sequence] {index + 1}/{len(variants)} {variant['name']}",
            flush=True,
        )
        records = _run_sweep(
            model,
            cases,
            topology,
            variant_options,
            module_source_overlays={
                **(module_source_overlays or {}),
                **variant.get("module_source_overlays", {}),
            },
            required_hardware=required_hardware,
        )
        results.append({"name": variant["name"], "records": records})
    return results


@app.function(
    image=qualification_image,
    cpu=16,
    memory=131072,
    timeout=4 * 60 * 60,
    single_use_containers=True,
)
def qualify(
    model: str,
    test_paths: list[str],
    *,
    environment: dict[str, str] | None = None,
    module_source_overlays: dict[str, str] | None = None,
) -> dict:
    """Run ordinary numerical/graph regression tests against a pinned GPU model."""
    return _run_qualification(
        model,
        test_paths,
        environment=environment,
        module_source_overlays=module_source_overlays,
    )


def _run_qualification(
    model: str,
    test_paths: list[str],
    *,
    environment: dict[str, str] | None = None,
    module_source_overlays: dict[str, str] | None = None,
) -> dict:
    import hashlib
    import importlib.util
    import subprocess
    import time

    from deploy.models import prepare_runtime, require_checkpoint

    spec = get_model(model)
    os.environ.update(image_environment(spec))
    MODEL_VOLUMES[spec.model_mount].reload()
    kernel_volume = KERNEL_VOLUMES.get(spec.kernel_mount)
    if kernel_volume is not None:
        kernel_volume.reload()
    checkpoint = require_checkpoint(spec, draft_tokens=0)
    prepare_runtime(spec)
    os.environ.update(environment or {})
    os.environ["RESIN_TEST_CHECKPOINT"] = str(checkpoint)
    overlays = {}
    for module_name, source in (module_source_overlays or {}).items():
        if not module_name.startswith(("resin.", "resin_models_cuda.")):
            raise ValueError("qualification overlays must belong to Resin modules")
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None or module_spec.origin is None:
            raise ValueError(f"unknown source module {module_name}")
        _write_source(Path(module_spec.origin), source)
        overlays[module_name] = hashlib.sha256(source.encode()).hexdigest()
    started = time.perf_counter()
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "pytest",
            "-o",
            "pythonpath=",
            *(
                str(Path("/opt/resin") / path) if not path.startswith("-") else path
                for path in test_paths
            ),
            "-v",
            "--tb=short",
        ],
        cwd="/root",
        env={**os.environ, "PYTHONPATH": "/opt/resin/tests"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            print(line, end="", flush=True)
        process.wait()
    finally:
        if kernel_volume is not None:
            kernel_volume.commit()
    return {
        "model": model,
        "checkpoint_revision": spec.revision,
        "test_paths": test_paths,
        "returncode": process.returncode,
        "elapsed_s": time.perf_counter() - started,
        "output": "".join(output),
        "module_source_overlays": overlays,
    }


@app.function(
    image=qualification_image,
    cpu=16,
    memory=131072,
    timeout=4 * 60 * 60,
    single_use_containers=True,
)
def qualify_sequence(
    model: str,
    test_paths: list[str],
    variants: list[dict],
    *,
    environment: dict[str, str] | None = None,
    stop_on_failure: bool = True,
) -> list[dict]:
    """Run source variants through fresh pytest processes on one GPU."""
    import importlib.metadata
    import importlib.util
    import subprocess

    if not variants:
        raise ValueError("a qualification sequence requires at least one variant")
    module_names = set(variants[0].get("module_source_overlays", {}))
    if any(
        set(variant.get("module_source_overlays", {})) != module_names
        for variant in variants
    ):
        raise ValueError(
            "every qualification variant must supply the same source modules"
        )
    allocation = {
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "task_id": os.getenv("MODAL_TASK_ID"),
        "versions": {},
    }
    for package in ("torch", "flashinfer-python"):
        allocation["versions"][package] = importlib.metadata.version(package)
    originals = {}
    for name in module_names:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"unknown source module {name}")
        path = Path(spec.origin)
        originals[path] = path.read_bytes()
    results = []
    try:
        for variant in variants:
            print(f"[qualification-sequence] {variant['name']}", flush=True)
            previous_environment = dict(os.environ)
            try:
                result = _run_qualification(
                    model,
                    test_paths,
                    environment={
                        **(environment or {}),
                        **variant.get("environment", {}),
                    },
                    module_source_overlays=variant.get("module_source_overlays"),
                )
            finally:
                for key in os.environ.keys() - previous_environment.keys():
                    del os.environ[key]
                os.environ.update(previous_environment)
            results.append(
                {"name": variant["name"], "allocation": allocation, **result}
            )
            if result["returncode"] and stop_on_failure:
                break
        return results
    finally:
        for path, source in originals.items():
            _write_source(path, source)


def download_traces(remote_directory: str, output: str | Path) -> list[str]:
    """Retrieve raw rank traces and merge each prefill/decode case for Perfetto."""
    from scripts.merge_bench_traces import merge_traces

    prefix = Path(remote_directory).relative_to("/bench-traces").as_posix()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    cases: dict[str, list[Path]] = {}
    for entry in TRACE_VOLUME.iterdir(prefix):
        if not entry.path.endswith(".json.gz"):
            continue
        target = destination / Path(entry.path).name
        with target.open("wb") as stream:
            for chunk in TRACE_VOLUME.read_file(entry.path):
                stream.write(chunk)
        cases.setdefault(target.name.rsplit("-rank", 1)[0], []).append(target)
    merged = []
    for case, sources in sorted(cases.items()):
        target = destination / f"{case}-all-ranks.json.gz"
        merge_traces(sources, target)
        merged.append(str(target))
    if not merged:
        raise RuntimeError(f"no benchmark traces found in {remote_directory}")
    return merged
