"""Persistent compiler namespaces and artifact loading for Modal runtimes."""

from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path

KERNEL_CACHE_VOLUME_NAME = "resin-kernel-cache-v2"
KERNEL_CACHE_VOLUME_VERSION = 2
KERNEL_CACHE_MOUNT = "/cache/resin-kernels"
CUDA_KERNEL_ABI = (
    "cuda13.3.1-torch2.15.0-cu132-triton3.9.0-flashinfer0.7.1-"
    "fa4b40-cutlassdsl4.8.1-tvmffi0.1.12-quack0.6.5-cutile1.5.0"
)
KERNEL_CACHE_ABI = (
    f"v12-blake3-localpublish1-h100-sm90a-py312-{CUDA_KERNEL_ABI}-"
    "fijit0.7.1cu130-fa4emptyaux1-qkvpack-tinygemm-top4-fiar"
)
KERNEL_CACHE_ROOT = f"{KERNEL_CACHE_MOUNT}/{KERNEL_CACHE_ABI}"
FLASHINFER_BUILD_WORKSPACE = "/tmp/resin-flashinfer-build"
_KERNEL_ARTIFACT_CACHE = None


def _cache_dir(name: str) -> str:
    return f"{KERNEL_CACHE_ROOT}/{name}"


# Each runtime owns a separate compiler namespace inside the shared volume.
KERNEL_CACHE_ENV = {
    # NVIDIA's driver cache for PTX -> cubin compilation.
    "CUDA_CACHE_PATH": _cache_dir("cuda-driver"),
    "CUDA_CACHE_MAXSIZE": str(4 * 1024**3),
    "CUDA_CACHE_DISABLE": "0",
    "TORCH_CUDA_ARCH_LIST": "9.0a",
    # PyTorch eager Jiterator and C++/CUDA extensions.
    "USE_PYTORCH_KERNEL_CACHE": "1",
    "PYTORCH_KERNEL_CACHE_PATH": _cache_dir("pytorch-jiterator"),
    "TORCH_EXTENSIONS_DIR": _cache_dir("torch-extensions"),
    # torch.compile / Inductor and its Triton kernels.
    "TORCHINDUCTOR_CACHE_DIR": _cache_dir("torchinductor"),
    "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    "TRITON_CACHE_DIR": _cache_dir("triton"),
    # FlashInfer builds in local scratch. Resin publishes only a completed,
    # content-validated shared library into the durable ``flashinfer``
    # namespace; Ninja never runs against the mounted cache filesystem.
    "FLASHINFER_WORKSPACE_BASE": FLASHINFER_BUILD_WORKSPACE,
    "FLASHINFER_CUDA_ARCH_LIST": "9.0a",
    # FA4's CuTe-DSL object cache is opt-in in the pinned beta wheel.
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": _cache_dir("flash-attention-4"),
    "FLASH_ATTENTION_ARCH": "sm_90a",
    # Underlying CUTLASS DSL MLIR cache and persistent CuTe executables.
    # RMSNorm and Resin's scorer both use QuACK's compiled-object cache.
    "CUTE_DSL_CACHE_DIR": _cache_dir("cute-dsl"),
    "CUTE_DSL_ARCH": "sm_90a",
    "QUACK_CACHE_ENABLED": "1",
    "QUACK_CACHE_DIR": _cache_dir("quack"),
    # Hugging Face kernels source packages (not model weights).
    "KERNELS_CACHE": _cache_dir("hf-kernels"),
}

_PATH_ENV_KEYS = (
    "CUDA_CACHE_PATH",
    "PYTORCH_KERNEL_CACHE_PATH",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "CUTE_DSL_CACHE_DIR",
    "QUACK_CACHE_DIR",
    "KERNELS_CACHE",
)
_LOCAL_BUILD_PATH_ENV_KEYS = ("FLASHINFER_WORKSPACE_BASE",)


def _host_cpu_cache_key(cpuinfo: str, machine: str) -> str | None:
    """Identify the host instructions available to CuTe's native launcher."""
    first_cpu = cpuinfo.partition("\n\n")[0]
    fields = {}
    for line in first_cpu.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip().lower()] = value.strip()
    features = fields.get("flags") or fields.get("features")
    if not features:
        return None
    identity = [machine]
    for key in ("vendor_id", "cpu family", "model", "cpu implementer", "cpu part"):
        identity.append(fields.get(key, ""))
    identity.extend(sorted(set(features.split())))
    return hashlib.sha256("\n".join(identity).encode()).hexdigest()[:24]


def scope_host_object_caches(mount: str) -> None:
    """Select CPU-compatible QuACK and FA4 objects before importing either."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        cpuinfo = ""
    cpu_key = _host_cpu_cache_key(cpuinfo, platform.machine())
    for backend in ("QUACK", "FLASH_ATTENTION_CUTE_DSL"):
        setting = f"{backend}_CACHE_DIR"
        value = os.environ.get(setting)
        if not value:
            continue
        base = Path(value)
        if not base.is_relative_to(mount):
            continue
        if cpu_key is None:
            os.environ[f"{backend}_CACHE_ENABLED"] = "0"
            continue
        # Workers can inherit an already-scoped path from a launcher.
        if base.parent.name == "host-cpu-v1":
            base = base.parent.parent
        target = base / "host-cpu-v1" / cpu_key
        target.mkdir(parents=True, exist_ok=True)
        os.environ[setting] = str(target)


def ensure_kernel_cache_dirs() -> None:
    """Create every compiler-specific namespace before importing GPU libraries."""

    paths = []
    for key in _PATH_ENV_KEYS:
        path = Path(os.environ.get(key, KERNEL_CACHE_ENV[key]))
        if not path.is_relative_to(KERNEL_CACHE_MOUNT):
            raise RuntimeError(f"{key} escaped the kernel-cache mount: {path}")
        paths.append(path)

    root = Path(KERNEL_CACHE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    for key in _LOCAL_BUILD_PATH_ENV_KEYS:
        path = Path(os.environ.get(key, KERNEL_CACHE_ENV[key]))
        if not path.is_absolute() or path.is_relative_to(KERNEL_CACHE_MOUNT):
            raise RuntimeError(
                f"{key} must be an absolute local build workspace outside "
                f"the durable cache mount: {path}"
            )
        path.mkdir(parents=True, exist_ok=True)

    # FlashInfer's trace-apply solutions currently bypass its workspace-base
    # setting. Redirect that one hard-coded cache without moving HOME (which
    # would also move model downloads into the kernel volume).
    trace_target = root / "flashinfer-trace-apply"
    trace_target.mkdir(parents=True, exist_ok=True)
    trace_source = Path.home() / ".cache/flashinfer/trace_apply/solutions"
    trace_source.parent.mkdir(parents=True, exist_ok=True)
    try:
        trace_source.symlink_to(trace_target, target_is_directory=True)
    except FileExistsError:
        # Multiple serving ranks can prepare the same mounted namespace at once.
        if (
            not trace_source.is_symlink()
            or trace_source.resolve() != trace_target.resolve()
        ):
            raise RuntimeError(
                f"unexpected FlashInfer trace cache: {trace_source}"
            ) from None


def install_flashinfer_hash_cache() -> None:
    """Use a content-validated JIT library without consulting Ninja on hits.

    Modal Volume v2 preserves bytes but not nanosecond mtimes, while Ninja uses
    mtimes in its dependency database. FlashInfer 0.6.15 also invokes Ninja
    unconditionally before loading a runtime-built library. This wrapper stores
    a BLAKE3 sidecar after a normal miss and loads the library directly only
    when the exact ABI, module name, size, and content hash still match.
    """

    from flashinfer.jit import core

    from resin.kernel_cache import ArtifactCache, wrap_flashinfer_build_and_load

    global _KERNEL_ARTIFACT_CACHE
    if _KERNEL_ARTIFACT_CACHE is None:
        _KERNEL_ARTIFACT_CACHE = ArtifactCache(
            KERNEL_CACHE_MOUNT,
            KERNEL_CACHE_ABI,
        )
        if _KERNEL_ARTIFACT_CACHE.root != Path(KERNEL_CACHE_ROOT).resolve(strict=False):
            raise RuntimeError(
                "artifact cache root disagrees with deployment namespace"
            )

    setattr(
        core.JitSpec,
        "build_and_load",
        wrap_flashinfer_build_and_load(
            core.JitSpec.build_and_load,
            _KERNEL_ARTIFACT_CACHE,
            durable_artifact_resolver=_flashinfer_durable_artifact,
        ),
    )


def _flashinfer_durable_artifact(spec) -> Path:
    """Map one scratch-built FlashInfer library into persistent storage."""

    workspace = Path(
        os.environ.get(
            "FLASHINFER_WORKSPACE_BASE",
            KERNEL_CACHE_ENV["FLASHINFER_WORKSPACE_BASE"],
        )
    ).resolve(strict=False)
    source = Path(spec.jit_library_path).resolve(strict=False)
    try:
        relative = source.relative_to(workspace)
    except ValueError as error:
        raise RuntimeError(
            f"FlashInfer JIT artifact escaped its local workspace: {source}"
        ) from error
    return Path("flashinfer") / relative
