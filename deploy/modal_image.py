"""Build the shared Modal image used by Resin GPU deployments."""

from __future__ import annotations

from pathlib import Path

import modal

_DEPENDENCY_ROOT = "/opt/resin-deps"
_SOURCE_ROOT = "/opt/resin"
_SYSTEM_PATH = (
    "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_PYTHON_SOURCE_IGNORE = ("**/__pycache__", "**/*.py[co]")


def build_modal_image(
    root: Path,
    *,
    env: dict[str, str] | None = None,
    cuda_source_ignore: tuple[str, ...] = (),
) -> modal.Image:
    """Build the common CUDA image and install one deployment's source tree.

    Modal's built-in ``Image.uv_sync`` currently rejects uv workspaces. Copying
    only the workspace manifests keeps dependency installation cacheable while
    retaining ``pyproject.toml`` and ``uv.lock`` as the single source of truth.
    """

    return (
        modal.Image.from_registry(
            "nvidia/cuda:13.3.1-devel-ubuntu22.04", add_python="3.12"
        )
        .apt_install(
            "build-essential",
            "curl",
            "git",
            "ninja-build",
            "pciutils",
            "software-properties-common",
            "libdw-dev",
            "libelf-dev",
        )
        .run_commands(
            "add-apt-repository -y ppa:ubuntu-toolchain-r/test",
            "apt-get update && apt-get install -y gcc-13 g++-13",
            "git config --global url.https://github.com/.insteadOf git@github.com:",
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --profile minimal",
        )
        .env(
            {
                "PATH": f"/root/.cargo/bin:{_SYSTEM_PATH}",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "MAX_JOBS": "16",
                "CMAKE_BUILD_PARALLEL_LEVEL": "16",
                "CUDA_HOME": "/usr/local/cuda",
                "CC": "gcc-13",
                "CXX": "g++-13",
                "NVCC_PREPEND_FLAGS": "-ccbin /usr/bin/g++-13",
            }
        )
        .pip_install("uv==0.12.5")
        .add_local_file(
            root / "pyproject.toml",
            f"{_DEPENDENCY_ROOT}/pyproject.toml",
            copy=True,
        )
        .add_local_file(root / "uv.lock", f"{_DEPENDENCY_ROOT}/uv.lock", copy=True)
        .add_local_file(
            root / "models/cuda/pyproject.toml",
            f"{_DEPENDENCY_ROOT}/models/cuda/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            root / "models/torch/pyproject.toml",
            f"{_DEPENDENCY_ROOT}/models/torch/pyproject.toml",
            copy=True,
        )
        .run_commands(
            # FlashMLA's post-build spill checker probes PCI devices even on
            # CPU builders. Modal hides PCI sysfs even in GPU-backed builds.
            # Give this build an empty inventory for the upstream build hook.
            "mkdir -p /tmp/resin-build-bin",
            "printf '%s\\n' '#!/bin/sh' "
            "'exec /usr/bin/lspci -F /dev/null \"$@\"' "
            "> /tmp/resin-build-bin/lspci",
            "chmod +x /tmp/resin-build-bin/lspci",
            "PATH=/tmp/resin-build-bin:$PATH uv sync "
            f"--project {_DEPENDENCY_ROOT} "
            "--frozen --no-install-workspace --no-default-groups "
            "--group deployment --compile-bytecode",
            "rm /tmp/resin-build-bin/lspci",
            "rmdir /tmp/resin-build-bin",
        )
        .env(
            {"PATH": (f"{_DEPENDENCY_ROOT}/.venv/bin:/root/.cargo/bin:{_SYSTEM_PATH}")}
        )
        # These accelerator-specific supplement wheels live outside PyPI.
        .uv_pip_install(
            "flashinfer-cubin==0.7.1",
            extra_index_url="https://flashinfer.ai/whl",
        )
        .uv_pip_install(
            "flashinfer-jit-cache==0.7.1+cu130",
            extra_index_url="https://flashinfer.ai/whl/cu130",
        )
        # Keep dependency layers independent of local source edits.
        .add_local_file(
            root / "deploy/__init__.py", "/root/deploy/__init__.py", copy=True
        )
        .add_local_file(
            root / "deploy/jit_cache.py", "/root/deploy/jit_cache.py", copy=True
        )
        .add_local_file(
            root / "deploy/modal_image.py", "/root/deploy/modal_image.py", copy=True
        )
        .add_local_file(
            root / "deploy/checkpoint_cache.py",
            "/root/deploy/checkpoint_cache.py",
            copy=True,
        )
        .add_local_file(root / "deploy/models.py", "/root/deploy/models.py", copy=True)
        .add_local_file(root / "Cargo.toml", f"{_SOURCE_ROOT}/Cargo.toml", copy=True)
        .add_local_file(root / "Cargo.lock", f"{_SOURCE_ROOT}/Cargo.lock", copy=True)
        .add_local_file(
            root / "pyproject.toml", f"{_SOURCE_ROOT}/pyproject.toml", copy=True
        )
        .add_local_file(root / "README.md", f"{_SOURCE_ROOT}/README.md", copy=True)
        .add_local_file(root / "LICENSE", f"{_SOURCE_ROOT}/LICENSE", copy=True)
        .add_local_file(root / "NOTICE", f"{_SOURCE_ROOT}/NOTICE", copy=True)
        .add_local_dir(root / "docs", f"{_SOURCE_ROOT}/docs", copy=True)
        .add_local_dir(root / "crates", f"{_SOURCE_ROOT}/crates", copy=True)
        .add_local_dir(
            root / "resin",
            f"{_SOURCE_ROOT}/resin",
            copy=True,
            ignore=list(_PYTHON_SOURCE_IGNORE),
        )
        .add_local_file(
            root / "models/torch/pyproject.toml",
            f"{_SOURCE_ROOT}/models/torch/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            root / "models/torch/resin_models_torch/__init__.py",
            f"{_SOURCE_ROOT}/models/torch/resin_models_torch/__init__.py",
            copy=True,
        )
        # uv resolves declared workspace sources even for --no-deps builds.
        # Only the MLX manifest is needed; its package and runtime stay excluded.
        .add_local_file(
            root / "models/mlx/pyproject.toml",
            f"{_SOURCE_ROOT}/models/mlx/pyproject.toml",
            copy=True,
        )
        .add_local_dir(
            root / "models/cuda",
            f"{_SOURCE_ROOT}/models/cuda",
            copy=True,
            ignore=["tests", *_PYTHON_SOURCE_IGNORE, *cuda_source_ignore],
        )
        .uv_pip_install(_SOURCE_ROOT, extra_options="--no-deps")
        .uv_pip_install(f"{_SOURCE_ROOT}/models/torch", extra_options="--no-deps")
        .uv_pip_install(f"{_SOURCE_ROOT}/models/cuda", extra_options="--no-deps")
        .env(env or {})
    )


__all__ = ["build_modal_image"]
