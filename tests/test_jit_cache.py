from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from deploy.jit_cache import (
    _LOCAL_BUILD_PATH_ENV_KEYS,
    _PATH_ENV_KEYS,
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MOUNT,
)


class TestKernelCacheContract:
    def test_every_runtime_cache_is_routed_below_the_volume_mount(self) -> None:
        for key in _PATH_ENV_KEYS:
            assert Path(KERNEL_CACHE_ENV[key]).is_relative_to(KERNEL_CACHE_MOUNT)

    def test_flashinfer_build_workspace_is_local_not_durable(self) -> None:
        for key in _LOCAL_BUILD_PATH_ENV_KEYS:
            path = Path(KERNEL_CACHE_ENV[key])
            assert path.is_absolute()
            assert not path.is_relative_to(KERNEL_CACHE_MOUNT)

    def test_required_opt_in_caches_are_enabled(self) -> None:
        assert KERNEL_CACHE_ENV["USE_PYTORCH_KERNEL_CACHE"] == "1"
        assert KERNEL_CACHE_ENV["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] == "1"
        assert KERNEL_CACHE_ENV["QUACK_CACHE_ENABLED"] == "1"

    def test_environment_override_cannot_escape_mount(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Import locally so the test never needs Modal or a mounted volume.
        from deploy.jit_cache import ensure_kernel_cache_dirs

        monkeypatch.setenv("CUDA_CACHE_PATH", str(tmp_path / "outside"))
        with pytest.raises(RuntimeError, match="escaped"):
            ensure_kernel_cache_dirs()

    def test_deployment_uses_one_abi_scoped_namespace(self) -> None:
        from deploy.jit_cache import (
            KERNEL_CACHE_ABI,
            KERNEL_CACHE_ROOT,
        )

        assert Path(KERNEL_CACHE_ROOT) == Path(KERNEL_CACHE_MOUNT) / KERNEL_CACHE_ABI
        assert KERNEL_CACHE_ABI.startswith("v12-blake3-")


def test_flashinfer_artifact_maps_scratch_path_to_durable_namespace(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from deploy.jit_cache import _flashinfer_durable_artifact

    workspace = tmp_path / "scratch"
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(workspace))
    spec = SimpleNamespace(jit_library_path=workspace / "module" / "kernel.so")
    assert _flashinfer_durable_artifact(spec) == Path("flashinfer/module/kernel.so")
    spec.jit_library_path = tmp_path / "outside" / "kernel.so"
    with pytest.raises(RuntimeError, match="escaped its local workspace"):
        _flashinfer_durable_artifact(spec)


def test_compiler_namespace_preparation_tolerates_concurrent_rank(
    monkeypatch, tmp_path
):
    from deploy import jit_cache

    mount = tmp_path / "volume"
    root = mount / "abi"
    monkeypatch.setattr(jit_cache, "KERNEL_CACHE_MOUNT", str(mount))
    monkeypatch.setattr(jit_cache, "KERNEL_CACHE_ROOT", str(root))
    for key in jit_cache._PATH_ENV_KEYS:
        monkeypatch.setenv(key, str(root / key))
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path / "scratch"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    original = Path.symlink_to

    def concurrent_create(path, target, **options):
        original(path, target, **options)
        raise FileExistsError("another rank won the creation race")

    monkeypatch.setattr(Path, "symlink_to", concurrent_create)
    jit_cache.ensure_kernel_cache_dirs()
    trace_link = tmp_path / "home/.cache/flashinfer/trace_apply/solutions"
    assert trace_link.is_symlink()
    assert trace_link.resolve() == root / "flashinfer-trace-apply"


def test_native_cache_key_uses_exposed_isa_even_with_generic_model_name():
    from deploy.jit_cache import _host_cpu_cache_key

    def key(flags):
        return _host_cpu_cache_key(f"model name: unknown\nflags: {flags}\n", "x86_64")

    assert key("avx2 fma") == key("fma avx2")
    assert key("avx2 fma") != key("avx2 fma avx512f")
    assert _host_cpu_cache_key("model name: unknown", "x86_64") is None


def test_shared_native_objects_are_scoped_before_model_import(monkeypatch, tmp_path):
    from deploy import jit_cache
    from deploy.models import get_model, prepare_runtime

    spec = replace(get_model("gemma4-e2b"), kernel_mount=str(tmp_path))
    for backend in ("QUACK", "FLASH_ATTENTION_CUTE_DSL"):
        monkeypatch.setenv(f"{backend}_CACHE_DIR", str(tmp_path / backend))
        monkeypatch.setenv(f"{backend}_CACHE_ENABLED", "1")
    monkeypatch.setenv("CUDA_CACHE_PATH", str(tmp_path / "cuda-driver"))

    for cpu in ("avx512", "avx2", "avx2", None):
        monkeypatch.setattr(
            jit_cache, "_host_cpu_cache_key", lambda *args, key=cpu: key
        )
        prepare_runtime(spec)
        for backend in ("QUACK", "FLASH_ATTENTION_CUTE_DSL"):
            if cpu:
                scoped = tmp_path / backend / "host-cpu-v1" / cpu
                assert os.environ[f"{backend}_CACHE_DIR"] == str(scoped)
                assert scoped.is_dir()
            else:
                assert os.environ[f"{backend}_CACHE_ENABLED"] == "0"
        assert os.environ["CUDA_CACHE_PATH"] == str(tmp_path / "cuda-driver")
