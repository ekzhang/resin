"""Test-only stubs for optional CUDA packages unavailable on macOS."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

if importlib.util.find_spec("quack") is None:
    quack = ModuleType("quack")
    rotary = ModuleType("quack.rotary")

    def _unavailable(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("Quack is unavailable in this test environment")

    setattr(quack, "rmsnorm", _unavailable)
    setattr(rotary, "apply_rotary", _unavailable)
    sys.modules["quack"] = quack
    sys.modules["quack.rotary"] = rotary

if importlib.util.find_spec("flashinfer") is None:
    flashinfer = ModuleType("flashinfer")
    comm = ModuleType("flashinfer.comm")
    mnnvl = ModuleType("flashinfer.comm.mnnvl")
    gemm = ModuleType("flashinfer.gemm")

    def _flashinfer_unavailable(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("FlashInfer is unavailable in this test environment")

    setattr(gemm, "tinygemm_bf16", _flashinfer_unavailable)

    class _AllReduceFusionPattern:
        kAllReduce = 0
        kARResidualRMSNorm = 1

    class _AllReduceFusionWorkspace:
        backend = "unavailable"

    class _TorchDistBackend:
        def __init__(self, group: object = None) -> None:
            self.group = group

    setattr(comm, "AllReduceFusionPattern", _AllReduceFusionPattern)
    setattr(comm, "AllReduceFusionWorkspace", _AllReduceFusionWorkspace)
    setattr(comm, "allreduce_fusion", _flashinfer_unavailable)
    setattr(comm, "create_allreduce_fusion_workspace", _flashinfer_unavailable)
    setattr(mnnvl, "TorchDistBackend", _TorchDistBackend)
    sys.modules["flashinfer"] = flashinfer
    sys.modules["flashinfer.comm"] = comm
    sys.modules["flashinfer.comm.mnnvl"] = mnnvl
    sys.modules["flashinfer.gemm"] = gemm
