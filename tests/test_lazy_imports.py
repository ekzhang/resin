from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


class TestImport:
    def test_import_resin_does_not_import_accelerator_stacks(self) -> None:
        code = (
            "import sys; sys.path[:0] = ['models/cuda', 'models/torch']; "
            "import resin; import resin.backend; "
            "from resin import Parallelism; Parallelism(world_size=4, attn_dp=4); "
            "import resin.backend.cuda; import resin_models_cuda; "
            "import resin_models_torch; "
            "print(*(int(name in sys.modules) for name in "
            "('torch', 'flashinfer', 'flash_attn')))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        assert result.stdout.strip() == "0 0 0"

    def test_torch_is_required_and_cuda_dependencies_are_optional(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
        assert project["dependencies"] == [
            "blake3>=1.0,<2",
            "numpy>=1.26,<3",
            "torch>=2.13.0",
        ]
        optional_dependencies = project["optional-dependencies"]
        assert "torch" not in optional_dependencies
        assert "demo" not in optional_dependencies
        assert "dev" not in optional_dependencies

        cuda = optional_dependencies["cuda"]
        assert cuda == [
            "flashinfer-python>=0.6.15.post1; sys_platform == 'linux'",
            "cuda-bindings>=13.3,<14; sys_platform == 'linux'",
        ]
        assert "gpu" not in optional_dependencies

        workspace = tomllib.loads(Path("pyproject.toml").read_text())
        dev = workspace["dependency-groups"]["dev"]
        assert "resin-models-torch" in dev
        assert "resin-models-cuda" in dev
        assert any(item.startswith("tokenizers") for item in dev)
        assert not any(item.startswith("hf-xet") for item in dev)
        overrides = workspace["tool"]["uv"]["override-dependencies"]
        assert all("sys_platform == 'linux'" in item for item in overrides)

        cuda_models = tomllib.loads(Path("models/cuda/pyproject.toml").read_text())[
            "project"
        ]
        cuda_dependencies = cuda_models["dependencies"]
        assert "resin[cuda]==0.1.0" in cuda_dependencies
        assert "resin-models-torch==0.1.0" in cuda_dependencies
        assert "huggingface-hub>=0.35" in cuda_dependencies
        assert "safetensors>=0.5" in cuda_dependencies
        assert any(
            item.startswith("flashinfer-python==0.6.15.post1")
            for item in cuda_dependencies
        )
        assert any(
            item.startswith("quack-kernels==0.6.4") for item in cuda_dependencies
        )
        assert any(
            item.startswith("nvidia-cutlass-dsl==4.6.2") for item in cuda_dependencies
        )
        assert any(
            item.startswith("flash-attn-4==4.0.0b26") for item in cuda_dependencies
        )
        assert not any(item.startswith("hf-xet") for item in cuda_dependencies)

        torch_models = tomllib.loads(Path("models/torch/pyproject.toml").read_text())[
            "project"
        ]
        assert "resin==0.1.0" in torch_models["dependencies"]
        assert any(item.startswith("torch>=") for item in torch_models["dependencies"])
        assert any(
            item.startswith("huggingface-hub>=")
            for item in torch_models["dependencies"]
        )
        assert any(
            item.startswith("safetensors>=") for item in torch_models["dependencies"]
        )
