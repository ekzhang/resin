"""Resolve, shard, and prepare NVIDIA's Gemma 4 NVFP4 checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.loading import (
    MODEL_FILES,
    assign_model_tensor,
    prepare_model,
    read_config,
    resolve_hub_model,
)
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.backend.torch import use_default_dtype
from resin.models.gemma4.config import Gemma4Nvfp4Config
from resin_models_cuda.dense_parallel import shard_dense_mlp
from resin_models_cuda.gemma4.model import (
    Gemma4Attention,
    Gemma4ForConditionalGeneration,
    Gemma4MLP,
)
from resin_models_cuda.gemma4.parallel import attention_geometry, shard_attention
from resin_models_cuda.nvfp4 import prepare_nvfp4_modules, select_nvfp4_backend

DEFAULT_MODEL = "nvidia/Gemma-4-31B-IT-NVFP4"
DEFAULT_REVISION = "4135a98a9b728a548947683219633b25682223ac"

PROCESSOR_FILES = (
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def resolve_model_path(model: str | Path, revision: str | None = None) -> Path:
    return resolve_hub_model(
        model,
        revision,
        default=(DEFAULT_MODEL, DEFAULT_REVISION),
        files=(*MODEL_FILES, *PROCESSOR_FILES),
    )


def build_meta_model(
    raw_config: dict[str, Any],
) -> tuple[Gemma4ForConditionalGeneration, Gemma4Nvfp4Config]:
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config,
        Gemma4VisionConfig,
    )

    config = Gemma4Nvfp4Config.from_dict(raw_config)
    transformers_config = Gemma4Config.from_dict(raw_config)
    vision_config = transformers_config.vision_config
    assert isinstance(vision_config, Gemma4VisionConfig)
    vision_config._attn_implementation = "sdpa"
    with use_default_dtype(torch.bfloat16):
        with torch.device("meta"):
            model = Gemma4ForConditionalGeneration(config, transformers_config)
    return model, config


def load_model(
    model_path: str | Path,
    *,
    device: str | torch.device,
    nvfp4_backend: str = "auto",
    tensor_parallel: TensorParallelContext | None = None,
    attention_parallel: TensorParallelContext | None = None,
) -> tuple[Gemma4ForConditionalGeneration, Gemma4Nvfp4Config, dict[str, Any]]:
    root = Path(model_path).expanduser().resolve()
    raw_config = read_config(root)
    model, config = build_meta_model(raw_config)
    if attention_parallel is not None:
        attention_geometry(config, attention_parallel.layout)
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("the Gemma 4 NVFP4 checkpoint can only be loaded on CUDA")
    tensor_files = indexed_tensor_files(root, require_index=True)
    expected = set(model.state_dict())
    actual = set(tensor_files)
    ignored_missing = {
        "model.vision_tower.encoder.rotary_emb.inv_freq",
        "model.vision_tower.encoder.rotary_emb.original_inv_freq",
    }
    if missing := expected - actual - ignored_missing:
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:8]}")
    if unexpected := actual - expected:
        raise ValueError(f"checkpoint has unexpected tensors: {sorted(unexpected)[:8]}")

    with SafetensorReader() as reader:
        for key, path in tensor_files.items():
            assign_model_tensor(model, key, reader.read(path, key), device="cpu")
    # Recreate nonpersistent language/vision rotary buffers before moving the
    # loaded model. Only rank-local weights should ever occupy device memory.
    model.materialize_buffers(torch.device("cpu"))

    def lower(module):
        if attention_parallel is not None and isinstance(module, Gemma4Attention):
            return shard_attention(module, attention_parallel)
        if tensor_parallel is not None and isinstance(module, Gemma4MLP):
            return shard_dense_mlp(module, tensor_parallel)
        return module

    prepared = prepare_model(model, lower).to(target)
    model.prepare_attention()
    model = cast(
        Gemma4ForConditionalGeneration,
        prepare_nvfp4_modules(
            prepared,
            select_nvfp4_backend(nvfp4_backend, torch.backends.cudnn.version()),
        ),
    )
    model.eval()
    return model, config, raw_config


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "build_meta_model",
    "load_model",
    "resolve_model_path",
]
