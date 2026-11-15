"""Resolve and stream Google's Gemma 4 31B assistant checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.loading import (
    assign_model_tensor,
    prepare_model,
    read_config,
    resolve_hub_model,
)
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.models.gemma4.assistant import (
    Gemma4AssistantAttention,
    Gemma4AssistantConfig,
    Gemma4AssistantForCausalLM,
    Gemma4AssistantMLP,
)
from resin_models_cuda.dense_parallel import shard_dense_mlp
from resin_models_cuda.gemma4.parallel import attention_geometry, shard_attention

DEFAULT_ASSISTANT = "google/gemma-4-31B-it-assistant"
DEFAULT_ASSISTANT_REVISION = "627c5ec1458b9086b841a91e0512fd31fd2fbbf1"


def resolve_assistant_path(model: str | Path, revision: str | None = None) -> Path:
    return resolve_hub_model(
        model,
        revision,
        default=(DEFAULT_ASSISTANT, DEFAULT_ASSISTANT_REVISION),
        files=("config.json", "model.safetensors"),
    )


def build_meta_assistant(
    raw_config: dict,
) -> tuple[Gemma4AssistantForCausalLM, Gemma4AssistantConfig]:
    config = Gemma4AssistantConfig.from_dict(raw_config)
    with torch.device("meta"):
        model = Gemma4AssistantForCausalLM(config)
    return model, config


def load_assistant(
    model_path: str | Path,
    *,
    device: str | torch.device,
    tensor_parallel: TensorParallelContext | None = None,
    attention_parallel: TensorParallelContext | None = None,
) -> tuple[Gemma4AssistantForCausalLM, Gemma4AssistantConfig]:
    root = Path(model_path).expanduser().resolve()
    raw_config = read_config(root)
    model, config = build_meta_assistant(raw_config)
    if attention_parallel is not None:
        attention_geometry(config, attention_parallel.layout)
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("the Gemma 4 assistant checkpoint can only be loaded on CUDA")
    tensor_files = indexed_tensor_files(root)
    expected = set(model.state_dict())
    actual = set(tensor_files)
    if missing := expected - actual:
        raise ValueError(
            f"assistant checkpoint is missing tensors: {sorted(missing)[:8]}"
        )
    if unexpected := actual - expected:
        raise ValueError(
            f"assistant checkpoint has unexpected tensors: {sorted(unexpected)[:8]}"
        )
    with SafetensorReader() as reader:
        for key, path in tensor_files.items():
            assign_model_tensor(model, key, reader.read(path, key), device="cpu")
    model.materialize_buffers(torch.device("cpu"))

    def lower(module):
        if attention_parallel is not None and isinstance(
            module, Gemma4AssistantAttention
        ):
            return shard_attention(module, attention_parallel)
        if tensor_parallel is not None and isinstance(module, Gemma4AssistantMLP):
            return shard_dense_mlp(module, tensor_parallel)
        return module

    model = cast(
        Gemma4AssistantForCausalLM,
        prepare_model(model, lower).to(target),
    )
    model.eval()
    return model, config


__all__ = [
    "DEFAULT_ASSISTANT",
    "DEFAULT_ASSISTANT_REVISION",
    "build_meta_assistant",
    "load_assistant",
    "resolve_assistant_path",
]
