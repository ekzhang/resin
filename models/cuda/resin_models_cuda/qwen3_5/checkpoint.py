"""Resolve and stream Qwen3.5-family checkpoints."""

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
    resolve_model,
)
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.backend.torch import use_default_dtype
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_cuda.dense_parallel import shard_dense_mlp
from resin_models_cuda.nvfp4 import prepare_nvfp4_modules, select_nvfp4_backend
from resin_models_cuda.qwen3_5.model import (
    Qwen3_5ForConditionalGeneration,
    Qwen35MLP,
)
from resin_models_cuda.qwen3_5.parallel import (
    prepare_attention_parallel,
    shard_attention_config,
)

DEFAULT_MODEL = "RadixArk/Qwen3.8-27B-NVFP4"
DEFAULT_REVISION = "554ebba9b5f1b79dc11246341960360e6ef05ef4"

TOKENIZER_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def _download_model(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[*MODEL_FILES, *TOKENIZER_FILES],
        )
    )


def resolve_model_path(model: str | Path, revision: str | None = None) -> Path:
    """Resolve a local directory or download a pinned Hub checkpoint."""

    selected = (
        DEFAULT_REVISION
        if revision is None and str(model) == DEFAULT_MODEL
        else revision
    )
    return resolve_model(model, selected, _download_model)


def load_config(model_path: str | Path) -> tuple[Qwen35Config, dict]:
    raw_config = read_config(Path(model_path))
    config = Qwen35Config.from_hugging_face(raw_config)
    config.require_dense()
    return config, raw_config


def is_modelopt_nvfp4(raw_config: dict[str, Any]) -> bool:
    """Recognize RadixArk's mixed FP8-projection/NVFP4-MLP contract."""

    quantization = raw_config.get("quantization_config") or {}
    groups = quantization.get("config_groups") or {}
    fp8 = groups.get("group_0") or {}
    nvfp4 = groups.get("group_1") or {}
    fp8_weights = fp8.get("weights") or {}
    nvfp4_weights = nvfp4.get("weights") or {}
    return (
        quantization.get("quant_method") == "modelopt"
        and quantization.get("quant_algo") == "MIXED_PRECISION"
        and fp8_weights.get("num_bits") == 8
        and fp8_weights.get("type") == "float"
        and nvfp4_weights.get("num_bits") == 4
        and nvfp4_weights.get("type") == "float"
        and nvfp4_weights.get("group_size") == 16
    )


def build_meta_model(
    raw_config: dict[str, Any], *, include_mtp: bool, include_vision: bool = False
) -> tuple[Qwen3_5ForConditionalGeneration, Qwen35Config]:
    config = Qwen35Config.from_hugging_face(raw_config)
    config.require_dense()
    if include_mtp:
        config.require_mtp()
    quantized = is_modelopt_nvfp4(raw_config)
    if raw_config.get("quantization_config") and not quantized:
        raise ValueError("unsupported Qwen3.5 checkpoint quantization contract")
    with use_default_dtype(torch.bfloat16), torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(
            config,
            include_mtp=include_mtp,
            vision_config=raw_config.get("vision_config") if include_vision else None,
            quantized=quantized,
        )
    return model, config


def _select_tensor_files(
    tensor_files: dict[str, Path],
    expected: set[str],
    *,
    include_mtp: bool,
    include_vision: bool = False,
) -> dict[str, Path]:
    checkpoint_scope = {
        key
        for key in tensor_files
        if key.startswith("model.language_model.")
        or key.startswith("lm_head.")
        or (include_mtp and key.startswith("mtp."))
        or (include_vision and key.startswith("model.visual."))
    }
    if missing := expected - checkpoint_scope:
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:8]}")
    if unexpected := checkpoint_scope - expected:
        raise ValueError(f"checkpoint has unexpected tensors: {sorted(unexpected)[:8]}")
    return {key: tensor_files[key] for key in expected}


def load_model(
    model_path: str | Path,
    *,
    device: str | torch.device,
    include_mtp: bool,
    include_vision: bool = True,
    nvfp4_backend: str = "auto",
    tensor_parallel: TensorParallelContext | None = None,
    attention_parallel: TensorParallelContext | None = None,
) -> tuple[Qwen3_5ForConditionalGeneration, Qwen35Config, dict[str, Any]]:
    """Stream the target, visual tower, and optional bundled MTP weights."""

    root = Path(model_path).expanduser().resolve()
    raw_config = read_config(root)
    model, config = build_meta_model(
        raw_config, include_mtp=include_mtp, include_vision=include_vision
    )
    if attention_parallel is not None:
        shard_attention_config(config, attention_parallel.layout)
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("the Qwen3.5 CUDA runner can only load weights on CUDA")
    if is_modelopt_nvfp4(raw_config):
        capability = torch.cuda.get_device_capability(target)
        if capability not in {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "Qwen3.5 NVFP4 weights require Blackwell-class FP4 hardware, "
                f"got sm{capability[0]}{capability[1]}"
            )
    expected = set(model.state_dict())
    tensor_files = _select_tensor_files(
        indexed_tensor_files(root, require_index=True),
        expected,
        include_mtp=include_mtp,
        include_vision=include_vision,
    )
    # Keep checkpoint storage mmap-backed until heads and FFNs are sharded.
    # Allocating the full model on every GPU first defeats TP's memory savings.
    host = torch.device("cpu")
    with SafetensorReader() as reader:
        for key, path in tensor_files.items():
            assign_model_tensor(model, key, reader.read(path, key), device=host)
    model.language_model.materialize_buffers(host)
    if model.visual is not None:
        rotary = model.visual.rotary_pos_emb
        rotary.inv_freq = 1.0 / (
            rotary.theta
            ** (
                torch.arange(0, rotary.dim, 2, dtype=torch.float32, device=host)
                / rotary.dim
            )
        )
    if attention_parallel is not None:
        prepare_attention_parallel(model, attention_parallel)
    prepared = prepare_model(
        model,
        lambda module: (
            shard_dense_mlp(module, tensor_parallel)
            if tensor_parallel is not None and isinstance(module, Qwen35MLP)
            else module
        ),
    )
    prepared.to(target)
    model = cast(
        Qwen3_5ForConditionalGeneration,
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
    "is_modelopt_nvfp4",
    "load_config",
    "load_model",
    "resolve_model_path",
]
