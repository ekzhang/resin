"""Tokenizer-free GPT-OSS checkpoint resolution and model loading."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.loading import (
    MODEL_FILES,
    read_config,
    resolve_dtype,
    resolve_model,
)
from resin.models.gpt_oss.config import GptOssConfig
from resin_models_cuda.gpt_oss.flashinfer import MXFP4Config
from resin_models_cuda.gpt_oss.model import MoEForCausalLM
from resin_models_cuda.gpt_oss.mxfp4 import (
    EXPERT_CHECKPOINT_NAMES,
    KernelExperts,
    shard_checkpoint_tensor,
    shard_model_config,
)
from resin_models_cuda.gpt_oss.weights import load_gpt_oss_weights

if TYPE_CHECKING:
    from resin.backend.topology import ParallelGroups

DEFAULT_MODEL = "openai/gpt-oss-20b"
MODEL_20B = "openai/gpt-oss-20b"
MODEL_120B = "openai/gpt-oss-120b"
MODEL_REVISIONS = {
    MODEL_20B: "6cee5e81ee83917806bbde320786a8fb61efebee",
    MODEL_120B: "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
}


def _download_model(
    repo_id: str,
    revision: str | None,
    *,
    kernel_moe=None,
) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    # Inspect the tiny config before asking the Hub for tens of gigabytes of
    # packed weights. Without an explicit kernel, fail before shard download.
    config_file = Path(
        hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
    )
    _check_quantization(
        json.loads(config_file.read_text()),
        kernel_moe=kernel_moe,
    )
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(MODEL_FILES),
        )
    )


def resolve_model_path(
    model: str | Path,
    revision: str | None,
    *,
    kernel_moe=None,
) -> Path:
    repo_id = str(model)
    selected_revision = MODEL_REVISIONS.get(repo_id) if revision is None else revision
    return resolve_model(
        model,
        selected_revision,
        lambda repository, selected: _download_model(
            repository,
            selected,
            kernel_moe=kernel_moe,
        ),
    )


def _check_quantization(
    config: dict[str, Any],
    *,
    kernel_moe=None,
) -> None:
    checkpoint_config = config.get("quantization_config") or config.get("quantization")
    method = checkpoint_config.get("quant_method") if checkpoint_config else None
    if method != "mxfp4":
        raise ValueError("GPT-OSS CUDA inference requires an MXFP4 checkpoint")
    if kernel_moe is None:
        raise NotImplementedError(
            "the released GPT-OSS checkpoint uses MXFP4 expert weights; "
            "pass an imported CUDA implementation as kernel_moe"
        )


def load_model(
    model_path: str | Path,
    *,
    device: str | torch.device,
    dtype: str | torch.dtype | None = None,
    kernel_moe=None,
    tensor_parallel: TensorParallelLayout | None = None,
    parallel_groups: ParallelGroups | None = None,
) -> tuple[MoEForCausalLM, dict[str, Any]]:
    root = Path(model_path).expanduser().resolve()
    config = read_config(root)
    args = GptOssConfig.from_dict(config)
    _check_quantization(
        config,
        kernel_moe=kernel_moe,
    )
    if parallel_groups is not None:
        group = parallel_groups.attn_tp
        attention_layout = TensorParallelLayout(rank=group.local_rank, size=group.size)
        if tensor_parallel is not None and tensor_parallel != attention_layout:
            raise ValueError("tensor_parallel does not match the attention TP group")
        tensor_parallel = attention_layout
        expert_parallel = TensorParallelLayout(
            rank=parallel_groups.ep.local_rank, size=parallel_groups.ep.size
        )
        expert_tensor_parallel = TensorParallelLayout(
            rank=parallel_groups.expert_tp.local_rank,
            size=parallel_groups.expert_tp.size,
        )
    else:
        expert_parallel = None
        tensor_parallel = tensor_parallel or TensorParallelLayout()
        expert_tensor_parallel = tensor_parallel
    target = torch.device(device)
    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype not in {None, torch.bfloat16}:
        raise ValueError("the packed SM90 GPT-OSS path requires bfloat16 activations")
    local_args = shard_model_config(
        args,
        tensor_parallel,
        expert_tensor_parallel=expert_tensor_parallel,
    )
    expert_args = (
        replace(
            local_args,
            num_local_experts=expert_parallel.even_shard(
                args.num_local_experts, name="num_experts"
            ).size,
        )
        if expert_parallel is not None
        else local_args
    )
    kernel_config = MXFP4Config.from_model(expert_args)
    assert kernel_moe is not None
    kernel_moe.validate_device(target)
    kernel_moe.validate(kernel_config)
    with torch.device("meta"):
        model = MoEForCausalLM(local_args)
    for layer in model.model.layers:
        layer.mlp.experts = KernelExperts(
            local_args,
            kernel_moe,
            kernel_config,
            parallel_groups=parallel_groups,
        )

    load_gpt_oss_weights(
        root,
        model,
        device=device,
        dtype=torch.bfloat16,
        num_hidden_layers=args.num_hidden_layers,
        expert_tensor_names=EXPERT_CHECKPOINT_NAMES,
        transform_tensor=lambda key, tensor: shard_checkpoint_tensor(
            key,
            tensor,
            args=args,
            tensor_parallel=tensor_parallel,
            expert_parallel=expert_parallel,
            expert_tensor_parallel=expert_tensor_parallel,
        ),
        install_expert_layer=lambda layer, tensors: cast(
            KernelExperts, model.model.layers[layer].mlp.experts
        ).load_weights(tensors),
    )

    model.materialize_buffers(target)
    unprepared = [
        index
        for index, layer in enumerate(model.model.layers)
        if not layer.mlp.experts.is_prepared
    ]
    if unprepared:
        raise RuntimeError(f"packed expert layers were not prepared: {unprepared}")
    remaining_meta = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    if remaining_meta:
        raise RuntimeError(
            "ordinary model parameters remain on meta: " + ", ".join(remaining_meta[:8])
        )
    model.eval()
    return model, config
