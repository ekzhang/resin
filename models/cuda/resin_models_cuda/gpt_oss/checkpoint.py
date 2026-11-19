"""Tokenizer-free GPT-OSS checkpoint resolution and model loading."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.loading import MODEL_FILES, read_config, resolve_model
from resin.models.gpt_oss.config import GptOssConfig
from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel, MXFP4Config
from resin_models_cuda.gpt_oss.model import MoEForCausalLM
from resin_models_cuda.gpt_oss.mxfp4 import CheckpointShards, KernelExperts
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


def _download_model(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    # Check the weight format before downloading tens of gigabytes of shards.
    config_file = Path(
        hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
    )
    _check_quantization(json.loads(config_file.read_text()))
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(MODEL_FILES),
        )
    )


def resolve_model_path(model: str | Path, revision: str | None) -> Path:
    repo_id = str(model)
    selected_revision = MODEL_REVISIONS.get(repo_id) if revision is None else revision
    return resolve_model(model, selected_revision, _download_model)


def _check_quantization(config: dict[str, Any]) -> None:
    checkpoint_config = config.get("quantization_config") or config.get("quantization")
    method = checkpoint_config.get("quant_method") if checkpoint_config else None
    if method != "mxfp4":
        raise ValueError("GPT-OSS CUDA inference requires an MXFP4 checkpoint")


def load_model(
    model_path: str | Path,
    *,
    device: str | torch.device,
    parallel_groups: ParallelGroups | None = None,
) -> tuple[MoEForCausalLM, dict[str, Any]]:
    root = Path(model_path).expanduser().resolve()
    config = read_config(root)
    args = GptOssConfig.from_dict(config)
    _check_quantization(config)

    def layout(group):
        return TensorParallelLayout(group.local_rank, group.size)

    shards = (
        CheckpointShards(args)
        if parallel_groups is None
        else CheckpointShards(
            args,
            layout(parallel_groups.attn_tp),
            expert_parallel=layout(parallel_groups.ep),
            expert_tensor_parallel=layout(parallel_groups.expert_tp),
        )
    )
    target = torch.device(device)
    local_args = shards.model_args
    kernel_config = MXFP4Config.from_model(shards.kernel_args)
    kernel = FlashInferSM90MXFP4Kernel()
    kernel.validate_device(target)
    kernel.validate(kernel_config)
    make_experts = partial(
        KernelExperts,
        local_args,
        kernel,
        kernel_config,
        parallel_groups=parallel_groups,
    )
    with torch.device("meta"):
        model = MoEForCausalLM(local_args, make_experts)

    load_gpt_oss_weights(root, model, device=target, shards=shards)

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
