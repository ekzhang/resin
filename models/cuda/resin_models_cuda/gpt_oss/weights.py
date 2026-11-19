"""Stream GPT-OSS weights into the final rank-local model layout."""

from __future__ import annotations

from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from resin.backend.loading import assign_model_tensor
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin_models_cuda.gpt_oss.mxfp4 import (
    EXPERT_CHECKPOINT_NAMES,
    CheckpointShards,
    KernelExperts,
)

if TYPE_CHECKING:
    from resin_models_cuda.gpt_oss.model import MoEForCausalLM


@torch.no_grad()
def load_gpt_oss_weights(
    root: Path, model: MoEForCausalLM, *, device: torch.device, shards: CheckpointShards
) -> None:
    """Load Q/K/V directly into packed storage and repack one expert layer at a time."""

    files = indexed_tensor_files(root)
    ordinary = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if ".mlp.experts." not in name
    }
    packed_sources = {}
    experts = []
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        experts.append(
            {name: f"{prefix}.mlp.experts.{name}" for name in EXPERT_CHECKPOINT_NAMES}
        )
        attention = layer.self_attn
        bounds = tuple(accumulate(attention.qkv_widths, initial=0))
        for name, _ in attention.qkv_proj.named_parameters():
            target = f"{prefix}.self_attn.qkv_proj.{name}"
            for projection, start, stop in zip(
                ("q_proj", "k_proj", "v_proj"), bounds[:-1], bounds[1:], strict=True
            ):
                source = f"{prefix}.self_attn.{projection}.{name}"
                packed_sources[source] = (target, slice(start, stop))

    ordinary_sources = (
        ordinary.keys() - {target for target, _ in packed_sources.values()}
        | packed_sources.keys()
    )
    expected = ordinary_sources | {key for layer in experts for key in layer.values()}
    if missing := expected - files.keys():
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:8]}")
    if unexpected := files.keys() - expected:
        raise ValueError(f"checkpoint has unexpected tensors: {sorted(unexpected)[:8]}")

    packed = {}
    with SafetensorReader() as reader:
        for key in sorted(ordinary_sources, key=lambda key: (files[key], key)):
            source = shards(key, reader.read(files[key], key))
            if key not in packed_sources:
                assign_model_tensor(
                    model, key, source, device=device, dtype=torch.bfloat16
                )
                continue
            target, interval = packed_sources[key]
            if target not in packed:
                packed[target] = torch.empty_like(
                    ordinary[target], device=device, dtype=torch.bfloat16
                )
                assign_model_tensor(model, target, packed[target], device=device)
            destination = packed[target][interval]
            if destination.shape != source.shape:
                raise ValueError(
                    f"shape mismatch for {key!r}: expected {destination.shape}, got {source.shape}"
                )
            destination.copy_(source.to(dtype=destination.dtype))

        for layer, keys in zip(model.model.layers, experts, strict=True):
            cast(KernelExperts, layer.mlp.experts).load_weights(
                {
                    name: shards(key, reader.read(files[key], key)).to(device=device)
                    for name, key in keys.items()
                }
            )
