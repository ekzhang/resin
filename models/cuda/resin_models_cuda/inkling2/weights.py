"""Stream only this rank's Inkling 2 attention and expert weight slices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin_models_cuda.inkling2.model import DecoderLayer, Inkling2Target


@dataclass(frozen=True)
class WeightSlice:
    tensor: Tensor
    # Independent (axis, rank, size) intervals compose EP and expert TP.
    shards: tuple[tuple[int, int, int], ...] = ()

    @property
    def source_shape(self) -> tuple[int, ...]:
        shape = list(self.tensor.shape)
        for axis, _, size in self.shards:
            shape[axis] *= size
        return tuple(shape)

    @property
    def slices(self) -> tuple[slice, ...]:
        intervals = [slice(None)] * self.tensor.ndim
        for axis, rank, _ in self.shards:
            width = self.tensor.shape[axis]
            intervals[axis] = slice(rank * width, (rank + 1) * width)
        return tuple(intervals)


def target_weight_slices(target: Inkling2Target) -> dict[str, WeightSlice]:
    """Map checkpoint tensors into packed projections and EP/ETP ownership."""
    result = {}
    tp = target.tensor_parallel

    def add(name, tensor, axis=None, *, rank=tp.rank, size=tp.size):
        shards = () if axis is None else ((axis, rank, size),)
        result["model.llm." + name] = WeightSlice(tensor, shards)

    for name, tensor in (
        ("embed.weight", target.embed_tokens.weight),
        ("embed_norm.weight", target.embed_norm.weight),
        ("norm.weight", target.norm.weight),
        ("unembed.weight", target.unembed.weight),
    ):
        add(name, tensor)
    for name, parameter in target.attn_res.named_parameters():
        add("attn_res." + name, parameter)
    for index, layer in enumerate(cast(Sequence[DecoderLayer], target.layers)):
        prefix, attention = f"layers.{index}.", layer.attn
        names = (
            ("attn_norm",)
            if layer.local
            else ("q_input_norm", "k_input_norm", "v_input_norm")
        )
        for name in (*names, "mlp_norm"):
            add(prefix + name + ".weight", getattr(layer, name).weight)
        for name in ("attn_sconv", "mlp_sconv"):
            add(prefix + name + ".weight", getattr(layer, name).weight)
        projections = (
            "wq_du",
            "wk_dv",
            "wv_dv",
            "wr_du",
            "gate_attn",
            "attn_sink_proj",
        )
        widths = attention.split_sizes
        offset = 0
        for name, width in zip(projections, widths, strict=True):
            if not layer.local and name in ("wk_dv", "wv_dv"):
                destination = getattr(attention, name).weight
            else:
                destination = attention.qkvr.weight[offset : offset + width]
                offset += width
            add(prefix + "attn." + name + ".weight", destination, 0)
        add(prefix + "attn.wo_ud.weight", attention.wo_ud.weight, 1)
        add(prefix + "attn.q_l2_clip.scale", attention.q_scale, 0)
        add(prefix + "attn.rel_l2_clip.scale", attention.rel_scale, 0)
        add(prefix + "attn.rel_logits_proj.proj", attention.rel_proj)
        for name in ("k_sconv", "v_sconv"):
            conv = getattr(attention, name)
            if layer.local:
                add(prefix + f"attn.{name}.weight", conv.weight, 0)
            else:
                add(prefix + f"attn.{name}.static_w", conv.static_w, 1)
                add(prefix + f"attn.{name}.weight_proj", conv.weight_proj, 0)
        add(prefix + "mlp.gate.weight", layer.mlp.gate.weight)
        add(prefix + "mlp.gate.bias", layer.mlp.gate_bias)
        experts = layer.mlp.experts
        expert_count = experts.w13_weight.shape[0]
        expert_rank = experts.expert_start // expert_count
        expert_size = target.config.num_routed_experts // expert_count
        etp_rank = experts.expert_tp_rank
        etp_size = experts.expert_tp_size
        for name, tensor in experts.named_parameters(recurse=False):
            shards = (
                ()
                if name.endswith("outer_scale_denominator")
                else ((0, expert_rank, expert_size),)
            )
            if name.endswith(("_weight", "_weight_scale")):
                axis = 1 if name.startswith("w13_") else 2
                shards += ((axis, etp_rank, etp_size),)
            result["model.llm." + prefix + "mlp.experts." + name] = WeightSlice(
                tensor, shards
            )
    return result


def load_target_weights(target: Inkling2Target, model: str | Path) -> int:
    files = indexed_tensor_files(Path(model), require_index=True)
    destinations = target_weight_slices(target)
    present = {name for name in files if name.startswith("model.llm.")}
    if missing := destinations.keys() - present:
        raise ValueError(
            f"Inkling 2 checkpoint is missing target weights: {sorted(missing)}"
        )
    if extra := present - destinations.keys():
        raise ValueError(
            f"Inkling 2 checkpoint has unexpected target weights: {sorted(extra)}"
        )
    loaded_bytes = 0
    with torch.no_grad(), SafetensorReader() as reader:
        # Sequential shard access keeps one mmap open and bounds host/GPU staging.
        for name in sorted(destinations, key=lambda key: (str(files[key]), key)):
            destination = destinations[name]
            value = reader.read_slice(
                files[name], name, destination.slices, shape=destination.source_shape
            )
            destination.tensor.copy_(value)
            loaded_bytes += value.numel() * value.element_size()
    return loaded_bytes


__all__ = ["WeightSlice", "target_weight_slices", "load_target_weights"]
