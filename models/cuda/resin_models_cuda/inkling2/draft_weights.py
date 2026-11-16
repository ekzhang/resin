"""Strict, rank-local loading of Inkling2's bundled BF16 MTP checkpoint."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from resin.backend.safetensors import SafetensorReader
from resin_models_cuda.inkling2.dspark import DSparkDraft, DSparkLayer


def draft_tensors(model: DSparkDraft) -> dict[str, tuple[Tensor, int | None]]:
    """Checkpoint name -> destination view and optional TP-sharded source axis."""
    tensors: dict[str, tuple[Tensor, int | None]] = {
        name: (parameter, None)
        for name, parameter in model.named_parameters()
        if not name.startswith("layers.")
    }
    for index, layer in enumerate(cast(Sequence[DSparkLayer], model.layers)):
        prefix = f"layers.{index}."
        for name in ("dflare_norm", "input_layernorm", "post_attention_layernorm"):
            tensors[prefix + name + ".weight"] = (getattr(layer, name).weight, None)
        attn = layer.self_attn
        for name in ("q_norm", "k_norm", "k_norm_ctx"):
            tensors[prefix + "self_attn." + name + ".weight"] = (
                getattr(attn, name).weight,
                None,
            )
        q, k, v = attn.qkv_proj.weight.split(
            (attn.q_width, attn.kv_width, attn.kv_width)
        )
        kc, vc = attn.kv_proj_ctx.weight.chunk(2)
        for name, parameter in (
            ("q_proj", q),
            ("k_proj", k),
            ("v_proj", v),
            ("k_proj_ctx", kc),
            ("v_proj_ctx", vc),
        ):
            tensors[prefix + "self_attn." + name + ".weight"] = (parameter, 0)
        tensors[prefix + "self_attn.o_proj.weight"] = (attn.o_proj.weight, 1)
        gate, up = layer.gate_up_proj.weight.chunk(2)
        for name, parameter, axis in (
            ("gate_proj", gate, 0),
            ("up_proj", up, 0),
            ("down_proj", layer.down_proj.weight, 1),
        ):
            tensors[prefix + "mlp." + name + ".weight"] = (parameter, axis)
    return tensors


@torch.no_grad()
def load_draft_weights(model_path: str | Path, model: DSparkDraft) -> dict[str, int]:
    from safetensors import safe_open

    path = Path(model_path).expanduser().resolve() / "mtp.safetensors"
    tensors = draft_tensors(model)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        actual = set(handle.keys())
    if actual != tensors.keys():
        raise ValueError(
            f"Inkling2 MTP tensors differ: missing={sorted(tensors.keys() - actual)}, "
            f"unexpected={sorted(actual - tensors.keys())}"
        )
    loaded_bytes = 0
    with SafetensorReader() as reader:
        for name, (destination, axis) in tensors.items():
            shape = list(destination.shape)
            intervals = [slice(None)] * len(shape)
            if axis is not None:
                width = shape[axis]
                shape[axis] *= model.parallel.size
                intervals[axis] = slice(
                    model.parallel.rank * width, (model.parallel.rank + 1) * width
                )
            source = reader.read_slice(path, name, tuple(intervals), shape=tuple(shape))
            if source.dtype != torch.bfloat16 or destination.dtype != torch.bfloat16:
                raise ValueError(f"{name}: Inkling2 MTP requires BF16 weights")
            destination.copy_(source)
            loaded_bytes += source.numel() * source.element_size()
    model.prepare()
    return {"tensors": len(tensors), "bytes": loaded_bytes}
