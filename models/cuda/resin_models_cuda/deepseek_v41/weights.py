"""Stream checkpoint-shaped tensors into prepared V4.1 attention/EP shards."""

from pathlib import Path
from typing import Any

import torch

from resin.backend.safetensors import SafetensorReader, indexed_tensor_files


def tensor_parallel_axis(name: str) -> int | None:
    """Return the axis used by DeepSeek's published TP=4 conversion."""

    if name in {"embed.weight", "head.weight"} or name.endswith(".attn_sink"):
        return 0
    if any(
        f".{component}." in name
        for component in ("wq_b", "wo_a", "weights_proj", "markov_w1", "markov_w2")
    ):
        return 0
    if ".wo_b." in name:
        return 1
    return None


def _shard_routed_expert(tensor: Any, name: str, model: Any):
    """Slice the logical intermediate axis, including packed weights/scales."""

    module_path, separator, expert_path = name.rpartition(".experts.")
    if not separator:
        if not name.startswith("experts."):
            return tensor
        module_path, expert_path = "", name.removeprefix("experts.")
    projection = expert_path.split(".")[1]
    axis = {"w1": 0, "w3": 0, "w2": 1}.get(projection)
    if axis is None:
        return tensor
    source = model.get_submodule(module_path) if module_path else model
    shard = getattr(source, "expert_intermediate_shard", None)
    if shard is None:
        return tensor
    total = source.routed_intermediate_size
    width = tensor.shape[axis]
    if total % width:
        raise ValueError(
            f"checkpoint tensor {name!r} has invalid expert block geometry"
        )
    packing = total // width
    if shard.start % packing or shard.size % packing:
        raise ValueError(f"checkpoint tensor {name!r} crosses an expert TP block")
    return tensor.narrow(
        axis, shard.start // packing, shard.size // packing
    ).contiguous()


def _coerce_for_target(tensor: Any, target: Any):
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is not None and target.dtype == fp4 and tensor.dtype == torch.int8:
        tensor = tensor.view(fp4)
    elif tensor.dtype != target.dtype:
        tensor = tensor.to(dtype=target.dtype)
    if tuple(tensor.shape) != tuple(target.shape):
        raise ValueError(
            f"checkpoint shape {tuple(tensor.shape)} does not match model "
            f"shape {tuple(target.shape)}"
        )
    return tensor


def _attention_shard(tensor, name, rank, size):
    axis = (
        0
        if name.endswith(("markov_head.embed.weight", "markov_head.head.weight"))
        else tensor_parallel_axis(name)
    )
    if axis is None or size == 1:
        return tensor
    if tensor.shape[axis] % size:
        raise ValueError(f"{name} does not divide attention TP={size}")
    width = tensor.shape[axis] // size
    return tensor.narrow(axis, rank * width, width)


def load_weights(model_path, model, *, rank, tensor_parallel_size):
    if not 0 <= rank < tensor_parallel_size:
        raise ValueError("attention rank is outside its group")
    files = indexed_tensor_files(Path(model_path), require_index=True)
    targets = dict(model.state_dict())
    if missing := set(targets) - files.keys():
        raise ValueError(f"checkpoint is missing V4.1 tensors: {sorted(missing)[:8]}")
    count = total = 0
    with SafetensorReader() as reader, torch.no_grad():
        for name in sorted(targets, key=lambda key: (files[key].name, key)):
            tensor = reader.read(files[name], name)
            target = targets[name]
            if ".engram.embed." in name:
                embedding = model.get_submodule(name.rsplit(".", 1)[0])
                width = target.shape[0]
                tensor = tensor[embedding.start : embedding.stop]
                # Last row-shard is padded exactly as in the official converter.
                if tensor.shape[0] < width:
                    tensor = torch.cat(
                        (
                            tensor,
                            tensor.new_full(
                                (width - tensor.shape[0], tensor.shape[1]),
                                1 if name.endswith("scale") else 0,
                            ),
                        )
                    )
            else:
                tensor = _attention_shard(tensor, name, rank, tensor_parallel_size)
                tensor = _shard_routed_expert(tensor, name, model)
            tensor = _coerce_for_target(tensor, target)
            target.copy_(tensor)
            total += target.numel() * target.element_size()
            count += 1
    return {"tensors": count, "bytes": total}
