"""Shared PyTorch model loading helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any, TypeVar, cast
from weakref import WeakKeyDictionary

import torch

MODEL_FILES = ("config.json", "model*.safetensors", "model.safetensors.index.json")
_ModelT = TypeVar("_ModelT", bound=torch.nn.Module)


def prepare_model(
    model: torch.nn.Module,
    lower: Callable[[torch.nn.Module], torch.nn.Module],
) -> torch.nn.Module:
    """Lower loaded checkpoint containers into execution-ready modules.

    Visit children first and preserve shared module identity. A rule returns
    ordinary modules unchanged, or replaces a checkpoint container with its
    prepared implementation. Weight layout and fusion belong in that rule;
    the executable model need not retain the checkpoint's module structure.
    """

    # Unchanged checkpoint children must not keep their backing storage alive
    # after a parent is replaced. Shared originals retain their replacement
    # until their last alias has been visited.
    prepared: WeakKeyDictionary[torch.nn.Module, torch.nn.Module | None] = (
        WeakKeyDictionary()
    )

    def visit(module: torch.nn.Module) -> torch.nn.Module:
        if module in prepared:
            replacement = prepared[module]
            return module if replacement is None else replacement
        for name, child in tuple(module._modules.items()):
            if child is not None:
                module._modules[name] = visit(child)
        replacement = lower(module)
        prepared[module] = None if replacement is module else replacement
        return replacement

    return visit(model)


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        raise TypeError("dtype must be a string, torch.dtype, or None")
    resolved = getattr(torch, dtype.removeprefix("torch."), None)
    if resolved is None or not isinstance(resolved, torch.dtype):
        raise ValueError(f"unknown torch dtype: {dtype!r}")
    return resolved


def resolve_model(
    model: str | Path,
    revision: str | None,
    download: Callable[[str, str | None], Path],
) -> Path:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate
    if candidate.exists():
        raise ValueError(f"model path is not a directory: {candidate}")
    return download(str(model), revision)


def read_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config: {config_path}")
    return json.loads(config_path.read_text())


def load_safetensor_weights(
    model_path: str | Path,
    model: torch.nn.Module,
    *,
    device: str | torch.device,
    dtype: str | torch.dtype | None = None,
    checkpoint_keys: Collection[str] | None = None,
    name_replacements: Mapping[str, str] | None = None,
    transform_tensor: Callable[[str, torch.Tensor], torch.Tensor] | None = None,
    require_index: bool = False,
) -> None:
    """Stream safetensors directly into a model's parameters and buffers.

    ``name_replacements`` maps checkpoint-name fragments to model-name
    fragments. Replacements apply in mapping order, so the same mechanism
    handles full names, prefixes, suffixes, and interior substrings.
    ``checkpoint_keys`` selects a partial checkpoint load when a model owns
    non-state tensors such as packed expert weights.
    """

    from resin.backend.safetensors import SafetensorReader, indexed_tensor_files

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    target = torch.device(device)
    if target.type == "meta":
        raise ValueError("safetensors must be loaded onto a material device")
    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype is not None and not resolved_dtype.is_floating_point:
        raise ValueError("model dtype must be floating point")

    replacements = tuple((name_replacements or {}).items())
    if any(not isinstance(old, str) or not old for old, _ in replacements):
        raise ValueError("safetensor name replacement keys must be non-empty strings")
    if any(not isinstance(new, str) for _, new in replacements):
        raise TypeError("safetensor name replacement values must be strings")

    tensor_files = indexed_tensor_files(root, require_index=require_index)
    if checkpoint_keys is None:
        selected_keys = tuple(tensor_files)
    else:
        selected_keys = tuple(dict.fromkeys(checkpoint_keys))
        if missing := set(selected_keys) - tensor_files.keys():
            raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:8]}")

    mapped: dict[str, tuple[str, Path]] = {}
    for checkpoint_name in selected_keys:
        model_name = checkpoint_name
        for old, new in replacements:
            model_name = model_name.replace(old, new)
        if model_name in mapped:
            raise ValueError(f"multiple checkpoint tensors map to {model_name!r}")
        mapped[model_name] = (checkpoint_name, tensor_files[checkpoint_name])

    model_names = set(model.state_dict())
    loaded_names = set(mapped)
    if checkpoint_keys is None and (missing := model_names - loaded_names):
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:8]}")
    if unexpected := loaded_names - model_names:
        raise ValueError(f"checkpoint has unexpected tensors: {sorted(unexpected)[:8]}")

    # Visit each shard once. Index files commonly order tensors by name rather
    # than by shard, which would otherwise reopen the same mmap repeatedly.
    by_file: dict[Path, list[tuple[str, str]]] = {}
    for model_name, (checkpoint_name, path) in mapped.items():
        by_file.setdefault(path, []).append((model_name, checkpoint_name))

    with SafetensorReader() as reader:
        for path, names in by_file.items():
            for model_name, checkpoint_name in names:
                tensor = reader.read(path, checkpoint_name)
                if transform_tensor is not None:
                    tensor = transform_tensor(checkpoint_name, tensor)
                assign_model_tensor(
                    model,
                    model_name,
                    tensor,
                    device=target,
                    dtype=resolved_dtype,
                )


def materialize_safetensor_model(
    model_path: Path,
    *,
    device: str | torch.device,
    dtype: str | torch.dtype | None,
    build: Callable[[], _ModelT],
    after_cast: Callable[[_ModelT], None] | None = None,
) -> _ModelT:
    """Build a meta model, assign safetensors, and finalize inference buffers."""

    target = torch.device(device)
    load_target = target if target.type in {"cpu", "cuda"} else torch.device("cpu")
    with torch.device("meta"):
        model = build()
    load_safetensor_weights(model_path, model, device=load_target)
    cast(Callable[[torch.device], None], model.materialize_buffers)(load_target)
    if target != load_target:
        model.to(target)

    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype is not None:
        if not resolved_dtype.is_floating_point:
            raise ValueError("model dtype must be floating point")
        model.to(dtype=resolved_dtype)
    if after_cast is not None:
        after_cast(model)
    # Module.to(dtype=...) casts floating-point buffers too. Each model's
    # materializer restores computed buffers such as RoPE frequencies.
    cast(Callable[[torch.device], None], model.materialize_buffers)(target)
    model.eval()
    return model


def assign_model_tensor(
    model: torch.nn.Module,
    key: str,
    tensor: torch.Tensor,
    *,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> None:
    """Validate and assign one parameter or buffer into a materialized module."""

    if "." in key:
        module_path, name = key.rsplit(".", 1)
        try:
            module = model.get_submodule(module_path)
        except AttributeError as error:
            raise ValueError(
                f"checkpoint tensor has no model target: {key!r}"
            ) from error
    else:
        module, name = model, key
    if name in module._parameters and module._parameters[name] is not None:
        existing, parameter = module._parameters[name], True
    elif name in module._buffers and module._buffers[name] is not None:
        existing, parameter = module._buffers[name], False
    else:
        raise ValueError(f"checkpoint tensor has no model target: {key!r}")
    existing = cast(torch.Tensor, existing)
    if tuple(existing.shape) != tuple(tensor.shape):
        raise ValueError(
            f"shape mismatch for {key!r}: checkpoint {tuple(tensor.shape)}, "
            f"model {tuple(existing.shape)}"
        )
    moved = tensor.to(
        device=device,
        dtype=dtype if dtype is not None and tensor.is_floating_point() else None,
    )
    if parameter:
        module._parameters[name] = torch.nn.Parameter(
            moved, requires_grad=existing.requires_grad
        )
    else:
        module._buffers[name] = moved
