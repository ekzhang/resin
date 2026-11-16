"""Checkpoint files and precision validation for Inkling 2 Small."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_target_path(model: str | Path) -> Path:
    root = Path(model).expanduser().resolve()
    required = ("config.json", "model.safetensors.index.json", "hf_quant_config.json")
    if missing := [name for name in required if not (root / name).is_file()]:
        raise FileNotFoundError(f"Inkling 2 checkpoint is missing files: {missing}")
    return root


def read_config(model: str | Path) -> dict:
    values = json.loads((Path(model) / "config.json").read_text())
    if not isinstance(values, dict):
        raise ValueError("Inkling 2 config.json must contain an object")
    return values


def load_precision(model: str | Path) -> dict[str, str]:
    """Validate expert activation formats instead of inferring them from repo names."""
    metadata = json.loads((Path(model) / "hf_quant_config.json").read_text())
    result = {}
    for group in metadata["model"]["parameter_groups"]:
        precision = group["compute_dtype"]
        quant = group.get("required_activation_quantization")
        if precision == "nvfp4" and quant != {
            "amax_type": "per_token",
            "block_size": 16,
            "format": "nvfp4_e2m1",
            "rounding": "nearest",
            "scale_dtype": "float8_e4m3fn",
            "scale_selection_rule": "fouroversix_mae",
        }:
            raise ValueError("unsupported Inkling 2 NVFP4 activation format")
        if precision == "mxfp8" and quant != {
            "block_size": 32,
            "format": "mxfp8_e4m3",
            "rounding": "nearest",
            "scale_dtype": "ue8m0",
        }:
            raise ValueError("unsupported Inkling 2 MXFP8 activation format")
        for name in group["tensors"]:
            if ".mlp.experts." in name and name.endswith("_weight"):
                if precision not in ("nvfp4", "mxfp8", "bfloat16") or name in result:
                    raise ValueError(f"invalid or duplicate expert precision: {name}")
                result[name] = "bf16" if precision == "bfloat16" else precision
    return result


__all__ = [
    "resolve_target_path",
    "read_config",
    "load_precision",
]
