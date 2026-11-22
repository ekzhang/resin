"""Load Inkling 2 target weights for MLX, keeping NVFP4/MXFP8 experts packed."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from resin.models.inkling2.config import Inkling2Config
from resin_models_mlx.inkling2.model import (
    Controller,
    Experts,
    Inkling2Weights,
    Layer,
)

PREFIX = "model.llm."

_NVFP4_ACTIVATIONS = {
    "amax_type": "per_token",
    "block_size": 16,
    "format": "nvfp4_e2m1",
    "rounding": "nearest",
    "scale_dtype": "float8_e4m3fn",
    "scale_selection_rule": "fouroversix_mae",
}
_MXFP8_ACTIVATIONS = {
    "block_size": 32,
    "format": "mxfp8_e4m3",
    "rounding": "nearest",
    "scale_dtype": "ue8m0",
}


def read_config(root: Path) -> Inkling2Config:
    values = json.loads((root / "config.json").read_text())
    if not isinstance(values, dict):
        raise ValueError("Inkling 2 config.json must contain an object")
    return Inkling2Config.from_hugging_face(values)


def expert_precision(root: Path) -> dict[str, str]:
    """Each expert weight's format, validated against its required activations."""
    metadata = json.loads((root / "hf_quant_config.json").read_text())
    result: dict[str, str] = {}
    for group in metadata["model"]["parameter_groups"]:
        precision = group["compute_dtype"]
        activations = group.get("required_activation_quantization")
        if precision == "nvfp4" and activations != _NVFP4_ACTIVATIONS:
            raise ValueError("unsupported Inkling 2 NVFP4 activation format")
        if precision == "mxfp8" and activations != _MXFP8_ACTIVATIONS:
            raise ValueError("unsupported Inkling 2 MXFP8 activation format")
        for name in group["tensors"]:
            if ".mlp.experts." in name and name.endswith("_weight"):
                if precision not in ("nvfp4", "mxfp8") or name in result:
                    raise ValueError(
                        f"unsupported or duplicate expert precision: {name}"
                    )
                result[name] = precision
    return result


def expected_tensors(config: Inkling2Config, precision: dict[str, str]) -> set[str]:
    names = {
        "embed.weight",
        "embed_norm.weight",
        "norm.weight",
        "unembed.weight",
        "attn_res.output_query",
    }
    loop = config.loop
    if loop is not None:
        names |= {
            "middle_cycle_cross_pass_queries",
            "middle_cycle_cross_pass_kv_queries",
        }
    for i in range(config.num_hidden_layers):
        local = i in config.local_layer_ids
        prefix = f"layers.{i}."
        if i:
            names.add(f"attn_res.attn_queries.{i - 1}")
        names.add(f"attn_res.mlp_queries.{i}")
        norms = (
            ("attn_norm",)
            if local
            else ("q_input_norm", "k_input_norm", "v_input_norm")
        )
        names |= {prefix + name + ".weight" for name in (*norms, "mlp_norm")}
        for name in (
            "wq_du",
            "wk_dv",
            "wv_dv",
            "wr_du",
            "gate_attn",
            "attn_sink_proj",
            "wo_ud",
        ):
            names.add(prefix + f"attn.{name}.weight")
        names |= {
            prefix + "attn.q_l2_clip.scale",
            prefix + "attn.rel_l2_clip.scale",
            prefix + "attn.rel_logits_proj.proj",
        }
        for name in ("k_sconv", "v_sconv"):
            if local:
                names.add(prefix + f"attn.{name}.weight")
            else:
                names |= {
                    prefix + f"attn.{name}.static_w",
                    prefix + f"attn.{name}.weight_proj",
                }
        names |= {prefix + "attn_sconv.weight", prefix + "mlp_sconv.weight"}
        names |= {prefix + "mlp.gate.weight", prefix + "mlp.gate.bias"}
        for matrix in ("w13", "w2"):
            base = prefix + f"mlp.experts.{matrix}_weight"
            names |= {base, base + "_scale"}
            if precision.get(PREFIX + base) == "nvfp4":
                names |= {base + "_amax", base + "_outer_scale_denominator"}
        if loop is not None and loop.core_start <= i < loop.core_end:
            for stream in ("attn", "k", "v", "mlp"):
                for part in ("down_weight", "gain_weight", "forget_weight"):
                    names.add(prefix + f"controllers.{stream}.{part}")
    return {PREFIX + name for name in names}


def _load_tensors(root: Path) -> dict[str, mx.array]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    files = sorted({str(name) for name in index["weight_map"].values()})
    tensors: dict[str, mx.array] = {}
    for name in files:
        loaded = mx.load(str(root / name))
        assert isinstance(loaded, dict)
        tensors.update(
            (key, value) for key, value in loaded.items() if key.startswith(PREFIX)
        )
    return tensors


def _experts(tensors: dict[str, mx.array], prefix: str, mode: str) -> Experts:
    def weight(name: str) -> mx.array:
        # NVFP4 bytes pack two codes (low nibble first); MXFP8 bytes are E4M3
        # codes. MLX reads both as packed uint32 words along the input axis.
        return tensors[prefix + name].view(mx.uint8).view(mx.uint32)

    experts = Experts(
        w13=weight("w13_weight"),
        w13_scales=tensors[prefix + "w13_weight_scale"].view(mx.uint8),
        w2=weight("w2_weight"),
        w2_scales=tensors[prefix + "w2_weight_scale"].view(mx.uint8),
        mode=mode,
    )
    if mode == "nvfp4":
        experts.w13_amax = tensors[prefix + "w13_weight_amax"].astype(mx.float32)
        experts.w2_amax = tensors[prefix + "w2_weight_amax"].astype(mx.float32)
        experts.w13_denominator = tensors[prefix + "w13_weight_outer_scale_denominator"]
        experts.w2_denominator = tensors[prefix + "w2_weight_outer_scale_denominator"]
    return experts


def load_weights(root: Path, config: Inkling2Config) -> Inkling2Weights:
    precision = expert_precision(root)
    tensors = _load_tensors(root)
    expected = expected_tensors(config, precision)
    if missing := expected - tensors.keys():
        raise ValueError(
            f"Inkling 2 checkpoint is missing target weights: {sorted(missing)}"
        )
    if extra := tensors.keys() - expected:
        raise ValueError(
            f"Inkling 2 checkpoint has unexpected target weights: {sorted(extra)}"
        )

    def get(name: str) -> mx.array:
        return tensors[PREFIX + name]

    layers = []
    loop = config.loop
    for i in range(config.num_hidden_layers):
        local = i in config.local_layer_ids
        prefix = f"layers.{i}."
        modes = {
            precision[PREFIX + prefix + f"mlp.experts.{matrix}_weight"]
            for matrix in ("w13", "w2")
        }
        if len(modes) != 1:
            raise ValueError(f"Inkling 2 layer {i} mixes expert precisions")
        norms = (
            ("attn_norm",)
            if local
            else ("q_input_norm", "k_input_norm", "v_input_norm")
        )
        attention = {
            name: get(prefix + f"attn.{name}.weight")
            for name in (
                "wq_du",
                "wk_dv",
                "wv_dv",
                "wr_du",
                "gate_attn",
                "attn_sink_proj",
            )
        }
        query = [
            attention[name]
            for name in ("wq_du", "wr_du", "gate_attn", "attn_sink_proj")
        ]
        if local:
            k_conv = get(prefix + "attn.k_sconv.weight").reshape(-1, 4)
            v_conv = get(prefix + "attn.v_sconv.weight").reshape(-1, 4)
            projection = mx.concatenate(
                [attention["wk_dv"], attention["wv_dv"], *query]
            )
            k_projection = v_projection = None
        else:
            k_conv = get(prefix + "attn.k_sconv.static_w").astype(mx.float32)
            v_conv = get(prefix + "attn.v_sconv.static_w").astype(mx.float32)
            projection = mx.concatenate(query)
            k_projection = mx.concatenate(
                [attention["wk_dv"], get(prefix + "attn.k_sconv.weight_proj")]
            )
            v_projection = mx.concatenate(
                [attention["wv_dv"], get(prefix + "attn.v_sconv.weight_proj")]
            )
        controllers = None
        if loop is not None and loop.core_start <= i < loop.core_end:
            controllers = {
                stream: Controller(
                    down=get(prefix + f"controllers.{stream}.down_weight"),
                    gain=get(prefix + f"controllers.{stream}.gain_weight"),
                    forget=get(prefix + f"controllers.{stream}.forget_weight"),
                )
                for stream in ("attn", "k", "v", "mlp")
            }
        layers.append(
            Layer(
                local=local,
                norms=tuple(get(prefix + name + ".weight") for name in norms),
                mlp_norm=get(prefix + "mlp_norm.weight"),
                projection=projection,
                k_projection=k_projection,
                v_projection=v_projection,
                wo=get(prefix + "attn.wo_ud.weight"),
                q_scale=get(prefix + "attn.q_l2_clip.scale"),
                rel_scale=get(prefix + "attn.rel_l2_clip.scale"),
                rel_proj=get(prefix + "attn.rel_logits_proj.proj"),
                k_conv=k_conv,
                v_conv=v_conv,
                attn_conv=get(prefix + "attn_sconv.weight").reshape(-1, 4),
                mlp_conv=get(prefix + "mlp_sconv.weight").reshape(-1, 4),
                # The router runs in FP32; keep one FP32 copy rather than casting per step.
                router=get(prefix + "mlp.gate.weight").astype(mx.float32),
                router_bias=get(prefix + "mlp.gate.bias").astype(mx.float32),
                experts=_experts(
                    tensors, PREFIX + prefix + "mlp.experts.", modes.pop()
                ),
                controllers=controllers,
            )
        )
    loop_queries = None
    if loop is not None:
        queries = get("middle_cycle_cross_pass_queries").astype(mx.float32)
        kv_queries = get("middle_cycle_cross_pass_kv_queries").astype(mx.float32)
        # Per core layer: query stream, key stream, value stream, MLP.
        loop_queries = mx.stack(
            [queries[0::2], kv_queries[0::2], kv_queries[1::2], queries[1::2]], axis=1
        )
    weights = Inkling2Weights(
        embed=get("embed.weight"),
        embed_norm=get("embed_norm.weight"),
        norm=get("norm.weight"),
        unembed=get("unembed.weight"),
        attn_queries=[
            get(f"attn_res.attn_queries.{i}").astype(mx.float32)
            for i in range(config.num_hidden_layers - 1)
        ],
        mlp_queries=[
            get(f"attn_res.mlp_queries.{i}").astype(mx.float32)
            for i in range(config.num_hidden_layers)
        ],
        output_query=get("attn_res.output_query").astype(mx.float32),
        layers=layers,
        loop_queries=loop_queries,
    )
    # Materialize every lazy load on this thread: a serving actor thread has
    # no CPU stream to run the deferred safetensors reads. One layer at a time
    # bounds the transient memory of the views.
    for layer in weights.layers:
        mx.eval(_arrays(layer))
    mx.eval(_arrays(weights))
    return weights


def _arrays(value) -> list[mx.array]:
    """Every MLX array reachable from a weights dataclass, list, tuple or dict."""
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (list, tuple)):
        return [array for item in value for array in _arrays(item)]
    if isinstance(value, dict):
        return [array for item in value.values() for array in _arrays(item)]
    if isinstance(value, (Layer, Experts, Controller)):
        return [
            array for name in value.__slots__ for array in _arrays(getattr(value, name))
        ]
    if isinstance(value, Inkling2Weights):
        return [
            array
            for name in value.__slots__
            if name != "layers"
            for array in _arrays(getattr(value, name))
        ]
    return []


def load_model(path: str | Path) -> tuple[Inkling2Config, Inkling2Weights]:
    root = Path(path).expanduser().resolve()
    for name in ("config.json", "model.safetensors.index.json", "hf_quant_config.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Inkling 2 checkpoint is missing {name}")
    config = read_config(root)
    return config, load_weights(root, config)


__all__ = [
    "expected_tensors",
    "expert_precision",
    "load_model",
    "load_weights",
    "read_config",
]
