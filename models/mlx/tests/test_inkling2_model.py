"""Inkling 2 MLX forward pass against the CUDA package's model on CPU PyTorch.

The reference is ``resin_models_cuda.inkling2.model.Inkling2Target`` loaded
from the same checkpoint files, with only its CUDA kernels replaced: dense
PyTorch attention and packed experts dequantized by the CUDA tests' independent
NVFP4/MXFP8 arithmetic.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuda" / "tests"))
from _inkling2_expert_reference import dequant, nvfp4_reference  # noqa: E402

from resin.models.inkling2.config import (  # noqa: E402
    Inkling2Config,
    Inkling2LoopConfig,
)
from resin_models_mlx.inkling2.checkpoint import load_model  # noqa: E402
from resin_models_mlx.inkling2.model import Inkling2Model  # noqa: E402

NVFP4_LAYERS = 4


def tiny_config() -> Inkling2Config:
    return Inkling2Config(
        vocab_size=320,
        unpadded_vocab_size=300,
        hidden_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=32,
        d_rel=4,
        rel_extent=12,
        local_layer_ids=(0, 1, 3, 4, 6),
        sliding_window_size=12,
        sconv_kernel_size=4,
        n_layers_per_attn_res_block=2,
        attn_res_normalize_values=True,
        intermediate_size=64,
        num_routed_experts=16,
        experts_per_token=8,
        route_scale=2.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        qk_l2_clip_threshold=3.0,
        rel_l2_clip_threshold=2.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        log_scaling_n_floor=16,
        log_scaling_alpha=0.1,
        final_logit_softcapping=30.0,
        eos_token_id=7,
        loop=Inkling2LoopConfig(core_start=2, core_end=6, num_passes=3, gate_rank=4),
    )


def _hf_config(config: Inkling2Config) -> dict:
    text = asdict(config)
    loop = text.pop("loop")
    text.update(
        torch_dtype="bfloat16",
        q_bias=False,
        o_bias=False,
        use_value_norm_no_gamma=True,
        n_shared_experts=0,
        use_gate_bias=True,
        n_routed_experts=config.num_routed_experts,
        num_experts_per_tok=config.experts_per_token,
        model_max_length=config.max_position_embeddings,
        loop_config={**loop, "is_varloop": False},
    )
    for name in ("num_routed_experts", "experts_per_token", "max_position_embeddings"):
        text.pop(name)
    eos = text.pop("eos_token_id")
    return dict(
        model_type="inkling2_loop",
        architectures=["Inkling2LoopForConditionalGeneration"],
        text_config=text,
        eos_token_id=eos,
    )


def _layer_mode(layer: int) -> str:
    return "nvfp4" if layer < NVFP4_LAYERS else "mxfp8"


def write_checkpoint(root: Path, config: Inkling2Config, seed: int = 0) -> None:
    """Random tensors in the published names, dtypes and packed expert layouts."""
    generator = torch.Generator().manual_seed(seed)
    h, d, r = config.hidden_size, config.head_dim, config.d_rel
    q, kv = config.num_attention_heads, config.num_key_value_heads
    experts, inter = config.num_routed_experts, config.intermediate_size
    loop = config.loop
    assert loop is not None
    tensors: dict[str, torch.Tensor] = {}

    def normal(name, shape, std=0.2, dtype=torch.bfloat16):
        tensors["model.llm." + name] = (
            torch.randn(shape, generator=generator) * std
        ).to(dtype)

    def positive(name, shape):
        tensors["model.llm." + name] = (
            1 + 0.2 * torch.randn(shape, generator=generator)
        ).to(torch.bfloat16)

    normal("embed.weight", (config.vocab_size, h), 1.0)
    positive("embed_norm.weight", (h,))
    positive("norm.weight", (h,))
    normal("unembed.weight", (config.vocab_size, h), 0.5)
    normal("attn_res.output_query", (h,), 1.0, torch.float32)
    core = 2 * (loop.core_end - loop.core_start)
    normal("middle_cycle_cross_pass_queries", (core, h), 1.0, torch.float32)
    normal("middle_cycle_cross_pass_kv_queries", (core, h), 1.0, torch.float32)
    for i in range(config.num_hidden_layers):
        local = i in config.local_layer_ids
        prefix = f"layers.{i}."
        if i:
            normal(
                f"attn_res.attn_queries.{i - 1}",
                (h,) if local else (3, h),
                1.0,
                torch.float32,
            )
        normal(f"attn_res.mlp_queries.{i}", (h,), 1.0, torch.float32)
        for name in (
            ("attn_norm",)
            if local
            else ("q_input_norm", "k_input_norm", "v_input_norm")
        ):
            positive(prefix + name + ".weight", (h,))
        positive(prefix + "mlp_norm.weight", (h,))
        for name, size in (
            ("wq_du", q * d),
            ("wk_dv", kv * d),
            ("wv_dv", kv * d),
            ("wr_du", q * r),
            ("gate_attn", q * d),
            ("attn_sink_proj", q),
        ):
            normal(prefix + f"attn.{name}.weight", (size, h), h**-0.5)
        normal(prefix + "attn.wo_ud.weight", (h, q * d), (q * d) ** -0.5)
        positive(prefix + "attn.q_l2_clip.scale", (q, d))
        positive(prefix + "attn.rel_l2_clip.scale", (q, r))
        normal(
            prefix + "attn.rel_logits_proj.proj",
            (r, config.sliding_window_size if local else config.rel_extent),
            0.5,
        )
        for name in ("k_sconv", "v_sconv"):
            if local:
                normal(prefix + f"attn.{name}.weight", (kv * d, 1, 4), 0.3)
            else:
                normal(prefix + f"attn.{name}.static_w", (4, kv * d), 0.3)
                normal(prefix + f"attn.{name}.weight_proj", (kv * 4, h), h**-0.5)
        for name in ("attn_sconv", "mlp_sconv"):
            normal(prefix + f"{name}.weight", (h, 1, 4), 0.3)
        normal(prefix + "mlp.gate.weight", (experts, h), h**-0.5)
        normal(prefix + "mlp.gate.bias", (experts,), 0.1, torch.float32)
        mode = _layer_mode(i)
        for matrix, (n, k) in (("w13", (2 * inter, h)), ("w2", (h, inter))):
            base = prefix + f"mlp.experts.{matrix}_weight"
            if mode == "nvfp4":
                tensors["model.llm." + base] = torch.randint(
                    0, 256, (experts, n, k // 2), generator=generator, dtype=torch.uint8
                )
                levels = torch.tensor([0.5, 0.75, 1.0, 1.5, 2.0])
                choice = torch.randint(0, 5, (experts, n, k // 16), generator=generator)
                tensors["model.llm." + base + "_scale"] = levels[choice].to(
                    torch.float8_e4m3fn
                )
                tensors["model.llm." + base + "_amax"] = 0.5 + 0.5 * torch.rand(
                    experts, generator=generator
                )
                tensors["model.llm." + base + "_outer_scale_denominator"] = (
                    torch.tensor([6.0 * k**0.5])
                )
            else:
                tensors["model.llm." + base] = (
                    torch.randn((experts, n, k), generator=generator) * 32
                ).to(torch.float8_e4m3fn)
                exponent = 127 - 5 - round(math.log2(k**0.5))
                tensors["model.llm." + base + "_scale"] = torch.full(
                    (experts, n, k // 32), exponent, dtype=torch.uint8
                )
        if loop.core_start <= i < loop.core_end:
            for stream in ("attn", "k", "v", "mlp"):
                controller = prefix + f"controllers.{stream}."
                normal(
                    controller + "down_weight", (loop.gate_rank, 2 * h), (2 * h) ** -0.5
                )
                normal(controller + "gain_weight", (h, loop.gate_rank), 0.5)
                normal(controller + "forget_weight", (h, loop.gate_rank), 0.5)
    root.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    (root / "config.json").write_text(json.dumps(_hf_config(config)))
    nvfp4: dict[str, Any] = {
        "compute_dtype": "nvfp4",
        "required_activation_quantization": {
            "amax_type": "per_token",
            "block_size": 16,
            "format": "nvfp4_e2m1",
            "rounding": "nearest",
            "scale_dtype": "float8_e4m3fn",
            "scale_selection_rule": "fouroversix_mae",
        },
        "tensors": [],
    }
    mxfp8: dict[str, Any] = {
        "compute_dtype": "mxfp8",
        "required_activation_quantization": {
            "block_size": 32,
            "format": "mxfp8_e4m3",
            "rounding": "nearest",
            "scale_dtype": "ue8m0",
        },
        "tensors": [],
    }
    for i in range(config.num_hidden_layers):
        group = nvfp4 if _layer_mode(i) == "nvfp4" else mxfp8
        group["tensors"] += [
            f"model.llm.layers.{i}.mlp.experts.{matrix}_weight"
            for matrix in ("w13", "w2")
        ]
    (root / "hf_quant_config.json").write_text(
        json.dumps({"model": {"parameter_groups": [nvfp4, mxfp8]}})
    )


# --- PyTorch reference ---------------------------------------------------------


def _mxfp8_activations(x):
    grouped = x.float().reshape(x.shape[0], -1, 32)
    raw = grouped.abs().amax(-1) * (1.0 / 448.0)
    exponent = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0**-127, 0, exponent.clamp(max=254))
    scale = (exponent << 23).view(torch.float32)
    inverse = torch.where(exponent == 0, torch.zeros_like(scale), 1.0 / scale)
    values = (
        (grouped * inverse[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    )
    return (values * scale[..., None]).flatten(1)


def _activations(x, mode):
    if mode == "nvfp4":
        packed, scales, amax = nvfp4_reference(x)
        return dequant(packed, scales, torch.ones_like(amax), 1.0), amax
    return _mxfp8_activations(x), None


def _weight(module, name, mode, expert):
    weight = getattr(module, name + "_weight")[expert]
    scales = getattr(module, name + "_weight_scale")[expert]
    if mode == "nvfp4":
        return dequant(weight, scales, torch.tensor(1.0), 1.0)
    return dequant(weight, scales)


def reference_experts(module, hidden, ids, weights):
    mode = module.precision["w13"]
    result = torch.zeros_like(hidden, dtype=torch.float32)

    def matmul(x, name, expert):
        base, amax = _activations(x, mode)
        output = base @ _weight(module, name, mode, expert).t()
        if mode == "nvfp4":
            denominator = getattr(module, name + "_weight_outer_scale_denominator")
            weight_amax = getattr(module, name + "_weight_amax")[expert]
            output = (
                output
                * ((amax * weight_amax) * (1.0 / (1536.0 * denominator)))[:, None]
            )
        return output.to(torch.bfloat16)

    for token in range(hidden.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate, up = (
                matmul(hidden[token : token + 1], "w13", expert)
                .float()
                .view(-1, 2)
                .unbind(-1)
            )
            activated = (
                ((4 * (gate / 4).tanh()) * gate.sigmoid()) * (25 * (up / 25).tanh())
            ) * weights[token, slot]
            down = matmul(activated.to(torch.bfloat16)[None], "w2", expert)
            result[token] += down[0].float()
    return result


class ReferenceCache:
    """One contiguous page per (row, visit): the reference reads keys [0, length)."""

    def __init__(self, config, batch, capacity):
        self.page_tokens, self.slab_units = capacity, 1
        self.counts = {
            0: config.execution_global_layer_count,
            1: config.execution_local_layer_count,
        }
        self.groups = [
            SimpleNamespace(first_blocks=torch.zeros(batch, dtype=torch.int64))
            for _ in range(2)
        ]
        shape = (
            batch * config.execution_layer_count,
            capacity,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.keys = torch.zeros(shape, dtype=torch.bfloat16)
        self.values = torch.zeros_like(self.keys)
        self.batch = batch

    def pages(self, group, layer, count):
        base = 0 if group == 0 else self.batch * self.counts[0]
        return (base + torch.arange(self.batch) * count + layer)[:, None]

    def store_kv(self, keys, values, positions, *, group_index, layer, layer_count):
        pages = self.pages(group_index, layer, layer_count).expand_as(positions)
        self.keys[pages, positions], self.values[pages, positions] = keys, values


def reference_attention(
    q,
    *,
    cache,
    group_index,
    layer,
    layer_count,
    positions,
    sequence_lengths,
    softmax_scale,
    window_left,
    relative_logits,
    **_,
):
    outputs, statistics = [], []
    for row in range(q.shape[0]):
        page = cache.pages(group_index, layer, layer_count)[row, 0]
        length = int(sequence_lengths[row])
        k = cache.keys[page, :length].float()
        v = cache.values[page, :length].float()
        groups = q.shape[2] // k.shape[1]
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        logits = torch.einsum("thd,shd->ths", q[row].float(), k) * softmax_scale
        distance = positions[row, :, None] - torch.arange(length)[None]
        rel = relative_logits.view(*q.shape[:3], -1)[row]
        bias = rel.gather(
            2, distance.clamp(0, rel.shape[-1] - 1)[:, None].expand(-1, q.shape[2], -1)
        )
        logits += bias * ((distance >= 0) & (distance < rel.shape[-1]))[:, None]
        mask = distance >= 0
        if window_left is not None:
            mask &= distance <= window_left
        logits.masked_fill_(~mask[:, None], float("-inf"))
        outputs.append(torch.einsum("ths,shd->thd", logits.softmax(-1), v))
        statistics.append(logits.logsumexp(-1))
    return torch.stack(outputs), torch.cat(statistics, 0).t()


def reference_target(root: Path, config: Inkling2Config, monkeypatch):
    from resin_models_cuda.inkling2 import attention
    from resin_models_cuda.inkling2.model import Inkling2Target
    from resin_models_cuda.inkling2.weights import load_target_weights

    monkeypatch.setattr(attention, "paged_attention", reference_attention)
    precision = {
        f"model.llm.layers.{i}.mlp.experts.{matrix}_weight": _layer_mode(i)
        for i in range(config.num_hidden_layers)
        for matrix in ("w13", "w2")
    }
    # One feature tap (for DSpark) keeps the reference's return shape; logits ignore it.
    target = Inkling2Target(config, (1,), precision=precision)
    load_target_weights(target, root)
    target.prepare()
    for layer in target.layers:
        experts = layer.mlp.experts  # ty: ignore[unresolved-attribute]
        monkeypatch.setattr(
            experts,
            "_local",
            lambda h, i, w, e=experts, max_rows_per_expert=None: reference_experts(
                e, h, i, w
            ),
        )
    return target


class DenseStore:
    """Unpaged MLX attention storage for one batch, indexed by absolute position."""

    def __init__(self, config: Inkling2Config, batch: int, capacity: int):
        shape = (batch, capacity, config.num_key_value_heads, config.head_dim)
        self.keys = {
            (local, slot): mx.zeros(shape, dtype=mx.bfloat16)
            for local, count in (
                (True, config.execution_local_layer_count),
                (False, config.execution_global_layer_count),
            )
            for slot in range(count)
        }
        self.values = {key: mx.zeros(shape, dtype=mx.bfloat16) for key in self.keys}
        self.positions: mx.array | None = None

    def attend(self, local, slot, q, k, v, bias, scale, window):
        assert self.positions is not None
        positions = self.positions
        batch, tokens = positions.shape
        rows = mx.arange(batch)[:, None]
        keys = self.keys[local, slot]
        values = self.values[local, slot]
        keys[rows, positions] = k
        values[rows, positions] = v
        length = int(np.array(positions).max()) + 1
        keys, values = (
            keys[:, :length].astype(mx.float32),
            values[:, :length].astype(mx.float32),
        )
        groups = q.shape[2] // keys.shape[2]
        keys = mx.repeat(keys, groups, axis=2)
        values = mx.repeat(values, groups, axis=2)
        logits = mx.einsum("bthd,bshd->bths", q.astype(mx.float32), keys) * scale
        distance = positions[:, :, None] - mx.arange(length)[None, None]
        extent = bias.shape[-1]
        index = mx.clip(distance, 0, extent - 1)
        gathered = mx.take_along_axis(
            bias.astype(mx.float32),
            mx.broadcast_to(index[:, :, None, :], (*bias.shape[:3], length)),
            axis=-1,
        )
        inside = (distance >= 0) & (distance < extent)
        logits = logits + mx.where(inside[:, :, None, :], gathered, 0.0)
        mask = distance >= 0
        if local:
            mask = mask & (distance <= window - 1)
        logits = mx.where(mask[:, :, None, :], logits, -mx.inf)
        lse = mx.logsumexp(logits, axis=-1)
        probabilities = mx.exp(logits - lse[..., None])
        output = mx.einsum("bths,bshd->bthd", probabilities, values)
        return output, lse


def _relative(actual: mx.array, expected) -> float:
    a = np.array(actual.astype(mx.float32)).reshape(-1)
    e = expected.float().numpy().reshape(-1)
    return float(np.abs(a - e).max() / max(np.abs(e).max(), 1e-6))


def _record(target):
    """Record every visit's attention and MLP inputs/outputs, in visit order."""
    records = {"sources": [], "attention": [], "mlp_input": [], "mlp": []}
    for layer in target.layers:
        layer.attn.register_forward_pre_hook(
            lambda _m, args: records["sources"].append(
                tuple(x.clone() for x in args[0])
            )
        )
        layer.attn.register_forward_hook(
            lambda _m, _args, output: records["attention"].append(output[0].clone())
        )
        layer.mlp.register_forward_pre_hook(
            lambda _m, args: records["mlp_input"].append(args[0].clone())
        )
        layer.mlp.register_forward_hook(
            lambda _m, _args, output: records["mlp"].append(output.clone())
        )
    return records


def _to_mlx(value) -> mx.array:
    return mx.array(value.float().numpy()).astype(mx.bfloat16)


# At 40 tokens (640 routes) the threshold is lowered so the segmented prefill
# experts run; the shorter prompts keep the gathered quantized GEMM.
@pytest.mark.parametrize("prompt_length", [5, 20, 40])
def test_tiny_model_matches_cuda_reference(tmp_path, monkeypatch, prompt_length):
    """Teacher-forced: each visit's mixing, attention and experts against the reference.

    Random tiny models are chaotic under FP4 activation quantization: a one-ULP
    change at the first visit moves the reference's own logits by more than the
    port does. Each MLX visit therefore (1) rebuilds the reference's attention
    and MLP inputs from the reference's earlier outputs, (2) recomputes attention
    and experts from the reference's exact inputs, and (3) continues from the
    reference's outputs, so differences cannot compound.
    """
    from resin_models_mlx.inkling2 import model as mlx_model

    if prompt_length == 40:
        monkeypatch.setattr(mlx_model, "SEGMENTED_ROUTES", 512)

    config = tiny_config()
    write_checkpoint(tmp_path, config)
    target = reference_target(tmp_path, config, monkeypatch)
    records = _record(target)
    loaded_config, weights = load_model(tmp_path)
    assert loaded_config == config
    model = Inkling2Model(config, weights)
    errors = {"sources": 0.0, "attention": 0.0, "mlp_input": 0.0, "mlp": 0.0}
    cursor = {"attention": 0, "mlp": 0}
    attention = Inkling2Model._attention
    sparse = mlx_model.sparse_mlp

    def forced_attention(self, layer, sources, state, tau, visit, slot, store):
        index = cursor["attention"]
        cursor["attention"] += 1
        reference = records["sources"][index]
        for actual, expected in zip(sources, reference, strict=True):
            errors["sources"] = max(errors["sources"], _relative(actual, expected))
        forced = tuple(_to_mlx(x) for x in reference)
        output, k_history, v_history = attention(
            self, layer, forced, state, tau, visit, slot, store
        )
        expected = records["attention"][index]
        errors["attention"] = max(errors["attention"], _relative(output, expected))
        return _to_mlx(expected), k_history, v_history

    def forced_mlp(x, layer, config):
        index = cursor["mlp"]
        cursor["mlp"] += 1
        reference = records["mlp_input"][index]
        errors["mlp_input"] = max(errors["mlp_input"], _relative(x, reference))
        output = sparse(_to_mlx(reference).reshape(x.shape), layer, config)
        expected = records["mlp"][index]
        errors["mlp"] = max(errors["mlp"], _relative(output, expected))
        return _to_mlx(expected).reshape(x.shape)

    monkeypatch.setattr(Inkling2Model, "_attention", forced_attention)
    monkeypatch.setattr(mlx_model, "sparse_mlp", forced_mlp)

    batch, capacity, decode = 2, 64, 6
    generator = torch.Generator().manual_seed(3)
    prompt = torch.randint(
        0, config.unpadded_vocab_size, (batch, prompt_length), generator=generator
    )
    cache = ReferenceCache(config, batch, capacity)
    store = DenseStore(config, batch, capacity)
    torch_state = torch.zeros(
        batch,
        config.execution_layer_count,
        3,
        config.state_width(),
        dtype=torch.bfloat16,
    )
    mlx_state = model.initial_state(batch)
    tokens, start, worst_logits, worst_state = prompt, 0, 0.0, 0.0
    for step in range(decode + 1):
        for values in records.values():
            values.clear()
        cursor.update(attention=0, mlp=0)
        count = tokens.shape[1]
        positions = torch.arange(start, start + count)[None].expand(batch, -1)
        with torch.no_grad():
            expected, _, torch_state, _ = target(
                tokens,
                positions,
                cache,
                torch_state,
                positions[:, -1] + 1,
                start + count,
                full_logits=True,
                prefill=step == 0,
            )
        store.positions = mx.array(positions.numpy())
        actual, mlx_state = model(
            mx.array(tokens.numpy()), store.positions, mlx_state, store, last_only=False
        )
        assert cursor == {
            "attention": config.execution_layer_count,
            "mlp": config.execution_layer_count,
        }
        worst_logits = max(worst_logits, _relative(actual, expected))
        worst_state = max(worst_state, _relative(mlx_state, torch_state))
        assert (np.array(actual).argmax(-1) == expected.numpy().argmax(-1)).all(), step
        start += count
        tokens = expected[:, -1:].argmax(-1)
    # One BF16 ULP is 2^-8 relative; GEMM accumulation order moves a few values by one.
    assert errors["sources"] < 1e-2, errors
    assert errors["mlp_input"] < 1e-2, errors
    assert errors["attention"] < 2e-2, errors
    assert errors["mlp"] < 2e-2, errors
    assert worst_logits < 1e-2, worst_logits
    assert worst_state < 1e-2, worst_state
