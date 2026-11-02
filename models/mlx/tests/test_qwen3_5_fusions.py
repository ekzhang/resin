"""Whole-model parity with the original immediate-residual native layer loop."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.gated_delta import (  # noqa: E402
    gated_delta_update as native_delta_update,
)
from mlx_lm.models.gated_delta import normalize_qk as native_normalize_qk  # noqa: E402
from test_qwen3_5_mlx import _text_config  # noqa: E402

import resin_models_mlx.qwen3_5.model as model_module  # noqa: E402
from resin.models.qwen3_5.config import Qwen35Config  # noqa: E402
from resin_models_mlx.qwen3_5.model import (  # noqa: E402
    AttentionHistory,
    LinearState,
    Qwen35Model,
    StepOutput,
)


def _configuration(dense):
    values = _text_config(dense=dense)
    if not dense:
        # All expert projection widths must support the packed group64 case.
        values.update(moe_intermediate_size=64, shared_expert_intermediate_size=64)
    return Qwen35Config.from_hugging_face(values)


def _model(dense, precision):
    mx.random.seed(3419)
    model = Qwen35Model(_configuration(dense))
    updates = []
    for name, value in cast(list[tuple[str, Any]], tree_flatten(model.parameters())):
        if name.endswith("linear_attn.norm.weight"):
            updates.append((name, 1.0 + 0.1 * mx.random.normal(value.shape)))
        elif name.endswith("norm.weight"):
            updates.append((name, 0.1 * mx.random.normal(value.shape)))
        elif name.endswith("A_log"):
            updates.append((name, mx.random.uniform(-8.0, 1.0, value.shape)))
        elif name.endswith("dt_bias"):
            updates.append((name, 0.1 * mx.random.normal(value.shape)))
    model.load_weights(updates, strict=False)
    decay = [
        (name, value)
        for name, value in cast(list[tuple[str, Any]], tree_flatten(model.parameters()))
        if name.endswith(("A_log", "dt_bias"))
    ]
    model.set_dtype(mx.float32 if precision == "fp32" else mx.bfloat16)
    model.load_weights(decay, strict=False)
    if precision == "int8":
        nn.quantize(model, bits=8, group_size=64, mode="affine")
    mx.eval(model.parameters())
    model.prepare_for_inference()
    return model


def _native_mixed(hidden, scale, eps):
    return mx.fast.rms_norm(hidden, scale, eps).astype(hidden.dtype)


def _native_conv(current, previous, weight):
    window = mx.concatenate([previous.astype(current.dtype), current], axis=1)
    kernel = weight.shape[1]
    carry = window[:, -(kernel - 1) :] if kernel > 1 else window[:, :0]
    return nn.silu(mx.conv1d(window, weight, groups=current.shape[-1])), carry


def _native_conv_qkv(
    current, previous, weight, *, key_heads, key_dim, value_heads, value_dim
):
    activation, carry = _native_conv(current, previous, weight)
    rows, tokens, _ = activation.shape
    width = key_heads * key_dim
    return (
        activation[..., :width].reshape(rows, tokens, key_heads, key_dim),
        activation[..., width : 2 * width].reshape(rows, tokens, key_heads, key_dim),
        activation[..., 2 * width :].reshape(rows, tokens, value_heads, value_dim),
        carry,
    )


def _native_delta(q, k, v, a, b, A_log, dt_bias, state):
    return native_delta_update(q, k, v, a.astype(mx.float32), b, A_log, dt_bias, state)


def _native_offset(norm, hidden):
    # Derive the reference directly from serialized weights, without consulting
    # the production effective-scale cache or fused normalization methods.
    scale = norm.weight.astype(mx.float32)
    if norm.offset:
        scale = 1.0 + scale
    return _native_mixed(hidden, scale, norm.eps)


def _old_layer_loop(model, token_ids, offsets, state, history):
    hidden = model.embed_tokens(token_ids)
    keys, values, recurrent, convolution = [], [], [], []
    linear_index = full_index = 0
    for layer in model.layers:
        normed = _native_offset(layer.input_layernorm, hidden)
        if layer.is_linear:
            mixed, next_recurrent, next_convolution = layer.linear_attn(
                normed, state.recurrent[linear_index], state.convolution[linear_index]
            )
            recurrent.append(next_recurrent)
            convolution.append(next_convolution)
            linear_index += 1
        else:
            mixed, key, value = layer.self_attn(
                normed, offsets, history[full_index], full_index
            )
            keys.append(key)
            values.append(value)
            full_index += 1
        hidden = hidden + mixed
        hidden = hidden + layer.mlp(
            _native_offset(layer.post_attention_layernorm, hidden)
        )
    return StepOutput(
        _native_offset(model.norm, hidden),
        tuple(keys),
        tuple(values),
        LinearState(tuple(recurrent), tuple(convolution)),
    )


def _arrays(output):
    return (
        output.hidden,
        *output.keys,
        *output.values,
        *output.state.recurrent,
        *output.state.convolution,
    )


def _equal_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    bits = mx.uint32 if expected.dtype == mx.float32 else mx.uint16
    np.testing.assert_array_equal(
        np.asarray(actual.view(bits)), np.asarray(expected.view(bits))
    )


def _append_history(history, output):
    result = []
    for previous, key, value in zip(history, output.keys, output.values, strict=True):
        if previous is not None:
            key = mx.concatenate([previous.keys, key], axis=1)
            value = mx.concatenate([previous.values, value], axis=1)
        result.append(AttentionHistory(key, value))
    return tuple(result)


@pytest.mark.parametrize(
    "dense,precision,batch",
    [
        (False, "bf16", 4),
        (True, "bf16", 1),
        (False, "fp32", 2),
        (True, "fp32", 4),
        (False, "int8", 1),
        (True, "int8", 2),
    ],
)
def test_fused_forward_matches_original_native_layer_loop(
    monkeypatch, dense, precision, batch
):
    model = _model(dense, precision)
    actual_state, expected_state = (
        model.initial_state(batch),
        model.initial_state(batch),
    )
    actual_history = expected_history = (None,) * model.config.full_layer_count
    rng = np.random.default_rng(9142 + batch)
    context = 0
    # Prefill, ordinary one-token decode (the Conv4 fast path), then a chunk
    # using both carried recurrent state and cached full-attention K/V.
    for width in (13, 1, 5):
        tokens = mx.array(rng.integers(2, 256, size=(batch, width), dtype=np.int32))
        offsets = mx.full((batch,), context, dtype=mx.int32)
        with monkeypatch.context() as native:
            native.setattr(model_module, "conv4_silu_qkv", _native_conv_qkv)
            native.setattr(model_module, "gated_delta_update", _native_delta)
            native.setattr(model_module, "normalize_qk", native_normalize_qk)
            expected = _old_layer_loop(
                model, tokens, offsets, expected_state, expected_history
            )
            expected_logits = model.logits(expected.hidden)
            # No reference graph may call a restored fused helper afterward.
            mx.eval(*_arrays(expected), expected_logits)
        actual = model(tokens, offsets, actual_state, actual_history)
        actual_logits = model.logits(actual.hidden)
        mx.eval(*_arrays(actual), actual_logits)
        for value, reference in zip(_arrays(actual), _arrays(expected), strict=True):
            _equal_bits(value, reference)
        _equal_bits(actual_logits, expected_logits)
        actual_state, expected_state = actual.state, expected.state
        actual_history = _append_history(actual_history, actual)
        expected_history = _append_history(expected_history, expected)
        context += width
