"""Compare cache packing with the original stack order, including raw BF16 bits."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin_models_mlx.cache_packing import pack_key_values  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


def _original(keys, values, count):
    return mx.stack(
        [
            mx.stack([key.reshape(count, -1), value.reshape(count, -1)], axis=1)
            for key, value in zip(keys, values, strict=True)
        ],
        axis=1,
    )


def _exact(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    unsigned = mx.uint32 if actual.dtype == mx.float32 else mx.uint16
    np.testing.assert_array_equal(
        np.asarray(actual.view(unsigned)), np.asarray(expected.view(unsigned))
    )


def _fixture(rows, tokens=1, layers=6, *, rank=3, strided=True):
    keys, values = [], []
    backing_width = 5120 if strided else 1024
    for layer in range(layers):
        bits = (
            np.arange(rows * tokens * backing_width, dtype=np.uint32) * 37 + layer * 171
        ).astype(np.uint16)
        packed = mx.array(bits.reshape(rows, tokens, backing_width)).view(mx.bfloat16)
        start = 4096 if strided else 0
        key = packed[..., start : start + 512].reshape(rows, tokens, 2, 256)
        value = packed[..., start + 512 : start + 1024].reshape(rows, tokens, 2, 256)
        if not strided:
            key, value = mx.contiguous(key), mx.contiguous(value)
        if rank == 3:
            key, value = (
                key.reshape(rows * tokens, 2, 256),
                value.reshape(rows * tokens, 2, 256),
            )
        keys.append(key)
        values.append(value)
    return keys, values


@pytest.mark.parametrize("rows,strided", [(1, False), (1, True), (4, True)])
def test_actual_rank3_six_layer_cache_order(rows, strided):
    keys, values = _fixture(rows, strided=strided)
    _exact(pack_key_values(keys, values, rows, 512), _original(keys, values, rows))


@pytest.mark.parametrize("rows,tokens", [(4, 7), (2, 129)])
def test_rank4_prefill_strides_and_flat_token_order(rows, tokens):
    keys, values = _fixture(rows, tokens, rank=4)
    count = rows * tokens
    _exact(pack_key_values(keys, values, count, 512), _original(keys, values, count))


def test_all_raw_bf16_patterns_are_copied_without_conversion():
    # Unlike arithmetic tests, every NaN payload and both signed zeros matter.
    bits = np.arange(65536, dtype=np.uint16).reshape(128, 2, 256)
    keys = [mx.array(np.roll(bits, layer * 17)).view(mx.bfloat16) for layer in range(6)]
    values = [
        mx.array(np.roll(bits, 65535 - layer * 31)).view(mx.bfloat16)
        for layer in range(6)
    ]
    actual = pack_key_values(keys, values, 128, 512)
    expected_bits = np.stack(
        [
            np.stack(
                [
                    np.roll(bits, layer * 17).reshape(128, 512),
                    np.roll(bits, 65535 - layer * 31).reshape(128, 512),
                ],
                axis=1,
            )
            for layer in range(6)
        ],
        axis=1,
    )
    np.testing.assert_array_equal(np.asarray(actual.view(mx.uint16)), expected_bits)


@pytest.mark.parametrize(
    "layers,dtype,layout",
    [
        (1, "bfloat16", "rank3"),
        (8, "bfloat16", "rank3"),
        (2, "float32", "rank3"),
        (2, "bfloat16", "mixed_dtype"),
        (2, "bfloat16", "rank2"),
        (2, "bfloat16", "mixed_shape"),
        (2, "bfloat16", "wrong_width"),
    ],
)
def test_dtype_layer_and_legacy_shape_contracts(layers, dtype, layout):
    keys, values = _fixture(3, layers=layers)
    keys, values = (
        [v.astype(getattr(mx, dtype)) for v in arrays] for arrays in (keys, values)
    )
    if layout == "mixed_dtype":
        values[1] = values[1].astype(mx.float32)
    elif layout == "rank2":
        keys, values = (
            [v.reshape(3, 512) for v in arrays] for arrays in (keys, values)
        )
    elif layout == "mixed_shape":
        values[0] = values[0].reshape(3, 512)
    width = 513 if layout == "wrong_width" else 512
    _exact(pack_key_values(keys, values, 3, width), _original(keys, values, 3))


def test_invalid_count_and_unpaired_layers_raise_as_original():
    keys, values = _fixture(3, layers=2)
    for candidate_keys, candidate_values, count in (
        (keys, values, 5),
        (keys, values[:1], 3),
        ([], [], 3),
    ):
        with pytest.raises(ValueError) as original_error:
            _original(candidate_keys, candidate_values, count)
        with pytest.raises(type(original_error.value)):
            pack_key_values(candidate_keys, candidate_values, count, 512)
