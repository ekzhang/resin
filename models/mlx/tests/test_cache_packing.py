"""Cache packing preserves order and bits, including every BF16 NaN payload."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin_models_mlx.cache_packing import pack_key_values  # noqa: E402


@pytest.mark.parametrize("layers,dtype", [(6, "bfloat16"), (8, "float32")])
def test_packed_cache_matches_original_layout(layers, dtype):
    # BF16 exercises the fused path; eight FP32 layers exercise native fallback.
    rows, width = 128, 512
    bits = np.arange(65536, dtype=np.uint16).reshape(rows, width)
    keys, values, expected = [], [], []
    for layer in range(layers):
        key = np.roll(bits, layer * 17).copy()
        value = np.roll(bits, 65535 - layer * 31).copy()
        if dtype == "float32":
            key, value = key.astype(np.float32), value.astype(np.float32)
        # The offset and unused columns keep the actual inputs noncontiguous.
        packed = np.zeros((rows, 2 * width + 32), dtype=key.dtype)
        packed[:, 16 : 16 + width], packed[:, 16 + width : 16 + 2 * width] = key, value
        backing = mx.array(packed)
        if dtype == "bfloat16":
            backing = backing.view(mx.bfloat16)
        keys.append(backing[:, 16 : 16 + width].reshape(rows, 2, 256))
        values.append(backing[:, 16 + width : 16 + 2 * width].reshape(rows, 2, 256))
        expected.append(np.stack([key, value], axis=1))
    result = pack_key_values(keys, values, rows, width)
    expected = np.stack(expected, axis=1)
    assert result.shape == expected.shape
    if dtype == "bfloat16":
        assert result.dtype == mx.bfloat16
        actual = np.asarray(result.view(mx.uint16))
    else:
        assert result.dtype == mx.float32
        actual = np.asarray(result)
    np.testing.assert_array_equal(actual, expected)
