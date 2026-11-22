"""Production-width scatter through a complete 4K-prefill payload."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal cache")

from resin_models_mlx._native import CacheStorage  # noqa: E402


def test_real_prefill_49152_rows_preserves_every_row_and_column():
    count, width = 49_152, 512  # 4096 tokens × six layers × key/value.
    rows = np.arange(count, dtype=np.uint16)
    payload = np.bitwise_xor(rows[:, None], np.arange(width, dtype=np.uint16))
    indices = np.arange(count, dtype=np.int32)[::-1].copy()
    row_bytes = width * payload.itemsize
    storage = CacheStorage((count + 1, row_bytes))
    epoch = storage.scatter_rows(
        storage.initial_epoch(),
        mx.array(indices),
        mx.array(payload).view(mx.bfloat16),
        row_bytes=row_bytes,
    )
    actual = np.asarray(storage.read_epoch(epoch).view(mx.uint16))
    np.testing.assert_array_equal(actual[:count], payload[::-1])
    np.testing.assert_array_equal(actual[count], np.zeros(width, np.uint16))
    np.testing.assert_array_equal(np.asarray(epoch), np.full(8, 1, np.uint32))
