"""Production-width scatter through a complete 4K-prefill payload."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal cache")

from resin_models_mlx.native import CacheStorage  # noqa: E402


def test_real_prefill_49152_rows_preserves_every_row_and_column():
    count, width = 49_152, 512  # 4096 tokens × six layers × key/value.
    rows = np.arange(count, dtype=np.uint16)
    payload = np.bitwise_xor(rows[:, None], np.arange(width, dtype=np.uint16))
    indices = np.arange(count, dtype=np.int32)[::-1].copy()
    storage = CacheStorage((count + 1, width), mx.bfloat16)
    epoch = storage.scatter_rows(
        storage.initial_epoch(), mx.array(indices), mx.array(payload).view(mx.bfloat16)
    )
    actual = np.asarray(storage.snapshot(epoch).view(mx.uint16))
    np.testing.assert_array_equal(actual[:count], payload[::-1])
    np.testing.assert_array_equal(actual[count], np.zeros(width, np.uint16))
    np.testing.assert_array_equal(np.asarray(epoch), np.full(8, 1, np.uint32))
    assert storage.buffer_identity != 0 and storage.allocation_count == 1
