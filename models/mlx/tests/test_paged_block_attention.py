"""Multi-token paged attention against independent gathered SDPA."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from resin_models_mlx.paged_attention import (  # noqa: E402
    paged_block_attention,
    paged_decode_attention,
)

PAGE = 128


def _bf16(array: np.ndarray):
    return mx.array(array.astype(np.float32)).astype(mx.bfloat16)


def _arena(rng, lengths, kv_heads, head_dim, tokens, layers=2, layer=1):
    rows = len(lengths)
    width = max(1, -(-(max(lengths) + tokens) // PAGE))
    # Unowned units, row tails and the block's own destinations stay poisoned:
    # the block's keys are not written to the arena before attention.
    units = np.full((rows * width * layers + 5, 2, PAGE, kv_heads * head_dim), np.nan)
    pages = (rng.permutation(rows * width) * layers).reshape(rows, width)
    keys = rng.standard_normal((rows, width * PAGE, kv_heads, head_dim))
    values = rng.standard_normal((rows, width * PAGE, kv_heads, head_dim))
    for row, length in enumerate(lengths):
        for position in range(length):
            unit = pages[row, position // PAGE] + layer
            units[unit, 0, position % PAGE] = keys[row, position].reshape(-1)
            units[unit, 1, position % PAGE] = values[row, position].reshape(-1)
    return units, pages.astype(np.int32), keys, values


def _reference(queries, keys, values, lengths, skip, scale):
    """Token ``j`` of row ``r`` attends to history ``skip..L`` and block ``0..j``."""

    rows, tokens = queries.shape[:2]
    shared = queries.shape[2] // keys.shape[2]
    output = []
    for row, length in enumerate(lengths):
        row_output = []
        for token in range(tokens):
            k = np.concatenate(
                [keys[row, skip:length], keys[row, length : length + token + 1]]
            )
            v = np.concatenate(
                [values[row, skip:length], values[row, length : length + token + 1]]
            )
            k = mx.repeat(_bf16(k).astype(mx.float32), shared, axis=1)
            v = mx.repeat(_bf16(v).astype(mx.float32), shared, axis=1)
            q = _bf16(queries[row, token]).astype(mx.float32)
            weights = mx.softmax(
                (q[:, None] * k.transpose(1, 0, 2)).sum(-1) * scale, axis=-1
            )
            row_output.append((weights[..., None] * v.transpose(1, 0, 2)).sum(1))
        output.append(mx.stack(row_output))
    return mx.stack(output)


@pytest.mark.parametrize(
    "heads,kv_heads,head_dim,lengths,tokens,skip",
    [
        (8, 2, 256, [126, 127, 128, 0], 3, 0),  # Native Qwen geometry.
        (8, 2, 256, [600, 1, 255], 4, 1),
        (8, 2, 256, [5], 1, 1),
        (4, 2, 64, [127, 128, 129], 2, 0),
        (6, 1, 64, [1023, 1], 4, 1),
        (24, 4, 256, [300, 2], 3, 0),  # Qwen3.8-27B heads, generic kernel.
    ],
)
def test_block_attention_matches_gathered_attention(
    heads, kv_heads, head_dim, lengths, tokens, skip
):
    rng = np.random.default_rng(heads + tokens + skip)
    rows = len(lengths)
    units, pages, keys, values = _arena(rng, lengths, kv_heads, head_dim, tokens)
    block_keys = np.stack(
        [keys[row, length : length + tokens] for row, length in enumerate(lengths)]
    )
    block_values = np.stack(
        [values[row, length : length + tokens] for row, length in enumerate(lengths)]
    )
    queries = rng.standard_normal((rows, tokens, heads, head_dim))
    scale = head_dim**-0.5
    actual = paged_block_attention(
        _bf16(queries),
        _bf16(block_keys),
        _bf16(block_values),
        _bf16(units),
        mx.array(pages),
        mx.array(np.array(lengths, np.int32)),
        1,
        scale,
        skip,
    )
    # The block's keys are exactly the history positions that follow it.
    expected = _reference(queries, keys, values, lengths, skip, scale)
    assert actual.shape == (rows, tokens, heads, head_dim)
    assert not mx.isnan(actual).any().item()
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(expected), atol=2e-2, rtol=2e-2
    )


def test_one_token_block_is_the_decode_kernel():
    rng = np.random.default_rng(3)
    lengths = [129, 4]
    units, pages, keys, values = _arena(rng, lengths, 2, 256, 1)
    queries = _bf16(rng.standard_normal((2, 1, 8, 256)))
    current_keys = _bf16(rng.standard_normal((2, 1, 2, 256)))
    current_values = _bf16(rng.standard_normal((2, 1, 2, 256)))
    arguments = (
        _bf16(units),
        mx.array(pages),
        mx.array(np.array(lengths, np.int32)),
        1,
        256**-0.5,
    )
    block = paged_block_attention(queries, current_keys, current_values, *arguments)
    decode = paged_decode_attention(
        queries[:, 0], current_keys[:, 0], current_values[:, 0], *arguments
    )
    assert mx.array_equal(block[:, 0], decode).item()


def test_prior_keys_precede_the_block():
    rng = np.random.default_rng(11)
    lengths, prior, tokens = [130, 7], 2, 2
    units, pages, keys, values = _arena(rng, lengths, 2, 256, prior + tokens)
    span = [
        (
            keys[row, length : length + prior + tokens],
            values[row, length : length + prior + tokens],
        )
        for row, length in enumerate(lengths)
    ]
    block_keys = np.stack([k for k, _ in span])
    block_values = np.stack([v for _, v in span])
    queries = rng.standard_normal((2, prior + tokens, 8, 256))
    arguments = (
        _bf16(units),
        mx.array(pages),
        mx.array(np.array(lengths, np.int32)),
        1,
        256**-0.5,
        1,
    )
    whole = paged_block_attention(
        _bf16(queries), _bf16(block_keys), _bf16(block_values), *arguments
    )
    tail = paged_block_attention(
        _bf16(queries[:, prior:]),
        _bf16(block_keys[:, prior:]),
        _bf16(block_values[:, prior:]),
        *arguments,
        prior=(_bf16(block_keys[:, :prior]), _bf16(block_values[:, :prior])),
    )
    np.testing.assert_allclose(
        np.array(tail.astype(mx.float32)),
        np.array(whole[:, prior:].astype(mx.float32)),
        atol=1e-2,
        rtol=1e-2,
    )
