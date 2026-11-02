"""Fresh state matches the null slab without changing restore precedence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from test_qwen3_5_mlx import _text_config  # noqa: E402

from resin.backend.plans import PagedCachePlan  # noqa: E402
from resin.models.qwen3_5.config import Qwen35Config  # noqa: E402
from resin_models_mlx.qwen3_5.cache import (  # noqa: E402
    STATE_GROUP,
    CacheArena,
    CacheGeometry,
)
from resin_models_mlx.qwen3_5.model import LinearState, Qwen35Model  # noqa: E402
from resin_models_mlx.qwen3_5.state import LiveStates  # noqa: E402


@pytest.fixture
def state_resources() -> tuple[Qwen35Model, CacheArena]:
    config = Qwen35Config.from_hugging_face(_text_config(dense=True))
    model = Qwen35Model(config)
    geometry = CacheGeometry.from_config(config)
    return model, CacheArena(geometry.layout(1), geometry)


def _plan(
    contexts: list[int], restores: tuple[tuple[int, int, int], ...], *, sequences=None
) -> PagedCachePlan:
    return cast(
        PagedCachePlan,
        SimpleNamespace(
            batch=SimpleNamespace(
                context_lengths=np.array(contexts, dtype=np.int64),
                sequence_ids=np.asarray(sequences)
                if sequences is not None
                else np.arange(len(contexts), dtype=np.int64) + 100,
            ),
            groups={
                STATE_GROUP: SimpleNamespace(
                    state=SimpleNamespace(restore_copies=restores, checkpoint_slots=())
                )
            },
        ),
    )


def _assert_state_bits_equal(left: LinearState, right: LinearState) -> None:
    for actual, expected in zip(
        (*left.recurrent, *left.convolution),
        (*right.recurrent, *right.convolution),
        strict=True,
    ):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        integer = mx.uint16 if actual.dtype == mx.bfloat16 else mx.uint32
        np.testing.assert_array_equal(
            np.array(actual.view(integer)), np.array(expected.view(integer))
        )


def _marked(model: Qwen35Model, markers) -> LinearState:
    markers = np.atleast_1d(markers).astype(np.float32)
    initial = model.initial_state(len(markers))

    def fill(value):
        row = mx.array(markers, value.dtype).reshape(-1, *[1] * (value.ndim - 1))
        return mx.contiguous(mx.broadcast_to(row, value.shape))

    return LinearState(
        tuple(map(fill, initial.recurrent)), tuple(map(fill, initial.convolution))
    )


@pytest.mark.parametrize("rows", [1, 4])
def test_initial_state_matches_native_null_slab_positive_zero_bits(
    state_resources: tuple[Qwen35Model, CacheArena], rows: int
) -> None:
    model, arena = state_resources
    request = _plan([0] * rows, tuple((row, 0, 0) for row in range(rows)))
    selected = np.arange(rows, dtype=np.int64)[::-1].copy()
    gathered = LiveStates(arena, model).gather(request, selected)
    initial = model.initial_state(rows)
    # This independent oracle reads the native zero allocation directly;
    # neither model.initial_state nor LiveStates.gather can shortcut it.
    null_slab = arena.read_states(np.zeros(rows, dtype=np.int64))
    mx.eval(
        gathered.recurrent,
        gathered.convolution,
        initial.recurrent,
        initial.convolution,
        null_slab.recurrent,
        null_slab.convolution,
    )
    _assert_state_bits_equal(gathered, null_slab)
    _assert_state_bits_equal(initial, null_slab)
    for value in (*gathered.recurrent, *gathered.convolution):
        integer = mx.uint16 if value.dtype == mx.bfloat16 else mx.uint32
        np.testing.assert_array_equal(
            np.array(value.view(integer)), np.zeros(value.shape)
        )


def test_real_restore_takes_precedence_even_at_context_zero(
    state_resources: tuple[Qwen35Model, CacheArena],
) -> None:
    model, arena = state_resources
    expected = _marked(model, 1.25)
    arena.write_states([arena.state_start(1, 0)], expected)
    actual = LiveStates(arena, model).gather(_plan([0], ((0, 1, 0),)), np.array([0]))
    mx.eval(
        actual.recurrent, actual.convolution, expected.recurrent, expected.convolution
    )
    _assert_state_bits_equal(actual, expected)


def test_mixed_null_held_and_restored_rows_keep_their_selected_order(state_resources):
    model, arena = state_resources
    arena.write_states([arena.state_start(1, 0)], _marked(model, 7))
    states = LiveStates(arena, model)
    # Retain rows from two different preceding batches; their other rows leave.
    for sequences, context in (([101, 103], 13), ([105, 102], 21)):
        request = _plan([context - 1] * 2, (), sequences=sequences)
        states.commit(
            request,
            np.arange(2),
            np.full(2, context),
            _marked(model, [sequence - 100 for sequence in sequences]),
        )
    request = _plan(
        [0, 13, 21, 4096], ((0, 0, 0), (3, 1, 0)), sequences=[100, 101, 102, 106]
    )
    actual = states.gather(request, np.array([3, 2, 0, 1]))
    mx.eval(actual.recurrent, actual.convolution)
    for value in (*actual.recurrent, *actual.convolution):
        values = np.array(value.astype(mx.float32)).reshape(4, -1)
        expected = np.broadcast_to(
            np.array([[7], [2], [0], [1]], np.float32), values.shape
        )
        np.testing.assert_array_equal(values, expected)
