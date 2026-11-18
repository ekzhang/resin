"""Exact data movement for accepted-prefix rollback; no recurrence replacement."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch

from resin_models_cuda.glm5_3_flash.speculative import SpeculativeTape

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _fixture(rows, width, *, heads=2, dim=16, dtype=torch.bfloat16):
    channels = heads * dim
    forget = torch.randn(rows * width, channels + 6, device="cuda", dtype=dtype)[
        :, 3 : 3 + channels
    ].reshape(rows * width, heads, dim)
    beta = torch.randn(rows * width, heads * 2, device="cuda", dtype=dtype)[:, ::2]
    history = torch.randn(rows, 3, channels, 6, device="cuda", dtype=dtype)[..., ::2]
    packed = torch.randn(rows * width, 3 * channels + 320, device="cuda", dtype=dtype)
    raw = packed[:, 7 : 7 + 3 * channels].reshape(rows * width, 3, channels)
    hidden = torch.randn(rows * width, 128, device="cuda", dtype=dtype)[:, 1::2]
    counts = ((torch.arange(rows, device="cuda") % width) + 1).repeat_interleave(2)[::2]
    contexts = torch.arange(rows * 2, device="cuda", dtype=torch.int32)[::2] * 8192 - 1
    if rows:
        forget[0, 0, 0] = -0.0
        beta[0, 0] = -0.0
        history[0, 0, 0, 1] = -0.0
        raw[0, 0, 0] = -0.0
        hidden[0, 0] = -0.0
        # Noncanonical NaN bits distinguish a copy from an unintended conversion.
        bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
        payload = 0x7FC1 if dtype == torch.bfloat16 else 0x7FC00001
        for value in (forget, beta, history, raw, hidden):
            value.view(bits)[tuple(-1 for _ in value.shape)] = payload
    return counts, contexts, forget, beta, history, raw, hidden


def _reference(data, width):
    counts, contexts, forget, beta, history, raw, hidden = data
    rows = counts.shape[0]
    accepted = (torch.arange(width, device="cuda")[None] < counts[:, None]).flatten()
    masked_f = forget.masked_fill(~accepted[:, None, None], -torch.inf)
    masked_b = beta.masked_fill(~accepted[:, None], -torch.inf)
    source = torch.cat(
        (history.permute(0, 3, 1, 2), raw.reshape(rows, width, *raw.shape[1:])), 1
    )
    indices = counts[:, None] + torch.arange(3, device="cuda")
    selected_history = (
        source[torch.arange(rows, device="cuda")[:, None], indices]
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    selected = hidden.reshape(rows, width, -1)[
        torch.arange(rows, device="cuda"), counts.long() - 1
    ]
    return masked_f, masked_b, selected_history, selected, contexts + counts


def _actual(data, width):
    from resin_models_cuda.glm5_3_flash.speculative_cuda import (
        finish_commit,
        prepare_replay,
        select_history,
    )

    counts, contexts, forget, beta, history, raw, hidden = data
    return (
        *prepare_replay(counts, width, forget, beta, history, raw)[:2],
        select_history(counts, width, history, raw),
        *finish_commit(counts, width, contexts, hidden),
    )


def _bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("width", range(1, 9))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_replay_preparation_matches_original_torch_bitwise(width, dtype):
    data = _fixture(9, width, dtype=dtype)
    saved = [tensor.clone() for tensor in data]
    actual, expected = _actual(data, width), _reference(data, width)
    for got, want in zip(actual, expected, strict=True):
        _bits(got, want)
    source_storages = {x.untyped_storage().data_ptr() for x in data}
    assert len({x.untyped_storage().data_ptr() for x in actual}) == len(actual)
    assert not source_storages.intersection(
        x.untyped_storage().data_ptr() for x in actual
    )
    for source, original in zip(data, saved, strict=True):
        _bits(source, original)
    # Include every final-row NaN payload in the accepted region as well.
    data[0].fill_(width)
    for got, want in zip(_actual(data, width), _reference(data, width), strict=True):
        _bits(got, want)


@torch.inference_mode()
def test_replay_preparation_real_geometry_and_graph_lifetime():
    data, width = _fixture(8, 4, heads=64, dim=128), 4
    for _ in range(2):
        _actual(data, width)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = _actual(data, width)
    gc.collect()
    # Keep the original inputs and graph outputs; all temporary helper objects die.
    pressure = [
        torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda") for _ in range(8)
    ]
    del pressure
    for counts in ([4] * 8, [1] * 8, [4, 3, 2, 1, 1, 2, 3, 4]):
        data[0].copy_(torch.tensor(counts, device="cuda"))
        for tensor in data[2:]:
            tensor.normal_()
        graph.replay()
        for got, want in zip(result, _reference(data, width), strict=True):
            _bits(got, want)


@torch.inference_mode()
def test_actual_tape_owns_initial_states_and_matches_preparation(monkeypatch):
    from resin_models_cuda.glm5_3_flash import kda_state_cuda

    data, width = _fixture(3, 4), 4
    counts, contexts, forget, beta, history, raw, hidden = data
    initial = torch.randn(3, 2, 16, 16, device="cuda")
    prior = initial.clone()
    q = torch.randn(12, 2, 16, device="cuda", dtype=torch.bfloat16)
    expected = _reference(data, width)

    def recurrence(key, value, raw_f, raw_b, a, dt, state, offsets, accepted, bound):
        assert state is initial
        assert raw_f is forget and raw_b is beta
        assert accepted is counts
        return initial.clone()

    monkeypatch.setattr(kda_state_cuda, "accepted_kda_state", recurrence)
    layer = SimpleNamespace(
        recurrent=initial,
        convolution=history,
        cu_seqlens=torch.arange(4, device="cuda", dtype=torch.int32) * width,
    )
    batch = SimpleNamespace(layers={0: layer}, context_lengths=contexts)
    tape = SpeculativeTape.attach(batch, width)
    tape.record(
        layer,
        raw,
        q,
        q,
        q,
        forget,
        beta,
        torch.zeros(2, device="cuda"),
        torch.zeros(32, device="cuda"),
        -5.0,
    )
    selected = tape.commit(counts, hidden)
    _bits(selected, expected[3])
    _bits(batch.lengths, expected[4])
    _bits(layer.convolution, expected[2])
    _bits(initial, prior)
    assert (
        layer.recurrent.untyped_storage().data_ptr()
        != initial.untyped_storage().data_ptr()
    )
    assert layer.speculative_tape is None


def _entries():
    from resin_models_cuda.glm5_3_flash import speculative_cuda as module

    device = torch.cuda.current_device()
    return {
        name: set(getattr(module, name).device_caches[device][0])
        for name in ("_prepare_replay", "_select_history", "_finish_commit")
    }


@torch.inference_mode()
def test_rows_width_counts_contexts_and_strides_do_not_specialize():
    first = _fixture(2, 1)
    _actual(first, 1)
    torch.cuda.synchronize()
    baseline = _entries()
    assert all(baseline.values())
    for rows, width in ((1, 2), (3, 3), (8, 4), (7, 5), (16, 6), (4, 7), (32, 8)):
        data = _fixture(rows, width)
        data[1].fill_(131071)
        for got, want in zip(
            _actual(data, width), _reference(data, width), strict=True
        ):
            _bits(got, want)
    torch.cuda.synchronize()
    assert _entries() == baseline


@torch.inference_mode()
def test_source_addresses_above_two_gib_are_not_truncated():
    from resin_models_cuda.glm5_3_flash.speculative_cuda import (
        finish_commit,
        prepare_replay,
        select_history,
    )

    if torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        pytest.skip("wide-address oracle needs a 2GiB strided allocation")  # ty: ignore[too-many-positional-arguments]
    # Only the two small rows are initialized; never compare or clone the arena.
    stride = 2**30 + 128
    storage = torch.empty(stride + 32, device="cuda", dtype=torch.bfloat16)
    raw = storage.as_strided((2, 3, 4), (stride, 4, 1), 7)
    raw[0].fill_(13)
    raw[1].fill_(27)
    counts = torch.tensor([2], device="cuda")
    forget = torch.zeros(2, 1, 4, device="cuda", dtype=torch.bfloat16)
    beta = torch.zeros(2, 1, device="cuda", dtype=torch.bfloat16)
    history = torch.full((1, 3, 4, 3), 5, device="cuda", dtype=torch.bfloat16)
    _, _, repaired = prepare_replay(counts, 2, forget, beta, history, raw)
    _bits(select_history(counts, 2, history, raw), repaired)
    _bits(repaired[..., 0], history[..., 2])
    _bits(repaired[..., 1], raw[0][None])
    _bits(repaired[..., 2], raw[1][None])
    hidden = storage.as_strided((2, 4), (stride, 1), 7)
    selected, lengths = finish_commit(
        counts, 2, torch.tensor([131071], device="cuda"), hidden
    )
    _bits(selected, hidden[1:2])
    assert lengths.item() == 131073
