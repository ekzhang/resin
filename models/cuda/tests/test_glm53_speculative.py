"""Native MTP alignment and block acceptance/state regression tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from resin.backend.cuda.slab import DevicePageGroup
from resin_models_cuda.glm5_3_flash import kernels
from resin_models_cuda.glm5_3_flash.attention import rms_norm
from resin_models_cuda.glm5_3_flash.cache import CacheArena, CacheGeometry, LayerCache
from resin_models_cuda.glm5_3_flash.speculative import (
    SpeculativeTape,
    prime_mtp,
    speculative_step,
)


def _template(positions, sequences, offsets, contexts):
    return LayerCache(
        positions,
        sequences,
        offsets,
        contexts,
        contexts + offsets[1:] - offsets[:-1],
    )


def test_mtp_prefill_uses_previous_target_and_current_image_embedding():
    positions = torch.tensor([0, 1, 2, 7, 8])
    sequences = torch.tensor([0, 0, 0, 1, 1])
    offsets, contexts = torch.tensor([0, 3, 5]), torch.tensor([0, 7])
    template = _template(positions, sequences, offsets, contexts)
    hidden = torch.arange(10).float().reshape(5, 2) + 1
    carry = torch.tensor([[90.0, 91.0], [80.0, 81.0]])
    embedded = torch.arange(10).float().reshape(5, 2) + 30
    recorded = []
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=1, rms_norm_eps=1e-6),
        norm=torch.tensor([1.2, 0.7]),
        mtp=lambda *args: recorded.append(args),
    )
    batch = SimpleNamespace(
        positions=positions,
        sequence_indices=sequences,
        cu_seqlens=offsets,
        cu_seqlens_cpu=offsets,
        context_lengths=contexts,
        previous_hidden=carry,
        layer=lambda _: template,
    )
    prime_mtp(
        model,
        batch,
        torch.tensor([3, 4, 5, 6, 7]),
        hidden,
        positions.tolist(),
        input_embeddings=embedded,
    )
    actual_embedded, previous, actual_positions, cache = recorded[0]
    torch.testing.assert_close(actual_embedded, embedded[1:])
    expected_previous = torch.stack((hidden[0], hidden[1], carry[1], hidden[3]))
    torch.testing.assert_close(previous, rms_norm(expected_previous, model.norm, 1e-6))
    assert actual_positions.tolist() == [0, 1, 6, 7]
    assert cache.layer(1).cu_seqlens.tolist() == [0, 2, 4]
    assert cache.layer(1).sequence_indices.tolist() == [0, 0, 1, 1]


class _ToyModel:
    """Distinct target and draft transitions expose every acceptance boundary."""

    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=1, rms_norm_eps=1e-6)
        self.norm = torch.tensor([1.0, 1.0])
        self.embedding = torch.arange(64).reshape(32, 2).float()
        self.target_calls, self.draft_calls, self.repairs = [], [], []

    @staticmethod
    def _logits(tokens):
        return torch.nn.functional.one_hot(tokens.long() % 32, 32).float() * 20

    def mtp_forward(self, ids, hidden, positions, cache):
        assert getattr(cache, "projection_rows", 0) == 0
        self.draft_calls.append((ids.clone(), hidden.clone(), positions.clone()))
        return self._logits(ids + 1), hidden + 1

    def forward_hidden(self, ids, positions, cache):
        assert cache.is_decode, "verification must use the target decode reduction"
        assert cache.projection_rows == 64
        self.target_calls.append(ids.clone())
        return torch.stack((ids, positions), dim=1).float()

    def logits(self, hidden, *, projection_rows=0):
        assert projection_rows == 64
        ids = hidden[:, 0].long()
        target = torch.where(ids == 1, 7, torch.where(ids == 9, 14, ids + 1))
        return self._logits(target)

    def mtp(self, embedded, previous, positions, cache):
        assert getattr(cache, "projection_rows", 0) == 0
        self.repairs.append((embedded.clone(), previous.clone(), positions.clone()))


class _ToyCache:
    committed_hidden: torch.Tensor
    committed_lengths: torch.Tensor

    def batch(self, positions, sequences, offsets, contexts):
        template = _template(positions, sequences, offsets, contexts)
        batch = SimpleNamespace(
            layers={1: template},
            layer=lambda _: template,
            previous_hidden=torch.ones(contexts.shape[0], 2),
            context_lengths=contexts,
        )

        def store(hidden):
            self.committed_hidden, self.committed_lengths = (
                hidden.clone(),
                batch.lengths.clone(),
            )

        batch.store = store
        return batch


def test_speculation_verifies_once_and_repairs_first_middle_all_acceptance():
    model, cache = _ToyModel(), _ToyCache()
    positions = torch.tensor([3, 6, 10])
    sampling_calls = []

    def sample(logits):
        sampling_calls.append(logits.shape)
        return logits.argmax(-1)

    acceptance, logits = speculative_step(
        model,
        cache,
        torch.tensor([1, 8, 16]),
        positions,
        3,
        sample,
        projection_rows=64,
    )
    tokens, counts, feedback = (
        acceptance.tokens,
        acceptance.counts,
        acceptance.feedback(),
    )
    assert len(model.target_calls) == 1
    assert model.target_calls[0].tolist() == [1, 2, 3, 4, 8, 9, 10, 11, 16, 17, 18, 19]
    assert logits.shape == (3, 4, 32)
    assert sampling_calls == [(3, 4, 32)]
    assert counts.tolist() == [1, 2, 4]
    assert [tokens[i, :n].tolist() for i, n in enumerate(counts)] == [
        [7],
        [9, 14],
        [17, 18, 19, 20],
    ]
    assert feedback.positions.tolist() == [4, 8, 14]
    assert cache.committed_lengths.tolist() == [4, 8, 14]
    assert cache.committed_hidden.tolist() == [[1.0, 3.0], [9.0, 7.0], [19.0, 13.0]]
    assert [call[2].tolist() for call in model.draft_calls] == [
        [2, 5, 9],
        [3, 6, 10],
        [4, 7, 11],
    ]
    repair_embedded, previous, repair_positions = model.repairs[0]
    repair_ids = (repair_embedded[:, 0] / 2).long().reshape(3, 4)
    assert repair_ids.tolist() == [[7, 3, 4, 5], [9, 14, 11, 12], [17, 18, 19, 20]]
    assert repair_positions.tolist() == [3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13]
    verified = torch.stack((model.target_calls[0], repair_positions), dim=1).float()
    torch.testing.assert_close(previous, rms_norm(verified, model.norm, 1e-6))


@pytest.mark.parametrize("depth", [-1, 0, 1, 3])
def test_fixed_acceptance_is_runtime_data_and_commits_matching_state(depth):
    model, cache = _ToyModel(), _ToyCache()
    positions = torch.tensor([3, 6, 10])
    acceptance, _ = speculative_step(
        model,
        cache,
        torch.tensor([1, 8, 16]),
        positions,
        3,
        lambda logits: logits.argmax(-1),
        projection_rows=64,
        fixed_acceptance=torch.tensor(depth),
    )
    expected_counts = (
        torch.tensor([1, 2, 4]) if depth < 0 else torch.full((3,), depth + 1)
    )
    torch.testing.assert_close(cache.committed_lengths, positions + expected_counts)
    torch.testing.assert_close(acceptance.counts.long(), expected_counts)
    torch.testing.assert_close(acceptance.feedback().positions, cache.committed_lengths)


def test_empty_mtp_priming_still_joins_expert_collective():
    calls = []
    model = SimpleNamespace(
        mtp=True, forward_mtp_experts_empty=lambda: calls.append("empty")
    )
    prime_mtp(model, None, torch.tensor([2]), torch.zeros(1, 8), [0])
    assert calls == ["empty"]


def _rollback_inputs(device="cpu"):
    torch.manual_seed(831)
    rows, width, heads, dim = 3, 4, 2, 128
    tensors = [
        torch.randn(rows * width, heads, dim, dtype=torch.bfloat16) * 0.2
        for _ in range(4)
    ]
    beta = torch.randn(rows * width, heads, dtype=torch.bfloat16)
    a_log, dt = torch.randn(heads), torch.randn(heads * dim)
    raw = torch.randn(rows * width, 3, heads * dim, dtype=torch.bfloat16)
    # Attention now retains QKV inside the larger packed projection allocation.
    # Exercise the same noncompact row stride through accepted-history repair,
    # including the changed-acceptance CUDA graph test below.
    projection = torch.full(
        (rows * width, 3 * heads * dim + 320),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    raw_view = projection[:, : 3 * heads * dim].reshape(rows * width, 3, heads * dim)
    raw_view.copy_(raw.to(device))
    state = torch.randn(rows, heads, dim, dim) * 0.01
    history = torch.randn(rows, 3, heads * dim, 3, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        recurrent=state.to(device),
        convolution=history.to(device),
        cu_seqlens=torch.arange(rows + 1, device=device, dtype=torch.int32) * width,
    )
    batch = SimpleNamespace(
        layers={0: layer}, context_lengths=torch.tensor([4, 20, 70], device=device)
    )
    tape = SpeculativeTape.attach(batch, width)
    tape.record(
        layer,
        raw_view,
        *[t.to(device) for t in tensors],
        beta.to(device),
        a_log.to(device),
        dt.to(device),
        -5.0,
    )
    hidden = (
        torch.arange(rows * width * 8, device=device).reshape(rows * width, 8).float()
    )
    return tape, hidden, tensors, beta, a_log, dt, raw, state, history


def _check_rollback(
    tape, selected, hidden, counts, tensors, beta, a_log, dt, raw, state, history
):
    layer = tape.records[0].layer
    for row, count in enumerate(counts):
        start, end = row * 4, row * 4 + count
        cache = SimpleNamespace(
            recurrent=state[row : row + 1], cu_seqlens=torch.tensor([0, count])
        )
        _, expected = kernels.kda(
            *[t[start:end] for t in tensors], beta[start:end], a_log, dt, -5.0, cache
        )
        torch.testing.assert_close(
            layer.recurrent[row].cpu(), expected[0], atol=0.003, rtol=0.025
        )
        source = torch.cat((history[row].permute(2, 0, 1), raw[start:end]))
        torch.testing.assert_close(
            layer.convolution[row].cpu(), source[-3:].permute(1, 2, 0), atol=0, rtol=0
        )
        torch.testing.assert_close(
            selected[row].cpu(), hidden[row * 4 + count - 1].cpu()
        )


def test_kda_rollback_matches_independent_accepted_prefix_recurrence():
    tape, hidden, *inputs = _rollback_inputs()
    counts = torch.tensor([1, 2, 4])
    selected = tape.commit(counts, hidden)
    _check_rollback(tape, selected, hidden, counts.tolist(), *inputs)


def test_accepted_block_crosses_checkpoint_and_preserves_cached_prefix():
    _, hidden, tensors, beta, a_log, dt, raw, state, history = _rollback_inputs()
    geometry = CacheGeometry(
        ("linear_attention", "deepseek_sparse_attention"),
        hidden_size=8,
        latent_dim=8,
        index_dim=4,
        heads=2,
        head_dim=128,
        conv_history=3,
        state_tokens=64,
    )
    arena = CacheArena("cpu", geometry.layout(9), geometry, max_batch_size=3)
    full = DevicePageGroup(
        torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.int32),
        torch.tensor([[0, 1]] * 3, dtype=torch.int32),
        torch.zeros(3, dtype=torch.int32),
    )
    recurrent = DevicePageGroup(
        torch.tensor([[4, 5], [6, 7], [8, 9]], dtype=torch.int32),
        torch.zeros(3, 2, dtype=torch.int32),
        torch.zeros(3, dtype=torch.int32),
    )
    cache = arena._make_graph_cache((full, recurrent))
    positions = torch.tensor([63, 64, 62])
    cache.store_state(
        positions - 1,
        state[:, None],
        history[:, None],
        torch.zeros(3, 8, dtype=torch.bfloat16),
    )
    block_positions = (positions[:, None] + torch.arange(4)).flatten()
    batch = cache.batch(
        block_positions,
        torch.arange(3).repeat_interleave(4),
        torch.arange(4, dtype=torch.int32) * 4,
        positions,
    )
    tape = SpeculativeTape.attach(batch, 4)
    tape.record(batch.layer(0), raw, *tensors, beta, a_log, dt, -5.0)
    counts = torch.tensor([2, 4, 1])
    selected = tape.commit(counts, hidden)
    batch.store(selected.to(torch.bfloat16))
    resumed = cache.batch(
        positions + counts,
        torch.arange(3),
        torch.arange(4, dtype=torch.int32),
        positions + counts,
    )
    torch.testing.assert_close(resumed.layer(0).recurrent, batch.layer(0).recurrent)
    torch.testing.assert_close(resumed.layer(0).convolution, batch.layer(0).convolution)
    torch.testing.assert_close(resumed.previous_hidden, selected.to(torch.bfloat16))
    # Row 1 started from a published prefix at boundary 64. Its checkpoint
    # remains untouched while accepted continuation is committed in block 1.
    prior, prior_conv, _ = cache.state_at(torch.tensor([62, 63, 61]))
    torch.testing.assert_close(prior[1, 0], state[1])
    torch.testing.assert_close(prior_conv[1, 0], history[1])
    _check_rollback(
        tape,
        selected,
        hidden,
        counts.tolist(),
        tensors,
        beta,
        a_log,
        dt,
        raw,
        state,
        history,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_kda_rollback_changed_acceptance_graph_replay():
    tape, hidden, *inputs = _rollback_inputs("cuda")
    counts = torch.tensor([1, 2, 4], device="cuda")
    for _ in range(3):
        tape.commit(counts, hidden)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = tape.commit(counts, hidden)
    for values in ([1, 4, 2], [4, 2, 1]):
        counts.copy_(torch.tensor(values, device="cuda"))
        graph.replay()
        _check_rollback(tape, selected, hidden, values, *inputs)
