"""Numerical state and page contracts for GLM-5.3-Flash attention."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from resin_models_cuda.glm5_3_flash import kernels
from resin_models_cuda.glm5_3_flash.attention import KdaAttention, SparseAttention
from resin_models_cuda.glm5_3_flash.speculative import SpeculativeTape


def _batch(lengths, contexts=None, device="cpu"):
    contexts = contexts or [0] * len(lengths)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return dict(
        positions=torch.tensor(
            [p for c, n in zip(contexts, lengths) for p in range(c, c + n)],
            device=device,
        ),
        sequence_indices=torch.tensor(
            [r for r, n in enumerate(lengths) for _ in range(n)], device=device
        ),
        cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device=device),
        cu_seqlens_cpu=torch.tensor(offsets, dtype=torch.int32),
    )


def _sparse_cache(
    lengths, contexts=None, *, width=8, index_dim=4, device="cpu", page_tokens=8
):
    # Disjoint request pages and a gap between physical pages exercise the same
    # envelope addressing used by the Engine's interleaved layer allocation.
    page_count = 9
    stride = page_tokens * width + width * 2
    storage = torch.full((page_count, stride), float("nan"), device=device)
    latent = storage.as_strided((page_count, page_tokens, width), (stride, width, 1))
    return SimpleNamespace(
        **_batch(lengths, contexts, device),
        page_tokens=page_tokens,
        page_table=torch.tensor(
            [[1, 3, 5, 7], [2, 4, 6, 8]], dtype=torch.int32, device=device
        )[: len(lengths)],
        latent=latent,
        index_keys=torch.full(
            (page_count, page_tokens, index_dim), float("nan"), device=device
        ),
        index_gates=torch.full(
            (page_count, page_tokens, index_dim), float("nan"), device=device
        ),
        pooled_keys=torch.full(
            (page_count, page_tokens // 4, index_dim), float("nan"), device=device
        ),
    )


def _kda_config():
    return SimpleNamespace(
        linear_num_heads=2,
        linear_head_dim=4,
        rms_norm_eps=1e-5,
        linear_lower_bound=-5.0,
        linear_conv_kernel_dim=4,
    )


def _kda_weights():
    torch.manual_seed(431)
    shapes = {
        "q_proj": (8, 8),
        "k_proj": (8, 8),
        "v_proj": (8, 8),
        "b_proj": (2, 8),
        "f_a_proj": (4, 8),
        "f_b_proj": (8, 4),
        "g_a_proj": (4, 8),
        "g_b_proj": (8, 4),
        "o_proj": (8, 8),
    }
    weights = {
        name + ".weight": torch.randn(shape) * 0.2 for name, shape in shapes.items()
    }
    weights.update(
        {
            name + "_conv1d.weight": torch.randn(8, 1, 4) * 0.2
            for name in ("q", "k", "v")
        }
    )
    weights.update(A_log=torch.randn(2), dt_bias=torch.randn(8))
    weights["o_norm.weight"] = torch.randn(4)
    return weights


def _kda_cache(lengths, contexts=None, states=None):
    return SimpleNamespace(
        **_batch(lengths, contexts),
        recurrent=torch.zeros(len(lengths), 2, 4, 4)
        if states is None
        else states.recurrent.clone(),
        convolution=torch.zeros(len(lengths), 3, 8, 3)
        if states is None
        else states.convolution.clone(),
    )


@pytest.mark.parametrize("counts", [(1, 3), (4, 4), (1, 1)])
def test_kda_attention_speculative_hook_restores_accepted_prefix(monkeypatch, counts):
    attention = KdaAttention(_kda_config(), 0, _kda_weights())
    owners = []
    original_linear = F.linear

    def observe_projection(value, weight, bias=None):
        output = original_linear(value, weight, bias)
        if weight is attention.in_weight:
            owners.append(weakref.ref(output))
        return output

    monkeypatch.setattr(F, "linear", observe_projection)
    hidden = torch.randn(8, 8)
    initial = _kda_cache([4, 4])
    initial.recurrent.normal_(std=0.1)
    initial.convolution.normal_(std=0.1)
    working = _kda_cache([4, 4], states=initial)
    batch = SimpleNamespace(layers={0: working}, context_lengths=torch.tensor([0, 0]))
    tape = SpeculativeTape.attach(batch, 4)
    output = attention(hidden, working.positions, working)
    raw = tape.records[0].raw_qkv
    assert raw.stride(0) > raw.shape[1] * raw.shape[2]
    expected_raw = raw.clone()
    gc.collect()
    assert owners[0]() is not None, "rollback lost the packed projection owner"
    # Allocate unrelated tensors after forward has released its local projected
    # variable. The tape must own the view until accepted-prefix history repair.
    pressure = [torch.randn(8, attention.in_weight.shape[0]) for _ in range(20)]
    torch.testing.assert_close(raw, expected_raw, atol=0, rtol=0)
    assert pressure
    selected = tape.commit(torch.tensor(counts), output)
    assert len(tape.records) == 1
    assert working.speculative_tape is None
    for row, count in enumerate(counts):
        source = SimpleNamespace(
            recurrent=initial.recurrent[row : row + 1],
            convolution=initial.convolution[row : row + 1],
        )
        reference = _kda_cache([count], states=source)
        expected = attention(
            hidden[row * 4 : row * 4 + count], reference.positions, reference
        )
        torch.testing.assert_close(selected[row], expected[-1])
        torch.testing.assert_close(working.recurrent[row], reference.recurrent[0])
        torch.testing.assert_close(working.convolution[row], reference.convolution[0])


def test_kda_chunk_continuation_preserves_conv_and_recurrence():
    model = KdaAttention(_kda_config(), 0, _kda_weights())
    hidden_a, hidden_b = torch.randn(7, 8), torch.randn(5, 8)
    whole = _kda_cache([7, 5])
    expected = model(torch.cat((hidden_a, hidden_b)), whole.positions, whole)
    first = _kda_cache([3, 2])
    out_first = model(torch.cat((hidden_a[:3], hidden_b[:2])), first.positions, first)
    second = _kda_cache([4, 3], [3, 2], first)
    out_second = model(
        torch.cat((hidden_a[3:], hidden_b[2:])), second.positions, second
    )
    actual = torch.cat((out_first[:3], out_second[:4], out_first[3:], out_second[4:]))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(second.recurrent, whole.recurrent, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(second.convolution, whole.convolution)


def test_kda_uses_per_key_decay_not_scalar_head_decay():
    q = torch.tensor([[[1.0, 2.0]]])
    k = torch.tensor([[[3.0, 4.0]]])
    v = torch.tensor([[[0.2, -0.1]]])
    state = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    g, beta = torch.tensor([[[-0.2, -2.0]]]), torch.tensor([[0.3]])
    output, final = kernels.reference_kda(q, k, v, g, beta, state, torch.tensor([0, 1]))
    normalized_k = k[0, 0] / (k[0, 0].square().sum() + 1e-6).sqrt()
    normalized_q = q[0, 0] / (q[0, 0].square().sum() + 1e-6).sqrt() / 2**0.5
    decayed = state[0, 0] * g[0, 0].exp()[:, None]
    golden = (
        decayed
        + normalized_k[:, None] * ((v[0, 0] - normalized_k @ decayed) * 0.3)[None]
    )
    torch.testing.assert_close(final[0, 0], golden)
    torch.testing.assert_close(output[0, 0], normalized_q @ golden)


def test_sparse_pool_tail_and_page_boundary_are_causal():
    torch.manual_seed(11)
    cache = _sparse_cache([11, 7])
    latent, keys, gates, ape = (
        torch.randn(18, 8),
        torch.randn(18, 4),
        torch.randn(18, 4),
        torch.randn(4, 4),
    )
    kernels.store_sparse(latent, keys, gates, ape, cache)
    selected = kernels.select_indices(
        torch.randn(18, 2, 4),
        torch.randn(18, 2),
        cache.positions,
        cache.sequence_indices,
        cache,
        8,
    )
    stride = cache.latent.stride(0) // 8
    for token, (seq, pos) in enumerate(
        zip(cache.sequence_indices.tolist(), cache.positions.tolist())
    ):
        actual = set(selected[token][selected[token] >= 0].tolist())
        # Up to eleven visible tokens: two complete pools plus a causal tail.
        expected = {
            int(cache.page_table[seq, p // 8]) * stride + p % 8 for p in range(pos + 1)
        }
        assert actual == expected
    assert torch.isnan(cache.latent[0]).all()
    assert torch.isnan(
        cache.pooled_keys[cache.page_table[0, 1], 0]
    ).all()  # incomplete pool remains unwritten


def test_sparse_pool_compression_keeps_partial_prefix_inputs():
    torch.manual_seed(15)
    cache = _sparse_cache([3])
    latent, keys, gates, ape = (
        torch.randn(7, 8),
        torch.randn(7, 4),
        torch.randn(7, 4),
        torch.randn(4, 4),
    )
    kernels.store_sparse(latent[:3], keys[:3], gates[:3], ape, cache)
    assert torch.isnan(cache.pooled_keys[1, 0]).all()
    for key, value in _batch([4], [3]).items():
        setattr(cache, key, value)
    kernels.store_sparse(latent[3:], keys[3:], gates[3:], ape, cache)
    expected = ((gates[:4] + ape).softmax(0) * keys[:4]).sum(0)
    torch.testing.assert_close(cache.pooled_keys[1, 0], expected)
    torch.testing.assert_close(cache.index_keys[1, :3], keys[:3])
    torch.testing.assert_close(cache.index_gates[1, :3], gates[:3])


def _sparse_weights():
    torch.manual_seed(21)
    shapes = {
        "q_a_proj": (6, 8),
        "q_b_proj": (8, 6),
        "kv_a_proj_with_mqa": (8, 8),
        "kv_b_proj": (16, 8),
        "o_proj": (8, 8),
        "indexer.wq_b": (8, 6),
        "indexer.wk": (4, 8),
        "indexer.weights_proj": (2, 8),
    }
    weights = {
        name + ".weight": torch.randn(shape) * 0.2 for name, shape in shapes.items()
    }
    weights.update(
        {
            "q_a_layernorm.weight": torch.ones(6),
            "kv_a_layernorm.weight": torch.ones(8),
            "indexer.k_norm.weight": torch.ones(4),
            "indexer.k_norm.bias": torch.zeros(4),
            "indexer.index_kpool_compress_gate": torch.randn(4, 8) * 0.2,
            "indexer.index_kpool_compress_ape": torch.randn(4, 4),
        }
    )
    return weights


def test_absorbed_mla_matches_independent_expanded_kv_reference():
    config = SimpleNamespace(
        num_attention_heads=2,
        qk_nope_head_dim=4,
        v_head_dim=4,
        kv_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=8,
        rms_norm_eps=1e-5,
        qk_rope_head_dim=0,
        index_kpool=4,
    )
    weights = _sparse_weights()
    model = SparseAttention(config, 3, weights)
    hidden, cache = torch.randn(7, 8), _sparse_cache([7])
    result = model(hidden, cache.positions, cache)
    qa = F.rms_norm(
        F.linear(hidden, weights["q_a_proj.weight"]),
        (6,),
        weights["q_a_layernorm.weight"],
        1e-5,
    )
    query = F.linear(qa, weights["q_b_proj.weight"]).view(7, 2, 4)
    kv = F.rms_norm(
        F.linear(hidden, weights["kv_a_proj_with_mqa.weight"]),
        (8,),
        weights["kv_a_layernorm.weight"],
        1e-5,
    )
    expanded = F.linear(kv, weights["kv_b_proj.weight"]).view(7, 2, 8)
    key, value = expanded.split(4, -1)
    golden = []
    for token in range(7):
        score = torch.einsum("hd,shd->hs", query[token], key[: token + 1]) / 2
        output = torch.einsum("hs,shd->hd", score.softmax(-1), value[: token + 1])
        golden.append(F.linear(output.flatten(), weights["o_proj.weight"]))
    torch.testing.assert_close(result, torch.stack(golden), rtol=2e-5, atol=2e-6)


def test_sparse_envelope_rejects_misaligned_physical_pages():
    storage = torch.zeros(3, 65)
    latent = storage.as_strided((3, 8, 8), (65, 8, 1))
    with pytest.raises(ValueError, match="aligned"):
        kernels.latent_envelope(latent)


_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not _BLACKWELL, reason="requires B200/B300 CUDA"),
        ),
    ],
)
def test_sparse_index_head_projection_preserves_fp32_scores(device, monkeypatch):
    config = SimpleNamespace(
        num_attention_heads=2,
        qk_nope_head_dim=4,
        v_head_dim=4,
        kv_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=8,
        rms_norm_eps=1e-5,
        qk_rope_head_dim=0,
        index_kpool=4,
    )
    weights = {
        key: value.bfloat16().to(device) for key, value in _sparse_weights().items()
    }
    attention = SparseAttention(config, 3, weights)
    assert attention.index_weight.dtype == torch.float32
    observed = []

    def select(query, head_weights, positions, sequences, cache, topk):
        observed.append(head_weights)
        return torch.empty(query.shape[0], 0, dtype=torch.int32, device=device)

    # Observe the actual forward's index-score boundary while isolating this
    # projection from page allocation, pool selection and FlashMLA geometry.
    monkeypatch.setattr(kernels, "store_sparse", lambda *args: None)
    monkeypatch.setattr(kernels, "select_indices", select)
    monkeypatch.setattr(
        kernels, "sparse_attention", lambda query, *args: torch.zeros_like(query)
    )
    hidden = torch.randn(7, 8, dtype=torch.bfloat16, device=device)
    positions = torch.arange(7, device=device)
    cache = SimpleNamespace(
        sequence_indices=torch.zeros(7, device=device, dtype=torch.long), latent=None
    )

    def launch():
        attention(hidden, positions, cache)
        return observed[-1]

    def check(actual):
        expected = F.linear(
            hidden.cpu().float(), weights["indexer.weights_proj.weight"].cpu().float()
        )
        assert actual.dtype == torch.float32
        assert not torch.equal(expected, expected.bfloat16().float())
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-6)

    check(launch())
    if device == "cuda":
        for _ in range(2):
            launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = launch()
        hidden.normal_()
        graph.replay()
        check(output)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not _BLACKWELL, reason="requires B200/B300 CUDA"),
        ),
    ],
)
def test_kda_fp32_beta_state_and_speculative_replay(device):
    # Orthogonal keys isolate each update: no prior prediction contributes,
    # so each written state row directly exposes the input-gate precision.
    tokens, heads, dim = 3, 2, 128
    q = torch.zeros(tokens, heads, dim, dtype=torch.bfloat16, device=device)
    for token in range(tokens):
        q[token, :, token] = 1
    k, v = q.clone(), torch.ones_like(q)
    forget = torch.full_like(q, -100)
    beta = torch.full((tokens, heads), 0.5, dtype=torch.bfloat16, device=device)
    a_log = torch.zeros(heads, device=device)
    dt = torch.zeros(heads * dim, device=device)
    initial = torch.zeros(1, heads, dim, dim, device=device)
    history = torch.zeros(1, 3, heads * dim, 3, dtype=torch.bfloat16, device=device)
    raw_qkv = torch.zeros(tokens, 3, heads * dim, dtype=torch.bfloat16, device=device)
    hidden = torch.zeros(tokens, dim, device=device)
    counts = torch.tensor([2], device=device)
    metadata = _batch([tokens], device=device)

    def launch():
        layer = SimpleNamespace(**metadata, recurrent=initial, convolution=history)
        batch = SimpleNamespace(
            layers={0: layer},
            context_lengths=torch.zeros(1, device=device, dtype=torch.long),
        )
        tape = SpeculativeTape.attach(batch, tokens)
        tape.record(layer, raw_qkv, q, k, v, forget, beta, a_log, dt, -5.0)
        _, final = kernels.kda(q, k, v, forget, beta, a_log, dt, -5.0, layer)
        layer.recurrent = final
        tape.commit(counts, hidden)
        return final, layer.recurrent

    def check(actual, accepted):
        expected = torch.zeros_like(initial, device="cpu")
        # Analytic unit-key norm with the serving reference's L2 epsilon.
        scale = torch.rsqrt(torch.tensor(1.0 + 1e-6))
        probabilities = beta.cpu().float().sigmoid()
        old_probabilities = beta.cpu().sigmoid().float()
        assert (probabilities - old_probabilities).abs().min() > 1e-4
        for token in range(accepted):
            expected[0, :, token, :] = probabilities[token, :, None] * scale
        # The old BF16 sigmoid misses this bound by more than two orders of
        # magnitude. Test FP32 state, before the output's intentional BF16 cast.
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-6)

    final, committed = launch()
    check(final, tokens)
    check(committed, 2)
    if device == "cuda":
        for _ in range(2):
            launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            final, committed = launch()
        beta.fill_(0.75)
        counts.fill_(1)
        graph.replay()
        check(final, tokens)
        check(committed, 1)


def _device_cache(source):
    data = {}
    for name, value in vars(source).items():
        if not isinstance(value, torch.Tensor):
            data[name] = value
        elif name == "cu_seqlens_cpu":
            data[name] = value
        elif name in ("latent", "index_keys", "pooled_keys"):
            data[name] = torch.empty_strided(
                value.shape, value.stride(), dtype=torch.bfloat16, device="cuda"
            )
            data[name].copy_(value)
        else:
            data[name] = value.cuda()
    return SimpleNamespace(**data)


@pytest.mark.parametrize("lengths", [[0], [1, 0, 4]])
def test_planar_convolution_matches_causal_equation_and_preserves_idle_history(lengths):
    source = torch.randn(sum(lengths), 3, 16, dtype=torch.bfloat16)[:, :, ::2]
    weights = torch.randn(3, 8, 4)
    history = torch.randn(len(lengths), 2, 3, 8, 3, dtype=torch.bfloat16)[:, 1]
    before = history.clone()
    cache = SimpleNamespace(**_batch(lengths), convolution=history)
    expected, final = torch.empty_like(source), history.clone()
    start = 0
    for row, length in enumerate(lengths):
        joined = torch.cat(
            (history[row], source[start : start + length].permute(1, 2, 0)), -1
        )
        for index in range(length):
            value = (joined[..., index : index + 4].float() * weights).sum(-1)
            expected[start + index] = F.silu(value)
        final[row] = joined[..., -3:]
        start += length
    actual, actual_final = kernels.convolution(source, weights, cache, planar=True)
    assert actual.shape == (3, sum(lengths), 8)
    assert all(plane.is_contiguous() for plane in actual)
    torch.testing.assert_close(
        actual.permute(1, 0, 2), expected, atol=0.008, rtol=0.008
    )
    torch.testing.assert_close(actual_final, final, atol=0, rtol=0)
    torch.testing.assert_close(history, before, atol=0, rtol=0)


@pytest.mark.skipif(not _BLACKWELL, reason="requires B200/B300 CUDA")
@pytest.mark.parametrize("planar", [False, True])
def test_cuda_conv_changed_input_replay_matches_reference(planar):
    torch.manual_seed(31)
    qkv = torch.randn(73, 3, 256, dtype=torch.bfloat16)
    weights = torch.randn(3, 256, 4)
    source = SimpleNamespace(
        **_batch([65, 8]), convolution=torch.randn(2, 3, 256, 3, dtype=torch.bfloat16)
    )
    cache = _device_cache(source)
    inputs = torch.empty(73, 3, 512, dtype=torch.bfloat16, device="cuda")[:, :, ::2]
    inputs.copy_(qkv)
    device_weights = weights.cuda()
    for _ in range(3):
        kernels.convolution(inputs, device_weights, cache, planar=planar)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, final = kernels.convolution(
            inputs, device_weights, cache, planar=planar
        )
    for seed in (32, 33):
        torch.manual_seed(seed)
        qkv.normal_()
        inputs.copy_(qkv)
        source.convolution.normal_()
        cache.convolution.copy_(source.convolution)
        graph.replay()
        expected, expected_final = kernels.convolution(qkv, weights, source)
        torch.testing.assert_close(
            (actual.permute(1, 0, 2) if planar else actual).cpu(),
            expected,
            atol=0.03125,
            rtol=0.008,
        )
        torch.testing.assert_close(final.cpu(), expected_final, atol=0, rtol=0)


@pytest.mark.skipif(not _BLACKWELL, reason="requires B200/B300 CUDA")
@pytest.mark.parametrize("lengths", [[7, 4], [65, 17], [0, 17]])
def test_cuda_kda_matches_sequential_state_oracle(lengths):
    torch.manual_seed(34)
    tokens, heads, dim = sum(lengths), 4, 128
    q, k, v, forget = [
        torch.randn(tokens, heads, dim, dtype=torch.bfloat16) * 0.2 for _ in range(4)
    ]
    beta, a_log, dt = (
        torch.randn(tokens, heads, dtype=torch.bfloat16),
        torch.randn(heads),
        torch.randn(heads * dim),
    )
    source = SimpleNamespace(
        **_batch(lengths), recurrent=torch.randn(len(lengths), heads, dim, dim) * 0.01
    )
    expected, final = kernels.kda(q, k, v, forget, beta, a_log, dt, -5.0, source)
    cache = _device_cache(source)
    actual, actual_final = kernels.kda(
        *[t.cuda() for t in (q, k, v, forget, beta, a_log, dt)], -5.0, cache
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0.004, rtol=0.025)
    torch.testing.assert_close(actual_final.cpu(), final, atol=0.003, rtol=0.025)


@pytest.mark.skipif(not _BLACKWELL, reason="requires B200/B300 CUDA")
@pytest.mark.parametrize("origin", [0, 1])
def test_cuda_sparse_pages_selection_and_mla_match_reference(origin):
    torch.manual_seed(41)
    source = _sparse_cache([75, 54], width=512, index_dim=128, page_tokens=64)
    source.storage_origin = origin
    for name in ("latent", "index_keys", "pooled_keys"):
        value = getattr(source, name)
        cast = torch.empty_strided(value.shape, value.stride(), dtype=torch.bfloat16)
        cast.copy_(value)
        setattr(source, name, cast)
    cache = _device_cache(source)
    latent = torch.randn(129, 512, dtype=torch.bfloat16)
    keys, gates = (
        torch.randn(129, 128, dtype=torch.bfloat16),
        torch.randn(129, 128, dtype=torch.bfloat16),
    )
    ape = torch.randn(4, 128, dtype=torch.bfloat16)
    kernels.store_sparse(latent, keys, gates, ape, source)
    kernels.store_sparse(latent.cuda(), keys.cuda(), gates.cuda(), ape.cuda(), cache)
    # Only compare complete pools; untouched memory deliberately contains NaNs.
    for seq, length in enumerate((75, 54)):
        for pool in range(length // 4):
            slot = pool + origin
            page = int(source.page_table[seq, slot // 16])
            torch.testing.assert_close(
                cache.pooled_keys[page, slot % 16].cpu(),
                source.pooled_keys[page, slot % 16],
                atol=0.015625,
                rtol=0.015625,
            )
    rows = torch.tensor([62, 63, 64, 74, 75 + 53])
    query, weights = (
        torch.randn(5, 32, 128, dtype=torch.bfloat16),
        torch.randn(5, 32, dtype=torch.bfloat16),
    )
    positions, sequences = source.positions[rows], source.sequence_indices[rows]
    expected_indices = kernels.select_indices(
        query, weights, positions, sequences, source, 64
    )
    indices = kernels.select_indices(
        query.cuda(), weights.cuda(), positions.cuda(), sequences.cuda(), cache, 64
    )
    torch.testing.assert_close(
        indices.cpu().sort(-1).values, expected_indices.sort(-1).values, atol=0, rtol=0
    )
    mla_query = torch.randn(5, 64, 512, dtype=torch.bfloat16) * 0.3
    expected = kernels.sparse_attention(mla_query, indices.cpu(), source.latent, 1 / 16)
    actual = kernels.sparse_attention(mla_query.cuda(), indices, cache.latent, 1 / 16)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.015625, rtol=0.025)
    assert torch.isnan(cache.latent[0]).all()


@pytest.mark.skipif(not _BLACKWELL, reason="requires B200/B300 CUDA")
@pytest.mark.parametrize("origin", [0, 1])
def test_cuda_indexer_reuses_compilation_across_context_capacities(origin):
    from resin_models_cuda.glm5_3_flash import kernels_cuda

    torch.manual_seed(44)
    cache = _device_cache(_sparse_cache([2], width=512, index_dim=128, page_tokens=64))
    cache.storage_origin = origin
    cache.pooled_keys.normal_()
    query = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([31, 63], device="cuda")
    sequences = torch.zeros(2, device="cuda", dtype=torch.int64)
    kernels.select_indices(query, weights, positions, sequences, cache, 64)
    device = torch.cuda.current_device()
    before = [
        set(kernel.device_caches[device][0])
        for kernel in (kernels_cuda._score, kernels_cuda._expand)
    ]
    cache.page_table = F.pad(cache.page_table, (0, 3))
    positions.copy_(torch.tensor([65, 95], device="cuda"))
    kernels.select_indices(query, weights, positions, sequences, cache, 64)
    after = [
        set(kernel.device_caches[device][0])
        for kernel in (kernels_cuda._score, kernels_cuda._expand)
    ]
    assert after == before


@pytest.mark.parametrize("boundary", [64, 512])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not _BLACKWELL, reason="requires B200/B300 CUDA"),
        ),
    ],
)
def test_mtp_origin_preserves_shared_prefix_and_logical_pooling(boundary, device):
    from resin.backend.cuda.slab import DevicePageGroup
    from resin_models_cuda.glm5_3_flash.cache import CacheArena, CacheGeometry

    torch.manual_seed(501)
    geometry = CacheGeometry(
        ("linear_attention", "deepseek_sparse_attention"),
        hidden_size=8,
        latent_dim=512,
        index_dim=128,
        heads=1,
        head_dim=4,
        conv_history=3,
        state_tokens=64,
        include_mtp=True,
    )
    prefix_parents = boundary // 64
    arena = CacheArena(
        device, geometry.layout(prefix_parents + 2), geometry, max_batch_size=1
    )

    def view(parents, start, count):
        full = DevicePageGroup(
            torch.tensor([parents], device=device, dtype=torch.int32),
            torch.zeros((1, len(parents)), device=device, dtype=torch.int32),
            torch.zeros(1, device=device, dtype=torch.int32),
        )
        state = DevicePageGroup(
            torch.zeros((1, 1), device=device, dtype=torch.int32),
            torch.zeros((1, 1), device=device, dtype=torch.int32),
            torch.zeros(1, device=device, dtype=torch.int32),
        )
        return (
            arena._make_graph_cache((full, state))
            .batch(
                torch.arange(start, start + count, device=device),
                torch.zeros(count, device=device, dtype=torch.long),
                torch.tensor([0, count], device=device, dtype=torch.int32),
                torch.tensor([start], device=device, dtype=torch.int32),
            )
            .layer(2)
        )

    # A cached target prefix ending at boundary contains MTP pairs only through
    # boundary-2. The next pair depends on the continuation's new anchor token.
    length = boundary - 1
    latent = torch.randn(length, 512, dtype=torch.bfloat16) * 0.2
    keys = (
        ((torch.arange(length) // 4 + 1).float()[:, None] / 4)
        .expand(-1, 128)
        .to(torch.bfloat16)
    )
    gates = torch.randn(length, 128)
    ape = torch.randn(4, 128, dtype=torch.bfloat16)
    shared = list(range(1, prefix_parents + 1))
    prefix = view(shared, 0, length)
    assert prefix.storage_origin == 1
    kernels.store_sparse(
        latent.to(device), keys.to(device), gates.to(device), ape.to(device), prefix
    )
    immutable = arena.storage[1 : prefix_parents + 1].clone()

    branch = view([*shared, prefix_parents + 1], length, 4)
    # This page-origin unit fixture omits Engine STATE groups; supply its
    # published tail explicitly. Engine checkpoint/fork coverage lives in the
    # bounded-tail tests.
    prefix_tail = prefix.index_tail.snapshot()
    branch_latent = torch.randn(4, 512, device=device, dtype=torch.bfloat16) * 0.2
    branch_keys = torch.full((4, 128), 64, device=device, dtype=torch.bfloat16)
    branch_gates = torch.randn(4, 128, device=device)
    device_ape = ape.to(device)
    positions = branch.positions[[0, -1]].clone()
    sequences = torch.zeros(2, device=device, dtype=torch.long)
    query = torch.ones(2, 32, 128, device=device, dtype=torch.bfloat16)
    weights = torch.ones(2, 32, device=device, dtype=torch.bfloat16)
    mla_query = torch.randn(2, 64, 512, device=device, dtype=torch.bfloat16) * 0.2

    def launch():
        branch.index_tail.assign(prefix_tail)
        kernels.store_sparse(
            branch_latent, branch_keys, branch_gates, device_ape, branch
        )
        selected = kernels.select_indices(
            query, weights, positions, sequences, branch, 64
        )
        return selected, kernels.sparse_attention(
            mla_query, selected, branch.latent, 1 / 16
        )

    if device == "cuda":
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            selected, attended = launch()
    for alternative, key_value in enumerate((64, -64)):
        # Replay changes both the branch data and its private physical parent;
        # the entire prefix still aliases exactly the same immutable parents.
        assert branch.page_table is not None
        branch.page_table[0, -1] = (
            prefix_parents + 1 + alternative
        ) * geometry.slab_units + 1
        branch_keys.fill_(key_value)
        branch_latent.normal_(std=0.2)
        branch_gates.normal_()
        if device == "cuda":
            graph.replay()
        else:
            selected, attended = launch()
        assert torch.equal(arena.storage[1 : prefix_parents + 1], immutable)

        # Independent oracle works only with logical token rows, without any
        # page/origin adapter. It preserves HF's BF16 softmax/product boundaries.
        logical_latent = torch.cat((latent, branch_latent.cpu()))
        logical_keys = torch.cat((keys, branch_keys.cpu()))
        logical_gates = torch.cat((gates, branch_gates.cpu()))
        table = branch.page_table[0].cpu().tolist()
        stride = arena.latent.stride(0) // geometry.latent_dim
        for row, pos in enumerate(positions.cpu().tolist()):
            pools = (pos + 1) // 4
            pool_keys = logical_keys[: pools * 4].reshape(pools, 4, 128)
            pool_gates = logical_gates[: pools * 4].reshape(pools, 4, 128)
            expected_pools = (
                (pool_gates + ape.float()).softmax(1).to(pool_keys.dtype) * pool_keys
            ).sum(1)
            for pool, expected in enumerate(expected_pools):
                slot = pool + 1
                actual = arena.pooled_keys[table[slot // 16], slot % 16].cpu()
                torch.testing.assert_close(actual, expected, atol=0.0625, rtol=0.008)
            scores = expected_pools.float().sum(-1) * (128**-0.5) * (32**0.5)
            chosen = scores.topk(min(16, pools), sorted=False).indices
            logical = torch.cat(
                (
                    (chosen[:, None] * 4 + torch.arange(4)).flatten(),
                    torch.arange(pools * 4, pos + 1),
                )
            )
            actual = selected[row].cpu()
            actual = [
                table.index(int(index) // stride) * 64 + int(index) % stride - 1
                for index in actual
                if index >= 0
            ]
            assert sorted(actual) == sorted(logical.tolist())
            values = logical_latent[logical].float()
            attention = (mla_query[row].cpu().float() @ values.T / 16).softmax(
                -1
            ) @ values
            torch.testing.assert_close(
                attended[row].cpu().float(), attention, atol=0.004, rtol=0.02
            )


@pytest.mark.skipif(not _BLACKWELL, reason="requires B200/B300 CUDA")
@pytest.mark.parametrize("origin", [0, 1])
@torch.inference_mode()
def test_sparse_cache_addresses_cross_signed32_boundaries_and_replay(origin):
    """Real strides cross 4 GiB BF16 and 8 GiB FP32 address boundaries.

    An 8 GiB prefix keeps the old signed-wrap addresses inside the allocation,
    so this regression fails numerically without an illegal CUDA access. Only
    a few pages are touched. Unused rows and wrapped-address canaries are NaNs,
    representing recycled FULL pages formerly used for recurrent checkpoints.
    """
    # Preserve the independently exercised legacy full-history adapter's wide
    # address regression. Production now keeps raw keys/gates in STATE tails;
    # their current-layout graph/address tests are in test_glm53_index_tail.py.
    geometry = SimpleNamespace(
        latent_dim=512,
        index_dim=128,
        unit_bytes=118784,
        latent_bytes=65536,
        index_bytes=16384,
        gate_bytes=32768,
    )
    assert (geometry.latent_dim, geometry.index_dim) == (512, 128)
    bf16_stride, fp32_stride = geometry.unit_bytes // 2, geometry.unit_bytes // 4
    first_bf16_overflow = (1 << 31) // bf16_stride + 1
    first_fp32_overflow = (1 << 31) // fp32_stride + 1
    pages = (2, first_bf16_overflow + 7, first_fp32_overflow + 11)
    assert (1 << 31) <= pages[1] * bf16_stride < (1 << 32)
    assert pages[2] * fp32_stride >= (1 << 31)
    prefix_bytes, page_count = 1 << 33, max(pages) + 1
    allocation_bytes = prefix_bytes + page_count * geometry.unit_bytes
    if torch.cuda.mem_get_info()[0] < allocation_bytes + (1 << 29):
        pytest.skip(
            "wide-address regression needs about 17 GiB of free CUDA memory"  # ty: ignore[too-many-positional-arguments]
        )
    storage = torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda")
    # Offsets are bytes within the production page unit, not separate buffers.
    fields = (
        ("latent", torch.bfloat16, 64, 512, 0),
        ("index_keys", torch.bfloat16, 64, 128, geometry.latent_bytes),
        (
            "index_gates",
            torch.float32,
            64,
            128,
            geometry.latent_bytes + geometry.index_bytes,
        ),
        (
            "pooled_keys",
            torch.bfloat16,
            16,
            128,
            geometry.latent_bytes + geometry.index_bytes + geometry.gate_bytes,
        ),
    )
    cache_fields, wrapped_canaries = {}, []
    for name, dtype, rows, width, byte_offset in fields:
        typed = storage.view(dtype)
        element_bytes = typed.element_size()
        stride = geometry.unit_bytes // element_bytes
        base = (prefix_bytes + byte_offset) // element_bytes
        cache_fields[name] = typed.as_strided(
            (page_count, rows, width), (stride, width, 1), base
        )
        cache_fields[name][0].fill_(float("nan"))
        for page in pages:
            cache_fields[name][page].fill_(float("nan"))
            exact = page * stride
            wrapped = (exact + (1 << 31)) % (1 << 32) - (1 << 31)
            if exact == wrapped:
                continue
            # Prove the legacy kernel can safely execute this case in the
            # regression harness; it writes only these in-bounds canaries.
            assert 0 <= base + wrapped
            assert base + wrapped + rows * width <= typed.numel()
            canary = typed.as_strided((rows, width), (width, 1), base + wrapped)
            canary.fill_(float("nan"))
            wrapped_canaries.append(canary)
    cache = SimpleNamespace(
        positions=torch.arange(16, device="cuda", dtype=torch.int64),
        sequence_indices=torch.zeros(16, device="cuda", dtype=torch.int64),
        page_table=torch.tensor([[pages[0]]], device="cuda", dtype=torch.int32),
        page_tokens=64,
        storage_origin=origin,
        **cache_fields,
    )
    latent = torch.empty(16, 512, device="cuda", dtype=torch.bfloat16)
    keys = torch.empty(16, 128, device="cuda", dtype=torch.bfloat16)
    gates = torch.empty(16, 128, device="cuda", dtype=torch.float32)
    ape = torch.zeros(4, 128, device="cuda", dtype=torch.bfloat16)
    index_query = torch.ones(1, 64, 128, device="cuda", dtype=torch.bfloat16)
    index_weight = torch.ones(1, 64, device="cuda", dtype=torch.float32)
    query_positions = torch.tensor([15], device="cuda", dtype=torch.int64)
    sequences = torch.zeros(1, device="cuda", dtype=torch.int64)
    mla_query = torch.zeros(1, 64, 512, device="cuda", dtype=torch.bfloat16)

    def inputs(page, phase):
        cache.page_table.fill_(page)
        row = torch.arange(16, device="cuda", dtype=torch.float32)
        pool = row.div(4, rounding_mode="floor")
        # Constant keys/gates within each pool give an exact, independent
        # pooling oracle. Reversing the keys changes the selected logical pool.
        key_value = 4 - pool if phase % 2 else pool + 1
        latent.copy_((row[:, None] / 16 + phase / 8).expand_as(latent))
        keys.copy_(key_value[:, None].expand_as(keys))
        gates.copy_((pool[:, None] / 8 + phase / 16).expand_as(gates))
        for tensor in cache_fields.values():
            tensor[page].fill_(float("nan"))

    def launch():
        kernels.store_sparse(latent, keys, gates, ape, cache)
        selected = kernels.select_indices(
            index_query, index_weight, query_positions, sequences, cache, topk=4
        )
        # This test selects one pool; the pinned SM100 head64 FlashMLA
        # implementation requires at least 128 index slots (normally 2112).
        mla_indices = F.pad(selected, (0, 128 - selected.shape[-1]), value=-1)
        attended = kernels.sparse_attention(mla_query, mla_indices, cache.latent, 0.125)
        return selected, attended

    def check(page, phase, selected, attended):
        # Check each store separately: FP32 gate offsets overflow at twice the
        # byte extent of BF16 latent/key/pool offsets.
        for name, expected in (
            ("latent", latent),
            ("index_keys", keys),
            ("index_gates", gates),
        ):
            actual = cache_fields[name][page, origin : origin + 16]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        pooled = cache.pooled_keys[page, origin : origin + 4]
        torch.testing.assert_close(pooled, keys[::4], atol=0, rtol=0)
        chosen_pool = 0 if phase % 2 else 3
        stride = cache.latent.stride(0) // cache.latent.shape[-1]
        expected_indices = (
            page * stride + chosen_pool * 4 + origin + torch.arange(4, device="cuda")
        )
        torch.testing.assert_close(selected[0, :4].long(), expected_indices)
        assert torch.all(selected[0, 4:] == -1)
        # Query zero makes attention an exact mean of the four selected rows.
        expected_value = (chosen_pool * 4 + 1.5) / 16 + phase / 8
        torch.testing.assert_close(
            attended, torch.full_like(attended, expected_value), atol=0, rtol=0
        )
        for name, tensor in cache_fields.items():
            last = origin + (4 if name == "pooled_keys" else 16)
            assert torch.isnan(tensor[page, last:]).all(), name
        for canary in wrapped_canaries:
            assert torch.isnan(canary).all(), "legacy signed-wrap address was written"

    for phase, page in enumerate(pages):
        inputs(page, phase)
        check(page, phase, *launch())
    inputs(pages[0], 0)
    for _ in range(2):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected, attended = launch()
    # Reuse the captured graph while changing page IDs and actual source data.
    for phase, page in enumerate((*pages[1:], pages[0], pages[2]), start=3):
        inputs(page, phase)
        graph.replay()
        check(page, phase, selected, attended)
