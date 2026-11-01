"""Bounded-warmup compilation and graph contracts for the whole KDA path."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from resin_models_cuda.glm5_3_flash.attention import KdaAttention

BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
pytestmark = pytest.mark.skipif(not BLACKWELL, reason="requires B200/B300 CUDA")


def _attention():
    """Actual 64×128 KDA geometry; smaller dense hidden width keeps tests cheap."""
    torch.manual_seed(947)
    hidden, heads, dim = 64, 64, 128
    width = heads * dim
    config = SimpleNamespace(
        linear_num_heads=heads,
        linear_head_dim=dim,
        rms_norm_eps=1e-5,
        linear_lower_bound=-5.0,
        linear_conv_kernel_dim=4,
    )
    shapes = {
        "q_proj": (width, hidden),
        "k_proj": (width, hidden),
        "v_proj": (width, hidden),
        "b_proj": (heads, hidden),
        "f_a_proj": (dim, hidden),
        "f_b_proj": (width, dim),
        "g_a_proj": (dim, hidden),
        "g_b_proj": (width, dim),
        "o_proj": (hidden, width),
    }
    weights = {
        key + ".weight": torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
        for key, shape in shapes.items()
    }
    weights.update(
        {
            name + "_conv1d.weight": torch.randn(width, 1, 4, device="cuda") * 0.2
            for name in ("q", "k", "v")
        }
    )
    weights.update(
        A_log=torch.randn(heads, device="cuda"),
        dt_bias=torch.randn(width, device="cuda"),
    )
    weights["o_norm.weight"] = torch.ones(dim, device="cuda", dtype=torch.bfloat16)
    return KdaAttention(config, 0, weights)


def _cache(lengths, contexts):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cpu_offsets = torch.tensor(offsets, dtype=torch.int32)
    return SimpleNamespace(
        positions=torch.tensor(
            [
                p
                for length, context in zip(lengths, contexts)
                for p in range(context, context + length)
            ],
            device="cuda",
            dtype=torch.long,
        ),
        sequence_indices=torch.tensor(
            [seq for seq, length in enumerate(lengths) for _ in range(length)],
            device="cuda",
            dtype=torch.long,
        ),
        cu_seqlens=cpu_offsets.cuda(),
        cu_seqlens_cpu=cpu_offsets,
        recurrent=torch.randn(len(lengths), 64, 128, 128, device="cuda") * 0.01,
        convolution=torch.randn(
            len(lengths), 3, 8192, 3, device="cuda", dtype=torch.bfloat16
        )
        * 0.1,
    )


def _jit_entries():
    """Attribute loaded compiled specializations to FLA or Resin, including wrappers.

    The pinned Triton 3.7 JIT device cache exposes compiled variants in slot 0.
    Counting those entries catches disk-cache hits too, unlike wall time alone.
    Heuristics and CachedAutotuner both wrap the underlying JIT in ``fn``.
    """
    from triton.runtime.jit import JITFunction

    device = torch.cuda.current_device()
    result, seen = {}, set()
    for name, module in tuple(sys.modules.items()):
        if module is None or not (
            name.startswith("fla.")
            or name.startswith("resin_models_cuda.glm5_3_flash.")
        ):
            continue
        for value in tuple(vars(module).values()):
            for _ in range(8):
                if isinstance(value, JITFunction):
                    if id(value) not in seen:
                        seen.add(id(value))
                        key = value.fn.__module__ + "." + value.fn.__qualname__
                        cache = value.device_caches.get(device)
                        result[key] = set() if cache is None else set(cache[0])
                    break
                value = getattr(value, "fn", None)
                if value is None:
                    break
    return result


@torch.inference_mode()
def test_complete_kda_path_reuses_jit_after_two_startup_regimes(
    tmp_path, record_property
):
    attention = _attention()

    def run(lengths, contexts):
        cache = _cache(lengths, contexts)
        hidden = torch.randn(sum(lengths), 64, device="cuda", dtype=torch.bfloat16)
        output = attention(hidden, cache.positions, cache)
        assert torch.isfinite(output).all()
        assert torch.isfinite(cache.recurrent).all()

    # Warm exactly two algorithms, not a list covering future context lengths.
    run([1, 1], [0, 16])
    run([65, 31], [0, 16])
    torch.cuda.synchronize()
    before = _jit_entries()
    assert any(
        "fused_recurrent_kda_fwd_kernel" in name and keys
        for name, keys in before.items()
    )
    assert any("chunk_kda_fwd_kernel" in name and keys for name, keys in before.items())
    assert any(
        name.endswith("kernels_cuda._l2norm") and keys for name, keys in before.items()
    )
    assert any(
        name.endswith("kernels_cuda._conv") and keys for name, keys in before.items()
    )
    for lengths, contexts in (
        ([3, 2], [63, 1025]),
        ([511, 2], [128, 8191]),
        ([1025, 17], [2047, 65535]),
        ([2049, 33], [8193, 1040000]),
    ):
        run(lengths, contexts)
    torch.cuda.synchronize()
    after = _jit_entries()
    additions = {
        name: sorted(map(str, keys - before.get(name, set())))
        for name, keys in after.items()
        if keys - before.get(name, set())
    }
    counts = {
        name: {"before": len(before.get(name, set())), "after": len(keys)}
        for name, keys in after.items()
        if keys
    }
    artifact = tmp_path / "glm-kda-compilation.json"
    artifact.write_text(
        json.dumps({"kernels": counts, "new_specializations": additions}, indent=2)
    )
    record_property("kda_compilation_counts", json.dumps(counts, sort_keys=True))
    print(
        f"KDA compilation counters: {json.dumps(counts, sort_keys=True)}; artifact={artifact}"
    )
    assert not additions, f"new context-specific KDA compilations: {additions}"


@torch.inference_mode()
def test_chunk_kda_attention_graph_replays_changed_inputs_and_initial_states():
    attention = _attention()
    cache = _cache([97, 33], [128, 4095])
    hidden = torch.randn(130, 64, device="cuda", dtype=torch.bfloat16)
    initial, history = cache.recurrent, cache.convolution

    def run():
        cache.recurrent, cache.convolution = initial, history
        output = attention(hidden, cache.positions, cache)
        return output, cache.recurrent, cache.convolution

    # This prepares FLA's CPU-derived chunk metadata before CUDA graph capture.
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_state, actual_history = run()
    for seed in (948, 949):
        torch.manual_seed(seed)
        hidden.normal_()
        initial.normal_(std=0.01)
        history.normal_(std=0.1)
        cache.positions.add_(67)
        graph.replay()
        expected, expected_state, expected_history = run()
        torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.02)
        torch.testing.assert_close(actual_state, expected_state, atol=0.001, rtol=0.015)
        torch.testing.assert_close(actual_history, expected_history, atol=0, rtol=0)
