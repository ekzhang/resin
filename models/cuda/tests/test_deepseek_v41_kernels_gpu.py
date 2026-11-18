"""Blackwell primitive and packed-cache attention numerical contracts."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="requires Blackwell CUDA",
)


def _check(actual, expected, *, atol=0, rtol=0):
    actual, expected = actual.float(), expected.float()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    difference = actual - expected
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / expected.norm().clamp_min(1e-20)).item(),
        "values": actual.numel(),
    }


@contextmanager
def _bf16_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _fp8_reference(values, block=32):
    grouped = values.float().unflatten(-1, (-1, block))
    scales = (grouped.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil().exp2()
    quantized = (grouped / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.flatten(-2), scales


def _primitives():
    from resin_models_cuda.deepseek_v41.kernels import (
        act_quant,
        fp4_act_quant,
        fp8_gemm,
        hc_split_sinkhorn,
    )

    values = torch.randn(33, 128, device="cuda", dtype=torch.bfloat16)
    values[0].zero_()
    values[1].fill_(1e-6)
    values[2] = torch.linspace(-6, 6, 128, device="cuda")
    expected_q, expected_s = _fp8_reference(values)
    quantized, scales = act_quant(values, 32, "ue8m0", torch.float8_e8m0fnu)
    result = {
        "fp8_values": _check(quantized, expected_q),
        "fp8_scales": _check(scales, expected_s),
    }
    inplace = values.clone()
    act_quant(inplace, 32, "ue8m0", torch.float8_e8m0fnu, True)
    result["fp8_inplace"] = _check(
        inplace,
        (expected_q.float() * expected_s.repeat_interleave(32, -1)).bfloat16(),
    )

    weight = torch.randn(160, 128, device="cuda").to(torch.float8_e4m3fn)
    weight_scales = torch.exp2(
        torch.arange(20, device="cuda").reshape(5, 4).remainder(5).float() - 2
    ).to(torch.float8_e8m0fnu)
    with _bf16_default():
        output = fp8_gemm(
            quantized,
            scales,
            weight,
            weight_scales,
            torch.float8_e8m0fnu,
            block_size=32,
        )
    left = quantized.float() * scales.float().repeat_interleave(32, -1)
    right = weight.float() * weight_scales.float().repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    result["fp8_gemm"] = _check(
        output, left.bfloat16() @ right.bfloat16().T, atol=0.0625, rtol=0.01
    )

    values = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    values[0].zero_()
    values[1].fill_(1e-6)
    # Unit scale with exact FP4 midpoints exercises ties-to-even, both signs.
    edges = torch.tensor(
        [
            0,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5,
            6,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5,
        ],
        device="cuda",
    )
    values[2] = edges.repeat(32)
    grouped = values.float().unflatten(-1, (-1, 16))
    expected_s = (
        (grouped.abs().amax(-1).clamp_min(6 * 2**-9) / 6)
        .to(torch.float8_e4m3fn)
        .float()
    )
    normalized = (grouped / expected_s[..., None]).clamp(-6, 6)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    distances = (normalized.abs()[..., None] - levels).abs()
    nearest = distances.argmin(-1)
    ties = (distances == distances.amin(-1, keepdim=True)).sum(-1) > 1
    nearest += (ties & (nearest.remainder(2) == 1)).long()
    expected = (
        (levels[nearest] * normalized.sign() * expected_s[..., None])
        .flatten(-2)
        .bfloat16()
    )
    packed, scales = fp4_act_quant(values, 16, scale_dtype=torch.float8_e4m3fn)
    result["fp4_scales"] = _check(scales, expected_s)
    # Independently unpack the two E2M1 nibbles, including negative zero.
    raw = packed.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
    unpacked = levels[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    result["fp4_packed"] = _check(
        (unpacked * expected_s.repeat_interleave(16, -1)).bfloat16(), expected
    )
    fp4_act_quant(values, 16, True, scale_dtype=torch.float8_e4m3fn)
    result["fp4_inplace"] = _check(values, expected)

    mixes = torch.randn(2, 7, 24, device="cuda", dtype=torch.float32)
    mixes[0, 0].zero_()
    mixes[0, 1] = torch.linspace(-80, 80, 24, device="cuda")
    scale = torch.tensor([0.4, -1.5, 1.2], device="cuda")
    base = torch.linspace(-3, 3, 24, device="cuda")
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    expected = (mixes[..., 8:] * scale[2] + base[8:]).unflatten(-1, (4, 4)).softmax(
        -1
    ) + 1e-6
    expected /= expected.sum(-2, keepdim=True) + 1e-6
    for _ in range(19):
        expected /= expected.sum(-1, keepdim=True) + 1e-6
        expected /= expected.sum(-2, keepdim=True) + 1e-6
    result["hc_pre"] = _check(
        pre,
        (mixes[..., :4] * scale[0] + base[:4]).sigmoid() + 1e-6,
        atol=2e-6,
        rtol=1e-5,
    )
    result["hc_post"] = _check(
        post,
        2 * (mixes[..., 4:8] * scale[1] + base[4:8]).sigmoid(),
        atol=2e-6,
        rtol=1e-5,
    )
    result["hc_sinkhorn"] = _check(comb, expected, atol=2e-6, rtol=1e-5)
    return result


def _decode(packed):
    data = packed[..., : 512 if packed.shape[-1] == 528 else 256].contiguous()
    if packed.shape[-1] == 528:
        values = data.view(torch.float8_e4m3fn).float()
        scales = packed[..., 512:].contiguous().view(torch.float8_e8m0fnu).float()
        return (values * scales.repeat_interleave(32, -1)).bfloat16()
    codes = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    values = levels[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    scales = packed[..., 256:].contiguous().view(torch.float8_e4m3fn).float()
    return (values * scales.repeat_interleave(16, -1)).bfloat16()


def _rotate(value, positions, cos_sin, *, inverse=False, round_bf16=True):
    value = value.float().clone()
    cosine, sine = cos_sin[positions].chunk(2, -1)
    if inverse:
        sine = -sine
    pair = value[..., -64:].unflatten(-1, (32, 2))
    left, right = pair[..., 0].clone(), pair[..., 1].clone()
    pair[..., 0] = left * cosine[:, None] - right * sine[:, None]
    pair[..., 1] = left * sine[:, None] + right * cosine[:, None]
    return value.bfloat16() if round_bf16 else value


def _reference(query, keys, indices, positions, attention):
    query = _rotate(query, positions, attention.cos_sin)
    selected = keys[indices.clamp_min(0).long()].float()
    scores = (
        torch.einsum("mhd,mkd->mhk", query.float(), selected) * attention.softmax_scale
    )
    scores.masked_fill_(indices[:, None] < 0, -torch.inf)
    scores = torch.cat(
        (scores, attention.attn_sink[None, :, None].expand(query.shape[0], -1, -1)), -1
    )
    output = torch.einsum("mhk,mkd->mhd", scores.softmax(-1)[..., :-1], selected)
    # The fused epilogue applies inverse RoPE to FP32 O, then quantizes directly.
    return _rotate(output, positions, attention.cos_sin, inverse=True, round_bf16=False)


def _weight(rows, columns):
    return SimpleNamespace(
        weight=torch.randn(rows, columns, device="cuda").to(torch.float8_e4m3fn),
        scale=torch.exp2(
            torch.randint(-6, -2, (rows // 32, columns // 32), device="cuda").float()
        ).to(torch.float8_e8m0fnu),
    )


def _decoded_weight(source):
    return source.weight.float() * source.scale.float().repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)


def _metrics(actual, expected):
    difference = actual.float() - expected.float()
    return {
        "relative_l2": (
            difference.norm() / expected.float().norm().clamp_min(1e-20)
        ).item(),
        "max_abs": difference.abs().max().item(),
        "values": actual.numel(),
    }


def _normalized_check(actual, expected, *, label, limit):
    metrics = {**_metrics(actual, expected), "relative_l2_limit": limit}
    assert metrics["relative_l2"] < limit, f"{label}: {metrics}"
    return metrics


def _projection_oracles(
    query_projection, output_projection, query_source, output_source, hidden
):
    # Invert the documented Q-B row permutation independently of FlashMLA.
    query = (
        query_projection(hidden)
        .reshape(4, 32, 64, 16)
        .transpose(1, 2)
        .reshape(4, 64, 512)
    )
    quantized, scales = _fp8_reference(hidden)
    query_reference = (
        (
            (quantized.float() * scales.repeat_interleave(32, -1))
            @ _decoded_weight(query_source).T
        )
        .bfloat16()
        .reshape_as(query)
    )
    result = {
        "query_projection": _normalized_check(
            query, query_reference, label="direct_query_projection", limit=0.003
        )
    }

    # Nonuniform independent head outputs expose WV's within-group permutation,
    # without attention averaging away a head/scale addressing mistake.
    from resin_models_cuda.deepseek_v41.kernels import act_quant

    source = torch.randn(4, 64, 512, device="cuda", dtype=torch.bfloat16)
    values, scales = act_quant(source, 32, "ue8m0", torch.float8_e8m0fnu)
    decoded = values.float() * scales.float().repeat_interleave(32, -1)
    permuted = values.reshape(4, 8, 8, 16, 32).transpose(2, 3).reshape(4, 8, 4096)
    packed_scales = (
        scales.reshape(4, 8, 8, 16)
        .transpose(2, 3)
        .contiguous()
        .view(torch.int32)
        .reshape(4, 8, 32)
    )
    actual = output_projection(permuted, packed_scales)
    expected = torch.einsum(
        "mgd,grd->mgr",
        decoded.reshape(4, 8, 4096),
        _decoded_weight(output_source).reshape(8, 128, 4096),
    ).bfloat16()
    result["output_projection"] = _normalized_check(
        actual, expected, label="direct_output_projection", limit=0.003
    )
    return result


def _attention_checks():
    """Check prepared projections, both packed formats and changing graph inputs."""
    from resin.backend.cuda.slab import DevicePageGroup
    from resin_models_cuda.deepseek_v41.attention import (
        OutputProjection,
        QueryProjection,
        fused_attention,
        gather_kv,
        pack_kv,
        store_kv,
    )
    from resin_models_cuda.deepseek_v41.cache import CacheArena, CacheGeometry
    from resin_models_cuda.deepseek_v41.upstream.fused_transform import (
        CastArgs,
        KVCacheOutArgs,
        fused_norm_rope_hadamard_cast_store_kvcache,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(4101)
    geometry = CacheGeometry.build(
        SimpleNamespace(
            n_layers=2,
            n_mtp_layers=0,
            compress_ratios=(1, 1),
            kv_source_layers=(0,),
            dspark_block_size=4,
            window_size=128,
        ),
        prefill_chunk_size=256,
        max_batch_size=4,
    )
    arena = CacheArena("cuda", geometry.layout(8), geometry)
    groups = tuple(
        DevicePageGroup(
            torch.tensor([[i + 1, i + 5]], device="cuda", dtype=torch.int32),
            torch.tensor([[packing - 1, 0]], device="cuda", dtype=torch.int32),
            torch.zeros(1, device="cuda", dtype=torch.int32),
        )
        for i, packing in enumerate(geometry.packings)
    )
    cache = arena._make_graph_cache(groups)
    logical = torch.arange(256, device="cuda", dtype=torch.int32)[None]
    window_ids, compressed_ids = (
        cache.window_rows(logical),
        cache.compressed_rows(0, 1, logical),
    )
    window, compressed = cache.window_cache(0), cache.compressed_cache(0)
    inputs = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    latent = torch.randn_like(inputs) * 0.5
    packed, packed4 = pack_kv(inputs), pack_kv(latent, fp4=True)
    store_kv(window, window_ids, packed)
    store_kv(compressed, compressed_ids, packed4)
    decoded, decoded4 = _decode(packed), _decode(packed4)
    result = {
        "parent_bytes": geometry.parent_bytes,
        "packings": geometry.packings,
        "fp8_gather": _check(gather_kv(window, window_ids), decoded),
        "fp4_gather": _check(gather_kv(compressed, compressed_ids), decoded4),
    }
    # The consumer's view aliases the source without copying page tables or KV.
    assert cache.compressed_cache(1).data_ptr() == compressed.data_ptr()
    query_source, output_source = _weight(64 * 512, 128), _weight(8 * 128, 4096)
    query_projection = QueryProjection(query_source, 64, 512)
    # This oracle isolates the grouped projection before its optional FP8 cast.
    output_projection = OutputProjection(output_source, 8, 128, fp8_output=False)
    frequencies = (
        torch.arange(256, device="cuda")[:, None]
        * torch.arange(32, device="cuda")[None]
        * 0.001
    )
    attention = SimpleNamespace(
        n_local_heads=64,
        head_dim=512,
        rope_head_dim=64,
        n_local_groups=8,
        cos_sin=torch.cat((frequencies.cos(), frequencies.sin()), -1),
        attn_sink=torch.linspace(-1, 1, 64, device="cuda"),
        softmax_scale=512**-0.5,
    )
    hidden = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([127, 128, 129, 255], device="cuda", dtype=torch.int32)
    selected = torch.arange(32, device="cuda", dtype=torch.int32).expand(4, -1).clone()
    extra = torch.arange(8, device="cuda", dtype=torch.int32).expand(4, -1).clone()
    selected[:, -1], extra[:, -1] = -1, -1
    draft_values = torch.randn(4, 5, 512, device="cuda", dtype=torch.bfloat16)
    draft_ids = (
        torch.arange(4, device="cuda")[:, None] * geometry.draft_tokens
        + torch.arange(5, device="cuda")[None]
    ).int()
    draft_dense = torch.zeros(
        4 * geometry.draft_tokens, 512, device="cuda", dtype=torch.bfloat16
    )
    draft_dense[draft_ids.flatten().long()] = _decode(pack_kv(draft_values)).flatten(
        0, 1
    )

    def execute(*, prefill=False, draft=False):
        query = query_projection(hidden).reshape(4, 64, 512)
        if prefill:
            ids = torch.cat((selected, torch.where(extra >= 0, extra + 256, -1)), -1)
            output = fused_attention(
                attention,
                query,
                positions,
                torch.cat((decoded, decoded4))[:, None],
                ids,
                prefill=True,
            )
        else:
            main_ids = torch.where(
                selected >= 0, window_ids.flatten()[selected.clamp_min(0).long()], -1
            )
            extra_ids = torch.where(
                extra >= 0, compressed_ids.flatten()[extra.clamp_min(0).long()], -1
            )
            if draft:
                # These fixtures are already normalized/rotated. Exercise the
                # production fused cache writer's cast/store stages, rather
                # than the reference store_kv helper's dynamic boolean gather.
                fused_norm_rope_hadamard_cast_store_kvcache(
                    draft_values.flatten(0, 1),
                    cast_args=CastArgs("e4m3", 32),
                    out_args=KVCacheOutArgs(cache.draft_cache, draft_ids.flatten()),
                )
            output = fused_attention(
                attention,
                query,
                positions,
                window,
                main_ids,
                extra_cache=cache.draft_cache if draft else compressed,
                extra_indices=draft_ids if draft else extra_ids,
            )
        return output_projection(*output)

    result["direct_projections"] = _projection_oracles(
        query_projection, output_projection, query_source, output_source, hidden
    )

    def reference(*, draft=False):
        quantized, scales = _fp8_reference(hidden)
        q = (quantized.float() * scales.repeat_interleave(32, -1)) @ _decoded_weight(
            query_source
        ).T
        ids = torch.cat(
            (
                selected,
                draft_ids + 256 if draft else torch.where(extra >= 0, extra + 256, -1),
            ),
            -1,
        )
        values = _reference(
            q.bfloat16().reshape(4, 64, 512),
            torch.cat((decoded, draft_dense if draft else decoded4)),
            ids,
            positions.long(),
            attention,
        )
        quantized, scales = _fp8_reference(values.flatten(1))
        decoded_output = (
            quantized.float() * scales.repeat_interleave(32, -1)
        ).reshape_as(values)
        quantization_error = _metrics(decoded_output, values)
        result["draft_output_quantization" if draft else "output_quantization"] = (
            quantization_error
        )
        values = decoded_output.reshape(4, 8, 4096)
        return torch.einsum(
            "mgd,grd->mgr",
            values,
            _decoded_weight(output_source).reshape(8, 128, 4096),
        ).bfloat16()

    expected = reference()
    # Small softmax/PV differences can cross an FP8 rounding midpoint. Check
    # their aggregate effect; elementwise relative errors explode at cancellation.
    result["decode_projection"] = _normalized_check(
        execute(), expected, label="decode_projection", limit=0.02
    )
    result["prefill_projection"] = _normalized_check(
        execute(prefill=True), expected, label="prefill_projection", limit=0.02
    )
    result["draft_projection"] = _normalized_check(
        execute(draft=True), reference(draft=True), label="draft_projection", limit=0.02
    )
    result["draft_history_unchanged"] = _check(gather_kv(window, window_ids), decoded)
    result["draft_private_rows"] = _check(
        gather_kv(cache.draft_cache, draft_ids),
        _decode(pack_kv(draft_values)).flatten(0, 1),
    )
    for draft in (False, True):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                execute(draft=draft)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = execute(draft=draft)
        replay = []
        for _ in range(5):
            hidden.add_(0.01)
            positions.add_(1).remainder_(256)
            selected[:, :-1].add_(7).remainder_(256)
            extra[:, :-1].add_(11).remainder_(256)
            if draft:
                draft_values.add_(0.005)
            eager = execute(draft=draft).clone()
            if draft:
                cache.draft_cache.zero_()
            graph.replay()
            torch.cuda.synchronize()
            replay.append(_check(captured, eager))
            if draft:
                _check(
                    gather_kv(cache.draft_cache, draft_ids),
                    _decode(pack_kv(draft_values)).flatten(0, 1),
                )
        result["draft_graph_replays" if draft else "graph_replays"] = replay
        graph.reset()
    result["ok"] = True
    return result


@torch.inference_mode()
def test_native_v41_quantization_and_hc():
    torch.manual_seed(41)
    _primitives()


@torch.inference_mode()
def test_native_v41_shared_paged_attention():
    _attention_checks()
