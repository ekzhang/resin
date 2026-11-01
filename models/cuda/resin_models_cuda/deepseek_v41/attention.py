"""Pinned FlashMLA packed KV and prepared fused attention projections."""

from functools import lru_cache, partial

import torch
from torch import nn
from torch.nn import functional as F

from resin_models_cuda.deepseek_v41.kernels import act_quant, fp4_act_quant, unpack_kv
from resin_models_cuda.deepseek_v41.linear import PreparedLinear, weight_scale_rows
from resin_models_cuda.mega_moe import prepare_weights


def pack_kv(value, *, fp4=False):
    data, scales = (
        fp4_act_quant(value, 16, scale_dtype=torch.float8_e4m3fn)
        if fp4
        else act_quant(value, 32, "ue8m0", torch.float8_e8m0fnu)
    )
    return torch.cat((data.view(torch.uint8), scales.view(torch.uint8)), -1)


def store_kv(cache, indices, values):
    indices = indices.flatten().long()
    valid = indices >= 0
    indices = indices[valid]
    cache[indices // cache.shape[1], indices % cache.shape[1], 0] = values.reshape(
        -1, values.shape[-1]
    )[valid]


def gather_kv(cache, indices):
    if not indices.numel():
        return cache.new_empty(0, 512, dtype=torch.bfloat16)
    return unpack_kv(cache, indices.flatten().int())


def prefill_inputs(attention, cache, metadata, capacity, runtime):
    """Reuse gather addresses and each shared compressed source within a forward."""
    layout = runtime.setdefault("prefill_layout", {})
    if "window" not in layout:
        positions = metadata.positions
        lower = runtime.get("swa_start", 0)
        start = max(lower, capacity - positions.shape[1] - attention.window_size + 1)
        history = torch.arange(start, max(start, capacity), device=positions.device)[
            None
        ]
        columns = torch.arange(attention.window_size, device=positions.device)
        counts = (positions + 1).clamp_max(attention.window_size)
        windows = positions[..., None] + 1 - counts[..., None] + columns
        visible = (
            (columns < counts[..., None])
            & (windows >= lower)
            & metadata.valid_token_mask.reshape_as(positions)[..., None]
        )
        layout["window"] = (
            cache.window_rows(attention.layer_id, history),
            torch.where(visible, windows - start, -1),
        )
    rows, indices = layout["window"]
    values = gather_kv(cache.window_cache(attention.layer_id), rows)
    if not attention.ratio:
        return values, indices
    ratio = attention.ratio
    if ratio not in layout:
        positions = (
            torch.arange(capacity // ratio, device=metadata.positions.device)[None]
            * ratio
        )
        layout[ratio] = cache.compressed_rows(attention.layer_id, ratio, positions)
    owner = cache.geometry.compressed_offsets[attention.layer_id]
    shared = runtime.setdefault("prefill_compressed", {})
    if owner not in shared:
        shared[owner] = gather_kv(
            cache.compressed_cache(attention.layer_id), layout[ratio]
        )
    selected = runtime["indices"]
    indices = torch.cat(
        (indices, torch.where(selected >= 0, selected + values.shape[0], -1)), -1
    )
    return torch.cat((values, shared[owner])), indices


def padded_indices(indices):
    indices = indices.int()
    padding = -indices.shape[-1] % 8
    return (F.pad(indices, (0, padding), value=-1) if padding else indices).contiguous()


class QueryProjection(PreparedLinear):
    def __init__(self, source, heads, dim, index_source=None):
        super().__init__(source, *((index_source,) if index_source is not None else ()))
        from flash_mla import fused_norm_rope_attn_rope_cast as fused

        self.query_width = heads * dim
        weight, scales = fused.permute_q_b_proj(
            (self.weight[: self.query_width], self.scale[: self.query_width]),
            heads,
            dim,
        )
        if index_source is None:
            # A single prepared projection shares its checkpoint storage.
            # Own the permuted tensors rather than modifying that source.
            self.weight, self.scale = weight.detach(), scales.detach()
        else:
            # Only attention Q is permuted; the appended index Q keeps its
            # ordinary head layout and the packed scales keep their TMA stride.
            self.weight[: self.query_width].copy_(weight)
            self.scale[: self.query_width].copy_(scales)

    def forward(self, value):
        return self.project(value)[0]

    def project(self, hidden):
        output = super().forward(hidden)
        return (
            output[..., : self.query_width],
            output[..., self.query_width :] if len(self.widths) > 1 else None,
        )


class OutputProjection(nn.Module):
    """Grouped O-A projection with a fused block32 FP8 epilogue for O-B."""

    weight: torch.Tensor
    scale: torch.Tensor

    def __init__(self, source, groups, rank, *, fp8_output=True):
        super().__init__()
        from flash_mla import fused_norm_rope_attn_rope_cast as fused

        if fp8_output and groups * rank % 128:
            raise ValueError(
                "packed output scales require groups * rank divisible by 128"
            )
        width = source.weight.shape[-1]
        weight = source.weight.reshape(groups, rank, width)
        scales = weight_scale_rows(source).reshape(groups, rank, width // 32)
        weight, scales = prepare_weights(weight, scales)
        weight, scales = fused.permute_wv_proj((weight, scales), width // 512, 512)
        self.register_buffer("weight", weight.detach())
        self.register_buffer("scale", scales.detach())
        self.fp8_output = fp8_output

    def forward(self, values, scales):
        import deep_gemm

        # This layout helper copies Float32 words; keep the packed UE8M0 bits.
        scales = (
            deep_gemm.get_mn_major_tma_aligned_tensor(
                scales.transpose(0, 1).view(torch.float32)
            )
            .view(torch.int32)
            .transpose(0, 1)
        )
        rows, groups, rank = values.shape[0], *self.weight.shape[:2]
        output = values.new_empty(
            rows,
            groups,
            rank,
            dtype=torch.float8_e4m3fn if self.fp8_output else torch.bfloat16,
        )
        if self.fp8_output:
            # Scales describe the flattened groups, not separate padded groups.
            # This is DeepInfer's fp8_einsum -> ReplicatedLinear tuple boundary.
            output_scales = torch.empty_strided(
                (rows, groups * rank // 128),
                (1, deep_gemm.get_tma_aligned_size(rows, 4)),
                dtype=torch.int32,
                device=values.device,
            )
            output = (output, output_scales)
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (values, scales),
            (self.weight, self.scale),
            output,
            recipe=(1, 1, 32),
        )
        return output


def finish_output(attention, context):
    """Keep grouped values and packed scales paired until the final GEMM."""
    output = attention.wo_a(*context)
    value = (
        (output[0].flatten(1), output[1])
        if isinstance(output, tuple)
        else output.flatten(1)
    )
    return attention.wo_b(value)


@lru_cache(2)
def _cos_sin(freqs):
    return torch.cat((freqs.real, freqs.imag), -1).contiguous()


def prepare_attention(attention):
    from resin_models_cuda.deepseek_v41.compressor import PreparedCompressor

    # DeepInfer merges these projections before lowering so their input cast
    # and GEMM are shared. The separate KV projection is retained for DFlash's
    # store-only pass, which must not compute Q.
    attention.qkv_projection = PreparedLinear(attention.wq_a, attention.wkv)
    attention.project_input = partial(project_input, attention)
    attention.wq_b = QueryProjection(
        attention.wq_b,
        attention.n_local_heads,
        attention.head_dim,
        attention.indexer.wq_b if hasattr(attention, "indexer") else None,
    )
    attention.project_query = attention.wq_b.project
    if hasattr(attention, "compressor"):
        attention.compressor = PreparedCompressor(attention.compressor)
    attention.wo_a = OutputProjection(
        attention.wo_a, attention.n_local_groups, attention.o_lora_rank
    )
    attention.register_buffer(
        "cos_sin",
        _cos_sin(attention.freqs_cis),
        persistent=False,
    )


def project_input(attention, hidden):
    query, kv = attention.qkv_projection(hidden).split(
        (attention.q_lora_rank, attention.head_dim), dim=-1
    )
    return attention.q_norm(query), kv


def store_attention_kv(attention, value, positions, cache, slots, *, compressed=False):
    """DeepInfer's one-pass RMSNorm/RoPE/quantization/paged store."""
    from resin_models_cuda.deepseek_v41.upstream.fused_transform import (
        CastArgs,
        KVCacheOutArgs,
        NormArgs,
        RopeArgs,
        fused_norm_rope_hadamard_cast_store_kvcache,
    )

    fused_norm_rope_hadamard_cast_store_kvcache(
        value.reshape(-1, attention.head_dim),
        norm_args=None
        if compressed
        else NormArgs(attention.kv_norm.weight, attention.kv_norm.eps),
        rope_args=RopeArgs(
            positions.flatten().int(),
            attention.cos_sin,
            attention.rope_head_dim,
            is_neox=False,
        ),
        cast_args=CastArgs(
            "e2m1" if compressed else "e4m3",
            16 if compressed else 32,
            round_sf=not compressed,
            sf_format="e4m3" if compressed else "ue8m0",
        ),
        out_args=KVCacheOutArgs(cache, slots.flatten().int()),
    )


def fused_attention(
    attention,
    query,
    positions,
    cache,
    indices,
    *,
    extra_cache=None,
    extra_indices=None,
    prefill=False,
):
    from flash_mla import fused_norm_rope_attn_rope_cast as fused

    common = (
        False,
        0.0,
        positions.flatten().int(),
        False,
        attention.rope_head_dim,
        attention.cos_sin,
        attention.n_local_groups,
        32,
        True,
        True,
        True,
    )
    query = query.reshape(-1, attention.n_local_heads, attention.head_dim)
    indices = padded_indices(indices.reshape(-1, indices.shape[-1]))
    if prefill:
        values, scales, _, _ = fused.prefill(
            *common,
            q=query,
            kv=cache,
            indices=indices[:, None],
            sm_scale=attention.softmax_scale,
            attn_sink=attention.attn_sink,
        )
    else:
        values, scales, _ = fused.decode(
            *common,
            q=query,
            k_cache=cache,
            indices_in_kvcache=indices,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=None
            if extra_indices is None
            else padded_indices(extra_indices.reshape(indices.shape[0], -1)),
            sm_scale=attention.softmax_scale,
            attn_sink=attention.attn_sink,
        )
    return values, scales
