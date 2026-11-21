"""Fused target preparation, optionally producing projected and padded Q/bias."""

from typing import Any, cast

import torch
import triton
import triton.language as tl

from resin_models_cuda.inkling2.fa4_bias import packed_decode_group
from resin_models_cuda.inkling2.triton import (
    _convolution_input,
    _convolution_value,
    _kv_location,
)


@triton.jit
def _clip_value(value, threshold: tl.constexpr):
    return value * tl.minimum(
        threshold / (tl.sqrt(tl.sum(value * value, 0)) + 1e-6), 1.0
    )


# Flat scalar arguments avoid Triton's recursive specialization of tuple strides.
@triton.jit(
    do_not_specialize=(
        "T layer WIDTH RP0 RP1 US QB QT QD RB RT RD KB KT KD VB VT VD "
        "KSB KST KSD VSB VST VSD KW0 KW1 VW0 VW1 "
        "KDB KDT KDH KDD VDB VDT VDH VDD QSH QSD RSH RSD "
        "TB TT PB PT GSB GSC GOB GOC GFB CKP CKT CKH CKD CVP CVT CVH CVD "
        "KNB KNT KND VNB VNT VND KIB KIT KID VIB VIT VID"
    ).split()
)
def _attention_prologue(
    Q,
    R,
    K,
    V,
    KState,
    VState,
    KW,
    VW,
    KDynamic,
    VDynamic,
    QScale,
    RScale,
    Tau,
    QOut,
    ROut,
    RProj,
    Used,
    KNext,
    VNext,
    KInputs,
    VInputs,
    Positions,
    Slabs,
    Slots,
    First,
    CacheK,
    CacheV,
    T,
    layer,
    WIDTH,
    RP0,
    RP1,
    US,
    QB,
    QT,
    QD,
    RB,
    RT,
    RD,
    KB,
    KT,
    KD,
    VB,
    VT,
    VD,
    KSB,
    KST,
    KSD,
    VSB,
    VST,
    VSD,
    KW0,
    KW1,
    VW0,
    VW1,
    KDB,
    KDT,
    KDH,
    KDD,
    VDB,
    VDT,
    VDH,
    VDD,
    QSH,
    QSD,
    RSH,
    RSD,
    TB,
    TT,
    PB,
    PT,
    GSB,
    GSC,
    GOB,
    GOC,
    GFB,
    CKP,
    CKT,
    CKH,
    CKD,
    CVP,
    CVT,
    CVH,
    CVD,
    KNB,
    KNT,
    KND,
    VNB,
    VNT,
    VND,
    KIB,
    KIT,
    KID,
    VIB,
    VIT,
    VID,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEADS_PER_KV: tl.constexpr,
    PADDED_GROUP: tl.constexpr,
    D: tl.constexpr,
    R_DIM: tl.constexpr,
    EXTENT: tl.constexpr,
    OUT_EXTENT: tl.constexpr,
    PRESHEAR: tl.constexpr,
    LOCAL: tl.constexpr,
    OUT_HEADS: tl.constexpr,
    OUT_HEADS_P: tl.constexpr,
    REL_WORKERS: tl.constexpr,
    PROJECT: tl.constexpr,
    QK_LIMIT: tl.constexpr,
    R_LIMIT: tl.constexpr,
    EPS: tl.constexpr,
    DYNAMIC: tl.constexpr,
    HAS_TAU: tl.constexpr,
    PAGE: tl.constexpr,
    UNITS: tl.constexpr,
    LAYERS: tl.constexpr,
    LIVE_STATE: tl.constexpr,
    SAVE_INPUTS: tl.constexpr,
):
    QS = QB, QT, QD
    KS = KB, KT, KD
    VS = VB, VT, VD
    KSS = KSB, KST, KSD
    VSS = VSB, VST, VSD
    KWS = KW0, KW1
    VWS = VW0, VW1
    KDS = KDB, KDT, KDH, KDD
    VDS = VDB, VDT, VDH, VDD
    QSS = QSH, QSD
    TS = TB, TT
    PS = PB, PT
    GS = GSB, GSC, GOB, GOC, GFB
    CKS = CKP, CKT, CKH, CKD
    CVS = CVP, CVT, CVH, CVD
    row, head = tl.program_id(0).to(tl.int64), tl.program_id(1)
    batch, token = row // T, row % T
    cols = tl.arange(0, D)
    if head < OUT_HEADS:
        source_head = (
            head // PADDED_GROUP * HEADS_PER_KV + head % PADDED_GROUP
            if PROJECT
            else head
        )
        valid = head % PADDED_GROUP < HEADS_PER_KV if PROJECT else True
        q = tl.load(
            Q + batch * QS[0] + token * QS[1] + (source_head * D + cols) * QS[2],
            valid,
            0,
        ).to(tl.float32)
        scale = tl.load(QScale + source_head * QSS[0] + cols * QSS[1], valid, 0)
        q = q * scale.to(tl.float32)
        q = _clip_value(q, QK_LIMIT).to(Q.dtype.element_ty)
        if HAS_TAU:
            tau = tl.load(Tau + batch * TS[0] + token * TS[1]).to(tl.float32)
            q = q.to(tl.float32) * tau
        tl.store(QOut + (row * OUT_HEADS + head) * D + cols, q)
        if not PROJECT:
            rel_cols = tl.arange(0, R_DIM)
            r = tl.load(R + batch * RB + token * RT + (head * R_DIM + rel_cols) * RD)
            r = r.to(tl.float32) * tl.load(RScale + head * RSH + rel_cols * RSD).to(
                tl.float32
            )
            r = _clip_value(r, R_LIMIT).to(R.dtype.element_ty)
            if HAS_TAU:
                r = r.to(tl.float32) * tau
            tl.store(ROut + (row * HEADS + head) * R_DIM + rel_cols, r)
    elif head < OUT_HEADS + KV_HEADS:
        kv_head = head - OUT_HEADS
        channels = kv_head * D + cols
        if SAVE_INPUTS:
            raw_k = tl.load(K + batch * KB + token * KT + channels * KD)
            raw_v = tl.load(V + batch * VB + token * VT + channels * VD)
            tl.store(KInputs + batch * KIB + token * KIT + channels * KID, raw_k)
            tl.store(VInputs + batch * VIB + token * VIT + channels * VID, raw_v)
        k = (
            _convolution_value(
                K,
                KState,
                KW,
                KDynamic,
                batch,
                token,
                channels,
                cols < D,
                KS,
                KSS,
                KWS,
                KDS,
                D,
                DYNAMIC,
                D,
            )
            .to(K.dtype.element_ty)
            .to(tl.float32)
        )
        v = (
            _convolution_value(
                V,
                VState,
                VW,
                VDynamic,
                batch,
                token,
                channels,
                cols < D,
                VS,
                VSS,
                VWS,
                VDS,
                D,
                DYNAMIC,
                D,
            )
            .to(V.dtype.element_ty)
            .to(tl.float32)
        )
        k = _clip_value(k, QK_LIMIT).to(K.dtype.element_ty)
        v = (v * tl.rsqrt(tl.sum(v * v, 0) / D + EPS)).to(V.dtype.element_ty)
        page, offset = _kv_location(
            Positions,
            Slabs,
            Slots,
            First,
            batch,
            token,
            PS,
            GS,
            WIDTH,
            layer,
            PAGE,
            UNITS,
            LAYERS,
            LIVE_STATE,
        )
        tl.store(
            CacheK + page * CKS[0] + offset * CKS[1] + kv_head * CKS[2] + cols * CKS[3],
            k,
        )
        tl.store(
            CacheV + page * CVS[0] + offset * CVS[1] + kv_head * CVS[2] + cols * CVS[3],
            v,
        )
        if token == 0:
            for index in tl.static_range(3):
                next_k = _convolution_input(
                    K, KState, batch, T - 3 + index, channels, cols < D, KS, KSS
                )
                next_v = _convolution_input(
                    V, VState, batch, T - 3 + index, channels, cols < D, VS, VSS
                )
                tl.store(KNext + batch * KNB + index * KNT + channels * KND, next_k)
                tl.store(VNext + batch * VNB + index * VNT + channels * VND, next_v)

    elif PROJECT:
        # Each worker projects every padded query head for one range of bias
        # columns. The K=16 tensor-core dot retains the BF16 clip/tau boundary.
        padded_head = tl.arange(0, OUT_HEADS_P)
        source_head = (
            padded_head // PADDED_GROUP * HEADS_PER_KV + padded_head % PADDED_GROUP
        )
        valid = (padded_head < OUT_HEADS) & (padded_head % PADDED_GROUP < HEADS_PER_KV)
        rel_cols = tl.arange(0, R_DIM)
        r = tl.load(
            R
            + batch * RB
            + token * RT
            + (source_head[:, None] * R_DIM + rel_cols[None, :]) * RD,
            valid[:, None],
            0,
        ).to(tl.float32)
        scale = tl.load(
            RScale + source_head[:, None] * RSH + rel_cols[None, :] * RSD,
            valid[:, None],
            0,
        ).to(tl.float32)
        r = r * scale
        norm = tl.sqrt(tl.sum(r * r, 1))
        r = r * tl.minimum(R_LIMIT / (norm + 1e-6), 1.0)[:, None]
        r = r.to(R.dtype.element_ty)
        if HAS_TAU:
            tau = tl.load(Tau + batch * TB + token * TT).to(tl.float32)
            r = (r.to(tl.float32) * tau).to(R.dtype.element_ty)
        worker = head - OUT_HEADS - KV_HEADS
        if PRESHEAR:
            used = tl.load(Used + batch * US)
            n_right = token + 1 + used - T
            values = tl.minimum(n_right, EXTENT)
            block_max = tl.minimum(
                tl.cdiv(used, 128), tl.cdiv(128 // PADDED_GROUP + used - T, 128)
            )
            right = tl.where(
                values > 0,
                tl.maximum(OUT_EXTENT // 128 - (block_max - (n_right - 1) // 128), 0)
                * 128
                + (n_right - 1) % 128
                + 1,
                0,
            )
            left = tl.maximum(right - values, 0)
        STEPS: tl.constexpr = tl.cdiv(OUT_EXTENT, 128 * REL_WORKERS)
        for step in tl.static_range(STEPS):
            ext = (worker * STEPS + step) * 128 + tl.arange(0, 128)
            if PRESHEAR:
                distance = right - 1 - ext
                valid_column = (
                    (ext >= left)
                    & (ext < right)
                    & (distance >= 0)
                    & (distance < EXTENT)
                )
            else:
                distance, valid_column = ext, ext < EXTENT
            projection = tl.load(
                RProj + rel_cols[:, None] * RP0 + distance[None, :] * RP1,
                valid_column[None, :],
                0,
            )
            projected = tl.dot(r, projection, out_dtype=tl.float32)
            if PRESHEAR:
                fill = tl.where(
                    ext >= right, -float("inf"), -float("inf") if LOCAL else 0.0
                )
                projected = tl.where(
                    ((ext < left) | (ext >= right))[None, :], fill[None, :], projected
                )
            tl.store(
                ROut
                + (row * OUT_HEADS + padded_head[:, None]) * OUT_EXTENT
                + ext[None, :],
                projected,
                (padded_head < OUT_HEADS)[:, None] & (ext < OUT_EXTENT)[None, :],
            )


def attention_prologue(
    q,
    k_input,
    v_input,
    relative,
    k_state,
    v_state,
    k_weight,
    v_weight,
    *,
    q_scale,
    rel_scale,
    rel_proj=None,
    seqused_k=None,
    local=False,
    qk_threshold,
    rel_threshold,
    rms_eps,
    cache,
    positions,
    group_index,
    layer,
    layer_count,
    k_dynamic=None,
    v_dynamic=None,
    tau=None,
    k_state_out=None,
    v_state_out=None,
    k_inputs_out=None,
    v_inputs_out=None,
):
    """Return clipped Q/R and KV tails, or padded Q/projected bias with rel_proj.

    Static convolution weights are [KV*D,4]; dynamic weights are [4,KV*D]
    plus per-token coefficients [B,T,KV,4]. K/V round to their input dtype
    after convolution and after normalization before the cache's storage cast.
    Clipped Q/R round before applying optional [B,T] long-context scaling.
    """
    batch, tokens, width = q.shape
    if rel_proj is not None and (
        q.dtype != torch.bfloat16 or rel_proj.dtype != torch.bfloat16
    ):
        raise ValueError("projected prologue requires native BF16 inputs/weights")
    heads, dim = q_scale.shape
    rel_dim, kv_heads = rel_scale.shape[-1], cache.keys.shape[-2]
    kv_width = kv_heads * dim
    heads_per_kv = heads // kv_heads
    padded_group = triton.next_power_of_2(heads_per_kv)
    out_heads = kv_heads * padded_group if rel_proj is not None else heads
    extent = rel_proj.shape[-1] if rel_proj is not None else rel_dim
    if seqused_k is not None and (
        rel_proj is None or extent != 1024 or seqused_k.shape != (batch,)
    ):
        raise ValueError(
            "presheared producer requires 1024 projected bias values and retained lengths"
        )
    out_extent = 1280 if seqused_k is not None else extent
    workers = 5 if seqused_k is not None else 4
    if rel_proj is not None and (
        packed_decode_group(heads, kv_heads) is None
        or rel_proj.shape[0] != 16
        or extent % 128
    ):
        raise ValueError(
            "projected prologue requires a supported packed head group and a 128-aligned extent"
        )
    if (
        width != heads * dim
        or dim != 128
        or rel_dim != 16
        or relative.shape != (batch, tokens, heads * rel_dim)
        or k_input.shape != (batch, tokens, kv_width)
        or v_input.shape != k_input.shape
        or k_state.shape != (batch, 3, kv_width)
        or v_state.shape != k_state.shape
        or positions.shape != (batch, tokens)
        or cache.keys.shape[-1] != dim
        or rel_scale.shape != (heads, rel_dim)
        or (k_dynamic is None) != (v_dynamic is None)
    ):
        raise ValueError("fused Inkling target prologue geometry is invalid")
    dynamic = k_dynamic is not None
    weight_shape = (4, kv_width) if dynamic else (kv_width, 4)
    if k_weight.shape != weight_shape or v_weight.shape != weight_shape:
        raise ValueError("fused Inkling convolution weight geometry is invalid")
    if k_dynamic is not None:
        assert v_dynamic is not None
        if (
            k_dynamic.shape != (batch, tokens, kv_heads, 4)
            or v_dynamic.shape != k_dynamic.shape
        ):
            raise ValueError("fused Inkling dynamic coefficient geometry is invalid")
    if tau is not None and tau.shape != (batch, tokens):
        raise ValueError("fused Inkling attention scaling geometry is invalid")
    for destination, source in (
        (k_state_out, k_state),
        (v_state_out, v_state),
        (k_inputs_out, k_input),
        (v_inputs_out, v_input),
    ):
        if destination is not None and destination.shape != source.shape:
            raise ValueError(
                "fused Inkling convolution destination has the wrong shape"
            )
    if (k_inputs_out is None) != (v_inputs_out is None):
        raise ValueError("fused Inkling rollback requires both K/V destinations")
    q_out = torch.empty((batch, tokens, out_heads, dim), dtype=q.dtype, device=q.device)
    r_out = torch.empty(
        (batch * tokens, out_heads, out_extent)
        if rel_proj is not None
        else (batch, tokens, heads, rel_dim),
        dtype=relative.dtype,
        device=relative.device,
    )
    k_next = (
        torch.empty_like(k_state, memory_format=torch.contiguous_format)
        if k_state_out is None
        else k_state_out
    )
    v_next = (
        torch.empty_like(v_state, memory_format=torch.contiguous_format)
        if v_state_out is None
        else v_state_out
    )
    if not batch or not tokens:
        k_next.copy_(k_state)
        v_next.copy_(v_state)
        return q_out, r_out, k_next, v_next
    group = cache.groups[group_index]
    cast(Any, _attention_prologue)[
        (
            batch * tokens,
            out_heads + kv_heads + (workers if rel_proj is not None else 0),
        )
    ](
        q,
        relative,
        k_input,
        v_input,
        k_state,
        v_state,
        k_weight,
        v_weight,
        k_dynamic,
        v_dynamic,
        q_scale,
        rel_scale,
        tau,
        q_out,
        r_out,
        rel_proj,
        seqused_k,
        k_next,
        v_next,
        k_inputs_out,
        v_inputs_out,
        positions,
        group.slab_ids,
        group.slotnums,
        group.first_blocks,
        cache.keys,
        cache.values,
        tokens,
        layer,
        group.slab_ids.shape[1],
        *(rel_proj.stride() if rel_proj is not None else (0, 0)),
        seqused_k.stride(0) if seqused_k is not None else 0,
        *q.stride(),
        *relative.stride(),
        *k_input.stride(),
        *v_input.stride(),
        *k_state.stride(),
        *v_state.stride(),
        *k_weight.stride(),
        *v_weight.stride(),
        *(k_dynamic.stride() if k_dynamic is not None else (0, 0, 0, 0)),
        *(v_dynamic.stride() if v_dynamic is not None else (0, 0, 0, 0)),
        *q_scale.stride(),
        *rel_scale.stride(),
        *(tau.stride() if tau is not None else (0, 0)),
        *positions.stride(),
        *group.slab_ids.stride(),
        *group.slotnums.stride(),
        group.first_blocks.stride(0),
        *cache.keys.stride(),
        *cache.values.stride(),
        *k_next.stride(),
        *v_next.stride(),
        *(k_inputs_out.stride() if k_inputs_out is not None else (0, 0, 0)),
        *(v_inputs_out.stride() if v_inputs_out is not None else (0, 0, 0)),
        heads,
        kv_heads,
        heads_per_kv,
        padded_group,
        dim,
        rel_dim,
        extent,
        out_extent,
        seqused_k is not None,
        local,
        out_heads,
        max(16, triton.next_power_of_2(out_heads)),
        workers,
        rel_proj is not None,
        qk_threshold,
        rel_threshold,
        rms_eps,
        dynamic,
        tau is not None,
        cache.page_tokens,
        cache.slab_units,
        layer_count,
        group.live_state,
        k_inputs_out is not None,
        num_warps=4,
        # Preserve the standalone clip/norm arithmetic. Convolution uses
        # explicit fma and a separate residual addition in the shared helper.
        enable_fp_fusion=True,
    )
    return q_out, r_out, k_next, v_next
