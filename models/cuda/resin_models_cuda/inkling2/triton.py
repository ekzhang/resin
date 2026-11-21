"""Fused Inkling 2 normalization, residual attention, convolution, and gating."""

from functools import cache
from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda, libdevice


@triton.jit
def _kv_location(
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
    PAGE: tl.constexpr,
    UNITS: tl.constexpr,
    LAYERS: tl.constexpr,
    LIVE_STATE: tl.constexpr,
):
    position = tl.load(Positions + batch * PS[0] + token * PS[1]).to(tl.int64)
    column = 0 if LIVE_STATE else position // PAGE - tl.load(First + batch * GS[4])
    valid = (column >= 0) & (column < WIDTH) & (position >= 0)
    slab = tl.load(Slabs + batch * GS[0] + column * GS[1], valid, 0).to(tl.int64)
    slot = tl.load(Slots + batch * GS[2] + column * GS[3], valid, 0).to(tl.int64)
    page = slab * UNITS + slot * LAYERS + layer
    offset = 0 if LIVE_STATE else tl.where(valid, position % PAGE, 0)
    return page, offset


# Flatten request-dependent strides: Triton specializes tuple elements recursively.
# Inner strides describe fixed tensor layouts and retain vectorized loads/stores.
@triton.jit(
    do_not_specialize=("T layer WIDTH KB VB PB GSB GOB").split(),
    do_not_specialize_on_alignment=["Positions", "Slabs", "Slots", "First"],
)
def _store_kv(
    K,
    V,
    Positions,
    Slabs,
    Slots,
    First,
    DestK,
    DestV,
    T,
    layer,
    KB,
    KT,
    KH,
    KD,
    VB,
    VT,
    VH,
    VD,
    CKP,
    CKT,
    CKH,
    CKD,
    CVP,
    CVT,
    CVH,
    CVD,
    PB,
    PT,
    GSB,
    GSC,
    GOB,
    GOC,
    GFB,
    WIDTH,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    UNITS: tl.constexpr,
    LAYERS: tl.constexpr,
    LIVE_STATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch, token = row // T, row % T
    page, offset = _kv_location(
        Positions,
        Slabs,
        Slots,
        First,
        batch,
        token,
        (PB, PT),
        (GSB, GSC, GOB, GOC, GFB),
        WIDTH,
        layer,
        PAGE,
        UNITS,
        LAYERS,
        LIVE_STATE,
    )
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    head, dim = col // DIM, col % DIM
    mask = col < HEADS * DIM
    key = tl.load(K + batch * KB + token * KT + head * KH + dim * KD, mask, 0)
    value = tl.load(V + batch * VB + token * VT + head * VH + dim * VD, mask, 0)
    tl.store(
        DestK + page * CKP + offset * CKT + head * CKH + dim * CKD,
        key,
        mask,
    )
    tl.store(
        DestV + page * CVP + offset * CVT + head * CVH + dim * CVD,
        value,
        mask,
    )


def store_kv(cache, keys, values, positions, group_index, layer, layer_count):
    group = cache.groups[group_index]
    batch, tokens = positions.shape
    heads, dim = cache.keys.shape[-2:]
    if keys.shape != (batch, tokens, heads, dim) or values.shape != keys.shape:
        raise ValueError("KV values do not match the cache and position geometry")
    if not batch or not tokens:
        return
    cast(Any, _store_kv)[(batch * tokens, triton.cdiv(heads * dim, 256))](
        keys,
        values,
        positions,
        group.slab_ids,
        group.slotnums,
        group.first_blocks,
        cache.keys,
        cache.values,
        tokens,
        layer,
        *keys.stride(),
        *values.stride(),
        *cache.keys.stride(),
        *cache.values.stride(),
        *positions.stride(),
        *group.slab_ids.stride(),
        *group.slotnums.stride(),
        group.first_blocks.stride(0),
        group.slab_ids.shape[1],
        heads,
        dim,
        cache.page_tokens,
        cache.slab_units,
        layer_count,
        group.live_state,
        256,
    )


@triton.jit
def _norm_row(
    X,
    W,
    Y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + cols, cols < D, 0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    if HAS_WEIGHT:
        y *= tl.load(W + cols, cols < D, 0).to(tl.float32)
    tl.store(Y + cols, y, cols < D)


@triton.jit(do_not_specialize=["stride"])
def _norm(
    X,
    W,
    Y,
    stride,
    D: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    _norm_row(X + row * stride, W, Y + row * D, D, EPS, HAS_WEIGHT, BLOCK)


def rms_norm(x, weight, eps, dtype):
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    out = torch.empty(shape, dtype=dtype, device=x.device)
    cast(Any, _norm)[(x.shape[0],)](
        x,
        weight,
        out,
        x.stride(0),
        x.shape[-1],
        eps,
        weight is not None,
        triton.next_power_of_2(x.shape[-1]),
    )
    return out


@triton.jit(do_not_specialize=["rows"])
def _attention_input_norm(
    Inputs, Weights, Y, rows, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row, stream = tl.program_id(0), tl.program_id(1)
    for index in tl.static_range(len(Inputs)):
        if stream == index:
            _norm_row(
                Inputs[index] + row * D,
                Weights[index],
                Y + (stream * rows + row) * D,
                D,
                EPS,
                True,
                BLOCK,
            )


def attention_input_norm(streams, weights, eps):
    """Normalize each source into adjacent BF16 rows for attention projections."""
    batch, tokens, width = streams[0].shape
    rows = batch * tokens
    output = streams[0].new_empty(
        (len(streams), batch, tokens, width), dtype=torch.bfloat16
    )
    cast(Any, _attention_input_norm)[(rows, len(streams))](
        tuple(stream.contiguous() for stream in streams),
        weights,
        output,
        rows,
        width,
        eps,
        triton.next_power_of_2(width),
    )
    return output


# Stable selection and the top-eight sum order follow SGLang e62ac627,
# kernels/ops/inkling2/route.py and ops/moe/gate_topk.py (Apache-2.0).
@triton.jit
def _route_pick(values, slots, index: tl.constexpr):
    return tl.sum(tl.where(slots == index, values, 0.0), 0)


@triton.jit
def _route_experts(
    X,
    Bias,
    Ids,
    Weights,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP: tl.constexpr,
):
    row, cols = tl.program_id(0), tl.arange(0, BLOCK)
    logits = tl.load(X + row * N + cols, cols < N, 0)
    selection = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
    selection += tl.load(Bias + cols, cols < N, 0)
    selection = tl.where(cols < N, selection, float("-inf"))
    bits = selection.to(tl.uint32, bitcast=True)
    ordered = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)
    keys = (ordered.to(tl.uint64) << 16) | (BLOCK - cols).to(tl.uint64)
    top = tl.sort(tl.topk(keys[None, :], TOP, dim=1), dim=1, descending=True)
    slots = tl.arange(0, TOP)
    ids = BLOCK - (tl.reshape(top, (TOP,)) & 0xFFFF).to(tl.int32)
    chosen = tl.load(X + row * N + ids, slots < K, 0)
    log_probs = tl.minimum(chosen, 0.0) - libdevice.log1p(
        libdevice.exp(-tl.abs(chosen))
    )
    log_probs = tl.where(slots < K, log_probs, float("-inf"))
    maximum = tl.max(log_probs, 0)
    shifted = libdevice.exp(log_probs - maximum)
    if K == 8:
        total = (
            (_route_pick(shifted, slots, 0) + _route_pick(shifted, slots, 2))
            + (_route_pick(shifted, slots, 1) + _route_pick(shifted, slots, 3))
        ) + (
            (_route_pick(shifted, slots, 4) + _route_pick(shifted, slots, 6))
            + (_route_pick(shifted, slots, 5) + _route_pick(shifted, slots, 7))
        )
    else:
        total = tl.sum(shifted, 0)
    weights = libdevice.exp(log_probs - (maximum + libdevice.log(total))) * SCALE
    tl.store(Ids + row * K + slots, ids, slots < K)
    tl.store(Weights + row * K + slots, weights, slots < K)


def route_experts(logits, bias, top_k, scale):
    rows, experts = logits.shape
    ids = torch.empty((rows, top_k), device=logits.device, dtype=torch.int64)
    weights = torch.empty((rows, top_k), device=logits.device, dtype=torch.float32)
    if rows:
        cast(Any, _route_experts)[(rows,)](
            logits.contiguous(),
            bias,
            ids,
            weights,
            experts,
            top_k,
            scale,
            triton.next_power_of_2(experts),
            triton.next_power_of_2(top_k),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return ids, weights


@triton.jit(do_not_specialize=["SX", "SH", "SD"])
def _clip(
    X,
    S,
    Tau,
    Y,
    SX,
    SH,
    SD,
    D: tl.constexpr,
    H: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_TAU: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    value = tl.load(X + (row // H) * SX + (row % H) * SH + cols * SD, cols < D, 0).to(
        tl.float32
    )
    if HAS_SCALE:
        value *= tl.load(S + (row % H) * D + cols, cols < D, 0).to(tl.float32)
    value *= tl.minimum(LIMIT / (tl.sqrt(tl.sum(value * value, 0)) + 1e-6), 1.0)
    value = value.to(X.dtype.element_ty)
    if HAS_TAU:
        value = value.to(tl.float32) * tl.load(Tau + row // H)
    tl.store(Y + row * D + cols, value, cols < D)


def l2_clip(x, scale, threshold, tau):
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    heads, dim = x.shape[-2:]
    x = x.reshape(-1, heads, dim)
    cast(Any, _clip)[(x.numel() // dim,)](
        x,
        scale,
        tau,
        out,
        *x.stride(),
        dim,
        heads,
        scale is not None,
        tau is not None,
        threshold,
        triton.next_power_of_2(dim),
    )
    return out


@triton.jit
def _convolution_input(X, State, batch, source, col, mask, XS, SS):
    live = tl.load(
        X + batch * XS[0] + source * XS[1] + col * XS[2], mask & (source >= 0), 0
    )
    past = tl.load(
        State + batch * SS[0] + (source + 3) * SS[1] + col * SS[2],
        mask & (source < 0),
        0,
    )
    return tl.where(source >= 0, live, past)


@triton.jit
def _convolution_value(
    X,
    State,
    W,
    Dynamic,
    batch,
    token,
    col,
    mask,
    XS,
    SS,
    WS,
    DS,
    HEAD_DIM: tl.constexpr,
    DYNAMIC: tl.constexpr,
    BLOCK: tl.constexpr,
    INPUT_DTYPE: tl.constexpr = None,  # ty: ignore[invalid-parameter-default]
):
    dtype: tl.constexpr = X.dtype.element_ty if INPUT_DTYPE is None else INPUT_DTYPE
    current = (
        tl.load(X + batch * XS[0] + token * XS[1] + col * XS[2], mask, 0)
        .to(dtype)
        .to(tl.float32)
    )
    acc = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(4):
        lag = tap if DYNAMIC else 3 - tap
        value = (
            _convolution_input(X, State, batch, token - lag, col, mask, XS, SS)
            .to(dtype)
            .to(tl.float32)
        )
        if DYNAMIC:
            coefficient = tl.load(W + tap * WS[0] + col * WS[1], mask, 0).to(tl.float32)
            coefficient += tl.load(
                Dynamic
                + batch * DS[0]
                + token * DS[1]
                + (col // HEAD_DIM) * DS[2]
                + tap * DS[3],
                mask,
                0,
            ).to(tl.float32)
        else:
            # Static checkpoint weights are stored in FP32 but consumed in the
            # input dtype, matching SGLang's prefill and fused decode prologues.
            coefficient = (
                tl.load(W + col * WS[0] + tap * WS[1], mask, 0).to(dtype).to(tl.float32)
            )
        if DYNAMIC and tap == 0:
            acc = value * coefficient
        else:
            acc = tl.fma(value, coefficient, acc)
    return (acc + current) * (1.0 / 3.0 if DYNAMIC else 0.4472135954999579)


@triton.jit(
    do_not_specialize="T XB XT XD SB ST SD NB NT ND IB IT ID AB AT AD PB PT PD".split()
)
def _conv(
    X,
    State,
    W,
    Dynamic,
    Attention,
    Partial,
    Y,
    Next,
    Inputs,
    T,
    D: tl.constexpr,
    H: tl.constexpr,
    XB,
    XT,
    XD,
    SB,
    ST,
    SD,
    NB,
    NT,
    ND,
    IB,
    IT,
    ID,
    AB,
    AT,
    AD,
    PB,
    PT,
    PD,
    BLOCK: tl.constexpr,
):
    FUSED: tl.constexpr = Attention is not None  # ty: ignore[invalid-assignment]
    DYNAMIC: tl.constexpr = Dynamic is not None  # ty: ignore[invalid-assignment]
    row = tl.program_id(0) if FUSED else tl.program_id(0).to(tl.int64)
    batch, token = row // T, row % T
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < D
    XS = (XB, XT, XD) if FUSED else (XB, XT, 1)
    value = _convolution_value(
        X,
        State,
        W,
        Dynamic,
        batch,
        token,
        col,
        mask,
        XS,
        (SB, ST, SD),
        (D if DYNAMIC else 4, 1),
        (T * H * 4, H * 4, 4, 1),
        D // H,
        DYNAMIC,
        BLOCK,
        tl.bfloat16 if FUSED else None,
    )
    if FUSED:
        value = value.to(tl.bfloat16).to(tl.float32)
        total = tl.full((BLOCK,), 0, tl.float32)
        if Partial is not None:
            total += tl.load(Partial + batch * PB + token * PT + col * PD, mask, 0).to(
                tl.float32
            )
        total += tl.load(Attention + batch * AB + token * AT + col * AD, mask, 0).to(
            tl.float32
        )
        value = total + value
    tl.store(Y + row * D + col, value, mask)
    if Inputs is not None:
        current = tl.load(X + batch * XB + token * XT + col * XS[2], mask, 0)
        if FUSED:
            current = current.to(tl.bfloat16)
        tl.store(Inputs + batch * IB + token * IT + col * ID, current, mask)
    if token == 0:
        for index in tl.static_range(3):
            tail = _convolution_input(
                X, State, batch, T - 3 + index, col, mask, XS, (SB, ST, SD)
            )
            if FUSED:
                tail = tail.to(tl.bfloat16)
            tl.store(Next + batch * NB + index * NT + col * ND, tail, mask)


def short_convolution(
    x,
    state,
    weight,
    dynamic,
    *,
    state_out=None,
    inputs_out=None,
    attention=None,
    partial=None,
):
    if state_out is not None and state_out.shape != state.shape:
        raise ValueError("convolution state destination has the wrong shape")
    if inputs_out is not None and inputs_out.shape != x.shape:
        raise ValueError("convolution input destination has the wrong shape")
    if attention is None and x.stride(-1) != 1:
        x = x.contiguous()
    if dynamic is not None:
        dynamic = dynamic.contiguous()
    dtype = x.dtype if attention is None else torch.bfloat16
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    next_state = state_out
    if next_state is None:
        layout = torch.contiguous_format if attention is None else torch.preserve_format
        next_state = torch.empty_like(state, memory_format=layout)
    b, t, d = x.shape
    cast(Any, _conv)[(b * t, triton.cdiv(d, 256))](
        x,
        state,
        weight,
        dynamic,
        attention,
        partial,
        out,
        next_state,
        inputs_out,
        t,
        d,
        dynamic.shape[-2] if dynamic is not None else 1,
        *x.stride(),
        *state.stride(),
        *next_state.stride(),
        *(inputs_out.stride() if inputs_out is not None else (0, 0, 0)),
        *(attention.stride() if attention is not None else (0, 0, 0)),
        *(partial.stride() if partial is not None else (0, 0, 0)),
        256,
        enable_fp_fusion=False,
    )
    return out, next_state


@triton.jit(do_not_specialize=["Rows"])
def _attn_res(
    Sources,
    Addend,
    Q,
    Y,
    Rows,
    D: tl.constexpr,
    NQ: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, query = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, BLOCK)
    q = tl.load(Q + query * D + col, col < D, 0).to(tl.float32) * (D**-0.5)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    output = tl.full((BLOCK,), 0, tl.float32)
    for i in tl.static_range(len(Sources)):
        x = tl.load(Sources[i] + row * D + col, col < D, 0).to(tl.float32)
        if HAS_ADDEND and i == len(Sources) - 1:
            x += tl.load(Addend + row * D + col, col < D, 0).to(tl.float32)
        normalized = x * tl.rsqrt(tl.sum(x * x, 0) / D + 1e-6)
        score = tl.sum(normalized * q, 0)
        next_max = tl.maximum(maximum, score)
        correction, probability = tl.exp(maximum - next_max), tl.exp(score - next_max)
        output = output * correction + (normalized if NORMALIZE else x) * probability
        denominator = denominator * correction + probability
        maximum = next_max
    tl.store(Y + (query * Rows + row) * D + col, output / denominator, col < D)


def block_attention_residual(sources, query, normalize_values, partial_addend):
    shape = sources[0].shape
    rows, width = sources[0].numel() // shape[-1], shape[-1]
    queries = query.numel() // width
    # At native decode widths, separate source scores distribute the sequential
    # mixer across more SMs. Keep smaller source sets and prefill on one launch.
    if width == 4608 and 0 < rows <= 8 and 4 <= len(sources) <= 9 and queries in (1, 3):
        from resin_models_cuda.inkling2.residual import split_mix

        return split_mix(sources, query, normalize_values, partial_addend)
    out = torch.empty((queries, *shape), device=sources[0].device, dtype=torch.float32)
    cast(Any, _attn_res)[(rows, queries)](
        tuple(x.contiguous() for x in sources),
        partial_addend,
        query,
        out,
        rows,
        width,
        queries,
        normalize_values,
        partial_addend is not None,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return out.unbind(0)


@triton.jit(do_not_specialize=["N"])
def _add(Inputs, Y, N, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for i in tl.static_range(len(Inputs)):
        value += tl.load(Inputs[i] + col, col < N, 0).to(tl.float32)
    tl.store(Y + col, value, col < N)


def residual_add(inputs):
    inputs = tuple(x.contiguous() for x in inputs)
    out = torch.empty_like(inputs[0])
    cast(Any, _add)[(triton.cdiv(out.numel(), 512),)](inputs, out, out.numel(), 512)
    return out


@triton.jit(do_not_specialize=["L0", "L1"])
def _gate_sink(
    X,
    G,
    LSE,
    S,
    Y,
    H: tl.constexpr,
    D: tl.constexpr,
    L0,
    L1,
    BLOCK: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    col = block * BLOCK + tl.arange(0, BLOCK)
    mask = col < H * D
    x = tl.load(X + row * H * D + col, mask, 0).to(tl.float32)
    g = tl.load(G + row * H * D + col, mask, 0).to(tl.float32)
    s = tl.load(S + row * H + col // D, mask, 0).to(tl.float32)
    lse = tl.load(LSE + row * L0 + col // D * L1, mask, 0)
    tl.store(Y + row * H * D + col, x * tl.sigmoid(g) * tl.sigmoid(lse - s), mask)


def gate_sink(output, gate, lse, sinks):
    rows, heads, dim = output.shape
    result = torch.empty_like(output, dtype=gate.dtype)
    cast(Any, _gate_sink)[(rows, triton.cdiv(heads * dim, 256))](
        output,
        gate.contiguous(),
        lse,
        sinks.contiguous(),
        result,
        heads,
        dim,
        *lse.stride(),
        256,
    )
    return result


@triton.jit(do_not_specialize=["T"])
def _select(
    Initial,
    Inputs,
    Accepted,
    Y,
    Positions,
    Slabs,
    Slots,
    T,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    SLAB_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    POSITION_ROW: tl.constexpr,
    SLAB_ROW: tl.constexpr,
    SLOT_ROW: tl.constexpr,
):
    batch, block = tl.program_id(0), tl.program_id(1)
    col = block * BLOCK + tl.arange(0, BLOCK)
    mask = col < L * 3 * D
    layer, position, channel = col // (3 * D), (col // D) % 3, col % D
    accepted = tl.load(Accepted + batch)
    source = accepted + position - 3
    original = tl.load(
        Initial + batch * L * 3 * D + layer * 3 * D + (source + 3) * D + channel,
        mask & (source < 0),
        0,
    )
    current = tl.load(
        Inputs + ((batch * T + source) * L + layer) * D + channel,
        mask & (source >= 0),
        0,
    )
    if Positions is None:
        output = Y + batch * L * 3 * D
    else:
        end = tl.load(Positions + batch * POSITION_ROW) + accepted - 1
        slab = tl.load(Slabs + batch * SLAB_ROW).to(tl.int64)
        slot = tl.load(Slots + batch * SLOT_ROW).to(tl.int64)
        output = Y + tl.where(end >= 0, slab * SLAB_STRIDE + slot * SLOT_STRIDE, 0)
    tl.store(output + col, tl.where(source < 0, original, current), mask)


def select_convolution_state(initial, inputs, accepted):
    b, layers, _, width = initial.shape
    out = torch.empty_like(initial)
    cast(Any, _select)[(b, triton.cdiv(layers * 3 * width, 512))](
        initial.contiguous(),
        inputs.contiguous(),
        accepted,
        out,
        None,
        None,
        None,
        inputs.shape[1],
        layers,
        width,
        512,
        0,
        0,
        0,
        0,
        0,
    )
    return out


def store_selected_convolution_state(
    storage, group, positions, initial, inputs, accepted
):
    batch, layers, _, width = initial.shape
    block = 2048
    cast(Any, _select)[(batch, triton.cdiv(layers * 3 * width, block))](
        initial,
        inputs,
        accepted,
        storage,
        positions,
        group.slab_ids,
        group.slotnums,
        inputs.shape[1],
        layers,
        width,
        block,
        storage.stride(0),
        storage.stride(1),
        positions.stride(0),
        group.slab_ids.stride(0),
        group.slotnums.stride(0),
    )


@triton.jit(do_not_specialize=["T"])
def _checkpoint_state(
    Storage,
    Addresses,
    Candidates,
    Ends,
    Initial,
    Inputs,
    Features,
    Positions,
    Flags,
    T,
    L: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    SLAB: tl.constexpr,
    SLOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    item, tile = tl.program_id(0), tl.program_id(1)
    row = tl.load(Candidates + item * 2)
    prefix = tl.load(Candidates + item * 2 + 1)
    count = prefix - tl.load(Positions + row)
    ready = (count > 0) & (count <= T) & (prefix <= tl.load(Ends + row))
    if tile == 0:
        tl.store(Flags + item, ready)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    state_mask = ready & (col < L * 3 * D)
    layer, position, channel = col // (3 * D), (col // D) % 3, col % D
    source = count + position - 3
    previous = tl.load(
        Initial + row * L * 3 * D + layer * 3 * D + (source + 3) * D + channel,
        state_mask & (source < 0),
        0,
    )
    current = tl.load(
        Inputs + ((row * T + source) * L + layer) * D + channel,
        state_mask & (source >= 0),
        0,
    )
    feature = tl.load(
        Features + (row * T + count - 1) * H + col - L * 3 * D,
        ready & (col >= L * 3 * D) & (col < L * 3 * D + H),
        0,
    )
    slab = tl.load(Addresses + item * 4 + 2)
    slot = tl.load(Addresses + item * 4 + 3)
    tl.store(
        Storage + slab * SLAB + slot * SLOT + col,
        tl.where(col < L * 3 * D, tl.where(source < 0, previous, current), feature),
        ready & (slab != 0) & (col < L * 3 * D + H),
    )


def copy_state_checkpoints(
    storage, addresses, candidates, ends, initial, inputs, features, positions
):
    _, layers, _, width = initial.shape
    flags = torch.empty(len(addresses), dtype=torch.bool, device=storage.device)
    cast(Any, _checkpoint_state)[(len(addresses), triton.cdiv(storage.shape[2], 512))](
        storage,
        addresses,
        candidates,
        ends,
        initial,
        inputs,
        features,
        positions,
        flags,
        inputs.shape[1],
        layers,
        width,
        features.shape[2],
        storage.stride(0),
        storage.stride(1),
        512,
    )
    return flags


# Loop controllers adapt SGLang PR 51926 under Apache-2.0,
# commit 92e894ef904bfdb5c0cf2f2a541cbd4eec023573,
# python/sglang/kernels/ops/elementwise/inkling2_loop.py.


@triton.jit(do_not_specialize=["T", "DEST_STRIDE", "MESSAGE_STRIDE"])
def _controller_norm(
    Dest,
    Message,
    Normed,
    T,
    DEST_STRIDE,
    MESSAGE_STRIDE,
    eps,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        cuda.gdc_wait()
        cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < W
    dest = tl.load(Dest + m * DEST_STRIDE + t * W + cols, valid, other=0)
    message = tl.load(Message + m * MESSAGE_STRIDE + t * W + cols, valid, other=0)
    dest_rstd = libdevice.rsqrt(tl.div_rn(tl.sum(dest * dest), W) + eps)
    message_rstd = libdevice.rsqrt(tl.div_rn(tl.sum(message * message), W) + eps)
    out = Normed + (m * T + t) * (2 * W)
    tl.store(out + cols, (dest * dest_rstd).to(tl.bfloat16), valid)
    tl.store(out + W + cols, (message * message_rstd).to(tl.bfloat16), valid)


@triton.jit(do_not_specialize=["T"])
def _controller_down(
    Normed,
    Down,
    Partials,
    T,
    R: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Write the FP32 projection partial for one 1,024-column group."""
    if USE_PDL:
        cuda.gdc_wait()
        cuda.gdc_launch_dependents()
    GROUPS: tl.constexpr = (K + 1023) // 1024
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    z = tl.program_id(2)
    m, group = z // GROUPS, z % GROUPS
    out = Partials + (z.to(tl.int64) * T) * R
    X = Normed + (m.to(tl.int64) * T) * K
    Wd = Down + m.to(tl.int64) * R * K
    ks = tl.arange(0, BLOCK_K)
    partial = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for tile in range(1024 // BLOCK_K):
        k = group * 1024 + tile * BLOCK_K + ks
        x = tl.load(
            X + rows[:, None].to(tl.int64) * K + k[None, :],
            (rows[:, None] < T) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            Wd + cols[None, :] * K + k[:, None],
            (cols[None, :] < R) & (k[:, None] < K),
            other=0,
        )
        partial = tl.dot(x, w, partial)
    tl.store(
        out + rows[:, None].to(tl.int64) * R + cols[None, :],
        partial,
        (rows[:, None] < T) & (cols[None, :] < R),
    )


@triton.jit(do_not_specialize=["T", "DEST_STRIDE", "MESSAGE_STRIDE"])
def _controller_mix(
    Partials,
    Gain,
    Forget,
    Dest,
    Message,
    Out,
    T,
    DEST_STRIDE,
    MESSAGE_STRIDE,
    W: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Merge the BF16 low-rank features with the FP32 destination and message."""
    if USE_PDL:
        cuda.gdc_wait()
        cuda.gdc_launch_dependents()
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    m = tl.program_id(1).to(tl.int64)
    rank = tl.arange(0, BLOCK_R)
    row_ok = rows[:, None] < T
    hidden_ok = row_ok & (rank[None, :] < R)
    p = Partials + (m * GROUPS * T + rows[:, None]) * R + rank[None, :]
    hidden = tl.load(p, hidden_ok, other=0)
    for group in tl.static_range(1, GROUPS):
        hidden = hidden + tl.load(p + group * T * R, hidden_ok, other=0)
    hidden = hidden.to(tl.bfloat16)
    cols = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = cols[None, :] < W
    weight_ok = col_ok & (rank[:, None] < R)
    weights = m * W * R + cols[None, :] * R + rank[:, None]
    offsets = rows[:, None].to(tl.int64) * W + cols[None, :]
    mask = row_ok & col_ok
    dest = tl.load(Dest + m * DEST_STRIDE + offsets, mask, other=0)
    message = tl.load(Message + m * MESSAGE_STRIDE + offsets, mask, other=0)
    gain = tl.dot(hidden, tl.load(Gain + weights, weight_ok, other=0))
    forget = tl.dot(hidden, tl.load(Forget + weights, weight_ok, other=0))
    output = (1.0 - libdevice.tanh(forget)) * dest + libdevice.tanh(gain) * message
    tl.store(Out + m * T * W + offsets, output, mask)


def _controller_rows(rows: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, int]:
    base = rows[0]
    if len(rows) == 1:
        return base, 0
    storage = base.untyped_storage().data_ptr()
    start = base.storage_offset()
    stride = int(rows[1].storage_offset() - start)
    if all(
        row.untyped_storage().data_ptr() == storage
        and row.storage_offset() == start + index * stride
        for index, row in enumerate(rows[1:], 1)
    ):
        return base, stride
    return torch.stack(rows), base.numel()


@cache
def _controller_wide_block_k(device_index: int) -> int:
    props = cast(Any, triton.runtime.driver.active).utils.get_device_properties(
        device_index
    )
    # The 64-by-64 projection's wide K tiles need 128 KiB of shared memory.
    return 256 if props["max_shared_mem"] >= 128 * 1024 else 128


def mix_loop_messages(dests, messages, down, gain, forget):
    """Blend one MLP stream or all three attention streams in three launches."""
    dest, dest_stride = _controller_rows(dests)
    message, message_stride = _controller_rows(messages)
    tokens, width = dests[0].shape
    streams, rank, pair_width = down.shape
    pdl = torch.cuda.get_device_capability(dest.device)[0] >= 9
    launch = {"launch_pdl": True} if pdl else {}
    normalized = torch.empty(
        (streams, tokens, pair_width), dtype=torch.bfloat16, device=dest.device
    )
    cast(Any, _controller_norm)[(tokens, streams)](
        dest,
        message,
        normalized,
        tokens,
        dest_stride,
        message_stride,
        1e-6,
        width,
        triton.next_power_of_2(width),
        pdl,
        num_warps=8,
        enable_fp_fusion=False,
        **launch,
    )
    groups = triton.cdiv(pair_width, 1024)
    small = tokens <= 16
    down_m, down_n = (16, 16) if small else (64, 64)
    down_k = 256 if small else _controller_wide_block_k(dest.device.index)
    partials = dest.new_empty((streams, groups, tokens, rank))
    cast(Any, _controller_down)[
        (triton.cdiv(tokens, down_m), triton.cdiv(rank, down_n), streams * groups)
    ](
        normalized,
        down,
        partials,
        tokens,
        rank,
        pair_width,
        down_m,
        down_n,
        down_k,
        pdl,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
        **launch,
    )
    output = dest.new_empty((streams, tokens, width))
    mix_m = 16 if small else 32
    cast(Any, _controller_mix)[
        (triton.cdiv(tokens, mix_m), streams, triton.cdiv(width, 64))
    ](
        partials,
        gain,
        forget,
        dest,
        message,
        output,
        tokens,
        dest_stride,
        message_stride,
        width,
        rank,
        max(16, triton.next_power_of_2(rank)),
        groups,
        mix_m,
        64,
        pdl,
        num_warps=4 if small else 8,
        enable_fp_fusion=False,
        **launch,
    )
    return output


@triton.jit
def _transfer_live_state(
    Cache,
    Slabs,
    Slots,
    Positions,
    State,
    SLAB_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    SLAB_ROW: tl.constexpr,
    SLOT_ROW: tl.constexpr,
    POSITION_ROW: tl.constexpr,
    WIDTH: tl.constexpr,
    STORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    slab = tl.load(Slabs + row * SLAB_ROW).to(tl.int64)
    slot = tl.load(Slots + row * SLOT_ROW).to(tl.int64)
    position = tl.load(Positions + row * POSITION_ROW)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offset < WIDTH
    if STORE:
        slab, slot = tl.where(position >= 0, slab, 0), tl.where(position >= 0, slot, 0)
    cache_row = Cache + slab * SLAB_STRIDE + slot * SLOT_STRIDE
    state_row = State + row.to(tl.int64) * WIDTH
    if STORE:
        value = tl.load(state_row + offset, mask, 0)
        tl.store(cache_row + offset, value, mask)
    else:
        value = tl.load(cache_row + offset, mask & (position != 0), 0)
        tl.store(state_row + offset, value, mask)


def transfer_live_state(storage, group, positions, state, *, store):
    """Copy a contiguous state field between per-request rows and live slots."""
    rows = positions.numel()
    width = state.numel() // rows
    block = 4096
    cast(Any, _transfer_live_state)[(rows, triton.cdiv(width, block))](
        storage,
        group.slab_ids,
        group.slotnums,
        positions,
        state,
        storage.stride(0),
        storage.stride(1),
        group.slab_ids.stride(0),
        group.slotnums.stride(0),
        positions.stride(0),
        width,
        store,
        block,
    )
