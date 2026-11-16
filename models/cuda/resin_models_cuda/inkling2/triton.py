"""Fused Inkling 2 normalization, residual attention, convolution, and gating."""

from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


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


@triton.jit(
    do_not_specialize=["T", "layer", "KS", "VS", "DKS", "DVS", "PS", "GS", "WIDTH"]
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
    KS,
    VS,
    DKS,
    DVS,
    PS,
    GS,
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
        PS,
        GS,
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
    key = tl.load(
        K + batch * KS[0] + token * KS[1] + head * KS[2] + dim * KS[3], mask, 0
    )
    value = tl.load(
        V + batch * VS[0] + token * VS[1] + head * VS[2] + dim * VS[3], mask, 0
    )
    tl.store(
        DestK + page * DKS[0] + offset * DKS[1] + head * DKS[2] + dim * DKS[3],
        key,
        mask,
    )
    tl.store(
        DestV + page * DVS[0] + offset * DVS[1] + head * DVS[2] + dim * DVS[3],
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
        keys.stride(),
        values.stride(),
        cache.keys.stride(),
        cache.values.stride(),
        positions.stride(),
        (
            *group.slab_ids.stride(),
            *group.slotnums.stride(),
            group.first_blocks.stride(0),
        ),
        group.slab_ids.shape[1],
        heads,
        dim,
        cache.page_tokens,
        cache.slab_units,
        layer_count,
        group.live_state,
        256,
    )


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
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride + cols, cols < D, 0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    if HAS_WEIGHT:
        y *= tl.load(W + cols, cols < D, 0).to(tl.float32)
    tl.store(Y + row * D + cols, y, cols < D)


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
    T,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch, block = tl.program_id(0), tl.program_id(1)
    col = block * BLOCK + tl.arange(0, BLOCK)
    mask = col < L * 3 * D
    layer, position, channel = col // (3 * D), (col // D) % 3, col % D
    source = tl.load(Accepted + batch) + position - 3
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
    tl.store(Y + batch * L * 3 * D + col, tl.where(source < 0, original, current), mask)


def select_convolution_state(initial, inputs, accepted):
    b, layers, _, width = initial.shape
    out = torch.empty_like(initial)
    cast(Any, _select)[(b, triton.cdiv(layers * 3 * width, 512))](
        initial.contiguous(),
        inputs.contiguous(),
        accepted,
        out,
        inputs.shape[1],
        layers,
        width,
        512,
    )
    return out
