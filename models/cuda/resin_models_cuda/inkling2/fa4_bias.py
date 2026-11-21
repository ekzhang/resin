"""Native relative-bias FA4 for prepared target decode inputs.

The producer writes padded BF16 queries and presheared BF16 relative bias.
Attention retains FP32 partials/LSE until the existing combined sink/gate step.
Only fixed model/decode geometry enters compilation; context lengths are data.
"""

from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

compile_count = 0
DECODE_TOKENS = (1, 8)
MIN_BIAS_TILE_ROWS = 8
MAX_BIAS_TILE_ROWS = 64
NARROW_PACKED_GROUP = 4


def packed_decode_group(heads: int, kv_heads: int) -> int | None:
    """Return the packed query group when it fits the decode bias tile."""
    if kv_heads not in (2, 4, 8) or heads <= 0 or heads % kv_heads:
        return None
    padded_group = 1 << ((heads // kv_heads) - 1).bit_length()
    return (
        padded_group
        if padded_group * max(DECODE_TOKENS) <= MAX_BIAS_TILE_ROWS
        else None
    )


def packed_decode_batch_supported(
    padded_group: int, batch: int, *, inputs_padded: bool
):
    if padded_group <= NARROW_PACKED_GROUP:
        return batch <= 64
    return inputs_padded or batch == 1


def decode_splits(batch, kv_heads, window_left):
    """Keep enough independent KV tiles as the request batch supplies more CTAs."""
    return max(1, (16 if window_left is not None or kv_heads == 8 else 32) // batch)


def packed_decode_splits(
    batch, kv_heads, padded_group, window_left, *, page_capacity=None
):
    if padded_group > NARROW_PACKED_GROUP:
        return decode_splits(batch, kv_heads, window_left)
    # Local decode visits at most ten retained pages. Give global decode more
    # CTAs for long contexts, bounded by the prepared table capacity.
    if window_left is not None:
        return max(1, 8 // batch)
    if page_capacity is None:
        return max(1, 16 // batch)
    return max(1, min(32, 72 // batch, page_capacity))


@cache
def _kernels():
    from flash_attn.cute.cache_utils import get_jit_cache

    directory = Path(__file__).parent
    source = b"".join(
        (directory / name).read_bytes()
        for name in ("fa4_bias.py", "fa4_bias_kernel.py", "fa4_bias_utils.py")
    )
    return get_jit_cache(
        "resin_inkling2_native_bias_" + sha256(source).hexdigest()[:16]
    )


def native_bias_attention(
    q,
    k,
    v,
    bias,
    gate,
    sinks,
    *,
    page_table,
    seqused_k,
    cu_seqlens_q,
    tokens,
    num_splits,
    window_left=None,
):
    import cutlass.cute as cute
    from flash_attn.cute.cute_dsl_utils import get_broadcast_dims, to_cute_tensor
    from flash_attn.cute.utils import AuxData

    from resin_models_cuda.inkling2.decode_attention import combine_gate_sink
    from resin_models_cuda.inkling2.fa4_bias_kernel import FlashAttentionForwardSm100

    batch, kv_heads = seqused_k.numel(), k.shape[2]
    rows = batch * tokens
    heads = sinks.shape[-1]
    padded_group = packed_decode_group(heads, kv_heads)
    if (
        not batch
        or tokens not in DECODE_TOKENS
        or padded_group is None
        or q.shape != (rows, kv_heads * padded_group, 128)
        or bias.shape != (rows, kv_heads * padded_group, 1280)
        or k.shape[1] != 128
        or k.shape != v.shape
        or gate.shape != (rows, heads, 128)
        or sinks.shape != gate.shape[:2]
        or seqused_k.shape != (batch,)
        or cu_seqlens_q.shape != (batch + 1,)
        or page_table.ndim != 2
        or page_table.shape[0] != batch
        or any(t.dtype != torch.bfloat16 for t in (q, k, v, bias))
        or not q.is_contiguous()
        or not bias.is_contiguous()
    ):
        raise ValueError(
            "native Inkling bias attention requires prepared decode inputs"
        )
    partials = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
    lse = torch.empty(
        (num_splits, q.shape[1], rows), dtype=torch.float32, device=q.device
    )
    args = [
        q,
        k,
        v,
        partials,
        lse,
        128**-0.5,
        cu_seqlens_q,
        None,
        None,
        seqused_k,
        page_table,
        window_left,
        0,
        None,
        None,
        None,
        AuxData(),
        bias,
        None,
        None,
        None,
        None,
        None,
    ]
    key = (
        torch.cuda.get_device_capability(q.device),
        tokens,
        kv_heads,
        padded_group,
        window_left is not None,
        tuple(get_broadcast_dims(arg) for arg in args if isinstance(arg, torch.Tensor)),
    )
    kernels = _kernels()
    if key not in kernels:
        kernel = FlashAttentionForwardSm100(
            is_causal=window_left is None,
            is_local=window_left is not None,
            qhead_per_kvhead=padded_group,
            bias_block_size=max(MIN_BIAS_TILE_ROWS, tokens * padded_group),
        )
        converted = [
            to_cute_tensor(
                arg,
                assumed_align=4 if i in (4, 6, 9, 10) else 16,
                leading_dim=0 if i in (6, 9) else -1,
            )
            if isinstance(arg, torch.Tensor)
            else arg
            for i, arg in enumerate(args)
        ]
        global compile_count
        compile_count += 1
        kernels[key] = cute.compile(
            kernel,
            *converted,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    launch: Any = kernels[key]
    launch(*args[:-2])
    return combine_gate_sink(
        partials, lse, gate, sinks, padded=True, heads_per_kv=heads // kv_heads
    )


@torch.inference_mode()
def warm_native_bias(kv_heads, device, *, heads_per_kv=6):
    """Compile query and broadcast-layout regimes before graph capture."""
    padded_group = packed_decode_group(kv_heads * heads_per_kv, kv_heads)
    if padded_group is None:
        return
    keys = torch.zeros((3, 128, kv_heads, 128), device=device, dtype=torch.bfloat16)
    values = torch.zeros_like(keys)
    for batch in (1, 2):
        pages = torch.arange(3, device=device, dtype=torch.int32).repeat(batch, 1)
        used = torch.full((batch,), 128, device=device, dtype=torch.int32)
        for tokens in DECODE_TOKENS:
            query = keys.new_zeros((batch * tokens, kv_heads * padded_group, 128))
            bias = keys.new_zeros((batch * tokens, kv_heads * padded_group, 1280))
            gate = keys.new_zeros((batch * tokens, kv_heads * heads_per_kv, 128))
            sinks = keys.new_zeros((batch * tokens, kv_heads * heads_per_kv))
            cu_query = (
                torch.arange(batch + 1, device=device, dtype=torch.int32) * tokens
            )
            for window in (None, 1023):
                # The combine kernel specializes on the next power of two of
                # the split count; CuTe only distinguishes broadcast layouts.
                for splits in (1, 2, 4, 8, 16, 32):
                    native_bias_attention(
                        query,
                        keys,
                        values,
                        bias,
                        gate,
                        sinks,
                        page_table=pages,
                        seqused_k=used,
                        cu_seqlens_q=cu_query,
                        tokens=tokens,
                        num_splits=splits,
                        window_left=window,
                    )
