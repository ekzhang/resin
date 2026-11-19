"""Blackwell FA4 launch preserving FP32 attention until Inkling's sink/gate.

FA4's public wrapper forces the query dtype. Use the SM100 kernel directly
for paged BF16 KV and the same FP32 output boundary as pinned SGLang.
Launch preparation follows flash_attn/cute/interface.py in flash-attn-4 4.0.0b40
(BSD-3-Clause, see LICENSE.flash-attention); SGLang reference: e62ac6279646c686d3b328f58d9b5c2c2d5927dc.
"""

from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

compile_count = 0


@cache
def _kernel_cache():
    from flash_attn.cute.cache_utils import get_jit_cache

    return get_jit_cache(
        "resin_inkling2_fp32_" + sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    )


def paged_attention_fp32(
    q,
    k,
    v,
    *,
    cu_seqlens_q,
    seqused_k,
    page_table,
    max_seqlen_q,
    softmax_scale,
    causal,
    window_size,
    score_mod=None,
    aux_tensors=None,
    num_splits=1,
):
    import cutlass.cute as cute
    from flash_attn.cute.config import FwdConfig, Sm100Registers
    from flash_attn.cute.cute_dsl_utils import (
        get_aux_tensor_metadata,
        get_broadcast_dims,
        to_cute_aux_tensor,
        to_cute_tensor,
    )
    from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
    from flash_attn.cute.interface import _resolve_causal_local_window
    from flash_attn.cute.utils import AuxData, hash_callable

    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Inkling 2 FP32 attention requires BF16 Q/K/V")
    if q.shape[-1] != 128 or v.shape[-1] != 128 or k.shape[1] != 128:
        raise ValueError("Inkling 2 FP32 attention requires head and page size 128")
    if type(num_splits) is not int or num_splits <= 0:
        raise ValueError("Inkling 2 attention split count must be positive")
    split = num_splits > 1
    # Split mode returns normalized FP32 partial outputs and natural-log LSEs;
    # the decode epilogue combines them together with the sink and channel gate.
    leading = (num_splits,) if split else ()
    out = torch.empty((*leading, *q.shape), device=q.device, dtype=torch.float32)
    lse = torch.empty(
        (*leading, q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32
    )
    left, right = window_size
    causal, local, left, right = _resolve_causal_local_window(causal, left, right)
    # Six query heads share a KV head in the released model. The generic FA4
    # packed tile requires an integral number of head groups per 128-row tile.
    ratio = q.shape[1] // k.shape[2]
    pack_gqa = 128 % ratio == 0 and ratio > 1
    aux = aux_tensors or []
    capability = torch.cuda.get_device_capability(q.device)
    key = (
        capability,
        ratio,
        causal,
        local,
        pack_gqa,
        split,
        hash_callable(score_mod) if score_mod is not None else None,
        get_aux_tensor_metadata(aux),
        tuple(get_broadcast_dims(t) for t in (q, k, v, out, lse, page_table)),
    )
    kernels = _kernel_cache()
    if key not in kernels:
        major, minor = capability
        sm103 = major == 10 and minor >= 3
        softmax_regs, correction_regs = 192, 80
        if causal:
            softmax_regs, correction_regs = (176, 64) if sm103 else (192, 72)
        emulation = 0 if sm103 else 32 if pack_gqa and not (causal or local) else 16
        # Keep the old 1-CTA, one-Q-stage launch. FP32 O shares the Q buffer in
        # split mode, leaving room for five rather than four BF16 KV stages.
        # The kernel reads the actual split count from the output tensor shape.
        config = FwdConfig(
            device_capacity=major,
            tile_m=128,
            tile_n=128,
            mma_pv_is_rs=False,
            intra_wg_overlap=False,
            q_stage=1,
            pack_gqa=pack_gqa,
            use_clc_scheduler=False,
            is_static_persistent=False,
            use_tma_o=False,
            use_s_ping_pong=False,
            num_splits=num_splits,
            use_2cta_instrs=False,
            registers=Sm100Registers(
                softmax_regs, correction_regs, 512 - 2 * softmax_regs - correction_regs
            ),
            ex2_emu_freq=emulation,
            ex2_emu_start_frg=int(bool(emulation)),
            overlap_sO_sQ=split,
            kv_stage=5 if split else 4,
            sm_stats_cross_stage=False,
        )
        kernel = FlashAttentionForwardSm100(
            128,
            128,
            qhead_per_kvhead=ratio,
            is_causal=causal,
            is_local=local,
            config=config,
            score_mod=score_mod,
            has_aux_tensors=bool(aux),
            paged_kv_non_tma=False,
            is_varlen_q=True,
        )
        # Layouts are dynamic: neither logical context nor packed query count
        # enters the compile key. The current stream is supplied by TVM FFI.
        args = [to_cute_tensor(t) for t in (q, k, v, out)]
        args.extend(
            [
                to_cute_tensor(lse, assumed_align=4),
                softmax_scale,
                to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0),
                None,
                None,
                to_cute_tensor(seqused_k, assumed_align=4, leading_dim=0),
                to_cute_tensor(page_table, assumed_align=4, leading_dim=1),
                left,
                right,
                None,
                None,
                None,
                AuxData([to_cute_aux_tensor(t) for t in aux] or None, None),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                max_seqlen_q,
                None,  # Tensor-map workspace for ragged V.
                None,  # Q scales for MXFP8.
                None,  # K scales for MXFP8.
                None,  # V scales for MXFP8.
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            ]
        )
        global compile_count
        compile_count += 1
        kernels[key] = cute.compile(kernel, *args, options="--enable-tvm-ffi")
    launch: Any = kernels[key]
    launch(
        q,
        k,
        v,
        out,
        lse,
        softmax_scale,
        cu_seqlens_q,
        None,
        None,
        seqused_k,
        page_table,
        left,
        right,
        None,
        None,
        None,
        AuxData(aux or None, None),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        max_seqlen_q,
        None,
        None,
        None,
        None,
    )
    return out, lse
