# ruff: noqa: F841
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026, Colfax International. (modifications)

# Specialized from SGLang e62ac6279646c686d3b328f58d9b5c2c2d5927dc.
# Only BF16 Q/K/V, FP32 output, head/page128, packed GQA8, split-KV,
# and pre-sheared1280-column bias are retained. See LICENSE.flash-attention.
# Original upstream features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# - varlen
# - sliding window
# - split-kv
#
# Colfax modifications:
# - relative bias
# - MXFP8 dtype
#
# Unsupported features that will be added later:
# - page size != 128
# - more hdim (192, 256)
#
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

# Native BF16/FP32 paged split-KV specialization, generated once from the pinned source.
from functools import partial
from typing import Callable, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import flash_attn.cute.pipeline as pipeline_custom
from cutlass import Boolean, Float32, Int32, Int64, const_expr, pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from flash_attn.cute import utils
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.flash_fwd_sm100 import DescaleTensors
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100 as FA4Forward
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.pack_gqa import pack_gqa_layout
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.tile_scheduler import (
    SchedulingMode,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
    TileSchedulerProtocol,
)
from flash_attn.cute.utils import AuxData
from quack import copy_utils, layout_utils
from quack.cute_dsl_utils import ParamsBase

from resin_models_cuda.inkling2.fa4_bias_utils import BlockInfo, NamedBarrierFwdSm100

_TUNING_CONFIG = {
    (True, False): {
        "ex2_emu_freq": 16,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 192,
        "num_regs_correction": 72,
    },
    (True, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 64,
    },
}


# Reuse the pinned FA4 MMA, correction, and output pipeline unchanged.
# Only the pre-sheared relative-bias producer/softmax path lives here.
class FlashAttentionForwardSm100(FA4Forward):
    def __init__(self, *, is_causal: bool, is_local: bool, bias_block_size: int):
        """Paged BF16 attention with GQA8 and pre-sheared relative bias."""
        assert bias_block_size in (8, 64)
        assert is_causal != is_local
        self.head_dim_padded = self.head_dim_v_padded = 128
        self.m_block_size = self.n_block_size = 128
        self.q_stage = self.bias_stage = self.cta_group_size = 1
        self.split_P_arrive = 96
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch.is_family_of(Arch.sm_100f) or self.arch.is_family_of(
            Arch.sm_110f
        ), "Only SM 10.x and 11.x are supported"
        self.cta_tiler = self.mma_tiler_qk = self.mma_tiler_pv = (128, 128, 128)
        self.qk_acc_dtype = self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_causal = is_causal
        self.is_local = is_local
        self.qhead_per_kvhead = 8
        self.is_split_kv = self.pack_gqa = self.has_bias = True
        self.is_persistent = self.use_2cta_instrs = False
        self.use_tma_Q = self.use_tma_KV = True
        self.use_tma_O = self.use_clc_scheduler = False
        self.check_hdim_oob = self.check_hdim_v_oob = False
        self.overlap_sO_sQ = True
        self.batch_invariant = self.rel_bias_scale_after_add = False
        self.use_tmem_load_red_max = False
        self.score_mod = None
        self.use_block_sparsity = self.uneven_kv_smem = False
        self.use_correction_warps_for_epi = self.use_varlen_scheduler = True
        self.q_subtile_factor = self.kv_subtile_factor = 1
        self.score_vec_size = 2
        self.mask_vec_size = 1
        self.bias_n_max = 10
        self.bias_block_size = bias_block_size
        self.scheduling_mode = SchedulingMode.STATIC
        self.TileScheduler = SingleTileVarlenScheduler
        self.is_sm103 = self.arch.is_family_of(Arch.sm_103f)
        self._tune = _TUNING_CONFIG.get((is_causal, self.is_sm103), {})
        self.enable_ex2_emu = self._tune.get("ex2_emu_freq", int(not self.is_sm103)) > 0
        # Four softmax and four correction warps; separate MMA and TMA warps.
        # Remaining warps still participate in the same 512-thread CTA.
        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = ()
        self.correction_warp_ids = (8, 9, 10, 11)
        self.epilogue_warp_ids = self.correction_warp_ids
        self.mma_warp_id = 12
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15, 4, 5, 6, 7, 13)
        self.threads_per_cta = 512
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_s_offset = [0, 128]
        self.tmem_o_offset = [256]
        self.tmem_total = 384
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = 64
        self.tmem_p_offset = [64, 192]
        self.tmem_vec_offset = self.tmem_s_offset
        self.num_regs_softmax = self._tune.get("num_regs_softmax", 192) + 8
        self.num_regs_correction = self._tune.get("num_regs_correction", 80)
        self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """
        smem_size_q = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_padded
            * self.q_dtype.width
            // 8
        )
        smem_size_bias = (
            self.bias_stage
            * self.bias_block_size
            * self.n_block_size
            * self.bias_dtype.width
            // 8
        )
        smem_size_o = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_v_padded
            * self.o_dtype.width
            // 8
        )
        smem_size_q_o_bias = max(smem_size_q + smem_size_bias, smem_size_o)
        smem_size_k_per_stage = (
            self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        )
        smem_size_v_per_stage = (
            self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        )
        smem_size_kv_per_stage = (
            max(smem_size_k_per_stage, smem_size_v_per_stage) // self.cta_group_size
        )
        kv_stage = min((224 * 1024 - smem_size_q_o_bias) // smem_size_kv_per_stage, 32)
        v_mma_stage = kv_stage
        self.kv_stage = kv_stage
        self.v_mma_stage = v_mma_stage
        self.s_stage = 2
        assert self.s_stage >= self.q_stage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        descale_tensors: Optional[DescaleTensors] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        mBias: Optional[cute.Tensor] = None,
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        mSFV: Optional[cute.Tensor] = None,
        qk_sf_vec_size: cutlass.Constexpr[Optional[int]] = None,
        v_sf_vec_size: cutlass.Constexpr[Optional[int]] = None,
        stream: cuda.CUstream = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        self.kv_size_ratio = self.v_dtype.width // self.k_dtype.width
        self.sfq_dtype = None
        self.sfk_dtype = None
        self.sfv_dtype = None
        self.qk_sf_vec_size = 1
        self.v_sf_vec_size = 1
        self.v_mma_dtype = self.v_dtype
        self.bias_dtype = mBias.element_type
        assert (
            self.q_dtype
            == self.k_dtype
            == self.v_dtype
            == self.bias_dtype
            == cutlass.BFloat16
        )
        assert self.o_dtype == Float32
        assert (
            blocksparse_tensors is None
            and mSFQ is None
            and mSFK is None
            and mSFV is None
        )
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        mBias = assume_tensor_aligned(mBias)
        mSFQ, mSFK, mSFV = [
            assume_tensor_aligned(t, align=4) if const_expr(t is not None) else None
            for t in (mSFQ, mSFK, mSFV)
        ]
        Q_layout_transpose = [0, 2, 1]
        mQ = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)
        )
        mBias = cute.make_tensor(
            mBias.iterator, cute.select(mBias.layout, mode=Q_layout_transpose)
        )
        KV_layout_transpose = [1, 3, 2, 0]
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        O_layout_transpose = [1, 3, 2, 0]
        LSE_layout_transpose = [2, 1, 0]
        num_splits = mO.shape[0]
        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        mLSE = cute.make_tensor(
            mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
        )
        mV_sf_shape = mV.shape
        V_layout_transpose = [1, 0, 2, 3]
        mV = cute.make_tensor(
            mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
        )
        self._setup_attributes()
        self.ex2_emu_freq = 0
        self.ex2_emu_start_frg = self._tune.get("ex2_emu_start_frg", 1)
        if const_expr(self.enable_ex2_emu):
            self.ex2_emu_freq = self._tune.get("ex2_emu_freq", 16)
            if const_expr(
                self.pack_gqa
                and self.head_dim_padded > 64
                and (not self.is_causal)
                and (not self.is_local)
            ):
                self.ex2_emu_freq = 32
        cta_group = tcgen05.CtaGroup.ONE
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_mma_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)
        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage * self.kv_size_ratio,
        )
        sSFQ_layout = None
        sSFK_layout = None
        sSFV_layout = None
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.v_mma_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_mma_dtype, self.v_mma_stage
        )
        tiled_mma_pv_vq = None
        sVq_layout = None
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )
        nheads_kv = mK.shape[2]
        mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
        mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
        mBias = pack_gqa_layout(mBias, self.qhead_per_kvhead, nheads_kv, head_idx=2)
        mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(
                mX.element_type, cute.select(layout, mode=[0, 1, 2])
            )
            if const_expr(mX is not None)
            else 0
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
                ("SFQ", mSFQ, sSFQ_layout),
                ("SFK", mSFK, sSFK_layout),
                ("SFV", mSFV, sSFV_layout),
            ]
        }
        for name in ("Q", "K", "V", "SFQ", "SFK", "SFV"):
            self.tma_copy_bytes[name] *= self.cta_group_size
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        gmem_tiled_copy_Q = None
        tma_atom_K = None
        tma_atom_V = None
        tma_atom_SFQ = None
        tma_atom_SFK = None
        tma_atom_SFV = None
        gmem_tiled_copy_SFQ = None
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            cta_layout_vmnk.shape,
        )
        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        tma_atom_O = None
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.o_dtype.width
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.o_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1), order=(1, 0)
        )
        assert self.m_block_size % tO_layout.shape[0] == 0
        vO_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )
        bias_layout_enum = cutlass.utils.LayoutEnum.from_tensor(mBias)
        self.bias_major_mode = bias_layout_enum.mma_major_mode()
        if const_expr(self.bias_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mBias is wrong")
        sBias_layout = sm100_utils_basic.make_smem_layout_epi(
            self.bias_dtype,
            bias_layout_enum,
            (self.bias_block_size, self.n_block_size),
            self.bias_stage,
        )
        sBias_size = cute.cosize(sBias_layout)
        self.tma_copy_bytes["bias"] = cute.size_in_bytes(
            self.bias_dtype, cute.select(sBias_layout, mode=[0, 1])
        )
        tma_atom_bias, mBias = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mBias,
            cute.select(sBias_layout, mode=[0, 1]),
            (self.bias_block_size, self.n_block_size),
            1,
        )
        bias_s2r_thr_layout = cute.make_ordered_layout(
            (self.bias_block_size, 1), order=(1, 0)
        )
        bias_s2r_val_layout = cute.make_ordered_layout(
            (1, 128 // self.bias_dtype.width), order=(1, 0)
        )
        bias_s2r_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.bias_dtype, num_bits_per_copy=128
        )
        bias_s2r_tiled_copy = cute.make_tiled_copy_tv(
            bias_s2r_copy_atom, bias_s2r_thr_layout, bias_s2r_val_layout
        )
        TileScheduler = self.TileScheduler
        _num_block_divisor = self.cta_tiler[0] * 1
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), _num_block_divisor),
            cute.size(mQ.shape[2]),
            cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[0],
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=not self.is_persistent and self.cta_group_size > 1,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        sO_size = 0
        sQ_size = cutlass.max(
            cute.cosize(sQ_layout),
            cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width,
        )
        sQ_size = cute.cosize(sQ_layout)
        clc_response_size = 0
        clc_mbar_size = 0
        sSFQ_size = 0
        sSFK_size = 0
        sSFV_size = 0
        sVq_size = 0
        sV_dequant_size = 0
        use_sf_mbar = 0
        use_vq_mbar = 0
        use_sfq_mbar = 0

        @cute.struct
        class SharedStorage:
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            mbar_load_bias: cute.struct.MemRange[Int64, self.bias_stage * 2]
            mbar_s0_s1_empty_for_sf: cute.struct.MemRange[
                Int64, self.q_stage * 2 * use_sf_mbar
            ]
            mbar_load_SFQ: cute.struct.MemRange[Int64, self.q_stage * 2 * use_sfq_mbar]
            mbar_load_Vq: cute.struct.MemRange[Int64, self.kv_stage * 2 * use_vq_mbar]
            mbar_v_upcast: cute.struct.MemRange[
                Int64, self.v_mma_stage * 2 * use_vq_mbar
            ]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.MemRange[Int32, clc_response_size]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sBias: cute.struct.Align[
                cute.struct.MemRange[self.bias_dtype, sBias_size],
                self.buffer_align_bytes,
            ]
            sVq: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, sVq_size],
                self.buffer_align_bytes,
            ]
            sV_dequant: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, sV_dequant_size],
                self.buffer_align_bytes,
            ]
            sSFQ: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, sSFQ_size],
                self.buffer_align_bytes,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, sSFK_size],
                self.buffer_align_bytes,
            ]
            sSFV: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, sSFV_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        base_softmax_scale = softmax_scale
        softmax_scale_log2, softmax_scale = (utils.LOG2_E, None)
        window_size_left = (
            Int32(window_size_left) if window_size_left is not None else None
        )
        window_size_right = (
            Int32(window_size_right) if window_size_right is not None else None
        )
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors, mPageTable
        )
        head_divmod = None
        head_divmod = FastDivmodDivisor(self.qhead_per_kvhead)
        self.use_block_sparsity = False
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            descale_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_data,
            fastdiv_mods,
            head_divmod,
            mBias,
            tma_atom_bias,
            sBias_layout,
            bias_s2r_tiled_copy,
            base_softmax_scale,
            tma_atom_SFQ,
            tma_atom_SFK,
            tma_atom_SFV,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk
            if cute.size(self.cluster_shape_mnk) > 1
            else None,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.has_bias,
        )

    def _generate_attention_mask_cls(self, window_size_left, window_size_right):
        return partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        descale_tensors: Optional[DescaleTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        mBias: Optional[cute.Tensor] = None,
        tma_atom_bias: Optional[cute.CopyAtom] = None,
        sBias_layout: Optional[cute.ComposedLayout] = None,
        bias_s2r_tiled_copy: Optional[cute.TiledCopy] = None,
        base_softmax_scale: Optional[Float32] = None,
        tma_atom_SFQ: Optional[cute.CopyAtom] = None,
        tma_atom_SFK: Optional[cute.CopyAtom] = None,
        tma_atom_SFV: Optional[cute.CopyAtom] = None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_O,
                tma_atom_SFQ,
                tma_atom_SFK,
                tma_atom_SFV,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len(
                (
                    self.mma_warp_id,
                    *self.softmax0_warp_ids,
                    *self.softmax1_warp_ids,
                    *self.correction_warp_ids,
                )
            ),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(
            len(self.load_warp_ids) * cute.arch.WARP_SIZE
        )
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        correction_warps = ThreadCooperativeGroup(len(self.correction_warp_ids))
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        )
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE
            * len(self.softmax0_warp_ids + self.correction_warp_ids)
            * self.cta_group_size
        )
        pipeline_q_tx_count = self.tma_copy_bytes["Q"]
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=pipeline_q_tx_count,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s0_s1_sequence = None
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )
        pipeline_o_epi = None
        pipeline_bias = None
        if const_expr(tma_atom_bias is not None):
            pipeline_bias = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_load_bias.data_ptr(),
                num_stages=self.bias_stage,
                producer_group=tma_warp,
                consumer_group=softmax_warps,
                tx_count=self.tma_copy_bytes["bias"],
                defer_sync=True,
            )
        pipeline_sfq = None
        pipeline_sf_overlap = None
        pipeline_vq = None
        pipeline_v_mma = None
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sVq, sSFQ, sSFK, sSFV = (None, None, None, None)
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sV_layout.inner, self.v_mma_dtype),
            sV_layout.outer,
        )
        sO = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer
        )
        sBias = storage.sBias.get_tensor(sBias_layout.outer, swizzle=sBias_layout.inner)
        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.q_stage * self.m_block_size * 2)
        )
        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)
        thr_mma_qk_sfk = None
        thr_mma_pv_sfv = None
        thr_mma_pv_vq = None
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        tP_width_ratio = Float32.width // self.v_mma_dtype.width
        tP_stage_stride = (
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )
        block_info = BlockInfo(
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead,
            batch_invariant=self.batch_invariant,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0][1],
            seqlen_k_static=mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=None,
            mCuBlockIdxOffsets=None,
        )
        AttentionMaskCls = self._generate_attention_mask_cls(
            window_size_left, window_size_right
        )
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)
        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(tile_scheduler, TileSchedulerProtocol), (
            f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"
        )
        for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
            if warp_idx == self.empty_warp_ids[i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                mBias=mBias,
                sBias=sBias,
                tma_atom_bias=tma_atom_bias,
                pipeline_bias=pipeline_bias,
            )
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                is_leader_cta,
                block_info,
                num_splits,
                SeqlenInfoCls,
                None,
                tile_scheduler=tile_scheduler,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
        if (
            const_expr(self.q_stage == 2)
            and warp_idx <= self.softmax1_warp_ids[-1]
            or (
                const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1]
            )
        ):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                descale_tensors=descale_tensors,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                tile_scheduler=tile_scheduler,
                base_softmax_scale=base_softmax_scale,
                sBias=sBias,
                bias_s2r_tiled_copy=bias_s2r_tiled_copy,
                pipeline_bias=pipeline_bias,
            )
            stage = Int32(0)
            softmax_loop(stage=stage, tStS=tStS)
            tmem_alloc_barrier.arrive()
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                None,  # Output and loader epilogues are handled by correction warps.
                None,
                None,  # Sinks remain in the external FP32 split-combine step.
                descale_tensors,
                gmem_tiled_copy_O,
                None,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
            )
            tmem_alloc_barrier.arrive()
        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mBias: Optional[cute.Tensor] = None,
        sBias: Optional[cute.Tensor] = None,
        tma_atom_bias: Optional[cute.CopyAtom] = None,
        pipeline_bias: Optional[pipeline.PipelineAsync] = None,
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        issue_kv_for_this_warp = (
            const_expr(not self.use_tma_KV or len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        )
        issue_q_for_this_warp = (
            const_expr(not self.use_tma_Q or len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        )
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        bias0_producer_state = pipeline_custom.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.bias_stage // self.q_stage
        )
        bias1_producer_state = pipeline_custom.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.bias_stage // self.q_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            head_idx_kv = head_idx
            load_SFQ_fn = None
            mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
            )
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
            )
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)
            tiler_gQ = (self.mma_tiler_qk[0] * self.q_stage, self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )
            load_Q = partial(
                self.load_Q,
                load_Q_fn,
                pipeline_q=pipeline_q,
                phase=q_producer_phase,
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )
            tKsSFK, tKgSFK = (None, None)
            tVsSFV, tVgSFV = (None, None)
            paged_kv_manager = None
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
                paged_kv_manager=None,
                sX=sK,
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
                paged_kv_manager=None,
                sX=sV,
            )
            if const_expr(tma_atom_bias is not None):
                mBias_cur = seqlen.offset_batch_Q(mBias, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                gBias = cute.local_tile(
                    mBias_cur, (self.bias_block_size, self.n_block_size), (None, None)
                )
                tBsBias, tBgBias = cpasync.tma_partition(
                    tma_atom_bias,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sBias, 0, 2),
                    cute.group_modes(gBias, 0, 2),
                )
                load_bias = partial(
                    self.load_bias,
                    tma_atom_bias,
                    tBgBias,
                    tBsBias,
                    pipeline_bias=pipeline_bias,
                )
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            if const_expr(tma_atom_bias is not None):
                _, n_block_max_abs = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits, absolute=True
                )
                _, n_block_max_abs0 = block_info.get_n_block_min_max(
                    seqlen,
                    self.q_stage * m_block,
                    split_idx,
                    num_splits,
                    half_tile_m=self.q_stage > 1,
                    absolute=True,
                )
                bias_idx_offset1 = n_block_max_abs - n_block_max
                bias_max_idx1 = self.bias_n_max - 1 - bias_idx_offset1
                bias_idx_offset0 = n_block_max_abs0 - n_block_max
                bias_max_idx0 = self.bias_n_max - 1 - max(bias_idx_offset0, 0)
                dummy_first_bias_load = bias_idx_offset0 < 0 and const_expr(
                    self.q_stage == 2
                )
                staggered_bias_loads = (
                    dummy_first_bias_load or bias_max_idx0 > bias_max_idx1
                ) and const_expr(self.q_stage == 2)
            if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                page_idx = mPageTable[batch_idx, n_block_first]
                if issue_kv_for_this_warp:
                    load_K(
                        block=n_block_max - 1,
                        producer_state=kv_producer_state,
                        page_idx=page_idx,
                    )
                if issue_q_for_this_warp:
                    load_Q(block=0, stage=0)
                if issue_kv_for_this_warp:
                    kv_producer_state.advance()
                if const_expr(tma_atom_bias is not None):
                    if const_expr(not self.is_split_kv) or bias_max_idx0 >= 0:
                        cute.arch.griddepcontrol_wait()
                        cute.arch.griddepcontrol_launch_dependents()
                    if issue_q_for_this_warp and (
                        const_expr(not self.is_split_kv) or bias_max_idx0 >= 0
                    ):
                        bias0_producer_state = load_bias(
                            m_block=self.q_stage * m_block + 0,
                            n_block=bias_max_idx0,
                            bias_producer_state=bias0_producer_state,
                            q_stage=0,
                        )
                q_producer_phase ^= 1
                if issue_kv_for_this_warp:
                    load_V(
                        block=n_block_max - 1,
                        producer_state=kv_producer_state,
                        page_idx=page_idx,
                    )
                    kv_producer_state.advance()
                if const_expr(tma_atom_bias is not None):
                    prologue_loads = min(bias_max_idx1, n_block_max - 1 - n_block_min)
                    tail_bias_load = (
                        staggered_bias_loads
                        and prologue_loads < n_block_max - 1 - n_block_min
                        and (prologue_loads >= 0)
                    )
                    prologue_loads = max(prologue_loads, 0)
                    for i in cutlass.range(prologue_loads, unroll=1):
                        n_block = n_block_max - 2 - i
                        page_idx = mPageTable[batch_idx, n_block]
                        if issue_kv_for_this_warp:
                            load_K(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            kv_producer_state.advance()
                        if issue_q_for_this_warp:
                            bias0_producer_state = load_bias(
                                m_block=self.q_stage * m_block + 0,
                                n_block=bias_max_idx0 - 1 - i + dummy_first_bias_load,
                                bias_producer_state=bias0_producer_state,
                                q_stage=0,
                            )
                        if issue_kv_for_this_warp:
                            load_V(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            kv_producer_state.advance()
                    if issue_q_for_this_warp and tail_bias_load:
                        bias0_producer_state = load_bias(
                            m_block=self.q_stage * m_block + 0,
                            n_block=bias_max_idx0
                            - 1
                            - prologue_loads
                            + dummy_first_bias_load,
                            bias_producer_state=bias0_producer_state,
                            q_stage=0,
                        )
                else:
                    prologue_loads = 0
                for i in cutlass.range(
                    prologue_loads, n_block_max - 1 - n_block_min, unroll=1
                ):
                    n_block = n_block_max - 2 - i
                    page_idx = mPageTable[batch_idx, n_block]
                    if issue_kv_for_this_warp:
                        load_K(
                            block=n_block,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )
                        kv_producer_state.advance()
                        load_V(
                            block=n_block,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )
                        kv_producer_state.advance()
            work_tile = tile_scheduler.advance_to_next_work()
        if issue_kv_for_this_warp:
            pipeline_kv.producer_tail(kv_producer_state)
        if issue_q_for_this_warp:
            pipeline_q.producer_acquire_w_index_phase(
                self.q_stage - 1, q_producer_phase
            )

    @cute.jit
    def load_bias(
        self,
        tma_atom_bias: cute.CopyAtom,
        tBgBias: cute.Tensor,
        tBsBias: cute.Tensor,
        pipeline_bias: pipeline.PipelineAsync,
        m_block: Int32,
        n_block: Int32,
        bias_producer_state: pipeline.PipelineState,
        q_stage: Int32,
    ):
        stage = bias_producer_state.index + q_stage
        phase = bias_producer_state.phase
        mbar_ptr = pipeline_bias.sync_object_full.get_barrier(stage)
        pipeline_bias.producer_acquire_w_index_phase(stage, phase)
        cute.copy(
            tma_atom_bias,
            tBgBias[None, m_block, n_block],
            tBsBias[None, stage],
            tma_bar_ptr=mbar_ptr,
        )
        bias_producer_state.advance()
        return bias_producer_state

    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        descale_tensors: Optional[DescaleTensors],
        thr_mma_qk: cute.ThrMma,
        tStS: cute.Tensor,
        sScale: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        head_divmod=None,
        tile_scheduler=None,
        base_softmax_scale: Optional[Float32] = None,
        sBias: Optional[cute.Tensor] = None,
        bias_s2r_tiled_copy: Optional[cute.TiledCopy] = None,
        pipeline_bias: Optional[pipeline.PipelineAsync] = None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        aux_tensors = aux_data.tensors
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tSAcc = tStS[(None, None), 0, 0, stage]
        tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_mma_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)
        tiled_tmem_load_red = None
        tStS_t2r_red = None
        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(
            tmem_store_scale_atom, tStScale
        ).get_slice(tidx)
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(
                    8 if const_expr(self.v_mma_dtype.width == 8) else 16
                )
            ),
            Float32,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)
        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)
        bias_si_consumer_state = pipeline_custom.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.bias_stage // self.q_stage
        )
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if const_expr(bias_s2r_tiled_copy is not None):
            bias_s2r_thr_copy = bias_s2r_tiled_copy.get_slice(tidx)
            tS2RsBias = bias_s2r_thr_copy.partition_S(sBias)
        else:
            bias_s2r_thr_copy = None
            tS2RsBias = None
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self._kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            _, n_block_max_abs_local = block_info.get_n_block_min_max(
                seqlen,
                self.q_stage * m_block + stage,
                split_idx,
                num_splits,
                half_tile_m=self.q_stage > 1,
                absolute=True,
            )
            bias_idx_offset = n_block_max_abs_local - n_block_max
            num_bias_loads = min(
                self.bias_n_max - bias_idx_offset, n_block_max - n_block_min
            )
            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_data=aux_data,
                vec_size=self.mask_vec_size,
            )
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )
            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
                )
            mask_mod = None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=mask_mod,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                **shared_mask_kwargs,
            )
            mask_fn_none = None
            qk_descale, _ = self._load_effective_descales(
                descale_tensors, batch_idx, kv_head_idx
            )
            max_offset = 8 if cutlass.const_expr(self.v_mma_dtype.width == 8) else 0
            softmax_scale_log2_eff = softmax_scale_log2
            softmax_scale_eff = base_softmax_scale * qk_descale
            rescale_threshold = (
                8.0
                if const_expr(self.v_mma_dtype.width == 16)
                else 4.0
                if const_expr(self.v_mma_dtype.width == 8)
                else 0.0
            )
            softmax = SoftmaxSm100.create(
                softmax_scale_log2_eff,
                rescale_threshold=rescale_threshold,
                softmax_scale=softmax_scale_eff,
                max_offset=max_offset,
            )
            softmax.reset()
            tile_block_count = n_block_max - n_block_min
            has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(0)
            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                tStS_t2r=tStS_t2r,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                tiled_tmem_load_red=tiled_tmem_load_red,
                tStS_t2r_red=tStS_t2r_red,
                stage=stage,
                bias_softmax_scale=softmax_scale_eff,
                tS2RsBias=tS2RsBias,
                bias_s2r_thr_copy=bias_s2r_thr_copy,
                pipeline_bias=pipeline_bias,
            )
            if const_expr(self.use_block_sparsity) or has_work:
                pipeline_sm_stats.producer_acquire_w_index_phase(
                    stage, sm_stats_producer_phase
                )
                sm_stats_producer_phase ^= 1
            if const_expr(not self.is_split_kv) or tile_block_count > Int32(0):
                if const_expr(self.has_bias) and (
                    const_expr(not self.is_split_kv) or num_bias_loads > 0
                ):
                    (
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        bias_si_consumer_state,
                    ) = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                        apply_bias=True,
                        bias_emu_off=Boolean(True),
                        bias_si_consumer_state=bias_si_consumer_state,
                    )
                else:
                    (
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                    ) = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                    )
                n_block_max -= 1
                n_block_min_causal_local_mask = max(
                    n_block_max + 1 - num_bias_loads, n_block_min
                )
                n_block_min_bias_emu_off = block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
                n_block_left_bias_emu_off = (
                    block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                )
                for n_tile in cutlass.range(
                    n_block_max - n_block_min_causal_local_mask, unroll=1
                ):
                    n_block = n_block_max - 1 - n_tile
                    (
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        bias_si_consumer_state,
                    ) = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        n_block,
                        mask_fn=None,
                        apply_bias=True,
                        bias_emu_off=Boolean(
                            n_block >= n_block_min_bias_emu_off
                            or n_block < n_block_left_bias_emu_off
                        ),
                        bias_si_consumer_state=bias_si_consumer_state,
                    )
                n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                n_block_min_before_local_mask = (
                    block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                )
                for n_tile in cutlass.range(
                    n_block_max - n_block_min_before_local_mask, unroll=1
                ):
                    n_block = n_block_max - n_tile - 1
                    (
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                    ) = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        n_block,
                    )
                if const_expr(
                    self.is_local and block_info.window_size_left is not None
                ):
                    n_block_max = cutlass.min(
                        n_block_max, n_block_min_before_local_mask
                    )
                    for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                        n_block = n_block_max - 1 - n_tile
                        (
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            s0_s1_sequence_phase,
                        ) = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            s0_s1_sequence_phase,
                            n_block,
                            mask_fn=partial(mask_fn, mask_seqlen=False),
                        )
                sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                sScale[
                    tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                ] = softmax.row_max[0]
                sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
            work_tile = tile_scheduler.advance_to_next_work()
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        bias_softmax_scale: Optional[Float32] = None,
        tS2RsBias: Optional[cute.Tensor] = None,
        bias_s2r_thr_copy: Optional[cute.CopyAtom] = None,
        apply_bias: bool = False,
        bias_emu_off: Optional[Boolean] = None,
        pipeline_bias: Optional[pipeline.PipelineAsync] = None,
        bias_si_consumer_state: Optional[pipeline.PipelineState] = None,
        tiled_tmem_load_red: Optional[cute.TiledCopy] = None,
        tStS_t2r_red: Optional[cute.Tensor] = None,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_mma_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tScS_shape = cta_qk_tiler
        tScP_shape = (tScS_shape[0], tilePlikeFP32)
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(
            thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype
        )
        use_red_max = const_expr(self.use_tmem_load_red_max and mask_fn is None)
        tile_row_max = None
        if const_expr(use_red_max):
            num_t2r = cute.size(tStS_t2r_red, mode=[2])
            tSrMax = cute.make_rmem_tensor(
                cute.make_layout((1, num_t2r)), self.qk_acc_dtype
            )
            for i in cutlass.range_constexpr(num_t2r):
                cute.copy_atom_call(
                    tiled_tmem_load_red,
                    tStS_t2r_red[None, 0, i],
                    (tSrS_t2r[None, 0, i], tSrMax[None, i]),
                )
            tile_row_max = softmax._compute_row_max(tSrMax.load())
        else:
            cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if const_expr(self.has_bias and apply_bias):
            tidx = cute.arch.thread_idx()[0] % (
                cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
            )
            bias_si_phase = bias_si_consumer_state.phase
            bias_si_stage = bias_si_consumer_state.index + stage
            pipeline_bias.consumer_wait_w_index_phase(bias_si_stage, bias_si_phase)
            tBrS = cute.make_tensor(
                tSrS_t2r.iterator,
                cute.make_fragment_like(tS2RsBias[None, None, None, 0].layout),
            )
            if const_expr(self.bias_block_size == 128) or tidx < self.bias_block_size:
                for i in cutlass.range_constexpr(cute.size(tS2RsBias.shape[2])):
                    tBrS_cur = tBrS[None, 0, i]
                    tS2RsBias_cur = tS2RsBias[None, 0, i, bias_si_stage]
                    tS2RrBias_cur = cute.make_fragment_like(tS2RsBias[None, 0, 0, 0])
                    cute.copy(bias_s2r_thr_copy, tS2RsBias_cur, tS2RrBias_cur)
                    assert cute.size(tBrS_cur.shape) % 2 == 0
                    for j in cutlass.range(
                        0, cute.size(tBrS_cur.shape), 2, unroll_full=True
                    ):
                        scores = (tBrS_cur[j], tBrS_cur[j + 1])
                        biases = (
                            tS2RrBias_cur[j].to(self.qk_acc_dtype),
                            tS2RrBias_cur[j + 1].to(self.qk_acc_dtype),
                        )
                        tBrS_cur[j], tBrS_cur[j + 1] = cute.arch.fma_packed_f32x2(
                            biases
                            if const_expr(self.rel_bias_scale_after_add)
                            else scores,
                            (bias_softmax_scale, bias_softmax_scale),
                            scores
                            if const_expr(self.rel_bias_scale_after_add)
                            else biases,
                        )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm100.Softmax) + stage,
                number_of_threads=128,
            )
            pipeline_bias.consumer_release_w_index(bias_si_stage)
        if const_expr(
            self.has_bias and (not apply_bias) and (not self.rel_bias_scale_after_add)
        ):
            assert cute.size(tSrS_t2r.shape) % 2 == 0
            for j in cutlass.range(0, cute.size(tSrS_t2r.shape), 2, unroll_full=True):
                tSrS_t2r[j], tSrS_t2r[j + 1] = cute.arch.mul_packed_f32x2(
                    (tSrS_t2r[j], tSrS_t2r[j + 1]),
                    (bias_softmax_scale, bias_softmax_scale),
                )
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape,
            Float32,
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.v_mma_dtype),
            tSrS_t2r.layout,
        )
        if const_expr(mask_fn is None and self.has_bias and apply_bias):
            if bias_emu_off:
                softmax.apply_exp2_convert(
                    tSrS_t2r,
                    tSrP_r2t,
                    ex2_emu_freq=0,
                    ex2_emu_start_frg=self.ex2_emu_start_frg,
                )
            else:
                softmax.apply_exp2_convert(
                    tSrS_t2r,
                    tSrP_r2t,
                    ex2_emu_freq=self.ex2_emu_freq,
                    ex2_emu_start_frg=self.ex2_emu_start_frg,
                )
        else:
            softmax.apply_exp2_convert(
                tSrS_t2r,
                tSrP_r2t,
                ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
                ex2_emu_start_frg=self.ex2_emu_start_frg,
            )
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive
                    // self.n_block_size
                )
                if const_expr(i + 1 == split_P_arrive_idx):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        if const_expr(bias_si_consumer_state is not None):
            if const_expr(self.has_bias and apply_bias):
                bias_si_consumer_state.advance()
            return (
                mma_si_consumer_phase ^ 1,
                sm_stats_producer_phase ^ 1,
                s0_s1_sequence_phase ^ 1,
                bias_si_consumer_state,
            )
        else:
            return (
                mma_si_consumer_phase ^ 1,
                sm_stats_producer_phase ^ 1,
                s0_s1_sequence_phase ^ 1,
            )
