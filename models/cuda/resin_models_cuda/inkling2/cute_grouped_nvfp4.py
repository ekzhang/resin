# Adapted from sgl-project/sglang e62ac6279646c686d3b328f58d9b5c2c2d5927dc,
# python/sglang/kernels/ops/inkling2/cute_grouped_nvfp4.py.
# Resin adds dependency-safe weight prefetch and bounded N64 token tiles.
# SFB stays tiled at128 rows, as in CUTLASS dense persistent GEMM.
# ruff: noqa
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Forked from FlashInfer 0.7.1 flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py
# (swap-AB, combine-fusion row store). Inkling2 routed NVFP4 experts: token rows are
# contiguous per expert (offsets), activation scales sit in 128-row blocks per expert,
# and each token row's FP32 accumulator is stored to its own output row.

from flashinfer.gemm.kernels.grouped_gemm_masked_blackwell import (
    Sm100BlockScaledPersistentDenseGemmKernel as _FlashInferGemm,
    make_fused_smem_layout_epi,
)

import functools
from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import (
    Int32,
    Int64,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo


class MaskedSchedulerParams:
    def __init__(
        self,
        masked_m: cute.Tensor,
        c: cute.Tensor,
        c_tiler: Tuple[int, int],
        schedule: cute.Tensor,
        *,
        loc=None,
        ip=None,
    ):
        self.schedule = schedule
        gc = cute.zipped_divide(c, tiler=c_tiler)
        problem_shape_ntile_mnl = gc[(0, (None, None, None))].shape
        self.masked_m = masked_m
        self.c = c
        self.c_tiler = c_tiler
        self.problem_shape_ntile_mnl = problem_shape_ntile_mnl
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.masked_m,
            self.c,
            self.c_tiler,
            self.schedule,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.masked_m,
                self.c,
                self.c_tiler,
                self.schedule,
            ],
            self._values_pos,
            strict=True,
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return MaskedSchedulerParams(*obj_list, loc=self._loc)

class MaskedScheduler:
    """CTA b executes logical tiles b, b+grid_size, ... in one-CTA clusters."""

    def __init__(self, params, num_persistent_clusters, current_work_linear_idx):
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.num_persistent_clusters) + extract_mlir_values(
            self._current_work_linear_idx
        )

    def __new_from_mlir_values__(self, values):
        assert len(values) == 2
        return MaskedScheduler(
            self.params,
            new_from_mlir_values(self.num_persistent_clusters, values[:1]),
            new_from_mlir_values(self._current_work_linear_idx, values[1:]),
        )

    @dsl_user_op
    @staticmethod
    def create(params, block_idx, grid_dim, *, loc=None, ip=None):
        return MaskedScheduler(
            params, cute.size(grid_dim, loc=loc, ip=ip), Int32(block_idx[2])
        )

    @cute.jit
    def _get_current_work_for_linear_idx(
        self,
        current_work_linear_idx: Int32,
    ) -> WorkTileInfo:
        num_tiles_n = self.params.problem_shape_ntile_mnl[0]
        # schedule[b] = (block << 16) | expert for 128-row token blocks, -1 past the end
        token_block = current_work_linear_idx // num_tiles_n
        is_valid = token_block < cute.size(self.params.schedule)
        expert = Int32(0)
        block = Int32(0)
        if is_valid:
            entry = self.params.schedule[token_block]
            is_valid = entry >= 0
            if is_valid:
                expert = entry & 0xFFFF
                block = entry >> 16
        cur_tile_coord = (
            current_work_linear_idx % num_tiles_n,
            block,
            expert,
        )
        return WorkTileInfo(cur_tile_coord, is_valid)

    @dsl_user_op
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self._get_current_work_for_linear_idx(self._current_work_linear_idx)

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        self._current_work_linear_idx += Int32(advance_count) * Int32(
            self.num_persistent_clusters
        )




def cp_bulk_store(dst_addr, src_addr, size, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_addr.toint(loc=loc, ip=ip).ir_value(),
            src_addr.toint(loc=loc, ip=ip).ir_value(),
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class Sm100BlockScaledPersistentDenseGemmKernel(_FlashInferGemm):
    """SM100 persistent block-scaled grouped GEMM (swap-AB, per-row combine store)."""

    def __init__(self, token_tile: int, sm_version: str, prefetch_weights: bool = False):
        super().__init__(16, (128, token_tile), (1, 1), sm_version,
                         is_combine_fusion=True, is_swap_ab=True)
        self.prefetch_weights = prefetch_weights

    def _setup_attributes(self):
        # Compute mma instruction shapes
        mma_inst_bits_k = 256
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_bits_k // self.a_dtype.width,
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mnk_sfb = (
            self.mma_inst_shape_mnk[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mnk[1], 128),
            self.mma_inst_shape_mnk[2],
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk[:2],
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb[:2],
        )

        # Compute mma/cluster/tile shapes
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mnk[0],
            self.mma_inst_shape_mnk[1],
            self.mma_inst_shape_mnk[2] * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mnk_sfb[0],
            self.mma_inst_shape_mnk_sfb[1],
            self.mma_inst_shape_mnk_sfb[2] * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma
        )

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = make_fused_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )
        return tiled_mma, tiled_mma_sfb

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        masked_m_tensor: cute.Tensor,
        offsets: cute.Tensor,
        schedule: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        tiled_mma, tiled_mma_sfb = self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            sfb_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            masked_m_tensor,  # add masked layout
            c_tensor,
            self.cta_tile_shape_mnk,
            max_active_clusters,
            schedule,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # Prefetch buffer for out_ptrs (one Int64 per M-row in the MMA tile)
            combine_out_ptrs: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int64,
                    self.mma_inst_shape_mnk[1],
                ],
                16,
            ]
            # Prefetch buffer for topk_weights (one Float32 per M-row in the MMA tile)
            combine_topk: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    self.mma_inst_shape_mnk[1],
                ],
                16,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c_tensor,
            offsets,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        offsets: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: MaskedSchedulerParams,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch tma desc
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Setup cta/thread coordinates
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_holding_buf = storage.tmem_holding_buf.ptr

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = (
            self.threads_per_warp
            * len(self.epilog_warp_id)
            * (2 if use_2cta_instrs else 1)
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        cute.arch.mbarrier_init_fence()

        # Setup smem tensor A/B/SFA/SFB/C
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        smem_out_ptrs = storage.combine_out_ptrs.get_tensor(
            cute.make_layout(self.mma_inst_shape_mnk[1])
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # Compute multicast mask for A/B/SFA/SFB buffer full
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None

        # Local_tile partition global tensors
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_block_cnt = cute.size(gA_mkl, mode=[3])

        # Partition global tensor for TiledMMA_A/B/C
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Partition global/shared tensor for TMA load A/B
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # Cluster wait before tensor memory alloc
        cute.arch.barrier(
            barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
        )
        if cutlass.const_expr(self.prefetch_weights):
            cute.arch.griddepcontrol_launch_dependents()
            if warp_idx != self.tma_warp_id:
                cute.arch.griddepcontrol_wait()
        else:
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # Specialized TMA load warp
        if warp_idx == self.tma_warp_id:
            # Persistent tile scheduling loop
            tile_sched = MaskedScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.get_current_work()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            dependency_pending = cutlass.Boolean(self.prefetch_weights)

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Slice to per mma tile index
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                expert = mma_tile_coord_mnl[2]
                row_start = offsets[expert]
                gB_expert = cute.local_tile(
                    cute.domain_offset((row_start, 0, 0), mB_nkl),
                    cute.slice_(self.mma_tiler, (0, None, None)),
                    (None, None, None),
                )
                _, tBgB_expert = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(thr_mma.partition_B(gB_expert), 0, 3),
                )
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB_expert[(None, mma_tile_coord_mnl[1], None, 0)]
                scale_tile = (row_start - offsets[0] + expert * 127) // 128

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[
                    (None, scale_tile + mma_tile_coord_mnl[1], None, 0)
                ]

                # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_block_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                # Tma load loop
                for k_block in cutlass.range(0, k_block_cnt, 1, unroll=1):  # noqa: B007
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    # Only immutable weights/scales may precede the producer
                    # wait. Routing offsets/schedule must already be ready.
                    # Keep the full A+B+SFA+SFB barrier byte count: weight
                    # arrivals alone must never release the MMA consumer.
                    if cutlass.const_expr(self.prefetch_weights):
                        if dependency_pending:
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, ab_producer_state.count)],
                                tAsSFA[(None, ab_producer_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                    ab_producer_state
                                ),
                                mcast_mask=sfa_full_mcast_mask,
                            )
                            cute.arch.griddepcontrol_wait()
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    if not dependency_pending:
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, ab_producer_state.count)],
                            tAsSFA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=sfa_full_mcast_mask,
                        )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )
                    dependency_pending = cutlass.Boolean(False)

                    # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt + k_block + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_block_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Inactive CTAs must honor the dependency even with no weight load.
            if dependency_pending:
                cute.arch.griddepcontrol_wait()
            # Wait A/B buffer empty
            ab_pipeline.producer_tail(ab_producer_state)

        # Specialized MMA warp
        if warp_idx == self.mma_warp_id:
            # Bar sync for retrieve tensor memory ptr from shared mem
            tmem_ptr_read_threads = self.threads_per_warp * len(
                (self.mma_warp_id, *self.epilog_warp_id)
            )
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            # Make accumulator tmem tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            # Partition for S2T copy of SFA/SFB
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            # Persistent tile scheduling loop
            tile_sched = MaskedScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.get_current_work()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_block = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_block_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                # Wait for accumulator buffer empty
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # Reset the ACCUMULATE field for each tile
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                # Mma mainloop
                for k_block in cutlass.range_constexpr(k_block_cnt):  # noqa: B007
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kphases = cute.size(tCrA, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (
                                None,
                                None,
                                kphase_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kphase_coord = (None, None, kphase_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kphase_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB[sf_kphase_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kphase_coord],
                                tCrB[kphase_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kphase
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_block = k_block + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_block_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                # Async arrive accumulator buffer full
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)
        # Specialized epilogue warps
        if warp_idx < self.mma_warp_id:
            # Alloc tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            # Bar sync for retrieve tensor memory ptr from shared memory
            tmem_ptr_read_threads = self.threads_per_warp * len(
                (self.mma_warp_id, *self.epilog_warp_id)
            )
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            # Retrieving tensor memory ptr and make accumulator tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Partition for epilogue
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            bSG_sC = self.epilog_gmem_copy_and_partition(sC)

            # Persistent tile scheduling loop
            tile_sched = MaskedScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.get_current_work()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            # Staging buffers rotate per stored subtile; skipped subtiles must not
            # advance it, or a buffer is rewritten while its bulk store still reads it.
            stored_subtiles = Int32(0)
            c_dtype_bytes: cutlass.Constexpr = self.c_dtype.width // 8
            epilog_threads = self.threads_per_warp * len(self.epilog_warp_id)

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx

                # Prefetch out_ptrs for this tile into SMEM
                # (overlaps GMEM loads with MMA pipeline stall)
                prefetch_m_base = cur_tile_coord[1] * self.mma_inst_shape_mnk[1]
                prefetch_num_m: cutlass.Constexpr = self.mma_inst_shape_mnk[1]
                prefetch_l_idx = cur_tile_coord[2]
                if tidx < prefetch_num_m:
                    prefetch_m_idx = prefetch_m_base + tidx
                    if prefetch_m_idx < tile_sched_params.masked_m[prefetch_l_idx]:
                        col_size = mC_mnl.shape[0]
                        out_row = offsets[prefetch_l_idx] + prefetch_m_idx
                        smem_out_ptrs[tidx] = (
                            mC_mnl.iterator.toint()
                            + Int64(out_row) * col_size * c_dtype_bytes
                        )
                    else:
                        smem_out_ptrs[tidx] = Int64(0)

                cute.arch.barrier(
                    barrier_id=self.epilog_sync_bar_id,
                    number_of_threads=epilog_threads,
                )

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                cur_tile_dim0_offset = cur_tile_coord[0] * self.mma_inst_shape_mnk[0]
                cur_tile_dim1_offset = cur_tile_coord[1] * self.mma_inst_shape_mnk[1]
                # Store accumulator to global memory in subtiles
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                # Token subtiles past the expert's rows hold nothing to store.
                active_subtiles = (
                    tile_sched_params.masked_m[cur_tile_coord[2]]
                    - cur_tile_dim1_offset
                    + epi_tile[1].shape
                    - 1
                ) // epi_tile[1].shape
                if active_subtiles > subtile_cnt:
                    active_subtiles = subtile_cnt
                for subtile_idx in cutlass.range(active_subtiles):
                    # Load accumulator from tensor memory buffer to register
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    # Convert to C type
                    acc_vec = tTR_rAcc.load()

                    acc_vec = acc_vec.to(self.c_dtype)
                    tRS_rC.store(acc_vec)

                    # Store C to shared memory
                    c_buffer = (stored_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    # TMA store C to global memory
                    src_smem_tensor = bSG_sC[(None, c_buffer)]
                    n_idx = cur_tile_dim0_offset
                    l_idx = cur_tile_coord[2]
                    with cute.arch.elect_one():
                        local_m = subtile_idx * epi_tile[1].shape + warp_idx * 8
                        smem_out_ptrs_ptr = cute.make_ptr(
                            Int64,
                            smem_out_ptrs.iterator.toint() + local_m * 8,
                            cute.AddressSpace.smem,
                        )
                        reg_out_ptrs = cute.make_tensor(smem_out_ptrs_ptr, (8,)).load()
                        # 8 elements per warp for a total of 32 elements per subtile
                        for idx_in_warp in cutlass.range_constexpr(8):
                            m_row = warp_idx * 8 + idx_in_warp
                            m_idx = (
                                cur_tile_dim1_offset
                                + subtile_idx * epi_tile[1].shape
                                + m_row
                            )
                            out_ptr = reg_out_ptrs[idx_in_warp]
                            if m_idx < tile_sched_params.masked_m[l_idx]:
                                src_ptr = cute.make_ptr(
                                    self.c_dtype,
                                    out_ptr + n_idx * c_dtype_bytes,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                )
                                cp_bulk_store(
                                    src_ptr,
                                    src_smem_tensor[((None, m_row),)].iterator,
                                    epi_tile[0].shape * c_dtype_bytes,
                                )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()

                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                stored_subtiles += active_subtiles

                # Async arrive accumulator buffer empty
                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Dealloc the tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            # Wait for C store complete
            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_wait_group(
                    0,
                    # Change `read` from True to False to also wait writes
                    read=False,
                )
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )




    def epilog_gmem_copy_and_partition(self, sC: cute.Tensor) -> cute.Tensor:
        """Group the C smem tile modes for the per-row bulk store."""
        return cute.group_modes(sC, 0, 2)

    def _compute_stages(self, tiled_mma):
        """Preserve the qualified one-CTA staging budget for N32/N64/N128."""
        a = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, 1
        )
        b = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, 1
        )
        sfa = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        sfb = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        ab_bytes = (
            cute.size_in_bytes(self.a_dtype, a)
            + cute.size_in_bytes(self.b_dtype, b)
            + cute.size_in_bytes(self.sf_dtype, sfa)
            + cute.size_in_bytes(self.sf_dtype, sfb)
        )
        c_layout = make_fused_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, 1)
        c_bytes = cute.size_in_bytes(self.c_dtype, c_layout)
        # Keep the original row-pointer/padding reservation as well as barriers;
        # changing this budget can change stage counts and kernel performance.
        reserved = 1024 + self.mma_tiler[1] * (8 + 4)
        ab_stages = (self.smem_capacity - (reserved + 2 * c_bytes)) // ab_bytes
        c_stages = 2 + (
            self.smem_capacity - ab_bytes * ab_stages - (reserved + 2 * c_bytes)
        ) // c_bytes
        return 2, ab_stages, c_stages

    @staticmethod
    def _compute_grid(
        masked_m_tensor: cute.Tensor,
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
        schedule: cute.Tensor,
    ) -> Tuple[MaskedSchedulerParams, Tuple[int, int, int]]:
        """Build the tile scheduler params and the persistent grid shape."""
        c_tiler = cute.slice_(cta_tile_shape_mnk, (None, None, 0))

        tile_sched_params = MaskedSchedulerParams(
            masked_m_tensor,
            c,
            c_tiler,
            schedule,
        )
        return tile_sched_params, (1, 1, max_active_clusters)


class _RoutedSwapGemm:
    def __init__(
        self,
        experts: int,
        n: int,
        k: int,
        sm_version: str,
        prefetch_weights: bool,
        token_tile: int,
    ):
        self.experts, self.n, self.k = experts, n, k
        self.gemm = Sm100BlockScaledPersistentDenseGemmKernel(
            token_tile=token_tile,
            sm_version=sm_version,
            prefetch_weights=prefetch_weights,
        )
        self.max_active_clusters = utils.HardwareInfo().get_max_active_clusters(1)

    @cute.jit
    def __call__(
        self,
        weight: cute.Tensor,
        weight_scales: cute.Tensor,
        x: cute.Tensor,
        x_scales: cute.Tensor,
        out: cute.Tensor,
        counts: cute.Tensor,
        offsets: cute.Tensor,
        schedule: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rows = x.shape[0]
        a = cute.make_tensor(
            cute.recast_ptr(weight.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_ordered_layout((self.n, self.k, self.experts), order=(1, 0, 2)),
        )
        b = cute.make_tensor(
            cute.recast_ptr(x.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2)),
        )
        sfa = cute.make_tensor(
            cute.recast_ptr(weight_scales.iterator, dtype=cutlass.Float8E4M3FN),
            cute.make_layout(1),
        )
        sfb = cute.make_tensor(
            cute.recast_ptr(x_scales.iterator, dtype=cutlass.Float8E4M3FN),
            cute.make_layout((x_scales.shape[0], self.k, 1)),
        )
        c = cute.make_tensor(
            out.iterator,
            cute.make_ordered_layout((self.n, rows, 1), order=(0, 1, 2)),
        )
        self.gemm(
            a,
            b,
            sfa,
            sfb,
            c,
            counts,
            offsets,
            schedule,
            self.max_active_clusters,
            stream,
        )


@functools.cache
def _compile(experts: int, n: int, k: int, prefetch_weights: bool, token_tile: int):
    # Warm the larger native fallback with the same prefetch contract once;
    # generic prefill uses a different contract and cannot populate this key.
    if token_tile in (32, 64):
        _compile(experts, n, k, prefetch_weights, 64 if token_tile == 32 else 128)
    major, minor = torch.cuda.get_device_capability()
    rows, scale_rows, out_rows = cute.sym_int64(), cute.sym_int64(), cute.sym_int64()
    blocks = cute.sym_int64()

    def fake(dtype, shape, align=16):
        return make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=align,
        )

    return cute.compile(
        _RoutedSwapGemm(
            experts, n, k, f"sm_{major}{minor}", prefetch_weights, token_tile
        ),
        fake(cutlass.Uint8, (experts, n, k // 2)),
        fake(cutlass.Uint8, (experts * ((n + 127) // 128 * 128) * (k // 16),)),
        fake(cutlass.Uint8, (rows, k // 2)),
        fake(cutlass.Uint8, (scale_rows, k // 16)),
        fake(cutlass.Float32, (out_rows, n)),
        fake(cutlass.Int32, (experts,), 4),
        fake(cutlass.Int32, (experts + 1,), 4),
        fake(cutlass.Int32, (blocks,), 4),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def routed_swap_matmul(
    values,
    x_scales,
    weight,
    scales,
    counts,
    offsets,
    schedule,
    out,
    *,
    prefetch_weights: bool = False,
    token_tile: int = 128,
    max_rows_per_expert: int | None = None,
):
    """out[r] = values[r] @ weight[e].T for the routed rows r of each local expert
    e (offsets[e] <= r < offsets[e] + counts[e]); values/x_scales/scales use the
    layouts of the CUTLASS swap path, schedule is route_metadata's 128-row block
    schedule. Other rows of out are left unset. Weight prefetch requires immutable
    weights/scales and routing metadata completed before the preceding producer;
    only activation values/scales may still be pending."""
    # N64 shares the128-row schedule/scale blocks only while each expert
    # fits its first tile. Bound total routed rows, including duplicate IDs.
    # N32 is internal first-tile use only: the caller must prove this
    # count bound (the target's unique top-k router uses its input-row count).
    if token_tile == 32 and (
        max_rows_per_expert is None or not 0 <= max_rows_per_expert <= 32
    ):
        raise ValueError("N32 requires an explicit per-expert row bound <=32")
    if token_tile not in (32, 64, 128) or (token_tile == 64 and values.shape[0] > 64):
        raise ValueError("N64 expert GEMM requires at most64 routed rows")
    experts, n, packed = weight.shape
    k = packed * 2
    scale_rows = (values.shape[0] + 127 * experts + 127) // 128 * 128
    _compile(experts, n, k, prefetch_weights, token_tile)(
        weight,
        scales.view(torch.uint8).view(-1),
        values,
        x_scales.view(torch.uint8)[: scale_rows * (k // 16)].view(scale_rows, k // 16),
        out,
        counts,
        offsets,
        schedule,
    )
