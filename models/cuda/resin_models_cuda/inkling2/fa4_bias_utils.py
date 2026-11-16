# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
import enum
from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from flash_attn.cute.block_info import BlockInfo as FA4BlockInfo
from flash_attn.cute.seqlen_info import SeqlenInfoQK


@dataclass(frozen=True)
class BlockInfo(FA4BlockInfo):
    batch_invariant: cutlass.Constexpr[bool] = False

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
        half_tile_m: bool = False,
        absolute: bool = False,
        half_tile_n: bool = False,
    ) -> Tuple[Int32, Int32]:
        tile_m = self.tile_m // 2 if const_expr(half_tile_m) else self.tile_m
        tile_n = self.tile_n // 2 if const_expr(half_tile_n) else self.tile_n
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, tile_n)
        if const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            m_idx_max = (m_block + 1) * tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = (
                n_idx if const_expr(self.is_causal) else n_idx + self.window_size_right
            )
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, tile_n))
        n_block_min = 0
        if const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // tile_n, 0)
        if cutlass.const_expr(self.is_split_kv and not absolute):
            num_n_blocks_per_split = (
                Int32(0)
                if n_block_max <= n_block_min
                else (n_block_max - n_block_min + num_splits - 1) // num_splits
            )
            n_block_min = n_block_min + split_idx * num_n_blocks_per_split
            n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
        return n_block_min, n_block_max


class NamedBarrierFwdSm100(enum.IntEnum):
    # Preserve the upstream IDs, including gaps used by other warp stages.
    Epilogue = 1
    TmemPtr = 2
    SoftmaxStatsW0 = 3
    Softmax = 11
