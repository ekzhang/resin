# Adapted from DeepSeek-V4-Flash-0731 (MIT).
"""Vocabulary sharding, rotary positions, and norms used by DeepSeek V4.1."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout


def _default_tensor_parallel() -> TensorParallelContext:
    """Compatibility for callers using the default process group as TP."""

    size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    return TensorParallelContext(
        TensorParallelLayout(rank=rank, size=size),
        dist.group.WORLD if size > 1 else None,
    )


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension. Each attention-TP rank holds one contiguous vocabulary shard.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        tensor_parallel: TensorParallelContext | None = None,
    ):
        super().__init__()
        self.tensor_parallel = tensor_parallel or _default_tensor_parallel()
        size, rank = self.tensor_parallel.size, self.tensor_parallel.rank
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % size == 0, (
            f"Vocabulary size must be divisible by attention TP (size={size})"
        )
        self.part_vocab_size = vocab_size // size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        self.allreduce: Callable[[torch.Tensor], torch.Tensor] = (
            self.tensor_parallel.sum
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel.enabled:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if self.tensor_parallel.enabled:
            y[mask] = 0
        return self.allreduce(y)


@lru_cache(2)
def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:
    """Precomputes complex exponentials for rotary embeddings with YaRN scaling.
    When original_seq_len > 0, applies frequency interpolation with a smooth
    linear ramp between beta_fast and beta_slow correction ranges."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Applies rotary positional embeddings in-place. Uses conjugate for inverse (de-rotation)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    row_aligned = x.size(1) == 1 and freqs_cis.size(0) == x.size(0)
    if freqs_cis.ndim == 3:
        if x.ndim == 4:
            freqs_cis = freqs_cis.unsqueeze(2)
    elif x.ndim == 3:
        freqs_cis = freqs_cis.view(
            x.size(0) if row_aligned else 1, x.size(1), x.size(-1)
        )
    else:
        freqs_cis = freqs_cis.view(
            x.size(0) if row_aligned else 1, x.size(1), 1, x.size(-1)
        )
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


class ParallelHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        tensor_parallel: TensorParallelContext | None = None,
    ):
        super().__init__()
        self.tensor_parallel = tensor_parallel or _default_tensor_parallel()
        part_vocab_size = vocab_size // self.tensor_parallel.size
        # The checkpoint stores lm_head in BF16; FP32 accumulation keeps the
        # final sampling logits stable across tensor-parallel ranks.
        self.weight = nn.Parameter(
            torch.empty(part_vocab_size, dim, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor, *, full_logits: bool = False):
        if not full_logits:
            x = x[:, -1]
        logits = F.linear(x.float(), self.weight)
        if self.tensor_parallel.enabled:
            all_logits = [
                torch.empty_like(logits) for _ in range(self.tensor_parallel.size)
            ]
            dist.all_gather(
                all_logits, logits, group=self.tensor_parallel.process_group
            )
            logits = torch.cat(all_logits, dim=-1)
        return logits
