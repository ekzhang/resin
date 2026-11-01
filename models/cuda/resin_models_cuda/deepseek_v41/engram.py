# Adapted from DeepSeek-V4.1-Flash-1017, MIT license.
# https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/
# 1bae0e1088993b9b82de08f59de5be5a34e42170/inference/engram.py
"""Checkpoint-exact n-gram hashing and row-sharded Engram residuals."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from math import isqrt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_compressed_token_map(tokenizer):
    """Collapse normalized spellings, retaining partial UTF-8 tokens by raw form."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    keys, lookup = {}, []
    for token_id in range(backend.get_vocab_size(with_added_tokens=True)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = (
            backend.id_to_token(token_id)
            if "\ufffd" in text
            else normalizer.normalize_str(text) or text
        )
        lookup.append(keys.setdefault(key, len(keys)))
    return lookup, len(keys)


def compute_hash_multipliers(layer_ids, max_ngram_size, vocab_size):
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    return torch.tensor(
        np.stack(
            [
                np.random.default_rng(10007 * layer).integers(
                    0, bound, size=max_ngram_size, dtype=np.int64
                )
                * 2
                + 1
                for layer in layer_ids
            ]
        )
    )


def _next_prime(start):
    candidate = start + 1
    while candidate < 2 or any(
        candidate % divisor == 0 for divisor in range(2, isqrt(candidate) + 1)
    ):
        candidate += 1
    return candidate


@dataclass(frozen=True)
class EngramLayout:
    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]

    @classmethod
    def from_args(cls, args):
        layers = tuple(args.engram_layer_ids)
        ngram, heads = args.engram_max_ngram_size, args.engram_n_heads
        if not layers or ngram < 2 or heads < 1 or args.engram_vocab_size < 2:
            raise ValueError(
                "Engram requires layers, n-grams >= 2 and positive buckets"
            )
        if len(args.engram_num_embeddings) != len(layers):
            raise ValueError("Engram table counts must match layer IDs")
        # Official allocation restarts each search at vocab_size and skips every
        # previously used prime. A single increasing cursor produces that order.
        cursor, primes = args.engram_vocab_size - 1, []
        for _ in layers:
            buckets = []
            for _ in range(ngram - 1):
                row = []
                for _ in range(heads):
                    cursor = _next_prime(cursor)
                    row.append(cursor)
                buckets.append(tuple(row))
            primes.append(tuple(buckets))
        if any(
            count < sum(sum(row) for row in layer)
            for count, layer in zip(args.engram_num_embeddings, primes, strict=True)
        ):
            raise ValueError("Engram table has fewer rows than its hash buckets")
        return cls(
            layers,
            ngram,
            heads,
            args.engram_head_dim,
            tuple(args.engram_num_embeddings),
            tuple(primes),
        )


class NgramHash(nn.Module):
    """Hash token history stored in the Engine's paged, rollback-safe token pool."""

    DEAD = -1
    primes: torch.Tensor
    offsets: torch.Tensor
    multipliers: torch.Tensor
    token_map: torch.Tensor

    def __init__(self, args, tokenizer):
        super().__init__()
        self.layout = EngramLayout.from_args(args)
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != args.engram_compressed_vocab_size:
            raise ValueError(
                f"Engram compressed vocabulary {vocab_size} != checkpoint "
                f"{args.engram_compressed_vocab_size}"
            )
        self.pad_id = token_map[args.engram_pad_id]
        primes = torch.tensor(self.layout.primes, dtype=torch.int64)
        flat = primes.flatten(1)
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", flat.cumsum(-1) - flat, persistent=False)
        self.register_buffer(
            "multipliers",
            compute_hash_multipliers(
                self.layout.layer_ids, self.layout.max_ngram_size, vocab_size
            ),
            persistent=False,
        )
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)

    @torch.inference_mode()
    def forward(self, input_ids, positions, cache, token_mask=None):
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = compressed.masked_fill(~token_mask, self.DEAD)
        # Write this invocation's entire prefix before looking back. Rejected
        # draft slots are overwritten at their logical positions on the next call.
        cache.token_values.index_copy_(
            0, cache.token_rows(positions).flatten(), compressed.flatten()
        )
        lookback = positions[..., None] - torch.arange(
            self.layout.max_ngram_size, device=input_ids.device
        )
        source = cache.token_values[cache.token_rows(lookback.clamp_min(0))]
        blocked = ((lookback < 0) | (source == self.DEAD)).cummax(-1).values
        tokens = source.masked_fill(blocked, self.pad_id)
        products = tokens.unsqueeze(-2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for shift in range(1, self.layout.max_ngram_size):
            rolling = rolling ^ products[..., shift]
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, shift - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class ParallelEngramEmbedding(nn.Module):
    """World-sharded FP8 rows, independent of attention or expert ownership."""

    def __init__(self, rows, dim, lookup=None):
        super().__init__()
        if dim % 32:
            raise ValueError("Engram head dimension must be divisible by block32")
        self.lookup = lookup
        group = None if lookup is None else lookup.group
        width = (rows + (1 if group is None else group.size) - 1) // (
            1 if group is None else group.size
        )
        rank = 0 if group is None else group.local_rank
        self.start, self.stop = (
            rank * width,
            (rank + 1) * width,
        )
        self.weight = nn.Parameter(torch.empty(width, dim, dtype=torch.float8_e4m3fn))
        self.scale = nn.Parameter(
            torch.empty(width, dim // 32, dtype=torch.float8_e8m0fnu)
        )

    def forward(self, indices):
        return (
            self.local(indices)
            if self.lookup is None
            else self.lookup(indices, self.local)
        )

    def local(self, indices, output=None):
        if indices.is_cuda:
            from resin_models_cuda.deepseek_v41.embedding_cuda import fp8_embedding

            return fp8_embedding(
                indices, self.weight, self.scale, self.start, self.stop, output
            )
        mask = (indices < self.start) | (indices >= self.stop)
        local = (indices - self.start).masked_fill(mask, 0)
        values = F.embedding(local, self.weight).float().unflatten(-1, (-1, 32))
        scales = F.embedding(local, self.scale).float().unsqueeze(-1)
        value = (
            (values * scales).flatten(-2).bfloat16().masked_fill(mask.unsqueeze(-1), 0)
        )
        if output is not None:
            output.copy_(value)
            return output
        return value

    def forward_empty(self, columns):
        return self(self.weight.new_empty((0, columns), dtype=torch.int64))


class Engram(nn.Module):
    def __init__(self, args, layer, *, lookup=None):
        super().__init__()
        from resin_models_cuda.deepseek_v41.linear import Linear

        layout = EngramLayout.from_args(args)
        self.layer_hash_index = layout.layer_ids.index(layer)
        self.dim, self.hc_mult, self.eps = args.dim, args.hc_mult, args.norm_eps
        self.embed = ParallelEngramEmbedding(
            layout.num_embeddings[self.layer_hash_index], layout.head_dim, lookup
        )
        self.hash_columns = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = Linear(
            (layout.max_ngram_size - 1) * layout.n_heads * layout.head_dim,
            args.dim * (args.hc_mult + 1),
        )
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.register_buffer("_gate_weight", None, persistent=False)
        self.prefill_projection: Callable | None = None

    def prepare_gate(self):
        """Fuse loaded RMS weights once, as in DeepInfer's Engram preparation."""
        self._gate_weight = (
            _gate_kernels().fused_weight(self.q_weight, self.k_weight)
            if self.q_weight.is_cuda
            else self.q_weight.float() * self.k_weight.float()
        )

    def forward(
        self, hidden, hash_ids=None, token_mask=None, *, embed_hook=None, prefill=False
    ):
        embedding = self.embed(hash_ids) if embed_hook is None else embed_hook()
        values = embedding.flatten(-2)
        if prefill and self.prefill_projection is not None:
            # Only the independent projection rows are padded. Its borrowed
            # graph output is consumed by the gate before another replay.
            projection = self.prefill_projection(values.reshape(-1, values.shape[-1]))
            projection = projection.view(*values.shape[:-1], projection.shape[-1])
        else:
            projection = self.wkv(values)
        if hidden.is_cuda:
            if self._gate_weight is None:
                raise RuntimeError("prepare Engram gate weights before execution")
            output = _gate_native(hidden, projection, self._gate_weight, self.eps)
            return (
                output
                if token_mask is None
                else torch.where(token_mask[..., None, None], output, hidden)
            )
        key, value = projection.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        stream = hidden.float()
        rstd = torch.rsqrt(stream.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        weight = self.q_weight.float() * self.k_weight.float()
        dot = (stream * weight * key).sum(-1)
        dot = dot * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (stream + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
            hidden.dtype
        )


@cache
def _gate_kernels():
    # Load the shared TileLang compatibility adapter before TileKernels 1.0.
    import_module("resin_models_cuda.deepseek_v41.tilelang_kernels")
    return import_module("tile_kernels.engram")


def _gate_native(hidden, projection, weight, eps):
    """Adapt DeepInfer's fused gate to TileKernels 1.0's separate K/V ABI.

    DeepInfer 774ce244's v4_modules/engram.py uses the same inference equation
    (Apache-2.0). This dependency version returns an out-of-place output and
    leaves optional image masking to its caller.
    """
    hc, dim = hidden.shape[-2:]
    kv = projection.reshape(-1, hc + 1, dim)
    output, *_ = _gate_kernels().engram_gate_fwd(
        hidden.reshape(-1, hc, dim),
        kv[:, :hc],
        kv[:, hc],
        weight,
        eps,
        1e-6,
        save_for_backward=False,
    )
    return output.view_as(hidden)


class EngramRuntime:
    """Fetch every layer's table rows, optionally overlapping model computation.

    The fetch/hook boundary follows DeepInfer v4_modules/engram.py at
    774ce244b87cc0eb8d4b94ab9f5de2ab22049f01 (Apache-2.0). This transport
    uses NCCL owner lookups and does not require DeepEP's RDMA devices.
    """

    def __init__(self, layers, *, overlap: bool = True):
        self.layers = tuple(layers.items())
        self.overlap = overlap
        self.lookup = self.layers[0][1].embed.lookup if self.layers else None
        if any(layer.embed.lookup is not self.lookup for _, layer in self.layers):
            raise ValueError("Engram layers must share one lookup transport")

    def fetch(self, hashes):
        """All ranks issue the same table order, including empty participants."""
        requests = [
            (
                layer.embed.weight.new_empty((0, layer.hash_columns), dtype=torch.int64)
                if hashes is None
                else hashes[..., layer.layer_hash_index, :],
                layer.embed.local,
            )
            for _, layer in self.layers
        ]
        hooks = (
            self.lookup.prefetch(requests)
            if self.lookup is not None
            else [
                lambda indices=indices, local=local: local(indices)
                for indices, local in requests
            ]
        )
        if not self.overlap:
            # These hooks add device-stream dependencies, without a host sync.
            values = [hook() for hook in hooks]
            hooks = [lambda value=value: value for value in values]
        return {
            layer: hook for (layer, _), hook in zip(self.layers, hooks, strict=True)
        }
