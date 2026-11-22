"""Inkling 2 target-only runner on Apple Silicon over Engine-owned MLX slabs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
import torch
from torch.profiler import record_function

from resin import BatchPhase, RunnerConfig
from resin.backend.capacity import plan_cache_capacity
from resin.backend.plans import PagedCachePlan
from resin.backend.sampling import PromptScores, format_output, prompt_scores
from resin.backend.topology import ParallelismCapabilities
from resin.models.inkling2.config import Inkling2Config
from resin_models_mlx.inkling2.attention import dense_attention
from resin_models_mlx.inkling2.cache import (
    GLOBAL_GROUP,
    LOCAL_GROUP,
    PAGE_TOKENS,
    STATE_GROUP,
    CacheArena,
    CacheGeometry,
    History,
)
from resin_models_mlx.inkling2.checkpoint import load_model
from resin_models_mlx.inkling2.decode_attention import decode_attention
from resin_models_mlx.inkling2.model import Inkling2Model, Inkling2Weights, logits
from resin_models_mlx.live_state import LiveStates
from resin_models_mlx.paged_attention import decode_metadata
from resin_models_mlx.sampling import MlxSampler

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput, ResourceBinding

_WORKSPACE_BYTES = 4 * 1024**3


def _to_torch(array: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(array.astype(mx.float32)))


class _StepStore:
    """Attention over one step's retained history plus its own new keys.

    The step's keys and values are kept for the runner to write after every
    reader of the arena view has been ordered.
    """

    def __init__(
        self,
        arena: CacheArena,
        plan: PagedCachePlan,
        rows: np.ndarray,
        contexts: np.ndarray,
        positions: mx.array,
        window: int,
    ) -> None:
        geometry = arena.geometry
        self.positions = positions
        self.sequence_ids = plan.batch.sequence_ids[rows]
        self.contexts = contexts
        self.history: dict[int, History | None] = {
            GLOBAL_GROUP: arena.history(plan, GLOBAL_GROUP, rows, contexts),
            LOCAL_GROUP: arena.history(
                plan, LOCAL_GROUP, rows, contexts, earliest=contexts - (window - 1)
            ),
        }
        self.keys: dict[int, list[mx.array | None]] = {
            group: [None] * geometry.layers(group)
            for group in (GLOBAL_GROUP, LOCAL_GROUP)
        }
        self.values: dict[int, list[mx.array | None]] = {
            group: [None] * geometry.layers(group)
            for group in (GLOBAL_GROUP, LOCAL_GROUP)
        }
        # Decode reads pages in place; prefill gathers history per visit.
        self.paged: dict[int, tuple[mx.array, mx.array, mx.array, int, int]] = {}
        if positions.shape[1] == 1:
            units = arena.units()
            for group in (GLOBAL_GROUP, LOCAL_GROUP):
                history = self.history[group]
                firsts = (
                    np.maximum(contexts - (window - 1), 0)
                    if group == LOCAL_GROUP
                    else np.zeros_like(contexts)
                )
                if history is None:
                    pages = np.zeros((len(rows), 1), dtype=np.int32)
                    starts = firsts // geometry.page_tokens
                    stride = geometry.blocks_per_slab(group)
                else:
                    pages = history.bases
                    starts = history.starts
                    stride = history.layer_stride
                meta = np.stack([contexts, firsts, starts], axis=1)
                longest = int(np.max(contexts + 1 - firsts))
                self.paged[group] = (
                    units,
                    decode_metadata(pages),
                    decode_metadata(meta),
                    longest,
                    stride,
                )

    def attend(self, local, slot, q, k, v, bias, scale, window):
        group = LOCAL_GROUP if local else GLOBAL_GROUP
        self.keys[group][slot] = k
        self.values[group][slot] = v
        if group in self.paged:
            units, pages, meta, longest, stride = self.paged[group]
            output, lse = decode_attention(
                q[:, 0],
                k[:, 0],
                v[:, 0],
                units,
                pages,
                meta,
                bias[:, 0],
                slot * stride,
                scale,
                longest,
            )
            return output[:, None], lse[:, None]
        batch, tokens = self.positions.shape
        current_valid = mx.ones((batch, tokens), dtype=mx.bool_)
        history = self.history[group]
        contiguous = None
        if history is None:
            keys, values = k, v
            key_positions, key_valid = self.positions, current_valid
            if batch == 1:
                context = int(self.contexts[0])
                contiguous = (context, context)
        else:
            hk, hv, hp, hvalid = history.gather(slot)
            if batch == 1:
                # One prefill row: drop the last page's unwritten tail so the
                # retained history runs contiguously into the chunk's keys.
                first = int(history.starts[0]) * PAGE_TOKENS
                count = int(self.contexts[0]) - first
                hk, hv, hp, hvalid = (
                    hk[:, :count],
                    hv[:, :count],
                    hp[:, :count],
                    hvalid[:, :count],
                )
                contiguous = (first, int(self.contexts[0]))
            keys = mx.concatenate([hk, k], axis=1)
            values = mx.concatenate([hv, v], axis=1)
            key_positions = mx.concatenate([hp, self.positions], axis=1)
            key_valid = mx.concatenate([hvalid, current_valid], axis=1)
        return dense_attention(
            q,
            keys,
            values,
            key_positions,
            key_valid,
            self.positions,
            bias,
            scale,
            window if local else None,
            contiguous,
        )

    def written(self, group: int) -> tuple[list[mx.array], list[mx.array]]:
        keys = [x for x in self.keys[group] if x is not None]
        values = [x for x in self.values[group] if x is not None]
        if len(keys) != len(self.keys[group]) or len(values) != len(self.values[group]):
            raise RuntimeError("an Inkling 2 visit did not produce its keys and values")
        return keys, values


@dataclass(slots=True)
class _Launch:
    """A step's outputs, retained through its GPU-ordered cache writes."""

    plan: PagedCachePlan
    rows: np.ndarray
    tokens: np.ndarray
    contexts: np.ndarray
    store: _StepStore
    hidden: mx.array
    state: mx.array
    last: mx.array
    receipts: tuple[int, ...] | None = field(default=None)

    @property
    def reads(self) -> tuple[mx.array, ...]:
        """Every output computed from the arena, including the held state."""
        keys = [x for group in self.store.keys.values() for x in group if x is not None]
        values = [
            x for group in self.store.values.values() for x in group if x is not None
        ]
        return (self.last, self.hidden, self.state, *keys, *values)


@dataclass(slots=True)
class _Completion:
    """One overlapped decode step, collected after the next one is launched."""

    _runner: Inkling2MlxRunner
    _plan: BatchPlan
    _launch: _Launch
    _tokens: mx.array
    _continues: bool
    _result: ForwardOutput | None = field(default=None, init=False)

    @property
    def feedback(self) -> mx.array | None:
        # None drains the Engine, so a step past max_seq_len fails in forward
        # rather than in an overlapped enqueue, which would poison the Engine.
        return self._tokens if self._continues else None

    def result(self) -> ForwardOutput:
        if self._result is None:
            if self._launch.receipts is None:
                # No continuation followed, so this step still owns its writes.
                if self._runner._pending is not self._launch:
                    raise RuntimeError("MLX step lost its pending cache writes")
                self._runner._write_pending()
            tokens = np.array(self._tokens).astype(np.int64)
            receipts = self._launch.receipts
            assert receipts is not None
            self._result = format_output(
                None, tokens, self._plan
            ).with_state_checkpoints(receipts)
        return self._result


class Inkling2MlxRunner:
    """Inkling 2 target with paged global/local attention and live conv history."""

    page_size = 128
    parallelism_capabilities = ParallelismCapabilities()

    def __init__(
        self,
        model: str | Path | tuple[Inkling2Config, Inkling2Weights],
        *,
        memory_utilization: float = 0.85,
        cache_slab_capacity: int | None = None,
        max_seq_len: int | None = None,
        max_batch_size: int = 8,
        prefill_chunk_size: int = 8192,
        workspace_bytes: int = _WORKSPACE_BYTES,
    ) -> None:
        if workspace_bytes < 0:
            raise ValueError("workspace_bytes must be non-negative")
        if isinstance(model, tuple):
            config, weights = model
            self.model_path = None
        else:
            self.model_path = Path(model)
            config, weights = load_model(self.model_path)
        self.config = config
        self._model = Inkling2Model(config, weights)
        limit_positions = config.max_position_embeddings
        self.max_seq_len = limit_positions if max_seq_len is None else max_seq_len
        if not 0 < self.max_seq_len <= limit_positions:
            raise ValueError(
                f"max_seq_len must be between 1 and the checkpoint's {limit_positions} positions"
            )
        self.prefill_chunk_size = prefill_chunk_size
        self.benchmark_metadata = {
            "quant": "checkpoint-nvfp4-mxfp8-experts",
            "weight_source": "prequantized_checkpoint",
            # Prefill chunks of 512 or more tokens instead run exactly
            # dequantized FP32 GEMMs per expert segment (model.SEGMENTED_ROUTES).
            "expert_backend": "mlx-gather-qmm",
            "graphs": {"prefill": "disabled", "decode": "disabled"},
        }
        # Logits cover the real vocabulary; embedding rows beyond it are padding.
        self.vocab_size = config.unpadded_vocab_size
        self.eos_token_id = config.eos_token_id
        self.cache_geometry = CacheGeometry.from_config(config, prefill_chunk_size)
        geometry = self.cache_geometry
        device = mx.device_info()
        limit = int(device["max_recommended_working_set_size"])
        self.cache_capacity_plan = plan_cache_capacity(
            limit,
            max(1, limit - mx.get_active_memory()),
            memory_utilization,
            geometry.slab_bytes,
            # Live convolution history is held for every running request, and
            # a step's output state exists beside its input.
            overhead_bytes=workspace_bytes + 2 * max_batch_size * geometry.state_bytes,
        )
        buffer_slabs = int(device["max_buffer_length"]) // geometry.slab_bytes
        # Room for every running request at full context, and as much again
        # for reusable prefixes; beyond that, memory is better left unwired.
        wanted = 2 * max_batch_size * geometry.sequence_slabs(self.max_seq_len) + 16
        slab_capacity = min(
            self.cache_capacity_plan.parent_capacity, buffer_slabs - 1, wanted
        )
        if cache_slab_capacity is not None:
            slab_capacity = min(slab_capacity, cache_slab_capacity)
        if slab_capacity < geometry.sequence_slabs(self.max_seq_len):
            raise ValueError(
                f"the Inkling 2 cache holds {slab_capacity} slabs, fewer than one "
                f"{self.max_seq_len}-token request needs; lower max_seq_len"
            )
        self.cache_layout = geometry.layout(slab_capacity)
        self._cache = CacheArena(self.cache_layout, geometry)
        self.runner_config = RunnerConfig(
            max_batch_tokens=prefill_chunk_size,
            prefill_chunk_size=prefill_chunk_size,
            continuation_prefill=True,
            cache_layout=self.cache_layout,
        )
        self._sampler = MlxSampler()
        self._states = LiveStates(self._cache, self._model.initial_state, STATE_GROUP)
        self._pending: _Launch | None = None

    def bind_resources(self, resources: ResourceBinding) -> None:
        self._cache.binding.bind_resources(resources)

    def release(self, sequence_ids: list[int]) -> None:
        # A pending step would hold the released rows' state again.
        self._write_pending()
        self._states.release(sequence_ids)

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        values = logits(
            mx.array(hidden.numpy()).astype(mx.bfloat16),
            self._model.weights,
            self.config,
        )
        return _to_torch(values)

    def _launch(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        tokens: np.ndarray,
        token_ids: mx.array | None = None,
    ) -> _Launch:
        """Build the step for ``tokens``, flattened plan indices ``[len(rows), width]``."""
        batch = plan.batch
        contexts = batch.context_lengths[rows].astype(np.int64)
        if token_ids is None:
            token_ids = mx.array(batch.token_ids[tokens].astype(np.int32))
        positions = contexts[:, None] + np.arange(tokens.shape[1])[None]
        positions_mx = mx.array(positions.astype(np.int32))
        store = _StepStore(
            self._cache,
            plan,
            rows,
            contexts,
            positions_mx,
            self.config.sliding_window_size,
        )
        hidden, state = self._model.hidden(
            token_ids.reshape(tokens.shape),
            positions_mx,
            self._states.gather(plan, rows),
            store,
        )
        last = logits(hidden[:, -1], self._model.weights, self.config)
        return _Launch(plan, rows, tokens, contexts, store, hidden, state, last)

    def _write(self, launch: _Launch) -> None:
        """Order writes after this graph's readers and hold its live state."""
        self._cache.order_after(launch.reads)
        flat = launch.tokens.reshape(-1)
        geometry = self.cache_geometry
        for group in (GLOBAL_GROUP, LOCAL_GROUP):
            keys, values = launch.store.written(group)
            width = (-1, geometry.kv_heads, geometry.head_dim)
            self._cache.write_tokens(
                launch.plan,
                group,
                [key.reshape(width) for key in keys],
                [value.reshape(width) for value in values],
                flat,
            )
        launch.receipts = self._states.commit(
            launch.plan,
            launch.rows,
            launch.contexts + launch.tokens.shape[1],
            launch.state,
        )

    def _write_pending(self) -> None:
        """Drain pending cache work before a terminal/ownership transition."""
        launch, self._pending = self._pending, None
        if launch is None:
            return
        self._write(launch)
        # No continuation retains the lease here: wait for snapshots and KV
        # writes, not only the sampled-token event.
        mx.eval(self._cache.epoch)

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self._write_pending()
        cache_plan = self._cache_plan(plan)
        batch = cache_plan.batch
        prefill = plan.phase == BatchPhase.PREFILL
        rows = np.arange(batch.num_sequences)
        groups = rows[:, None] if prefill else (rows,)
        last_logits = []
        prompt_rows = []
        receipts: list[int] = []
        for rows in groups:
            if prefill:
                selected = batch.token_slice(int(rows[0]))
                tokens = np.arange(selected.start, selected.stop)[None]
            else:
                tokens = rows[:, None]
            with record_function("resin.mlx.forward"):
                launch = self._launch(cache_plan, rows, tokens)
            with record_function("resin.mlx.cache_write"):
                self._write(launch)
                mx.async_eval(self._cache.epoch, *launch.reads)
            last_logits.append(launch.last)
            assert launch.receipts is not None
            receipts += launch.receipts
            if prefill and plan.needs_prompt_logprobs:
                scores = prompt_scores(
                    plan,
                    _to_torch(launch.hidden[0]),
                    self._project,
                    token_start=int(tokens[0, 0]),
                )
                if scores is not None:
                    prompt_rows.append(scores)
        # A synchronous result can release the lease immediately: wait for
        # every issued cache write, not just the sampled-token event.
        combined = (
            last_logits[0] if len(last_logits) == 1 else mx.concatenate(last_logits)
        )
        combined = mx.depends(combined, self._cache.epoch)
        with record_function("resin.mlx.sample"):
            sampled = self._sampler(combined, plan)
        return format_output(
            _to_torch(combined) if plan.needs_logprobs else None,
            sampled,
            plan,
            prompt_logprobs=PromptScores.concat(prompt_rows),
        ).with_state_checkpoints(receipts)

    def _cache_plan(self, plan: BatchPlan) -> PagedCachePlan:
        cache_plan = PagedCachePlan.from_plan(plan, self.cache_layout)
        # Validate before any cache write: rejected plans must leave no state.
        if any(media is not None for media in plan.media):
            raise ValueError("the Inkling 2 MLX runner serves text only")
        if np.any(cache_plan.token_ids >= self.vocab_size):
            raise ValueError("token id is out of range for Inkling 2")
        if int(np.max(cache_plan.batch.post_lengths)) > self.max_seq_len:
            raise ValueError(
                f"sequence exceeds this runner's {self.max_seq_len}-token context"
            )
        return cache_plan

    def enqueue(
        self, plan: BatchPlan, *, feedback: mx.array | None = None
    ) -> _Completion | ForwardOutput:
        """Launch a decode step while the previous one still runs on the GPU."""
        if plan.phase != BatchPhase.DECODE or plan.needs_logprobs:
            if feedback is not None:
                raise RuntimeError("an MLX continuation requires unscored decode")
            return self.forward(plan)
        cache_plan = self._cache_plan(plan)
        previous, self._pending = self._pending, None
        rows = np.arange(cache_plan.batch.num_sequences)
        with record_function("resin.mlx.enqueue"):
            if previous is not None:
                # Epoch dependencies order previous reads -> writes -> new
                # reads without waiting for the GPU on the calling thread.
                self._write(previous)
            launch = self._launch(cache_plan, rows, rows[:, None], feedback)
            sampled = self._sampler.tokens(launch.last, plan)
            mx.async_eval(self._cache.epoch, sampled, *launch.reads)
        self._pending = launch
        continues = int(np.max(cache_plan.batch.post_lengths)) < self.max_seq_len
        return _Completion(self, plan, launch, sampled, continues)


__all__ = ["Inkling2MlxRunner"]
