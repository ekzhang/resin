"""Paged prefill, overlapped decode and cache lifetime shared by the MLX runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, Protocol, TypeVar

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten
from torch.profiler import record_function

from resin import BatchPhase, RunnerConfig
from resin.backend.capacity import plan_cache_capacity
from resin.backend.plans import PagedCachePlan
from resin.backend.sampling import PromptScores, format_output, prompt_scores
from resin.backend.topology import ParallelismCapabilities
from resin_models_mlx.cache_arena import PAGE_TOKENS, ByteArena, SlabGeometry
from resin_models_mlx.live_state import LiveStates
from resin_models_mlx.sampling import MlxSampler

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput, ResourceBinding

DEFAULT_WORKSPACE_BYTES = 4 * 1024**3
_Arena = TypeVar("_Arena", bound=ByteArena)


def _to_torch(array: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(array.astype(mx.float32)))


def context_limit(requested: int | None, positions: int) -> int:
    if requested is None:
        requested = positions
    if not 0 < requested <= positions:
        raise ValueError(
            f"max_seq_len must be between 1 and the checkpoint's {positions} positions"
        )
    return requested


class ModelStep(NamedTuple):
    """Model outputs that must be evaluated before the next arena mutation."""

    hidden: mx.array
    state: Any
    kv: Mapping[int, tuple[Sequence[mx.array], Sequence[mx.array]]]


@dataclass(slots=True)
class _Launch:
    plan: PagedCachePlan
    rows: np.ndarray
    tokens: np.ndarray
    ends: np.ndarray
    step: ModelStep
    last: mx.array
    reads: tuple[mx.array, ...]
    receipts: tuple[int, ...] | None = None


class MlxCompletion(Protocol):
    @property
    def feedback(self) -> object | None: ...

    def result(self) -> ForwardOutput: ...


@dataclass(slots=True)
class _Completion:
    """One overlapped decode step, collected after the next one is launched."""

    _runner: MlxRunner
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


class MlxRunner(Generic[_Arena], ABC):
    """Execute model steps against a single Engine-owned cache arena."""

    model_name: str
    max_seq_len: int
    vocab_size: int
    page_size = PAGE_TOKENS
    parallelism_capabilities = ParallelismCapabilities()

    def __init__(
        self,
        cache: _Arena,
        states: LiveStates,
        logits: Callable[[mx.array], mx.array],
        prefill_chunk_size: int,
        *,
        max_accepted_per_step: int = 1,
    ) -> None:
        self._cache = cache
        self._states = states
        self._logits = logits
        self.prefill_chunk_size = prefill_chunk_size
        self.cache_layout = cache.layout
        self.runner_config = RunnerConfig(
            max_batch_tokens=prefill_chunk_size,
            prefill_chunk_size=prefill_chunk_size,
            continuation_prefill=True,
            max_accepted_per_step=max_accepted_per_step,
            cache_layout=self.cache_layout,
        )
        self._sampler = MlxSampler()
        self._pending: _Launch | None = None

    def _slab_capacity(
        self,
        geometry: SlabGeometry,
        memory_utilization: float,
        workspace_bytes: int,
        max_batch_size: int,
        cache_slab_capacity: int | None,
    ) -> int:
        device = mx.device_info()
        limit = int(device["max_recommended_working_set_size"])
        self.cache_capacity_plan = plan_cache_capacity(
            limit,
            max(1, limit - mx.get_active_memory()),
            memory_utilization,
            geometry.slab_bytes,
            # The output recurrent state exists beside the held input state.
            overhead_bytes=workspace_bytes + 2 * max_batch_size * geometry.state_bytes,
        )
        # The arena is one Metal buffer, including the reserved null slab.
        buffer_slabs = int(device["max_buffer_length"]) // geometry.slab_bytes
        capacity = min(self.cache_capacity_plan.parent_capacity, buffer_slabs - 1)
        return (
            capacity
            if cache_slab_capacity is None
            else min(capacity, cache_slab_capacity)
        )

    def bind_resources(self, resources: ResourceBinding) -> None:
        self._cache.binding.bind_resources(resources)

    def release(self, sequence_ids: list[int]) -> None:
        # A pending step would hold the released rows' state again.
        self._write_pending()
        self._states.release(sequence_ids)

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        return _to_torch(self._logits(mx.array(hidden.numpy()).astype(mx.bfloat16)))

    @abstractmethod
    def _step(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        tokens: np.ndarray,
        contexts: np.ndarray,
        token_ids: mx.array,
    ) -> ModelStep:
        """Run one prefill chunk or one decode token per row."""

    def _launch(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        tokens: np.ndarray,
        token_ids: mx.array | None = None,
    ) -> _Launch:
        contexts = plan.batch.context_lengths[rows].astype(np.int64)
        if token_ids is None:
            token_ids = mx.array(plan.batch.token_ids[tokens].astype(np.int32))
        step = self._step(plan, rows, tokens, contexts, token_ids.reshape(tokens.shape))
        last = self._logits(step.hidden[:, -1])
        # Keep every computed reader, including all leaves of the held state.
        state_leaves: list[tuple[str, mx.array]] = []
        tree_flatten(step.state, destination=state_leaves)
        reads = (
            last,
            step.hidden,
            *(v for _, v in state_leaves),
            *(v for pair in step.kv.values() for layers in pair for v in layers),
        )
        return _Launch(
            plan, rows, tokens, contexts + tokens.shape[1], step, last, reads
        )

    def _write(self, launch: _Launch) -> None:
        self._cache.order_after(launch.reads)
        for group, (keys, values) in launch.step.kv.items():
            self._cache.write_tokens(
                launch.plan, keys, values, launch.tokens, group=group
            )
        launch.receipts = self._states.commit(
            launch.plan, launch.rows, launch.ends, launch.step.state
        )

    def _write_pending(self) -> None:
        """Drain pending cache work before a terminal/ownership transition."""

        launch, self._pending = self._pending, None
        if launch is None:
            return
        self._write(launch)
        # This path is used when there is no continuation to retain the lease.
        # Include snapshots and KV writes, not only the sampled-token event.
        mx.eval(self._cache.epoch)

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self._write_pending()
        cache_plan = self._cache_plan(plan)
        batch = cache_plan.batch
        prefill = plan.phase == BatchPhase.PREFILL
        rows = np.arange(batch.num_sequences)
        groups = rows[:, None] if prefill else (rows,)
        logits = []
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
            logits.append(launch.last)
            assert launch.receipts is not None
            receipts += launch.receipts
            if prefill and plan.needs_prompt_logprobs:
                scores = prompt_scores(
                    plan,
                    _to_torch(launch.step.hidden[0]),
                    self._project,
                    token_start=int(tokens[0, 0]),
                )
                if scores is not None:
                    prompt_rows.append(scores)
        # A synchronous result can release the lease immediately: wait for
        # every issued cache write, not just the sampled-token event.
        combined = logits[0] if len(logits) == 1 else mx.concatenate(logits)
        combined = mx.depends(combined, self._cache.epoch)
        with record_function("resin.mlx.sample"):
            tokens = self._sampler(combined, plan)
        return format_output(
            _to_torch(combined) if plan.needs_logprobs else None,
            tokens,
            plan,
            prompt_logprobs=PromptScores.concat(prompt_rows),
        ).with_state_checkpoints(receipts)

    def _cache_plan(self, plan: BatchPlan) -> PagedCachePlan:
        lookahead = self.runner_config.max_accepted_per_step - 1
        lookahead += self.cache_layout.decode_write_extra_tokens
        cache_plan = PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=lookahead
        )
        if np.any(cache_plan.token_ids >= self.vocab_size):
            raise ValueError(f"token id is out of range for {self.model_name}")
        if int(np.max(cache_plan.batch.post_lengths)) > self.max_seq_len:
            raise ValueError(
                f"sequence exceeds this runner's {self.max_seq_len}-token context"
            )
        return cache_plan

    def enqueue(
        self, plan: BatchPlan, *, feedback: mx.array | None = None
    ) -> MlxCompletion | ForwardOutput:
        """Launch a decode step while the previous one still runs on the GPU.

        ``feedback`` holds the previous step's sampled tokens, which replace
        the continuation plan's token IDs. The previous step's arena writes are
        ordered after its reads and before this step's reads on the GPU.
        """

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
            tokens = self._sampler.tokens(launch.last, plan)
            mx.async_eval(self._cache.epoch, tokens, *launch.reads)
        self._pending = launch
        continues = int(launch.ends.max()) < self.max_seq_len
        return _Completion(self, plan, launch, tokens, continues)
