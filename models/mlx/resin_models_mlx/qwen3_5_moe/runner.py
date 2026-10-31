"""Qwen3.5-MoE runner on Apple Silicon over Engine-owned MLX slabs."""

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
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.qwen3_5_moe.cache import (
    PREFILL_CHUNK_SIZE,
    CacheArena,
    CacheGeometry,
)
from resin_models_mlx.qwen3_5_moe.checkpoint import load_model
from resin_models_mlx.qwen3_5_moe.model import Qwen35MoeModel, StepOutput
from resin_models_mlx.qwen3_5_moe.state import LiveStates
from resin_models_mlx.sampling import MlxSampler

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput, ResourceBinding

_WORKSPACE_BYTES = 4 * 1024**3


def _to_torch(array: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(array.astype(mx.float32)))


@dataclass(slots=True)
class _Launch:
    """A step's unevaluated outputs; its arena writes wait until its reads finish."""

    plan: PagedCachePlan
    rows: np.ndarray
    tokens: np.ndarray
    contexts: np.ndarray
    output: StepOutput
    last: mx.array
    receipts: tuple[int, ...] | None = field(default=None)

    @property
    def reads(self) -> tuple[mx.array, ...]:
        """Every output computed from the arena, including the held state."""

        output = self.output
        return (
            self.last,
            output.hidden,
            *output.keys,
            *output.values,
            *output.state.recurrent,
            *output.state.convolution,
        )


class _Completion:
    """One overlapped decode step, collected after the next one is launched."""

    def __init__(
        self,
        runner: Qwen3_5MoeMlxRunner,
        plan: BatchPlan,
        launch: _Launch,
        tokens: mx.array,
        continues: bool,
    ) -> None:
        self._runner = runner
        self._plan = plan
        self._launch = launch
        self._tokens = tokens
        self._continues = continues
        self._result: ForwardOutput | None = None

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


class Qwen3_5MoeMlxRunner:
    """Text-only Qwen3.5-MoE runner with paged attention and DeltaNet state."""

    page_size = 128
    prefill_chunk_size = PREFILL_CHUNK_SIZE
    parallelism_capabilities = ParallelismCapabilities()

    def __init__(
        self,
        model: str | Path | Qwen35MoeModel,
        *,
        weight_bits: int | None = 8,
        group_size: int = 64,
        memory_utilization: float = 0.85,
        cache_slab_capacity: int | None = None,
        max_seq_len: int | None = None,
        max_batch_size: int = 16,
    ) -> None:
        if isinstance(model, Qwen35MoeModel):
            self._model = model
            self.config: Qwen35Config = model.config
            self.model_path = None
            quantization = "preloaded"
        else:
            self.model_path = Path(model)
            self._model, self.config, _ = load_model(
                self.model_path, bits=weight_bits, group_size=group_size
            )
            quantization = (
                "bfloat16"
                if weight_bits is None
                else f"mlx-int{weight_bits}-g{group_size}"
            )
        limit_positions = self.config.max_position_embeddings
        self.max_seq_len = limit_positions if max_seq_len is None else max_seq_len
        if not 0 < self.max_seq_len <= limit_positions:
            raise ValueError(
                f"max_seq_len must be between 1 and the checkpoint's "
                f"{limit_positions} positions"
            )
        # Describes how this runner executes for benchmark records.
        self.benchmark_metadata = {
            "quant": quantization,
            "expert_backend": "mlx-gather-qmm",
            "graphs": {"prefill": "disabled", "decode": "disabled"},
        }
        self.vocab_size = self.config.vocab_size
        self.eos_token_id = self.config.eos_token_id
        self.cache_geometry = CacheGeometry.from_config(self.config)
        device = mx.device_info()
        limit = int(device["max_recommended_working_set_size"])
        self.cache_capacity_plan = plan_cache_capacity(
            limit,
            max(1, limit - mx.get_active_memory()),
            memory_utilization,
            self.cache_geometry.slab_bytes,
            # Live DeltaNet state is held for every running request, and a
            # step's output exists beside its input.
            overhead_bytes=_WORKSPACE_BYTES
            + 2 * max_batch_size * self.cache_geometry.state_bytes,
        )
        # The arena is one Metal buffer, including the reserved null slab.
        buffer_slabs = (
            int(device["max_buffer_length"]) // self.cache_geometry.slab_bytes
        )
        slab_capacity = min(self.cache_capacity_plan.parent_capacity, buffer_slabs - 1)
        if cache_slab_capacity is not None:
            slab_capacity = min(slab_capacity, cache_slab_capacity)
        self.cache_layout = self.cache_geometry.layout(slab_capacity)
        self._cache = CacheArena(self.cache_layout, self.cache_geometry)
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            cache_layout=self.cache_layout,
        )
        self._sampler = MlxSampler()
        self._states = LiveStates(self._cache, self._model)
        # An enqueued decode step whose arena writes have not been issued.
        self._pending: _Launch | None = None

    def bind_resources(self, resources: ResourceBinding) -> None:
        self._cache.binding.bind_resources(resources)

    def release(self, sequence_ids: list[int]) -> None:
        # A pending step would hold the released rows' state again.
        self._write_pending()
        self._states.release(sequence_ids)

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self._model.logits(mx.array(hidden.numpy()).astype(mx.bfloat16))
        return _to_torch(logits)

    def _launch(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        tokens: np.ndarray,
        token_ids: mx.array | None = None,
    ) -> _Launch:
        """Build the step for ``tokens``, flattened plan indices ``[len(rows), width]``.

        Decode passes one token per row; prefill passes one row's chunk.
        ``token_ids`` replaces the plan's IDs with tokens still on the GPU.
        """

        batch = plan.batch
        contexts = batch.context_lengths[rows].astype(np.int64)
        if token_ids is None:
            token_ids = mx.array(batch.token_ids[tokens].astype(np.int32))
        # Decode attends to history in place; prefill gathers one row's
        # history layer by layer for the dense causal kernel. Only the graph
        # holds these views of the arena, so they end with its evaluation.
        history = (
            self._cache.paged_history(plan, rows, contexts)
            if tokens.shape[1] == 1
            else self._cache.history(plan, int(rows[0]), int(contexts[0]))
        )
        output = self._model(
            token_ids.reshape(tokens.shape),
            mx.array(contexts.astype(np.int32)),
            self._states.gather(plan, rows),
            history,
        )
        last = self._model.logits(output.hidden[:, -1])
        return _Launch(plan, rows, tokens, contexts, output, last)

    def _write(self, launch: _Launch) -> None:
        """Queue ``launch``'s arena writes and hold its state; reads must be done."""

        output = launch.output
        self._cache.write_tokens(
            launch.plan,
            tuple(key.reshape(-1, *key.shape[2:]) for key in output.keys),
            tuple(value.reshape(-1, *value.shape[2:]) for value in output.values),
            launch.tokens.reshape(-1),
        )
        launch.receipts = self._states.commit(
            launch.plan,
            launch.rows,
            launch.contexts + launch.tokens.shape[1],
            output.state,
        )

    def _write_pending(self) -> None:
        """Finish an overlapped step's reads, then update the arena in place."""

        launch, self._pending = self._pending, None
        if launch is None:
            return
        mx.eval(*launch.reads)
        self._write(launch)
        mx.async_eval(self._cache.storage)

    def _step(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        tokens: np.ndarray,
    ) -> tuple[StepOutput, mx.array, tuple[int, ...]]:
        """Run one step synchronously.

        Returns the output, last-token logits and written snapshot receipts.
        """

        with record_function("resin.mlx.forward"):
            launch = self._launch(plan, rows, tokens)
            # Finish every arena read before scattering, so MLX can donate the
            # arena buffer to the writes instead of copying it.
            mx.eval(*launch.reads)
        with record_function("resin.mlx.cache_write"):
            self._write(launch)
            # Reads of this step are complete, so the writes can update the arena
            # in place while the host samples and commits. The next step's reads
            # depend on the written arena and therefore run after them.
            mx.async_eval(self._cache.storage)
        assert launch.receipts is not None
        return launch.output, launch.last, launch.receipts

    def _output(
        self, plan: BatchPlan, logits: mx.array, prompt_logprobs=None
    ) -> ForwardOutput:
        with record_function("resin.mlx.sample"):
            tokens = self._sampler(logits, plan)
        # Logits leave the GPU only when the plan asks for scores.
        host_logits = _to_torch(logits) if plan.needs_logprobs else None
        return format_output(host_logits, tokens, plan, prompt_logprobs=prompt_logprobs)

    def _prefill(self, plan: BatchPlan, cache_plan: PagedCachePlan) -> ForwardOutput:
        batch = cache_plan.batch
        logits = []
        prompt_rows = []
        receipts: list[int] = []
        for row in range(batch.num_sequences):
            selected = batch.token_slice(row)
            output, last, written = self._step(
                cache_plan,
                np.array([row]),
                np.arange(selected.start, selected.stop)[None],
            )
            logits.append(last)
            receipts += written
            if (
                plan.needs_prompt_logprobs
                and (
                    scores := prompt_scores(
                        plan,
                        _to_torch(output.hidden[0]),
                        self._project,
                        token_start=selected.start,
                    )
                )
                is not None
            ):
                prompt_rows.append(scores)
        return self._output(
            plan,
            mx.concatenate(logits),
            prompt_logprobs=PromptScores.concat(prompt_rows),
        ).with_state_checkpoints(receipts)

    def _decode(self, plan: BatchPlan, cache_plan: PagedCachePlan) -> ForwardOutput:
        batch = cache_plan.batch
        rows = np.arange(batch.num_sequences)
        _, last, receipts = self._step(cache_plan, rows, rows[:, None])
        return self._output(plan, last).with_state_checkpoints(receipts)

    def _cache_plan(self, plan: BatchPlan) -> PagedCachePlan:
        cache_plan = PagedCachePlan.from_plan(plan, self.cache_layout)
        if np.any(cache_plan.token_ids >= self.vocab_size):
            raise ValueError("token id is out of range for Qwen3.5-MoE")
        if int(np.max(cache_plan.batch.post_lengths)) > self.max_seq_len:
            raise ValueError(
                f"sequence exceeds this runner's {self.max_seq_len}-token context"
            )
        return cache_plan

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self._write_pending()
        cache_plan = self._cache_plan(plan)
        if plan.phase == BatchPhase.PREFILL:
            return self._prefill(plan, cache_plan)
        return self._decode(plan, cache_plan)

    def enqueue(
        self, plan: BatchPlan, *, feedback: mx.array | None = None
    ) -> _Completion | ForwardOutput:
        """Launch a decode step while the previous one still runs on the GPU.

        ``feedback`` holds the previous step's sampled tokens, which replace
        the continuation plan's token IDs. The previous step's arena writes are
        queued once its reads finish, ahead of this step's reads.
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
                # The writes stay unevaluated until the previous reads finish,
                # but this step's history and state must already include them.
                self._write(previous)
            launch = self._launch(cache_plan, rows, rows[:, None], feedback)
            tokens = self._sampler.tokens(launch.last, plan)
            if previous is not None:
                mx.eval(*previous.reads)
            mx.async_eval(self._cache.storage, tokens, *launch.reads)
        self._pending = launch
        continues = int(np.max(cache_plan.batch.post_lengths)) < self.max_seq_len
        return _Completion(self, plan, launch, tokens, continues)


__all__ = ["Qwen3_5MoeMlxRunner"]
