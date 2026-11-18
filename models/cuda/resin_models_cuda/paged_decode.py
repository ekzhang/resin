"""Host execution shared by paged decode runners."""

from __future__ import annotations

from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import torch

from resin import BatchPhase
from resin.backend.cuda.completion import (
    CudaContinuation,
    DeviceCompletion,
    GraphDeviceFeedback,
)
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.runner import CudaRunner, pad_sampling_inputs
from resin.backend.cuda.scoring import score_completion
from resin.backend.cuda.slab import GraphPageTableArena
from resin.backend.cuda.state import execute_state
from resin.backend.plans import PagedCachePlan
from resin.backend.torch import use_default_dtype

if TYPE_CHECKING:
    from resin.backend.speculative import SpeculativeStats
    from resin.backend.topology import ParallelGroups

_Cache = TypeVar("_Cache", bound=GraphPageTableArena)


@dataclass(frozen=True)
class PreparedBatch:
    cache_plan: PagedCachePlan | None


def complete_graph_decode(acceptance, logits, plan, *, stats, continuation=None):
    feedback = None
    if continuation is not None:
        next_inputs = acceptance.feedback()
        feedback = GraphDeviceFeedback(
            next_inputs.token_ids, next_inputs.positions, continuation
        )
    return DeviceCompletion.from_acceptance(
        acceptance,
        feedback=feedback,
        rows=0 if plan is None else plan.num_sequences,
        stats=stats if acceptance.draft_tokens and plan is not None else None,
        plan=plan,
        scores=score_completion(logits, acceptance, plan),
    )


def paged_decode_graph_inputs(runner, plan, cache_plan, shape, graph_cache):
    rows = plan.num_sequences
    padding = shape[0] - rows
    inputs = {
        "token_ids": np.pad(plan.token_ids, (0, padding)),
        "positions": np.pad(plan.positions, (0, padding), constant_values=1),
        **runner._cache.graph_plan_inputs(graph_cache, cache_plan),
        **pad_sampling_inputs(runner._sampler, plan, runner.vocab_size, padding),
    }
    if runner.parallel_groups is not None:
        inputs["ep_valid_rows"] = rows
    return inputs


def replay_distributed_decode(
    runner, plan, cache_plan, shape, *, enqueue=False, feedback=None
):
    if shape is None:
        return False, None
    if feedback is not None:
        inputs = runner._continuation.resume(
            () if plan is None else plan.sequence_ids,
            feedback.continuation,
            shape=shape,
        )
        return True, runner._enqueue_graph_output(plan, shape, inputs, feedback)
    if plan is not None:
        return True, runner._graph_forward(
            plan, cache_plan, shape=shape, enqueue=enqueue
        )
    # Idle replicas join collective operations with scratch cache rows.
    inputs = runner._decode_graph_capture_inputs(shape)
    inputs["ep_valid_rows"] = 0
    if enqueue:
        return True, runner._enqueue_graph_output(None, shape, inputs, None)
    runner._run_cuda_graph("decode", shape, inputs)
    return True, None


class PagedDecodeRunner(CudaRunner, Generic[_Cache]):
    """Stage one paged decode batch and complete it before the next replay."""

    _decode_model: str
    _cache: _Cache
    _speculation: SpeculativeStats
    parallel_groups: ParallelGroups | None
    speculative_draft_tokens: int
    vocab_size: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._continuation = CudaContinuation()

    @abstractmethod
    def _prepare(self, plan): ...

    @abstractmethod
    def _eager(self, plan, cache_plan): ...

    @abstractmethod
    def _decode_graph_shape(self, plan, cache_plan): ...

    @abstractmethod
    def _distributed_decode_shape(
        self, plan, context, prepared, *, enqueue, feedback
    ): ...

    @abstractmethod
    def _decode_graph_capture_inputs(self, shape): ...

    @abstractmethod
    def forward_distributed(self, plan, context, *, prepared=None): ...

    def _decode_graph_result(self, shape, inputs):
        return self._run_cuda_graph("decode", shape, inputs)

    def _graph_completion(self, plan, shape, inputs, continuation=None):
        acceptance, logits = self._decode_graph_result(shape, inputs)
        return complete_graph_decode(
            acceptance,
            logits,
            plan,
            stats=self._speculation,
            continuation=continuation,
        )

    def _graph_forward(self, plan, cache_plan, *, shape=None, enqueue=False):
        if shape is None:
            shape = self._decode_graph_shape(plan, cache_plan)
        if shape is None:
            return None
        graph = self._captured_cuda_graph("decode", shape)
        inputs = paged_decode_graph_inputs(
            self, plan, cache_plan, shape, graph.resources
        )
        if enqueue:
            return self._enqueue_graph_output(plan, shape, inputs, None)
        return self._graph_completion(plan, shape, inputs).result()

    def _enqueue_graph_output(self, plan, shape, inputs, feedback):
        continuation = (
            feedback.continuation
            if feedback is not None
            else self._continuation.begin(
                () if plan is None else plan.sequence_ids,
                shape,
                self._captured_cuda_graph("decode", shape).inputs,
            )
        )
        if feedback is not None:
            inputs.update(feedback.inputs())
        completion = self._graph_completion(plan, shape, inputs, continuation)
        # Both steps have to fit the captured page tables. The lookahead for
        # auxiliary writes is reserved in both the plan and the graph.
        maximum = self.runner_config.max_accepted_per_step
        if plan is not None and np.any(
            np.asarray(plan.positions, dtype=np.int64) + 2 * maximum
            > shape[1] + self.speculative_draft_tokens
        ):
            completion.feedback = None
            self._continuation.drain(continuation)
        return completion

    @observe_execution
    @execute_state
    def forward(self, plan):
        self._check_cuda_graphs(plan.phase)
        self._continuation.drain()
        cache_plan = self._prepare(plan)
        with (
            self._cuda_failure,
            torch.inference_mode(),
            use_default_dtype(torch.bfloat16),
            torch.device(self.device),
        ):
            if plan.phase == BatchPhase.DECODE:
                result = self._graph_forward(plan, cache_plan)
                if result is not None:
                    return result
            with (
                self._prefill_graphs()
                if plan.phase == BatchPhase.PREFILL
                else nullcontext()
            ):
                return self._eager(plan, cache_plan)

    @observe_execution
    @execute_state
    def enqueue(self, plan, *, feedback=None):
        if feedback is not None:
            self._validate_graph_feedback(feedback)
            inputs = self._continuation.resume(plan.sequence_ids, feedback.continuation)
            with self._cuda_failure, torch.inference_mode():
                return self._enqueue_graph_output(
                    plan, feedback.continuation.shape, inputs, feedback
                )
        self._continuation.drain()
        if plan.phase != BatchPhase.DECODE or plan.needs_prompt_logprobs:
            return self.forward(plan)
        self._check_cuda_graphs(plan.phase)
        cache_plan = self._prepare(plan)
        with self._cuda_failure, torch.inference_mode():
            result = self._graph_forward(plan, cache_plan, enqueue=True)
        return self.forward(plan) if result is None else result

    def _validate_graph_feedback(self, feedback):
        if not isinstance(feedback, GraphDeviceFeedback):
            raise ValueError(
                f"{self._decode_model} continuation requires owned graph feedback"
            )

    def validate_distributed(self, plan, context):
        if self.parallel_groups is None or context.groups is not self.parallel_groups:
            raise ValueError(
                f"distributed groups do not match {self._decode_model} runner"
            )
        if plan is None:
            if context.num_sequences or context.num_tokens:
                raise ValueError("nonempty distributed replica needs a plan")
            cache_plan = None
        else:
            if (
                plan.phase != context.phase
                or plan.num_tokens != context.num_tokens
                or plan.num_sequences != context.num_sequences
            ):
                raise ValueError("distributed metadata does not match plan")
            cache_plan = self._prepare(plan)
        self._check_cuda_graphs(context.phase)
        return PreparedBatch(cache_plan)

    def _distributed_graph(
        self, plan, context, prepared, *, enqueue=False, feedback=None
    ):
        shape = self._distributed_decode_shape(
            plan, context, prepared, enqueue=enqueue, feedback=feedback
        )
        return replay_distributed_decode(
            self,
            plan,
            None if prepared is None else prepared.cache_plan,
            shape,
            enqueue=enqueue,
            feedback=feedback,
        )

    @observe_execution
    @execute_state
    def enqueue_distributed(self, plan, context, *, prepared=None, feedback=None):
        if feedback is not None:
            self._validate_graph_feedback(feedback)
        else:
            self._continuation.drain()
        if context.phase != BatchPhase.DECODE or (
            plan is not None and plan.needs_prompt_logprobs
        ):
            return self.forward_distributed(plan, context, prepared=prepared)
        prepared = (
            self.validate_distributed(plan, context) if prepared is None else prepared
        )
        with self._cuda_failure, torch.inference_mode():
            used, result = self._distributed_graph(
                plan, context, prepared, enqueue=True, feedback=feedback
            )
        if used:
            return result
        if feedback is not None:
            raise RuntimeError("a device continuation requires a captured graph")
        return self.forward_distributed(plan, context, prepared=prepared)
