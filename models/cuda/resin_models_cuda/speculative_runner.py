"""Host execution for paged CUDA models with speculative decode continuations."""

from __future__ import annotations

from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from resin import BatchPhase
from resin.backend.cuda.completion import (
    CudaContinuation,
    DeviceAcceptance,
    DeviceCompletion,
    GraphDeviceFeedback,
)
from resin.backend.cuda.graphs import GraphShape
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.runner import (
    CudaRunner,
    capture_sampling_inputs,
    pad_sampling_inputs,
)
from resin.backend.cuda.scoring import score_completion
from resin.backend.cuda.slab import GraphPageTableArena, SlabDeviceCache
from resin.backend.cuda.state import execute_state
from resin.backend.plans import PagedCachePlan
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelGroups
from resin.backend.torch import use_default_dtype


@dataclass(frozen=True)
class PreparedBatch:
    cache_plan: PagedCachePlan | None


class SpeculativeCudaRunner(CudaRunner):
    """Prepare plans, replay an owned continuation, or fall back to eager execution."""

    _cache: GraphPageTableArena
    parallel_groups: ParallelGroups | None
    vocab_size: int

    def __init__(self, *, speculative_draft_tokens: int, label: str, **kwargs):
        super().__init__(label=label, **kwargs)
        self._runner_label = label
        self.speculative_draft_tokens = speculative_draft_tokens
        self._speculation = SpeculativeStats(speculative_draft_tokens)
        self._graph_device_caches: dict[GraphShape, SlabDeviceCache] = {}
        self._graph_logits: dict[GraphShape, torch.Tensor] = {}
        self._graph_acceptance: dict[GraphShape, DeviceAcceptance] = {}
        self._graph_inputs: dict[GraphShape, dict] = {}
        self._continuation = CudaContinuation()

    def speculative_stats(self):
        return self._speculation.snapshot(include_draft_tokens=True)

    @abstractmethod
    def _prepare(self, plan) -> PagedCachePlan: ...

    @abstractmethod
    def _eager(self, plan, cache_plan): ...

    @abstractmethod
    def _decode_graph_shape(self, plan, cache_plan): ...

    @abstractmethod
    def _distributed_decode_shape(
        self, plan, context, prepared, *, enqueue, feedback
    ): ...

    @abstractmethod
    def _eager_distributed(self, plan, context, cache_plan): ...

    def _decode_graph_capture_inputs(self, shape):
        inputs = {
            "token_ids": (0,) * shape[0],
            "positions": (1,) * shape[0],
            **capture_sampling_inputs(shape[0], self.vocab_size, unique_seeds=True),
            **self._cache.graph_shape_capture_inputs(
                shape[0], shape[1] + self.speculative_draft_tokens
            ),
        }
        if self.parallel_groups is not None:
            inputs["ep_valid_rows"] = shape[0]
        return inputs

    def _graph_forward(self, plan, cache_plan, *, shape=None, enqueue=False):
        batch_size = plan.num_sequences
        if shape is None:
            shape = self._decode_graph_shape(plan, cache_plan)
        if shape is None:
            return None
        padding = shape[0] - batch_size
        inputs = {
            "token_ids": np.pad(plan.token_ids, (0, padding)),
            "positions": np.pad(plan.positions, (0, padding), constant_values=1),
            **self._cache.graph_plan_inputs(
                self._graph_device_caches[shape], cache_plan
            ),
            **pad_sampling_inputs(self._sampler, plan, self.vocab_size, padding),
        }
        if self.parallel_groups is not None:
            inputs["ep_valid_rows"] = batch_size
        if enqueue:
            return self._enqueue_graph_output(plan, shape, inputs, None)
        return self._graph_completion(plan, shape, inputs).result()

    def _decode_acceptance(self, shape, inputs):
        self._run_cuda_graph("decode", shape, inputs)
        return self._graph_acceptance[shape]

    def _graph_completion(self, plan, shape, inputs, continuation=None):
        acceptance = self._decode_acceptance(shape, inputs)
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
            device=self.device,
            stats=self._speculation
            if self.speculative_draft_tokens and plan is not None
            else None,
            plan=plan,
            scores=score_completion(self._graph_logits[shape], acceptance, plan),
        )

    def _enqueue_graph_output(self, plan, shape, inputs, feedback):
        ids = () if plan is None else tuple(map(int, plan.sequence_ids))
        continuation = (
            feedback.continuation
            if feedback is not None
            else self._continuation.begin(ids, shape, self._graph_inputs[shape])
        )
        if feedback is not None:
            inputs.update(feedback.inputs())
        completion = self._graph_completion(plan, shape, inputs, continuation)
        # Both this step and its successor must fit in the captured page tables.
        maximum = self.runner_config.max_accepted_per_step
        capacity = shape[1] + self.speculative_draft_tokens
        if plan is not None and np.any(
            np.asarray(plan.positions, dtype=np.int64) + 2 * maximum > capacity
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
            if not isinstance(feedback, GraphDeviceFeedback):
                raise ValueError(
                    f"{self._runner_label} continuation requires owned graph feedback"
                )
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

    def validate_distributed(self, plan, context):
        if self.parallel_groups is None or context.groups is not self.parallel_groups:
            raise ValueError(
                f"distributed groups do not match {self._runner_label} runner"
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

    @observe_execution
    @execute_state
    def enqueue_distributed(self, plan, context, *, prepared=None, feedback=None):
        if feedback is not None:
            if not isinstance(feedback, GraphDeviceFeedback):
                raise ValueError(
                    f"{self._runner_label} continuation requires owned graph feedback"
                )
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

    def _distributed_graph(
        self, plan, context, prepared, *, enqueue=False, feedback=None
    ):
        shape = self._distributed_decode_shape(
            plan, context, prepared, enqueue=enqueue, feedback=feedback
        )
        if shape is None:
            return False, None
        if feedback is not None:
            inputs = self._continuation.resume(
                () if plan is None else plan.sequence_ids,
                feedback.continuation,
                shape=shape,
            )
            return True, self._enqueue_graph_output(plan, shape, inputs, feedback)
        if plan is not None:
            return True, self._graph_forward(
                plan, prepared.cache_plan, shape=shape, enqueue=enqueue
            )
        inputs = self._decode_graph_capture_inputs(shape)
        inputs["ep_valid_rows"] = 0
        if enqueue:
            return True, self._enqueue_graph_output(None, shape, inputs, None)
        self._run_cuda_graph("decode", shape, inputs)
        return True, None

    @observe_execution
    @execute_state
    def forward_distributed(self, plan, context, *, prepared=None):
        self._continuation.drain()
        prepared = (
            self.validate_distributed(plan, context) if prepared is None else prepared
        )
        with (
            self._cuda_failure,
            torch.inference_mode(),
            use_default_dtype(torch.bfloat16),
            torch.device(self.device),
        ):
            if context.phase == BatchPhase.DECODE:
                used, result = self._distributed_graph(plan, context, prepared)
                if used:
                    return result
            return self._eager_distributed(plan, context, prepared.cache_plan)
