"""Captured decode scores and eager bounded prompt output-head projections."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from resin.backend.cuda.completion import (
    DeviceAcceptance,
    DeviceCompletion,
    DeviceFeedback,
    DeviceScores,
)
from resin.backend.cuda.staging import TensorStager, TensorStagerPool
from resin.backend.sampling import PromptScores, _ScoreColumns

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput

# How many tokens to score for prompt_logprobs per loop iteration. We chunk it
# to avoid having large logit tensors in memory.
_PROMPT_SCORE_CHUNK_SIZE = 256


def _owner(value: torch.Tensor) -> torch.Tensor:
    while value._base is not None:
        value = value._base
    return value


def _capacity(count: int, vocab: int) -> int:
    # Capacity controls output storage and graph binding, never kernel compilation.
    return min(vocab, 0 if count == 0 else max(32, 1 << (count - 1).bit_length()))


def _capture(device: torch.device, launch: Callable[[], None]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream(device=device)
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(2):
            launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    current.wait_stream(stream)
    return graph


class _DecodeScores:
    def __init__(self, logits: torch.Tensor, width: int, capacity: int) -> None:
        from resin_models_cuda._scoring_cutedsl import score_logits

        self.scores: DeviceScores
        batch = logits.shape[0]
        self.selected = torch.zeros(
            (batch, width), dtype=torch.int64, device=logits.device
        )
        self.counts = torch.full(
            (batch,), width, dtype=torch.int32, device=logits.device
        )
        self.requested = torch.zeros(batch, dtype=torch.bool, device=logits.device)
        self.top_counts = torch.zeros(batch, dtype=torch.int32, device=logits.device)
        self.staging = TensorStagerPool(logits.device)
        self.vocab_size = logits.shape[-1]

        def launch():
            result = score_logits(
                logits,
                self.selected,
                score_mask=self.requested,
                top_counts=self.top_counts,
                max_top_k=capacity,
                valid_counts=self.counts,
            )
            self.scores = DeviceScores(*result[:3], self.vocab_size)

        self.graph = _capture(logits.device, launch)

    def run(self, acceptance, requested, top_counts) -> DeviceScores:
        self.staging.copy_into(
            {
                "selected": self.selected,
                "counts": self.counts,
                "requested": self.requested,
                "top_counts": self.top_counts,
            },
            {
                "selected": acceptance.tokens,
                "counts": acceptance.counts,
                "requested": requested,
                "top_counts": top_counts,
            },
        )
        self.graph.replay()
        return self.scores


@torch.inference_mode()
def score_completion(
    logits: torch.Tensor,
    acceptance: DeviceAcceptance,
    plan: BatchPlan | None,
) -> DeviceScores | None:
    """Score actual emitted tokens; unscored plans do no CUDA scoring work."""
    if plan is None or not plan.needs_logprobs or not logits.is_cuda:
        return None
    # Inference-mode views do not track _base. Bind the cache to the caller's
    # persistent tensor before adding the ordinary decode's width dimension.
    owner = _owner(logits)
    if logits.ndim == 2:
        logits = logits[:, None, :]
    batch, width = acceptance.tokens.shape
    if logits.ndim != 3 or logits.shape[:2] != (batch, width):
        raise ValueError("target logits must cover the accepted token grid")
    policies = plan.sampling_params
    requested = np.zeros(batch, dtype=np.bool_)
    top_counts = np.zeros(batch, dtype=np.int32)
    for row, (sample, params) in enumerate(
        zip(plan.sampling_mask, policies, strict=True)
    ):
        requested[row] = sample and params.logprobs
        if requested[row]:
            top_counts[row] = min(params.top_logprobs, logits.shape[-1])
    capacity = _capacity(int(max(top_counts, default=0)), logits.shape[-1])
    cache = getattr(owner, "_resin_score_graphs", None)
    if cache is None:
        cache = {}
        setattr(owner, "_resin_score_graphs", cache)
    key = (logits.data_ptr(), tuple(logits.shape), logits.stride(), capacity)
    entry = cache.get(key)
    if entry is None:
        entry = cache[key] = _DecodeScores(logits, width, capacity)
    return entry.run(acceptance, requested, top_counts)


@torch.inference_mode()
def prompt_scores(
    plan: BatchPlan,
    hidden: torch.Tensor,
    project: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    token_start: int = 0,
) -> PromptScores | None:
    """Eagerly project and score requested predecessors in bounded CUDA chunks."""
    if not plan.needs_prompt_logprobs:
        return None
    if not hidden.is_cuda:
        from resin.backend.sampling import prompt_scores as reference

        return reference(plan, hidden, project, token_start=token_start)
    from resin_models_cuda._scoring_cutedsl import score_logits

    indices = plan.prompt_score_indices
    keep = (indices >= token_start) & (indices < token_start + hidden.shape[0])
    selected_rows = indices[keep].astype(np.int64) - token_start
    selected_ids = plan.prompt_score_token_ids[keep].astype(np.int64)
    top_counts = plan.prompt_score_top_counts[keep]
    target_indptr = plan.prompt_target_indptr
    target_counts = np.diff(target_indptr)[keep]
    source_rows = np.flatnonzero(keep)
    parts = []
    for start in range(0, len(selected_rows), _PROMPT_SCORE_CHUNK_SIZE):
        stop = min(start + _PROMPT_SCORE_CHUNK_SIZE, len(selected_rows))
        staging = TensorStager(hidden.device)
        indices = staging.device("indices", selected_rows[start:stop], torch.int64)
        values = hidden.index_select(0, indices)
        if project is not None:
            values = project(values)
        vocab = values.shape[-1]
        counts = target_counts[start:stop]
        ids = (
            np.concatenate(
                [
                    plan.prompt_target_token_ids[
                        target_indptr[row] : target_indptr[row + 1]
                    ]
                    for row in source_rows[start:stop]
                ]
            ).astype(np.int64)
            if np.any(counts)
            else np.empty(0, np.int64)
        )
        if np.any(ids >= vocab) or np.any(selected_ids[start:stop] >= vocab):
            raise ValueError(
                f"prompt target token IDs must be less than vocabulary size {vocab}"
            )
        clipped = np.minimum(top_counts[start:stop], vocab).astype(np.uint32)
        maximum = int(np.max(clipped, initial=0))
        target_ids = target_rows = None
        if len(ids):
            target_ids = staging.device("target_ids", ids, torch.int64)
            rows = np.repeat(np.arange(stop - start, dtype=np.int64), counts)
            target_rows = staging.device("target_rows", rows, torch.int64)
        chosen, top_ids, top_values, targets = score_logits(
            values,
            staging.device("selected", selected_ids[start:stop], torch.int64),
            top_counts=staging.device(
                "top_counts", clipped.astype(np.int32), torch.int32
            ),
            max_top_k=maximum,
            target_rows=target_rows,
            target_ids=target_ids,
        )
        top = None
        if maximum:
            offsets = np.concatenate(
                (np.zeros(1, np.int64), clipped.cumsum(dtype=np.int64))
            )
            positions = np.arange(offsets[-1]) - np.repeat(offsets[:-1], clipped)
            packed = np.repeat(np.arange(stop - start), clipped) * maximum + positions
            top_indices = staging.device("top_indices", packed, torch.int64)
            top_ids = top_ids.reshape(-1).index_select(0, top_indices)
            top_values = top_values.reshape(-1).index_select(0, top_indices)
            top = _ScoreColumns(clipped, top_ids, top_values)
        target_columns = (
            None if target_ids is None else _ScoreColumns(counts, target_ids, targets)
        )
        parts.append(PromptScores(chosen, top, target_columns))
    return (
        PromptScores.concat(parts)
        if parts
        else PromptScores(hidden.new_empty(0, dtype=torch.float32))
    )


@torch.inference_mode()
def format_output(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    plan: BatchPlan,
    *,
    prompt_logprobs: PromptScores | None = None,
) -> ForwardOutput:
    """Use CUDA scoring for generated columns; CPU runners retain their reference."""
    if not plan.needs_logprobs or logits is None or not logits.is_cuda:
        from resin.backend.sampling import format_output as reference

        return reference(logits, token_ids, plan, prompt_logprobs=prompt_logprobs)
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("CUDA scored outputs require sampled token tensors")
    tokens = token_ids[:, None] if token_ids.ndim == 1 else token_ids
    counts = torch.full(
        (tokens.shape[0],), tokens.shape[1], dtype=torch.int32, device=logits.device
    )
    accepted = DeviceAcceptance(
        tokens,
        (counts - 1).long(),
        counts,
        torch.zeros_like(counts, dtype=torch.int64),
        tokens.shape[1] - 1,
    )
    # Formatting does not produce a continuation's bonus-token gather.
    feedback = DeviceFeedback(torch.zeros_like(accepted.positions), counts.long())
    output = DeviceCompletion(
        tokens,
        counts,
        feedback,
        rows=plan.num_sequences,
        plan=plan,
        scores=score_completion(logits, accepted, plan),
    ).result()
    if not plan.needs_prompt_logprobs:
        return output
    if prompt_logprobs is None:
        raise ValueError("prefill did not provide requested prompt scores")
    from resin import ForwardOutput

    return ForwardOutput(
        token_ids=output.token_ids,
        token_indptr=output.token_indptr,
        logprobs=output.logprobs,
        logprob_indptr=output.logprob_indptr,
        top_indptr=output.top_indptr,
        top_token_ids=output.top_token_ids,
        top_logprobs=output.top_logprobs,
        **prompt_logprobs.output(plan),
    )
