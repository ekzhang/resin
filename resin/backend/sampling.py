"""PyTorch sampling operations shared by model runners."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch

from resin import SamplingParams

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput

_TokenIds = (
    torch.Tensor
    | npt.NDArray
    | Sequence[int | np.integer]
    | Sequence[Sequence[int | np.integer]]
)

SAMPLING_OFFSET_STRIDE = 32


def _offsets(counts: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
    return np.concatenate((np.zeros(1, np.uint32), counts.cumsum(dtype=np.uint32)))


@dataclass(frozen=True)
class _ScoreColumns:
    counts: npt.NDArray[np.uint32]
    token_ids: torch.Tensor
    logprobs: torch.Tensor

    def output(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_indptr": _offsets(self.counts),
            f"{prefix}_token_ids": self.token_ids.detach()
            .to(device="cpu", dtype=torch.uint32)
            .numpy(),
            f"{prefix}_logprobs": self.logprobs.detach().cpu().numpy(),
        }

    @classmethod
    def concat(
        cls, parts: Sequence[_ScoreColumns | None], lengths: Sequence[int]
    ) -> _ScoreColumns | None:
        present = [part for part in parts if part is not None]
        if not present:
            return None
        return cls(
            np.concatenate(
                [
                    np.zeros(length, np.uint32) if part is None else part.counts
                    for part, length in zip(parts, lengths, strict=True)
                ]
            ),
            torch.cat([part.token_ids for part in present]),
            torch.cat([part.logprobs for part in present]),
        )


@dataclass(frozen=True)
class PromptScores:
    """Packed selected, top, and target scores from the same raw distributions."""

    logprobs: torch.Tensor
    top: _ScoreColumns | None = None
    targets: _ScoreColumns | None = None

    @classmethod
    def concat(cls, parts: Sequence[PromptScores]) -> PromptScores | None:
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        lengths = [len(part.logprobs) for part in parts]
        return cls(
            torch.cat([part.logprobs for part in parts]),
            _ScoreColumns.concat([part.top for part in parts], lengths),
            _ScoreColumns.concat([part.targets for part in parts], lengths),
        )

    def output(self, plan: BatchPlan) -> dict[str, Any]:
        columns = {
            "prompt_logprobs": self.logprobs.detach().cpu().numpy(),
            "prompt_logprob_indptr": np.asarray(
                plan.prompt_score_indptr, dtype=np.uint32
            ),
        }
        if self.top is not None:
            columns.update(self.top.output("prompt_top"))
        if self.targets is not None:
            columns.update(self.targets.output("prompt_target"))
        return columns


@torch.compile(  # ty: ignore[no-matching-overload]
    dynamic=True, fullgraph=True, recompile_limit=128
)
def _score_tensors(
    logits: torch.Tensor,
    selected: torch.Tensor,
    max_k: int,
    target_rows: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize once and return only the requested score columns."""
    torch._dynamo.mark_static(logits, -1)
    values = logits.float().log_softmax(dim=-1)
    chosen = values.gather(1, selected[:, None]).squeeze(1)
    if max_k:
        top_values, top_ids = values.topk(max_k, dim=-1)
    else:
        top_values = logits.new_empty((logits.shape[0], 0), dtype=torch.float32)
        top_ids = logits.new_empty((logits.shape[0], 0), dtype=torch.int64)
    targets = values[target_rows, target_ids]
    return chosen, top_values, top_ids, targets


def _score_distribution(
    logits: torch.Tensor,
    selected: torch.Tensor,
    top_counts: npt.NDArray[np.uint32],
    target_counts: npt.NDArray[np.uint32] | None = None,
    target_ids: npt.NDArray[np.uint32] | None = None,
) -> PromptScores:
    """Normalize once, then gather all requested views of each distribution."""
    vocab = logits.shape[-1]
    # Check on CPU before gather: an invalid CUDA index poisons the device.
    if target_ids is not None and np.any(target_ids >= vocab):
        raise ValueError(
            f"prompt target token IDs must be less than vocabulary size {vocab}"
        )
    top_counts = np.minimum(top_counts, vocab).astype(np.uint32)
    max_k = int(top_counts.max()) if top_counts.size else 0
    if target_ids is not None and target_counts is not None and len(target_ids):
        target_rows = torch.as_tensor(
            np.repeat(np.arange(len(selected)), target_counts), device=logits.device
        )
        target_tokens = torch.as_tensor(
            target_ids.astype(np.int64), device=logits.device
        )
    else:
        target_rows = torch.empty(0, dtype=torch.int64, device=logits.device)
        target_tokens = torch.empty_like(target_rows)
    chosen, top_values, top_ids, target_values = _score_tensors(
        logits, selected, max_k, target_rows, target_tokens
    )
    top = None
    if max_k:
        keep = (
            torch.arange(max_k, device=logits.device)[None, :]
            < torch.as_tensor(top_counts.astype(np.int64), device=logits.device)[
                :, None
            ]
        )
        top = _ScoreColumns(top_counts, top_ids[keep], top_values[keep])
    targets = None
    if target_ids is not None and target_counts is not None and len(target_ids):
        targets = _ScoreColumns(target_counts, target_tokens, target_values)
    return PromptScores(chosen, top, targets)


def prompt_scores(
    plan: BatchPlan,
    hidden: torch.Tensor,
    project: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    token_start: int = 0,
) -> PromptScores | None:
    """Score this forward's prompt predecessors, projecting at most 128 rows."""
    if not plan.needs_prompt_logprobs:
        return None
    indices = plan.prompt_score_indices
    keep = (indices >= token_start) & (indices < token_start + hidden.shape[0])
    rows = np.flatnonzero(keep)
    selected = torch.as_tensor(
        indices[keep].astype(np.int64) - token_start, device=hidden.device
    )
    targets = torch.as_tensor(
        plan.prompt_score_token_ids[keep].astype(np.int64), device=hidden.device
    )
    top_counts = plan.prompt_score_top_counts[keep]
    target_indptr = plan.prompt_target_indptr
    target_counts = np.diff(target_indptr)[keep]
    target_ids = plan.prompt_target_token_ids
    parts = []
    for start in range(0, len(selected), 128):
        end = start + 128
        values = hidden.index_select(0, selected[start:end])
        values = project(values) if project is not None else values
        ids = (
            np.concatenate(
                [
                    target_ids[target_indptr[row] : target_indptr[row + 1]]
                    for row in rows[start:end]
                ]
            )
            if np.any(target_counts[start:end])
            else None
        )
        parts.append(
            _score_distribution(
                values,
                targets[start:end],
                top_counts[start:end],
                target_counts[start:end],
                ids,
            )
        )
    return (
        PromptScores.concat(parts)
        if parts
        else PromptScores(hidden.new_empty(0, dtype=torch.float32))
    )


def _token_columns(
    token_ids: _TokenIds, counts: npt.ArrayLike | None = None
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """Pack accepted tokens into native output columns without Python scalars."""
    if isinstance(token_ids, torch.Tensor):
        # A graph may reuse its device output immediately after this call.
        # Preserve its dtype until host validation; narrowing on CUDA could hide
        # invalid values and launch an otherwise unnecessary conversion kernel.
        token_ids = token_ids.detach().to(device="cpu", copy=True).numpy()
    if isinstance(token_ids, np.ndarray):
        if token_ids.ndim not in (1, 2):
            raise ValueError(
                "output tokens must have shape [rows] or [rows, positions]"
            )
        rows = token_ids.shape[0]
        width = 1 if token_ids.ndim == 1 else token_ids.shape[1]
        if counts is None:
            lengths = np.full(rows, width, np.uint32)
            ids = token_ids.reshape(-1)
        else:
            lengths = np.asarray(counts)
            if (
                lengths.ndim != 1
                or len(lengths) != rows
                or lengths.dtype.kind not in "iu"
            ):
                raise ValueError(
                    "accepted counts must contain one integer per output row"
                )
            if np.any(lengths < 0) or np.any(lengths > width):
                raise ValueError("accepted counts must fit the output token width")
            lengths = lengths.astype(np.uint32, copy=False)
            keep = np.arange(width)[None, :] < lengths[:, None]
            ids = token_ids.reshape(rows, width)[keep]
        if ids.dtype.kind not in "iu":
            raise TypeError("output token ids must be integers")
        if ids.size and (ids.min() < 0 or ids.max() > 0xFFFFFFFF):
            raise ValueError("output token ids must fit uint32")
        ids = np.ascontiguousarray(ids, dtype=np.uint32)
    else:
        if counts is not None:
            raise TypeError("accepted counts require a token array")
        rows = (
            [[token] for token in token_ids]
            if token_ids and np.isscalar(token_ids[0])
            else token_ids
        )
        rows = cast(Sequence[Sequence[int]], rows)
        lengths = np.fromiter(map(len, rows), dtype=np.uint32)

        def checked_tokens() -> Iterator[int | np.integer]:
            for row in rows:
                for token in row:
                    if not isinstance(token, (int, np.integer)) or isinstance(
                        token, (bool, np.bool_)
                    ):
                        raise TypeError("output token ids must be integers")
                    if not 0 <= token <= 0xFFFFFFFF:
                        raise ValueError("output token ids must fit uint32")
                    yield token

        ids = np.fromiter(checked_tokens(), dtype=np.uint32)
    return ids, _offsets(lengths)


def token_output(
    token_ids: _TokenIds, *, counts: npt.ArrayLike | None = None
) -> ForwardOutput:
    """Materialize token-only results using the shared array-based output contract."""
    from resin import ForwardOutput

    ids, offsets = _token_columns(token_ids, counts)
    return ForwardOutput(token_ids=ids, token_indptr=offsets)


def format_output(
    logits: torch.Tensor | None,
    token_ids: _TokenIds,
    plan: BatchPlan,
    *,
    prompt_logprobs: PromptScores | None = None,
) -> ForwardOutput:
    """Materialize accepted tokens and optional raw-model logprob columns.

    Logits have shape [rows, vocab] or [rows, verification positions, vocab].
    Scores use the untempered model distribution, before top-k/top-p filtering.
    Only accepted output positions are returned; unsampled rows carry no scores.
    Token-only output uses the same columns without scoring or transferring logits.
    """
    from resin import ForwardOutput

    ids, offsets = _token_columns(token_ids)
    lengths = np.diff(offsets)
    prompt_columns: dict[str, Any] = {}
    if plan.needs_prompt_logprobs:
        if prompt_logprobs is None:
            raise ValueError("prefill did not provide requested prompt scores")
        prompt_columns = prompt_logprobs.output(plan)
    if not plan.needs_logprobs:
        return ForwardOutput(token_ids=ids, token_indptr=offsets, **prompt_columns)
    logits = cast(torch.Tensor, logits)
    if logits.ndim == 2:
        logits = logits[:, None, :]
    if logits.ndim != 3 or len(lengths) != logits.shape[0]:
        raise ValueError("logits must align with output token rows")
    if np.any(lengths > logits.shape[1]):
        raise ValueError("verification logits do not cover the accepted token row")

    policies = plan.sampling_params
    requested = np.fromiter(
        (
            sample and params.logprobs
            for sample, params in zip(plan.sampling_mask, policies, strict=True)
        ),
        dtype=np.bool_,
    )
    scored_lengths = lengths * requested
    score_offsets = np.concatenate(
        (np.zeros(1, np.uint32), scored_lengths.cumsum(dtype=np.uint32))
    )
    row_ids = np.repeat(np.arange(len(lengths)), scored_lengths)
    positions = np.arange(len(row_ids)) - np.repeat(score_offsets[:-1], scored_lengths)
    device = logits.device
    if len(row_ids) == len(lengths) and np.all(scored_lengths == 1):
        values = logits[:, 0].float()
    else:
        values = logits[
            torch.as_tensor(row_ids, device=device),
            torch.as_tensor(positions, device=device),
        ].float()
    selected = torch.as_tensor(
        ids[offsets[row_ids] + positions].astype(np.int64), device=device
    )
    counts = np.fromiter(
        (min(params.top_logprobs, values.shape[-1]) for params in policies),
        dtype=np.uint32,
    )[row_ids]
    scores = _score_distribution(values, selected, counts)
    columns = {} if scores.top is None else scores.top.output("top")
    return ForwardOutput(
        token_ids=ids,
        token_indptr=offsets,
        logprobs=scores.logprobs.detach().cpu().numpy(),
        logprob_indptr=score_offsets,
        **columns,
        **prompt_columns,
    )


class _SamplingInputs(TypedDict):
    temperatures: tuple[float, ...]
    top_ps: tuple[float, ...]
    top_ks: tuple[int, ...]
    seeds: tuple[int, ...]
    offsets: tuple[int, ...]
    sampling_mask: npt.NDArray[np.bool_]


def batch_sampling_inputs(plan: BatchPlan, vocab_size: int) -> _SamplingInputs:
    """Extract backend-friendly primitive arrays from one immutable plan."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    params: tuple[SamplingParams, ...] = tuple(plan.sampling_params)
    rows = plan.num_sequences
    indptr = plan.token_indptr
    context_lengths = plan.context_lengths
    if len(params) != rows:
        raise ValueError("sampling params must contain one item per sequence")
    if len(context_lengths) != rows:
        raise ValueError("context_lengths must contain one item per sequence")
    if len(indptr) != rows + 1 or indptr[0] != 0 or indptr[-1] != len(plan.token_ids):
        raise ValueError("token_indptr does not partition token_ids")
    offsets = tuple(
        (int(context_lengths[index]) + int(indptr[index + 1] - indptr[index]))
        * SAMPLING_OFFSET_STRIDE
        for index in range(rows)
    )
    seeds = []
    for item in params:
        if item.seed is None:
            raise ValueError("sampling params must contain effective seeds")
        seeds.append(item.seed)
    return {
        "temperatures": tuple(float(item.temperature) for item in params),
        "top_ps": tuple(float(item.top_p) for item in params),
        "top_ks": tuple(
            min(vocab_size, vocab_size if item.top_k is None else int(item.top_k))
            for item in params
        ),
        "seeds": tuple(seeds),
        "offsets": offsets,
        "sampling_mask": plan.sampling_mask,
    }


@torch.compile(  # ty: ignore[no-matching-overload]
    dynamic=True, fullgraph=True, recompile_limit=128
)
def _sampling_distribution(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    nucleus: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare probabilities without capturing the row-local random draw."""
    torch._dynamo.mark_static(logits, -1)
    scores = logits.float() / temperature
    if top_k < scores.numel() and not nucleus:
        scores, token_ids = torch.topk(scores, top_k, sorted=True)
    else:
        scores, token_ids = torch.sort(scores, descending=True)
    probabilities = torch.softmax(scores, dim=-1)
    if nucleus:
        # Select the nucleus from the full temperature distribution before
        # applying the top-k limit and normalizing its retained mass.
        cumulative = probabilities.cumsum(dim=-1)
        probabilities = probabilities * (cumulative - probabilities < top_p)
        probabilities = probabilities[:top_k]
        token_ids = token_ids[:top_k]
        probabilities = probabilities / probabilities.sum()
    return probabilities, token_ids


def sampling_seed(seed: int, offset: int) -> int:
    """Mix a request seed and absolute sampling offset for reference backends."""
    value = (seed + offset * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


class TorchSampler:
    """Readable reference sampler with row-stable stateless randomness."""

    def __call__(
        self,
        logits: torch.Tensor,
        plan: BatchPlan | None = None,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if plan is None:
            result = logits.argmax(dim=-1)
            if out is None:
                return result
            out.copy_(result)
            return out
        if logits.ndim != 2:
            raise ValueError("sampling logits must have shape [batch, vocab]")
        values = batch_sampling_inputs(plan, logits.shape[-1])
        if logits.shape[0] != len(values["temperatures"]):
            raise ValueError("sampling logits must contain one row per plan sequence")

        result = logits.argmax(dim=-1)
        sampled = [
            row
            for row, (temperature, should_sample) in enumerate(
                zip(values["temperatures"], values["sampling_mask"], strict=True)
            )
            if should_sample and temperature != 0.0
        ]
        if sampled:
            temperatures = torch.as_tensor(
                values["temperatures"], dtype=torch.float32, device=logits.device
            )
            top_ps = torch.as_tensor(
                values["top_ps"], dtype=torch.float32, device=logits.device
            )
            for row in sampled:
                probabilities, token_ids = _sampling_distribution(
                    logits[row],
                    temperatures[row],
                    top_ps[row],
                    values["top_ks"][row],
                    values["top_ps"][row] < 1.0,
                )
                generator = torch.Generator(device=logits.device)
                generator.manual_seed(
                    sampling_seed(values["seeds"][row], values["offsets"][row])
                )
                selected = torch.multinomial(
                    probabilities, 1, generator=generator
                ).squeeze(0)
                result[row] = token_ids[selected]

        if out is None:
            return result
        out.copy_(result)
        return out


__all__ = [
    "SAMPLING_OFFSET_STRIDE",
    "PromptScores",
    "TorchSampler",
    "batch_sampling_inputs",
    "format_output",
    "prompt_scores",
    "sampling_seed",
    "token_output",
]
