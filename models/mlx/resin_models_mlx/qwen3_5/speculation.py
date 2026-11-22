"""Qwen MTP verification over the MLX runner's leased KV and live state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from torch.profiler import record_function

from resin.backend.sampling import format_output
from resin.backend.speculative import SpeculativeStats
from resin_models_mlx.qwen3_5.cache import MTP_GROUP, STATE_GROUP
from resin_models_mlx.runner import _to_torch

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput
    from resin.backend.plans import PagedCachePlan
    from resin_models_mlx.live_state import LiveStates
    from resin_models_mlx.qwen3_5.cache import CacheArena
    from resin_models_mlx.qwen3_5.media import MediaEncoder
    from resin_models_mlx.qwen3_5.model import BlockInputs, Qwen35Model
    from resin_models_mlx.runner import _Launch
    from resin_models_mlx.sampling import MlxSampler


@dataclass(frozen=True, slots=True)
class SpeculativeFeedback:
    sequence_ids: tuple[int, ...]
    anchors: mx.array
    positions: mx.array


@dataclass(slots=True)
class _Step:
    plan: BatchPlan
    drafts: mx.array
    targets: mx.array
    accepted: mx.array
    logits: mx.array
    receipts: tuple[int, ...]
    epoch: mx.array
    next_feedback: SpeculativeFeedback
    state: tuple[mx.array, ...]
    stats: SpeculativeStats
    feedback: SpeculativeFeedback | None = None
    _result: ForwardOutput | None = field(default=None, init=False)

    @property
    def arrays(self) -> tuple[mx.array, ...]:
        return (
            self.drafts,
            self.targets,
            self.accepted,
            self.next_feedback.anchors,
            self.next_feedback.positions,
            *self.state,
        )

    def result(self) -> ForwardOutput:
        if self._result is None:
            mx.eval(self.epoch, *self.arrays)
            acceptance = self.stats.accept(
                np.array(self.drafts).tolist(), np.array(self.targets).tolist()
            )
            if acceptance.accepted_counts != np.array(self.accepted).tolist():
                error = RuntimeError(
                    "MLX speculative acceptance diverged from the host"
                )
                error._resin_post_accept = True  # ty: ignore[unresolved-attribute]
                raise error
            self._result = format_output(
                _to_torch(self.logits) if self.plan.needs_logprobs else None,
                acceptance.token_rows,
                self.plan,
            ).with_state_checkpoints(self.receipts)
        return self._result


class MtpDecode:
    def __init__(
        self,
        model: Qwen35Model,
        cache: CacheArena,
        states: LiveStates,
        sampler: MlxSampler,
        media: MediaEncoder | None,
        drafts: int,
        max_seq_len: int,
    ) -> None:
        self.model, self.cache, self.states = model, cache, states
        self.sampler, self.media = sampler, media
        self.drafts, self.max_seq_len = drafts, max_seq_len
        self.stats = SpeculativeStats(drafts)
        self._last_hidden: dict[int, mx.array] = {}
        self._proposals: dict[int, tuple[mx.array, mx.array]] = {}

    def release(self, sequences: list[int]) -> None:
        for sequence in sequences:
            self._last_hidden.pop(sequence, None)
            self._proposals.pop(sequence, None)

    def prefill(self, launch: _Launch) -> None:
        """MTP slot p pairs target hidden state p-1 with token p; slot zero is empty."""
        plan, rows, tokens = launch.plan, launch.rows, launch.tokens
        batch = plan.batch
        row = int(rows[0])
        sequence = int(batch.sequence_ids[row])
        start = int(batch.context_lengths[row])
        hidden = launch.step.hidden
        token_ids = mx.array(batch.token_ids[tokens].astype(np.int32))
        if start == 0:
            pair_ids, previous, first = token_ids[:, 1:], hidden[:, :-1], 1
        else:
            # A prefix hit has no predecessor hidden state. A zero stand-in
            # can affect proposal quality, but cannot change target verification.
            last = self._last_hidden.get(sequence)
            if last is None:
                last = mx.zeros(hidden.shape[-1:], dtype=hidden.dtype)
            pair_ids, first = token_ids, start
            previous = mx.concatenate([last[None, None], hidden[:, :-1]], axis=1)
        last = self._last_hidden[sequence] = mx.contiguous(hidden[0, -1])
        mx.async_eval(last)
        if pair_ids.shape[1] > 0:
            key, value = self.model.mtp.keys_values(
                self.model.embed_tokens(pair_ids),
                previous,
                mx.array([first - 1], dtype=mx.int32),
            )
            self.cache.write_positions(
                plan,
                rows,
                np.arange(first, start + tokens.shape[1])[None],
                (key,),
                (value,),
                MTP_GROUP,
            )

    def _mtp(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        token_ids: mx.array,
        previous: mx.array,
        first: mx.array,
        *,
        written: mx.array | None = None,
        prior: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        history = self.cache.leased_history(
            plan, rows, first if written is None else written, MTP_GROUP, skip=1
        )
        if prior is not None:
            history = replace(history, prior=prior)
        hidden, key, value = self.model.mtp(
            self.model.embed_tokens(token_ids), previous, first - 1, history
        )
        self.cache.order_after([hidden])
        return hidden, key, value

    def _first_proposals(self, plan, sequences, anchors, positions, bounds, exact):
        missing = np.array(
            [
                row
                for row, sequence in enumerate(sequences)
                if sequence not in self._proposals
            ],
            dtype=np.int64,
        )
        if not len(missing):
            return
        if not exact:
            raise RuntimeError("an MLX speculative continuation lost its draft")
        zeros = mx.zeros((self.model.config.hidden_size,), dtype=mx.bfloat16)
        previous = mx.stack(
            [self._last_hidden.get(sequences[row], zeros) for row in missing]
        )[:, None]
        selected = mx.array(missing)
        slots = positions[selected]
        hidden, key, value = self._mtp(
            plan, missing, anchors[selected][:, None], previous, slots
        )
        self.cache.write_leased(
            plan, missing, slots[:, None], bounds[missing], (key,), (value,), MTP_GROUP
        )
        proposals = self.sampler.block_tokens(
            self.model.logits(hidden),
            plan.batch._plan,
            rows=missing,
            positions=positions,
        )[:, 0]
        for index, row in enumerate(missing.tolist()):
            self._proposals[sequences[row]] = proposals[index], hidden[index, 0]

    def _draft(self, plan, rows, sequences, positions):
        columns = [mx.stack([self._proposals[sequence][0] for sequence in sequences])]
        draft_hidden = mx.stack(
            [self._proposals[sequence][1] for sequence in sequences]
        )
        prior = None
        for offset in range(1, self.drafts):
            hidden, key, value = self._mtp(
                plan,
                rows,
                columns[-1][:, None],
                draft_hidden[:, None],
                positions + offset,
                written=positions + 1,
                prior=prior,
            )
            prior = (
                (key, value)
                if prior is None
                else (
                    mx.concatenate([prior[0], key], axis=1),
                    mx.concatenate([prior[1], value], axis=1),
                )
            )
            draft_hidden = hidden[:, 0]
            columns.append(
                self.sampler.block_tokens(
                    self.model.logits(hidden),
                    plan.batch._plan,
                    offset,
                    positions=positions,
                )[:, 0]
            )
        return mx.stack(columns, axis=1).astype(mx.int32)

    def _launch(
        self, plan: PagedCachePlan, feedback: SpeculativeFeedback | None
    ) -> _Step:
        """Verify [anchor, drafts] once and commit the accepted DeltaNet prefix."""
        batch = plan.batch
        native = batch._plan
        has_media = any(media is not None for media in native.media)
        if has_media and self.media is None:
            raise ValueError("this Qwen3.5 runner did not load a vision tower")
        if has_media and feedback is not None:
            raise RuntimeError("image rows do not continue on the GPU")
        rows = np.arange(batch.num_sequences)
        sequences = tuple(map(int, batch.sequence_ids))
        bounds = batch.context_lengths.astype(np.int64)
        exact = feedback is None
        if feedback is None:
            anchors = mx.array(batch.token_ids.astype(np.int32))
            positions = mx.array(bounds.astype(np.int32))
        else:
            if feedback.sequence_ids != sequences:
                raise RuntimeError("an MLX speculative continuation changed its rows")
            anchors, positions = feedback.anchors, feedback.positions

        with record_function("resin.mlx.draft"):
            self._first_proposals(plan, sequences, anchors, positions, bounds, exact)
            drafts = self._draft(plan, rows, sequences, positions)
            mx.async_eval(drafts)

        with record_function("resin.mlx.target_verify"):
            verify_ids = mx.concatenate([anchors[:, None], drafts], axis=1)
            embeddings = rotary = None
            if has_media:
                assert self.media is not None
                embeddings, rotary = self.media.prepare(
                    native,
                    rows,
                    np.zeros(verify_ids.shape, np.int64),
                    bounds,
                    self.model.embed_tokens(verify_ids),
                )
            retained: list[BlockInputs] = []
            output = self.model(
                verify_ids,
                positions,
                self.states.gather(plan, rows, exact=exact),
                self.cache.leased_history(plan, rows, positions),
                input_embeddings=embeddings,
                rotary_positions=rotary,
                retain=retained,
            )
            logits = self.model.logits(output.hidden)
            targets = self.sampler.block_tokens(logits, native, positions=positions)
            self.cache.order_after(
                [
                    logits,
                    output.hidden,
                    *output.keys,
                    *output.values,
                    *output.state.recurrent,
                    *output.state.convolution,
                    *(inputs.recurrent for inputs in retained),
                    *(inputs.window for inputs in retained),
                ]
            )
            accepted = mx.cumprod(
                (drafts == targets[:, : self.drafts]).astype(mx.int32), axis=1
            ).sum(axis=1)
            mx.async_eval(accepted, targets)

        with record_function("resin.mlx.commit"):
            block = positions[:, None] + mx.arange(self.drafts + 1, dtype=mx.int32)
            self.cache.write_leased(
                plan, rows, block, bounds + self.drafts, output.keys, output.values
            )
            committed = self.model.commit_block(retained, accepted)
            ends = positions + accepted.astype(mx.int32) + 1
            receipts = self.states.commit(plan, rows, ends, committed)

        with record_function("resin.mlx.draft"):
            # Fill shifted MTP slots from the verified target hidden states.
            hidden, key, value = self._mtp(
                plan, rows, targets.astype(mx.int32), output.hidden, positions + 1
            )
            self.cache.write_leased(
                plan,
                rows,
                block + 1,
                bounds + self.drafts + 1,
                (key,),
                (value,),
                MTP_GROUP,
            )
            selected = mx.arange(len(sequences))
            next_hidden = hidden[selected, accepted]
            proposals = self.sampler.chosen_tokens(
                self.model.logits(next_hidden),
                native,
                1,
                self.drafts + 1,
                accepted,
                positions=positions,
            )
            for row, sequence in enumerate(sequences):
                self._proposals[sequence] = proposals[row], next_hidden[row]
        return _Step(
            native,
            drafts,
            targets,
            accepted,
            logits,
            receipts,
            self.cache.epoch,
            SpeculativeFeedback(
                sequences, targets[selected, accepted].astype(mx.int32), ends
            ),
            (proposals, next_hidden, *committed.recurrent, *committed.convolution),
            self.stats,
        )

    def forward(self, plan: PagedCachePlan) -> ForwardOutput:
        with record_function("resin.mlx.speculative_decode"):
            return self._launch(plan, None).result()

    def enqueue(
        self, plan: PagedCachePlan, feedback: SpeculativeFeedback | None
    ) -> _Step | ForwardOutput:
        native = plan.batch._plan
        checkpoints = plan.groups[STATE_GROUP].state
        if (
            native.needs_logprobs
            or any(media is not None for media in native.media)
            or (checkpoints is not None and len(checkpoints.checkpoint_slots))
        ):
            if feedback is not None:
                raise RuntimeError("an MLX speculative continuation needs host state")
            return self.forward(plan)
        with record_function("resin.mlx.speculative_enqueue"):
            step = self._launch(plan, feedback)
            mx.async_eval(step.epoch, *step.arrays)
        span = 2 * (self.drafts + 1) + self.cache.layout.decode_write_extra_tokens
        if int(np.max(plan.batch.context_lengths)) + span <= self.max_seq_len:
            step.feedback = step.next_feedback
        return step
