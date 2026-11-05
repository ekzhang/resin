"""Qwen3.5 runner on Apple Silicon over Engine-owned MLX slabs."""

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
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelismCapabilities
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.qwen3_5.cache import (
    KV_GROUP,
    MTP_GROUP,
    PREFILL_CHUNK_SIZE,
    CacheArena,
    CacheGeometry,
)
from resin_models_mlx.qwen3_5.checkpoint import checkpoint_quantization, load_model
from resin_models_mlx.qwen3_5.model import (
    BlockInputs,
    PagedHistory,
    Qwen35Model,
    StepOutput,
)
from resin_models_mlx.qwen3_5.state import LiveStates
from resin_models_mlx.sampling import MlxSampler

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput, ResourceBinding

_WORKSPACE_BYTES = 4 * 1024**3


def _to_torch(array: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(array.astype(mx.float32)))


@dataclass(slots=True)
class _Launch:
    """A step's outputs, retained through its GPU-ordered cache writes."""

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


@dataclass(slots=True)
class _Completion:
    """One overlapped decode step, collected after the next one is launched."""

    _runner: Qwen35MlxRunner
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


class Qwen35MlxRunner:
    """Qwen3.5 runner with paged attention, DeltaNet state and optional vision."""

    page_size = 128
    prefill_chunk_size = PREFILL_CHUNK_SIZE
    parallelism_capabilities = ParallelismCapabilities()

    def __init__(
        self,
        model: str | Path | Qwen35Model,
        *,
        weight_bits: int | None = 8,
        group_size: int = 64,
        memory_utilization: float = 0.85,
        cache_slab_capacity: int | None = None,
        max_seq_len: int | None = None,
        max_batch_size: int = 16,
        workspace_bytes: int = _WORKSPACE_BYTES,
        include_vision: bool = True,
        compile_vision: bool = True,
        speculative_draft_tokens: int = 0,
    ) -> None:
        self._media = None
        if workspace_bytes < 0:
            raise ValueError("workspace_bytes must be non-negative")
        if (
            isinstance(speculative_draft_tokens, bool)
            or not isinstance(speculative_draft_tokens, int)
            or speculative_draft_tokens < 0
        ):
            raise ValueError("speculative_draft_tokens must be a non-negative integer")
        self.speculative_draft_tokens = speculative_draft_tokens
        if isinstance(model, Qwen35Model):
            if speculative_draft_tokens and not model.has_mtp:
                raise ValueError("MTP decoding requires a model with its MTP head")
            self._model = model
            self.config: Qwen35Config = model.config
            self.model_path = None
            quantization = "preloaded"
            weight_source = "preloaded"
        else:
            self.model_path = Path(model)
            self._model, self.config, raw_config = load_model(
                self.model_path,
                bits=weight_bits,
                group_size=group_size,
                include_mtp=speculative_draft_tokens > 0,
            )
            packed = checkpoint_quantization(raw_config)
            bits, group = (
                (packed["bits"], packed["group_size"])
                if packed is not None
                else (weight_bits, group_size)
            )
            quantization = "bfloat16" if bits is None else f"mlx-int{bits}-g{group}"
            weight_source = (
                "bf16_checkpoint" if packed is None else "prequantized_checkpoint"
            )
            if include_vision and raw_config.get("vision_config"):
                from resin_models_mlx.qwen3_5.media import MediaEncoder
                from resin_models_mlx.qwen3_5.vision import load_vision_model

                visual = load_vision_model(
                    self.model_path, raw_config, compiled=compile_vision
                )
                self._media = MediaEncoder(visual, self.config.hidden_size)
                if compile_vision:
                    side = visual.config.spatial_merge_size * 8
                    mx.eval(
                        visual(
                            mx.zeros(
                                (side * side, visual.config.patch_width),
                                dtype=mx.bfloat16,
                            ),
                            np.array([[1, side, side]], dtype=np.int64),
                        )
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
            "weight_source": weight_source,
            "expert_backend": "mlx-gather-qmm" if self.config.is_moe else "none",
            "graphs": {"prefill": "disabled", "decode": "disabled"},
            "speculative": (
                {"method": "mtp", "draft_tokens": speculative_draft_tokens}
                if speculative_draft_tokens
                else None
            ),
        }
        self.vocab_size = self.config.vocab_size
        self.eos_token_id = self.config.eos_token_id
        self.cache_geometry = CacheGeometry.from_config(
            self.config, include_mtp=speculative_draft_tokens > 0
        )
        device = mx.device_info()
        limit = int(device["max_recommended_working_set_size"])
        self.cache_capacity_plan = plan_cache_capacity(
            limit,
            max(1, limit - mx.get_active_memory()),
            memory_utilization,
            self.cache_geometry.slab_bytes,
            # Live DeltaNet state is held for every running request, and a
            # step's output exists beside its input; a verified block also
            # holds the replayed state of its accepted prefix.
            overhead_bytes=workspace_bytes
            + (3 if speculative_draft_tokens else 2)
            * max_batch_size
            * self.cache_geometry.state_bytes,
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
        if self.config.head_dim == 256:
            from resin_models_mlx.paged_attention import _warm_fused_merge

            _warm_fused_merge()
        if (
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ) == (8, 2, 256):
            from resin_models_mlx._native_cache import _warm_paged_attention

            _warm_paged_attention()
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            max_accepted_per_step=speculative_draft_tokens + 1,
            cache_layout=self.cache_layout,
        )
        self._sampler = MlxSampler()
        self._states = LiveStates(self._cache, self._model)
        # An enqueued decode step whose arena writes have not been issued.
        self._pending: _Launch | None = None
        self._speculation = SpeculativeStats(speculative_draft_tokens)
        # Per sequence: the target's last hidden state, and the next MTP
        # proposal with the MTP hidden state that produced it.
        self._last_hidden: dict[int, mx.array] = {}
        self._drafts: dict[int, tuple[mx.array, mx.array]] = {}

    def speculative_stats(self) -> dict[str, int | float]:
        return self._speculation.snapshot(include_draft_tokens=True)

    def bind_resources(self, resources: ResourceBinding) -> None:
        self._cache.binding.bind_resources(resources)

    def release(self, sequence_ids: list[int]) -> None:
        # A pending step would hold the released rows' state again.
        self._write_pending()
        self._states.release(sequence_ids)
        for sequence in sequence_ids:
            self._last_hidden.pop(int(sequence), None)
            self._drafts.pop(int(sequence), None)

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
        embeddings = None
        rotary_positions = None
        if any(batch._plan.media[int(row)] is not None for row in rows):
            if self._media is None:
                raise ValueError("this Qwen3.5 runner did not load a vision tower")
            embeddings, rotary_positions = self._media.prepare(
                batch._plan,
                rows,
                batch.token_ids[tokens],
                contexts,
                self._model.embed_tokens(token_ids.reshape(tokens.shape)),
            )
        output = self._model(
            token_ids.reshape(tokens.shape),
            mx.array(contexts.astype(np.int32)),
            self._states.gather(plan, rows),
            history,
            input_embeddings=embeddings,
            rotary_positions=rotary_positions,
        )
        last = self._model.logits(output.hidden[:, -1])
        return _Launch(plan, rows, tokens, contexts, output, last)

    def _write(self, launch: _Launch) -> None:
        """Order writes after this graph's readers and hold its live state."""

        output = launch.output
        self._cache.order_after(launch.reads)
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
        if self.speculative_draft_tokens and not prefill:
            with record_function("resin.mlx.speculative_decode"):
                return self._speculative_decode(plan, cache_plan)
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
                if prefill and self.speculative_draft_tokens:
                    self._prefill_mtp(cache_plan, int(rows[0]), tokens, launch)
                mx.async_eval(self._cache.epoch, *launch.reads)
            logits.append(launch.last)
            assert launch.receipts is not None
            receipts += launch.receipts
            if prefill and plan.needs_prompt_logprobs:
                scores = prompt_scores(
                    plan,
                    _to_torch(launch.output.hidden[0]),
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

    def _prefill_mtp(
        self, plan: PagedCachePlan, row: int, tokens: np.ndarray, launch: _Launch
    ) -> None:
        """Write the MTP pairs whose tokens this prefill chunk schedules.

        Slot ``p`` pairs the target hidden state at ``p - 1`` with token ``p``.
        A chunk starting at ``s`` needs the hidden state at ``s - 1``: the
        previous chunk's, or zeros after a prefix hit, which does not store
        it. That only weakens one pair's proposals, never the output.
        """

        batch = plan.batch
        sequence = int(batch.sequence_ids[row])
        start, count = int(launch.contexts[0]), tokens.shape[1]
        hidden = launch.output.hidden
        token_ids = mx.array(batch.token_ids[tokens].astype(np.int32))
        if start == 0:
            # Slot 0 holds no pair.
            pair_ids, previous, first = token_ids[:, 1:], hidden[:, :-1], 1
        else:
            last = self._last_hidden.get(sequence)
            if last is None:
                last = mx.zeros((hidden.shape[-1],), dtype=hidden.dtype)
            pair_ids = token_ids
            previous = mx.concatenate([last[None, None], hidden[:, :-1]], axis=1)
            first = start
        self._last_hidden[sequence] = hidden[0, -1]
        if pair_ids.shape[1] == 0:
            return
        key, value = self._model.mtp.keys_values(
            self._model.embed_tokens(pair_ids),
            previous,
            mx.array([first - 1], dtype=mx.int32),
        )
        self._cache.write_positions(
            plan,
            np.array([row]),
            np.arange(first, start + count)[None],
            (key,),
            (value,),
            MTP_GROUP,
        )

    def _mtp_step(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        token_ids: mx.array,
        previous: mx.array,
        first: np.ndarray,
        written: np.ndarray | None = None,
        prior: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Run MTP pairs ``[rows, tokens]`` at consecutive slots ``first + j``.

        Each attends to the written pairs before slot ``written`` (default
        ``first``), then ``prior`` unwritten ones, then the block's earlier
        pairs. A pair at slot ``p`` has rotary position ``p - 1``.
        """

        history = self._cache.paged_history(
            plan, rows, first if written is None else written, MTP_GROUP, skip=1
        )
        if prior is not None:
            history = PagedHistory(
                history.units,
                history.pages,
                history.lengths,
                history.layer_stride,
                history.skip,
                prior,
            )
        hidden, key, value = self._model.mtp(
            self._model.embed_tokens(token_ids),
            previous,
            mx.array((first - 1).astype(np.int32)),
            history,
        )
        # The hidden state is the only output computed from the arena view.
        self._cache.order_after([hidden])
        return hidden, key, value

    def _speculative_decode(
        self, plan: BatchPlan, cache_plan: PagedCachePlan
    ) -> ForwardOutput:
        """Draft with MTP, verify ``[anchor, drafts]`` once, commit the accepted.

        The anchor at position ``P`` was sampled by the previous step, and its
        first draft was proposed then. Verification writes K/V for all
        ``K + 1`` leased positions; only the accepted prefix becomes readable,
        through positions. DeltaNet state is recomputed for the accepted
        prefix from the block's retained inputs. MTP then re-reads the
        verified hidden states to write slots ``P + 1 ..= P + K + 1`` and
        propose the next anchor's first draft.
        """

        batch = cache_plan.batch
        model = self._model
        drafts_per_step = self.speculative_draft_tokens
        count = batch.num_sequences
        rows = np.arange(count)
        contexts = batch.context_lengths.astype(np.int64)
        sequences = [int(sequence) for sequence in batch.sequence_ids]
        anchors = mx.array(batch.token_ids.astype(np.int32))

        missing = np.array(
            [
                row
                for row, sequence in enumerate(sequences)
                if sequence not in self._drafts
            ]
        )
        if len(missing):
            # The anchor's pair, at slot P, after prefill or a retraction.
            zeros = mx.zeros((self.config.hidden_size,), dtype=mx.bfloat16)
            previous = mx.stack(
                [self._last_hidden.get(sequences[row], zeros) for row in missing]
            )[:, None]
            slots = contexts[missing]
            hidden, key, value = self._mtp_step(
                cache_plan,
                missing,
                anchors[mx.array(missing)][:, None],
                previous,
                slots,
            )
            self._cache.write_positions(
                cache_plan, missing, slots[:, None], (key,), (value,), MTP_GROUP
            )
            proposals = self._sampler.block_tokens(
                model.logits(hidden), plan, 0, rows=missing
            )[:, 0]
            for index, row in enumerate(missing.tolist()):
                self._drafts[sequences[row]] = (proposals[index], hidden[index, 0])

        with record_function("resin.mlx.draft"):
            columns = [mx.stack([self._drafts[sequence][0] for sequence in sequences])]
            draft_hidden = mx.stack(
                [self._drafts[sequence][1] for sequence in sequences]
            )
            prior = None
            for step in range(1, drafts_per_step):
                # Chain MTP on its own hidden state and proposal, at slot P + step.
                hidden, key, value = self._mtp_step(
                    cache_plan,
                    rows,
                    columns[-1][:, None],
                    draft_hidden[:, None],
                    contexts + step,
                    contexts + 1,
                    prior,
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
                    self._sampler.block_tokens(model.logits(hidden), plan, step)[:, 0]
                )
            drafts = mx.stack(columns, axis=1).astype(mx.int32)
            # Start drafting on the GPU while verification is built.
            mx.async_eval(drafts)

        with record_function("resin.mlx.target_verify"):
            verify_ids = mx.concatenate([anchors[:, None], drafts], axis=1)
            embeddings = rotary_positions = None
            if any(batch._plan.media[int(row)] is not None for row in rows):
                if self._media is None:
                    raise ValueError("this Qwen3.5 runner did not load a vision tower")
                embeddings, rotary_positions = self._media.prepare(
                    batch._plan,
                    rows,
                    np.zeros((count, drafts_per_step + 1), dtype=np.int64),
                    contexts,
                    model.embed_tokens(verify_ids),
                )
            retained: list[BlockInputs] = []
            state = self._states.gather(cache_plan, rows)
            output = model(
                verify_ids,
                mx.array(contexts.astype(np.int32)),
                state,
                self._cache.paged_history(cache_plan, rows, contexts),
                input_embeddings=embeddings,
                rotary_positions=rotary_positions,
                retain=retained,
            )
            logits = model.logits(output.hidden)
            targets = self._sampler.block_tokens(logits, plan)
            self._cache.order_after(
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
            # Matching draft prefix, on the GPU: the rest of the step is
            # issued without waiting for it.
            accepted = mx.cumprod(
                (drafts == targets[:, :drafts_per_step]).astype(mx.int32), axis=1
            ).sum(axis=1)
            # Verify on the GPU while the commit and MTP repair are built.
            mx.async_eval(accepted, targets)

        with record_function("resin.mlx.commit"):
            positions = contexts[:, None] + np.arange(drafts_per_step + 1)
            self._cache.write_positions(
                cache_plan, rows, positions, output.keys, output.values, KV_GROUP
            )
            committed = model.commit_block(retained, accepted)
            # A due snapshot needs the exact accepted length on the host.
            ends = mx.array(contexts.astype(np.int32)) + accepted + 1
            receipts = self._states.commit(cache_plan, rows, ends, committed)
            selected = mx.arange(count)
            last_hidden = output.hidden[selected, accepted]
            for row, sequence in enumerate(sequences):
                self._last_hidden[sequence] = last_hidden[row]

        with record_function("resin.mlx.draft"):
            # Pairs (h[P + j], target[j]) fill slots P + 1 ..= P + K + 1; the
            # accepted prefix's last one proposes the next anchor's draft.
            hidden, key, value = self._mtp_step(
                cache_plan, rows, targets.astype(mx.int32), output.hidden, contexts + 1
            )
            self._cache.write_positions(
                cache_plan, rows, positions + 1, (key,), (value,), MTP_GROUP
            )
            next_hidden = hidden[selected, accepted]
            proposals = self._sampler.chosen_tokens(
                model.logits(next_hidden),
                plan,
                1,
                drafts_per_step + 1,
                accepted,
            )
            for row, sequence in enumerate(sequences):
                self._drafts[sequence] = (proposals[row], next_hidden[row])
        # A synchronous result releases its lease: finish every cache write.
        mx.eval(
            self._cache.epoch,
            drafts,
            targets,
            accepted,
            proposals,
            next_hidden,
            *committed.recurrent,
            *committed.convolution,
        )
        acceptance = self._speculation.accept(
            np.array(drafts).tolist(), np.array(targets).tolist()
        )
        if acceptance.accepted_counts != np.array(accepted).tolist():
            raise RuntimeError("MLX speculative acceptance diverged from the host")
        return format_output(
            _to_torch(logits) if plan.needs_logprobs else None,
            acceptance.token_rows,
            plan,
        ).with_state_checkpoints(receipts)

    def _cache_plan(self, plan: BatchPlan) -> PagedCachePlan:
        lookahead = (
            self.speculative_draft_tokens + self.cache_layout.decode_write_extra_tokens
            if self.speculative_draft_tokens
            else 0
        )
        cache_plan = PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=lookahead
        )
        if np.any(cache_plan.token_ids >= self.vocab_size):
            raise ValueError("token id is out of range for Qwen3.5")
        if int(np.max(cache_plan.batch.post_lengths)) > self.max_seq_len:
            raise ValueError(
                f"sequence exceeds this runner's {self.max_seq_len}-token context"
            )
        return cache_plan

    def enqueue(
        self, plan: BatchPlan, *, feedback: mx.array | None = None
    ) -> _Completion | ForwardOutput:
        """Launch a decode step while the previous one still runs on the GPU.

        ``feedback`` holds the previous step's sampled tokens, which replace
        the continuation plan's token IDs. The previous step's arena writes are
        ordered after its reads and before this step's reads on the GPU.
        """

        if (
            plan.phase != BatchPhase.DECODE
            or plan.needs_logprobs
            or self.speculative_draft_tokens
        ):
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
        continues = int(np.max(cache_plan.batch.post_lengths)) < self.max_seq_len
        return _Completion(self, plan, launch, tokens, continues)


__all__ = ["Qwen35MlxRunner"]
