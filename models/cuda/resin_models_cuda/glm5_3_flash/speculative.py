"""Native NextN block speculation with KDA-only state rollback.

Token/hidden alignment follows SGLang 5971f47bc8eacd8989cb6dd67e46b4fc6dbec2ad
``eagle_worker_v2.py`` and ``deepseek_nextn.py`` (Apache-2.0): hidden h_i
and embedding x_(i+1) occupy draft cache position i. All acceptance decisions
remain tensors; the runner owns graph capture, sampling, and completion fences.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from resin.backend.cuda.completion import DeviceAcceptance
from resin_models_cuda.glm5_3_flash import kernels
from resin_models_cuda.glm5_3_flash.attention import rms_norm


@dataclass
class KdaRecord:
    layer: Any
    initial: Tensor
    history: Tensor
    raw_qkv: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
    forget: Tensor
    beta: Tensor
    a_log: Tensor
    dt_bias: Tensor
    lower_bound: float | None


class SpeculativeTape:
    """Retain projections, then replay only accepted recurrent updates.

    Unlike retaining every intermediate recurrent state, storage is independent
    of draft length times the large K×V state. Initial states are already owned
    by the batch and FLA never mutates them. Accepted lengths are runtime inputs.
    """

    def __init__(self, batch, width: int, *, reuse_initial_state: bool = False):
        self.batch, self.width = batch, width
        self.records: list[KdaRecord] = []
        self.reuse_initial_state = reuse_initial_state
        self._owned_regions = []
        self._page_storage = None
        if reuse_initial_state:
            arena = getattr(
                getattr(batch, "device_cache", None), "recurrent_states", None
            )
            if not isinstance(arena, Tensor):
                raise ValueError("KDA scratch reuse requires the Engine cache arena")
            self._page_storage = arena.untyped_storage().data_ptr()

    @classmethod
    def attach(cls, batch, width: int, *, reuse_initial_state: bool = False):
        if not 1 <= width <= 8:
            raise ValueError("KDA speculative blocks must contain 1–8 tokens")
        tape = cls(batch, width, reuse_initial_state=reuse_initial_state)
        for layer in batch.layers.values():
            if layer.recurrent is not None:
                layer.speculative_tape = tape
        return tape

    def record(self, layer, raw_qkv, q, k, v, forget, beta, a_log, dt_bias, bound):
        if self.reuse_initial_state:
            initial = layer.recurrent
            if initial.dtype != torch.float32 or not initial.is_contiguous():
                raise ValueError(
                    "KDA scratch reuse requires packed FP32 initial states"
                )
            pointer = initial.untyped_storage().data_ptr()
            if pointer == self._page_storage:
                raise ValueError(
                    "KDA scratch reuse cannot overwrite Engine cache pages"
                )
            start = initial.storage_offset()
            end = start + initial.numel()
            if any(
                pointer == prior and start < stop and begin < end
                for prior, begin, stop in self._owned_regions
            ):
                raise ValueError("KDA scratch reuse requires disjoint layer states")
            self._owned_regions.append((pointer, start, end))
        self.records.append(
            KdaRecord(
                layer,
                layer.recurrent,
                layer.convolution,
                raw_qkv,
                q,
                k,
                v,
                forget,
                beta,
                a_log,
                dt_bias,
                bound,
            )
        )

    def commit(self, counts: Tensor, hidden: Tensor) -> Tensor:
        """Restore accepted KDA states and return the final accepted raw hidden.

        ``counts`` includes the anchor target input: range [1,width]. The target
        was evaluated once over the entire block. Sparse cache rows beyond the
        accepted prefix remain invisible under causal selection and are replaced
        when their positions are next written, including partial pool boundaries.
        """
        from resin_models_cuda.glm5_3_flash.index_tail import commit_tail

        for layer in self.batch.layers.values():
            commit_tail(layer, counts)
        rows = counts.shape[0]
        accepted = None
        if not counts.is_cuda:
            within = torch.arange(self.width, device=counts.device)[None, :]
            accepted = (within < counts[:, None]).flatten()
        for record in self.records:
            if counts.is_cuda:
                from resin_models_cuda.glm5_3_flash.kda_state_cuda import (
                    accepted_kda_state,
                )
                from resin_models_cuda.glm5_3_flash.speculative_cuda import (
                    select_history,
                )

                # Only state survives verification. Avoid producing discarded
                # attention outputs or executing rejected recurrent updates.
                reuse: dict[str, Any] = (
                    {"out": record.initial, "initial_is_owned": True}
                    if self.reuse_initial_state
                    else {}
                )
                record.layer.recurrent = accepted_kda_state(
                    record.key,
                    record.value,
                    record.forget,
                    record.beta,
                    record.a_log,
                    record.dt_bias,
                    record.initial,
                    record.layer.cu_seqlens,
                    counts,
                    record.lower_bound,
                    **reuse,
                )
                history = select_history(
                    counts,
                    self.width,
                    record.history,
                    record.raw_qkv,
                )
            else:
                # Independent reference: mask rejected updates and execute the
                # original full recurrence, retaining only its final state.
                assert accepted is not None
                forget = record.forget.masked_fill(~accepted[:, None, None], -torch.inf)
                # Keep raw logits: kernels.kda applies sigmoid in FP32 for both
                # verification and replay; rejected logits produce exactly zero.
                beta = record.beta.masked_fill(~accepted[:, None], -torch.inf)
                replay = SimpleNamespace(
                    recurrent=record.initial,
                    cu_seqlens=record.layer.cu_seqlens,
                    cu_seqlens_cpu=None,
                )
                _, record.layer.recurrent = kernels.kda(
                    record.query,
                    record.key,
                    record.value,
                    forget,
                    beta,
                    record.a_log,
                    record.dt_bias,
                    record.lower_bound,
                    replay,
                )
                source = torch.cat(
                    (
                        record.history.permute(0, 3, 1, 2),
                        record.raw_qkv.reshape(
                            rows, self.width, *record.raw_qkv.shape[1:]
                        ),
                    ),
                    dim=1,
                )
                history_indices = counts[:, None] + torch.arange(
                    3, device=counts.device
                )
                history = (
                    source[
                        torch.arange(rows, device=counts.device)[:, None],
                        history_indices,
                    ]
                    .permute(0, 2, 3, 1)
                    .contiguous()
                )
            record.layer.convolution = history
            # Avoid a tape→record→layer→tape cycle retaining GPU projections
            # between uncaptured decode steps until Python's cyclic GC runs.
            record.layer.speculative_tape = None
        if counts.is_cuda:
            from resin_models_cuda.glm5_3_flash.speculative_cuda import finish_commit

            selected, self.batch.lengths = finish_commit(
                counts, self.width, self.batch.context_lengths, hidden
            )
            return selected
        self.batch.lengths = self.batch.context_lengths + counts
        hidden = hidden.reshape(rows, self.width, -1)
        return hidden[torch.arange(rows, device=counts.device), counts.long() - 1]


class MtpCache:
    """A draft-only metadata view sharing the Engine-owned sparse pages."""

    def __init__(self, layer_id, template, positions, sequences, offsets, contexts):
        self.layer_id = layer_id
        self.view = replace(
            template,
            positions=positions,
            sequence_indices=sequences,
            cu_seqlens=offsets,
            context_lengths=contexts,
            lengths=contexts + offsets[1:] - offsets[:-1],
        )

    def layer(self, layer_id):
        if layer_id != self.layer_id:
            raise ValueError("MTP cache cannot supply a target attention layer")
        return self.view


def prime_mtp(
    model, batch, token_ids, raw_hidden, positions_cpu, *, input_embeddings=None
):
    """Prime known prompt pairs, including the preceding chunk's final pair.

    Host positions come from the Engine plan, before capture. The final prompt
    hidden is paired with the generated anchor by the first draft step. This
    avoids guessing the token after an intermediate prefill chunk. Vision token
    embeddings, when supplied, follow the same shift as ordinary embeddings.
    """
    if model.mtp is None:
        raise RuntimeError("native MTP weights were not loaded")
    rows_cpu = [i for i, position in enumerate(positions_cpu) if position > 0]
    if not rows_cpu:
        # Attention-DP replicas must still join the MTP expert collective when
        # another replica has prompt pairs to prime (e.g. a one-token prompt).
        model.forward_mtp_experts_empty()
        return
    if batch.cu_seqlens_cpu is None:
        raise ValueError("MTP priming requires host sequence offsets")
    device = token_ids.device
    rows = torch.tensor(rows_cpu, device=device, dtype=torch.long)
    sequences = batch.sequence_indices[rows]
    previous = raw_hidden[(rows - 1).clamp_min(0)]
    is_start = rows == batch.cu_seqlens[sequences]
    previous = torch.where(
        is_start[:, None], batch.previous_hidden[sequences], previous
    )
    previous = rms_norm(previous, model.norm, model.config.rms_norm_eps)
    offsets_cpu = batch.cu_seqlens_cpu.tolist()
    offsets = [0]
    for start, end in zip(offsets_cpu, offsets_cpu[1:]):
        offsets.append(offsets[-1] + sum(p > 0 for p in positions_cpu[start:end]))
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    positions = batch.positions[rows] - 1
    layer_id = model.config.num_hidden_layers
    cache = MtpCache(
        layer_id,
        batch.layer(layer_id),
        positions,
        sequences,
        offsets,
        (batch.context_lengths - 1).clamp_min(0),
    )
    # Logits are unnecessary for priming; invoking the block directly avoids a
    # vocabulary projection over the entire prompt.
    embedded = (
        torch.nn.functional.embedding(token_ids[rows], model.embedding)
        if input_embeddings is None
        else input_embeddings[rows]
    )
    model.mtp(embedded, previous, positions, cache)


def speculative_step(
    model,
    device_cache,
    anchors: Tensor,
    positions: Tensor,
    draft_tokens: int,
    sample,
    *,
    capacity=None,
    phase_timing=None,
    projection_rows=0,
):
    """One greedy draft chain, one target block, and one native-MTP repair block.

    ``sample(logits)`` samples the complete [rows, width, vocab] verification
    block with deterministic per-position RNG offsets. Greedy proposals are accepted only
    when they equal those target samples, preserving the target distribution.
    ``capacity(width)`` optionally enters the runner's EP dispatch graph context.
    Returns the shared accepted block and target logits.
    """
    if not 1 <= draft_tokens <= 7:
        raise ValueError("native MTP currently supports 1–7 draft tokens")
    width, rows = draft_tokens + 1, anchors.numel()
    device = anchors.device
    sequences = torch.arange(rows, device=device, dtype=torch.long)
    unit_offsets = torch.arange(rows + 1, device=device, dtype=torch.int32)
    block_positions = (
        positions[:, None] + torch.arange(width, device=device)
    ).flatten()
    block_sequences = sequences.repeat_interleave(width)
    block_offsets = unit_offsets * width
    batch = device_cache.batch(
        block_positions, block_sequences, block_offsets, positions
    )
    batch.is_decode = True
    batch.projection_rows = projection_rows
    layer_id = model.config.num_hidden_layers
    template = batch.layer(layer_id)
    hidden = rms_norm(batch.previous_hidden, model.norm, model.config.rms_norm_eps)
    current = anchors.flatten()
    drafts = []
    repair_initial_tail = None
    if phase_timing is not None:
        phase_timing.mark(0)
    for step in range(draft_tokens):
        draft_positions = positions + step - 1
        cache = MtpCache(
            layer_id,
            template,
            draft_positions,
            sequences,
            unit_offsets,
            draft_positions,
        )
        with capacity(1) if capacity is not None else nullcontext():
            logits, hidden = model.mtp_forward(current, hidden, draft_positions, cache)
        if step == 0 and getattr(template, "index_tail", None) is not None:
            # The first draft's p-1 pair is exact; later recursive pairs are
            # provisional. Repair must start from this owned tail snapshot.
            repair_initial_tail = template.index_tail.snapshot()
        current = logits.argmax(-1)
        drafts.append(current)
    drafts = torch.stack(drafts, dim=1)
    if phase_timing is not None:
        phase_timing.mark(1)
    verify_ids = torch.cat((anchors.reshape(rows, 1), drafts), dim=1)
    tape = SpeculativeTape.attach(batch, width, reuse_initial_state=anchors.is_cuda)
    with capacity(width) if capacity is not None else nullcontext():
        verified_hidden = model.forward_hidden(
            verify_ids.flatten(), block_positions, batch
        )
    logits = model.logits(verified_hidden, projection_rows=projection_rows).reshape(
        rows, width, -1
    )
    if phase_timing is not None:
        phase_timing.mark(2)
    targets = sample(logits)
    acceptance = DeviceAcceptance.from_samples(drafts, targets, positions)
    accepted, counts = acceptance.accepted_drafts, acceptance.counts
    selected_hidden = tape.commit(counts, verified_hidden)
    # Rebuild the native MTP cache using verified target states, replacing the
    # approximate recursively generated draft states. Only accepted positions
    # can be observed subsequently; spare writes are inside the lookahead lease.
    bonus = targets.gather(1, accepted[:, None])
    repair_ids = torch.cat((drafts, targets[:, -1:]), dim=1)
    repair_ids = repair_ids.scatter(1, accepted[:, None], bonus)
    if repair_initial_tail is not None:
        template.index_tail.assign(repair_initial_tail)
    repair = MtpCache(
        layer_id, template, block_positions, block_sequences, block_offsets, positions
    )
    previous = rms_norm(verified_hidden, model.norm, model.config.rms_norm_eps)
    with capacity(width) if capacity is not None else nullcontext():
        embedded = torch.nn.functional.embedding(repair_ids.flatten(), model.embedding)
        model.mtp(embedded, previous, block_positions, repair)
    from resin_models_cuda.glm5_3_flash.index_tail import commit_tail

    commit_tail(repair.view, counts)
    # Persist both the accepted target tails and the repaired native-MTP tail
    # together with the recurrent state. Prefix snapshots now own all history.
    batch.store(selected_hidden)
    if phase_timing is not None:
        phase_timing.mark(3)
    return acceptance, logits
