"""Inkling-Small CUDA target with gamma-five DSpark decoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from resin import BatchPhase, RunnerConfig
from resin.backend.capacity import plan_cache_capacity
from resin.backend.cuda.graphs import (
    CudaGraphMode,
    CudaGraphStep,
    GraphShape,
    resolve_decode_graph_shapes,
)
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.runner import (
    CudaRunner,
    capture_sampling_inputs,
    graph_sample,
    pad_sampling_inputs,
    validate_hybrid_plan,
)
from resin.backend.cuda.sampling import (
    SAMPLING_OFFSET_STRIDE,
)
from resin.backend.cuda.staging import TensorStager
from resin.backend.plans import PagedCachePlan
from resin.backend.sampling import (
    PromptScores,
    format_output,
    prompt_scores,
    token_output,
)
from resin.backend.speculative import SpeculativeStats
from resin.backend.torch import use_default_dtype
from resin.recipe import InklingMedia
from resin_models_cuda.inkling.cache import (
    PREFILL_CHUNK_SIZE,
    STATE_WIDTH_PER_LAYER,
    CacheArena,
    CacheGeometry,
    DeviceCache,
)
from resin_models_cuda.inkling.checkpoint import (
    load_draft_config,
    load_target_config,
    resolve_draft_path,
    resolve_target_path,
)
from resin_models_cuda.inkling.dspark import DSparkDraft
from resin_models_cuda.inkling.model import (
    DecoderLayer,
    InklingForCausalLM,
    select_convolution_state,
)
from resin_models_cuda.inkling.weights import load_draft_weights, load_target_weights

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput


class InklingCudaRunner(CudaRunner):
    """Inkling-Small NVFP4 runner with image input and DSpark speculation."""

    speculative_tokens = DSparkDraft.gamma

    def __init__(
        self,
        target_model: str | Path,
        draft_model: str | Path,
        *,
        max_seq_len: int,
        max_batch_size: int = 128,
        device: str | torch.device,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        gpu_memory_utilization: float = 0.90,
        cache_slab_capacity: int | None = None,
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
        cuda_graph_decode_shapes: object | None = None,
    ) -> None:
        for name, value in (
            ("max_seq_len", max_seq_len),
            ("max_batch_size", max_batch_size),
            ("prefill_chunk_size", prefill_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if prefill_chunk_size != PREFILL_CHUNK_SIZE:
            raise ValueError("Inkling recurrent checkpoints require 4096-token chunks")
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if cache_slab_capacity is not None and (
            isinstance(cache_slab_capacity, bool)
            or not isinstance(cache_slab_capacity, int)
            or cache_slab_capacity <= 0
        ):
            raise ValueError("cache_slab_capacity must be a positive integer")
        super().__init__(
            device=device,
            label="Inkling",
            kv_cache_dtype="bfloat16",
            cuda_graph_prefill=cuda_graph_prefill,
            cuda_graph_decode=cuda_graph_decode,
        )
        torch.cuda.set_device(self.device)
        if torch.cuda.get_device_capability(self.device)[0] < 10:
            raise RuntimeError("Inkling-Small NVFP4 requires a Blackwell GPU")
        self.target_path = resolve_target_path(target_model)
        self.draft_path = resolve_draft_path(draft_model)
        self.config = load_target_config(self.target_path)
        self.draft_config = load_draft_config(self.draft_path)
        self.draft_config.validate_target(self.config)
        if max_seq_len > min(
            self.config.max_position_embeddings,
            self.draft_config.max_position_embeddings,
        ):
            raise ValueError("max_seq_len exceeds the target or draft context length")

        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
        self.physical_seq_len = max_seq_len + self.speculative_tokens
        self.vocab_size = self.config.unpadded_vocab_size
        self.eos_token_id = self.config.eos_token_id
        self._speculation = SpeculativeStats(self.speculative_tokens)
        self._graph_device_caches: dict[GraphShape, DeviceCache] = {}
        self._media_features: dict[int, tuple[InklingMedia, torch.Tensor]] = {}

        with use_default_dtype(torch.bfloat16), torch.device(self.device):
            self._model = InklingForCausalLM(
                self.config, self.draft_config.target_layer_ids
            )
            self._draft = DSparkDraft(
                self.draft_config,
                self.config,
                self._model.target,
                self.physical_seq_len,
            )
        self._model.eval()
        self._draft.eval()
        print("[inkling] loading target weights", flush=True)
        self.weight_load = {
            "target": load_target_weights(self.target_path, self._model)
        }
        print("[inkling] loading DSpark weights", flush=True)
        self.weight_load["draft"] = load_draft_weights(self.draft_path, self._draft)
        print("[inkling] preparing routed MoE weights", flush=True)
        self._model.prepare()

        self.cache_geometry = CacheGeometry()
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity = plan_cache_capacity(
            total_memory,
            free_memory,
            gpu_memory_utilization,
            self.cache_geometry.bytes_per_slab,
        )
        slab_capacity = capacity.parent_capacity
        if cache_slab_capacity is not None:
            slab_capacity = min(slab_capacity, cache_slab_capacity)
        self.cache_capacity_plan = replace(
            capacity,
            parent_capacity=slab_capacity,
            allocation_bytes=(slab_capacity + capacity.reserved_parent_count)
            * capacity.bytes_per_parent,
        )
        self.cache_layout = self.cache_geometry.layout(slab_capacity)
        self._cache = CacheArena(self.device, self.cache_layout)
        self._configure_cache_binding(self._cache.binding)
        decode_graph_shapes = resolve_decode_graph_shapes(
            cuda_graph_decode_shapes,
            policy=self.cuda_graph_decode,
            width=self.physical_seq_len,
            speculative=True,
            max_batch_size=max_batch_size,
        )
        if any(shape[0] > max_batch_size for shape in decode_graph_shapes):
            raise ValueError("CUDA graph batch capacity exceeds max_batch_size")
        if any(shape[1] > self.physical_seq_len for shape in decode_graph_shapes):
            raise ValueError("CUDA graph sequence capacity exceeds max_seq_len")
        if any(
            shape[1] <= self.speculative_tokens + 1 for shape in decode_graph_shapes
        ):
            raise ValueError("Inkling CUDA graph sequence capacity is too small")
        self._configure_cuda_graphs(
            {
                "decode": CudaGraphMode(
                    decode_graph_shapes,
                    self._new_decode_graph_step,
                    self._decode_graph_capture_inputs,
                )
            }
        )
        self._graph_state_inputs = torch.empty(
            (
                max_batch_size,
                self.speculative_tokens + 1,
                self.config.num_hidden_layers,
                STATE_WIDTH_PER_LAYER,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=self.prefill_chunk_size,
            prefill_chunk_size=self.prefill_chunk_size,
            continuation_prefill=True,
            cache_write_lookahead_tokens=self.speculative_tokens,
            cache_layout=self.cache_layout,
        )
        layers = cast(Sequence[DecoderLayer], self._model.target.layers)
        prefill_ffns = self._configure_prefill_graphs(
            tuple(layer.feed_forward for layer in layers),
            hidden_size=self.config.hidden_size,
            max_tokens=self.prefill_chunk_size,
        )
        for layer, callback in zip(layers, prefill_ffns, strict=True):
            layer.prefill_ffn = callback

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        return self._speculation.snapshot()

    @staticmethod
    def _capture_inputs(shape: GraphShape, vocab_size: int) -> dict[str, object]:
        batch_size, _ = shape
        return {
            "token_ids": (0,) * batch_size,
            "positions": (1,) * batch_size,
            **capture_sampling_inputs(batch_size, vocab_size, unique_seeds=True),
        }

    def _after_cuda_graph_capture(self) -> None:
        self._cache.reset_null()
        torch.cuda.synchronize(self.device)

    def _decode_graph_capture_inputs(self, shape: GraphShape) -> dict[str, object]:
        inputs = self._capture_inputs(shape, self.vocab_size)
        inputs.update(self._cache.graph_shape_capture_inputs(*shape))
        return inputs

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        batch_size, graph_seq_len = shape
        cache = self._cache.graph_cache(batch_size, graph_seq_len)
        self._graph_device_caches[shape] = cache
        token_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
        positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        sampling = self._sampler.empty(batch_size, self.device)
        output_width = 2 * self.speculative_tokens + 1
        output_ids = torch.empty(
            (batch_size, output_width), dtype=torch.long, device=self.device
        )

        phase_timing = self._new_phase_timing("draft", "target_verify", "decode")

        def launch() -> None:
            with use_default_dtype(torch.bfloat16), torch.device(self.device):
                if phase_timing is not None:
                    phase_timing.mark(0)
                initial_state = self._initial_state(cache, positions)
                state_inputs = self._graph_state_inputs[:batch_size]
                drafts, _ = self._draft.propose(
                    token_ids,
                    positions,
                    cache,
                    graph_seq_len,
                    lambda logits, step: graph_sample(
                        self._sampler, logits, sampling, step
                    ),
                )
                if phase_timing is not None:
                    phase_timing.mark(1)
                verify_ids = torch.cat((token_ids[:, None], drafts), dim=1)
                offsets = torch.arange(self.speculative_tokens + 1, device=self.device)
                verify_positions = positions[:, None] + offsets
                logits, taps, final_state, _ = self._model.target(
                    verify_ids,
                    verify_positions,
                    cache,
                    initial_state,
                    positions + self.speculative_tokens + 1,
                    graph_seq_len,
                    full_logits=True,
                    state_inputs_out=state_inputs,
                )
                if phase_timing is not None:
                    phase_timing.mark(2)
                targets = torch.stack(
                    [
                        graph_sample(
                            self._sampler, logits[:, step].contiguous(), sampling, step
                        )
                        for step in range(self.speculative_tokens + 1)
                    ],
                    dim=1,
                )
                matches = drafts == targets[:, : self.speculative_tokens]
                accepted = torch.cumprod(matches.to(torch.int32), dim=1).sum(1)
                processed = accepted.to(torch.long) + 1
                select_convolution_state(
                    initial_state,
                    state_inputs,
                    processed,
                    out=final_state,
                )
                self._store_state(cache, positions + processed - 1, final_state)
                self._draft.inject_target(taps, verify_positions, cache)
                output_ids[:, : self.speculative_tokens].copy_(drafts)
                output_ids[:, self.speculative_tokens :].copy_(targets)

                if phase_timing is not None:
                    phase_timing.mark(3)

        inputs = {"token_ids": token_ids, "positions": positions}
        inputs.update(self._cache.graph_device_inputs(cache))
        inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(
            inputs=inputs,
            output=output_ids,
            launch=launch,
            phase_timing=phase_timing,
            prepare=self._cache.reset_null,
        )

    def _graph_forward(self, plan: BatchPlan, cache_plan: Any):
        if plan.needs_logprobs:
            self._cuda_graph_fallback("decode", "logprobs")
            return None
        batch_size = plan.num_sequences
        required_length = int(np.max(plan.positions)) + self.speculative_tokens + 1
        shape = self._select_cuda_graph(
            "decode",
            match=lambda candidate: (
                candidate[0] >= batch_size and candidate[1] >= required_length
            ),
        )
        if shape is None:
            return None
        padding = shape[0] - batch_size
        inputs: dict[str, object] = {
            "token_ids": np.pad(plan.token_ids, (0, padding)),
            "positions": np.pad(plan.positions, (0, padding), constant_values=1),
        }
        inputs.update(
            self._cache.graph_plan_inputs(self._graph_device_caches[shape], cache_plan)
        )
        inputs.update(
            pad_sampling_inputs(self._sampler, plan, self.vocab_size, padding)
        )
        output = (
            self._run_cuda_graph("decode", shape, inputs)[:batch_size].cpu().tolist()
        )
        width = self.speculative_tokens
        accepted = self._speculation.accept(
            [row[:width] for row in output],
            [row[width:] for row in output],
        ).token_rows
        return token_output(accepted)

    def _validate_plan(self, plan: BatchPlan) -> tuple[tuple[int, int], ...]:
        return validate_hybrid_plan(
            plan,
            label="Inkling",
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            decode_lookahead=0,
        )

    def _sample(self, logits: torch.Tensor, plan: BatchPlan, step: int = 0):
        return self._sampler(
            logits.float(),
            plan,
            offset_delta=step * SAMPLING_OFFSET_STRIDE,
        )

    def _initial_state(
        self, cache: DeviceCache, contexts: torch.Tensor
    ) -> torch.Tensor:
        positions = (contexts - 1).clamp_min(0)[:, None]
        state = cache.state_at(positions)[:, 0].clone()
        return torch.where(
            (contexts == 0)[:, None, None, None],
            torch.zeros((), dtype=state.dtype, device=state.device),
            state,
        )

    @staticmethod
    def _store_state(
        cache: DeviceCache, positions: torch.Tensor, state: torch.Tensor
    ) -> None:
        cache.store_state(positions[:, None], state[:, None])

    def _prefill(
        self,
        plan: BatchPlan,
        rows: tuple[tuple[int, int], ...],
        cache: DeviceCache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> ForwardOutput:
        logits = []
        prompt_rows = []
        for row, (start, end) in enumerate(rows):
            row_cache = cache.row(row)
            row_ids = token_ids[start:end][None]
            row_positions = positions[start:end][None]
            context = int(plan.context_lengths[row])
            contexts = torch.tensor([context], dtype=torch.long, device=self.device)
            state = self._initial_state(row_cache, contexts)
            input_embeddings = self._model.target.embed_tokens(row_ids)
            self._merge_media_embeddings(
                plan,
                row,
                row_ids,
                row_positions,
                input_embeddings,
            )
            row_logits, taps, final_state, _ = self._model.target(
                row_ids,
                row_positions,
                row_cache,
                state,
                row_positions[:, -1] + 1,
                context + end - start,
                full_logits=plan.needs_prompt_logprobs,
                **({"return_hidden": True} if plan.needs_prompt_logprobs else {}),
                prefill=True,
                input_embeddings=input_embeddings,
            )
            self._store_state(row_cache, row_positions[:, -1], final_state)
            self._draft.inject_target(taps, row_positions, row_cache)
            if plan.needs_prompt_logprobs:

                def project(hidden):
                    return self._model.target.unembed(hidden)[
                        ..., : self.config.unpadded_vocab_size
                    ]

                prompt_rows.append(
                    prompt_scores(plan, row_logits[0], project, token_start=start)
                )
                row_logits = project(row_logits[:, -1:])
            logits.append(row_logits[:, -1])
        values = torch.cat(logits)
        scores = PromptScores.concat(prompt_rows)
        return format_output(
            values, self._sample(values, plan), plan, prompt_logprobs=scores
        )

    def _features_for(self, sequence_id: int, media: InklingMedia) -> torch.Tensor:
        existing = self._media_features.get(sequence_id)
        if existing is not None:
            if existing[0] is not media:
                raise ValueError("a live Inkling sequence changed its media object")
            return existing[1]
        pixels = media.pixel_values
        if pixels is None:
            raise ValueError("Inkling image placeholders require pixel_values")
        features = self._model.vision(
            torch.tensor(pixels, device=self.device, dtype=torch.bfloat16)
        ).detach()
        if features.shape != (len(media.image_positions), self.config.hidden_size):
            raise ValueError("Inkling vision features do not match placeholders")
        self._media_features[sequence_id] = (media, features)
        return features

    def _merge_media_embeddings(
        self,
        plan: BatchPlan,
        row: int,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> None:
        media = plan.media[row]
        if media is None:
            return
        if not isinstance(media, InklingMedia):
            raise TypeError("InklingCudaRunner media must be InklingMedia")
        flat_ids = token_ids[0]
        flat_positions = positions[0]
        position_list = flat_positions.tolist()
        prompt_ids = media.token_ids
        for token_id, position in zip(flat_ids.tolist(), position_list):
            if position < len(prompt_ids) and token_id != prompt_ids[position]:
                raise ValueError("scheduled token does not match InklingMedia prompt")
        feature_indices = {
            position: index for index, position in enumerate(media.image_positions)
        }
        if not any(position in feature_indices for position in position_list):
            return
        features = self._features_for(int(plan.sequence_ids[row]), media)
        for query_index, position in enumerate(position_list):
            feature_index = feature_indices.get(position)
            if feature_index is not None:
                hidden[0, query_index].copy_(features[feature_index])

    def _speculative_decode(
        self,
        plan: BatchPlan,
        cache: DeviceCache,
        anchors: torch.Tensor,
        positions: torch.Tensor,
    ) -> ForwardOutput:
        phase_timing = self._eager_phase_timing("draft", "target_verify", "decode")
        if phase_timing is not None:
            phase_timing.mark(0)
        drafts, _ = self._draft.propose(
            anchors,
            positions,
            cache,
            int(positions.max().item()) + self.speculative_tokens,
            lambda logits, step: self._sample(logits, plan, step),
        )
        if phase_timing is not None:
            phase_timing.mark(1)
        verify_ids = torch.cat((anchors[:, None], drafts), dim=1)
        verify_positions = positions[:, None] + torch.arange(
            self.speculative_tokens + 1, device=self.device
        )
        contexts = torch.as_tensor(
            plan.context_lengths, dtype=torch.long, device=self.device
        )
        initial_state = self._initial_state(cache, contexts)
        state_inputs = torch.empty(
            (
                anchors.shape[0],
                self.speculative_tokens + 1,
                self.config.num_hidden_layers,
                STATE_WIDTH_PER_LAYER,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )
        logits, taps, final_state, _ = self._model.target(
            verify_ids,
            verify_positions,
            cache,
            initial_state,
            positions + self.speculative_tokens + 1,
            int(positions.max().item()) + self.speculative_tokens + 1,
            full_logits=True,
            state_inputs_out=state_inputs,
        )
        if phase_timing is not None:
            phase_timing.mark(2)
        targets = torch.stack(
            [
                self._sample(logits[:, step].contiguous(), plan, step)
                for step in range(self.speculative_tokens + 1)
            ],
            dim=1,
        )
        acceptance = self._speculation.accept(drafts.tolist(), targets.tolist())
        processed = torch.tensor(
            [count + 1 for count in acceptance.accepted_counts],
            dtype=torch.long,
            device=self.device,
        )
        select_convolution_state(
            initial_state,
            state_inputs,
            processed,
            out=final_state,
        )
        self._store_state(cache, positions + processed - 1, final_state)
        self._draft.inject_target(taps, verify_positions, cache)
        if phase_timing is not None:
            phase_timing.mark(3)
        return format_output(logits, acceptance.token_rows, plan)

    @observe_execution
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self._check_cuda_graphs(plan.phase)
        rows = self._validate_plan(plan)
        cache_plan = PagedCachePlan.from_plan(
            plan,
            self.cache_layout,
            decode_lookahead=(
                self.speculative_tokens if plan.phase == BatchPhase.DECODE else 0
            ),
        )
        with self._cuda_failure:
            if plan.phase == BatchPhase.DECODE:
                graph_output = self._graph_forward(plan, cache_plan)
                if graph_output is not None:
                    return graph_output
            staging = TensorStager(self.device)
            cache = self._cache.tensorize(cache_plan, staging)
            token_ids = staging.device("token_ids", plan.token_ids, torch.long)
            positions = staging.device("positions", plan.positions, torch.long)
            with (
                torch.inference_mode(),
                use_default_dtype(torch.bfloat16),
                torch.device(self.device),
            ):
                if plan.phase == BatchPhase.PREFILL:
                    with self._prefill_graphs():
                        return self._prefill(plan, rows, cache, token_ids, positions)
                return self._speculative_decode(plan, cache, token_ids, positions)

    def release(self, sequence_ids: list[int]) -> None:
        """The Engine retires each sequence's slabs after this callback."""

        for sequence_id in sequence_ids:
            self._media_features.pop(sequence_id, None)


__all__ = ["InklingCudaRunner"]
