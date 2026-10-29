"""Qwen3.5 CUDA runner with optional bundled-MTP decoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from resin import BatchPhase, RunnerConfig
from resin.backend.capacity import plan_cache_capacity
from resin.backend.cuda.completion import (
    DeviceCompletion,
    DeviceFeedback,
    accepted_block,
)
from resin.backend.cuda.graphs import (
    CudaGraphMode,
    CudaGraphStep,
    GraphShape,
    resolve_decode_graph_shapes,
)
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.parallel import (
    TensorParallelContext,
    replicated_cache_capacity,
)
from resin.backend.cuda.runner import (
    CudaRunner,
    capture_sampling_inputs,
    graph_sample,
    graph_sample_block,
    pad_sampling_inputs,
    validate_hybrid_plan,
)
from resin.backend.cuda.sampling import (
    SAMPLING_OFFSET_STRIDE,
)
from resin.backend.cuda.staging import TensorStager
from resin.backend.cuda.vision_cache import VisionFeatureCache
from resin.backend.plans import PagedCachePlan
from resin.backend.sampling import (
    PromptScores,
    format_output,
    prompt_scores,
    token_output,
)
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelGroups, ParallelismCapabilities
from resin.backend.torch import use_default_dtype
from resin.recipe import Qwen35Media
from resin_models_cuda.nvfp4 import select_nvfp4_backend
from resin_models_cuda.qwen3_5.attention import (
    GraphAttentionSpec,
    GraphPagedAttentionRuntime,
    PagedAttentionRuntime,
)
from resin_models_cuda.qwen3_5.cache import (
    PAGE_TOKENS,
    PREFILL_CHUNK_SIZE,
    CacheArena,
    CacheGeometry,
    DeviceCache,
)
from resin_models_cuda.qwen3_5.checkpoint import (
    DEFAULT_MODEL,
    is_modelopt_nvfp4,
    load_model,
    resolve_model_path,
)
from resin_models_cuda.qwen3_5.model import Qwen35DecoderLayer

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput

_CACHE_WORKSPACE_BYTES = 512 * 1024**2
_GRAPH_WORKSPACE_BYTES = 128 * 1024**2


def _accepted_verification_state(states, accepted):
    """Select the state after the first token plus the accepted draft prefix."""
    selected = states[0]
    shape = (accepted.shape[0],) + (1,) * (selected.ndim - 1)
    for step, state in enumerate(states[1:], start=1):
        selected = torch.where((accepted >= step).reshape(shape), state, selected)
    return selected


def _one_draft_verification(drafts, targets, *, accepted_drafts=None):
    """Choose the real commit depth and MTP inputs for natural or forced acceptance."""
    if accepted_drafts is None:
        return (drafts == targets[:, 0]).long(), targets
    if type(accepted_drafts) is not int or accepted_drafts not in (0, 1):
        raise ValueError("benchmark accepted drafts must be zero or one")
    accepted = torch.full_like(drafts, accepted_drafts)
    mtp_inputs = (
        torch.stack((drafts, targets[:, 1]), dim=1) if accepted_drafts else targets
    )
    return accepted, mtp_inputs


@dataclass(frozen=True, slots=True)
class _Feedback(DeviceFeedback):
    """Device continuation inputs; the one-draft MTP path also carries drafts."""

    sequence_ids: tuple[int, ...]
    shape: GraphShape
    draft_ids: torch.Tensor | None = None

    def restrict(self, rows):
        DeviceFeedback.restrict(self, rows)
        if self.draft_ids is not None:
            self.draft_ids[rows:].zero_()
        return self

    def inputs(self):
        inputs = {
            "token_ids": self.token_ids,
            "positions": self.positions,
            "sampling_offsets": (self.positions + 1) * SAMPLING_OFFSET_STRIDE,
        }
        if self.draft_ids is not None:
            inputs["draft_ids"] = self.draft_ids
        return inputs


class _Completion(DeviceCompletion):
    """Shared device completion that also harvests the next MTP drafts on the host."""

    def __init__(self, runner, sequence_ids, next_drafts, *args, **kwargs):
        # Queue the draft copy before the base class records its completion event.
        self._draft_stager = TensorStager(runner.device)
        self._next_drafts = (
            None
            if next_drafts is None
            else self._draft_stager.output("next_drafts", next_drafts)
        )
        self._runner = runner
        self._sequence_ids = sequence_ids
        super().__init__(*args, **kwargs)

    def result(self):
        fresh = self._result is None
        output = super().result()
        if not fresh:
            return output
        runner = self._runner
        if self._next_drafts is not None:
            for sequence, draft in zip(
                self._sequence_ids,
                self._next_drafts[: self._rows].tolist(),
                strict=True,
            ):
                runner._drafts[sequence] = int(draft)
        return output


class Qwen3_5CudaRunner(CudaRunner):
    """Dense Qwen3.5/Qwen3.8 text runner over paged FA4 and DeltaNet state."""

    page_size = 128
    prefill_chunk_size = PREFILL_CHUNK_SIZE
    parallelism_capabilities = ParallelismCapabilities(attn_tp=True, expert_tp=True)

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        max_seq_len: int | None = None,
        max_batch_size: int = 32,
        speculative_tokens: int = 0,
        device: str | torch.device,
        gpu_memory_utilization: float = 0.90,
        cache_slab_capacity: int | None = None,
        kv_cache_dtype: str = "fp8",
        nvfp4_backend: str = "auto",
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
        cuda_graph_decode_shapes: object | None = None,
        cache_write_reservation_tokens: int = 136,
        parallel_groups: ParallelGroups | None = None,
    ) -> None:
        if parallel_groups is not None:
            self.parallelism_capabilities.validate(
                parallel_groups.topology, model="Qwen3.5"
            )
        self.parallel_groups = parallel_groups
        self.attention_parallel = TensorParallelContext.from_group(
            None if parallel_groups is None else parallel_groups.attn_tp
        )
        self.expert_parallel = TensorParallelContext.from_group(
            None if parallel_groups is None else parallel_groups.expert_tp
        )
        if (
            isinstance(speculative_tokens, bool)
            or not isinstance(speculative_tokens, int)
            or speculative_tokens < 0
        ):
            raise ValueError("speculative_tokens must be a non-negative integer")
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if not 0.0 < gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if cache_slab_capacity is not None and (
            isinstance(cache_slab_capacity, bool)
            or not isinstance(cache_slab_capacity, int)
            or cache_slab_capacity <= 0
        ):
            raise ValueError("cache_slab_capacity must be a positive integer")
        if nvfp4_backend not in {"auto", "cudnn", "cute-dsl"}:
            raise ValueError("nvfp4_backend must be 'auto', 'cudnn', or 'cute-dsl'")

        super().__init__(
            device=device,
            label="Qwen3.5",
            kv_cache_dtype=kv_cache_dtype,
            cuda_graph_prefill=cuda_graph_prefill,
            cuda_graph_decode=cuda_graph_decode,
        )
        torch.cuda.set_device(self.device)
        if torch.cuda.get_device_capability(self.device)[0] < 9:
            raise RuntimeError("Qwen3_5CudaRunner requires Hopper or newer CUDA")
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        except ImportError as error:
            raise ImportError(
                "Qwen3_5CudaRunner requires flash-linear-attention and FlashInfer"
            ) from error
        del chunk_gated_delta_rule, BatchDecodeWithPagedKVCacheWrapper

        self.model_path = resolve_model_path(model, revision)
        self.speculative_tokens = speculative_tokens
        nvfp4_backend = select_nvfp4_backend(
            nvfp4_backend, torch.backends.cudnn.version()
        )
        self._model, self.config, self.raw_config = load_model(
            self.model_path,
            device=self.device,
            include_mtp=speculative_tokens > 0,
            include_vision=True,
            nvfp4_backend=nvfp4_backend,
            tensor_parallel=self.expert_parallel,
            attention_parallel=self.attention_parallel,
        )
        self.attention_config = self._model.language_model.config
        self._configure_parallel_graphs(self._model, parallel_groups)
        self.image_token_id = self.raw_config.get("image_token_id")
        vision_config = self.raw_config.get("vision_config") or {}
        self.spatial_merge_size = int(vision_config.get("spatial_merge_size", 1))
        self._media_features: dict[int, tuple[Qwen35Media, torch.Tensor]] = {}
        self._vision_feature_cache = VisionFeatureCache()
        self.weight_quantization = (
            "modelopt-fp8-nvfp4" if is_modelopt_nvfp4(self.raw_config) else "bfloat16"
        )
        self.nvfp4_backend = (
            nvfp4_backend if self.weight_quantization != "bfloat16" else None
        )
        selected_max_seq_len = (
            self.config.max_position_embeddings if max_seq_len is None else max_seq_len
        )
        if (
            isinstance(selected_max_seq_len, bool)
            or not isinstance(selected_max_seq_len, int)
            or selected_max_seq_len <= 0
        ):
            raise ValueError("max_seq_len must be a positive integer")
        if selected_max_seq_len > self.config.max_position_embeddings:
            raise ValueError("max_seq_len exceeds the checkpoint context length")
        self.max_seq_len = selected_max_seq_len
        self.max_batch_size = max_batch_size
        self.vocab_size = self.config.vocab_size
        self.eos_token_id = self.config.eos_token_id
        self._speculation = SpeculativeStats(speculative_tokens)
        self._graph_device_caches: dict[GraphShape, DeviceCache] = {}
        self._graph_attention: dict[GraphShape, GraphPagedAttentionRuntime] = {}
        cache_dtype = self.kv_cache_torch_dtype

        self.cache_geometry = CacheGeometry.from_config(
            self.attention_config,
            include_mtp=speculative_tokens > 0,
            kv_element_bytes=cache_dtype.itemsize,
        )
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity = plan_cache_capacity(
            total_memory,
            free_memory,
            gpu_memory_utilization,
            self.cache_geometry.slab_bytes,
            overhead_bytes=_CACHE_WORKSPACE_BYTES,
        )
        slab_capacity = capacity.parent_capacity
        if cache_slab_capacity is not None:
            slab_capacity = min(slab_capacity, cache_slab_capacity)
        slab_capacity = replicated_cache_capacity(
            slab_capacity, None if parallel_groups is None else parallel_groups.replica
        )
        self.cache_capacity_plan = replace(
            capacity,
            parent_capacity=slab_capacity,
            allocation_bytes=(slab_capacity + capacity.reserved_parent_count)
            * capacity.bytes_per_parent,
        )
        self.cache_layout = self.cache_geometry.layout(slab_capacity)
        self._cache = CacheArena(
            self.device,
            self.cache_layout,
            self.cache_geometry,
            kv_dtype=cache_dtype,
        )
        self._configure_cache_binding(self._cache.binding)
        self._attention = PagedAttentionRuntime(
            self.device,
            num_attention_heads=self.attention_config.num_attention_heads,
            num_key_value_heads=self.attention_config.num_key_value_heads,
            head_dim=self.config.head_dim,
            workspace_bytes=_CACHE_WORKSPACE_BYTES,
        )
        decode_graph_shapes = resolve_decode_graph_shapes(
            cuda_graph_decode_shapes,
            policy=self.cuda_graph_decode,
            width=self.max_seq_len,
            speculative=bool(speculative_tokens),
            max_batch_size=max_batch_size,
        )
        if any(shape[0] > max_batch_size for shape in decode_graph_shapes):
            raise ValueError("Qwen3.5 CUDA graph batch capacity exceeds max_batch_size")
        if any(shape[1] > self.max_seq_len for shape in decode_graph_shapes):
            raise ValueError("Qwen3.5 CUDA graph sequence capacity exceeds max_seq_len")
        if any(shape[1] <= speculative_tokens + 1 for shape in decode_graph_shapes):
            raise ValueError("Qwen3.5 CUDA graph sequence capacity is too small")
        self._graph_workspace = torch.empty(
            _GRAPH_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
        )
        # One-draft MTP: the next cycle's draft per sequence, and pending
        # (key, value, decay, beta) GDN updates of the verified draft token.
        self._drafts: dict[int, int] = {}
        self._pair_offsets = torch.arange(2, device=self.device)[None]
        self._window_offsets = torch.arange(3, device=self.device)
        geometry = self.attention_config
        self._pending = (
            torch.empty(
                (
                    geometry.linear_layer_count,
                    max_batch_size,
                    1,
                    geometry.linear_num_value_heads,
                    geometry.linear_key_head_dim + geometry.linear_value_head_dim + 2,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            if speculative_tokens == 1
            else None
        )
        self._configure_cuda_graphs(
            {
                "decode": CudaGraphMode(
                    decode_graph_shapes,
                    self._new_decode_graph_step,
                    self._decode_graph_capture_inputs,
                )
            }
        )
        lookahead = speculative_tokens + 1 if speculative_tokens else 0
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            cache_write_lookahead_tokens=lookahead,
            cache_write_reservation_tokens=(
                cache_write_reservation_tokens if speculative_tokens in (0, 1) else 0
            ),
            cache_layout=self.cache_layout,
        )
        layers = tuple(
            cast(Sequence[Qwen35DecoderLayer], self._model.language_model.layers)
        )
        if self._model.mtp is not None:
            layers += tuple(cast(Sequence[Qwen35DecoderLayer], self._model.mtp.layers))
        prefill_ffns = self._configure_prefill_graphs(
            tuple(layer.feed_forward for layer in layers),
            hidden_size=self.config.hidden_size,
            max_tokens=self.prefill_chunk_size,
        )
        for layer, callback in zip(layers, prefill_ffns, strict=True):
            layer.prefill_ffn = callback

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        return self._speculation.snapshot(include_draft_tokens=True)

    @property
    def benchmark_counters(self) -> dict[str, int]:
        return {
            "async.enqueues": getattr(self, "_async_enqueues", 0),
            "async.continuations": getattr(self, "_async_continuations", 0),
            "host.attention_plans": getattr(self, "_host_attention_plans", 0),
        }

    @staticmethod
    def _media_at(plan: BatchPlan, row: int) -> Qwen35Media | None:
        media = plan.media[row]
        if media is None:
            return None
        if not isinstance(media, Qwen35Media):
            raise TypeError("Qwen3_5CudaRunner media must be Qwen35Media")
        return media

    def _host_rope_deltas(self, plan: BatchPlan) -> np.ndarray:
        return np.asarray(
            [
                media.rope_delta if (media := self._media_at(plan, row)) else 0
                for row in range(plan.num_sequences)
            ],
            dtype=np.int64,
        )

    def _rope_deltas(self, plan: BatchPlan) -> torch.Tensor:
        return torch.as_tensor(self._host_rope_deltas(plan), device=self.device)

    def _decode_rotary_positions(
        self, plan: BatchPlan, positions: torch.Tensor
    ) -> torch.Tensor:
        adjusted = positions + self._rope_deltas(plan)
        return adjusted[None, :, None].expand(3, -1, -1)

    def _prefill_rotary_positions(
        self,
        media: Qwen35Media | None,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if media is None:
            return positions[None].expand(3, -1, -1)
        absolute = positions[0].cpu().numpy()
        columns = np.broadcast_to(
            absolute + media.rope_delta, (3, len(absolute))
        ).copy()
        prompt = absolute < len(media.token_ids)
        columns[:, prompt] = media.position_ids[:, absolute[prompt]]
        return torch.from_numpy(columns).to(self.device)[:, None]

    def _features_for(self, sequence_id: int, media: Qwen35Media) -> torch.Tensor:
        existing = self._media_features.get(sequence_id)
        if existing is not None:
            if existing[0] is not media:
                raise ValueError("a live Qwen sequence changed its media object")
            return existing[1]
        cached = self._vision_feature_cache.get(media.digest)
        if cached is None:
            visual = self._model.visual
            if visual is None:
                raise RuntimeError("Qwen checkpoint did not load a visual tower")
            with torch.inference_mode():
                output = visual(
                    torch.from_numpy(media.pixel_values).to(
                        self.device, dtype=torch.bfloat16
                    ),
                    torch.from_numpy(media.image_grid_thw).to(self.device),
                )
                features = output.pooler_output.detach()
            if features.shape != (len(media.image_positions), self.config.hidden_size):
                raise ValueError("Qwen vision features do not match image placeholders")
            cached = self._vision_feature_cache.put(media.digest, (features,))
        (features,) = cached
        assert features is not None
        if features.shape != (len(media.image_positions), self.config.hidden_size):
            raise ValueError("Qwen vision features do not match image placeholders")
        self._media_features[sequence_id] = (media, features)
        return features

    def _merge_media_embeddings(
        self,
        plan: BatchPlan,
        row: int,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Qwen35Media | None:
        media = self._media_at(plan, row)
        if media is None:
            return None
        position_list = [int(position) for position in positions[0].tolist()]
        prompt_ids = media.token_ids
        for token_id, position in zip(
            token_ids[0].tolist(), position_list, strict=True
        ):
            if position < len(prompt_ids) and int(token_id) != prompt_ids[position]:
                raise ValueError("scheduled token does not match Qwen35Media prompt")
        feature_indices = {
            position: index for index, position in enumerate(media.image_positions)
        }
        if not any(position in feature_indices for position in position_list):
            return media
        features = self._features_for(int(plan.sequence_ids[row]), media)
        for query_index, position in enumerate(position_list):
            feature_index = feature_indices.get(position)
            if feature_index is not None:
                embeddings[0, query_index].copy_(features[feature_index])
        return media

    @staticmethod
    def _capture_inputs(shape: GraphShape, vocab_size: int) -> dict[str, object]:
        batch_size, _ = shape
        return {
            "token_ids": (0,) * batch_size,
            "positions": (1,) * batch_size,
            "rope_deltas": (0,) * batch_size,
            "draft_ids": (0,) * batch_size,
            **capture_sampling_inputs(batch_size, vocab_size, unique_seeds=True),
        }

    def _after_cuda_graph_capture(self) -> None:
        self._cache.reset_null()
        torch.cuda.synchronize(self.device)

    def _decode_graph_capture_inputs(self, shape: GraphShape) -> dict[str, object]:
        inputs = self._capture_inputs(shape, self.vocab_size)
        if not self._fast_speculation:
            del inputs["draft_ids"]
        inputs.update(self._cache.graph_shape_capture_inputs(*shape))
        return inputs

    @property
    def _fast_speculation(self) -> bool:
        """One MTP draft verified with the target in a single two-token forward."""
        return self.speculative_tokens == 1

    def _graph_specs(self) -> tuple[GraphAttentionSpec, ...]:
        target = self.config.full_layer_count
        if not self.speculative_tokens:
            return (GraphAttentionSpec(0, target, 1, 1),)
        if self._fast_speculation:
            return (GraphAttentionSpec(0, target, 2, 2), GraphAttentionSpec(2, 1, 2, 2))
        width = self.speculative_tokens + 1
        return (
            GraphAttentionSpec(0, target, 1, 1),
            *(
                GraphAttentionSpec(2, 1, 1, step + 1)
                for step in range(self.speculative_tokens)
            ),
            *(GraphAttentionSpec(0, target, 1, step + 2) for step in range(width)),
            GraphAttentionSpec(2, 1, width, width + 1),
        )

    def _graph_attention_for(self, shape: GraphShape) -> GraphPagedAttentionRuntime:
        runtime = self._graph_attention.get(shape)
        if runtime is None:
            cache = self._graph_device_caches.get(shape)
            if cache is None:
                cache = self._cache.graph_cache(*shape)
                self._graph_device_caches[shape] = cache
            runtime = GraphPagedAttentionRuntime(
                self.device,
                cache,
                self._graph_specs(),
                num_attention_heads=self.attention_config.num_attention_heads,
                num_key_value_heads=self.attention_config.num_key_value_heads,
                head_dim=self.config.head_dim,
                workspace=self._graph_workspace,
            )
            self._graph_attention[shape] = runtime
        return runtime

    def _refresh_graph_attention(self, attention, positions: torch.Tensor) -> None:
        """Resolve exact readable pages from GPU positions and the frozen lease."""
        attention.refresh(positions)

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        batch_size, graph_seq_len = shape
        attention = self._graph_attention_for(shape)
        cache = self._graph_device_caches[shape]
        token_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
        positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        rope_deltas = torch.empty(batch_size, dtype=torch.long, device=self.device)
        draft_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
        sampling = self._sampler.empty(batch_size, self.device)
        output_width = 2 * self.speculative_tokens + 2 if self.speculative_tokens else 1
        output_ids = torch.empty(
            (batch_size, output_width), dtype=torch.long, device=self.device
        )

        if self._fast_speculation:
            phase_timing = self._new_phase_timing("target_verify", "decode", "draft")
        elif self.speculative_tokens:
            phase_timing = self._new_phase_timing(
                "decode", "draft", "target_verify", "decode"
            )
        else:
            phase_timing = None

        def ordinary_launch() -> None:
            attention.reset()
            self._refresh_graph_attention(attention, positions)
            # GDN recurrent and convolution state update in place in the arena.
            previous = (positions - 1).clamp_min(0)
            hidden, _, _ = self._target_forward(
                cache,
                token_ids[:, None],
                positions[:, None],
                None,
                None,
                max_sequence_length=graph_seq_len,
                attention_runtime=attention,
                rotary_positions=(positions + rope_deltas)[None, :, None].expand(
                    3, -1, -1
                ),
                recurrent_pool=cache.recurrent_pool(previous, positions),
            )
            cache.store_hidden(positions, hidden[:, -1])
            output_ids[:, 0].copy_(
                graph_sample(
                    self._sampler, self._model.logits(hidden[:, -1]), sampling, 0
                )
            )

        def speculative_launch() -> None:
            if phase_timing is not None:
                phase_timing.mark(0)
            attention.reset()
            self._refresh_graph_attention(attention, positions)
            recurrent, convolution, _ = self._initial_state(cache, positions)
            anchor_hidden, recurrent, convolution = self._target_forward(
                cache,
                token_ids[:, None],
                positions[:, None],
                recurrent,
                convolution,
                max_sequence_length=graph_seq_len,
                attention_runtime=attention,
                rotary_positions=(positions + rope_deltas)[None, :, None].expand(
                    3, -1, -1
                ),
            )
            first = graph_sample(
                self._sampler, self._model.logits(anchor_hidden[:, -1]), sampling, 0
            )

            if phase_timing is not None:
                phase_timing.mark(1)
            draft_columns = []
            draft_input = first
            draft_hidden = anchor_hidden[:, -1:]
            for step in range(self.speculative_tokens):
                draft_position = positions + step
                draft_hidden = self._mtp_forward(
                    cache,
                    draft_input[:, None],
                    draft_hidden,
                    draft_position[:, None],
                    attention_runtime=attention,
                    max_sequence_length=graph_seq_len,
                    rope_deltas=rope_deltas,
                )
                draft_input = graph_sample(
                    self._sampler,
                    self._model.logits(draft_hidden[:, -1]),
                    sampling,
                    step + 1,
                )
                draft_columns.append(draft_input)
            drafts = torch.stack(draft_columns, dim=1)
            if phase_timing is not None:
                phase_timing.mark(2)
            verify_ids = torch.cat((first[:, None], drafts), dim=1)

            verify_hidden = []
            recurrent_steps = []
            convolution_steps = []
            work_recurrent = recurrent
            work_convolution = convolution
            for step in range(self.speculative_tokens + 1):
                verify_position = positions + step + 1
                hidden, work_recurrent, work_convolution = self._target_forward(
                    cache,
                    verify_ids[:, step : step + 1],
                    verify_position[:, None],
                    work_recurrent,
                    work_convolution,
                    max_sequence_length=graph_seq_len,
                    attention_runtime=attention,
                    rotary_positions=(verify_position + rope_deltas)[
                        None, :, None
                    ].expand(3, -1, -1),
                )
                current_hidden = hidden[:, -1]
                # Forward work does not depend on acceptance. Retain each
                # returned state, then select its accepted prefix after one
                # batched draw of the complete verification block.
                recurrent_steps.append(work_recurrent)
                convolution_steps.append(work_convolution)
                verify_hidden.append(current_hidden)
            verify_hidden = torch.stack(verify_hidden, dim=1)
            targets = graph_sample_block(
                self._sampler,
                self._model.logits(verify_hidden),
                sampling,
                start_step=1,
            )
            matches = drafts == targets[:, : self.speculative_tokens]
            accepted = torch.cumprod(matches.to(torch.int64), dim=1).sum(1)
            processed = accepted + 1
            selected_recurrent = _accepted_verification_state(recurrent_steps, accepted)
            selected_convolution = _accepted_verification_state(
                convolution_steps, accepted
            )
            selected_hidden = verify_hidden.gather(
                1, accepted[:, None, None].expand(-1, 1, verify_hidden.shape[-1])
            ).squeeze(1)
            if phase_timing is not None:
                phase_timing.mark(3)

            repair_ids = torch.cat((drafts, targets[:, -1:]), dim=1)
            correction = targets.gather(1, accepted[:, None])
            repair_ids = repair_ids.scatter(1, accepted[:, None], correction)
            repair_positions = positions[:, None] + torch.arange(
                1,
                self.speculative_tokens + 2,
                dtype=torch.long,
                device=self.device,
            )
            self._mtp_forward(
                cache,
                repair_ids,
                verify_hidden,
                repair_positions,
                attention_runtime=attention,
                max_sequence_length=graph_seq_len,
                rope_deltas=rope_deltas,
            )
            self._store_state(
                cache,
                positions + processed,
                selected_recurrent,
                selected_convolution,
                selected_hidden,
            )
            output_ids[:, 0].copy_(first)
            output_ids[:, 1 : self.speculative_tokens + 1].copy_(drafts)
            output_ids[:, self.speculative_tokens + 1 :].copy_(targets)

            if phase_timing is not None:
                phase_timing.mark(4)

        def fast_speculative_launch() -> None:
            from resin_models_cuda.qwen3_5.gdn_decode import gdn_commit

            # Verify [anchor, draft] in one target forward, commit the accepted
            # prefix of GDN/conv state, then draft the next token with MTP.
            if self._pending is None:
                raise RuntimeError(
                    "one-draft verification requires a pending state buffer"
                )
            if phase_timing is not None:
                phase_timing.mark(0)
            attention.reset()
            self._refresh_graph_attention(attention, positions)
            config = self.attention_config
            previous = (positions - 1).clamp_min(0)
            pool = replace(
                cache.recurrent_pool(previous, positions),
                pending=self._pending[:, :batch_size],
            )
            pair = torch.stack((token_ids, draft_ids), dim=1)
            pair_positions = positions[:, None] + self._pair_offsets
            pair_rotary = (pair_positions + rope_deltas[:, None])[None].expand(
                3, -1, -1
            )
            hidden, _, windows = self._target_forward(
                cache,
                pair,
                pair_positions,
                None,
                cache.convolution_at(previous),
                max_sequence_length=graph_seq_len,
                attention_runtime=attention,
                rotary_positions=pair_rotary,
                recurrent_pool=pool,
            )
            targets = graph_sample_block(
                self._sampler, self._model.logits(hidden), sampling, start_step=0
            )
            accepted, mtp_inputs = _one_draft_verification(
                draft_ids,
                targets,
                accepted_drafts=self._speculation.benchmark_accepted_drafts,
            )
            if phase_timing is not None:
                phase_timing.mark(1)
            gdn_commit(
                pool,
                config.linear_layer_count,
                pool.destination,
                cache.state_offsets(positions + accepted),
                accepted,
                num_value_heads=config.linear_num_value_heads,
                key_dim=config.linear_key_head_dim,
                value_dim=config.linear_value_head_dim,
            )
            window = (accepted + 1)[:, None, None, None] + self._window_offsets
            convolution = windows.gather(-1, window.expand(*windows.shape[:-1], 3))
            selected = hidden.gather(
                1, accepted[:, None, None].expand(-1, 1, hidden.shape[-1])
            ).squeeze(1)
            cache.store_convolution(positions + accepted, convolution, selected)
            if phase_timing is not None:
                phase_timing.mark(2)
            mtp_hidden = self._mtp_forward(
                cache,
                mtp_inputs,
                hidden,
                pair_positions,
                attention_runtime=attention,
                max_sequence_length=graph_seq_len,
                rope_deltas=rope_deltas,
            )
            mtp_selected = mtp_hidden.gather(
                1, accepted[:, None, None].expand(-1, 1, mtp_hidden.shape[-1])
            ).squeeze(1)
            next_draft = graph_sample(
                self._sampler, self._model.logits(mtp_selected), sampling, 2
            )
            # [draft, t1, t2] is accepted_block's [drafts, targets] layout.
            output_ids[:, 0].copy_(draft_ids)
            output_ids[:, 1:3].copy_(targets)
            output_ids[:, 3].copy_(next_draft)
            if phase_timing is not None:
                phase_timing.mark(3)

        if self._fast_speculation:
            launch = fast_speculative_launch
        else:
            launch = speculative_launch if self.speculative_tokens else ordinary_launch
        inputs = {
            "token_ids": token_ids,
            "positions": positions,
            "rope_deltas": rope_deltas,
        }
        if self._fast_speculation:
            inputs["draft_ids"] = draft_ids
        inputs.update(self._cache.graph_device_inputs(cache))
        inputs.update(attention.graph_inputs())
        inputs.update(self._sampler.graph_inputs(sampling))
        if not hasattr(self, "_feedback_graph_inputs"):
            self._feedback_graph_inputs = {}
        self._feedback_graph_inputs[shape] = inputs

        def prepare() -> None:
            attention.reset()
            self._cache.reset_null()

        return CudaGraphStep(
            inputs=inputs,
            output=output_ids,
            launch=launch,
            phase_timing=phase_timing,
            prepare=prepare,
        )

    def _graph_forward(
        self,
        plan: BatchPlan,
        cache_plan: PagedCachePlan,
        *,
        enqueue: bool = False,
        feedback=None,
    ):
        if plan.needs_logprobs:
            self._cuda_graph_fallback("decode", "logprobs")
            return None
        batch_size = plan.num_sequences
        history_pages = max(
            int(np.diff(cache_plan.groups[index].block_indptr).max())
            for index in (0, 2)
            if index < len(cache_plan.groups)
        )
        required_length = int(np.max(plan.positions)) + self.speculative_tokens + 2
        if not self.speculative_tokens:
            required_length = int(np.max(plan.positions)) + 1
        shape = self._select_cuda_graph(
            "decode",
            match=lambda candidate: (
                candidate[0] >= batch_size
                and candidate[1] >= required_length
                and (candidate[1] + PAGE_TOKENS - 1) // PAGE_TOKENS >= history_pages
            ),
        )
        if shape is None:
            return None
        padding = shape[0] - batch_size
        inputs: dict[str, object] = {
            "token_ids": np.pad(plan.token_ids, (0, padding)),
            "positions": np.pad(plan.positions, (0, padding), constant_values=1),
            "rope_deltas": np.pad(self._host_rope_deltas(plan), (0, padding)),
        }
        inputs.update(
            self._cache.graph_plan_inputs(self._graph_device_caches[shape], cache_plan)
        )
        inputs.update(
            self._graph_attention[shape].plan(cache_plan, np.asarray(plan.positions))
        )
        self._host_attention_plans = getattr(self, "_host_attention_plans", 0) + 1
        inputs.update(
            pad_sampling_inputs(self._sampler, plan, self.vocab_size, padding)
        )
        sequence_ids = [int(sequence) for sequence in plan.sequence_ids]
        if self._fast_speculation:
            # Rows without a draft yet (first decode) propose their anchor; a
            # rejected draft costs only that cycle's second token.
            inputs["draft_ids"] = np.pad(
                np.asarray(
                    [
                        self._drafts.get(sequence, int(token))
                        for sequence, token in zip(
                            sequence_ids, plan.token_ids, strict=True
                        )
                    ],
                    dtype=np.int64,
                ),
                (0, padding),
            )
        if enqueue:
            return self._enqueue_graph_output(plan, shape, inputs, feedback=feedback)
        output = self._run_cuda_graph("decode", shape, inputs)[:batch_size]
        if not self.speculative_tokens:
            return token_output(output[:, 0])
        if self._fast_speculation:
            return self._fast_speculative_output(sequence_ids, output.cpu().tolist())
        output = output.cpu().tolist()
        drafts = [row[1 : self.speculative_tokens + 1] for row in output]
        targets = [row[self.speculative_tokens + 1 :] for row in output]
        accepted = self._speculation.accept(drafts, targets)
        return token_output(
            [
                [row[0], *tokens]
                for row, tokens in zip(output, accepted.token_rows, strict=True)
            ]
        )

    def _fast_speculative_output(self, sequence_ids, rows) -> ForwardOutput:
        acceptance = self._speculation.accept(
            [[row[0]] for row in rows], [row[1:3] for row in rows]
        )
        for sequence, row in zip(sequence_ids, rows, strict=True):
            self._drafts[sequence] = row[3]
        return token_output(acceptance.token_rows)

    def _enqueue_graph_output(self, plan, shape, inputs, *, feedback=None):
        sequence_ids = tuple(map(int, plan.sequence_ids))
        if feedback is not None:
            inputs.update(feedback.inputs())
            positions = feedback.positions
        else:
            positions = torch.as_tensor(
                inputs["positions"], device=self.device, dtype=torch.long
            )
        output = self._run_cuda_graph("decode", shape, inputs)
        draft_width = 1 if self._fast_speculation else 0
        stats = self._speculation if draft_width else None
        tokens, counts, base = accepted_block(
            output[:, : 2 * draft_width + 1],
            positions,
            draft_width,
            fixed_acceptance=None if stats is None else stats.benchmark_accepted_drafts,
        )
        next_drafts = output[:, 3].clone() if draft_width else None
        completion = _Completion(
            self,
            sequence_ids,
            next_drafts,
            tokens,
            counts,
            _Feedback(base.token_ids, base.positions, sequence_ids, shape, next_drafts),
            rows=plan.num_sequences,
            device=self.device,
            stats=stats,
        )
        # A later conservative plan can advance by the Engine's maximum output
        # width. Drain before it could exceed this graph's token capacity;
        # the next normal batch may select a different graph bucket or go eager.
        maximum = self.runner_config.cache_write_lookahead_tokens + 1
        if np.any(np.asarray(plan.positions, dtype=np.int64) + 2 * maximum > shape[1]):
            completion.feedback = None
        self._async_enqueues = getattr(self, "_async_enqueues", 0) + 1
        self._async_continuations = getattr(self, "_async_continuations", 0) + int(
            feedback is not None
        )
        return completion

    def _validate_plan(self, plan: BatchPlan) -> tuple[tuple[int, int], ...]:
        return validate_hybrid_plan(
            plan,
            label="Qwen3.5",
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            decode_lookahead=self.speculative_tokens + 1
            if self.speculative_tokens
            else 0,
        )

    def _sample(self, logits: torch.Tensor, plan: BatchPlan, step: int = 0):
        return self._sampler(
            logits,
            plan,
            offset_delta=step * SAMPLING_OFFSET_STRIDE,
        )

    def _initial_state(
        self, cache: DeviceCache, contexts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = (contexts - 1).clamp_min(0)[:, None]
        recurrent, convolution, target_hidden = cache.state_at(positions)
        recurrent = recurrent[:, 0].clone()
        convolution = convolution[:, 0].clone()
        target_hidden = target_hidden[:, 0].clone()
        new = contexts == 0
        recurrent.masked_fill_(new[:, None, None, None, None], 0)
        convolution.masked_fill_(new[:, None, None, None], 0)
        target_hidden.masked_fill_(new[:, None], 0)
        return recurrent, convolution, target_hidden

    @staticmethod
    def _store_state(
        cache: DeviceCache,
        positions: torch.Tensor,
        recurrent: torch.Tensor,
        convolution: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> None:
        cache.store_state(
            positions[:, None],
            recurrent[:, None],
            convolution[:, None],
            target_hidden[:, None],
        )

    def _target_forward(
        self,
        cache: DeviceCache,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        recurrent: torch.Tensor | None,
        convolution: torch.Tensor | None,
        *,
        attention_runtime: object | None = None,
        max_sequence_length: int | None = None,
        rotary_positions: torch.Tensor | None = None,
        input_embeddings: torch.Tensor | None = None,
        recurrent_pool=None,
    ):
        """Run target tokens through the phase-independent model operation."""

        sequence_lengths = positions[:, -1] + 1
        if max_sequence_length is None:
            max_sequence_length = int(sequence_lengths.max().item())
        return self._model.language_model(
            input_ids,
            positions,
            cache=cache,
            recurrent_states=recurrent,
            convolution_states=convolution,
            sequence_lengths=sequence_lengths,
            max_sequence_length=max_sequence_length,
            attention_runtime=(
                self._attention if attention_runtime is None else attention_runtime
            ),
            rotary_positions=rotary_positions,
            input_embeddings=input_embeddings,
            recurrent_pool=recurrent_pool,
        )

    def _mtp_forward(
        self,
        cache: DeviceCache,
        input_ids: torch.Tensor,
        previous_hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        attention_runtime: object | None = None,
        max_sequence_length: int | None = None,
        rope_deltas: torch.Tensor | None = None,
        rotary_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mtp = self._model.mtp
        if mtp is None:
            raise RuntimeError("Qwen3.5 MTP was not loaded")
        embeddings = self._model.language_model.embed_tokens(input_ids)
        selected_rotary_positions = positions
        if rotary_positions is not None:
            selected_rotary_positions = rotary_positions
        elif rope_deltas is not None:
            selected_rotary_positions = (positions + rope_deltas[:, None])[None].expand(
                3, -1, -1
            )
        rotary = self._model.language_model.rotary(
            selected_rotary_positions, embeddings.dtype
        )
        sequence_lengths = positions[:, -1] + 1
        if max_sequence_length is None:
            max_sequence_length = int(sequence_lengths.max().item())
        return mtp(
            embeddings,
            previous_hidden,
            positions,
            rotary=rotary,
            cache=cache,
            sequence_lengths=sequence_lengths,
            max_sequence_length=max_sequence_length,
            attention_runtime=attention_runtime or self._attention,
        )

    def _prefill_mtp(
        self,
        cache: DeviceCache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
        previous_hidden: torch.Tensor,
        context: int,
        sampled_token: torch.Tensor | None,
        media: Qwen35Media | None,
    ) -> None:
        if self._model.mtp is None:
            return
        if context:
            self._mtp_forward(
                cache,
                token_ids[:, :1],
                previous_hidden[:, None],
                positions[:, :1] - 1,
                rotary_positions=self._prefill_rotary_positions(
                    media, positions[:, :1] - 1
                ),
            )
        if token_ids.shape[1] > 1:
            self._mtp_forward(
                cache,
                token_ids[:, 1:],
                target_hidden[:, :-1],
                positions[:, :-1],
                rotary_positions=self._prefill_rotary_positions(
                    media, positions[:, :-1]
                ),
            )
        if sampled_token is not None:
            self._mtp_forward(
                cache,
                sampled_token[:, None],
                target_hidden[:, -1:],
                positions[:, -1:],
                rotary_positions=self._prefill_rotary_positions(
                    media, positions[:, -1:]
                ),
            )

    def _prefill(
        self,
        plan: BatchPlan,
        rows: tuple[tuple[int, int], ...],
        cache: DeviceCache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> ForwardOutput:
        final_logits = []
        prompt_rows = []
        outputs = []
        states = []
        for row, (start, end) in enumerate(rows):
            row_cache = cache.row(row)
            row_ids = token_ids[start:end][None]
            row_positions = positions[start:end][None]
            context = int(plan.context_lengths[row])
            contexts = torch.tensor([context], dtype=torch.long, device=self.device)
            recurrent, convolution, previous_hidden = self._initial_state(
                row_cache, contexts
            )
            embeddings = self._model.language_model.embed_tokens(row_ids)
            media = self._merge_media_embeddings(
                plan, row, row_ids, row_positions, embeddings
            )
            hidden, recurrent, convolution = self._target_forward(
                row_cache,
                row_ids,
                row_positions,
                recurrent,
                convolution,
                max_sequence_length=context + end - start,
                input_embeddings=embeddings,
                rotary_positions=self._prefill_rotary_positions(media, row_positions),
            )
            final_logits.append(self._model.logits(hidden[:, -1]))
            if plan.needs_prompt_logprobs:
                prompt_rows.append(
                    prompt_scores(
                        plan, hidden[0], self._model.logits, token_start=start
                    )
                )
            states.append(
                (
                    row_cache,
                    row_ids,
                    row_positions,
                    hidden,
                    recurrent,
                    convolution,
                    previous_hidden,
                    context,
                )
            )
        logits = torch.cat(final_logits)
        sampled = self._sample(logits, plan)
        for row, state in enumerate(states):
            (
                row_cache,
                row_ids,
                row_positions,
                hidden,
                recurrent,
                convolution,
                previous_hidden,
                context,
            ) = state
            sampled_token = sampled[row : row + 1] if plan.sampling_mask[row] else None
            media = self._media_at(plan, row)
            self._prefill_mtp(
                row_cache,
                row_ids,
                row_positions,
                hidden,
                previous_hidden,
                context,
                sampled_token,
                media,
            )
            self._store_state(
                row_cache,
                row_positions[:, -1],
                recurrent,
                convolution,
                hidden[:, -1],
            )
            outputs.append(int(sampled[row]))
        scores = PromptScores.concat(prompt_rows)
        return format_output(logits, outputs, plan, prompt_logprobs=scores)

    def _ordinary_decode(
        self,
        plan: BatchPlan,
        cache: DeviceCache,
        anchors: torch.Tensor,
        positions: torch.Tensor,
    ) -> ForwardOutput:
        previous = (positions - 1).clamp_min(0)
        hidden, _, _ = self._target_forward(
            cache,
            anchors[:, None],
            positions[:, None],
            None,
            None,
            rotary_positions=self._decode_rotary_positions(plan, positions),
            recurrent_pool=cache.recurrent_pool(previous, positions),
        )
        cache.store_hidden(positions, hidden[:, -1])
        logits = self._model.logits(hidden[:, -1])
        return format_output(logits, self._sample(logits, plan), plan)

    def _speculative_decode(
        self,
        plan: BatchPlan,
        cache: DeviceCache,
        anchors: torch.Tensor,
        positions: torch.Tensor,
    ) -> ForwardOutput:
        phase_timing = self._eager_phase_timing(
            "decode", "draft", "target_verify", "decode"
        )
        if phase_timing is not None:
            phase_timing.mark(0)
        contexts = torch.as_tensor(
            plan.context_lengths, dtype=torch.long, device=self.device
        )
        rope_deltas = self._rope_deltas(plan)
        initial_recurrent, initial_convolution, _ = self._initial_state(cache, contexts)
        anchor_hidden, anchor_recurrent, anchor_convolution = self._target_forward(
            cache,
            anchors[:, None],
            positions[:, None],
            initial_recurrent,
            initial_convolution,
            rotary_positions=(positions + rope_deltas)[None, :, None].expand(3, -1, -1),
        )
        first_logits = self._model.logits(anchor_hidden[:, -1])
        first = self._sample(first_logits, plan, 0)

        if phase_timing is not None:
            phase_timing.mark(1)
        drafts = []
        draft_input = first
        draft_hidden = anchor_hidden[:, -1:]
        for step in range(self.speculative_tokens):
            draft_position = positions + step
            draft_hidden = self._mtp_forward(
                cache,
                draft_input[:, None],
                draft_hidden,
                draft_position[:, None],
                rope_deltas=rope_deltas,
            )
            draft_input = self._sample(
                self._model.logits(draft_hidden[:, -1]), plan, step + 1
            )
            drafts.append(draft_input)
        draft_ids = torch.stack(drafts, dim=1)

        if phase_timing is not None:
            phase_timing.mark(2)
        verify_ids = torch.cat((first[:, None], draft_ids), dim=1)
        verify_positions = positions[:, None] + torch.arange(
            1, self.speculative_tokens + 2, device=self.device
        )
        verify_hidden, verify_recurrent, verify_convolution = self._target_forward(
            cache,
            verify_ids,
            verify_positions,
            anchor_recurrent,
            anchor_convolution,
            rotary_positions=(verify_positions + rope_deltas[:, None])[None].expand(
                3, -1, -1
            ),
        )
        verify_logits = self._model.logits(verify_hidden)
        if phase_timing is not None:
            phase_timing.mark(3)
        targets = self._sample(verify_logits, plan, 1)
        acceptance = self._speculation.accept(draft_ids.tolist(), targets.tolist())

        rows = []
        for row, accepted in enumerate(acceptance.accepted_counts):
            processed = accepted + 1
            row_cache = cache.row(row)
            if accepted == self.speculative_tokens:
                committed_hidden = verify_hidden[row : row + 1]
                recurrent = verify_recurrent[row : row + 1]
                convolution = verify_convolution[row : row + 1]
            else:
                committed_hidden, recurrent, convolution = self._target_forward(
                    row_cache,
                    verify_ids[row : row + 1, :processed],
                    verify_positions[row : row + 1, :processed],
                    anchor_recurrent[row : row + 1],
                    anchor_convolution[row : row + 1],
                    rotary_positions=(
                        verify_positions[row : row + 1, :processed]
                        + rope_deltas[row : row + 1, None]
                    )[None].expand(3, -1, -1),
                )
            correction = targets[row, accepted : accepted + 1]
            repair_ids = torch.cat((draft_ids[row, :accepted], correction))[None]
            repair_positions = verify_positions[row : row + 1, :processed]
            self._mtp_forward(
                row_cache,
                repair_ids,
                committed_hidden[:, :processed],
                repair_positions,
                rope_deltas=rope_deltas[row : row + 1],
            )
            commit_position = positions[row : row + 1] + processed
            self._store_state(
                row_cache,
                commit_position,
                recurrent,
                convolution,
                committed_hidden[:, processed - 1],
            )
            rows.append([int(first[row]), *acceptance.token_rows[row]])
        if phase_timing is not None:
            phase_timing.mark(4)
        if plan.needs_logprobs:
            logits = torch.cat((first_logits[:, None], verify_logits), dim=1)
            return format_output(logits, rows, plan)
        return token_output(rows)

    @observe_execution
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self._check_cuda_graphs(plan.phase)
        rows = self._validate_plan(plan)
        lookahead = (
            self.speculative_tokens + 1
            if plan.phase == BatchPhase.DECODE and self.speculative_tokens
            else 0
        )
        cache_plan = PagedCachePlan.from_plan(
            plan,
            self.cache_layout,
            decode_lookahead=lookahead,
            decode_reservation_tokens=self.runner_config.cache_write_reservation_tokens,
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
                if self.speculative_tokens:
                    return self._speculative_decode(plan, cache, token_ids, positions)
                return self._ordinary_decode(plan, cache, token_ids, positions)

    @observe_execution
    def enqueue(self, plan: BatchPlan, *, feedback=None):
        """Device-feedback decode for plain and one-draft MTP graphs; else forward()."""
        if (
            plan.phase != BatchPhase.DECODE
            or plan.needs_logprobs
            or plan.needs_prompt_logprobs
            or self.speculative_tokens not in (0, 1)
        ):
            if feedback is not None:
                raise RuntimeError(
                    "a device continuation requires unscored captured decode"
                )
            return self.forward(plan)
        self._check_cuda_graphs(plan.phase)
        if feedback is not None:
            if not isinstance(feedback, _Feedback):
                raise ValueError("Qwen continuation requires owned device feedback")
            if feedback.sequence_ids != tuple(map(int, plan.sequence_ids)):
                raise ValueError("Qwen continuation changed its row ownership")
            shape = feedback.shape
            with self._cuda_failure, torch.inference_mode():
                # The Engine retains the same lease and row order. All dense
                # tables, policies and rotary deltas already occupy graph-owned
                # buffers; stage only owned token/position/draft feedback.
                inputs = dict(self._feedback_graph_inputs[shape])
                return self._enqueue_graph_output(
                    plan, shape, inputs, feedback=feedback
                )
        self._validate_plan(plan)
        lookahead = self.speculative_tokens + 1 if self.speculative_tokens else 0
        cache_plan = PagedCachePlan.from_plan(
            plan,
            self.cache_layout,
            decode_lookahead=lookahead,
            decode_reservation_tokens=self.runner_config.cache_write_reservation_tokens,
        )
        with self._cuda_failure, torch.inference_mode():
            result = self._graph_forward(
                plan, cache_plan, enqueue=True, feedback=feedback
            )
        if result is None and feedback is not None:
            raise RuntimeError("a device continuation requires a captured graph")
        return self.forward(plan) if result is None else result

    def release(self, sequence_ids: list[int]) -> None:
        """Release request-local image embeddings after Engine retirement."""

        for sequence_id in sequence_ids:
            self._media_features.pop(sequence_id, None)
            self._drafts.pop(sequence_id, None)


__all__ = ["Qwen3_5CudaRunner"]
