"""Qwen3.5 CUDA runner with optional bundled-MTP decoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from numpy.typing import NDArray

from resin import BatchPhase, RunnerConfig
from resin.backend.capacity import plan_cache_capacity
from resin.backend.cuda.completion import (
    ContinuationState,
    CudaContinuation,
    DeviceAcceptance,
    DeviceCompletion,
    DeviceFeedback,
)
from resin.backend.cuda.graphs import (
    CudaGraphMode,
    CudaGraphResult,
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
    forward_paged,
    graph_sample,
    pad_sampling_inputs,
    validate_hybrid_plan,
)
from resin.backend.cuda.scoring import format_output, prompt_scores, score_completion
from resin.backend.cuda.slab import graph_scratch_parent_count
from resin.backend.cuda.staging import TensorStager
from resin.backend.cuda.state import execute_state
from resin.backend.cuda.vision_cache import VisionFeatureCache
from resin.backend.plans import PagedCachePlan
from resin.backend.sampling import (
    SAMPLING_OFFSET_STRIDE,
    PromptScores,
)
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelGroups, ParallelismCapabilities
from resin.models.vision import (
    image_grid,
    media_identity,
    media_key,
    mrope_delta,
    mrope_positions,
    packed_images,
    span_slices,
)
from resin.recipe import Media
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


@dataclass(frozen=True, slots=True)
class _DecodeResult:
    logits: torch.Tensor
    acceptance: DeviceAcceptance
    next_draft: torch.Tensor | None = None
    next_hidden: torch.Tensor | None = None

    @property
    def parity_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.acceptance.parity_tensors + tuple(
            tensor
            for tensor in (self.next_draft, self.next_hidden)
            if tensor is not None
        )


@dataclass(frozen=True, slots=True)
class _Feedback(DeviceFeedback):
    """Owned model payload attached to the shared admission generation."""

    continuation: ContinuationState
    draft_ids: torch.Tensor | None = None
    draft_hidden: torch.Tensor | None = None

    def restrict(self, rows):
        DeviceFeedback.restrict(self, rows)
        if self.draft_ids is not None:
            self.draft_ids[rows:].zero_()
        if self.draft_hidden is not None:
            self.draft_hidden[rows:].zero_()
        return self

    def inputs(self):
        inputs = {
            "token_ids": self.token_ids,
            "positions": self.positions,
            "sampling_offsets": (self.positions + 1) * SAMPLING_OFFSET_STRIDE,
        }
        if self.draft_ids is not None:
            inputs["draft_ids"] = self.draft_ids
        if self.draft_hidden is not None:
            inputs["draft_hidden"] = self.draft_hidden
        return inputs


class _Completion(DeviceCompletion):
    """Shared device completion that also harvests the next MTP drafts on the host."""

    def __init__(
        self, runner, sequence_ids, next_drafts, *args, next_hidden=None, **kwargs
    ):
        # Queue the draft copy before the base class records its completion event.
        self._draft_stager = TensorStager(runner.device)
        self._next_drafts = (
            None
            if next_drafts is None
            else self._draft_stager.output("next_drafts", next_drafts)
        )
        self._next_hidden = next_hidden
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
        if self._next_hidden is not None:
            for row, sequence in enumerate(self._sequence_ids):
                runner._draft_hidden[sequence] = self._next_hidden[row]
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
        speculative_draft_tokens: int = 0,
        device: str | torch.device,
        gpu_memory_utilization: float = 0.90,
        cache_slab_capacity: int | None = None,
        kv_cache_dtype: str = "fp8",
        nvfp4_backend: str = "auto",
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
        cuda_graph_decode_shapes: object | None = None,
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
            isinstance(speculative_draft_tokens, bool)
            or not isinstance(speculative_draft_tokens, int)
            or speculative_draft_tokens < 0
        ):
            raise ValueError("speculative_draft_tokens must be a non-negative integer")
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
        self.speculative_draft_tokens = speculative_draft_tokens
        nvfp4_backend = select_nvfp4_backend(
            nvfp4_backend, torch.backends.cudnn.version()
        )
        self._model, self.config, self.raw_config = load_model(
            self.model_path,
            device=self.device,
            include_mtp=speculative_draft_tokens > 0,
            include_vision=True,
            nvfp4_backend=nvfp4_backend,
            tensor_parallel=self.expert_parallel,
            attention_parallel=self.attention_parallel,
        )
        self.attention_config = self._model.language_model.config
        self._configure_parallel_graphs(self._model, parallel_groups)
        self.image_token_id = self.raw_config.get("image_token_id")
        vision_config = self.raw_config.get("vision_config") or {}
        self.spatial_merge_size = int(vision_config.get("spatial_merge_size", 2))
        self.vision_patch_size = int(vision_config.get("patch_size", 16))
        self.vision_temporal_patch_size = int(
            vision_config.get("temporal_patch_size", 2)
        )
        self._media_features: dict[int, tuple[tuple[Media, ...], torch.Tensor]] = {}
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
        self._speculation = SpeculativeStats(speculative_draft_tokens)
        self._continuation = CudaContinuation()
        cache_dtype = self.kv_cache_torch_dtype

        self.cache_geometry = CacheGeometry.from_config(
            self.attention_config,
            include_mtp=speculative_draft_tokens > 0,
            kv_element_bytes=cache_dtype.itemsize,
        )
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity = plan_cache_capacity(
            total_memory,
            free_memory,
            gpu_memory_utilization,
            self.cache_geometry.slab_bytes,
            reserved_parent_count=1
            + graph_scratch_parent_count(self.cache_geometry.packings, max_batch_size),
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
            max_batch_size=max_batch_size,
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
            speculative=bool(speculative_draft_tokens),
            max_batch_size=max_batch_size,
        )
        if any(shape[0] > max_batch_size for shape in decode_graph_shapes):
            raise ValueError("Qwen3.5 CUDA graph batch capacity exceeds max_batch_size")
        if any(shape[1] > self.max_seq_len for shape in decode_graph_shapes):
            raise ValueError("Qwen3.5 CUDA graph sequence capacity exceeds max_seq_len")
        if any(
            shape[1] <= speculative_draft_tokens + 1 for shape in decode_graph_shapes
        ):
            raise ValueError("Qwen3.5 CUDA graph sequence capacity is too small")
        self._graph_workspace = torch.empty(
            _GRAPH_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
        )
        # Retain the next cycle's first proposal and, for K>1, its MTP hidden
        # seed. One-draft verification also keeps deferred GDN updates.
        self._drafts: dict[int, int] = {}
        self._draft_hidden: dict[int, torch.Tensor] = {}
        self._verify_offsets = torch.arange(
            self.speculative_draft_tokens + 1, device=self.device
        )
        self._window_offsets = torch.arange(3, device=self.device)
        geometry = self.attention_config
        self._pending = (
            torch.empty(
                (
                    geometry.linear_layer_count,
                    max_batch_size,
                    speculative_draft_tokens,
                    geometry.linear_num_value_heads,
                    geometry.linear_key_head_dim + geometry.linear_value_head_dim + 2,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            if speculative_draft_tokens
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
        self.runner_config = RunnerConfig(
            max_batch_tokens=PREFILL_CHUNK_SIZE,
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            continuation_prefill=True,
            max_accepted_per_step=speculative_draft_tokens + 1,
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

    def _host_rope_deltas(self, plan: BatchPlan) -> NDArray[np.int64]:
        return np.asarray(
            [
                mrope_delta(media, self.vision_patch_size, self.spatial_merge_size)
                if (media := plan.media[row])
                else 0
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
        media: tuple[Media, ...] | None,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if media is None:
            return positions[None].expand(3, -1, -1)
        absolute = positions[0].cpu().numpy()
        columns = mrope_positions(
            media, absolute, self.vision_patch_size, self.spatial_merge_size
        )
        return torch.from_numpy(columns).to(self.device)[:, None]

    def _features_for(self, sequence_id: int, media: tuple[Media, ...]) -> torch.Tensor:
        existing = self._media_features.get(sequence_id)
        if existing is not None:
            if media_identity(existing[0]) != media_identity(media):
                raise ValueError("a live Qwen sequence changed its media object")
            return existing[1]
        image_grid(media, self.vision_patch_size, self.spatial_merge_size)
        cached = self._vision_feature_cache.get(media_key(media))
        if cached is None:
            visual = self._model.visual
            if visual is None:
                raise RuntimeError("Qwen checkpoint did not load a visual tower")
            pixels, grid = packed_images(
                media,
                self.vision_patch_size,
                merge=self.spatial_merge_size,
                temporal=self.vision_temporal_patch_size,
            )
            with torch.inference_mode():
                output = visual(
                    torch.from_numpy(pixels).to(self.device, dtype=torch.bfloat16),
                    torch.from_numpy(grid).to(self.device),
                )
                features = output.pooler_output.detach()
            if features.shape != (
                sum(item.span[1] for item in media),
                self.config.hidden_size,
            ):
                raise ValueError("Qwen vision features do not match image placeholders")
            cached = self._vision_feature_cache.put(media_key(media), (features,))
        (features,) = cached
        assert features is not None
        if features.shape != (
            sum(item.span[1] for item in media),
            self.config.hidden_size,
        ):
            raise ValueError("Qwen vision features do not match image placeholders")
        self._media_features[sequence_id] = (media, features)
        return features

    def _merge_media_embeddings(
        self,
        plan: BatchPlan,
        row: int,
        token_ids: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> tuple[Media, ...] | None:
        media = plan.media[row]
        if media is None:
            return None
        start = int(plan.context_lengths[row])
        slices = tuple(
            span_slices(
                (item.span for item in media), start, start + token_ids.shape[1]
            )
        )
        if slices:
            features = self._features_for(int(plan.sequence_ids[row]), media)
            for destination, source in slices:
                embeddings[0, destination].copy_(features[source])
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
        if not self.speculative_draft_tokens:
            del inputs["draft_ids"]
        elif not self._fast_speculation:
            inputs["draft_hidden"] = torch.zeros(
                shape[0],
                self.config.hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
        inputs.update(self._cache.graph_shape_capture_inputs(*shape))
        return inputs

    @property
    def _fast_speculation(self) -> bool:
        """One MTP draft verified with the target in a single two-token forward."""
        return self.speculative_draft_tokens == 1

    def _graph_specs(self) -> tuple[GraphAttentionSpec, ...]:
        target = self.config.full_layer_count
        if not self.speculative_draft_tokens:
            return (GraphAttentionSpec(0, target, 1, 1),)
        width = self.speculative_draft_tokens + 1
        return (
            *(
                GraphAttentionSpec(2, 1, 1, step, 1)
                for step in range(1, self.speculative_draft_tokens)
            ),
            GraphAttentionSpec(0, target, width, width),
            GraphAttentionSpec(2, 1, width, width, 1),
        )

    def _refresh_graph_attention(self, attention, positions: torch.Tensor) -> None:
        """Resolve exact readable pages from GPU positions and the frozen lease."""
        attention.refresh(positions)

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        batch_size, graph_seq_len = shape
        cache = self._cache.graph_cache(*shape)
        attention = GraphPagedAttentionRuntime(
            self.device,
            cache,
            self._graph_specs(),
            num_attention_heads=self.attention_config.num_attention_heads,
            num_key_value_heads=self.attention_config.num_key_value_heads,
            head_dim=self.config.head_dim,
            workspace=self._graph_workspace,
        )
        token_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
        positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        rope_deltas = torch.empty(batch_size, dtype=torch.long, device=self.device)
        draft_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
        sampling = self._sampler.empty(batch_size, self.device)
        draft_seed = (
            torch.empty(
                batch_size,
                self.config.hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            if self.speculative_draft_tokens > 1
            else None
        )
        output = CudaGraphResult()

        if self._fast_speculation:
            phase_timing = self._new_phase_timing("target_verify", "decode", "draft")
        elif self.speculative_draft_tokens:
            phase_timing = self._new_phase_timing(
                "draft", "target_verify", "decode", "draft"
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
            logits = self._model.logits(hidden[:, -1])
            samples = graph_sample(self._sampler, logits, sampling, 0)[:, None]
            result = _DecodeResult(
                logits,
                DeviceAcceptance.from_samples(samples[:, :0], samples, positions),
            )
            output.publish(result, parity=result.parity_tensors)

        def speculative_launch() -> None:
            if phase_timing is not None:
                phase_timing.mark(0)
            attention.reset()
            self._refresh_graph_attention(attention, positions)

            def sample(logits, step):
                if logits.ndim == 3:
                    return self._sampler.sample_block(logits, sampling, start_step=step)
                return graph_sample(self._sampler, logits, sampling, step)

            verify_phase = int(not self._fast_speculation)
            result = self._mtp_step(
                cache,
                token_ids,
                positions,
                rope_deltas,
                draft_ids,
                draft_seed,
                sample,
                timing=phase_timing,
                verify_phase=verify_phase,
                attention_runtime=attention,
                max_sequence_length=graph_seq_len,
            )
            if phase_timing is not None:
                phase_timing.mark(verify_phase + 3)
            output.publish(result, parity=result.parity_tensors)

        launch = (
            speculative_launch if self.speculative_draft_tokens else ordinary_launch
        )
        inputs = {
            "token_ids": token_ids,
            "positions": positions,
            "rope_deltas": rope_deltas,
        }
        if self.speculative_draft_tokens:
            inputs["draft_ids"] = draft_ids
            if draft_seed is not None:
                inputs["draft_hidden"] = draft_seed
        inputs.update(self._cache.graph_device_inputs(cache))
        inputs.update(attention.graph_inputs())
        inputs.update(self._sampler.graph_inputs(sampling))

        def prepare() -> None:
            attention.reset()
            self._cache.reset_null()

        return CudaGraphStep(
            inputs=inputs,
            output=output,
            launch=launch,
            phase_timing=phase_timing,
            prepare=prepare,
            resources=(cache, attention),
        )

    def _graph_forward(
        self,
        plan: BatchPlan,
        cache_plan: PagedCachePlan,
        *,
        enqueue: bool = False,
        feedback=None,
    ):
        batch_size = plan.num_sequences
        history_pages = max(
            int(np.diff(cache_plan.groups[index].block_indptr).max())
            for index in (0, 2)
            if index < len(cache_plan.groups)
        )
        required_length = (
            int(np.max(plan.positions))
            + self.speculative_draft_tokens
            + 1
            + self.cache_layout.decode_write_extra_tokens
        )
        if not self.speculative_draft_tokens:
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
        cache, attention = self._captured_cuda_graph("decode", shape).resources
        inputs.update(self._cache.graph_plan_inputs(cache, cache_plan))
        inputs.update(attention.plan(cache_plan, np.asarray(plan.positions)))
        self._host_attention_plans = getattr(self, "_host_attention_plans", 0) + 1
        inputs.update(
            pad_sampling_inputs(self._sampler, plan, self.vocab_size, padding)
        )
        sequence_ids = [int(sequence) for sequence in plan.sequence_ids]
        if self.speculative_draft_tokens:
            if any(sequence not in self._drafts for sequence in sequence_ids):
                staging = TensorStager(self.device)
                cache = self._cache.tensorize(cache_plan, staging)
                self._initialize_drafts(
                    plan,
                    cache,
                    staging.device("initial_anchors", plan.token_ids, torch.long),
                    staging.device("initial_positions", plan.positions, torch.long),
                )
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
            if not self._fast_speculation:
                inputs["draft_hidden"] = self._persistent_draft_hidden(
                    sequence_ids, shape[0]
                )
        completion = self._enqueue_graph_output(
            plan, shape, inputs, feedback=feedback, asynchronous=enqueue
        )
        return completion if enqueue else completion.result()

    def _enqueue_graph_output(
        self, plan, shape, inputs, *, feedback=None, asynchronous=True
    ):
        sequence_ids = tuple(map(int, plan.sequence_ids))
        if feedback is not None:
            inputs.update(feedback.inputs())
        result = self._run_cuda_graph("decode", shape, inputs)
        acceptance = result.acceptance
        base = acceptance.feedback()
        next_drafts = (
            result.next_draft.clone() if result.next_draft is not None else None
        )
        next_hidden = (
            result.next_hidden.clone() if result.next_hidden is not None else None
        )
        state = (
            feedback.continuation
            if feedback is not None
            else self._continuation.begin(
                sequence_ids, shape, self._captured_cuda_graph("decode", shape).inputs
            )
        )
        completion = _Completion(
            self,
            sequence_ids,
            next_drafts,
            acceptance.tokens,
            acceptance.counts,
            _Feedback(base.token_ids, base.positions, state, next_drafts, next_hidden),
            rows=plan.num_sequences,
            device=self.device,
            stats=self._speculation if self.speculative_draft_tokens else None,
            next_hidden=next_hidden,
            plan=plan,
            scores=score_completion(result.logits, acceptance, plan),
        )
        # A later conservative plan can advance by the Engine's maximum output
        # width. Drain before it could exceed this graph's token capacity;
        # the next normal batch may select a different graph bucket or go eager.
        maximum = self.runner_config.max_accepted_per_step
        if np.any(
            np.asarray(plan.positions, dtype=np.int64)
            + 2 * maximum
            + self.cache_layout.decode_write_extra_tokens
            > shape[1]
        ):
            completion.feedback = None
            self._continuation.drain(state)
        if asynchronous:
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
            decode_lookahead=self.speculative_draft_tokens,
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
        media: tuple[Media, ...] | None,
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

    def _prefill(
        self,
        plan: BatchPlan,
        rows: tuple[tuple[int, int], ...],
        cache: DeviceCache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> ForwardOutput:
        final_logits: list[torch.Tensor] = []
        prompt_rows: list[PromptScores | None] = []
        states: list[tuple] = []
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
            media = self._merge_media_embeddings(plan, row, row_ids, embeddings)
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
            media = plan.media[row]
            self._prefill_mtp(
                row_cache,
                row_ids,
                row_positions,
                hidden,
                previous_hidden,
                context,
                media,
            )
            self._store_state(
                row_cache,
                row_positions[:, -1],
                recurrent,
                convolution,
                hidden[:, -1],
            )
        scores = PromptScores.concat(cast(list[PromptScores], prompt_rows))
        return format_output(logits, sampled, plan, prompt_logprobs=scores)

    @torch.inference_mode()
    def _initialize_drafts(self, plan, cache, anchors, positions):
        sequence_ids = tuple(map(int, plan.sequence_ids))
        missing = [
            row
            for row, sequence in enumerate(sequence_ids)
            if sequence not in self._drafts
        ]
        if not missing:
            return
        # Prefill cannot append the sampled anchor beyond its query lease. Its
        # MTP pair is initialized now, at physical P inside the decode lease.
        previous = (positions - 1).clamp_min(0)
        hidden = self._mtp_forward(
            cache,
            anchors[:, None],
            cache.hidden_at(previous)[:, None],
            previous[:, None],
            rope_deltas=self._rope_deltas(plan),
        )[:, -1]
        drafts = self._sample(self._model.logits(hidden), plan, 1).tolist()
        for row in missing:
            sequence = sequence_ids[row]
            self._drafts[sequence] = int(drafts[row])
            if self.speculative_draft_tokens > 1:
                self._draft_hidden[sequence] = hidden[row].clone()

    def _persistent_draft_hidden(self, sequence_ids, capacity):
        hidden = torch.zeros(
            capacity, self.config.hidden_size, dtype=torch.bfloat16, device=self.device
        )
        for row, sequence in enumerate(sequence_ids):
            previous = self._draft_hidden.get(sequence)
            if previous is not None:
                hidden[row].copy_(previous)
        return hidden

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

    def _verification_pool(self, cache, previous, positions, rows):
        if self._pending is None:
            raise RuntimeError("speculative verification requires pending GDN state")
        return replace(
            cache.recurrent_pool(previous, positions), pending=self._pending[:, :rows]
        )

    def _commit_verification(self, cache, pool, positions, hidden, windows, acceptance):
        from resin_models_cuda.qwen3_5.gdn_decode import gdn_commit

        config = self.attention_config
        accepted = acceptance.accepted_drafts
        commit_positions = positions + accepted
        gdn_commit(
            pool,
            config.linear_layer_count,
            pool.destination,
            cache.state_offsets(commit_positions),
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
        cache.store_convolution(commit_positions, convolution, selected)

    def _mtp_step(
        self,
        cache,
        anchors,
        positions,
        rope_deltas,
        first_draft,
        draft_seed,
        sample,
        *,
        timing,
        verify_phase,
        attention_runtime=None,
        max_sequence_length=None,
    ) -> _DecodeResult:
        """Verify the proposed block, commit accepted state, and repair MTP for the next step."""
        execution = dict(
            attention_runtime=attention_runtime, max_sequence_length=max_sequence_length
        )
        if self._fast_speculation:
            drafts = first_draft[:, None]
        else:
            columns = [first_draft]
            draft, hidden = first_draft, draft_seed[:, None]
            for step in range(1, self.speculative_draft_tokens):
                hidden = self._mtp_forward(
                    cache,
                    draft[:, None],
                    hidden,
                    (positions + step - 1)[:, None],
                    rope_deltas=rope_deltas,
                    **execution,
                )
                draft = sample(self._model.logits(hidden[:, -1]), step + 1)
                columns.append(draft)
            drafts = torch.stack(columns, dim=1)
        if timing is not None and verify_phase:
            timing.mark(verify_phase)
        verify_ids = torch.cat((anchors[:, None], drafts), dim=1)
        verify_positions = positions[:, None] + self._verify_offsets
        previous = (positions - 1).clamp_min(0)
        pool = self._verification_pool(cache, previous, positions, len(positions))
        hidden, _, windows = self._target_forward(
            cache,
            verify_ids,
            verify_positions,
            None,
            cache.convolution_at(previous),
            rotary_positions=(verify_positions + rope_deltas[:, None])[None].expand(
                3, -1, -1
            ),
            recurrent_pool=pool,
            **execution,
        )
        logits = self._model.logits(hidden)
        targets = sample(logits, 0)
        acceptance = DeviceAcceptance.from_samples(drafts, targets, positions)
        if timing is not None:
            timing.mark(verify_phase + 1)
        self._commit_verification(cache, pool, positions, hidden, windows, acceptance)
        if timing is not None:
            timing.mark(verify_phase + 2)
        repaired = self._mtp_forward(
            cache,
            acceptance.tokens,
            hidden,
            verify_positions,
            rope_deltas=rope_deltas,
            **execution,
        )
        selected = acceptance.accepted_drafts[:, None, None].expand(
            -1, 1, repaired.shape[-1]
        )
        next_hidden = repaired.gather(1, selected).squeeze(1)
        next_draft = sample(
            self._model.logits(next_hidden), self.speculative_draft_tokens + 1
        )
        return _DecodeResult(
            logits,
            acceptance,
            next_draft,
            None if self._fast_speculation else next_hidden,
        )

    def _speculative_decode(
        self,
        plan: BatchPlan,
        cache: DeviceCache,
        anchors: torch.Tensor,
        positions: torch.Tensor,
    ) -> ForwardOutput:
        phase_timing = self._eager_phase_timing(
            "draft", "target_verify", "decode", "draft"
        )
        if phase_timing is not None:
            phase_timing.mark(0)
        rope_deltas = self._rope_deltas(plan)
        sequence_ids = tuple(map(int, plan.sequence_ids))
        self._initialize_drafts(plan, cache, anchors, positions)
        draft_input = torch.tensor(
            [
                self._drafts.get(sequence, int(anchor))
                for sequence, anchor in zip(sequence_ids, plan.token_ids, strict=True)
            ],
            device=self.device,
        )
        draft_seed = (
            None
            if self._fast_speculation
            else self._persistent_draft_hidden(sequence_ids, len(sequence_ids))
        )
        result = self._mtp_step(
            cache,
            anchors,
            positions,
            rope_deltas,
            draft_input,
            draft_seed,
            lambda logits, step: self._sample(logits, plan, step),
            timing=phase_timing,
            verify_phase=1,
        )
        acceptance = result.acceptance
        if phase_timing is not None:
            phase_timing.mark(4)
        return _Completion(
            self,
            sequence_ids,
            result.next_draft,
            acceptance.tokens,
            acceptance.counts,
            acceptance.feedback(),
            rows=plan.num_sequences,
            stats=self._speculation,
            device=self.device,
            plan=plan,
            scores=score_completion(result.logits, acceptance, plan),
            next_hidden=result.next_hidden,
        ).result()

    @observe_execution
    @execute_state
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self._check_cuda_graphs(plan.phase)
        self._continuation.drain()
        rows, cache_plan = self._prepare_batch(plan)
        return forward_paged(self, plan, rows, cache_plan)

    def _prepare_batch(self, plan):
        rows = self._validate_plan(plan)
        lookahead = (
            self.speculative_draft_tokens + self.cache_layout.decode_write_extra_tokens
            if plan.phase == BatchPhase.DECODE
            else 0
        )
        return rows, PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=lookahead
        )

    def _eager_forward(self, plan, rows, cache, token_ids, positions):
        if plan.phase == BatchPhase.PREFILL:
            with self._prefill_graphs():
                return self._prefill(plan, rows, cache, token_ids, positions)
        if self.speculative_draft_tokens:
            return self._speculative_decode(plan, cache, token_ids, positions)
        return self._ordinary_decode(plan, cache, token_ids, positions)

    @observe_execution
    @execute_state
    def enqueue(self, plan: BatchPlan, *, feedback=None):
        """Device-feedback decode over one retained graph admission."""
        if plan.phase != BatchPhase.DECODE or plan.needs_prompt_logprobs:
            if feedback is not None:
                raise RuntimeError("a device continuation requires captured decode")
            return self.forward(plan)
        self._check_cuda_graphs(plan.phase)
        if feedback is not None:
            if not isinstance(feedback, _Feedback):
                raise ValueError("Qwen continuation requires owned device feedback")
            inputs = self._continuation.resume(
                tuple(map(int, plan.sequence_ids)), feedback.continuation
            )
            shape = feedback.continuation.shape
            with self._cuda_failure, torch.inference_mode():
                # The Engine retains the same lease and row order. All dense
                # tables, policies and rotary deltas already occupy graph-owned
                # buffers; stage only owned token/position/draft feedback.
                return self._enqueue_graph_output(
                    plan, shape, inputs, feedback=feedback
                )
        self._continuation.drain()
        _, cache_plan = self._prepare_batch(plan)
        with self._cuda_failure, torch.inference_mode():
            result = self._graph_forward(plan, cache_plan, enqueue=True)
        return self.forward(plan) if result is None else result

    def release(self, sequence_ids: list[int]) -> None:
        """Release request-local image embeddings after Engine retirement."""

        self._continuation.drain()
        for sequence_id in sequence_ids:
            self._media_features.pop(sequence_id, None)
            self._drafts.pop(sequence_id, None)
            self._draft_hidden.pop(sequence_id, None)


__all__ = ["Qwen3_5CudaRunner"]
