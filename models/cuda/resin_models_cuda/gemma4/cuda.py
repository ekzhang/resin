"""Gemma 4 31B NVFP4 CUDA runner for Blackwell."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from resin import (
    BatchPhase,
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
    RunnerConfig,
)
from resin.backend.capacity import CacheCapacityPlan, plan_cache_capacity
from resin.backend.cuda.completion import DeviceAcceptance, DeviceCompletion
from resin.backend.cuda.failures import GraphInputError
from resin.backend.cuda.graphs import (
    DEFAULT_MAX_NUM_SEQS,
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
    graph_sample_block,
    pad_sampling_inputs,
)
from resin.backend.cuda.vision_cache import VisionFeatureCache
from resin.backend.resources import CachePoolBinding
from resin.backend.sampling import (
    SAMPLING_OFFSET_STRIDE,
    format_output,
    prompt_scores,
    token_output,
)
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelGroups, ParallelismCapabilities
from resin.models.gemma4.config import SLIDING_ATTENTION
from resin.models.gemma4.plan import (
    PAGE_SIZE,
    PREFILL_CHUNK_SIZE,
    SLIDING_WINDOW,
    WINDOW_RETENTION_TOKENS,
    Gemma4BatchPlan,
    PagedAttentionPlan,
    build_gemma4_batch_plan,
    expand_gemma4_decode_batch,
    gemma4_assistant_context,
)
from resin.recipe import Gemma4Media
from resin_models_cuda.gemma4.assistant_checkpoint import (
    DEFAULT_ASSISTANT,
    load_assistant,
    resolve_assistant_path,
)
from resin_models_cuda.gemma4.checkpoint import (
    DEFAULT_MODEL,
    load_model,
    resolve_model_path,
)
from resin_models_cuda.gemma4.model import (
    Gemma4ForConditionalGeneration,
)
from resin_models_cuda.gemma4.parallel import attention_geometry
from resin_models_cuda.nvfp4 import select_nvfp4_backend as _select_nvfp4_backend

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput

_GRAPH_WORKSPACE_BYTES = 128 * 1024 * 1024
_PREFILL_WORKSPACE_BYTES_PER_ELEMENT = 16
_DEFAULT_PREFILL_WORKSPACE_BYTES = (
    PREFILL_CHUNK_SIZE * 32 * (256 + 512) * _PREFILL_WORKSPACE_BYTES_PER_ELEMENT
)
_DEFAULT_WORKSPACE_BYTES = 2 * _GRAPH_WORKSPACE_BYTES + _DEFAULT_PREFILL_WORKSPACE_BYTES
_LOCAL_GRAPH_PAGE_CAPACITY = (
    1 + (WINDOW_RETENTION_TOKENS - 1 + PAGE_SIZE - 1) // PAGE_SIZE
)


def plan_gemma4_kv_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    static_memory_fraction: float,
    *,
    workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
    cache_element_bytes: int = 1,
    tensor_parallel_size: int = 1,
    config=None,
) -> CacheCapacityPlan:
    if cache_element_bytes not in {1, 2}:
        raise ValueError("cache_element_bytes must be one or two")
    if (
        type(tensor_parallel_size) is not int
        or tensor_parallel_size <= 0
        or 4 % tensor_parallel_size
    ):
        raise ValueError("tensor_parallel_size must divide both Gemma KV head counts")
    # One parent holds all 50 local layers for one 64-token page. Global
    # history pages use an equal-sized, differently shaped view of a parent.
    local_layers = getattr(config, "local_layer_count", 50)
    kv_heads = 16 if config is None else config.num_key_value_heads
    head_dim = 256 if config is None else config.head_dim
    bytes_per_parent = (
        2
        * local_layers
        * PAGE_SIZE
        * (kv_heads // tensor_parallel_size)
        * head_dim
        * cache_element_bytes
    )
    return plan_cache_capacity(
        total_memory_bytes,
        free_memory_bytes,
        static_memory_fraction,
        bytes_per_parent,
        overhead_bytes=workspace_bytes,
    )


class _Gemma4KvArena:
    def __init__(
        self, device, config, parent_capacity: int, *, dtype, layout=None
    ) -> None:
        self.dtype = dtype
        self.parent_capacity = parent_capacity
        self.layout = layout
        physical_parent_capacity = parent_capacity + 1
        local_shape = (
            physical_parent_capacity,
            2,
            config.local_layer_count,
            PAGE_SIZE,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.storage = torch.empty(local_shape, dtype=self.dtype, device=device)
        local = self.storage.permute(1, 2, 0, 3, 4, 5)
        self.local_keys, self.local_values = local[0], local[1]

        packing = config.history_blocks_per_parent
        global_shape = (
            physical_parent_capacity * packing,
            2,
            config.global_layer_count,
            PAGE_SIZE,
            config.num_global_key_value_heads,
            config.global_head_dim,
        )
        global_view = self.storage.view(global_shape).permute(1, 2, 0, 3, 4, 5)
        self.global_keys, self.global_values = global_view[0], global_view[1]
        parent_bytes = self.storage[0].numel() * self.storage.element_size()
        self._storage_block_bytes = (parent_bytes // packing, parent_bytes)
        self._storage_geometry = (str(dtype), local_shape[1:], global_shape[1:])

    @property
    def cache_storage_signature(self):
        from resin.backend.cache_storage import cache_signature

        if self.layout is None:
            raise ValueError("cache storage requires the Engine's frozen cache layout")
        return cache_signature(
            "resin.gemma4-kv.v1",
            self.layout,
            self._storage_block_bytes,
            self._storage_geometry,
        )

    def cache_storage_buffers(self, transfer):
        from resin.backend.cache_storage import slab_buffers

        if self.layout is None:
            raise ValueError("cache storage requires the Engine's frozen cache layout")
        return slab_buffers(
            transfer, self.storage, self.layout, self._storage_block_bytes
        )


@dataclass(frozen=True, slots=True)
class _DevicePagedPlan:
    qo_indptr: Tensor
    kv_indptr: Tensor
    page_indices: Tensor
    last_page_lengths: Tensor
    append_pages: Tensor
    append_offsets: Tensor


@dataclass(frozen=True, slots=True)
class _MediaFeatures:
    media: Gemma4Media
    image: Tensor | None
    video: Tensor | None
    positions: dict[int, tuple[str, int]]


class Gemma4CudaRunner(CudaRunner):
    """Dynamic paged forward pass with FP8 KV and W4A16 NVFP4 MLPs."""

    page_size = PAGE_SIZE
    prefill_chunk_size = PREFILL_CHUNK_SIZE
    sliding_window = SLIDING_WINDOW
    local_graph_page_capacity = _LOCAL_GRAPH_PAGE_CAPACITY
    max_num_seqs = DEFAULT_MAX_NUM_SEQS
    assistant_geometry_fields = (
        "num_attention_heads",
        "num_key_value_heads",
        "num_global_key_value_heads",
        "head_dim",
        "global_head_dim",
    )
    parallelism_capabilities = ParallelismCapabilities(attn_tp=True, expert_tp=True)

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        cache_parent_capacity: int | None = None,
        revision: str | None = None,
        assistant_model: str | Path | None = None,
        assistant_revision: str | None = None,
        speculative_draft_tokens: int = 0,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        device: str | torch.device,
        nvfp4_backend: str = "auto",
        kv_cache_dtype: str = "fp8",
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
        cuda_graph_decode_shapes: object | None = None,
        static_memory_fraction: float = 0.95,
        parallel_groups: ParallelGroups | None = None,
    ) -> None:
        if type(prefill_chunk_size) is not int or prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be a positive integer")
        self.prefill_chunk_size = prefill_chunk_size
        if parallel_groups is not None:
            self.parallelism_capabilities.validate(
                parallel_groups.topology, model="Gemma 4"
            )
        self.parallel_groups = parallel_groups
        self.attention_parallel = TensorParallelContext.from_group(
            None if parallel_groups is None else parallel_groups.attn_tp
        )
        self.expert_parallel = TensorParallelContext.from_group(
            None if parallel_groups is None else parallel_groups.expert_tp
        )
        if cache_parent_capacity is not None and (
            isinstance(cache_parent_capacity, bool)
            or not isinstance(cache_parent_capacity, int)
            or cache_parent_capacity <= 0
        ):
            raise ValueError("cache_parent_capacity must be a positive integer")
        if (
            isinstance(speculative_draft_tokens, bool)
            or not isinstance(speculative_draft_tokens, int)
            or speculative_draft_tokens < 0
        ):
            raise ValueError("speculative_draft_tokens must be a non-negative integer")
        super().__init__(
            device=device,
            label="Gemma 4",
            kv_cache_dtype=kv_cache_dtype,
            cuda_graph_prefill=cuda_graph_prefill,
            cuda_graph_decode=cuda_graph_decode,
        )
        nvfp4_backend = self._prepare_weight_backend(nvfp4_backend)
        cache_dtype = self.kv_cache_torch_dtype

        self.model_path = resolve_model_path(model, revision)
        self._model, self.config, self.raw_config = self._load_target(
            self.model_path,
            device=self.device,
            nvfp4_backend=nvfp4_backend,
            tensor_parallel=self.expert_parallel,
            attention_parallel=self.attention_parallel,
        )
        self.attention_config = attention_geometry(
            self.config, self.attention_parallel.layout
        )
        self.sliding_window = getattr(self.config, "sliding_window", SLIDING_WINDOW)
        self.window_retention_tokens = self.prefill_chunk_size + self.sliding_window - 1
        self.local_graph_page_capacity = (
            1 + (self.window_retention_tokens - 1 + PAGE_SIZE - 1) // PAGE_SIZE
        )
        self._configure_parallel_graphs(self._model, parallel_groups)
        self.nvfp4_backend = nvfp4_backend
        self.speculative_draft_tokens = speculative_draft_tokens
        self._assistant = None
        self.assistant_path: Path | None = None
        if speculative_draft_tokens:
            selected_assistant = assistant_model or DEFAULT_ASSISTANT
            self.assistant_path = resolve_assistant_path(
                selected_assistant, assistant_revision
            )
            self._assistant, assistant_config = self._load_assistant(
                self.assistant_path,
                device=self.device,
                tensor_parallel=self.expert_parallel,
                attention_parallel=self.attention_parallel,
            )
            if assistant_config.backbone_hidden_size != self.config.hidden_size:
                raise ValueError(
                    "assistant backbone width does not match Gemma 4 target"
                )
            if assistant_config.vocab_size != self.config.vocab_size:
                raise ValueError("assistant vocabulary does not match Gemma 4 target")
            for field in self.assistant_geometry_fields:
                if getattr(assistant_config, field) != getattr(self.config, field):
                    raise ValueError(f"assistant {field} does not match Gemma 4 target")
        self._speculation = SpeculativeStats(speculative_draft_tokens)
        self._graph_attention: dict[GraphShape, tuple[Any, Any]] = {}
        self._graph_spec_attention: dict[GraphShape, tuple[Any, Any, Any, Any]] = {}
        self._graph_verify_hidden: dict[GraphShape, Tensor] = {}
        self._graph_acceptance: dict[GraphShape, DeviceAcceptance] = {}
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity_plan = plan_gemma4_kv_capacity(
            total_memory,
            free_memory,
            static_memory_fraction,
            workspace_bytes=(
                2 * _GRAPH_WORKSPACE_BYTES
                + self.prefill_chunk_size
                * self.attention_config.num_attention_heads
                * (self.config.head_dim + self.config.global_head_dim)
                * _PREFILL_WORKSPACE_BYTES_PER_ELEMENT
            ),
            cache_element_bytes=cache_dtype.itemsize,
            tensor_parallel_size=self.attention_parallel.size,
            config=self.config,
        )
        local_capacity = capacity_plan.parent_capacity
        if cache_parent_capacity is not None:
            local_capacity = min(local_capacity, cache_parent_capacity)
        local_capacity = replicated_cache_capacity(
            local_capacity, None if parallel_groups is None else parallel_groups.replica
        )
        self.cache_parent_capacity = local_capacity
        self.cache_capacity_plan = replace(
            capacity_plan,
            parent_capacity=local_capacity,
            allocation_bytes=(local_capacity + 1) * capacity_plan.bytes_per_parent,
        )

        maximum_history_pages = (
            self.config.max_position_embeddings + PAGE_SIZE - 1
        ) // PAGE_SIZE
        decode_graph_shapes = resolve_decode_graph_shapes(
            cuda_graph_decode_shapes,
            policy=self.cuda_graph_decode,
            width=min(
                maximum_history_pages,
                self.cache_parent_capacity * self.config.history_blocks_per_parent,
            ),
            speculative=bool(speculative_draft_tokens),
            max_batch_size=self.max_num_seqs,
        )
        if any(batch > self.max_num_seqs for batch, _ in decode_graph_shapes):
            raise ValueError(
                f"Gemma 4 CUDA graph batch capacity exceeds {self.max_num_seqs}"
            )
        if any(width > maximum_history_pages for _, width in decode_graph_shapes):
            raise ValueError("Gemma 4 CUDA graph page width exceeds context length")
        self._configure_cuda_graphs(
            {
                "decode": CudaGraphMode(
                    decode_graph_shapes,
                    self._new_decode_graph_step,
                    lambda shape: self._graph_capture_inputs(
                        shape, self.config.vocab_size
                    ),
                )
            }
        )

        self.cache_layout = CacheLayout(
            parent_capacity=self.cache_parent_capacity,
            groups=[
                CacheGroupLayout(
                    PAGE_SIZE,
                    blocks_per_slab=self.config.history_blocks_per_parent,
                ),
                CacheGroupLayout(
                    PAGE_SIZE,
                    retention=CacheRetention.WINDOW,
                    window_tokens=self.window_retention_tokens,
                    read_window_tokens=self.sliding_window,
                ),
            ],
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=self.prefill_chunk_size,
            prefill_chunk_size=self.prefill_chunk_size,
            continuation_prefill=True,
            max_accepted_per_step=speculative_draft_tokens + 1,
            cache_layout=self.cache_layout,
        )
        self._configure_cache_binding(
            CachePoolBinding(self.cache_layout, resource="Gemma 4 KV")
        )
        self._cache = _Gemma4KvArena(
            self.device,
            self.attention_config,
            self.cache_parent_capacity,
            dtype=cache_dtype,
            layout=self.cache_layout,
        )
        self._graph_attention_workspaces = tuple(
            torch.zeros(_GRAPH_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
            for _ in range(2)
        )
        self._media_features: dict[int, _MediaFeatures] = {}
        self._vision_feature_cache = VisionFeatureCache()
        # CPU registry contains only slot numbers, never one Tensor per request.
        self._target_hidden: dict[int, int] = {}
        self._hidden_bank: Tensor | None = None
        self._hidden_free_slots: list[int] = []
        self._hidden_slot_generation = 0
        self._hidden_index_cache: dict[bool, tuple[object, Tensor]] = {}
        self._hidden_input_buffers: dict[int, Tensor] = {}
        self._build_attention_wrappers()
        self._prefill_feed_forward = self._configure_feed_forward_graphs()

    def _prepare_weight_backend(self, backend: str) -> str:
        capability = torch.cuda.get_device_capability(self.device)
        if capability not in {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}:
            raise RuntimeError(
                "Gemma 4 NVFP4 W4A16 requires Blackwell-class FP4 hardware, "
                f"got sm{capability[0]}{capability[1]}"
            )
        return _select_nvfp4_backend(backend, torch.backends.cudnn.version())

    def _load_target(self, *args, **kwargs):
        return load_model(*args, **kwargs)

    def _load_assistant(self, *args, **kwargs):
        return load_assistant(*args, **kwargs)

    def _configure_feed_forward_graphs(self):
        return self._configure_prefill_graphs(
            tuple(
                lambda hidden, _ids, layer=layer: self._feed_forward(layer, hidden)
                for layer in self._model.language_model.layers
            ),
            hidden_size=self.config.hidden_size,
            max_tokens=self.prefill_chunk_size,
        )

    @property
    def model(self) -> Gemma4ForConditionalGeneration:
        return self._model

    @property
    def model_memory_bytes(self) -> int:
        tensors = list(self._model.parameters()) + list(self._model.buffers())
        if self._assistant is not None:
            tensors += list(self._assistant.parameters()) + list(
                self._assistant.buffers()
            )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        return self._speculation.snapshot(include_draft_tokens=True)

    def _build_attention_wrappers(self) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        def prefill_workspace(head_dim: int) -> Tensor:
            size = (
                self.prefill_chunk_size
                * self.attention_config.num_attention_heads
                * head_dim
                * _PREFILL_WORKSPACE_BYTES_PER_ELEMENT
            )
            return torch.zeros(size, dtype=torch.uint8, device=self.device)

        local_workspace = prefill_workspace(self.config.head_dim)
        global_workspace = prefill_workspace(self.config.global_head_dim)
        self._workspaces = (local_workspace, global_workspace)
        self._local_prefill = BatchPrefillWithPagedKVCacheWrapper(
            local_workspace, "NHD", backend="auto"
        )
        self._global_prefill = BatchPrefillWithPagedKVCacheWrapper(
            global_workspace, "NHD", backend="auto"
        )
        self._local_decode = BatchDecodeWithPagedKVCacheWrapper(
            local_workspace, "NHD", backend="auto"
        )
        self._global_decode = BatchDecodeWithPagedKVCacheWrapper(
            global_workspace, "NHD", backend="auto"
        )

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._media_features.pop(request_id, None)
            slot = self._target_hidden.pop(request_id, None)
            if slot is not None:
                # Engine releases requests only after their writer leases finish.
                self._hidden_free_slots.append(slot)
                self._hidden_slot_generation += 1
        self._hidden_index_cache.clear()

    def _device_plan(self, plan: PagedAttentionPlan) -> _DevicePagedPlan:
        def tensor(values, dtype):
            return torch.as_tensor(values, dtype=dtype, device=self.device)

        append_slots = tensor(plan.append_slots, torch.int64)
        return _DevicePagedPlan(
            qo_indptr=tensor(plan.qo_indptr, torch.int32),
            kv_indptr=tensor(plan.kv_indptr, torch.int32),
            page_indices=tensor(plan.page_indices, torch.int32),
            last_page_lengths=tensor(plan.last_page_lengths, torch.int32),
            append_pages=torch.div(append_slots, PAGE_SIZE, rounding_mode="floor"),
            append_offsets=append_slots.remainder(PAGE_SIZE),
        )

    def _plan_attention(
        self,
        batch: Gemma4BatchPlan,
        local: _DevicePagedPlan,
        global_: _DevicePagedPlan,
        custom_mask: Tensor | None,
    ) -> tuple[Any, Any]:
        if batch.phase == BatchPhase.PREFILL:
            self._local_prefill.plan(
                local.qo_indptr,
                local.kv_indptr,
                local.page_indices,
                local.last_page_lengths,
                self.attention_config.num_attention_heads,
                self.attention_config.num_key_value_heads,
                self.config.head_dim,
                PAGE_SIZE,
                custom_mask=custom_mask,
                causal=custom_mask is None,
                sm_scale=1.0,
                window_left=self.sliding_window - 1,
                q_data_type=torch.bfloat16,
                kv_data_type=self._cache.dtype,
                o_data_type=torch.bfloat16,
            )
            self._global_prefill.plan(
                global_.qo_indptr,
                global_.kv_indptr,
                global_.page_indices,
                global_.last_page_lengths,
                self.attention_config.num_attention_heads,
                self.attention_config.num_global_key_value_heads,
                self.config.global_head_dim,
                PAGE_SIZE,
                causal=True,
                sm_scale=1.0,
                q_data_type=torch.bfloat16,
                kv_data_type=self._cache.dtype,
                o_data_type=torch.bfloat16,
            )
            return self._local_prefill, self._global_prefill

        common = {
            "pos_encoding_mode": "NONE",
            "q_data_type": torch.bfloat16,
            "kv_data_type": self._cache.dtype,
            "o_data_type": torch.bfloat16,
            "sm_scale": 1.0,
        }
        self._local_decode.plan(
            local.kv_indptr,
            local.page_indices,
            local.last_page_lengths,
            self.attention_config.num_attention_heads,
            self.attention_config.num_key_value_heads,
            self.config.head_dim,
            PAGE_SIZE,
            window_left=self.sliding_window - 1,
            **common,
        )
        self._global_decode.plan(
            global_.kv_indptr,
            global_.page_indices,
            global_.last_page_lengths,
            self.attention_config.num_attention_heads,
            self.attention_config.num_global_key_value_heads,
            self.config.global_head_dim,
            PAGE_SIZE,
            **common,
        )
        return self._local_decode, self._global_decode

    def _features_for(self, sequence_id: int, media: Gemma4Media) -> _MediaFeatures:
        existing = self._media_features.get(sequence_id)
        if existing is not None:
            if existing.media is not media:
                raise ValueError("a live Gemma 4 sequence changed its media object")
            return existing
        media.validate_placeholder_tokens(
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
        )
        cached = self._vision_feature_cache.get(media.vision_digest)
        if cached is None:
            with torch.inference_mode():
                image = None
                if media.pixel_values is not None:
                    assert media.image_position_ids is not None
                    image = self._model.get_image_features(
                        torch.from_numpy(media.pixel_values).to(
                            self.device, dtype=torch.bfloat16
                        ),
                        torch.from_numpy(media.image_position_ids).to(self.device),
                    )
                video = None
                if media.pixel_values_videos is not None:
                    assert media.video_position_ids is not None
                    video = self._model.get_video_features(
                        torch.from_numpy(media.pixel_values_videos).to(
                            self.device, dtype=torch.bfloat16
                        ),
                        torch.from_numpy(media.video_position_ids).to(self.device),
                    )
            if image is not None and len(image) != len(media.image_positions):
                raise ValueError("vision features do not match image placeholder count")
            if video is not None and len(video) != len(media.video_positions):
                raise ValueError("vision features do not match video placeholder count")
            cached = self._vision_feature_cache.put(media.vision_digest, (image, video))
        image, video = cached
        if image is not None and len(image) != len(media.image_positions):
            raise ValueError("vision features do not match image placeholder count")
        if video is not None and len(video) != len(media.video_positions):
            raise ValueError("vision features do not match video placeholder count")
        positions = {
            position: ("image", index)
            for index, position in enumerate(media.image_positions)
        }
        positions.update(
            {
                position: ("video", index)
                for index, position in enumerate(media.video_positions)
            }
        )
        features = _MediaFeatures(media, image, video, positions)
        self._media_features[sequence_id] = features
        return features

    def _plan_assistant_attention(self, *args, **kwargs):
        return self._plan_attention(*args, **kwargs)

    @staticmethod
    def _validate_unsplit_vision_blocks(
        media: Gemma4Media, query_start: int, query_end: int
    ) -> None:
        blocks = media.block_ids
        for position in range(query_start, min(query_end, len(blocks))):
            block = blocks[position]
            if block < 0:
                continue
            start = position
            while start and blocks[start - 1] == block:
                start -= 1
            end = position + 1
            while end < len(blocks) and blocks[end] == block:
                end += 1
            if start < query_start or end > query_end:
                raise ValueError(
                    "a Gemma 4 vision block crosses a prefill chunk boundary; "
                    "reduce earlier prompt length or increase the runner chunk size"
                )

    def _merge_media_embeddings(
        self,
        native: BatchPlan,
        batch: Gemma4BatchPlan,
        hidden: Tensor,
    ) -> dict[int, _MediaFeatures]:
        features_by_row = {}
        sequence_ids = np.asarray(native.sequence_ids, dtype=np.uint64)
        for row, media in enumerate(native.media):
            if media is None:
                continue
            if not isinstance(media, Gemma4Media):
                raise TypeError("Gemma4CudaRunner media must be Gemma4Media")
            start, end = map(int, batch.token_indptr[row : row + 2])
            positions = batch.positions[start:end]
            prompt_tokens = media.token_ids
            for flat_index, position in zip(range(start, end), positions, strict=True):
                absolute = int(position)
                if absolute < len(prompt_tokens) and (
                    int(batch.token_ids[flat_index]) != prompt_tokens[absolute]
                ):
                    raise ValueError(
                        "scheduled token does not match Gemma4Media prompt"
                    )
            vision_positions = set(media.image_positions + media.video_positions)
            if not any(int(position) in vision_positions for position in positions):
                continue
            self._validate_unsplit_vision_blocks(
                media, int(positions[0]), int(positions[-1]) + 1
            )
            features = self._features_for(int(sequence_ids[row]), media)
            features_by_row[row] = features
            for flat_index, position in zip(range(start, end), positions, strict=True):
                absolute = int(position)
                location = features.positions.get(absolute)
                if location is None:
                    continue
                kind, feature_index = location
                source = features.image if kind == "image" else features.video
                assert source is not None
                hidden[flat_index].copy_(source[feature_index])
        return features_by_row

    def _local_custom_mask(
        self,
        batch: Gemma4BatchPlan,
        features_by_row: dict[int, _MediaFeatures],
    ) -> Tensor | None:
        if batch.phase != BatchPhase.PREFILL or not features_by_row:
            return None
        masks = []
        any_bidirectional = False
        for row in range(len(batch.token_indptr) - 1):
            start, end = map(int, batch.token_indptr[row : row + 2])
            query = batch.positions[start:end, None]
            first = batch.local.first_token_positions[row]
            keys = np.arange(first, first + batch.local.kv_lengths[row])[None, :]
            allowed = (keys <= query) & (keys >= query - self.sliding_window + 1)
            features = features_by_row.get(row)
            if features is not None:
                block_ids = np.asarray(features.media.block_ids, dtype=np.int64)
                query_blocks = np.full(query.shape, -1, dtype=np.int64)
                key_blocks = np.full(keys.shape, -1, dtype=np.int64)
                valid_q = query[:, 0] < len(block_ids)
                valid_k = keys[0] < len(block_ids)
                query_blocks[valid_q, 0] = block_ids[query[valid_q, 0]]
                key_blocks[0, valid_k] = block_ids[keys[0, valid_k]]
                same = (query_blocks >= 0) & (query_blocks == key_blocks)
                if np.any(same & ~allowed):
                    any_bidirectional = True
                allowed |= same & (keys >= query - self.sliding_window + 1)
            masks.append(allowed.reshape(-1))
        if not any_bidirectional:
            return None
        return torch.as_tensor(
            np.concatenate(masks), dtype=torch.bool, device=self.device
        )

    def _run_transformer(
        self,
        hidden: Tensor,
        positions: Tensor,
        local: _DevicePagedPlan,
        global_: _DevicePagedPlan,
        local_wrapper: Any,
        global_wrapper: Any,
        token_ids: Tensor | None = None,
    ) -> Tensor:
        rotary = self._model.language_model.rotary_emb
        embeddings = {
            kind: rotary(positions, kind, hidden.dtype)
            for kind in set(self.config.layer_types)
        }
        local_layer = 0
        global_layer = 0
        for layer_index, layer in enumerate(self._model.language_model.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            cos, sin = embeddings[layer.self_attn.layer_type]
            query, key, value = layer.self_attn.project_qkv(normalized, cos, sin)
            if layer.self_attn.layer_type == SLIDING_ATTENTION:
                keys = self._cache.local_keys[local_layer]
                values = self._cache.local_values[local_layer]
                append_pages = local.append_pages
                append_offsets = local.append_offsets
                wrapper = local_wrapper
                local_layer += 1
            else:
                keys = self._cache.global_keys[global_layer]
                values = self._cache.global_values[global_layer]
                append_pages = global_.append_pages
                append_offsets = global_.append_offsets
                wrapper = global_wrapper
                global_layer += 1
            keys[append_pages, append_offsets] = key.to(self._cache.dtype)
            values[append_pages, append_offsets] = value.to(self._cache.dtype)
            attended = wrapper.run(query, (keys, values))
            hidden = layer.self_attn.o_proj(attended.reshape(len(hidden), -1))
            hidden = residual + layer.post_attention_layernorm(hidden)

            hidden = self._prefill_feed_forward[layer_index](hidden)

        return self._model.language_model.norm(hidden)

    @staticmethod
    def _feed_forward(layer, hidden: Tensor) -> Tensor:
        residual = hidden
        hidden = layer.pre_feedforward_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + layer.post_feedforward_layernorm(hidden)
        hidden *= layer.layer_scalar
        return hidden

    def _graph_capture_inputs(
        self, shape: GraphShape, vocab_size: int = 262_208
    ) -> dict[str, object]:
        """Synthetic decode rows backed only by graph-reserved parent zero."""

        batch_size, history_width = shape
        rows = tuple(range(batch_size))
        zeros = (0,) * batch_size
        token_ids = (2,) * batch_size if self.speculative_draft_tokens else zeros
        positions = (1,) * batch_size if self.speculative_draft_tokens else zeros
        inputs: dict[str, object] = {
            "token_ids": token_ids,
            "positions": positions,
            **capture_sampling_inputs(batch_size, vocab_size),
        }
        if self.speculative_draft_tokens:
            width = self.speculative_draft_tokens + 1
            inputs.update(
                {
                    "fixed_acceptance": -1,
                    "target_hidden": torch.linspace(
                        -1.0,
                        1.0,
                        self.config.hidden_size,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    .expand(batch_size, -1)
                    .contiguous(),
                    "verify_qo_indptr": tuple(
                        index * width for index in range(batch_size + 1)
                    ),
                }
            )
            stages = (
                ("assistant_", 1, None),
                ("verify_", width + 1, tuple(range(1, width + 1)) * batch_size),
            )
        else:
            stages = (("", 1, tuple(row % PAGE_SIZE for row in rows)),)
        for prefix, last_page_length, append_offsets in stages:
            for name, page_width in (
                ("local", self.local_graph_page_capacity),
                ("global", history_width),
            ):
                name = prefix + name
                inputs.update(
                    {
                        f"{name}_kv_indptr": tuple(range(batch_size + 1)),
                        f"{name}_page_indices": (0,) * (batch_size * page_width),
                        f"{name}_last_page_lengths": (last_page_length,) * batch_size,
                    }
                )
                if append_offsets is not None:
                    inputs[f"{name}_append_pages"] = (0,) * len(append_offsets)
                    inputs[f"{name}_append_offsets"] = append_offsets
        return inputs

    def _before_cuda_graph_capture(self) -> None:
        if self.speculative_draft_tokens:
            self._cache.storage[0].fill_(0.125)

    def _after_cuda_graph_capture(self) -> None:
        self._cache.storage[0].zero_()
        torch.cuda.synchronize(self.device)

    def _plan_graph_attention(
        self,
        wrapper,
        indptr,
        indices,
        last,
        *,
        local: bool,
        qo_indptr=None,
        assistant: bool = False,
        non_blocking: bool = False,
    ) -> None:
        """Use the same FlashInfer attention contract at capture and replay."""

        args = (indptr, indices, last)
        if qo_indptr is not None:
            args = (qo_indptr, *args)
        wrapper.plan(
            *args,
            (
                getattr(
                    getattr(self._assistant, "config", None),
                    "num_attention_heads",
                    self.attention_config.num_attention_heads,
                )
                if assistant
                else self.attention_config.num_attention_heads
            ),
            (
                self.attention_config.num_key_value_heads
                if local
                else self.attention_config.num_global_key_value_heads
            ),
            self.config.head_dim if local else self.config.global_head_dim,
            PAGE_SIZE,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=self._cache.dtype,
            o_data_type=torch.bfloat16,
            sm_scale=1.0,
            window_left=self.sliding_window - 1 if local else -1,
            disable_split_kv=True,
            non_blocking=non_blocking,
            **({"causal": True} if qo_indptr is not None else {}),
        )

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        if self.speculative_draft_tokens:
            return self._new_speculative_graph_step(shape)
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        batch_size, history_width = shape

        def empty(dtype, *dimensions):
            return torch.empty(dimensions, dtype=dtype, device=self.device)

        def attention(width: int, *, local: bool):
            workspace = self._graph_attention_workspaces[0 if local else 1]
            kv_indptr = empty(torch.int32, batch_size + 1)
            page_indices = empty(torch.int32, batch_size * width)
            last_page_lengths = empty(torch.int32, batch_size)
            append_pages = empty(torch.int64, batch_size)
            append_offsets = empty(torch.int64, batch_size)
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=kv_indptr,
                paged_kv_indices_buffer=page_indices,
                paged_kv_last_page_len_buffer=last_page_lengths,
                backend="fa2",
            )
            initial_indptr = torch.arange(
                batch_size + 1, dtype=torch.int32, device=self.device
            )
            initial_indices = torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            )
            initial_last = torch.ones(batch_size, dtype=torch.int32, device=self.device)
            self._plan_graph_attention(
                wrapper,
                initial_indptr,
                initial_indices,
                initial_last,
                local=local,
            )
            plan = _DevicePagedPlan(
                qo_indptr=torch.arange(
                    batch_size + 1, dtype=torch.int32, device=self.device
                ),
                kv_indptr=kv_indptr,
                page_indices=page_indices,
                last_page_lengths=last_page_lengths,
                append_pages=append_pages,
                append_offsets=append_offsets,
            )
            return wrapper, plan

        local_wrapper, local = attention(self.local_graph_page_capacity, local=True)
        global_wrapper, global_ = attention(history_width, local=False)
        self._graph_attention[shape] = (local_wrapper, global_wrapper)
        token_ids = empty(torch.int64, batch_size)
        positions = empty(torch.int64, batch_size)
        output_ids = empty(torch.int64, batch_size)
        sampling = self._sampler.empty(batch_size, self.device)

        def launch() -> None:
            hidden = self._model.embed_tokens(token_ids)
            hidden = self._run_transformer(
                hidden,
                positions,
                local,
                global_,
                local_wrapper,
                global_wrapper,
                token_ids,
            )
            logits = self._model.logits(hidden)
            self._sampler.sample_tensors(logits, sampling, out=output_ids)

        inputs = {
            "token_ids": token_ids,
            "positions": positions,
            "local_kv_indptr": local.kv_indptr,
            "local_page_indices": local.page_indices,
            "local_last_page_lengths": local.last_page_lengths,
            "local_append_pages": local.append_pages,
            "local_append_offsets": local.append_offsets,
            "global_kv_indptr": global_.kv_indptr,
            "global_page_indices": global_.page_indices,
            "global_last_page_lengths": global_.last_page_lengths,
            "global_append_pages": global_.append_pages,
            "global_append_offsets": global_.append_offsets,
        }
        inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(inputs=inputs, output=output_ids, launch=launch)

    def _new_assistant_graph_attention_wrapper(self, *args, **kwargs):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        return BatchDecodeWithPagedKVCacheWrapper(*args, **kwargs)

    def _new_verify_attention_wrapper(self, *args, **kwargs):
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        return BatchPrefillWithPagedKVCacheWrapper(*args, **kwargs)

    def _new_speculative_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        if not hasattr(self, "_graph_acceptance"):
            self._graph_acceptance = {}

        assistant = self._assistant
        if assistant is None:
            raise RuntimeError("speculative graph requires the Gemma 4 assistant")
        batch_size, history_width = shape
        width = self.speculative_draft_tokens + 1

        def empty(dtype, *dimensions):
            return torch.empty(dimensions, dtype=dtype, device=self.device)

        def attention(page_width: int, *, local: bool):
            workspace = self._graph_attention_workspaces[0 if local else 1]
            assistant_kv_indptr = empty(torch.int32, batch_size + 1)
            assistant_page_indices = empty(torch.int32, batch_size * page_width)
            assistant_last = empty(torch.int32, batch_size)
            assistant_wrapper = self._new_assistant_graph_attention_wrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                paged_kv_indptr_buffer=assistant_kv_indptr,
                paged_kv_indices_buffer=assistant_page_indices,
                paged_kv_last_page_len_buffer=assistant_last,
                backend="auto",
            )
            verify_qo_indptr = empty(torch.int32, batch_size + 1)
            verify_kv_indptr = empty(torch.int32, batch_size + 1)
            verify_page_indices = empty(torch.int32, batch_size * page_width)
            verify_last = empty(torch.int32, batch_size)
            verify_append_pages = empty(torch.int64, batch_size * width)
            verify_append_offsets = empty(torch.int64, batch_size * width)
            verify_wrapper = self._new_verify_attention_wrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=verify_qo_indptr,
                paged_kv_indptr_buf=verify_kv_indptr,
                paged_kv_indices_buf=verify_page_indices,
                paged_kv_last_page_len_buf=verify_last,
                backend="auto",
            )
            initial_kv_indptr = torch.arange(
                batch_size + 1, dtype=torch.int32, device=self.device
            )
            initial_indices = torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            )
            initial_assistant_last = torch.ones(
                batch_size, dtype=torch.int32, device=self.device
            )
            initial_verify_qo = torch.arange(
                0,
                (batch_size + 1) * width,
                width,
                dtype=torch.int32,
                device=self.device,
            )
            initial_verify_last = torch.full(
                (batch_size,), width + 1, dtype=torch.int32, device=self.device
            )
            self._plan_graph_attention(
                assistant_wrapper,
                initial_kv_indptr,
                initial_indices,
                initial_assistant_last,
                local=local,
                assistant=True,
            )
            self._plan_graph_attention(
                verify_wrapper,
                initial_kv_indptr,
                initial_indices,
                initial_verify_last,
                local=local,
                qo_indptr=initial_verify_qo,
            )
            assistant_plan = _DevicePagedPlan(
                qo_indptr=torch.arange(
                    batch_size + 1, dtype=torch.int32, device=self.device
                ),
                kv_indptr=assistant_kv_indptr,
                page_indices=assistant_page_indices,
                last_page_lengths=assistant_last,
                append_pages=empty(torch.int64, 0),
                append_offsets=empty(torch.int64, 0),
            )
            verify_plan = _DevicePagedPlan(
                qo_indptr=verify_qo_indptr,
                kv_indptr=verify_kv_indptr,
                page_indices=verify_page_indices,
                last_page_lengths=verify_last,
                append_pages=verify_append_pages,
                append_offsets=verify_append_offsets,
            )
            return assistant_wrapper, assistant_plan, verify_wrapper, verify_plan

        (
            assistant_local_wrapper,
            assistant_local,
            verify_local_wrapper,
            verify_local,
        ) = attention(self.local_graph_page_capacity, local=True)
        (
            assistant_global_wrapper,
            assistant_global,
            verify_global_wrapper,
            verify_global,
        ) = attention(history_width, local=False)
        self._graph_spec_attention[shape] = (
            assistant_local_wrapper,
            assistant_global_wrapper,
            verify_local_wrapper,
            verify_global_wrapper,
        )

        token_ids = empty(torch.int64, batch_size)
        positions = empty(torch.int64, batch_size)
        target_hidden = empty(torch.bfloat16, batch_size, self.config.hidden_size)
        output_ids = empty(
            torch.int64, batch_size, 2 * self.speculative_draft_tokens + 1
        )
        fixed_acceptance = empty(torch.int64)
        verify_hidden_output = empty(
            torch.bfloat16, batch_size, width, self.config.hidden_size
        )
        self._graph_verify_hidden[shape] = verify_hidden_output
        sampling = self._sampler.empty(batch_size, self.device)
        shared_local = (self._cache.local_keys[-1], self._cache.local_values[-1])
        shared_global = (self._cache.global_keys[-1], self._cache.global_values[-1])

        phase_timing = self._new_phase_timing("draft", "target_verify", "decode")

        def launch() -> None:
            if phase_timing is not None:
                phase_timing.mark(0)
            last_ids = token_ids
            last_hidden = target_hidden
            draft_columns = []
            for _ in range(self.speculative_draft_tokens):
                token_embedding = self._model.embed_tokens(last_ids)
                inputs_embeds = torch.cat((token_embedding, last_hidden), dim=-1)
                draft_logits, last_hidden = assistant.forward_step(
                    inputs_embeds,
                    positions,
                    local_wrapper=assistant_local_wrapper,
                    global_wrapper=assistant_global_wrapper,
                    local_kv=shared_local,
                    global_kv=shared_global,
                )
                last_ids = draft_logits.argmax(dim=-1)
                draft_columns.append(last_ids)
            drafts = torch.stack(draft_columns, dim=1)
            if phase_timing is not None:
                phase_timing.mark(1)
            verify_ids = torch.cat((token_ids[:, None], drafts), dim=1).reshape(-1)
            verify_positions = (
                positions[:, None]
                + torch.arange(width, dtype=torch.int64, device=self.device)[None]
            ).reshape(-1)
            hidden = self._model.embed_tokens(verify_ids)
            hidden = self._run_transformer(
                hidden,
                verify_positions,
                verify_local,
                verify_global,
                verify_local_wrapper,
                verify_global_wrapper,
                verify_ids,
            )
            verify_hidden_output.copy_(
                hidden.view(batch_size, width, self.config.hidden_size)
            )
            verify_logits = self._model.logits(hidden).view(
                batch_size, width, self.config.vocab_size
            )
            if phase_timing is not None:
                phase_timing.mark(2)
            output_ids[:, : self.speculative_draft_tokens].copy_(drafts)
            output_ids[:, self.speculative_draft_tokens :].copy_(
                graph_sample_block(self._sampler, verify_logits, sampling)
            )
            self._graph_acceptance[shape] = DeviceAcceptance.from_samples(
                drafts,
                output_ids[:, self.speculative_draft_tokens :],
                positions,
                fixed_acceptance=fixed_acceptance,
            )

            if phase_timing is not None:
                phase_timing.mark(3)

        inputs = {
            "token_ids": token_ids,
            "positions": positions,
            "target_hidden": target_hidden,
            "fixed_acceptance": fixed_acceptance,
            "assistant_local_kv_indptr": assistant_local.kv_indptr,
            "assistant_local_page_indices": assistant_local.page_indices,
            "assistant_local_last_page_lengths": assistant_local.last_page_lengths,
            "assistant_global_kv_indptr": assistant_global.kv_indptr,
            "assistant_global_page_indices": assistant_global.page_indices,
            "assistant_global_last_page_lengths": assistant_global.last_page_lengths,
            "verify_qo_indptr": verify_local.qo_indptr,
            "verify_local_kv_indptr": verify_local.kv_indptr,
            "verify_local_page_indices": verify_local.page_indices,
            "verify_local_last_page_lengths": verify_local.last_page_lengths,
            "verify_local_append_pages": verify_local.append_pages,
            "verify_local_append_offsets": verify_local.append_offsets,
            "verify_global_kv_indptr": verify_global.kv_indptr,
            "verify_global_page_indices": verify_global.page_indices,
            "verify_global_last_page_lengths": verify_global.last_page_lengths,
            "verify_global_append_pages": verify_global.append_pages,
            "verify_global_append_offsets": verify_global.append_offsets,
        }
        inputs.update(self._sampler.graph_inputs(sampling))
        # At non-power-of-two row counts the captured NVFP4 path can cross an
        # argmax tie differently from eager warmup. Require repeatable graph
        # output here; end-to-end graph/eager correctness is covered with real
        # prompts instead of this deliberately synthetic null-page state.
        return CudaGraphStep(
            inputs=inputs,
            output=output_ids,
            launch=launch,
            phase_timing=phase_timing,
            validate_eager_output=False,
        )

    def _prepare_graph_attention(
        self, shape: GraphShape, inputs: dict[str, object]
    ) -> None:
        if self.speculative_draft_tokens:
            wrappers = self._graph_spec_attention[shape]
            stages = (
                ("assistant_", wrappers[:2], None),
                ("verify_", wrappers[2:], inputs["verify_qo_indptr"]),
            )
        else:
            stages = (("", self._graph_attention[shape], None),)

        def tensor(values):
            return torch.as_tensor(values, dtype=torch.int32, device=self.device)

        for prefix, wrappers, queries in stages:
            for local, wrapper in zip((True, False), wrappers, strict=True):
                name = prefix + ("local" if local else "global")
                indptr = inputs[f"{name}_kv_indptr"]
                page_count = int(np.asarray(indptr)[-1])
                self._plan_graph_attention(
                    wrapper,
                    tensor(indptr),
                    tensor(inputs[f"{name}_page_indices"])[:page_count],
                    tensor(inputs[f"{name}_last_page_lengths"]),
                    local=local,
                    qo_indptr=None if queries is None else tensor(queries),
                    assistant=prefix == "assistant_",
                )

    @staticmethod
    def _graph_page_inputs(
        name: str,
        plan: PagedAttentionPlan,
        *,
        active_rows: int,
        graph_rows: int,
        width: int,
    ) -> dict[str, object]:
        row_widths = np.diff(plan.kv_indptr)
        if len(row_widths) != active_rows or np.any(row_widths > width):
            raise GraphInputError(f"{name} page table exceeds CUDA graph shape")
        indices = np.zeros(graph_rows * width, dtype=np.int32)
        indptr = np.zeros(graph_rows + 1, dtype=np.int32)
        cursor = 0
        for row in range(graph_rows):
            if row < active_rows:
                start, end = map(int, plan.kv_indptr[row : row + 2])
                count = end - start
                indices[cursor : cursor + count] = plan.page_indices[start:end]
            else:
                count = 1
            cursor += count
            indptr[row + 1] = cursor
        last_page_lengths = np.ones(graph_rows, dtype=np.int32)
        last_page_lengths[:active_rows] = plan.last_page_lengths
        append_pages = np.zeros(graph_rows, dtype=np.int64)
        append_offsets = np.arange(graph_rows, dtype=np.int64) % PAGE_SIZE
        append_pages[:active_rows] = plan.append_slots // PAGE_SIZE
        append_offsets[:active_rows] = plan.append_slots % PAGE_SIZE
        return {
            f"{name}_kv_indptr": indptr,
            f"{name}_page_indices": indices,
            f"{name}_last_page_lengths": last_page_lengths,
            f"{name}_append_pages": append_pages,
            f"{name}_append_offsets": append_offsets,
        }

    @staticmethod
    def _spec_graph_page_inputs(
        name: str,
        assistant: PagedAttentionPlan,
        verify: PagedAttentionPlan,
        *,
        active_rows: int,
        graph_rows: int,
        page_width: int,
        query_width: int,
    ) -> dict[str, object]:
        def pack_pages(
            plan: PagedAttentionPlan, prefix: str, *, padding_last: int = 1
        ) -> dict[str, object]:
            row_widths = np.diff(plan.kv_indptr)
            if len(row_widths) != active_rows or np.any(row_widths > page_width):
                raise GraphInputError(f"{prefix} page table exceeds CUDA graph shape")
            indices = np.zeros(graph_rows * page_width, dtype=np.int32)
            indptr = np.zeros(graph_rows + 1, dtype=np.int32)
            cursor = 0
            for row in range(graph_rows):
                if row < active_rows:
                    start, end = map(int, plan.kv_indptr[row : row + 2])
                    count = end - start
                    indices[cursor : cursor + count] = plan.page_indices[start:end]
                else:
                    count = 1
                cursor += count
                indptr[row + 1] = cursor
            last = np.full(graph_rows, padding_last, dtype=np.int32)
            last[:active_rows] = plan.last_page_lengths
            return {
                f"{prefix}_kv_indptr": indptr,
                f"{prefix}_page_indices": indices,
                f"{prefix}_last_page_lengths": last,
            }

        inputs = pack_pages(assistant, f"assistant_{name}")
        inputs.update(
            pack_pages(verify, f"verify_{name}", padding_last=query_width + 1)
        )
        append_pages = np.zeros(graph_rows * query_width, dtype=np.int64)
        append_offsets = np.zeros(graph_rows * query_width, dtype=np.int64)
        for row in range(graph_rows):
            start = row * query_width
            end = start + query_width
            if row < active_rows:
                query_start, query_end = map(int, verify.qo_indptr[row : row + 2])
                slots = verify.append_slots[query_start:query_end]
                if len(slots) != query_width:
                    raise GraphInputError("verify append rows do not match graph width")
                append_pages[start:end] = slots // PAGE_SIZE
                append_offsets[start:end] = slots % PAGE_SIZE
            else:
                append_offsets[start:end] = np.arange(1, query_width + 1)
        inputs.update(
            {
                f"verify_{name}_append_pages": append_pages,
                f"verify_{name}_append_offsets": append_offsets,
            }
        )
        return inputs

    def _graph_forward(
        self, native: BatchPlan, batch: Gemma4BatchPlan
    ) -> ForwardOutput | None:
        if native.needs_logprobs:
            self._cuda_graph_fallback("decode", "logprobs")
            return None
        if not self._cuda_graph_mode_enabled("decode"):
            return None
        active_rows = native.num_sequences
        history_width = int(np.diff(batch.global_.kv_indptr).max())
        local_width = int(np.diff(batch.local.kv_indptr).max())
        if local_width > self.local_graph_page_capacity:
            raise GraphInputError("local page table exceeds CUDA graph shape")
        shape = self._select_cuda_graph(
            "decode",
            match=lambda candidate: (
                candidate[0] >= active_rows and candidate[1] >= history_width
            ),
        )
        if shape is None:
            return None
        graph_rows, graph_history_width = shape
        padding_rows = graph_rows - active_rows
        inputs: dict[str, object] = {
            "token_ids": np.pad(batch.token_ids, (0, padding_rows)),
            "positions": np.pad(batch.positions, (0, padding_rows)),
        }
        inputs.update(
            self._graph_page_inputs(
                "local",
                batch.local,
                active_rows=active_rows,
                graph_rows=graph_rows,
                width=self.local_graph_page_capacity,
            )
        )
        inputs.update(
            self._graph_page_inputs(
                "global",
                batch.global_,
                active_rows=active_rows,
                graph_rows=graph_rows,
                width=graph_history_width,
            )
        )
        inputs.update(
            pad_sampling_inputs(
                self._sampler, native, self.config.vocab_size, padding_rows
            )
        )
        self._prepare_graph_attention(shape, inputs)
        return token_output(self._run_cuda_graph("decode", shape, inputs)[:active_rows])

    def _speculative_graph_forward(
        self, native: BatchPlan, batch: Gemma4BatchPlan
    ) -> ForwardOutput | None:
        if native.needs_logprobs:
            self._cuda_graph_fallback("decode", "logprobs")
            return None
        if not self._cuda_graph_mode_enabled("decode"):
            return None
        sequence_ids = [int(value) for value in native.sequence_ids]
        if any(sequence_id not in self._target_hidden for sequence_id in sequence_ids):
            self._cuda_graph_fallback("decode", "target_hidden")
            return None
        active_rows = native.num_sequences
        history_width = int(np.diff(batch.global_.kv_indptr).max())
        local_width = int(np.diff(batch.local.kv_indptr).max())
        if local_width > self.local_graph_page_capacity:
            raise GraphInputError("local page table exceeds CUDA graph shape")
        shape = self._select_cuda_graph(
            "decode",
            match=lambda candidate: (
                candidate[0] >= active_rows and candidate[1] >= history_width
            ),
        )
        if shape is None:
            return None
        graph_rows, graph_history_width = shape
        padding_rows = graph_rows - active_rows
        width = self.speculative_draft_tokens + 1
        assistant = gemma4_assistant_context(
            batch, local_window_tokens=self.sliding_window
        )
        verify = expand_gemma4_decode_batch(
            batch,
            np.zeros((active_rows, self.speculative_draft_tokens), dtype=np.int64),
        )
        hidden = self._gather_target_hidden(sequence_ids, graph_rows)
        fixed = self._speculation.validated_benchmark_accepted_drafts
        inputs: dict[str, object] = {
            "token_ids": np.pad(batch.token_ids, (0, padding_rows)),
            "positions": np.pad(batch.positions, (0, padding_rows), constant_values=1),
            "target_hidden": hidden,
            "fixed_acceptance": -1 if fixed is None else fixed,
            "verify_qo_indptr": np.arange(
                0, (graph_rows + 1) * width, width, dtype=np.int32
            ),
        }
        inputs.update(
            self._spec_graph_page_inputs(
                "local",
                assistant.local,
                verify.local,
                active_rows=active_rows,
                graph_rows=graph_rows,
                page_width=self.local_graph_page_capacity,
                query_width=width,
            )
        )
        inputs.update(
            self._spec_graph_page_inputs(
                "global",
                assistant.global_,
                verify.global_,
                active_rows=active_rows,
                graph_rows=graph_rows,
                page_width=graph_history_width,
                query_width=width,
            )
        )
        inputs.update(
            pad_sampling_inputs(
                self._sampler, native, self.config.vocab_size, padding_rows
            )
        )
        self._prepare_graph_attention(shape, inputs)
        self._run_cuda_graph("decode", shape, inputs)
        accepted = self._graph_acceptance[shape]
        self._store_accepted_hidden(
            sequence_ids,
            accepted.accepted_drafts[:active_rows],
            self._graph_verify_hidden[shape][:active_rows],
        )
        return DeviceCompletion.from_acceptance(
            accepted,
            rows=active_rows,
            stats=self._speculation,
            device=self.device,
        ).result()

    def _eager_transformer(self, native: BatchPlan, batch: Gemma4BatchPlan) -> Tensor:
        token_ids = torch.as_tensor(
            batch.token_ids, dtype=torch.int64, device=self.device
        )
        positions = torch.as_tensor(
            batch.positions, dtype=torch.int64, device=self.device
        )
        hidden = self._model.embed_tokens(token_ids)
        features = self._merge_media_embeddings(native, batch, hidden)
        local_mask = self._local_custom_mask(batch, features)
        local = self._device_plan(batch.local)
        global_ = self._device_plan(batch.global_)
        local_wrapper, global_wrapper = self._plan_attention(
            batch, local, global_, local_mask
        )
        with (
            self._prefill_graphs()
            if batch.phase == BatchPhase.PREFILL
            else nullcontext()
        ):
            return self._run_transformer(
                hidden,
                positions,
                local,
                global_,
                local_wrapper,
                global_wrapper,
                token_ids,
            )

    def _hidden_slot_indices(self, sequence_ids, rows: int, *, write: bool) -> Tensor:
        sequence_ids = tuple(map(int, sequence_ids))
        if rows < len(sequence_ids):
            raise ValueError("hidden-state buffer has fewer rows than requests")
        if write:
            missing = tuple(
                sequence
                for sequence in sequence_ids
                if sequence not in self._target_hidden
            )
            if len(set(sequence_ids)) != len(sequence_ids):
                raise ValueError("hidden-state requests must be distinct")
            if self._hidden_bank is None:
                capacity = max(self.max_num_seqs, len(missing), 1)
                self._hidden_bank = torch.empty(
                    (capacity + 1, self.config.hidden_size),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                # Slot zero is immutable padding and never belongs to a request.
                self._hidden_bank[0].zero_()
                self._hidden_free_slots = list(range(capacity, 0, -1))
            if len(missing) > len(self._hidden_free_slots):
                previous = self._hidden_bank
                old_capacity = len(previous) - 1
                capacity = max(
                    old_capacity * 2,
                    old_capacity + len(missing) - len(self._hidden_free_slots),
                )
                self._hidden_bank = previous.new_empty(
                    (capacity + 1, previous.shape[1])
                )
                self._hidden_bank[: len(previous)].copy_(previous)
                self._hidden_free_slots.extend(range(capacity, old_capacity, -1))
            if missing:
                for sequence in missing:
                    self._target_hidden[sequence] = self._hidden_free_slots.pop()
                self._hidden_slot_generation += 1
        key = (sequence_ids, rows, self._hidden_slot_generation)
        cached = self._hidden_index_cache.get(write)
        if cached is not None and cached[0] == key:
            return cached[1]
        slots = [self._target_hidden[sequence] for sequence in sequence_ids]
        slots.extend([0] * (rows - len(slots)))
        indices = torch.tensor(slots, dtype=torch.int64, device=self.device)
        self._hidden_index_cache[write] = (key, indices)
        return indices

    def _set_target_hidden(self, sequence_ids, hidden: Tensor) -> None:
        """Save an entire accepted batch with one GPU scatter.

        Slots survive batch permutation and admission. Graph output buffers never
        alias this bank; all accesses follow the runner's serialized CUDA stream.
        """
        if hidden.shape != (len(sequence_ids), self.config.hidden_size):
            raise ValueError("accepted hidden states must match the request batch")
        indices = self._hidden_slot_indices(sequence_ids, len(sequence_ids), write=True)
        assert self._hidden_bank is not None
        self._hidden_bank.index_copy_(0, indices, hidden)

    def _gather_target_hidden(self, sequence_ids, graph_rows: int) -> Tensor:
        """Gather saved rows into reusable input storage, padding with slot zero.

        A later gather is queued after the previous graph's input copy on the same
        stream. Device feedback separately owns in-flight continuation state.
        """
        indices = self._hidden_slot_indices(sequence_ids, graph_rows, write=False)
        assert self._hidden_bank is not None
        output = self._hidden_input_buffers.get(graph_rows)
        if output is None:
            output = self._hidden_bank.new_empty((graph_rows, self.config.hidden_size))
            self._hidden_input_buffers[graph_rows] = output
        torch.index_select(self._hidden_bank, 0, indices, out=output)
        return output

    def _store_last_hidden(
        self, native: BatchPlan, batch: Gemma4BatchPlan, hidden: Tensor
    ) -> None:
        indices = torch.as_tensor(
            batch.last_token_indices, dtype=torch.int64, device=self.device
        )
        self._set_target_hidden(native.sequence_ids, hidden.index_select(0, indices))

    def _sample_target(self, logits: Tensor, plan: BatchPlan, step: int) -> Tensor:
        return self._sampler(
            logits,
            plan,
            offset_delta=step * SAMPLING_OFFSET_STRIDE,
        )

    def _single_target_decode(
        self, native: BatchPlan, batch: Gemma4BatchPlan
    ) -> ForwardOutput:
        rows = native.num_sequences
        target = expand_gemma4_decode_batch(batch, np.empty((rows, 0), dtype=np.int64))
        hidden = self._eager_transformer(native, target)
        self._store_last_hidden(native, target, hidden)
        logits = self._model.logits(hidden[target.last_token_indices])
        return format_output(logits, self._sample_target(logits, native, 0), native)

    def _store_accepted_hidden(self, sequence_ids, accepted_drafts, verify_hidden):
        indices = accepted_drafts[:, None, None].expand(-1, 1, verify_hidden.shape[-1])
        selected = verify_hidden.gather(1, indices).squeeze(1)
        self._set_target_hidden(sequence_ids, selected)

    def _speculative_decode(
        self, native: BatchPlan, batch: Gemma4BatchPlan
    ) -> ForwardOutput:
        assistant = self._assistant
        if assistant is None:
            raise RuntimeError("speculative decode requires the Gemma 4 assistant")
        sequence_ids = [int(value) for value in native.sequence_ids]
        if any(sequence_id not in self._target_hidden for sequence_id in sequence_ids):
            # A complete prefix-cache hit has target K/V but no sequence-local
            # final hidden state. Materialize it with one ordinary target step;
            # all subsequent steps can draft normally.
            return self._single_target_decode(native, batch)

        context = gemma4_assistant_context(
            batch, local_window_tokens=self.sliding_window
        )
        local = self._device_plan(context.local)
        global_ = self._device_plan(context.global_)
        local_wrapper, global_wrapper = self._plan_assistant_attention(
            context, local, global_, custom_mask=None
        )
        anchors = torch.as_tensor(
            batch.token_ids, dtype=torch.int64, device=self.device
        )
        positions = torch.as_tensor(
            batch.positions, dtype=torch.int64, device=self.device
        )
        target_hidden = self._gather_target_hidden(sequence_ids, len(sequence_ids))
        phase_timing = self._eager_phase_timing("draft", "target_verify", "decode")
        if phase_timing is not None:
            phase_timing.mark(0)
        last_ids = anchors
        last_hidden = target_hidden
        draft_columns = []
        shared_local = (self._cache.local_keys[-1], self._cache.local_values[-1])
        shared_global = (self._cache.global_keys[-1], self._cache.global_values[-1])
        for _ in range(self.speculative_draft_tokens):
            token_embedding = self._model.embed_tokens(last_ids)
            inputs_embeds = torch.cat((token_embedding, last_hidden), dim=-1)
            draft_logits, last_hidden = assistant.forward_step(
                inputs_embeds,
                positions,
                local_wrapper=local_wrapper,
                global_wrapper=global_wrapper,
                local_kv=shared_local,
                global_kv=shared_global,
            )
            last_ids = draft_logits.argmax(dim=-1)
            draft_columns.append(last_ids)
        drafts = torch.stack(draft_columns, dim=1)

        if phase_timing is not None:
            phase_timing.mark(1)
        verify = expand_gemma4_decode_batch(batch, drafts.cpu().numpy())
        verify_hidden = self._eager_transformer(native, verify)
        width = self.speculative_draft_tokens + 1
        verify_hidden_rows = verify_hidden.view(
            native.num_sequences, width, self.config.hidden_size
        )
        verify_logits = self._model.logits(verify_hidden).view(
            native.num_sequences, width, self.config.vocab_size
        )
        if phase_timing is not None:
            phase_timing.mark(2)
        targets = self._sampler(verify_logits, native)
        accepted = DeviceAcceptance.from_samples(
            drafts,
            targets,
            positions,
            fixed_acceptance=self._speculation.validated_benchmark_accepted_drafts,
        )
        self._store_accepted_hidden(
            sequence_ids, accepted.accepted_drafts, verify_hidden_rows
        )
        if phase_timing is not None:
            phase_timing.mark(3)
        return DeviceCompletion.from_acceptance(
            accepted,
            rows=native.num_sequences,
            stats=self._speculation,
            device=self.device,
            plan=native,
            logits=verify_logits,
        ).result()

    def _build_batch_plan(self, plan):
        return build_gemma4_batch_plan(
            plan,
            self.cache_layout,
            prefill_chunk_size=self.prefill_chunk_size,
            decode_lookahead=self.speculative_draft_tokens
            if plan.phase == BatchPhase.DECODE
            else 0,
        )

    @torch.inference_mode()
    @observe_execution
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self._check_cuda_graphs(plan.phase)
        batch = self._build_batch_plan(plan)
        with self._cuda_failure:
            if batch.phase == BatchPhase.DECODE:
                if self.speculative_draft_tokens:
                    graph_output = self._speculative_graph_forward(plan, batch)
                    if graph_output is not None:
                        return graph_output
                    return self._speculative_decode(plan, batch)
                graph_output = self._graph_forward(plan, batch)
                if graph_output is not None:
                    return graph_output
            hidden = self._eager_transformer(plan, batch)
            self._store_last_hidden(plan, batch, hidden)
            last = torch.as_tensor(
                batch.last_token_indices, dtype=torch.int64, device=self.device
            )
            logits = self._model.logits(hidden[last])
            scores = prompt_scores(plan, hidden, self._model.logits)
            return format_output(
                logits,
                self._sample_target(logits, plan, 0),
                plan,
                prompt_logprobs=scores,
            )


__all__ = [
    "Gemma4CudaRunner",
    "plan_gemma4_kv_capacity",
]
