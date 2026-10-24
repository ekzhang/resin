"""Qwen3.5 CUDA runner with optional bundled-MTP decoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
from resin.backend.cuda.parallel import (
    TensorParallelContext,
    replicated_cache_capacity,
)
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
            **capture_sampling_inputs(batch_size, vocab_size, unique_seeds=True),
        }

    def _after_cuda_graph_capture(self) -> None:
        self._cache.reset_null()
        torch.cuda.synchronize(self.device)

    def _decode_graph_capture_inputs(self, shape: GraphShape) -> dict[str, object]:
        inputs = self._capture_inputs(shape, self.vocab_size)
        inputs.update(self._cache.graph_shape_capture_inputs(*shape))
        for index, call in enumerate(self._graph_attention_for(shape).calls):
            inputs[f"attention.{index}.base_pages"] = (0,) * len(
                call.prepared.base_pages
            )
        return inputs

    def _graph_specs(self) -> tuple[GraphAttentionSpec, ...]:
        target = self.config.full_layer_count
        if not self.speculative_tokens:
            return (GraphAttentionSpec(0, target, 1, 1),)
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

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        batch_size, graph_seq_len = shape
        attention = self._graph_attention_for(shape)
        cache = self._graph_device_caches[shape]
        token_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
        positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        rope_deltas = torch.empty(batch_size, dtype=torch.long, device=self.device)
        sampling = self._sampler.empty(batch_size, self.device)
        output_width = 2 * self.speculative_tokens + 2 if self.speculative_tokens else 1
        output_ids = torch.empty(
            (batch_size, output_width), dtype=torch.long, device=self.device
        )

        phase_timing = (
            self._new_phase_timing("decode", "draft", "target_verify", "decode")
            if self.speculative_tokens
            else None
        )

        def ordinary_launch() -> None:
            attention.reset()
            recurrent, convolution, _ = self._initial_state(cache, positions)
            hidden, recurrent, convolution = self._target_forward(
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
            self._store_state(cache, positions, recurrent, convolution, hidden[:, -1])
            output_ids[:, 0].copy_(
                graph_sample(
                    self._sampler, self._model.logits(hidden[:, -1]), sampling, 0
                )
            )

        def speculative_launch() -> None:
            if phase_timing is not None:
                phase_timing.mark(0)
            attention.reset()
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

            active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            selected_recurrent = recurrent
            selected_convolution = convolution
            selected_hidden = anchor_hidden[:, -1]
            verify_hidden = []
            target_columns = []
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
                selected_recurrent = torch.where(
                    active[:, None, None, None, None],
                    work_recurrent,
                    selected_recurrent,
                )
                selected_convolution = torch.where(
                    active[:, None, None, None],
                    work_convolution,
                    selected_convolution,
                )
                selected_hidden = torch.where(
                    active[:, None], current_hidden, selected_hidden
                )
                verify_hidden.append(current_hidden)
                target = graph_sample(
                    self._sampler,
                    self._model.logits(current_hidden),
                    sampling,
                    step + 1,
                )
                target_columns.append(target)
                if step < self.speculative_tokens:
                    active = active & (drafts[:, step] == target)
            if phase_timing is not None:
                phase_timing.mark(3)
            targets = torch.stack(target_columns, dim=1)
            matches = drafts == targets[:, : self.speculative_tokens]
            accepted = torch.cumprod(matches.to(torch.int64), dim=1).sum(1)
            processed = accepted + 1

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
                torch.stack(verify_hidden, dim=1),
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

        launch = speculative_launch if self.speculative_tokens else ordinary_launch
        inputs = {
            "token_ids": token_ids,
            "positions": positions,
            "rope_deltas": rope_deltas,
        }
        inputs.update(self._cache.graph_device_inputs(cache))
        inputs.update(attention.graph_inputs())
        inputs.update(self._sampler.graph_inputs(sampling))

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
        self, plan: BatchPlan, cache_plan: PagedCachePlan
    ) -> ForwardOutput | None:
        if plan.needs_logprobs:
            self._cuda_graph_fallback("decode", "logprobs")
            return None
        batch_size = plan.num_sequences
        required_length = int(np.max(plan.positions)) + self.speculative_tokens + 2
        if not self.speculative_tokens:
            required_length = int(np.max(plan.positions)) + 1
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
            "rope_deltas": np.pad(self._host_rope_deltas(plan), (0, padding)),
        }
        inputs.update(
            self._cache.graph_plan_inputs(self._graph_device_caches[shape], cache_plan)
        )
        inputs.update(
            self._graph_attention[shape].plan(cache_plan, np.asarray(plan.positions))
        )
        inputs.update(
            pad_sampling_inputs(self._sampler, plan, self.vocab_size, padding)
        )
        output = self._run_cuda_graph("decode", shape, inputs)[:batch_size]
        if not self.speculative_tokens:
            return token_output(output[:, 0])
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
            logits.float(),
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
        recurrent: torch.Tensor,
        convolution: torch.Tensor,
        *,
        attention_runtime: object | None = None,
        max_sequence_length: int | None = None,
        rotary_positions: torch.Tensor | None = None,
        input_embeddings: torch.Tensor | None = None,
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
        contexts = torch.as_tensor(
            plan.context_lengths, dtype=torch.long, device=self.device
        )
        recurrent, convolution, _ = self._initial_state(cache, contexts)
        hidden, recurrent, convolution = self._target_forward(
            cache,
            anchors[:, None],
            positions[:, None],
            recurrent,
            convolution,
            rotary_positions=self._decode_rotary_positions(plan, positions),
        )
        self._store_state(cache, positions, recurrent, convolution, hidden[:, -1])
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
        targets = torch.stack(
            [
                self._sample(verify_logits[:, step].contiguous(), plan, step + 1)
                for step in range(self.speculative_tokens + 1)
            ],
            dim=1,
        )
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
            plan, self.cache_layout, decode_lookahead=lookahead
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

    def release(self, sequence_ids: list[int]) -> None:
        """Release request-local image embeddings after Engine retirement."""

        for sequence_id in sequence_ids:
            self._media_features.pop(sequence_id, None)


__all__ = ["Qwen3_5CudaRunner"]
