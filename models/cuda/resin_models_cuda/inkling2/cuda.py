"""Inkling 2 Engine runner: attention TP/DP, expert parallelism, and bundled DSpark."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import torch

from resin import BatchPhase, BatchPlan, RunnerConfig
from resin.backend.capacity import plan_cache_capacity
from resin.backend.cuda.completion import DeviceAcceptance, DeviceCompletion
from resin.backend.cuda.graphs import (
    CudaGraphMode,
    CudaGraphResult,
    CudaGraphStep,
    resolve_decode_graph_shapes,
)
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.parallel import (
    FlashInferAllReduce,
    TensorParallelContext,
    replicated_cache_capacity,
)
from resin.backend.cuda.runner import (
    capture_sampling_inputs,
    graph_sample,
    validate_hybrid_plan,
)
from resin.backend.cuda.scoring import format_output, prompt_scores, score_completion
from resin.backend.cuda.slab import graph_scratch_parent_count
from resin.backend.cuda.staging import TensorStager
from resin.backend.cuda.state import execute_state
from resin.backend.cuda.vision_cache import VisionFeatureCache
from resin.backend.plans import PagedCachePlan
from resin.backend.sampling import SAMPLING_OFFSET_STRIDE, PromptScores
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelismCapabilities
from resin.backend.torch import use_default_dtype
from resin.models.inkling2.config import Inkling2Config
from resin.models.inkling2.dspark import Inkling2DSparkConfig
from resin.models.vision import (
    image_grid,
    media_identity,
    media_key,
    packed_images,
    span_slices,
)
from resin.recipe import Media
from resin_models_cuda.inkling2.cache import (
    PREFILL_CHUNK_SIZE,
    CacheArena,
    CacheGeometry,
)
from resin_models_cuda.inkling2.checkpoint import (
    load_precision,
    read_config,
    resolve_target_path,
)
from resin_models_cuda.inkling2.draft_weights import load_draft_weights
from resin_models_cuda.inkling2.dspark import DSparkAttention, DSparkDraft, DSparkLayer
from resin_models_cuda.inkling2.kernels import select_convolution_state
from resin_models_cuda.inkling2.model import DecoderLayer, Inkling2Target
from resin_models_cuda.inkling2.vision import Inkling2VisionEncoder
from resin_models_cuda.inkling2.vision_weights import load_vision_weights
from resin_models_cuda.inkling2.weights import load_target_weights
from resin_models_cuda.paged_decode import PagedDecodeRunner


class Inkling2CudaRunner(PagedDecodeRunner[CacheArena]):
    _decode_model = "Inkling 2"
    parallelism_capabilities = ParallelismCapabilities(
        attn_tp=True, attn_dp=True, ep=True, expert_tp=True
    )

    def __init__(
        self,
        model: str | Path,
        *,
        device,
        max_seq_len: int,
        max_batch_size: int = 8,
        parallel_groups=None,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        speculative_draft_tokens: int = 7,
        expert_backend: str = "triton",
        fast_allreduce: bool = False,
        fused_attention_prologue: bool = False,
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
        cuda_graph_decode_shapes=None,
        gpu_memory_utilization: float = 0.85,
        cache_parent_capacity: int | None = None,
        profile_graph_phases: bool = False,
    ):
        if expert_backend not in ("triton", "cute-dsl"):
            raise ValueError(f"unsupported Inkling 2 expert backend: {expert_backend}")
        self.expert_backend = expert_backend
        self.expert_kernel_backends = {
            "nvfp4": "sglang-cute-dsl" if expert_backend == "cute-dsl" else "triton",
            "mxfp8": "triton",
            "bf16": "triton",
        }
        self.benchmark_metadata = {
            "expert_backend": self.expert_backend,
            "expert_kernel_backends": self.expert_kernel_backends,
        }
        for name, value in (
            ("max_seq_len", max_seq_len),
            ("max_batch_size", max_batch_size),
            ("prefill_chunk_size", prefill_chunk_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if prefill_chunk_size % 128:
            raise ValueError("Inkling 2 prefill chunks must align to 128-token pages")
        if type(
            speculative_draft_tokens
        ) is not int or speculative_draft_tokens not in (0, 7):
            raise ValueError(
                "Inkling 2 supports target-only or seven-token DSpark decoding"
            )
        if parallel_groups is not None:
            self.parallelism_capabilities.validate(
                parallel_groups.topology, model="Inkling 2"
            )
        super().__init__(
            device=device,
            label="Inkling 2",
            kv_cache_dtype="bfloat16",
            cuda_graph_prefill=cuda_graph_prefill,
            cuda_graph_decode=cuda_graph_decode,
        )
        torch.cuda.set_device(self.device)
        if torch.cuda.get_device_capability(self.device)[0] < 10:
            raise RuntimeError("Inkling 2 packed experts require a Blackwell GPU")
        self.parallel_groups = parallel_groups
        parallel = (
            TensorParallelContext.from_group(parallel_groups.attn_tp)
            if parallel_groups
            else TensorParallelContext()
        )
        self.max_seq_len, self.max_batch_size = max_seq_len, max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
        self.speculative_draft_tokens = speculative_draft_tokens
        self.physical_seq_len = max_seq_len + speculative_draft_tokens + 1
        self._speculation = SpeculativeStats(speculative_draft_tokens)
        self._media_features = {}
        self._vision_feature_cache = VisionFeatureCache()
        if profile_graph_phases:
            self.configure_metrics(enabled=True)
        self.model_path = resolve_target_path(model)
        self.raw_config = read_config(self.model_path)
        self.config = Inkling2Config.from_hugging_face(self.raw_config)
        self.draft_config = Inkling2DSparkConfig.from_hugging_face(self.raw_config)
        if max_seq_len > min(
            self.config.max_position_embeddings,
            self.draft_config.max_position_embeddings,
        ):
            raise ValueError("max_seq_len exceeds the checkpoint context length")
        self.vocab_size = self.config.unpadded_vocab_size
        self.eos_token_id = self.config.eos_token_id
        with (
            torch.inference_mode(),
            use_default_dtype(torch.bfloat16),
            torch.device("meta"),
        ):
            self._model = Inkling2Target(
                self.config,
                self.draft_config.target_layer_ids,
                tensor_parallel=parallel,
                parallel_groups=parallel_groups,
                precision=load_precision(self.model_path),
                expert_backend=expert_backend,
            )
            self._draft = (
                DSparkDraft(
                    self.draft_config,
                    self.config,
                    self._model,
                    self.physical_seq_len,
                    tensor_parallel=parallel,
                )
                if speculative_draft_tokens
                else None
            )
            self._vision = Inkling2VisionEncoder(
                self.raw_config["vision_config"], self.config.hidden_size
            )
        self._model.to_empty(device=self.device).eval()
        self.weight_load = {"target": load_target_weights(self._model, self.model_path)}
        self._model.prepare()
        self._vision.to_empty(device=self.device).eval()
        self.weight_load["vision"] = load_vision_weights(self.model_path, self._vision)
        if self._draft is not None:
            self._draft.to_empty(device=self.device).eval()
            self.weight_load["draft"] = load_draft_weights(self.model_path, self._draft)
        self._allreduce = FlashInferAllReduce(
            parallel,
            max_token_num=max_batch_size * (speculative_draft_tokens + 1),
            hidden_dim=self.config.hidden_size,
            dtype=torch.bfloat16,
            enabled=fast_allreduce,
        )
        layers = cast(Sequence[DecoderLayer], self._model.layers)
        self._expert_allreduce = (
            FlashInferAllReduce(
                parallel,
                max_token_num=max_batch_size * (speculative_draft_tokens + 1),
                hidden_dim=self.config.hidden_size,
                dtype=torch.float32,
                enabled=fast_allreduce,
            )
            if layers[0].mlp.experts.replicated_ep is not None
            else None
        )
        for layer in layers:
            layer.attn.allreduce = self._allreduce.sum
            layer.attn.fused_prologue = fused_attention_prologue
            if self._expert_allreduce is not None:
                layer.mlp.experts.fast_sum = self._expert_allreduce.sum
        if self._draft is not None:
            for module in self._draft.modules():
                if isinstance(module, (DSparkAttention, DSparkLayer)):
                    module.allreduce = self._allreduce.sum
        if fused_attention_prologue:
            from resin_models_cuda.inkling2.fa4_bias import warm_native_bias

            for kv_heads in sorted({layer.attn.kv_heads for layer in layers}):
                warm_native_bias(kv_heads, self.device)
        self.benchmark_metadata["allreduce_backend"] = self._allreduce.backend
        self.benchmark_metadata["attention_prologue"] = (
            "fused" if fused_attention_prologue else "separate"
        )
        self.benchmark_metadata["target_attention"] = (
            "native-relative-bias"
            if fused_attention_prologue and layers[0].attn.kv_heads in (2, 4, 8)
            else "generic-fa4"
        )
        self.benchmark_metadata["expert_allreduce_backend"] = (
            self._expert_allreduce.backend
            if self._expert_allreduce is not None
            else "not-applicable"
        )
        self._configure_parallel_graphs(self._model, parallel_groups)
        geometry = CacheGeometry.from_config(
            self.config,
            self.draft_config,
            attn_tp=parallel.size,
            prefill_chunk_size=prefill_chunk_size,
            draft_enabled=bool(speculative_draft_tokens),
        )
        self.cache_geometry = geometry
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        reserve = 1 + graph_scratch_parent_count(geometry.packings, max_batch_size)
        capacity = plan_cache_capacity(
            total,
            free,
            gpu_memory_utilization,
            geometry.bytes_per_slab,
            reserved_parent_count=reserve,
            overhead_bytes=4 * 1024**3,
        )
        parent_capacity = capacity.parent_capacity
        if cache_parent_capacity is not None:
            if type(cache_parent_capacity) is not int or cache_parent_capacity <= 0:
                raise ValueError("cache_parent_capacity must be positive")
            parent_capacity = min(parent_capacity, cache_parent_capacity)
        parent_capacity = replicated_cache_capacity(
            parent_capacity,
            None if parallel_groups is None else parallel_groups.replica,
        )
        capacity = replace(
            capacity,
            parent_capacity=parent_capacity,
            allocation_bytes=(parent_capacity + reserve) * geometry.bytes_per_slab,
        )
        self.cache_capacity_plan = capacity
        self.cache_layout = geometry.layout(parent_capacity)
        self._cache = CacheArena(
            self.device, self.cache_layout, geometry, max_batch_size=max_batch_size
        )
        self._configure_cache_binding(self._cache.binding)
        self.runner_config = RunnerConfig(
            max_batch_tokens=prefill_chunk_size,
            prefill_chunk_size=prefill_chunk_size,
            continuation_prefill=True,
            max_accepted_per_step=speculative_draft_tokens + 1,
            cache_layout=self.cache_layout,
        )
        # Graphs replay serially. Share verification history across all buckets
        # instead of pinning a separate multi-megabyte state per row per graph.
        self._checkpoint_buffers = (
            tuple(
                torch.empty(shape, dtype=torch.bfloat16, device=self.device)
                for shape in (
                    (
                        max_batch_size,
                        geometry.target_layers,
                        geometry.state_length,
                        geometry.state_width,
                    ),
                    (
                        max_batch_size,
                        speculative_draft_tokens + 1,
                        geometry.target_layers,
                        geometry.state_width,
                    ),
                )
            )
            if self._draft is not None
            else None
        )
        self._checkpoint_inputs = None
        shapes = resolve_decode_graph_shapes(
            cuda_graph_decode_shapes,
            policy=self.cuda_graph_decode,
            width=max_seq_len,
            speculative=False,
            max_batch_size=max_batch_size,
        )
        if any(
            batch > max_batch_size or width > max_seq_len for batch, width in shapes
        ):
            raise ValueError("Inkling 2 graph shape exceeds configured capacity")
        shapes = tuple((*shape, algorithm) for shape in shapes for algorithm in (1, 2))
        self._configure_cuda_graphs(
            {
                "decode": CudaGraphMode(
                    shapes,
                    self._new_decode_graph_step,
                    self._decode_graph_capture_inputs,
                )
            }
        )
        layers = cast(Sequence[DecoderLayer], self._model.layers)
        callbacks = self._configure_prefill_graphs(
            tuple(layer.feed_forward for layer in layers),
            hidden_size=self.config.hidden_size,
            max_tokens=prefill_chunk_size,
            pad_up_to=0,
        )
        for layer, callback in zip(layers, callbacks, strict=True):
            layer.prefill_ffn = callback

    def speculative_stats(self):
        return self._speculation.snapshot(include_draft_tokens=True)

    def _initial_state(self, cache, positions, *, out=None):
        state = cache.state_at((positions - 1).clamp_min(0)[:, None])[:, 0]
        return torch.where(
            (positions == 0)[:, None, None, None],
            torch.zeros((), dtype=state.dtype, device=state.device),
            state,
            out=out,
        )

    def _decode(
        self,
        ids,
        positions,
        cache,
        max_sequence_length,
        sample,
        *,
        valid_rows=None,
        phase_timing=None,
    ):
        self._checkpoint_inputs = None
        if phase_timing is not None:
            phase_timing.mark(0)
        buffers = self._checkpoint_buffers
        initial = self._initial_state(
            cache, positions, out=None if buffers is None else buffers[0][: len(ids)]
        )
        width = self.speculative_draft_tokens + 1
        if self._draft is None:
            drafts = ids[:, None][:, :0]
        else:
            features = cache.draft_features_at((positions - 1).clamp_min(0)[:, None])[
                :, 0
            ]
            drafts = self._draft.propose(
                ids,
                positions,
                cache,
                max_sequence_length,
                sample,
                f_rnn=features,
            )
        verify_ids = torch.cat((ids[:, None], drafts), dim=1)
        if phase_timing is not None:
            phase_timing.mark(1)
        verify_positions = positions[:, None] + torch.arange(width, device=self.device)
        conv_inputs = None if buffers is None else buffers[1][: len(ids)]
        with (
            self._parallel_graph_capacity(
                ids.shape[0] * width, valid_rows=valid_rows * width
            )
            if valid_rows is not None
            else nullcontext()
        ):
            logits, taps, final_state, _ = self._model(
                verify_ids,
                verify_positions,
                cache,
                initial,
                positions + width,
                max_sequence_length,
                full_logits=True,
                state_inputs_out=conv_inputs,
            )
        if phase_timing is not None:
            phase_timing.mark(2)
        targets = sample(logits, None)
        acceptance = DeviceAcceptance.from_samples(drafts, targets, positions)
        accepted = acceptance.counts.long()
        if self._draft is not None:
            assert conv_inputs is not None
            final_state = select_convolution_state(initial, conv_inputs, accepted)
            features = self._draft.inject_target(taps, verify_positions, cache)
            self._checkpoint_inputs = initial, conv_inputs, features, positions
            selected = features[
                torch.arange(ids.shape[0], device=self.device), accepted - 1
            ]
            cache.store_draft_features(
                (positions + accepted - 1)[:, None], selected[:, None]
            )
        cache.store_state((positions + accepted - 1)[:, None], final_state[:, None])
        if phase_timing is not None:
            phase_timing.mark(3)
        return acceptance, logits

    def _copy_state_checkpoints(self, storage, addresses, *, candidates, ends):
        if self._checkpoint_inputs is None:
            from resin.backend.cuda.state_triton import copy_state

            return copy_state(storage, addresses, candidates=candidates, ends=ends)
        from resin_models_cuda.inkling2.kernels import copy_state_checkpoints

        return copy_state_checkpoints(
            storage, addresses, candidates, ends, *self._checkpoint_inputs
        )

    def _new_decode_graph_step(self, shape):
        size, width, algorithm = shape
        cache = self._cache.graph_cache(size, width + self.speculative_draft_tokens)
        ids = torch.empty((size, 1), device=self.device, dtype=torch.long)
        positions = torch.empty(size, device=self.device, dtype=torch.long)
        sampling = self._sampler.empty(size, self.device)
        output = CudaGraphResult()
        valid = (
            torch.empty((), device=self.device, dtype=torch.int32)
            if self.parallel_groups
            else None
        )

        phase_timing = self._new_phase_timing("draft", "target_verify", "decode")

        def launch():
            def sample(logits, step):
                if algorithm == 1:
                    return logits.argmax(dim=-1)
                return (
                    self._sampler.sample_block(logits, sampling)
                    if step is None
                    else graph_sample(self._sampler, logits, sampling, step)
                )

            acceptance, logits = self._decode(
                ids[:, 0],
                positions,
                cache,
                width + self.speculative_draft_tokens,
                sample,
                valid_rows=valid,
                phase_timing=phase_timing,
            )
            output.publish(
                (acceptance, logits, self._checkpoint_inputs),
                parity=acceptance.parity_tensors,
            )
            self._checkpoint_inputs = None

        inputs = {
            "token_ids": ids,
            "positions": positions,
            **self._cache.graph_device_inputs(cache),
            **self._sampler.graph_inputs(sampling),
        }
        if valid is not None:
            inputs["ep_valid_rows"] = valid
        return CudaGraphStep(
            inputs=inputs,
            output=output,
            launch=launch,
            prepare=self._cache.reset_null,
            phase_timing=phase_timing,
            resources=cache,
        )

    def _validate_plan(self, plan: BatchPlan) -> tuple[tuple[int, int], ...]:
        return validate_hybrid_plan(
            plan,
            label="Inkling 2",
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            decode_lookahead=0,
        )

    def _prepare(self, plan):
        self._validate_plan(plan)
        return PagedCachePlan.from_plan(
            plan, self.cache_layout, decode_lookahead=self.speculative_draft_tokens
        )

    def _sample(self, logits, plan, step=None):
        return self._sampler(
            logits,
            plan,
            offset_delta=0 if step is None else step * SAMPLING_OFFSET_STRIDE,
        )

    def _prefill(self, plan, cache_plan, *, context=None):
        self._checkpoint_inputs = None
        if plan is not None:
            staging = TensorStager(self.device)
            cache = self._cache.tensorize(cache_plan, staging)
            ids = staging.device("token_ids", plan.token_ids, torch.long)
            positions = staging.device("positions", plan.positions, torch.long)
        count = plan.num_sequences if context is None else context.max_num_sequences
        logits, scores = [], []
        for row in range(count):
            if plan is None or row >= plan.num_sequences:
                if self._prefill_graph_active:
                    self._empty_prefill_regions()
                else:
                    self._model.forward_experts_empty()
                continue
            start, end = map(int, plan.token_indptr[row : row + 2])
            row_cache, row_ids, row_positions = (
                cache.row(row),
                ids[start:end][None],
                positions[start:end][None],
            )
            state = self._initial_state(row_cache, row_positions[:, 0])
            embeddings = self._media_embeddings(plan, row, row_ids)
            hidden, taps, final_state, _ = self._model(
                row_ids,
                row_positions,
                row_cache,
                state,
                row_positions[:, -1] + 1,
                int(plan.context_lengths[row]) + end - start,
                full_logits=True,
                return_hidden=True,
                prefill=True,
                input_embeddings=embeddings,
            )
            last_positions = row_positions[:, -1:]
            row_cache.store_state(last_positions, final_state[:, None])
            if self._draft is not None:
                features = self._draft.inject_target(taps, row_positions, row_cache)
                row_cache.store_draft_features(last_positions, features[:, -1:])
            logits.append(self._model.logits(hidden[:, -1]))
            if plan.needs_prompt_logprobs:
                scores.append(
                    prompt_scores(
                        plan, hidden[0], self._model.logits, token_start=start
                    )
                )
        if plan is None:
            return None
        values = torch.cat(logits)
        return format_output(
            values,
            self._sample(values, plan),
            plan,
            prompt_logprobs=PromptScores.concat(scores),
        )

    def _features_for(self, sequence_id: int, media: tuple[Media, ...]) -> torch.Tensor:
        existing = self._media_features.get(sequence_id)
        if existing is not None and existing[0] != media_identity(media):
            raise ValueError("Inkling 2 image payload changed during a request")
        if existing is None:
            side = self._vision.patch_size * self._vision.merge_size
            image_grid(media, side)
            cached = self._vision_feature_cache.get(media_key(media))
            if cached is None:
                pixels, grid = packed_images(media, side, channels_last=True)
                pixels = torch.as_tensor(
                    pixels, device=self.device, dtype=torch.bfloat16
                )
                features = self._vision(pixels, torch.from_numpy(grid))
                if features.shape != (
                    sum(item.span[1] for item in media),
                    self.config.hidden_size,
                ):
                    raise ValueError(
                        "Inkling 2 vision features do not match image placeholders"
                    )
                cached = self._vision_feature_cache.put(media_key(media), (features,))
            (features,) = cached
            assert features is not None
            existing = (media_identity(media), features)
            self._media_features[sequence_id] = existing
        return existing[1]

    def _media_embeddings(
        self, plan: BatchPlan, row: int, ids: torch.Tensor
    ) -> torch.Tensor | None:
        media = plan.media[row]
        if media is None:
            return None
        start = int(plan.context_lengths[row])
        slices = tuple(
            span_slices((item.span for item in media), start, start + ids.shape[1])
        )
        if not slices:
            return None
        features = self._features_for(int(plan.sequence_ids[row]), media)
        embeddings = self._model.embed_norm(self._model.embed_tokens(ids))
        for destination, source in slices:
            embeddings[0, destination].copy_(features[source])
        return embeddings

    def _eager(self, plan, cache_plan):
        if plan.phase == BatchPhase.PREFILL:
            return self._prefill(plan, cache_plan)
        staging = TensorStager(self.device)
        cache = self._cache.tensorize(cache_plan, staging)
        ids = staging.device("token_ids", plan.token_ids, torch.long)
        positions = staging.device("positions", plan.positions, torch.long)
        acceptance, logits = self._decode(
            ids,
            positions,
            cache,
            int(np.max(plan.positions)) + self.speculative_draft_tokens + 1,
            lambda values, step: self._sample(values, plan, step),
            phase_timing=self._eager_phase_timing("draft", "target_verify", "decode"),
        )
        return DeviceCompletion.from_acceptance(
            acceptance,
            rows=plan.num_sequences,
            device=self.device,
            stats=self._speculation if self.speculative_draft_tokens else None,
            plan=plan,
            scores=score_completion(logits, acceptance, plan),
        ).result()

    def _after_cuda_graph_capture(self):
        self._cache.reset_null()
        torch.cuda.synchronize(self.device)

    def _decode_graph_capture_inputs(self, shape):
        inputs = {
            "token_ids": (0,) * shape[0],
            "positions": (1,) * shape[0],
            **capture_sampling_inputs(shape[0], self.vocab_size, unique_seeds=True),
            **self._cache.graph_shape_capture_inputs(
                shape[0], shape[1] + self.speculative_draft_tokens
            ),
        }
        if self.parallel_groups is not None:
            inputs["ep_valid_rows"] = shape[0]
        return inputs

    def _sampling_graph_algorithm(self, plan):
        if plan is None:
            return 0
        values = self._sampler.host_inputs(plan, self.vocab_size)
        return 2 if any(values["sampling_stochastic_mask"]) else 1

    def _decode_graph_shape(self, plan, cache_plan):
        algorithm = self._sampling_graph_algorithm(plan)
        return self._select_cuda_graph(
            "decode",
            match=lambda shape: (
                shape[0] >= plan.num_sequences
                and shape[2] >= algorithm
                and shape[1] > int(np.max(plan.positions))
                and self._cache.graph_can_fit_plan(
                    cache_plan, (shape[0], shape[1] + self.speculative_draft_tokens)
                )
            ),
        )

    def _decode_graph_result(self, shape, inputs):
        acceptance, logits, self._checkpoint_inputs = self._run_cuda_graph(
            "decode", shape, inputs
        )
        return acceptance, logits

    def _enqueue_graph_output(self, plan, shape, inputs, feedback):
        if self._sampling_graph_algorithm(plan) > shape[2]:
            raise ValueError("sampling policy changed inside a greedy continuation")
        return super()._enqueue_graph_output(plan, shape, inputs, feedback)

    def release(self, sequence_ids):
        self._continuation.drain()
        for sequence_id in sequence_ids:
            self._media_features.pop(int(sequence_id), None)

    def _distributed_decode_shape(self, plan, context, prepared, *, enqueue, feedback):
        local_shape = (
            feedback.continuation.shape
            if feedback is not None
            else self._decode_graph_shape(plan, prepared.cache_plan)
            if plan is not None
            else None
        )
        return self._select_distributed_cuda_graph(
            "decode",
            context,
            required_shape=local_shape
            if local_shape is not None
            else (
                0 if plan is None else plan.num_sequences,
                0 if plan is None else int(np.max(plan.positions)) + 1,
                self._sampling_graph_algorithm(plan),
            ),
            eligible=plan is None or local_shape is not None,
            validation_error=(
                "sampling policy changed inside a greedy continuation"
                if feedback is not None
                and self._sampling_graph_algorithm(plan)
                > feedback.continuation.shape[2]
                else None
            ),
        )

    @observe_execution
    @execute_state
    def forward_distributed(self, plan, context, *, prepared=None):
        self._continuation.drain()
        prepared = (
            self.validate_distributed(plan, context) if prepared is None else prepared
        )
        with (
            self._cuda_failure,
            torch.inference_mode(),
            use_default_dtype(torch.bfloat16),
            torch.device(self.device),
        ):
            if context.phase == BatchPhase.DECODE:
                used, result = self._distributed_graph(plan, context, prepared)
                if used:
                    return result
                if plan is None:
                    self._model.forward_experts_empty()
                    return None
                return self._eager(plan, prepared.cache_plan)
            with self._distributed_prefill_graphs(context, per_sequence=True):
                return self._prefill(plan, prepared.cache_plan, context=context)
