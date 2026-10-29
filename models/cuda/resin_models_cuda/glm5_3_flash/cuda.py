"""GLM-5.3-Flash NVFP4 Engine runner with attention-DP and expert parallelism."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from resin import BatchPhase, RunnerConfig
from resin.backend.capacity import plan_cache_capacity
from resin.backend.cuda.completion import (
    CudaContinuation,
    DeviceAcceptance,
    DeviceCompletion,
    GraphDeviceFeedback,
)
from resin.backend.cuda.graphs import (
    CudaGraphMode,
    CudaGraphStep,
    resolve_decode_graph_shapes,
)
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.runner import (
    CudaRunner,
    capture_sampling_inputs,
    graph_sample_block,
    pad_sampling_inputs,
    validate_hybrid_plan,
)
from resin.backend.cuda.slab import graph_scratch_parent_count
from resin.backend.cuda.staging import TensorStager
from resin.backend.cuda.state import execute_state
from resin.backend.cuda.vision_cache import VisionFeatureCache
from resin.backend.plans import PagedCachePlan
from resin.backend.sampling import format_output, prompt_scores
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelismCapabilities
from resin.backend.torch import use_default_dtype
from resin_models_cuda.glm5_3_flash.cache import CacheArena, CacheGeometry
from resin_models_cuda.glm5_3_flash.checkpoint import load_model, resolve_model_path


@dataclass(frozen=True)
class PreparedBatch:
    cache_plan: PagedCachePlan | None


class Glm53FlashCudaRunner(CudaRunner):
    parallelism_capabilities = ParallelismCapabilities(
        attn_dp=True, ep=True, expert_tp=True
    )

    def __init__(
        self,
        model: str | Path,
        *,
        device,
        max_seq_len: int,
        max_batch_size: int = 8,
        parallel_groups=None,
        prefill_chunk_size: int = 8192,
        speculative_draft_tokens: int = 0,
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
        cuda_graph_decode_shapes=None,
        gpu_memory_utilization: float = 0.85,
        cache_parent_capacity: int | None = None,
        profile_graph_phases: bool = False,
    ):
        for name, value in (
            ("max_seq_len", max_seq_len),
            ("max_batch_size", max_batch_size),
            ("prefill_chunk_size", prefill_chunk_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if prefill_chunk_size % 64:
            raise ValueError("GLM prefill chunks must align to 64-token pages")
        if (
            type(speculative_draft_tokens) is not int
            or not 0 <= speculative_draft_tokens <= 7
        ):
            raise ValueError("GLM native MTP supports zero to seven draft tokens")
        if parallel_groups is not None:
            self.parallelism_capabilities.validate(
                parallel_groups.topology, model="GLM-5.3-Flash"
            )
            mesh = parallel_groups.topology
            if mesh.attn_dp != mesh.world_size:
                raise ValueError("GLM requires attention-DP=world size")
        super().__init__(
            device=device,
            label="GLM-5.3-Flash",
            kv_cache_dtype="fp8",
            cuda_graph_prefill=cuda_graph_prefill,
            cuda_graph_decode=cuda_graph_decode,
        )
        self.parallel_groups = parallel_groups
        self.max_seq_len, self.max_batch_size = max_seq_len, max_batch_size
        # Compatibility floor, not the capacity of every executing batch.
        # The selected graph bucket supplies the target projection row count.
        self.target_projection_rows = 64
        self.prefill_chunk_size, self.speculative_draft_tokens = (
            prefill_chunk_size,
            speculative_draft_tokens,
        )
        self._media_features = {}
        self._vision_feature_cache = VisionFeatureCache()
        self._speculation = SpeculativeStats(speculative_draft_tokens)
        if profile_graph_phases:
            self.configure_metrics(enabled=True)
        self.model_path = resolve_model_path(model)
        with (
            torch.inference_mode(),
            use_default_dtype(torch.bfloat16),
            torch.device(self.device),
        ):
            self._model, self.config, self.raw_config = load_model(
                self.model_path,
                device=self.device,
                parallel_groups=parallel_groups,
                include_mtp=bool(speculative_draft_tokens),
                max_tokens=max(
                    prefill_chunk_size, max_batch_size * (speculative_draft_tokens + 1)
                ),
            )
        self.vocab_size = self.config.vocab_size
        self._configure_parallel_graphs(self._model, parallel_groups)
        geometry = CacheGeometry.from_config(
            self.config,
            state_tokens=prefill_chunk_size,
            include_mtp=bool(speculative_draft_tokens),
        )
        self.cache_geometry = geometry
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        reserve = 1 + graph_scratch_parent_count(geometry.packings, max_batch_size)
        capacity = plan_cache_capacity(
            total,
            free,
            gpu_memory_utilization,
            geometry.slab_bytes,
            reserved_parent_count=reserve,
            overhead_bytes=4 * 1024**3,
        )
        parent_capacity = capacity.parent_capacity
        if cache_parent_capacity is not None:
            if type(cache_parent_capacity) is not int or cache_parent_capacity <= 0:
                raise ValueError("cache_parent_capacity must be positive")
            parent_capacity = min(parent_capacity, cache_parent_capacity)
            capacity = replace(
                capacity,
                parent_capacity=parent_capacity,
                allocation_bytes=(parent_capacity + reserve) * geometry.slab_bytes,
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
        self._graph_device_caches, self._graph_logits = {}, {}
        self._graph_acceptance = {}
        self._graph_inputs = {}
        self._continuation = CudaContinuation()
        shapes = resolve_decode_graph_shapes(
            cuda_graph_decode_shapes,
            policy=cuda_graph_decode,
            width=max_seq_len,
            # K0/MTP share physical batch buckets and therefore target
            # projection arithmetic independently of the native draft width.
            speculative=False,
            max_batch_size=max_batch_size,
        )
        if any(
            batch > max_batch_size or width > max_seq_len for batch, width in shapes
        ):
            raise ValueError("GLM graph shape exceeds configured capacity")
        self._projection_batch_buckets = tuple(sorted({batch for batch, _ in shapes}))
        self._configure_cuda_graphs(
            {
                "decode": CudaGraphMode(
                    shapes,
                    self._new_decode_graph_step,
                    self._decode_graph_capture_inputs,
                )
            }
        )
        layers = tuple(self._model.layers)
        if self._model.mtp is not None:
            layers += (self._model.mtp,)
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
        if self.speculative_draft_tokens:
            inputs["fixed_acceptance"] = -1
        return inputs

    def _projection_rows_for_bucket(self, batch_bucket):
        """Keep one arithmetic profile per physical batch bucket for K0..K7."""
        return max(self.target_projection_rows, batch_bucket * 8)

    def _eager_projection_rows(self, num_sequences):
        # A graph may miss on context capacity or be disabled. Use only the
        # existing CPU batch catalog, without a graph vote or device read.
        # Distributed callers supply their already-agreed maximum request count.
        # Missing/disabled catalogs use bounded power-of-two batch profiles.
        bucket = next(
            (size for size in self._projection_batch_buckets if size >= num_sequences),
            1 << max(0, num_sequences - 1).bit_length(),
        )
        return self._projection_rows_for_bucket(bucket)

    def _new_decode_graph_step(self, shape):
        size, width = shape
        projection_rows = self._projection_rows_for_bucket(size)
        # Shape widths describe the public context bound. Verification may
        # compute K further rows which the Engine clips at request completion.
        cache = self._cache.graph_cache(size, width + self.speculative_draft_tokens)
        self._graph_device_caches[shape] = cache
        ids = torch.empty((size, 1), device=self.device, dtype=torch.long)
        positions = torch.empty(size, device=self.device, dtype=torch.long)
        sequence_indices = torch.arange(size, device=self.device, dtype=torch.long)
        indptr = torch.arange(size + 1, device=self.device, dtype=torch.int32)
        sampling = self._sampler.empty(size, self.device)
        draft_tokens = self.speculative_draft_tokens
        output = torch.empty(
            (size, draft_tokens + 1) if draft_tokens else (size,),
            device=self.device,
            dtype=torch.long,
        )
        logits_buffer = torch.empty(
            (size, draft_tokens + 1, self.vocab_size)
            if draft_tokens
            else (size, self.vocab_size),
            device=self.device,
            dtype=torch.float32,
        )
        self._graph_logits[shape] = logits_buffer
        valid = (
            torch.empty((), device=self.device, dtype=torch.int32)
            if self.parallel_groups is not None
            else None
        )

        fixed_acceptance = (
            torch.empty((), dtype=torch.long, device=self.device)
            if draft_tokens
            else None
        )
        phase_timing = (
            self._new_phase_timing("draft", "target_verify", "decode")
            if draft_tokens
            else None
        )

        def launch():
            with (
                use_default_dtype(torch.bfloat16),
                torch.device(self.device),
            ):
                if draft_tokens:
                    from resin_models_cuda.glm5_3_flash.speculative import (
                        speculative_step,
                    )

                    def sample(logits):
                        return graph_sample_block(self._sampler, logits, sampling)

                    def capacity(width):
                        return self._parallel_graph_capacity(
                            size * width,
                            valid_rows=None if valid is None else valid * width,
                        )

                    acceptance, logits = speculative_step(
                        self._model,
                        cache,
                        ids.flatten(),
                        positions,
                        draft_tokens,
                        sample,
                        capacity=capacity,
                        fixed_acceptance=fixed_acceptance,
                        phase_timing=phase_timing,
                        projection_rows=projection_rows,
                    )
                    output.copy_(acceptance.tokens)
                    logits_buffer.copy_(logits)
                else:
                    with self._parallel_graph_capacity(size, valid_rows=valid):
                        batch = cache.batch(
                            positions, sequence_indices, indptr, positions
                        )
                        batch.is_decode = True
                        batch.projection_rows = projection_rows
                        hidden = self._model.forward_hidden(
                            ids.flatten(), positions, batch
                        )
                        batch.store(hidden)
                        logits = self._model.logits(
                            hidden, projection_rows=projection_rows
                        ).float()
                        logits_buffer.copy_(logits)
                        self._sampler.sample_tensors(logits, sampling, out=output)
                    acceptance = DeviceAcceptance.from_samples(
                        output[:, None][:, :0], output[:, None], positions
                    )
                self._graph_acceptance[shape] = acceptance

        inputs = {
            "token_ids": ids,
            "positions": positions,
            **self._cache.graph_device_inputs(cache),
            **self._sampler.graph_inputs(sampling),
        }
        if valid is not None:
            inputs["ep_valid_rows"] = valid
        if fixed_acceptance is not None:
            inputs["fixed_acceptance"] = fixed_acceptance
        self._graph_inputs[shape] = inputs
        return CudaGraphStep(
            inputs=inputs,
            output=output,
            launch=launch,
            prepare=self._cache.reset_null,
            phase_timing=phase_timing,
        )

    def _validate_plan(self, plan):
        rows = validate_hybrid_plan(
            plan,
            label="GLM-5.3-Flash",
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            # The public limit bounds committed tokens, not speculative work.
            # Engine leases and graph page tables reserve the verification tail.
            decode_lookahead=0,
        )
        from resin.recipe import Glm53FlashMedia

        if any(
            value is not None and not isinstance(value, Glm53FlashMedia)
            for value in plan.media
        ):
            raise TypeError("GLM requires its native prepared image payload")
        return rows

    def _prepare(self, plan):
        self._validate_plan(plan)
        return PagedCachePlan.from_plan(
            plan,
            self.cache_layout,
            # Validate the Engine's physical lease, including native MTP repair
            # writes. This is separate from the public committed-token limit.
            decode_lookahead=(
                self.runner_config.max_accepted_per_step
                - 1
                + self.cache_layout.decode_write_extra_tokens
            ),
        )

    def _decode_graph_shape(self, plan, cache_plan):
        return self._select_cuda_graph(
            "decode",
            match=lambda shape: (
                shape[0] >= plan.num_sequences
                and shape[1] > int(np.max(plan.positions))
                and self._cache.graph_can_fit_plan(
                    cache_plan, (shape[0], shape[1] + self.speculative_draft_tokens)
                )
            ),
        )

    def _graph_forward(
        self, plan, cache_plan, *, shape=None, enqueue=False, feedback=None
    ):
        batch_size = plan.num_sequences
        if shape is None:
            shape = self._decode_graph_shape(plan, cache_plan)
        if shape is None:
            return None
        padding = shape[0] - batch_size
        inputs = {
            "token_ids": np.pad(plan.token_ids, (0, padding)),
            "positions": np.pad(plan.positions, (0, padding), constant_values=1),
            **self._cache.graph_plan_inputs(
                self._graph_device_caches[shape], cache_plan
            ),
            **pad_sampling_inputs(self._sampler, plan, self.vocab_size, padding),
        }
        if self.parallel_groups is not None:
            inputs["ep_valid_rows"] = batch_size
        if self.speculative_draft_tokens:
            fixed = self._speculation.validated_benchmark_accepted_drafts
            inputs["fixed_acceptance"] = -1 if fixed is None else fixed
        if enqueue:
            return self._enqueue_graph_output(plan, shape, inputs, feedback)
        self._run_cuda_graph("decode", shape, inputs)
        completion = DeviceCompletion.from_acceptance(
            self._graph_acceptance[shape],
            rows=batch_size,
            device=self.device,
            stats=self._speculation if self.speculative_draft_tokens else None,
            plan=plan,
            logits=self._graph_logits[shape] if plan.needs_logprobs else None,
        )
        return completion.result()

    def _enqueue_graph_output(self, plan, shape, inputs, feedback):
        ids = () if plan is None else tuple(map(int, plan.sequence_ids))
        continuation = (
            feedback.continuation
            if feedback is not None
            else self._continuation.begin(ids, shape, self._graph_inputs[shape])
        )
        if feedback is not None:
            inputs.update(feedback.inputs())
        self._run_cuda_graph("decode", shape, inputs)
        acceptance = self._graph_acceptance[shape]
        next_inputs = acceptance.feedback()
        completion = DeviceCompletion.from_acceptance(
            acceptance,
            feedback=GraphDeviceFeedback(
                next_inputs.token_ids, next_inputs.positions, continuation
            ),
            rows=0 if plan is None else plan.num_sequences,
            device=self.device,
            stats=self._speculation
            if self.speculative_draft_tokens and plan is not None
            else None,
            plan=plan,
            logits=self._graph_logits[shape]
            if plan is not None and plan.needs_logprobs
            else None,
        )
        maximum = self.runner_config.max_accepted_per_step
        writes = maximum + self.cache_layout.decode_write_extra_tokens
        capacity = (
            shape[1]
            + self.speculative_draft_tokens
            + self.cache_layout.decode_write_extra_tokens
        )
        if plan is not None and np.any(
            np.asarray(plan.positions, dtype=np.int64) + maximum + writes > capacity
        ):
            completion.feedback = None
            self._continuation.drain(continuation)
        return completion

    def _eager(self, plan, cache_plan, *, decode_batch_size=None):
        projection_rows = 0
        if plan.phase == BatchPhase.DECODE:
            batch_size = (
                plan.num_sequences if decode_batch_size is None else decode_batch_size
            )
            if batch_size < plan.num_sequences:
                raise ValueError("decode projection bucket cannot omit local requests")
            projection_rows = self._eager_projection_rows(batch_size)
        staging = TensorStager(self.device)
        cache = self._cache.tensorize(cache_plan, staging)
        ids = staging.device("token_ids", plan.token_ids, torch.long)
        positions = staging.device("positions", plan.positions, torch.long)
        if plan.phase == BatchPhase.DECODE and self.speculative_draft_tokens:
            from resin_models_cuda.glm5_3_flash.speculative import speculative_step

            fixed = self._speculation.validated_benchmark_accepted_drafts
            acceptance, logits = speculative_step(
                self._model,
                cache,
                ids,
                positions,
                self.speculative_draft_tokens,
                lambda values: self._sampler(values, plan),
                fixed_acceptance=torch.tensor(
                    -1 if fixed is None else fixed, device=self.device
                ),
                phase_timing=self._eager_phase_timing(
                    "draft", "target_verify", "decode"
                ),
                projection_rows=projection_rows,
            )
            return DeviceCompletion.from_acceptance(
                acceptance,
                rows=plan.num_sequences,
                device=self.device,
                stats=self._speculation,
                plan=plan,
                logits=logits,
            ).result()
        contexts = staging.device("context_lengths", plan.context_lengths, torch.int32)
        indptr = staging.device("indptr", plan.token_indptr, torch.int32)
        sequence_indices = staging.device(
            "sequence_indices",
            np.repeat(
                np.arange(plan.num_sequences, dtype=np.int64),
                np.diff(plan.token_indptr),
            ),
            torch.long,
        )
        cpu_indptr = torch.from_numpy(
            np.asarray(plan.token_indptr, dtype=np.int32).copy()
        )
        batch = cache.batch(positions, sequence_indices, indptr, contexts, cpu_indptr)
        batch.is_decode = plan.phase == BatchPhase.DECODE
        batch.projection_rows = projection_rows
        embeddings = (
            self._media_embeddings(plan, ids)
            if plan.phase == BatchPhase.PREFILL
            else None
        )
        hidden = self._model.forward_hidden(ids, positions, batch, hidden=embeddings)
        if self.speculative_draft_tokens:
            from resin_models_cuda.glm5_3_flash.speculative import prime_mtp

            prime_mtp(
                self._model,
                batch,
                ids,
                hidden,
                plan.positions,
                input_embeddings=embeddings,
            )
        last = (indptr[1:] - 1).long()
        batch.store(hidden[last])
        logits = self._model.logits(
            hidden[last], projection_rows=batch.projection_rows
        ).float()
        scores = (
            prompt_scores(plan, hidden, self._model.logits)
            if plan.needs_prompt_logprobs
            else None
        )
        output = self._sampler(logits, plan)
        return format_output(logits, output, plan, prompt_logprobs=scores)

    def _media_embeddings(self, plan, token_ids):
        if all(value is None for value in plan.media):
            return None
        from resin_models_cuda.glm5_3_flash.vision import replace_image_embeddings

        embeddings = torch.nn.functional.embedding(token_ids, self._model.embedding)
        for row, media in enumerate(plan.media):
            if media is None:
                continue
            sequence_id = int(plan.sequence_ids[row])
            stored = self._media_features.get(sequence_id)
            if stored is not None and stored[0] != media.digest:
                raise ValueError("GLM image payload changed during a request")
            if stored is None:
                cached = self._vision_feature_cache.get(media.digest)
                if cached is None:
                    vision = self._model.vision
                    if vision is None:
                        raise ValueError("GLM checkpoint has no vision encoder")
                    pixels = torch.as_tensor(
                        np.array(media.pixel_values),
                        device=self.device,
                        dtype=torch.bfloat16,
                    )
                    features = vision.encode(pixels, media.image_grid_thw.tolist())
                    if features.shape != (
                        len(media.image_positions),
                        embeddings.shape[-1],
                    ):
                        raise ValueError(
                            "GLM vision features do not match image placeholders"
                        )
                    cached = self._vision_feature_cache.put(media.digest, (features,))
                (features,) = cached
                assert features is not None
                stored = media.digest, features
                self._media_features[sequence_id] = stored
            start, end = int(plan.token_indptr[row]), int(plan.token_indptr[row + 1])
            replace_image_embeddings(
                embeddings[start:end],
                stored[1],
                media.image_positions,
                int(plan.context_lengths[row]),
            )
        return embeddings

    def release(self, sequence_ids):
        self._continuation.drain()
        for sequence_id in sequence_ids:
            self._media_features.pop(int(sequence_id), None)

    @observe_execution
    @execute_state
    def forward(self, plan):
        self._check_cuda_graphs(plan.phase)
        self._continuation.drain()
        cache_plan = self._prepare(plan)
        with (
            self._cuda_failure,
            torch.inference_mode(),
            use_default_dtype(torch.bfloat16),
            torch.device(self.device),
        ):
            if plan.phase == BatchPhase.DECODE:
                result = self._graph_forward(plan, cache_plan)
                if result is not None:
                    return result
            with (
                self._prefill_graphs()
                if plan.phase == BatchPhase.PREFILL
                else nullcontext()
            ):
                return self._eager(plan, cache_plan)

    @observe_execution
    @execute_state
    def enqueue(self, plan, *, feedback=None):
        if feedback is not None:
            if not isinstance(feedback, GraphDeviceFeedback):
                raise ValueError("GLM continuation requires owned graph feedback")
            inputs = self._continuation.resume(plan.sequence_ids, feedback.continuation)
            with self._cuda_failure, torch.inference_mode():
                return self._enqueue_graph_output(
                    plan, feedback.continuation.shape, inputs, feedback
                )
        self._continuation.drain()
        if plan.phase != BatchPhase.DECODE or plan.needs_prompt_logprobs:
            return self.forward(plan)
        self._check_cuda_graphs(plan.phase)
        cache_plan = self._prepare(plan)
        with self._cuda_failure, torch.inference_mode():
            result = self._graph_forward(
                plan, cache_plan, enqueue=True, feedback=feedback
            )
        if result is None and feedback is not None:
            raise RuntimeError("a device continuation requires a captured graph")
        return self.forward(plan) if result is None else result

    def validate_distributed(self, plan, context):
        if self.parallel_groups is None or context.groups is not self.parallel_groups:
            raise ValueError("distributed groups do not match GLM runner")
        if plan is None:
            if context.num_sequences or context.num_tokens:
                raise ValueError("nonempty distributed replica needs a plan")
            cache_plan = None
        else:
            if (
                plan.phase != context.phase
                or plan.num_tokens != context.num_tokens
                or plan.num_sequences != context.num_sequences
            ):
                raise ValueError("distributed metadata does not match plan")
            cache_plan = self._prepare(plan)
        self._check_cuda_graphs(context.phase)
        return PreparedBatch(cache_plan)

    def _distributed_graph(
        self, plan, context, prepared, *, enqueue=False, feedback=None
    ):
        local_shape = (
            feedback.continuation.shape
            if feedback is not None
            else self._decode_graph_shape(plan, prepared.cache_plan)
            if enqueue and plan is not None
            else None
        )
        shape = self._select_distributed_cuda_graph(
            "decode",
            context,
            required_shape=local_shape
            if local_shape is not None
            else (
                0 if plan is None else plan.num_sequences,
                0 if plan is None else int(np.max(plan.positions)) + 1,
            ),
            eligible=not enqueue or plan is None or local_shape is not None,
        )
        if shape is None:
            return False, None
        if feedback is not None:
            inputs = self._continuation.resume(
                () if plan is None else plan.sequence_ids,
                feedback.continuation,
                shape=shape,
            )
            return True, self._enqueue_graph_output(plan, shape, inputs, feedback)
        if plan is not None:
            return True, self._graph_forward(
                plan,
                prepared.cache_plan,
                shape=shape,
                enqueue=enqueue,
                feedback=feedback,
            )
        inputs = self._decode_graph_capture_inputs(shape)
        inputs["ep_valid_rows"] = 0
        if self.speculative_draft_tokens:
            fixed = self._speculation.validated_benchmark_accepted_drafts
            inputs["fixed_acceptance"] = -1 if fixed is None else fixed
        if enqueue:
            return True, self._enqueue_graph_output(None, shape, inputs, None)
        self._run_cuda_graph("decode", shape, inputs)
        return True, None

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
            scope = (
                self._distributed_prefill_graphs(context)
                if context.phase == BatchPhase.PREFILL
                else nullcontext()
            )
            with scope:
                if plan is not None:
                    return self._eager(
                        plan,
                        prepared.cache_plan,
                        decode_batch_size=context.max_num_sequences
                        if context.phase == BatchPhase.DECODE
                        else None,
                    )
                if self._prefill_graph_active:
                    self._empty_prefill_regions()
                else:
                    if context.phase == BatchPhase.DECODE:
                        for _ in range(self.speculative_draft_tokens):
                            self._model.forward_mtp_experts_empty()
                    self._model.forward_experts_empty()
                    if self.speculative_draft_tokens:
                        self._model.forward_mtp_experts_empty()
                return None

    @observe_execution
    @execute_state
    def enqueue_distributed(self, plan, context, *, prepared=None, feedback=None):
        if feedback is not None:
            if not isinstance(feedback, GraphDeviceFeedback):
                raise ValueError("GLM continuation requires owned graph feedback")
        else:
            self._continuation.drain()
        if context.phase != BatchPhase.DECODE or (
            plan is not None and plan.needs_prompt_logprobs
        ):
            return self.forward_distributed(plan, context, prepared=prepared)
        prepared = (
            self.validate_distributed(plan, context) if prepared is None else prepared
        )
        with self._cuda_failure, torch.inference_mode():
            used, result = self._distributed_graph(
                plan, context, prepared, enqueue=True, feedback=feedback
            )
        if used:
            return result
        if feedback is not None:
            raise RuntimeError("a device continuation requires a captured graph")
        return self.forward_distributed(plan, context, prepared=prepared)
