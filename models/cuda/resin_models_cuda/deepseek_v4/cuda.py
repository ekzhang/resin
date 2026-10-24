"""Native DeepSeek-V4-Flash-0731 CUDA execution with explicit rank groups.

Resin owns the model hierarchy, kernels, request lifetime, sampling, and
scheduling. SWA, compressed history, and recurrent compressor tails are
logical cache groups backed by one fungible Engine-owned parent pool.
"""

from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from resin import BatchPhase, RunnerConfig
from resin.backend.cuda.completion import DeviceCompletion
from resin.backend.cuda.graphs import (
    CudaGraphMode,
    CudaGraphProgram,
    CudaGraphRegion,
    CudaGraphStep,
    GraphShape,
    resolve_decode_graph_shapes,
)
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.parallel import (
    FlashInferAllReduce,
    TensorParallelContext,
    TensorParallelLayout,
)
from resin.backend.cuda.runner import (
    CudaRunner,
    capture_sampling_inputs,
    graph_sample,
    pad_sampling_inputs,
)
from resin.backend.cuda.sampling import (
    SAMPLING_OFFSET_STRIDE,
)
from resin.backend.cuda.staging import TensorStager
from resin.backend.cuda.tensor_graphs import CudaGraphTensorRegion, TensorSpec
from resin.backend.plans import BatchView, PagedCachePlan
from resin.backend.sampling import (
    PromptScores,
    format_output,
    prompt_scores,
    token_output,
)
from resin.backend.speculative import SpeculativeStats
from resin.backend.topology import ParallelGroups, ParallelismCapabilities
from resin.backend.torch import use_default_dtype
from resin_models_cuda.deepseek_v4.cache import (
    CacheGeometry,
    plan_cache_capacity,
)
from resin_models_cuda.deepseek_v4.checkpoint import (
    load_config,
    resolve_model_path,
)
from resin_models_cuda.deepseek_v4.weights import load_deepseek_v4_weights

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput
    from resin.backend.distributed import StepContext


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    """Rank-local host metadata accepted by distributed readiness validation."""

    rows: tuple[tuple[int, int], ...]
    cache_plan: PagedCachePlan | None


class DeepSeekV4CudaRunner(CudaRunner):
    """Dynamically batched runner over Resin's DeepSeek-V4 implementation."""

    parallelism_capabilities = ParallelismCapabilities(
        attn_tp=False,
        attn_dp=True,
        cp=True,
        ep=True,
        expert_tp=True,
    )

    def __init__(
        self,
        model: str | Path,
        *,
        max_seq_len: int,
        max_batch_size: int = 1,
        tensor_parallel: TensorParallelContext | None = None,
        parallel_groups: ParallelGroups | None = None,
        device: str | torch.device,
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
        cuda_graph_decode_shapes: object | None = None,
        fast_allreduce: bool = True,
        speculative_tokens: int = 0,
        profile_graph_phases: bool = False,
        gpu_memory_utilization: float = 0.95,
        cache_parent_capacity: int | None = None,
        prefill_chunk_size: int = 4096,
    ) -> None:
        if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
            raise ValueError("max_seq_len must be a positive integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        if (
            isinstance(prefill_chunk_size, bool)
            or not isinstance(prefill_chunk_size, int)
            or prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer")
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
            raise ValueError("max_batch_size must be a positive integer")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if max_batch_size * max_seq_len > 2**32 - 1:
            raise ValueError("max_batch_size * max_seq_len exceeds scheduler metadata")
        if (
            isinstance(speculative_tokens, bool)
            or not isinstance(speculative_tokens, int)
            or speculative_tokens < 0
        ):
            raise ValueError("speculative_tokens must be a non-negative integer")
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        if cache_parent_capacity is not None and (
            isinstance(cache_parent_capacity, bool)
            or not isinstance(cache_parent_capacity, int)
            or cache_parent_capacity <= 0
        ):
            raise ValueError("cache_parent_capacity must be a positive integer")

        if parallel_groups is not None:
            if tensor_parallel is not None:
                raise ValueError("pass either tensor_parallel or parallel_groups")
            self.parallelism_capabilities.validate(
                parallel_groups.topology, model="DeepSeek-V4"
            )
            tensor_parallel = TensorParallelContext(
                TensorParallelLayout(
                    rank=parallel_groups.attn_tp.local_rank,
                    size=parallel_groups.attn_tp.size,
                ),
                parallel_groups.attn_tp.handle
                if parallel_groups.attn_tp.size > 1
                else None,
            )
        elif tensor_parallel is None:
            tensor_parallel = TensorParallelContext()
        if tensor_parallel.size != 1:
            raise ValueError("DeepSeek MLA has one KV head and requires attn_tp=1")
        self.parallel_groups = parallel_groups
        self.tensor_parallel = tensor_parallel
        self.tensor_parallel_rank = tensor_parallel.rank
        self.tensor_parallel_size = tensor_parallel.size
        super().__init__(
            device=device,
            label="DeepSeek-V4",
            kv_cache_dtype="bfloat16",
            cuda_graph_prefill=cuda_graph_prefill,
            cuda_graph_decode=cuda_graph_decode,
        )
        torch.cuda.set_device(self.device)
        if torch.cuda.get_device_capability(self.device)[0] < 10:
            raise RuntimeError("the DeepSeek-V4 runner requires Blackwell")
        if parallel_groups is None and self.tensor_parallel_size > 1:
            if not dist.is_initialized():
                raise RuntimeError(
                    "DeepSeek-V4 TP requires an initialized process group"
                )
            if (
                dist.get_world_size() != self.tensor_parallel_size
                or dist.get_rank() != self.tensor_parallel_rank
            ):
                raise ValueError("the default process group does not match TP layout")

        self.model_path, self.config = self._checkpoint(model)
        self.config.validate_tensor_parallel(self.tensor_parallel_size)
        if speculative_tokens > self.config.dspark_block_size:
            raise ValueError(
                "speculative_tokens exceeds this checkpoint's DSpark block size"
            )
        if speculative_tokens and not self.config.dspark_target_layer_ids:
            raise ValueError("speculative_tokens requires a DSpark checkpoint")
        if max_seq_len > 1_048_576:
            raise ValueError("max_seq_len exceeds the checkpoint context length")
        self.max_seq_len = max_seq_len
        self.prefill_chunk_size = min(prefill_chunk_size, max_seq_len)
        self.max_batch_size = max_batch_size
        self.speculative_tokens = speculative_tokens
        self.profile_graph_phases = profile_graph_phases
        self._dspark_phase_events: dict[GraphShape, tuple[torch.cuda.Event, ...]] = {}
        self._last_dspark_phase_ms: dict[str, float] | None = None
        self._graph_device_caches = {}
        self._graph_logits = {}
        self.vocab_size = self.config.vocab_size
        self.eos_token_id = 1
        self._speculation = SpeculativeStats(speculative_tokens)
        physical_seq_len = max_seq_len + speculative_tokens
        if speculative_tokens:
            physical_seq_len += 1
        decode_graph_shapes = resolve_decode_graph_shapes(
            cuda_graph_decode_shapes,
            policy=self.cuda_graph_decode,
            width=physical_seq_len,
            speculative=bool(speculative_tokens),
            max_batch_size=max_batch_size,
        )
        if any(shape[0] > max_batch_size for shape in decode_graph_shapes):
            raise ValueError("CUDA graph batch capacity exceeds max_batch_size")
        if any(shape[1] < speculative_tokens + 2 for shape in decode_graph_shapes):
            raise ValueError(
                "DeepSeek-V4 CUDA graph sequence capacity must cover position 1 "
                "and the complete verification block"
            )
        if any(shape[1] > physical_seq_len for shape in decode_graph_shapes):
            raise ValueError("CUDA graph sequence capacity exceeds max_seq_len")
        self._configure_cuda_graphs(
            {
                "decode": CudaGraphMode(
                    decode_graph_shapes,
                    self._new_decode_graph_step,
                    self._decode_graph_capture_inputs,
                )
            }
        )
        inference_config_path = self.model_path / "inference/config.json"
        try:
            inference_values = json.loads(inference_config_path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError("DeepSeek-V4 inference/config.json is invalid") from error
        if not isinstance(inference_values, dict):
            raise ValueError("DeepSeek-V4 inference config must contain an object")

        inference_values.update(
            {
                "max_batch_size": max_batch_size,
                "max_seq_len": physical_seq_len,
                "dspark_block_size": speculative_tokens,
            }
        )
        if speculative_tokens:
            if int(inference_values.get("n_mtp_layers", 0)) <= 0:
                raise ValueError("DSpark checkpoint has no speculative stages")
        else:
            inference_values["n_mtp_layers"] = 0
        args, self._model = self._build_model(inference_values)
        self._allreduce = FlashInferAllReduce(
            self.tensor_parallel,
            max_token_num=max_batch_size * max(1, speculative_tokens + 1),
            hidden_dim=self.config.hidden_size,
            dtype=torch.bfloat16,
            enabled=fast_allreduce,
        )
        index_widths = sorted(
            {
                min(
                    ((args.max_seq_len + 255) // 256) * 64,
                    ((shape[1] + 255) // 256) * 64,
                )
                for shape in decode_graph_shapes
            }
        )
        self._index_allreduces = {
            width: FlashInferAllReduce(
                self.tensor_parallel,
                max_token_num=max_batch_size * max(1, speculative_tokens + 1),
                hidden_dim=(width + 7) // 8 * 8,
                dtype=torch.bfloat16,
                enabled=fast_allreduce,
            )
            for width in index_widths
        }

        def reduce_index_scores(tensor: torch.Tensor) -> torch.Tensor:
            reducer = self._index_allreduces.get(tensor.shape[-1])
            if reducer is None:
                return self.tensor_parallel.sum(tensor)
            padding = reducer.hidden_dim - tensor.shape[-1]
            if padding == 0:
                return reducer.sum(tensor)
            padded = F.pad(tensor, (0, padding))
            return reducer.sum(padded)[..., : tensor.shape[-1]].contiguous()

        self.allreduce_backend = self._allreduce.backend
        self._model.set_allreduce(
            self._allreduce.sum,
            indexer_reducer=reduce_index_scores,
        )
        self.weight_load = self._load_weights()
        self._prepare_model_weights()
        self._configure_parallel_graphs(self._model, parallel_groups)

        cache_backend = self._cache_backend()
        self.cache_geometry = self._build_cache_geometry(args)
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity_plan = plan_cache_capacity(
            total_memory,
            free_memory,
            gpu_memory_utilization,
            self.cache_geometry,
            reserved_parent_count=1 + self.cache_geometry.graph_scratch_parents,
        )
        local_capacity = capacity_plan.parent_capacity
        if cache_parent_capacity is not None:
            local_capacity = min(local_capacity, cache_parent_capacity)
        if parallel_groups is not None and parallel_groups.replica.size > 1:
            # Replica control groups use Gloo, including ranks that replicate
            # attention. Every copy must back the same logical cache capacity.
            capacity_tensor = torch.tensor(local_capacity, dtype=torch.int64)
            parallel_groups.distributed.all_reduce(
                capacity_tensor,
                op=dist.ReduceOp.MIN,
                group=parallel_groups.replica.handle,
            )
            local_capacity = int(capacity_tensor.item())
        elif self.tensor_parallel.enabled:
            capacity_tensor = torch.tensor(
                local_capacity, dtype=torch.int64, device=self.device
            )
            dist.all_reduce(
                capacity_tensor,
                op=dist.ReduceOp.MIN,
                group=self.tensor_parallel.process_group,
            )
            local_capacity = int(capacity_tensor.item())
        self.cache_capacity_plan = replace(
            capacity_plan,
            parent_capacity=local_capacity,
            allocation_bytes=(local_capacity + capacity_plan.reserved_parent_count)
            * capacity_plan.bytes_per_parent,
        )
        self.cache_layout = self.cache_geometry.layout(local_capacity)
        self._cache = cache_backend.CacheArena(
            self.device,
            self.cache_layout,
            self.cache_geometry,
            dtype=torch.bfloat16,
        )
        self._configure_cache_binding(self._cache.binding)
        self._model.bind_cache_arena(self._cache)
        self.runner_config = RunnerConfig(
            max_batch_tokens=max_batch_size * self.prefill_chunk_size,
            prefill_chunk_size=self.prefill_chunk_size,
            continuation_prefill=True,
            cache_write_lookahead_tokens=speculative_tokens,
            cache_layout=self.cache_layout,
        )
        prefill_ffns = self._configure_prefill_graphs(
            tuple(layer.ffn for layer in self._model.layers),
            hidden_size=self.config.hidden_size,
            max_tokens=self.prefill_chunk_size,
            token_ids=True,
            pad_up_to=0,
            capture_context=self._prefill_compute_context,
        )
        for layer, callback in zip(self._model.layers, prefill_ffns, strict=True):
            layer.prefill_ffn = callback
        self._configure_prefill_attention_graphs()

    @staticmethod
    def _checkpoint(model):
        path = resolve_model_path(model)
        return path, load_config(path)

    @staticmethod
    def _cache_backend():
        from resin_models_cuda.deepseek_v4 import cache

        return cache

    def _build_cache_geometry(self, args):
        return CacheGeometry.build(
            args.compress_ratios,
            target_layers=args.n_layers,
            dspark_layers=args.n_mtp_layers,
            window_size=args.window_size,
            decode_input_tokens=max(1, self.speculative_tokens + 1),
            prefill_chunk_size=self.prefill_chunk_size,
            max_batch_size=self.max_batch_size,
        )

    def _build_model(self, values):
        # Imported after device validation; cuTile initialization is GPU-only.
        from resin_models_cuda.deepseek_v4 import model as implementation

        args = implementation.ModelArgs(
            **{
                key: value
                for key, value in values.items()
                if key in implementation.ModelArgs.__dataclass_fields__
            }
        )
        with use_default_dtype(torch.bfloat16), torch.device(self.device):
            model = implementation.Transformer(
                args,
                tensor_parallel=self.tensor_parallel,
                parallel_groups=self.parallel_groups,
            )
        return args, model

    def _load_weights(self):
        return load_deepseek_v4_weights(
            self.model_path,
            self._model,
            rank=self.tensor_parallel_rank,
            tensor_parallel_size=self.tensor_parallel_size,
            device=self.device,
        )

    def _prepare_model_weights(self):
        self._model.prepare_weights(self._allreduce.sum)

    def _configure_prefill_attention_graphs(self) -> None:
        """Capture dense projections around the ordinary eager sparse attention."""

        regions = []
        for layer in self._model.layers:
            attention = layer.attn

            def project(hidden, attention=attention):
                return tuple(
                    value.squeeze(0) for value in attention._project(hidden[None])
                )

            def finish(output, positions, attention=attention):
                return attention._finish(output[None], positions[None]).squeeze(0)

            regions.extend(
                (
                    CudaGraphTensorRegion(
                        f"layer.{layer.layer_id}.project",
                        project,
                        (TensorSpec((attention.dim,)),),
                        (
                            TensorSpec((attention.q_lora_rank,)),
                            TensorSpec((attention.head_dim,)),
                        ),
                    ),
                    CudaGraphTensorRegion(
                        f"layer.{layer.layer_id}.query",
                        attention.wq_b,
                        (TensorSpec((attention.q_lora_rank,)),),
                        TensorSpec((attention.n_local_heads * attention.head_dim,)),
                    ),
                    CudaGraphTensorRegion(
                        f"layer.{layer.layer_id}.finish",
                        finish,
                        (
                            TensorSpec((attention.n_local_heads, attention.head_dim)),
                            TensorSpec((), torch.int64),
                        ),
                        TensorSpec((attention.dim,)),
                    ),
                )
            )
        callbacks = self._configure_prefill_tensor_graphs(
            tuple(regions),
            max_tokens=self.prefill_chunk_size,
            pad_up_to=0,
            capture_context=self._prefill_compute_context,
        )
        for index, layer in enumerate(self._model.layers):
            project, query, finish = callbacks[3 * index : 3 * index + 3]
            layer.attn.prefill_project = lambda hidden, invoke=project: tuple(
                value.unsqueeze(0) for value in invoke(hidden.squeeze(0))
            )
            layer.attn.prefill_query = lambda qr, invoke=query: invoke(qr.squeeze(0))[
                None
            ]
            layer.attn.prefill_finish = lambda output, positions, invoke=finish: invoke(
                output.squeeze(0), positions.squeeze(0)
            )[None]

    @contextmanager
    def _prefill_compute_context(self):
        from resin_models_cuda.deepseek_v4.gemm import dynamic_gemm_rows

        with (
            dynamic_gemm_rows(),
            use_default_dtype(torch.bfloat16),
            torch.device(self.device),
        ):
            yield

    @property
    def speculative_stats(self) -> dict[str, int | float]:
        return self._speculation.snapshot()

    @property
    def last_dspark_phase_ms(self) -> dict[str, float] | None:
        """Return device timings from the most recent profiled graph replay."""

        if self._last_dspark_phase_ms is None:
            return None
        return dict(self._last_dspark_phase_ms)

    def _cache_plan(self, plan: BatchPlan):
        return PagedCachePlan.from_plan(
            plan,
            self.cache_layout,
            decode_lookahead=self.speculative_tokens,
        )

    @staticmethod
    def _graph_capture_inputs(
        shape: GraphShape, vocab_size: int = 129_280
    ) -> dict[str, object]:
        """Return deterministic synthetic inputs for one decode graph."""

        batch_size, sequence_capacity = shape
        if sequence_capacity < 2:
            raise ValueError("DeepSeek-V4 decode graphs require capacity >= 2")
        inputs = {
            "token_ids": (0,) * batch_size,
            "positions": (1,) * batch_size,
            **capture_sampling_inputs(batch_size, vocab_size, unique_seeds=True),
        }
        return inputs

    def _reset_model_state(self) -> None:
        """Restore the graph-only null parent after synthetic writes."""

        self._cache.initialize_state_sentinels()

    def _after_cuda_graph_capture(self) -> None:
        self._reset_model_state()
        torch.cuda.synchronize(self.device)

    def _decode_graph_capture_inputs(self, shape: GraphShape) -> dict[str, object]:
        inputs = self._graph_capture_inputs(shape, self.vocab_size)
        inputs.update(self._cache.graph_shape_capture_inputs(*shape))
        if self.parallel_groups is not None:
            inputs["ep_valid_rows"] = shape[0]
        return inputs

    def _new_decode_graph_step(
        self, shape: GraphShape
    ) -> CudaGraphStep | CudaGraphProgram:
        if self.speculative_tokens:
            return self._new_dspark_graph_step(shape)
        batch_size, graph_seq_len = shape
        cache = self._cache.graph_cache(batch_size, graph_seq_len)
        self._graph_device_caches[shape] = cache
        token_ids = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        output_ids = torch.empty(batch_size, dtype=torch.long, device=self.device)
        logits_buffer = torch.empty(
            (batch_size, self.vocab_size), dtype=torch.float32, device=self.device
        )
        self._graph_logits[shape] = logits_buffer
        valid_rows = (
            torch.empty((), dtype=torch.int32, device=self.device)
            if self.parallel_groups is not None
            else None
        )
        sampling = self._sampler.empty(batch_size, self.device)

        def launch() -> None:
            with (
                self._parallel_graph_capacity(batch_size, valid_rows=valid_rows),
                use_default_dtype(torch.bfloat16),
                torch.device(self.device),
            ):
                logits = self._model.forward_decode(
                    token_ids,
                    positions,
                    graph_seq_len,
                    cache=cache,
                )
                logits_buffer.copy_(logits)
                self._sampler.sample_tensors(logits, sampling, out=output_ids)
                if self.parallel_groups is not None:
                    self._broadcast_tokens(output_ids)

        def broadcast() -> None:
            dist.broadcast(
                output_ids,
                src=0,
                group=self.tensor_parallel.process_group,
            )

        inputs = {"token_ids": token_ids, "positions": positions}
        if valid_rows is not None:
            inputs["ep_valid_rows"] = valid_rows
        inputs.update(self._cache.graph_device_inputs(cache))
        inputs.update(self._sampler.graph_inputs(sampling))
        if self.parallel_groups is not None or not self.tensor_parallel.enabled:
            return CudaGraphStep(
                launch=launch,
                inputs=inputs,
                output=output_ids,
                prepare=self._reset_model_state,
            )
        if self.cuda_graph_decode == "full":

            def full_launch() -> None:
                launch()
                broadcast()

            return CudaGraphStep(
                launch=full_launch,
                inputs=inputs,
                output=output_ids,
                prepare=self._reset_model_state,
            )
        return CudaGraphProgram(
            regions=(
                CudaGraphRegion("model", launch),
                CudaGraphRegion("broadcast", broadcast, capture=False),
            ),
            inputs=inputs,
            output=output_ids,
            prepare=self._reset_model_state,
        )

    def _new_dspark_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        batch_size, graph_seq_len = shape
        cache = self._cache.graph_cache(batch_size, graph_seq_len)
        self._graph_device_caches[shape] = cache
        block_size = self.speculative_tokens
        token_ids = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
        positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        output_ids = torch.empty(
            (batch_size, 2 * block_size + 1),
            dtype=torch.long,
            device=self.device,
        )
        logits_buffer = torch.empty(
            (batch_size, block_size + 1, self.vocab_size),
            dtype=torch.float32,
            device=self.device,
        )
        self._graph_logits[shape] = logits_buffer
        valid_rows = (
            torch.empty((), dtype=torch.int32, device=self.device)
            if self.parallel_groups is not None
            else None
        )
        sampling = self._sampler.empty(batch_size, self.device)
        phase_timing = self._new_phase_timing("draft", "target_verify", "decode")
        phase_events: tuple[torch.cuda.Event, ...] | None = None
        if self.profile_graph_phases:
            phase_events = tuple(
                torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)
            )
            self._dspark_phase_events[shape] = phase_events

        def sample(logits: torch.Tensor, step: int) -> torch.Tensor:
            return self._broadcast_tokens(
                graph_sample(self._sampler, logits, sampling, step)
            )

        def launch() -> None:
            with use_default_dtype(torch.bfloat16), torch.device(self.device):
                if phase_timing is not None:
                    phase_timing.mark(0)
                if phase_events is not None:
                    phase_events[0].record()
                # Expert dispatch counts flattened token rows. Draft and
                # verification have different block widths, while the live
                # replica batch is a replay input that may also be zero.
                with self._parallel_graph_capacity(
                    batch_size * block_size,
                    valid_rows=None if valid_rows is None else valid_rows * block_size,
                ):
                    drafts = self._broadcast_tokens(
                        self._model.draft(token_ids, positions, cache=cache)
                    )
                if phase_timing is not None:
                    phase_timing.mark(1)
                if phase_events is not None:
                    phase_events[1].record()
                verify_ids = torch.cat((token_ids, drafts), dim=1)
                with self._parallel_graph_capacity(
                    batch_size * (block_size + 1),
                    valid_rows=None
                    if valid_rows is None
                    else valid_rows * (block_size + 1),
                ):
                    verify_logits, verify_hidden = self._model.forward_verify(
                        verify_ids, positions, graph_seq_len, cache=cache
                    )
                logits_buffer.copy_(verify_logits)
                if phase_timing is not None:
                    phase_timing.mark(2)
                if phase_events is not None:
                    phase_events[2].record()
                if verify_hidden is None:
                    raise RuntimeError(
                        "target did not produce DSpark verification states"
                    )
                self._model.commit_dspark_main_block(
                    verify_hidden, positions, cache=cache
                )
                output_ids[:, :block_size].copy_(drafts)
                for index in range(block_size + 1):
                    output_ids[:, block_size + index].copy_(
                        sample(verify_logits[:, index].contiguous(), index)
                    )
                if phase_timing is not None:
                    phase_timing.mark(3)
                if phase_events is not None:
                    phase_events[3].record()

        inputs = {"token_ids": token_ids, "positions": positions}
        if valid_rows is not None:
            inputs["ep_valid_rows"] = valid_rows
        inputs.update(self._cache.graph_device_inputs(cache))
        inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(
            inputs=inputs,
            output=output_ids,
            launch=launch,
            prepare=self._reset_model_state,
            phase_timing=phase_timing,
        )

    @overload
    def _graph_forward(
        self,
        plan: BatchPlan,
        cache_plan,
        *,
        shape: GraphShape | None = None,
        enqueue: Literal[False] = False,
        feedback=None,
    ) -> ForwardOutput | None: ...

    @overload
    def _graph_forward(
        self,
        plan: BatchPlan,
        cache_plan,
        *,
        shape: GraphShape | None = None,
        enqueue: Literal[True],
        feedback=None,
    ) -> DeviceCompletion | None: ...

    def _graph_forward(
        self,
        plan: BatchPlan,
        cache_plan,
        *,
        shape: GraphShape | None = None,
        enqueue: bool = False,
        feedback=None,
    ) -> ForwardOutput | DeviceCompletion | None:
        batch_size = plan.num_sequences
        positions = plan.positions
        sequence_length = int(np.max(positions)) + 1
        if shape is None:
            shape = self._select_cuda_graph(
                "decode",
                match=lambda candidate: (
                    candidate[0] >= batch_size
                    and candidate[1] >= sequence_length + self.speculative_tokens
                ),
            )
        if shape is None:
            return None
        padding_rows = shape[0] - batch_size
        inputs: dict[str, object] = {
            "token_ids": np.pad(plan.token_ids, (0, padding_rows)),
            "positions": np.pad(positions, (0, padding_rows), constant_values=1),
        }
        graph_cache = self._graph_device_caches[shape]
        inputs.update(self._cache.graph_plan_inputs(graph_cache, cache_plan))
        inputs.update(
            pad_sampling_inputs(self._sampler, plan, self.vocab_size, padding_rows)
        )
        if getattr(self, "parallel_groups", None) is not None:
            inputs["ep_valid_rows"] = batch_size
        if enqueue:
            from resin.backend.cuda.completion import accepted_block

            self._last_dspark_phase_ms = {}
            if feedback is not None:
                inputs.update(feedback.inputs())
                device_positions = feedback.positions
            else:
                device_positions = torch.as_tensor(
                    inputs["positions"], device=self.device, dtype=torch.long
                )
            output = self._run_cuda_graph("decode", shape, inputs)
            tokens, counts, next_inputs = accepted_block(
                output,
                device_positions,
                self.speculative_tokens,
                fixed_acceptance=self._speculation.benchmark_accepted_drafts,
            )
            if self.speculative_tokens:
                self._model.clear_rejected_speculative_state(
                    device_positions,
                    counts,
                    self.speculative_tokens + 1,
                    cache=graph_cache,
                )
            return DeviceCompletion(
                tokens,
                counts,
                next_inputs,
                rows=batch_size,
                stats=self._speculation if self.speculative_tokens else None,
                device=self.device,
            )
        output = self._run_cuda_graph("decode", shape, inputs)[:batch_size]
        phase_events = self._dspark_phase_events.get(shape)
        if phase_events is not None:
            start, draft_end, verify_end, step_end = phase_events
            step_end.synchronize()
            self._last_dspark_phase_ms = {
                "draft": start.elapsed_time(draft_end),
                "target_verify": draft_end.elapsed_time(verify_end),
                "commit_and_sampling": verify_end.elapsed_time(step_end),
                "device_total": start.elapsed_time(step_end),
            }
        if not self.speculative_tokens:
            if plan.needs_logprobs:
                return format_output(
                    self._graph_logits[shape][:batch_size], output, plan
                )
            return token_output(output)
        output = output.cpu().tolist()
        block_size = self.speculative_tokens
        drafts = [row[:block_size] for row in output]
        targets = [row[block_size:] for row in output]
        accepted = self._accept_speculative(
            drafts,
            targets,
            torch.tensor(positions, dtype=torch.long, device=self.device),
            graph_cache,
        )
        if plan.needs_logprobs:
            return format_output(self._graph_logits[shape][:batch_size], accepted, plan)
        return token_output(accepted)

    def _validate_plan(self, plan: BatchPlan) -> tuple[tuple[int, int], ...]:
        batch = BatchView.from_plan(plan)
        row_count = batch.num_sequences
        if row_count > self.max_batch_size:
            raise ValueError("DeepSeek-V4 batch size is outside runner capacity")
        rows = tuple(
            (int(start), int(end))
            for start, end in zip(batch.token_indptr, batch.token_indptr[1:])
        )

        if batch.phase == BatchPhase.PREFILL:
            if any(end - start > self.prefill_chunk_size for start, end in rows):
                raise ValueError(
                    "DeepSeek-V4 prefill row exceeds its cache write capacity"
                )
        elif batch.phase == BatchPhase.DECODE:
            if any(end - start != 1 for start, end in rows):
                raise ValueError("DeepSeek-V4 decode rows must contain one token")
        else:
            raise ValueError("DeepSeek-V4 plan phase must be prefill or decode")
        return rows

    def _prefill_region_tokens(self, plan, rows):
        """Optional per-region row demands, agreed before distributed prefill."""
        return {}

    def _prefill_row(self, plan, slot, input_ids, positions, cache):
        """Execute one sequence and commit its model-specific speculative state."""
        context = int(plan.context_lengths[slot])
        logits, main = self._model.forward_prefill(
            input_ids,
            context,
            cache=cache,
            **({"return_hidden": True} if plan.needs_prompt_logprobs else {}),
        )
        if self.speculative_tokens:
            if main is None:
                raise RuntimeError("target did not produce DSpark hidden states")
            if context == 0:
                self._model.prefill_dspark(main, cache=cache)
            else:
                self._model.commit_dspark_main_block(main, positions[:1], cache=cache)
        return logits

    def _prefill(
        self,
        plan: BatchPlan,
        rows: tuple[tuple[int, int], ...],
        cache,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, PromptScores | None]:
        """Prefill fresh prompts eagerly and replay only a restored suffix."""

        logits = []
        prompt_rows = []
        for slot, (start, end) in enumerate(rows):
            row_cache = cache.row(slot)
            input_ids = token_ids[start:end].unsqueeze(0)
            row_logits = self._prefill_row(
                plan, slot, input_ids, positions[start:end], row_cache
            )
            if plan.needs_prompt_logprobs:

                def project(hidden):
                    return self._model.head(hidden[None], full_logits=True)[0]

                prompt_rows.append(
                    prompt_scores(plan, row_logits[0], project, token_start=start)
                )
                row_logits = self._model.head(row_logits[:, -1:])
            logits.append(row_logits)
        return torch.cat(logits, dim=0), PromptScores.concat(prompt_rows)

    def _broadcast_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        groups = self.parallel_groups
        if groups is not None:
            if groups.device_replica.size > 1:
                groups.distributed.broadcast(
                    token_ids,
                    src=groups.device_replica.source,
                    group=groups.device_replica.handle,
                )
        elif self.tensor_parallel.enabled:
            dist.broadcast(
                token_ids,
                src=0,
                group=self.tensor_parallel.process_group,
            )
        return token_ids

    def _sample_target(
        self, logits: torch.Tensor, plan: BatchPlan, step: int
    ) -> torch.Tensor:
        return self._broadcast_tokens(
            self._sampler(
                logits,
                plan,
                offset_delta=step * SAMPLING_OFFSET_STRIDE,
            )
        )

    def _accept_speculative(
        self,
        draft_rows: list[list[int]],
        target_rows: list[list[int]],
        positions: torch.Tensor,
        cache,
    ) -> list[list[int]]:
        acceptance = self._speculation.accept(draft_rows, target_rows)
        processed_counts = torch.tensor(
            acceptance.processed_counts,
            dtype=torch.int32,
            device=self.device,
        )
        self._model.clear_rejected_speculative_state(
            positions,
            processed_counts,
            self.speculative_tokens + 1,
            cache=cache,
        )
        return acceptance.token_rows

    def _speculative_decode(
        self, plan: BatchPlan, cache, token_ids, positions
    ) -> ForwardOutput:
        """Draft greedily and verify a fixed block against target samples."""

        batch_size = plan.num_sequences
        anchors = token_ids.view(batch_size, 1)
        graph_seq_len = int(np.max(plan.positions)) + self.speculative_tokens + 1
        phase_timing = self._eager_phase_timing("draft", "target_verify", "decode")
        if phase_timing is not None:
            phase_timing.mark(0)
        drafts = self._broadcast_tokens(
            self._model.draft(anchors, positions, cache=cache)
        )
        if phase_timing is not None:
            phase_timing.mark(1)
        verify_ids = torch.cat((anchors, drafts), dim=1)
        verify_logits, verify_hidden = self._model.forward_verify(
            verify_ids,
            positions,
            graph_seq_len,
            cache=cache,
        )
        if verify_hidden is None:
            raise RuntimeError("target did not produce DSpark verification states")
        if phase_timing is not None:
            phase_timing.mark(2)
        self._model.commit_dspark_main_block(verify_hidden, positions, cache=cache)

        target_columns = [
            self._sample_target(verify_logits[:, index].contiguous(), plan, index)
            for index in range(self.speculative_tokens + 1)
        ]
        accepted = self._accept_speculative(
            drafts.tolist(),
            torch.stack(target_columns, dim=1).tolist(),
            positions,
            cache,
        )
        if phase_timing is not None:
            phase_timing.mark(3)
        return format_output(verify_logits, accepted, plan)

    @observe_execution
    def enqueue(self, plan: BatchPlan, *, feedback=None):
        """Submit graph decode and keep exact continuation inputs on the GPU.

        The Engine owns overlap and page leases. Unsupported shapes and scored
        outputs retain the ordinary synchronous forward contract.
        """
        if self.parallel_groups is not None:
            raise RuntimeError("explicit rank groups require enqueue_distributed")
        if (
            plan.phase != BatchPhase.DECODE
            or plan.needs_logprobs
            or plan.needs_prompt_logprobs
        ):
            return self.forward(plan)
        self._check_cuda_graphs(plan.phase)
        self._validate_plan(plan)
        cache_plan = self._cache_plan(plan)
        with self._cuda_failure:
            result = self._graph_forward(
                plan, cache_plan, enqueue=True, feedback=feedback
            )
        if result is None and feedback is not None:
            raise RuntimeError(
                "a device continuation requires its captured decode graph"
            )
        return self.forward(plan) if result is None else result

    @observe_execution
    def enqueue_distributed(self, plan, context, *, prepared=None, feedback=None):
        """Generic asynchronous graph submission across composed rank meshes."""
        if context.phase != BatchPhase.DECODE or (
            plan is not None and (plan.needs_logprobs or plan.needs_prompt_logprobs)
        ):
            return self.forward_distributed(plan, context, prepared=prepared)
        prepared = (
            self.validate_distributed(plan, context) if prepared is None else prepared
        )
        with self._cuda_failure:
            shape = self._select_distributed_cuda_graph(
                "decode",
                context,
                required_shape=(
                    0 if plan is None else plan.num_sequences,
                    0
                    if plan is None
                    else int(np.max(plan.positions)) + self.speculative_tokens + 1,
                ),
            )
            if shape is not None:
                if plan is not None:
                    return self._graph_forward(
                        plan,
                        prepared.cache_plan,
                        shape=shape,
                        enqueue=True,
                        feedback=feedback,
                    )
                from resin.backend.cuda.completion import (
                    DeviceCompletion,
                    accepted_block,
                )

                inputs = self._decode_graph_capture_inputs(shape)
                inputs["ep_valid_rows"] = 0
                output = self._run_cuda_graph("decode", shape, inputs)
                positions = torch.ones(shape[0], device=self.device, dtype=torch.long)
                tokens, counts, next_inputs = accepted_block(
                    output,
                    positions,
                    self.speculative_tokens,
                    fixed_acceptance=self._speculation.benchmark_accepted_drafts,
                )
                return DeviceCompletion(
                    tokens, counts, next_inputs, rows=0, device=self.device
                )
        if feedback is not None:
            raise RuntimeError(
                "a device continuation requires its captured decode graph"
            )
        return self.forward_distributed(plan, context, prepared=prepared)

    @observe_execution
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        """Run one dynamically batched prefill or decode sampling step."""

        self._check_cuda_graphs(plan.phase)
        if self.parallel_groups is not None:
            raise RuntimeError("explicit rank groups require forward_distributed")
        rows = self._validate_plan(plan)
        cache_plan = self._cache_plan(plan)
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
                        logits, scores = self._prefill(
                            plan, rows, cache, token_ids, positions
                        )
                elif self.speculative_tokens:
                    return self._speculative_decode(plan, cache, token_ids, positions)
                else:
                    input_ids = token_ids.view(plan.num_sequences, 1)
                    logits = self._model.forward_decode(
                        input_ids,
                        positions,
                        int(np.max(plan.positions)) + 1,
                        cache=cache,
                    )
                    scores = None
                output = self._sample_target(logits, plan, 0)
            return format_output(logits, output, plan, prompt_logprobs=scores)

    def validate_distributed(
        self, plan: BatchPlan | None, context: StepContext
    ) -> PreparedBatch:
        """Reject incompatible batches before the coordinator's readiness vote."""

        if self.parallel_groups is None or context.groups is not self.parallel_groups:
            raise ValueError("distributed step groups do not match this runner")
        if context.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
            raise ValueError("DeepSeek-V4 plan phase must be prefill or decode")
        rows, cache_plan = (), None
        if plan is None:
            if context.num_sequences or context.num_tokens:
                raise ValueError("a nonempty replica requires a batch plan")
        else:
            rows = self._validate_plan(plan)
            if (
                plan.phase != context.phase
                or plan.num_sequences != context.num_sequences
                or plan.num_tokens != context.num_tokens
            ):
                raise ValueError("distributed step metadata does not match the plan")
            cache_plan = self._cache_plan(plan)
        # Worker readiness votes on validation errors before any rank enters
        # a graph-selection collective or launches captured expert exchanges.
        self._check_cuda_graphs(context.phase)
        if context.phase == BatchPhase.DECODE and self.cuda_graph_decode != "disabled":
            self._require_cuda_graph_executor().mode("decode").check_ready()
        return PreparedBatch(rows, cache_plan)

    @observe_execution
    def forward_distributed(
        self,
        plan: BatchPlan | None,
        context: StepContext,
        *,
        prepared: PreparedBatch | None = None,
    ) -> ForwardOutput | None:
        """Join every EP layer even when this attention replica has no work.

        Prefill executes one sequence per model call. Replicas with shorter
        queues still enter each routed-expert round, without touching a cache.
        Token counts within a round may differ; expert dispatch carries them.
        Speculative decode joins draft stages before one target verification
        round, including replicas that have no local requests.
        """

        prepared = (
            self.validate_distributed(plan, context) if prepared is None else prepared
        )
        rows, cache_plan = prepared.rows, prepared.cache_plan
        with self._cuda_failure:
            if context.phase == BatchPhase.DECODE:
                used_graph, graph_output = self._distributed_decode_graph(
                    plan, context, cache_plan
                )
                if used_graph:
                    return graph_output
            staging = TensorStager(self.device)
            if plan is not None:
                cache = self._cache.tensorize(cache_plan, staging)
                token_ids = staging.device("token_ids", plan.token_ids, torch.long)
                positions = staging.device("positions", plan.positions, torch.long)
            with (
                torch.inference_mode(),
                use_default_dtype(torch.bfloat16),
                torch.device(self.device),
                self._prefill_compute_context()
                if context.phase == BatchPhase.PREFILL
                else nullcontext(),
            ):
                if context.phase == BatchPhase.PREFILL:
                    with self._distributed_prefill_graphs(
                        context,
                        per_sequence=True,
                        region_tokens=self._prefill_region_tokens(plan, rows),
                    ):
                        prefill_output = (
                            self._prefill(plan, rows, cache, token_ids, positions)
                            if plan is not None
                            else None
                        )
                        output_logits, scores = (
                            prefill_output
                            if prefill_output is not None
                            else (None, None)
                        )
                        for _ in range(context.max_num_sequences - len(rows)):
                            if self._prefill_graph_active:
                                self._empty_prefill_regions()
                            else:
                                self._model.forward_experts_empty()
                elif plan is not None and self.speculative_tokens:
                    return self._speculative_decode(plan, cache, token_ids, positions)
                elif plan is not None:
                    output_logits = self._model.forward_decode(
                        token_ids.view(plan.num_sequences, 1),
                        positions,
                        int(np.max(plan.positions)) + 1,
                        cache=cache,
                    )
                    scores = None
                else:
                    if self.speculative_tokens:
                        self._model.forward_experts_empty(stage="draft")
                    self._model.forward_experts_empty()
                    output_logits = None
                if plan is None:
                    # Active replicas synchronize when copying sampled tokens
                    # to the host. Empty replicas have no output transfer.
                    self._cuda_failure()
                    return None
                assert output_logits is not None
                output = self._sample_target(output_logits, plan, 0)
            return format_output(output_logits, output, plan, prompt_logprobs=scores)

    def _distributed_decode_graph(
        self, plan: BatchPlan | None, context: StepContext, cache_plan
    ) -> tuple[bool, ForwardOutput | None]:
        """Replay one common graph bucket across independent attention replicas."""

        shape = self._select_distributed_cuda_graph(
            "decode",
            context,
            required_shape=(
                0 if plan is None else plan.num_sequences,
                0
                if plan is None
                else int(np.max(plan.positions)) + self.speculative_tokens + 1,
            ),
        )
        if shape is None:
            return False, None
        if plan is not None:
            return True, self._graph_forward(plan, cache_plan, shape=shape)

        # Idle replicas use only reserved scratch parents and send zero-weight
        # expert rows. Their local experts still execute work for active peers.
        inputs = self._decode_graph_capture_inputs(shape)
        inputs["ep_valid_rows"] = 0
        self._run_cuda_graph("decode", shape, inputs)
        return True, None


__all__ = ["DeepSeekV4CudaRunner"]
