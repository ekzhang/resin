"""GPT-OSS CUDA runner over paged FA4 and one Engine-owned cache pool."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from flashinfer.comm import (
    AllReduceFusionPattern,
    AllReduceFusionWorkspace,
    allreduce_fusion,
    create_allreduce_fusion_workspace,
)
from flashinfer.comm.mnnvl import TorchDistBackend
from flashinfer.gemm import tinygemm_bf16
from quack import rmsnorm
from quack.rotary import apply_rotary
from torch import Tensor

from resin import (
    BatchPhase,
    CacheGroupLayout,
    CacheLayout,
    CacheRetention,
    RunnerConfig,
)
from resin.backend.capacity import CacheCapacityPlan, plan_cache_capacity
from resin.backend.cuda.context_parallel import (
    ContextParallelTokens,
    context_parallel_query_batch,
)
from resin.backend.cuda.failures import GraphInputError
from resin.backend.cuda.graphs import (
    DEFAULT_MAX_NUM_SEQS,
    CudaGraphMode,
    CudaGraphStep,
    GraphShape,
    resolve_decode_graph_shapes,
)
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.paged_kv import (
    DevicePagedAttentionBatch,
    PagedKvArena,
    empty_decode_paged_attention_batch,
    pad_paged_attention_table,
    paged_attention_graph_inputs,
    paged_attention_host_inputs,
    stage_paged_attention_batch,
)
from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.backend.cuda.runner import (
    CudaRunner,
    capture_sampling_inputs,
    pad_sampling_inputs,
)
from resin.backend.cuda.staging import TensorStager
from resin.backend.cuda.tensor_graphs import CudaGraphTensorRegion, TensorSpec
from resin.backend.paged_attention import PagedAttentionBatch
from resin.backend.quantization import QuantizationConfig
from resin.backend.sampling import format_output, prompt_scores
from resin.backend.topology import ParallelGroups, ParallelismCapabilities
from resin.backend.torch import benchmark_expert_indices
from resin.models.gpt_oss.config import GptOssConfig
from resin.models.gpt_oss.plan import (
    PAGE_SIZE,
    PREFILL_CHUNK_SIZE,
    SLIDING_WINDOW,
    WINDOW_RETENTION_TOKENS,
    WINDOW_TABLE_WIDTH,
    build_attention_batches,
)
from resin_models_cuda.gpt_oss.checkpoint import (
    DEFAULT_MODEL,
    load_model,
    resolve_model_path,
)
from resin_models_cuda.gpt_oss.cutile import top4_softmax
from resin_models_cuda.gpt_oss.model import MLP, TopKRouter

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput
    from resin.backend.distributed import StepContext


_BF16_BYTES = 2
_TINYGEMM_MAX_TOKENS = 128
_AttentionBatches = tuple[tuple[PagedAttentionBatch, ...], PagedAttentionBatch | None]


class _EmptyFa4AuxData:
    """Zero-width CUTLASS JIT argument for GPT-OSS's unused FA4 auxiliary data."""

    def __init__(self, value) -> None:
        if value.tensors is not None or value.scalars is not None:
            raise NotImplementedError(
                "GPT-OSS attention does not support FA4 auxiliary data"
            )
        self._value = value

    def __getattr__(self, name: str):
        return getattr(self._value, name)

    def __c_pointers__(self) -> list[object]:
        return []

    def __get_mlir_types__(self) -> list[object]:
        return []

    def __new_from_mlir_values__(self, _values: list[object]):
        return self


def _register_empty_fa4_aux_data() -> None:
    """Adapt FA4's empty auxiliary bundle to the CUTLASS JIT boundary."""

    from cutlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry
    from flash_attn.cute.utils import AuxData

    if AuxData in JitArgAdapterRegistry.jit_arg_adapter_registry:
        return
    JitArgAdapterRegistry.register_jit_arg_adapter(AuxData)(_EmptyFa4AuxData)


def plan_gpt_oss_kv_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    gpu_memory_utilization: float,
    *,
    attention_layers_per_group: int = 18,
    tensor_parallel_size: int = 1,
) -> CacheCapacityPlan:
    """Size the fungible KV pool from post-load free device memory.

    ``gpu_memory_utilization`` is the total model-plus-cache fraction. Model,
    communication, and persistent kernel allocations have already been made,
    so protecting ``(1 - utilization) * total`` from the current free memory
    gives the exact remaining KV budget. History and sliding-window groups use
    one shared physical parent pool and therefore need no capacity split.
    """

    for name, value in (
        ("attention_layers_per_group", attention_layers_per_group),
        ("tensor_parallel_size", tensor_parallel_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if 8 % tensor_parallel_size:
        raise ValueError("tensor_parallel_size must divide the model's KV heads")

    bytes_per_parent = (
        2
        * attention_layers_per_group
        * PAGE_SIZE
        * (8 // tensor_parallel_size)
        * 64
        * _BF16_BYTES
    )
    return plan_cache_capacity(
        total_memory_bytes,
        free_memory_bytes,
        gpu_memory_utilization,
        bytes_per_parent,
    )


class GptOssCudaRunner(CudaRunner):
    """GPT-OSS BF16 attention on FA4 over one native cache pool."""

    model_label = "GPT-OSS"
    parallelism_capabilities = ParallelismCapabilities(
        attn_tp=True,
        attn_dp=True,
        cp=True,
        ep=True,
        expert_tp=True,
    )
    page_size = PAGE_SIZE
    sliding_window = SLIDING_WINDOW
    prefill_chunk_size = PREFILL_CHUNK_SIZE
    window_retention_tokens = WINDOW_RETENTION_TOKENS
    window_table_width = WINDOW_TABLE_WIDTH
    default_vocab_size = 201_088

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        cache_parent_capacity: int | None = None,
        revision: str | None = None,
        device: str | torch.device,
        dtype=None,
        quantization: QuantizationConfig | None = None,
        kernel_attention=None,
        kernel_moe=None,
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
        cuda_graph_decode_shapes: object = None,
        tensor_parallel: TensorParallelContext | None = None,
        parallel_groups: ParallelGroups | None = None,
        fast_allreduce: bool = True,
        gpu_memory_utilization: float = 0.95,
    ) -> None:
        if parallel_groups is not None:
            if tensor_parallel is not None:
                raise ValueError("pass either tensor_parallel or parallel_groups")
            self.parallelism_capabilities.validate(
                parallel_groups.topology, model=self.model_label
            )
            group = parallel_groups.attn_tp
            tensor_parallel = TensorParallelContext(
                TensorParallelLayout(rank=group.local_rank, size=group.size),
                group.handle if group.size > 1 else None,
            )
        self.parallel_groups = parallel_groups
        if kernel_attention is None:
            from flash_attn.cute import flash_attn_varlen_func

            kernel_attention = flash_attn_varlen_func
        if not callable(kernel_attention):
            raise TypeError("kernel_attention must be callable")
        if getattr(kernel_attention, "__module__", "").startswith("flash_attn."):
            _register_empty_fa4_aux_data()

        if cache_parent_capacity is not None and (
            isinstance(cache_parent_capacity, bool)
            or not isinstance(cache_parent_capacity, int)
            or cache_parent_capacity <= 0
        ):
            raise ValueError("cache_parent_capacity must be a positive integer")
        if not 0.0 < gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be between zero and one")
        self.tensor_parallel = tensor_parallel or TensorParallelContext()
        self.tensor_parallel_rank = self.tensor_parallel.rank
        self.tensor_parallel_size = self.tensor_parallel.size
        super().__init__(
            device=device,
            label=self.model_label,
            kv_cache_dtype="bfloat16",
            cuda_graph_prefill=cuda_graph_prefill,
            cuda_graph_decode=cuda_graph_decode,
        )

        self._fa4 = kernel_attention
        if torch.cuda.get_device_capability(self.device)[0] != 9:
            raise RuntimeError(
                f"the {self.model_label} FA4 runner requires an SM90 GPU"
            )
        resolved_dtype = torch.bfloat16 if dtype is None else dtype
        if isinstance(resolved_dtype, str):
            resolved_dtype = getattr(torch, resolved_dtype.removeprefix("torch."), None)
        if resolved_dtype is not torch.bfloat16:
            raise ValueError(
                f"the {self.model_label} FA4 runner requires bfloat16 activations"
            )

        selected_model = DEFAULT_MODEL if model is None else model
        self.model_path = resolve_model_path(
            selected_model,
            revision,
            quantization=quantization,
            kernel_moe=kernel_moe,
        )
        self._model, self.config = load_model(
            self.model_path,
            device=self.device,
            dtype=torch.bfloat16,
            quantization=quantization,
            kernel_moe=kernel_moe,
            tensor_parallel=self.tensor_parallel.layout,
            parallel_groups=parallel_groups,
        )
        self._configure_parallel_graphs(self._model, parallel_groups)
        self._graph_logits = {}
        self._last_graph_logits = None
        args = self._model.args
        checkpoint_args = GptOssConfig.from_dict(self.config)
        expected_attention = (
            checkpoint_args.num_attention_heads // self.tensor_parallel_size,
            checkpoint_args.num_key_value_heads // self.tensor_parallel_size,
            checkpoint_args.head_dim,
            self.sliding_window,
        )
        actual_attention = (
            args.num_attention_heads,
            args.num_key_value_heads,
            args.head_dim,
            args.sliding_window,
        )
        if actual_attention != expected_attention:
            raise ValueError(
                f"the {self.model_label} FA4 geometry is "
                f"{expected_attention}, got {actual_attention}"
            )

        self.num_query_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.eos_token_id = self.config.get("eos_token_id")

        self._allreduce_workspace: AllReduceFusionWorkspace | None = None
        self.allreduce_backend = "local"
        if self.tensor_parallel.enabled and fast_allreduce and parallel_groups is None:
            self._allreduce_workspace = create_allreduce_fusion_workspace(
                backend="auto",
                world_size=self.tensor_parallel_size,
                rank=self.tensor_parallel_rank,
                max_token_num=self.prefill_chunk_size,
                hidden_dim=self.hidden_size,
                dtype=torch.bfloat16,
                comm_backend=TorchDistBackend(group=self.tensor_parallel.process_group),
                group=self.tensor_parallel.process_group,
            )
            self.allreduce_backend = self._allreduce_workspace.backend
        elif self.tensor_parallel.enabled:
            self.allreduce_backend = "nccl"

        for layer in self._model.model.layers:
            layer.self_attn.pack_qkv_projection()
            # MNNVL's fused RMSNorm reads gamma in the communication tensor's
            # dtype. The checkpoint values are BF16 even though RMSNorm
            # parameters are initially materialized as FP32.
            layer.input_layernorm.to(dtype=torch.bfloat16)
            layer.post_attention_layernorm.to(dtype=torch.bfloat16)
        self._model.model.norm.to(dtype=torch.bfloat16)

        self._full_layers = tuple(
            index
            for index, layer_type in enumerate(args.layer_types)
            if layer_type == "full_attention"
        )
        self._sliding_layers = tuple(
            index
            for index, layer_type in enumerate(args.layer_types)
            if layer_type == "sliding_attention"
        )
        self._full_layer_slots = {
            layer_index: slot for slot, layer_index in enumerate(self._full_layers)
        }
        self._sliding_layer_slots = {
            layer_index: slot for slot, layer_index in enumerate(self._sliding_layers)
        }
        if len(self._full_layers) != len(self._sliding_layers):
            raise ValueError(
                f"{self.model_label} History and Window groups must have equal geometry"
            )
        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity_plan = plan_gpt_oss_kv_capacity(
            total_memory,
            free_memory,
            gpu_memory_utilization,
            attention_layers_per_group=len(self._full_layers),
            tensor_parallel_size=self.tensor_parallel_size,
        )
        local_capacity = capacity_plan.parent_capacity
        if cache_parent_capacity is not None:
            local_capacity = min(local_capacity, cache_parent_capacity)
        if parallel_groups is not None and parallel_groups.replica.size > 1:
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
        self.cache_parent_capacity = local_capacity
        self.cache_capacity_plan = replace(
            capacity_plan,
            parent_capacity=local_capacity,
            allocation_bytes=(local_capacity + 1) * capacity_plan.bytes_per_parent,
        )
        self.cache_layout = CacheLayout(
            parent_capacity=self.cache_parent_capacity,
            groups=[
                CacheGroupLayout(self.page_size),
                CacheGroupLayout(
                    self.page_size,
                    retention=CacheRetention.WINDOW,
                    window_tokens=self.window_retention_tokens,
                    read_window_tokens=self.sliding_window,
                ),
            ],
        )
        # Sparse native plans remain chronological; the runner only derives
        # dense FA4 tables and page-aligned kernel rounds from each frozen plan.
        self.runner_config = RunnerConfig(
            max_batch_tokens=self.prefill_chunk_size,
            prefill_chunk_size=self.prefill_chunk_size,
            continuation_prefill=True,
            cache_layout=self.cache_layout,
        )
        decode_graph_shapes = resolve_decode_graph_shapes(
            cuda_graph_decode_shapes,
            policy=self.cuda_graph_decode,
            width=min(
                self.cache_parent_capacity,
                (checkpoint_args.max_position_embeddings + self.page_size - 1)
                // self.page_size,
            ),
            speculative=False,
            max_batch_size=DEFAULT_MAX_NUM_SEQS,
        )
        if any(batch > DEFAULT_MAX_NUM_SEQS for batch, _ in decode_graph_shapes):
            raise ValueError(
                f"CUDA graph batch capacity exceeds {DEFAULT_MAX_NUM_SEQS}"
            )
        if any(width > self.cache_parent_capacity for _, width in decode_graph_shapes):
            raise ValueError("CUDA graph table width exceeds cache capacity")
        self._configure_cuda_graphs(
            {
                "decode": CudaGraphMode(
                    decode_graph_shapes,
                    self._new_decode_graph_step,
                    self._decode_graph_capture_inputs,
                )
            }
        )
        # FA4's SM90 TMA path can load a complete physical page even when
        # ``seqused_k`` masks its unused tail.  Keep unwritten rows finite so a
        # masked PV operation can never encounter uninitialized NaN/Inf data.
        self._kv = PagedKvArena(
            self.device,
            self.cache_layout,
            layer_count=len(self._full_layers),
            page_size=self.page_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            resource=f"{self.model_label} KV",
        )
        self._configure_cache_binding(self._kv.binding)

        for layer in self._model.model.layers:
            if layer.self_attn.sinks.dtype is not torch.bfloat16:
                raise TypeError("FA4 learnable sink tensors must be bfloat16")
        self._prefill_feed_forward = self._configure_prefill_graphs(
            tuple(
                lambda hidden, _ids, layer=layer: self._feed_forward(
                    layer.mlp, hidden.unsqueeze(0)
                ).squeeze(0)
                for layer in self._model.model.layers
            ),
            hidden_size=self.hidden_size,
            max_tokens=self.prefill_chunk_size,
        )
        self._prefill_attention = self._configure_prefill_tensor_graphs(
            tuple(
                CudaGraphTensorRegion(
                    f"layer.{index}.qkv_rope",
                    lambda hidden, cos, sin, attention=layer.self_attn: (
                        self._attention_project(attention, hidden, cos, sin)
                    ),
                    inputs=(
                        TensorSpec((self.hidden_size,)),
                        TensorSpec((self.head_dim // 2,)),
                        TensorSpec((self.head_dim // 2,)),
                    ),
                    output=TensorSpec(
                        (
                            (self.num_query_heads + 2 * self.num_kv_heads)
                            * self.head_dim,
                        )
                    ),
                )
                for index, layer in enumerate(self._model.model.layers)
            ),
            max_tokens=self.prefill_chunk_size,
            mode="prefill_attention",
            pad_up_to=0,
        )

    @classmethod
    def _graph_capture_inputs(
        cls, shape: GraphShape, vocab_size: int | None = None
    ) -> dict[str, tuple]:
        """Return valid synthetic decode inputs backed only by null page zero."""

        batch_size, full_width = shape
        vocab_size = cls.default_vocab_size if vocab_size is None else vocab_size
        rows = tuple(range(batch_size))
        zeros = (0,) * batch_size
        return {
            "token_ids": zeros,
            "positions": zeros,
            "token_indices": rows,
            "sequence_indices": rows,
            "full_page_table": (0,) * (batch_size * full_width),
            "full_seqused_k": (1,) * batch_size,
            "sliding_page_table": (0,) * (batch_size * cls.window_table_width),
            "sliding_seqused_k": (1,) * batch_size,
            "full_append_slots": rows,
            "sliding_append_slots": rows,
            **capture_sampling_inputs(batch_size, vocab_size, stochastic=True),
        }

    def _after_cuda_graph_capture(self) -> None:
        # Capture writes synthetic K/V to parent zero, which the native
        # allocator permanently reserves. Restore its null-page contract
        # before a request can reach the runner.
        self._kv.storage[:, :, 0].zero_()
        torch.cuda.synchronize(self.device)

    def _decode_graph_capture_inputs(self, shape):
        inputs = self._graph_capture_inputs(shape, self.vocab_size)
        if self.parallel_groups is not None:
            inputs["ep_valid_rows"] = shape[0]
        return inputs

    def _tensorize(
        self, batch: PagedAttentionBatch, staging: TensorStager, prefix: str
    ) -> DevicePagedAttentionBatch:
        return stage_paged_attention_batch(batch, staging, prefix)

    def _apply_yarn_rotary(self, hidden, cos, sin):
        """Apply the model's already-materialized YaRN cos/sin tensors."""

        # The runner already gathers the YaRN rows for every packed token. A
        # synthetic batch dimension lets Quack consume those rows directly,
        # independent of the requests they came from.
        if hidden.shape[0] == 0:
            return hidden
        apply_rotary(
            hidden.unsqueeze(0),
            cos.squeeze(0),
            sin.squeeze(0),
            interleaved=False,
            inplace=True,
        )
        return hidden

    def _attention(
        self,
        layer_index: int,
        attention,
        hidden,
        position_embeddings,
        batch: DevicePagedAttentionBatch,
        *,
        context_parallel: ContextParallelTokens | None = None,
        query_batch: DevicePagedAttentionBatch | None = None,
    ):
        token_count = hidden.shape[1]
        flat = hidden.reshape(token_count, self.hidden_size)
        cos, sin = position_embeddings
        projections = self._prefill_attention if self._prefill_graph_active else None
        projected = (
            projections[layer_index](flat, cos.squeeze(0), sin.squeeze(0))
            if projections is not None
            else self._attention_project(
                attention, flat, cos.squeeze(0), sin.squeeze(0)
            )
        )
        output = self._attention_core(
            layer_index,
            attention,
            projected,
            batch,
            context_parallel=context_parallel,
            query_batch=query_batch,
        )
        if token_count == 0:
            return hidden.new_empty((1, 0, self.hidden_size))
        return self._attention_output_projection(attention.o_proj, output).unsqueeze(0)

    def _attention_project(self, attention, hidden, cos, sin) -> Tensor:
        """Project token rows and rotate Q/K without touching cache metadata."""

        token_count = hidden.shape[0]
        projected = attention.qkv_proj(hidden)
        query, key, _value = projected.split(
            (
                self.num_query_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ),
            dim=-1,
        )
        query = query.view(token_count, self.num_query_heads, self.head_dim)
        key = key.view(token_count, self.num_kv_heads, self.head_dim)
        self._apply_yarn_rotary(query, cos.unsqueeze(0), sin.unsqueeze(0))
        self._apply_yarn_rotary(key, cos.unsqueeze(0), sin.unsqueeze(0))
        return projected

    def _attention_core(
        self,
        layer_index: int,
        attention,
        projected: Tensor,
        batch: DevicePagedAttentionBatch,
        *,
        context_parallel: ContextParallelTokens | None = None,
        query_batch: DevicePagedAttentionBatch | None = None,
    ) -> Tensor:
        """Append actual K/V rows and attend with the current request metadata."""

        token_count = projected.shape[0]
        query, key, value = projected.split(
            (
                self.num_query_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ),
            dim=-1,
        )
        query = query.view(token_count, self.num_query_heads, self.head_dim)
        key = key.view(token_count, self.num_kv_heads, self.head_dim)
        value = value.view(token_count, self.num_kv_heads, self.head_dim)
        if context_parallel is not None:
            key, value = context_parallel.gather(key), context_parallel.gather(value)
        layer_type = attention.layer_type
        if query.dtype is not torch.bfloat16:
            raise TypeError("FA4 query tensor must be bfloat16")
        if key.dtype is not torch.bfloat16 or value.dtype is not torch.bfloat16:
            raise TypeError("FA4 K/V tensors must be bfloat16")
        if attention.sinks.dtype is not torch.bfloat16:
            raise TypeError("FA4 learnable sink tensor must be bfloat16")

        sliding = layer_type == "sliding_attention"
        if sliding:
            slot = self._sliding_layer_slots[layer_index]
            group = batch.group("sliding")
        else:
            slot = self._full_layer_slots[layer_index]
            group = batch.group("full")
        key_cache = self._kv.keys[slot]
        value_cache = self._kv.values[slot]

        key_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
            0, group.append_slots, key
        )
        value_cache.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
            0, group.append_slots, value
        )
        if token_count == 0:
            return projected.new_empty((0, self.num_query_heads * self.head_dim))
        if query_batch is not None:
            batch = query_batch
            group = batch.group("sliding" if sliding else "full")
        output = self._fa4(
            query,
            key_cache,
            value_cache,
            cu_seqlens_q=batch.cu_seqlens_q,
            cu_seqlens_k=None,
            max_seqlen_q=batch.max_seqlen_q,
            max_seqlen_k=group.max_length,
            min_seqlen_k=group.min_length,
            seqused_q=None,
            seqused_k=group.lengths,
            page_table=group.page_table,
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=(self.sliding_window - 1, 0) if sliding else (None, None),
            softcap=0.0,
            num_splits=1,
            pack_gqa=True,
            deterministic=False,
            return_lse=False,
            learnable_sink=attention.sinks,
        )
        if isinstance(output, tuple):
            output = output[0]
        if output.dtype is not torch.bfloat16:
            raise TypeError("FA4 output tensor must be bfloat16")
        return output.reshape(token_count, -1)

    def _attention_output_projection(self, projection, hidden: Tensor) -> Tensor:
        """Keep row-parallel GEMM partials unrounded until their TP reduction."""

        if not self.tensor_parallel.enabled:
            return projection(hidden)
        partial = torch.mm(hidden, projection.weight.t(), out_dtype=torch.float32)
        if projection.bias is not None:
            partial.add_(projection.bias.float())
        return partial

    def _allreduce_residual_rmsnorm(
        self,
        partial: Tensor,
        residual: Tensor,
        weight: Tensor,
        eps: float,
        *,
        reduce: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if partial.numel() == 0:
            return partial, residual
        workspace = (
            self._allreduce_workspace
            if reduce and partial.dtype == residual.dtype
            else None
        )
        if workspace is None:
            reduced = self.tensor_parallel.sum(partial) if reduce else partial
            reduced = reduced.to(residual.dtype)
            normalized, combined = rmsnorm(
                reduced,
                weight,
                residual=residual,
                eps=eps,
                prenorm=True,
            )
            return normalized, combined

        shape = partial.shape
        flat_partial = partial.reshape(-1, shape[-1])
        flat_residual = residual.reshape(-1, shape[-1])
        combined = torch.empty_like(flat_partial)
        normalized = torch.empty_like(flat_partial)
        allreduce_fusion(
            input=flat_partial,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=False,
            trigger_completion_at_end=True,
            residual_in=flat_residual,
            residual_out=combined,
            norm_out=normalized,
            rms_gamma=weight,
            rms_eps=eps,
        )
        return normalized.view(shape), combined.view(shape)

    @staticmethod
    def _feed_forward(mlp: MLP, hidden: Tensor) -> Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        if flat.shape[0] == 0:
            getattr(mlp.experts, "forward_empty")()
            return hidden
        logits = _router_logits(mlp.router, flat)
        indices, weights = _route(mlp.router, logits)
        return mlp.experts(flat, indices, weights).reshape(shape)

    def _forward_batch(
        self,
        token_ids,
        positions,
        batch: DevicePagedAttentionBatch,
        *,
        context_parallel: ContextParallelTokens | None = None,
        full_hidden: bool = False,
    ):
        selected_tokens = token_ids.index_select(0, batch.token_indices)
        selected_positions = positions.index_select(0, batch.token_indices)
        query_batch = None
        if context_parallel is not None:
            selected_tokens = context_parallel.select(selected_tokens)
            selected_positions = context_parallel.select(selected_positions)
            query_batch = context_parallel_query_batch(batch, context_parallel)
        hidden = self._model.model.embed_tokens(selected_tokens).unsqueeze(0)
        position_embeddings = self._model.model.rotary_emb(
            selected_positions.unsqueeze(0), hidden.dtype
        )
        layers = self._model.model.layers
        normalized = (
            rmsnorm(
                hidden,
                layers[0].input_layernorm.weight,
                eps=layers[0].input_layernorm.eps,
            )
            if hidden.numel()
            else hidden
        )
        for layer_index, layer in enumerate(layers):
            attention_out = self._attention(
                layer_index,
                layer.self_attn,
                normalized,
                position_embeddings,
                batch,
                context_parallel=context_parallel,
                query_batch=query_batch,
            )
            normalized_mlp_input, hidden = self._allreduce_residual_rmsnorm(
                attention_out,
                hidden,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.eps,
            )
            feed_forward_output = self._prefill_feed_forward[layer_index](
                normalized_mlp_input
            )
            next_norm = (
                layers[layer_index + 1].input_layernorm
                if layer_index + 1 < len(layers)
                else self._model.model.norm
            )
            normalized, hidden = self._allreduce_residual_rmsnorm(
                feed_forward_output,
                hidden,
                next_norm.weight,
                next_norm.eps,
                reduce=self.parallel_groups is None,
            )
        final_hidden = normalized.squeeze(0)
        if context_parallel is not None:
            final_hidden = context_parallel.gather(final_hidden)
        return (
            final_hidden
            if full_hidden
            else final_hidden.index_select(0, batch.last_token_indices)
        )

    @staticmethod
    def _pad_page_table(batch: PagedAttentionBatch, width: int):
        """Pad each row by repeating one page already owned by that row."""

        try:
            return pad_paged_attention_table(batch.group("full"), width)
        except ValueError as error:
            raise GraphInputError(str(error)) from error

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        batch_size, full_width = shape

        def empty(dtype, *shape):
            return torch.empty(shape, dtype=dtype, device=self.device)

        batch = empty_decode_paged_attention_batch(
            self.device,
            batch_size,
            {
                "full": (full_width, self.page_size),
                "sliding": (self.window_table_width, self.page_size),
            },
        )
        token_ids = empty(torch.long, batch_size)
        positions = empty(torch.long, batch_size)
        final_hidden = empty(torch.bfloat16, batch_size, self.hidden_size)
        output_ids = empty(torch.long, batch_size)
        logits_buffer = empty(torch.float32, batch_size, self.vocab_size)
        self._graph_logits[shape] = logits_buffer
        valid_rows = (
            torch.empty((), dtype=torch.int32, device=self.device)
            if self.parallel_groups is not None
            else None
        )
        sampling = (
            self._sampler.empty(batch_size, self.device)
            if hasattr(self._sampler, "empty")
            else None
        )

        def launch() -> None:
            with self._parallel_graph_capacity(batch_size, valid_rows=valid_rows):
                hidden = self._forward_batch(token_ids, positions, batch)
                final_hidden.index_copy_(0, batch.sequence_indices, hidden)
                logits = self._model.lm_head(final_hidden)
                logits_buffer.copy_(logits)
                if sampling is None:
                    self._sampler(logits, out=output_ids)
                else:
                    self._sampler.sample_tensors(logits, sampling, out=output_ids)
                groups = self.parallel_groups
                if groups is not None and groups.device_replica.size > 1:
                    groups.distributed.broadcast(
                        output_ids,
                        src=groups.device_replica.source,
                        group=groups.device_replica.handle,
                    )

        inputs = {
            "token_ids": token_ids,
            "positions": positions,
            **paged_attention_graph_inputs(batch),
        }
        if valid_rows is not None:
            inputs["ep_valid_rows"] = valid_rows
        if sampling is not None:
            inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(
            inputs=inputs,
            output=output_ids,
            launch=launch,
        )

    def _graph_forward(
        self,
        plan: BatchPlan,
        host: PagedAttentionBatch,
        *,
        shape: GraphShape | None = None,
    ) -> ForwardOutput | None:
        if not self._cuda_graph_mode_enabled("decode"):
            return None
        rows = len(plan.sequence_ids)
        full = host.group("full")
        sliding = host.group("sliding")
        if (
            host.phase != BatchPhase.DECODE
            or host.query_lengths != (1,) * rows
            or sliding.width != self.window_table_width
        ):
            raise GraphInputError("plan is not a fixed-shape decode batch")
        if shape is None:
            shape = self._select_cuda_graph(
                "decode",
                match=lambda candidate: (
                    candidate[0] >= rows and candidate[1] >= full.width
                ),
            )
        if shape is None:
            return None
        inputs = {
            "token_ids": plan.token_ids,
            "positions": plan.positions,
            **paged_attention_host_inputs(
                host,
                {"full": shape[1], "sliding": self.window_table_width},
            ),
        }
        if self.parallel_groups is not None:
            inputs["ep_valid_rows"] = rows
        padding = shape[0] - rows
        if padding:
            for name in ("token_ids", "positions"):
                inputs[name] = np.pad(inputs[name], (0, padding))
            inactive_rows = np.arange(rows, shape[0], dtype=np.int64)
            for name in ("token_indices", "sequence_indices"):
                inputs[name] = np.concatenate((inputs[name], inactive_rows))
            for name, width in (
                ("full", shape[1]),
                ("sliding", self.window_table_width),
            ):
                inputs[f"{name}_page_table"] = np.pad(
                    inputs[f"{name}_page_table"], (0, padding * width)
                )
                inputs[f"{name}_seqused_k"] = np.pad(
                    inputs[f"{name}_seqused_k"], (0, padding), constant_values=1
                )
                # Parent zero is reserved. Distinct offsets keep inactive
                # writes from racing while leaving every live page untouched.
                inputs[f"{name}_append_slots"] = np.concatenate(
                    (inputs[f"{name}_append_slots"], inactive_rows)
                )
        if hasattr(self._sampler, "host_inputs"):
            inputs.update(
                pad_sampling_inputs(self._sampler, plan, self.vocab_size, padding)
            )
        output = self._run_cuda_graph("decode", shape, inputs)[:rows]
        self._last_graph_logits = self._graph_logits[shape][:rows]
        return format_output(self._last_graph_logits, output, plan)

    @observe_execution
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        if getattr(self, "parallel_groups", None) is not None:
            raise RuntimeError("explicit rank groups require forward_distributed")
        return self._forward_plan(plan)

    def _attention_batches(self, plan: BatchPlan) -> _AttentionBatches:
        prefill_batches, decode_batch = build_attention_batches(plan, self.cache_layout)
        host_batches = prefill_batches + (
            () if decode_batch is None else (decode_batch,)
        )
        covered = {index for batch in host_batches for index in batch.sequence_indices}
        if covered != set(range(len(plan.sequence_ids))):
            raise ValueError(
                "native attention batches do not cover every plan sequence"
            )
        return prefill_batches, decode_batch

    def _forward_plan(
        self,
        plan: BatchPlan,
        *,
        use_graph=True,
        prepared: _AttentionBatches | None = None,
    ) -> ForwardOutput:
        self._check_cuda_graphs(plan.phase)
        host_positions = plan.positions

        prefill_batches, decode_batch = (
            self._attention_batches(plan) if prepared is None else prepared
        )

        with self._cuda_failure:
            if use_graph and not prefill_batches and decode_batch is not None:
                graph_output = self._graph_forward(plan, decode_batch)
                if graph_output is not None:
                    return graph_output

            staging = TensorStager(self.device)
            token_ids = staging.device("token_ids", plan.token_ids, torch.long)
            positions = staging.device("positions", host_positions, torch.long)
            final_hidden = torch.empty(
                len(plan.sequence_ids),
                self.hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            prompt_hidden = (
                torch.empty(
                    (len(plan.token_ids), self.hidden_size),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                if plan.needs_prompt_logprobs
                else None
            )
            with torch.inference_mode():
                with self._prefill_graphs():
                    for index, host_batch in enumerate(prefill_batches):
                        batch = self._tensorize(host_batch, staging, f"prefill.{index}")
                        groups = self.parallel_groups
                        context_parallel = None
                        if groups is not None and groups.cp.size > 1:
                            context_parallel = ContextParallelTokens(
                                host_batch.query_lengths,
                                groups.cp,
                                device=self.device,
                                distributed=groups.distributed,
                            )
                        last_hidden = self._forward_batch(
                            token_ids,
                            positions,
                            batch,
                            context_parallel=context_parallel,
                            full_hidden=prompt_hidden is not None,
                        )
                        if prompt_hidden is not None:
                            prompt_hidden.index_copy_(
                                0, batch.token_indices, last_hidden
                            )
                            last_hidden = last_hidden.index_select(
                                0, batch.last_token_indices
                            )
                        final_hidden.index_copy_(0, batch.sequence_indices, last_hidden)
                if decode_batch is not None:
                    batch = self._tensorize(decode_batch, staging, "decode")
                    last_hidden = self._forward_batch(token_ids, positions, batch)
                    final_hidden.index_copy_(0, batch.sequence_indices, last_hidden)
                logits = self._model.lm_head(final_hidden)
                scores = (
                    prompt_scores(plan, prompt_hidden, self._model.lm_head)
                    if prompt_hidden is not None
                    else None
                )

            tokens = self._sampler(logits, plan)
            groups = getattr(self, "parallel_groups", None)
            if groups is not None and groups.device_replica.size > 1:
                groups.distributed.broadcast(
                    tokens,
                    src=groups.device_replica.source,
                    group=groups.device_replica.handle,
                )
            return format_output(logits, tokens, plan, prompt_logprobs=scores)

    def validate_distributed(
        self, plan: BatchPlan | None, context: StepContext
    ) -> _AttentionBatches:
        """Prepare rank-local attention metadata before the readiness vote."""

        if self.parallel_groups is None or context.groups is not self.parallel_groups:
            raise ValueError("distributed step groups do not match this runner")
        if context.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
            raise ValueError("GPT-OSS plan phase must be prefill or decode")
        prepared: _AttentionBatches = ((), None)
        if plan is None:
            if context.num_sequences or context.num_tokens:
                raise ValueError("a nonempty replica requires a batch plan")
        else:
            if (
                plan.phase != context.phase
                or plan.num_sequences != context.num_sequences
                or len(plan.token_ids) != context.num_tokens
            ):
                raise ValueError("distributed step metadata does not match the plan")
            prepared = self._attention_batches(plan)
            prefills, decode = prepared
            if len(prefills) + (decode is not None) != 1:
                raise ValueError(
                    "distributed GPT-OSS requires one packed attention batch"
                )
        # Worker readiness votes on validation errors before any rank enters
        # a graph-selection collective or launches captured expert exchanges.
        self._check_cuda_graphs(context.phase)
        if context.phase == BatchPhase.DECODE and self.cuda_graph_decode != "disabled":
            self._require_cuda_graph_executor().mode("decode").check_ready()
        return prepared

    @observe_execution
    def forward_distributed(
        self,
        plan: BatchPlan | None,
        context: StepContext,
        *,
        prepared: _AttentionBatches | None = None,
    ) -> ForwardOutput | None:
        """Run one packed batch, or join its expert layers without KV work."""

        prepared = (
            self.validate_distributed(plan, context) if prepared is None else prepared
        )
        self._last_graph_logits = None
        with self._cuda_failure:
            if context.phase == BatchPhase.DECODE:
                host = prepared[1]
                shape = self._select_distributed_cuda_graph(
                    "decode",
                    context,
                    required_shape=(
                        context.num_sequences,
                        0 if host is None else host.group("full").width,
                    ),
                )
                if shape is not None:
                    if plan is not None:
                        assert host is not None
                        return self._graph_forward(plan, host, shape=shape)
                    inputs = self._decode_graph_capture_inputs(shape)
                    inputs["ep_valid_rows"] = 0
                    self._run_cuda_graph("decode", shape, inputs)
                    return None
            if context.phase == BatchPhase.PREFILL:
                with self._distributed_prefill_graphs(context):
                    if plan is not None:
                        return self._forward_plan(plan, prepared=prepared)
                    with torch.inference_mode():
                        if self._prefill_graph_active:
                            self._empty_prefill_regions()
                        else:
                            self._forward_experts_empty()
            elif plan is not None:
                return self._forward_plan(plan, use_graph=False, prepared=prepared)
            else:
                with torch.inference_mode():
                    self._forward_experts_empty()
            self._cuda_failure()
            return None

    def _forward_experts_empty(self):
        for layer in self._model.model.layers:
            getattr(layer.mlp.experts, "forward_empty")()


def _router_logits(router: TopKRouter, hidden: Tensor) -> Tensor:
    """Use Hopper TinyGEMM for skinny decode routers and GEMM otherwise."""

    if hidden.shape[0] <= _TINYGEMM_MAX_TOKENS:
        output = torch.empty(
            (hidden.shape[0], router.weight.shape[0]),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        tinygemm_bf16(
            hidden,
            router.weight,
            output,
            router.bias,
            use_pdl=False,
        )
        return output
    return torch.nn.functional.linear(hidden, router.weight, router.bias)


def _route(router: TopKRouter, logits: Tensor) -> tuple[Tensor, Tensor]:
    if router.benchmark_balanced_experts:
        indices = benchmark_expert_indices(
            logits,
            router.top_k,
            router.benchmark_replica,
            router.benchmark_expert_parallel_size,
        )
        return indices, torch.softmax(logits.gather(1, indices), dim=-1)
    if router.top_k == 4 and logits.shape[-1] in (32, 128):
        return top4_softmax(logits)
    values, indices = torch.topk(logits, router.top_k, dim=-1, sorted=True)
    return indices, torch.softmax(values, dim=-1)


__all__ = [
    "GptOssCudaRunner",
    "plan_gpt_oss_kv_capacity",
    "_route",
    "_router_logits",
]
