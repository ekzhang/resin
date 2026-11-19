"""GPT-OSS MXFP4 expert construction, sharding, and checkpoint adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from resin.backend.cuda.parallel import TensorParallelLayout
from resin.backend.expert_parallel import ExpertParallelDispatch
from resin.models.gpt_oss.config import GptOssConfig
from resin_models_cuda.gpt_oss.flashinfer import MXFP4Config, MXFP4Weights

if TYPE_CHECKING:
    from resin.backend.topology import ParallelGroups

_CHECKPOINT_TO_STANDARD = {
    "gate_up_proj_blocks": "gate_up",
    "gate_up_proj_scales": "gate_up_scales",
    "gate_up_proj_bias": "gate_up_bias",
    "down_proj_blocks": "down",
    "down_proj_scales": "down_scales",
    "down_proj_bias": "down_bias",
}
EXPERT_CHECKPOINT_NAMES = tuple(_CHECKPOINT_TO_STANDARD)


class CheckpointShards:
    """Use the same rank-local geometry for model construction and checkpoint reads."""

    def __init__(
        self,
        args: GptOssConfig,
        tensor_parallel: TensorParallelLayout = TensorParallelLayout(),
        *,
        expert_parallel: TensorParallelLayout = TensorParallelLayout(),
        expert_tensor_parallel: TensorParallelLayout = TensorParallelLayout(),
    ) -> None:
        query = tensor_parallel.even_shard(
            args.num_attention_heads, name="num_attention_heads"
        )
        kv = tensor_parallel.even_shard(
            args.num_key_value_heads, name="num_key_value_heads"
        )
        experts = expert_parallel.even_shard(args.num_local_experts, name="num_experts")
        intermediate = expert_tensor_parallel.aligned_shard(
            args.intermediate_size, alignment=32, name="intermediate_size"
        )
        self.model_args = replace(
            args,
            num_attention_heads=query.size,
            num_key_value_heads=kv.size,
            intermediate_size=intermediate.size,
        )
        self.kernel_args = replace(self.model_args, num_local_experts=experts.size)

        all_rows = slice(None)
        query_rows = slice(query.start * args.head_dim, query.stop * args.head_dim)
        kv_rows = slice(kv.start * args.head_dim, kv.stop * args.head_dim)
        owner = experts.as_slice() if expert_parallel.enabled else all_rows
        pairs = groups = all_rows
        if expert_tensor_parallel.enabled:
            pairs = slice(2 * intermediate.start, 2 * intermediate.stop)
            groups = slice(intermediate.start // 32, intermediate.stop // 32)
        self._slices: dict[str, tuple[slice, ...]] = (
            {
                "q_proj.weight": (query_rows,),
                "q_proj.bias": (query_rows,),
                "k_proj.weight": (kv_rows,),
                "k_proj.bias": (kv_rows,),
                "v_proj.weight": (kv_rows,),
                "v_proj.bias": (kv_rows,),
                "o_proj.weight": (all_rows, query_rows),
                "self_attn.sinks": (query.as_slice(),),
            }
            if tensor_parallel.enabled
            else {}
        )
        self._slices.update(
            {
                "experts.gate_up_proj_blocks": (owner, pairs),
                "experts.gate_up_proj_scales": (owner, pairs),
                "experts.gate_up_proj_bias": (owner, pairs),
                "experts.down_proj_blocks": (owner, all_rows, groups),
                "experts.down_proj_scales": (owner, all_rows, groups),
                "experts.down_proj_bias": (owner,),
            }
        )
        self._remote_bias = {
            "o_proj.bias": tensor_parallel.rank != 0,
            "experts.down_proj_bias": expert_tensor_parallel.rank != 0,
        }

    def __call__(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        name = ".".join(key.rsplit(".", 2)[-2:])
        if (selection := self._slices.get(name)) is not None:
            tensor = tensor[selection].contiguous()
        return torch.zeros_like(tensor) if self._remote_bias.get(name) else tensor


class KernelExperts(torch.nn.Module):
    """Prepared MXFP4 experts with optional distributed dispatch."""

    storage_format = "mxfp4"

    def __init__(
        self,
        args,
        kernel,
        config: MXFP4Config,
        *,
        parallel_groups: ParallelGroups | None = None,
    ) -> None:
        super().__init__()
        self._kernel = kernel
        self._config = config
        actual = (
            args.num_local_experts,
            args.hidden_size,
            args.intermediate_size,
            args.num_experts_per_tok,
        )
        expected = (
            config.num_experts * parallel_groups.ep.size
            if parallel_groups is not None
            else config.num_experts,
            config.hidden_size,
            config.intermediate_size,
            config.top_k,
        )
        if actual != expected:
            raise ValueError(
                f"expert module dimensions {actual} do not match kernel {expected}"
            )
        self._prepared_weights = None
        self._device = None
        self._expert_start = (
            parallel_groups.ep.local_rank * config.num_experts
            if parallel_groups is not None
            else 0
        )
        self._dispatch = (
            ExpertParallelDispatch(
                self.forward_local,
                num_experts=args.num_local_experts,
                groups=parallel_groups,
                # CP prefill rows differ; head shards and attention copies
                # share the same complete FFN input rows.
                replication=(
                    parallel_groups.attn_tp,
                    parallel_groups.attn_replication,
                ),
            )
            if parallel_groups is not None
            else None
        )

    def load_weights(self, checkpoint_tensors: Mapping[str, Any]) -> None:
        if self._prepared_weights is not None:
            raise RuntimeError("packed expert weights are already installed")
        raw_tensors = {}
        for name, tensor in checkpoint_tensors.items():
            standard_name = _CHECKPOINT_TO_STANDARD[name]
            if name.endswith("_blocks"):
                if tensor.ndim != 4 or tensor.shape[-1] != 16:
                    raise ValueError(
                        f"{name} must have checkpoint shape [E, N, K/32, 16], "
                        f"got {tuple(tensor.shape)}"
                    )
                tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
            raw_tensors[standard_name] = tensor

        raw = MXFP4Weights(
            gate_up=raw_tensors["gate_up"],
            down=raw_tensors["down"],
            gate_up_scales=raw_tensors["gate_up_scales"],
            down_scales=raw_tensors["down_scales"],
            gate_up_bias=raw_tensors["gate_up_bias"],
            down_bias=raw_tensors["down_bias"],
        )
        self._prepared_weights = self._kernel.prepare_weights(raw, self._config)
        self._device = raw.gate_up.device

    @property
    def is_prepared(self) -> bool:
        return self._prepared_weights is not None

    def forward(self, hidden, indices, weights):
        if self._prepared_weights is None:
            raise RuntimeError("packed expert weights have not been loaded")
        if self._dispatch is not None:
            # Keep the collective ABI identical for cuTile and torch
            # routers, and for ranks entering an empty expert round.
            return self._dispatch(
                hidden, indices.to(torch.int32), weights.to(torch.float32)
            )
        return self._kernel.forward(hidden, indices, weights, self._prepared_weights)

    def forward_local(self, hidden, indices, weights):
        """Execute owned experts with their original global routing weights."""

        if self._prepared_weights is None:
            raise RuntimeError("packed expert weights have not been loaded")
        if hidden.shape[0] == 0:
            return torch.zeros_like(hidden, dtype=torch.float32)
        config = self._config
        local = indices - self._expert_start
        owned = (local >= 0) & (local < config.num_experts)
        # Local num_experts is the inactive sentinel: substituting expert
        # zero could hide a real route because CUTLASS takes the first match.
        return self._kernel.forward_partial(
            hidden,
            torch.where(owned, local, config.num_experts),
            weights.masked_fill(~owned, 0),
            self._prepared_weights,
        )

    def forward_empty(self) -> None:
        """Participate in EP while this attention replica has no token rows."""

        if self._dispatch is None or self._device is None:
            raise RuntimeError("empty expert rounds require prepared EP weights")
        config = self._config
        hidden = torch.empty(
            (0, config.hidden_size), device=self._device, dtype=torch.bfloat16
        )
        indices = torch.empty((0, config.top_k), device=self._device, dtype=torch.int32)
        weights = torch.empty(
            (0, config.top_k), device=self._device, dtype=torch.float32
        )
        self._dispatch(hidden, indices, weights)


__all__ = [
    "EXPERT_CHECKPOINT_NAMES",
    "KernelExperts",
    "CheckpointShards",
]
