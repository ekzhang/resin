"""GPT-OSS MXFP4 expert construction, sharding, and checkpoint adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from resin.backend.cuda.parallel import TensorParallelLayout, TensorShard
from resin.backend.expert_parallel import ExpertParallelDispatch
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


def _slice_axis(tensor, axis: int, shard: TensorShard):
    selection = [slice(None)] * tensor.ndim
    selection[axis] = shard.as_slice()
    return tensor[tuple(selection)].contiguous()


def shard_model_config(
    args,
    tensor_parallel: TensorParallelLayout,
    *,
    expert_tensor_parallel: TensorParallelLayout | None = None,
):
    """Keep attention heads and expert channels on their independent meshes."""

    expert_tensor_parallel = expert_tensor_parallel or tensor_parallel
    query_heads = tensor_parallel.even_shard(
        args.num_attention_heads, name="num_attention_heads"
    )
    kv_heads = tensor_parallel.even_shard(
        args.num_key_value_heads, name="num_key_value_heads"
    )
    intermediate = expert_tensor_parallel.aligned_shard(
        args.intermediate_size,
        alignment=32,
        name="intermediate_size",
    )
    return replace(
        args,
        num_attention_heads=query_heads.size,
        num_key_value_heads=kv_heads.size,
        intermediate_size=intermediate.size,
    )


def shard_checkpoint_tensor(
    key: str,
    tensor,
    *,
    args,
    tensor_parallel: TensorParallelLayout,
    expert_parallel: TensorParallelLayout | None = None,
    expert_tensor_parallel: TensorParallelLayout | None = None,
):
    """Return one rank's canonical contiguous checkpoint shard."""

    expert_name = key.rsplit(".", 1)[-1]
    if expert_name in EXPERT_CHECKPOINT_NAMES and expert_parallel is not None:
        shard = expert_parallel.even_shard(args.num_local_experts, name="num_experts")
        tensor = _slice_axis(tensor, 0, shard)
    expert_tensor_parallel = expert_tensor_parallel or tensor_parallel

    if expert_name in EXPERT_CHECKPOINT_NAMES and expert_tensor_parallel.enabled:
        intermediate = expert_tensor_parallel.aligned_shard(
            args.intermediate_size,
            alignment=32,
            name="intermediate_size",
        )
        if expert_name in {
            "gate_up_proj_blocks",
            "gate_up_proj_scales",
            "gate_up_proj_bias",
        }:
            paired = TensorShard(2 * intermediate.start, 2 * intermediate.stop)
            return _slice_axis(tensor, 1, paired)
        if expert_name in {"down_proj_blocks", "down_proj_scales"}:
            grouped = TensorShard(intermediate.start // 32, intermediate.stop // 32)
            return _slice_axis(tensor, 2, grouped)
        if expert_name == "down_proj_bias":
            return (
                tensor if expert_tensor_parallel.rank == 0 else torch.zeros_like(tensor)
            )

    if not tensor_parallel.enabled:
        return tensor

    query_heads = tensor_parallel.even_shard(
        args.num_attention_heads, name="num_attention_heads"
    )
    kv_heads = tensor_parallel.even_shard(
        args.num_key_value_heads, name="num_key_value_heads"
    )
    query_width = TensorShard(
        query_heads.start * args.head_dim,
        query_heads.stop * args.head_dim,
    )
    kv_width = TensorShard(
        kv_heads.start * args.head_dim,
        kv_heads.stop * args.head_dim,
    )

    if key.endswith(".self_attn.q_proj.weight") or key.endswith(
        ".self_attn.q_proj.bias"
    ):
        return _slice_axis(tensor, 0, query_width)
    if key.endswith(".self_attn.k_proj.weight") or key.endswith(
        ".self_attn.k_proj.bias"
    ):
        return _slice_axis(tensor, 0, kv_width)
    if key.endswith(".self_attn.v_proj.weight") or key.endswith(
        ".self_attn.v_proj.bias"
    ):
        return _slice_axis(tensor, 0, kv_width)
    if key.endswith(".self_attn.o_proj.weight"):
        return _slice_axis(tensor, 1, query_width)
    if key.endswith(".self_attn.sinks"):
        return _slice_axis(tensor, 0, query_heads)
    if key.endswith(".self_attn.o_proj.bias"):
        return tensor if tensor_parallel.rank == 0 else torch.zeros_like(tensor)

    return tensor


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
    "shard_checkpoint_tensor",
    "shard_model_config",
]
