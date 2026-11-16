"""Inkling 2 packed experts with shared graphable expert-parallel transport."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
from torch import nn

from resin.backend.expert_parallel import ExpertParallelDispatch


class Inkling2Experts(nn.Module):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor

    def __init__(
        self,
        config,
        *,
        precision=None,
        parallel_groups=None,
        expert_backend="triton",
    ):
        super().__init__()
        if expert_backend not in ("triton", "cute-dsl"):
            raise ValueError(f"unsupported Inkling 2 expert backend: {expert_backend}")
        self.expert_backend = expert_backend
        self.precision = dict(precision or {"w13": "bf16", "w2": "bf16"})
        self.hidden, self.full_intermediate = (
            config.hidden_size,
            config.intermediate_size,
        )
        self.gate_cap, self.up_cap = config.mlp_gate_softcap, config.mlp_up_softcap
        self.num_experts = config.num_routed_experts
        ep_size = parallel_groups.ep.size if parallel_groups else 1
        ep_rank = parallel_groups.ep.local_rank if parallel_groups else 0
        self.expert_tp_size = parallel_groups.expert_tp.size if parallel_groups else 1
        self.expert_tp_rank = (
            parallel_groups.expert_tp.local_rank if parallel_groups else 0
        )
        self.expert_tp_group = parallel_groups.expert_tp if parallel_groups else None
        self.direct_expert_tp = bool(
            parallel_groups
            and ep_size == 1
            and self.expert_tp_size > 1
            and parallel_groups.expert_tp.ranks == parallel_groups.attn_tp.ranks
        )
        # Attention TP already holds identical input rows and routing on every
        # rank. When those ranks are exactly the expert owners, only their
        # FP32 local sums need communicating; no row exchange is necessary.
        self.replicated_ep = (
            parallel_groups.ep
            if parallel_groups
            and ep_size > 1
            and self.expert_tp_size == 1
            and parallel_groups.ep.ranks == parallel_groups.attn_tp.ranks
            else None
        )
        self.fast_sum: Callable[[torch.Tensor], torch.Tensor] | None = None
        if self.num_experts % ep_size:
            raise ValueError("Inkling 2 expert count must be divisible by EP")
        if self.full_intermediate % self.expert_tp_size:
            raise ValueError(
                "Inkling 2 intermediate width must be divisible by expert TP"
            )
        self.intermediate = self.full_intermediate // self.expert_tp_size
        self.expert_count = self.num_experts // ep_size
        self.expert_start = ep_rank * self.expert_count

        def parameter(name, shape, dtype):
            self.register_parameter(
                name, nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            )

        for name, n, k in (
            ("w13", 2 * self.intermediate, self.hidden),
            ("w2", self.hidden, self.intermediate),
        ):
            mode = self.precision[name]
            if mode not in ("nvfp4", "mxfp8", "bf16"):
                raise ValueError(f"unsupported Inkling 2 expert precision: {mode}")
            group = 16 if mode == "nvfp4" else 32
            if mode != "bf16" and k % group:
                raise ValueError(f"{name} input width must be divisible by {group}")
            dtype = {
                "nvfp4": torch.uint8,
                "mxfp8": torch.float8_e4m3fn,
                "bf16": torch.bfloat16,
            }[mode]
            parameter(
                name + "_weight",
                (self.expert_count, n, k // (2 if mode == "nvfp4" else 1)),
                dtype,
            )
            if mode != "bf16":
                parameter(
                    name + "_weight_scale",
                    (self.expert_count, n, k // group),
                    torch.float8_e4m3fn if mode == "nvfp4" else torch.uint8,
                )
            if mode == "nvfp4":
                for suffix, size in (
                    ("amax", self.expert_count),
                    ("outer_scale_denominator", 1),
                ):
                    parameter(
                        name + "_weight_" + suffix,
                        (size,),
                        torch.float32,
                    )
        self.dispatch = (
            ExpertParallelDispatch(
                self._local,
                num_experts=self.num_experts,
                groups=parallel_groups,
                replication=(parallel_groups.attn_tp, parallel_groups.attn_replication),
            )
            if parallel_groups
            and parallel_groups.device_domain.size > 1
            and not self.direct_expert_tp
            and self.replicated_ep is None
            else None
        )

    def reduce_activation_amax(self, amax):
        """NVFP4's per-route activation scale covers the full intermediate axis."""
        group = self.expert_tp_group
        if group is not None and group.size > 1 and amax.numel():
            dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=group.handle)
        return amax

    def reduce_down_partials(self, partials):
        """Complete each expert route in FP32 before its single BF16 rounding."""
        if partials.dtype != torch.float32:
            raise ValueError(
                "Inkling 2 W2 partials must remain FP32 until expert TP reduction"
            )
        group = self.expert_tp_group
        if group is not None and group.size > 1 and partials.numel():
            dist.all_reduce(partials, op=dist.ReduceOp.SUM, group=group.handle)
        output = partials.to(torch.bfloat16)
        if not self.direct_expert_tp and self.expert_tp_rank:
            # Shared expert transport sums contributions from every ETP rank.
            # Only one rank contributes the already completed expert routes.
            output.zero_()
        return output

    def prepare(self):
        """Prepare native packed GEMMs; only NVFP4 block scales change layout."""
        if not self.w13_weight.is_cuda:
            return
        if torch.cuda.get_device_capability(self.w13_weight.device)[0] < 10:
            raise ValueError("Inkling 2 packed experts require a Blackwell GPU")
        from resin_models_cuda.inkling2.expert_kernels import warm_routing

        warm_routing(self.expert_start, self.expert_count, self.w13_weight.device)
        if self.expert_backend == "cute-dsl" and "nvfp4" in self.precision.values():
            from resin_models_cuda.inkling2.expert_cute import prepare_backend

            prepare_backend(self)

    def _local(self, hidden, ids, weights, *, max_rows_per_expert=None):
        from resin_models_cuda.inkling2.expert_kernels import local_experts

        return local_experts(
            self, hidden, ids, weights, max_rows_per_expert=max_rows_per_expert
        )

    def forward(
        self, hidden, ids, weights, *, defer_rounding=False, max_rows_per_expert=None
    ):
        # Empty sources must still participate in distributed expert exchange.
        if self.dispatch is not None:
            return self.dispatch(hidden, ids, weights)
        # Only the direct path preserves the router's local-row bound.
        # Exchanged routes reach _local through dispatch with the default None.
        result = self._local(
            hidden, ids, weights, max_rows_per_expert=max_rows_per_expert
        )
        if self.replicated_ep is not None:
            # Each complete expert route was rounded to BF16 by the local
            # kernel. Retain its FP32 top-k sum through the owner reduction.
            if result.numel():
                if self.fast_sum is None:
                    dist.all_reduce(result, group=self.replicated_ep.handle)
                else:
                    result = self.fast_sum(result)
            return result if defer_rounding else result.to(hidden.dtype)
        return result
