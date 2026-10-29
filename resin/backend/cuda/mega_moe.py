"""DeepGEMM's fused EP/shared-expert FFN behind the prepared-layer boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from resin.backend.collectives import LayerCollective
from resin.backend.topology import RankGroup

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer


def prepare_weights(
    weight: Tensor, scales: Tensor, *, block_rows: int = 1
) -> tuple[Tensor, Tensor]:
    """Preserve checkpoint values while preparing per-row packed UE8M0 scales."""
    import deep_gemm

    n = weight.shape[-2]
    k = scales.shape[-1] * 32
    expanded = scales.float().repeat_interleave(block_rows, dim=-2)[..., :n, :]
    packed = deep_gemm.transform_sf_into_required_layout(
        expanded, n, k, (1, 32), weight.shape[0] if weight.ndim == 3 else None
    )
    return weight, packed


def shared_scale_rows(rows: int, block_m: int, *, device: str | torch.device) -> Tensor:
    """DeepGEMM's shared-expert UTCCP scale-row permutation (upstream test ABI)."""
    source = torch.arange(rows, device=device)
    within = source.remainder(block_m)
    return (
        source.div(block_m, rounding_mode="floor") * ((block_m + 127) // 128 * 128)
        + within.div(128, rounding_mode="floor") * 128
        + within.remainder(32) * 4
        + within.remainder(128).div(32, rounding_mode="floor")
    )


class MegaMoeCollective(LayerCollective):
    """One symmetric workspace per expert layout, shared by serial FFN layers."""

    def __init__(self, group: RankGroup, max_tokens: int) -> None:
        super().__init__()
        self.group, self.max_tokens = group, max_tokens
        self.buffers = {}

    @property
    def allocated(self) -> bool:
        return bool(self.buffers)

    def buffer(
        self, experts: int, topk: int, hidden: int, intermediate: int
    ) -> SymmBuffer:
        import deep_gemm

        key = experts, topk, hidden, intermediate
        if key not in self.buffers:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare mega-MoE workspaces before graph capture")
            self.buffers[key] = deep_gemm.SymmBuffer(
                self.group.handle,
                experts,
                self.max_tokens,
                topk,
                hidden,
                intermediate,
                num_shared_experts=1,
                mma_type="fp8xfp4",
                activation="swiglu",
            )
        return self.buffers[key]

    def close(self) -> None:
        for buffer in self.buffers.values():
            buffer.destroy()
        self.buffers.clear()


class MegaMoe(nn.Module):
    """Gate, sparse EP exchange, routed/shared GEMMs and combine in DeepGEMM."""

    gate_up_weight: Tensor
    gate_up_scale: Tensor
    down_weight: Tensor
    down_scale: Tensor
    shared_gate_up_weight: Tensor
    shared_gate_up_scale: Tensor
    shared_down_weight: Tensor
    shared_down_scale: Tensor

    def __init__(
        self,
        gate: nn.Module,
        collective: MegaMoeCollective,
        quantize: Callable[[Tensor], tuple[Tensor, Tensor]],
        *,
        route: Callable[
            [Tensor, Tensor | None, Tensor | None, tuple[Tensor, Tensor]], None
        ],
        gate_up: tuple[Tensor, Tensor],
        down: tuple[Tensor, Tensor],
        shared_gate_up: tuple[Tensor, Tensor],
        shared_down: tuple[Tensor, Tensor],
        experts: int,
        topk: int,
        hidden: int,
        intermediate: int,
        clamp: float,
    ) -> None:
        super().__init__()
        self.gate, self.collective, self.quantize = gate, collective, quantize
        self.route = route
        self.experts, self.topk, self.dim, self.clamp = experts, topk, hidden, clamp
        for name, pair in (
            ("gate_up", gate_up),
            ("down", down),
            ("shared_gate_up", shared_gate_up),
            ("shared_down", shared_down),
        ):
            for suffix, value in zip(("weight", "scale"), pair, strict=True):
                self.register_buffer(f"{name}_{suffix}", value.detach())
        # Every layer prepares its layout before capture; subsequent forwards
        # select this already allocated workspace without communicator setup.
        self._workspace_key = experts, topk, hidden, intermediate
        collective.buffer(*self._workspace_key)

    def _shared_block_m(self, rows: int) -> int:
        import deep_gemm

        return deep_gemm.get_block_m_for_mega_moe(
            self.collective.group.size,
            self.experts,
            self.workspace.num_max_tokens_per_rank,
            rows,
            self.topk,
            "fp8xfp4",
        )

    def _weights(self, name: str) -> tuple[Tensor, Tensor]:
        return getattr(self, f"{name}_weight"), getattr(self, f"{name}_scale")

    @property
    def workspace(self) -> SymmBuffer:
        return self.collective.buffers[self._workspace_key]

    def fused_norm_cast_kwargs(self, rows: int) -> dict:
        """Let mHC write the routed and shared FP8 inputs in their final layout.

        Matches DeepInfer's v4_modules/mlp.py::fused_norm_cast_kwargs contract.
        The BF16 normalization output remains the routing gate's input.
        """
        self._validate_rows(rows)
        workspace = self.workspace
        # DeepGEMM's native constructor fills these views; destroy() clears them.
        return {
            "y_fp8": workspace.x[:rows],  # ty: ignore[not-subscriptable]
            "y_routed_sf": workspace.x_sf[:rows],  # ty: ignore[not-subscriptable]
            "y_shared_sf": workspace.shared_l1_acts_sf[:rows],
            "shared_sf_block_m": self._shared_block_m(rows),
        }

    def _validate_rows(self, rows: int) -> None:
        if rows > self.collective.max_tokens:
            raise ValueError("mega-MoE rows exceed its prepared workspace capacity")
        if self.collective._capacity is not None and rows > self.collective._capacity:
            raise ValueError("mega-MoE rows exceed the current graph capacity")

    def forward(self, hidden: Tensor, input_ids: Tensor) -> Tensor:
        with torch.cuda.device(hidden.device):
            return self._forward(hidden, input_ids, prepared=False)

    def forward_prepared(self, hidden: Tensor, input_ids: Tensor | None) -> Tensor:
        """Consume the symmetric inputs just written by fused mHC normalization."""
        with torch.cuda.device(hidden.device):
            return self._forward(hidden, input_ids, prepared=True)

    def _forward(
        self, hidden: Tensor, input_ids: Tensor | None, *, prepared: bool
    ) -> Tensor:
        import deep_gemm

        shape = hidden.shape
        hidden = hidden.reshape(-1, self.dim)
        rows, workspace = hidden.shape[0], self.workspace
        self._validate_rows(rows)
        valid = self.collective._valid_rows
        active = (
            None if valid is None else torch.arange(rows, device=hidden.device) < valid
        )
        self.route(
            hidden,
            None if input_ids is None else input_ids.reshape(-1),
            active,
            (workspace.topk_idx[:rows], workspace.topk_weights[:rows]),
        )
        if rows and not prepared:
            values, scales = self.quantize(hidden)
            packed = scales.contiguous().view(torch.int32)
            workspace.x[:rows].copy_(values)  # ty: ignore[not-subscriptable]
            workspace.x_sf[:rows].copy_(packed)  # ty: ignore[not-subscriptable]
            workspace.shared_l1_acts_sf.zero_()
            workspace.shared_l1_acts_sf.index_copy_(
                0,
                shared_scale_rows(
                    rows, self._shared_block_m(rows), device=hidden.device
                ),
                packed,
            )
        output = torch.empty_like(hidden)
        deep_gemm.fp8_fp4_mega_moe(
            output,
            self._weights("gate_up"),
            self._weights("down"),
            workspace,
            shared_l1_weights=self._weights("shared_gate_up"),
            shared_l2_weights=self._weights("shared_down"),
            activation="swiglu",
            activation_clamp=self.clamp,
            fast_math=True,
        )
        return output.reshape(shape)

    def forward_empty(self) -> None:
        weight = self._weights("gate_up")[0]
        self(
            weight.new_empty((0, self.dim), dtype=torch.bfloat16),
            weight.new_empty((0,), dtype=torch.long),
        )
