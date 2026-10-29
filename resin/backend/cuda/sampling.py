"""Shared graph-safe fused sampling for CUDA model runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from resin.backend.sampling import SAMPLING_OFFSET_STRIDE, batch_sampling_inputs

if TYPE_CHECKING:
    from resin import BatchPlan


@dataclass(frozen=True, slots=True)
class SamplingTensors:
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    seeds: torch.Tensor
    offsets: torch.Tensor
    stochastic_mask: torch.Tensor


class FusedSampler:
    """Temperature, top-k, and top-p sampling with request-local Philox streams.

    Runtime device parameters may change between CUDA graph replays. Unfiltered
    rows sample directly from logits. Top-p uses the full temperature-scaled
    distribution, followed by top-k; filtered rows retain every token tied at
    either cutoff. Greedy rows use the lowest argmax token ID.
    """

    def __init__(self) -> None:
        from resin.backend.cuda.filtered_sampling import (
            prepare_filters,
            validate_filtered_inputs,
        )
        from resin.backend.cuda.logits_sampling import sample_logits

        self._sample_logits = sample_logits
        self._prepare_filters = prepare_filters
        self._validate_inputs = validate_filtered_inputs

    @staticmethod
    def host_inputs(
        plan: BatchPlan, vocab_size: int, *, offset_delta: int = 0
    ) -> dict[str, tuple]:
        if offset_delta < 0:
            raise ValueError("sampling offset delta cannot be negative")
        values = batch_sampling_inputs(plan, vocab_size)
        return {
            "sampling_temperatures": tuple(
                temperature if temperature > 0.0 else 1.0
                for temperature in values["temperatures"]
            ),
            "sampling_top_ps": values["top_ps"],
            "sampling_top_ks": values["top_ks"],
            "sampling_seeds": values["seeds"],
            "sampling_offsets": tuple(
                offset + offset_delta for offset in values["offsets"]
            ),
            "sampling_stochastic_mask": tuple(
                should_sample and temperature > 0.0
                for temperature, should_sample in zip(
                    values["temperatures"], values["sampling_mask"], strict=True
                )
            ),
        }

    def empty(self, batch_size: int, device: str | torch.device) -> SamplingTensors:
        return SamplingTensors(
            temperatures=torch.empty(batch_size, dtype=torch.float32, device=device),
            top_ps=torch.empty(batch_size, dtype=torch.float32, device=device),
            top_ks=torch.empty(batch_size, dtype=torch.int32, device=device),
            seeds=torch.empty(batch_size, dtype=torch.int64, device=device),
            offsets=torch.empty(batch_size, dtype=torch.int64, device=device),
            stochastic_mask=torch.empty(batch_size, dtype=torch.bool, device=device),
        )

    @staticmethod
    def graph_inputs(tensors: SamplingTensors) -> dict[str, torch.Tensor]:
        return {
            "sampling_temperatures": tensors.temperatures,
            "sampling_top_ps": tensors.top_ps,
            "sampling_top_ks": tensors.top_ks,
            "sampling_seeds": tensors.seeds,
            "sampling_offsets": tensors.offsets,
            "sampling_stochastic_mask": tensors.stochastic_mask,
        }

    def sample_tensors(
        self,
        logits: torch.Tensor,
        tensors: SamplingTensors,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(
            logits,
            tensors.temperatures,
            tensors.top_ks,
            tensors.top_ps,
            tensors.seeds,
            tensors.offsets,
            tensors.stochastic_mask,
            out,
        )
        filters = self._prepare_filters(
            logits,
            tensors.temperatures,
            tensors.top_ks,
            tensors.top_ps,
            tensors.stochastic_mask,
        )
        return self._sample_logits(
            logits,
            tensors.temperatures,
            tensors.seeds,
            tensors.offsets,
            tensors.stochastic_mask,
            filter_state=filters,
            out=out,
        )

    def sample_block(
        self,
        logits: torch.Tensor,
        tensors: SamplingTensors,
        *,
        out: torch.Tensor | None = None,
        start_step: int = 0,
    ) -> torch.Tensor:
        """Sample all [request, verified position, vocabulary] logits together.

        Each request keeps its seed and policy. Column i uses its original
        offset plus (start_step + i) * SAMPLING_OFFSET_STRIDE. Metadata is
        expanded in one device launch; no input or RNG state is mutated.
        The first two logits dimensions must flatten as a view, including
        padded or strided vocabulary storage. No vocabulary copy is made.
        """
        from resin.backend.cuda.logits_sampling import _check_no_overlap
        from resin.backend.cuda.sampling_blocks import expand_sampling

        if type(start_step) is not int or start_step < 0:
            raise ValueError("sampling start_step must be a nonnegative integer")
        if logits.ndim != 3 or logits.shape[1] < 1:
            raise ValueError("block logits must have shape [rows, width>=1, vocab]")
        rows, width, vocab = logits.shape
        self._validate_inputs(
            logits[:, 0, :],
            tensors.temperatures,
            tensors.top_ks,
            tensors.top_ps,
            tensors.seeds,
            tensors.offsets,
            tensors.stochastic_mask,
        )
        try:
            flat_logits = logits.view(rows * width, vocab)
        except RuntimeError as error:
            raise ValueError("block logits must flatten without copying") from error
        flat_out = None
        if out is not None:
            if (
                out.shape != (rows, width)
                or out.dtype != torch.int64
                or out.device != logits.device
                or not out.is_contiguous()
            ):
                raise ValueError(
                    "block output must be contiguous CUDA int64 [rows, width]"
                )
            _check_no_overlap(
                (
                    ("logits", logits),
                    ("temperatures", tensors.temperatures),
                    ("top_ps", tensors.top_ps),
                    ("top_ks", tensors.top_ks),
                    ("seeds", tensors.seeds),
                    ("offsets", tensors.offsets),
                    ("stochastic_mask", tensors.stochastic_mask),
                ),
                (),
                out,
            )
            flat_out = out.view(rows * width)
        expanded = expand_sampling(tensors, width, start_step)
        result = self.sample_tensors(flat_logits, expanded, out=flat_out)
        return out if out is not None else result.view(rows, width)

    def __call__(
        self,
        logits: torch.Tensor,
        plan: BatchPlan | None = None,
        *,
        out: torch.Tensor | None = None,
        offset_delta: int = 0,
    ) -> torch.Tensor:
        if plan is None:
            result = logits.argmax(dim=-1)
            if out is None:
                return result
            out.copy_(result)
            return out
        if logits.ndim not in (2, 3) or (logits.ndim == 3 and logits.shape[1] < 1):
            raise ValueError(
                "sampling logits must have shape [batch, vocab] or [batch, width, vocab]"
            )
        values = batch_sampling_inputs(plan, logits.shape[-1])
        if logits.shape[0] != len(values["temperatures"]):
            raise ValueError("sampling logits must contain one row per plan sequence")

        if not any(
            should_sample and temperature > 0.0
            for temperature, should_sample in zip(
                values["temperatures"], values["sampling_mask"], strict=True
            )
        ):
            result = logits.argmax(dim=-1)
        else:
            device = logits.device
            host = self.host_inputs(plan, logits.shape[-1], offset_delta=offset_delta)
            tensors = SamplingTensors(
                temperatures=torch.tensor(
                    host["sampling_temperatures"],
                    dtype=torch.float32,
                    device=device,
                ),
                top_ps=torch.tensor(
                    host["sampling_top_ps"], dtype=torch.float32, device=device
                ),
                top_ks=torch.tensor(
                    host["sampling_top_ks"], dtype=torch.int32, device=device
                ),
                seeds=torch.tensor(
                    host["sampling_seeds"], dtype=torch.int64, device=device
                ),
                offsets=torch.tensor(
                    host["sampling_offsets"], dtype=torch.int64, device=device
                ),
                stochastic_mask=torch.tensor(
                    host["sampling_stochastic_mask"],
                    dtype=torch.bool,
                    device=device,
                ),
            )
            result = (
                self.sample_block(logits, tensors)
                if logits.ndim == 3
                else self.sample_tensors(logits, tensors)
            )

        if out is None:
            return result
        out.copy_(result)
        return out


__all__ = [
    "FusedSampler",
    "SAMPLING_OFFSET_STRIDE",
    "SamplingTensors",
]
