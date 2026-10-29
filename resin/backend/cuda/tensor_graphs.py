"""Fixed-address tensor operations around eager, request-dependent work."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import cast

import torch

from resin.backend.collectives import layer_collectives
from resin.backend.cuda.graphs import CudaGraphMode, CudaGraphStep, GraphShape


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Trailing dimensions and dtype of a tensor with a leading token axis."""

    shape: tuple[int, ...]
    dtype: torch.dtype = torch.bfloat16


@dataclass(frozen=True, slots=True)
class CudaGraphTensorRegion:
    """A row-independent operation; all request metadata stays with its caller.

    Regions exchanging tokens across attention-DP replicas require coordinated
    capacities; local projections and tensor-parallel reductions use local rows.
    """

    name: str
    operation: Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]
    inputs: tuple[TensorSpec, ...]
    output: TensorSpec | tuple[TensorSpec, ...]


class PrefillTensorGraphs:
    """Bucket tensor operations using the runner's existing graph executor.

    Matching layouts share buffers across layers. Results are views of these
    reusable buffers and must be consumed before the next matching operation.
    No page table, request count, or CP ownership is captured by this class.
    Coordinated regions use the runner's agreed capacity and valid-row scalar;
    local projection regions choose their own buckets and skip empty inputs.
    """

    def __init__(
        self,
        runner,
        regions: tuple[CudaGraphTensorRegion, ...],
        *,
        max_tokens: int,
        pad_up_to: int = 128,
        capture_context: Callable = nullcontext,
        mode: str = "prefill_attention",
        coordinated: bool = False,
    ) -> None:
        self.runner, self.regions, self.mode = runner, tuple(regions), mode
        self.pad_up_to, self.capture_context = pad_up_to, capture_context
        self.coordinated = coordinated
        self.buffers = {}
        self.buckets = (
            tuple(
                sorted(
                    {min(2**p, max_tokens) for p in range(max_tokens.bit_length() + 1)}
                )
            )
            if runner.cuda_graph_prefill == "breakable"
            else ()
        )
        runner._require_cuda_graph_executor().add_mode(
            mode,
            CudaGraphMode(
                tuple((n, i + 1) for i in range(len(regions)) for n in self.buckets),
                self.build,
                lambda shape: self._buffers(*shape)[0],
            ),
        )

    def _buffers(
        self, tokens: int, region: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | tuple[torch.Tensor, ...]]:
        spec = self.regions[region - 1]
        key = tokens, spec.inputs, spec.output
        if key not in self.buffers:

            def allocate(layout: TensorSpec) -> torch.Tensor:
                return torch.zeros(
                    (tokens, *layout.shape),
                    dtype=layout.dtype,
                    device=self.runner.device,
                )

            inputs = {str(i): allocate(layout) for i, layout in enumerate(spec.inputs)}
            if (
                self.coordinated
                and getattr(self.runner, "_parallel_graph_groups", None) is not None
                and any(layer_collectives(self.runner._parallel_graph_model))
            ):
                inputs["ep_valid_rows"] = torch.tensor(
                    tokens, dtype=torch.int32, device=self.runner.device
                )
            output = (
                tuple(allocate(layout) for layout in spec.output)
                if isinstance(spec.output, tuple)
                else allocate(spec.output)
            )
            self.buffers[key] = inputs, output
        return self.buffers[key]

    def build(self, shape: GraphShape) -> CudaGraphStep:
        inputs, output = self._buffers(*shape)
        spec = self.regions[shape[1] - 1]

        def launch() -> None:
            with (
                self.runner._parallel_graph_capacity(
                    shape[0], valid_rows=inputs.get("ep_valid_rows")
                )
                if self.coordinated
                else nullcontext(),
                self.capture_context(),
            ):
                result = spec.operation(
                    *(inputs[str(i)] for i in range(len(spec.inputs)))
                )
                if isinstance(spec.output, tuple):
                    for target, value in zip(output, result, strict=True):
                        target.copy_(value)
                else:
                    cast(torch.Tensor, output).copy_(cast(torch.Tensor, result))

        return CudaGraphStep(inputs, output, launch)

    def invoke(
        self, region: int, *values: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        spec = self.regions[region]
        if not self.runner._prefill_graph_active:
            return spec.operation(*values)
        if len(values) != len(spec.inputs):
            raise ValueError(
                "tensor graph region input count does not match its layout"
            )
        tokens = values[0].shape[0]
        if not tokens and not self.coordinated:
            return spec.operation(*values)
        distributed = self.coordinated and getattr(
            self.runner, "_distributed_prefill_scope", False
        )
        bucket = (
            self.runner._distributed_prefill_region_buckets.get(
                region, self.runner._distributed_prefill_bucket
            )
            if distributed
            else next((n for n in self.buckets if n >= tokens), None)
        )
        if distributed and bucket is not None and tokens > bucket:
            raise ValueError("prefill region exceeds its agreed graph token capacity")
        reason = None
        if bucket is None or any(
            value.shape != (tokens, *layout.shape) or value.dtype != layout.dtype
            for value, layout in zip(values, spec.inputs, strict=True)
        ):
            reason = "tokens_or_layout"
        elif not distributed and tokens > self.pad_up_to and tokens != bucket:
            reason = "padding"
        if reason is not None:
            self.runner._cuda_graph_fallback(self.mode, reason)
            return spec.operation(*values)
        inputs, _ = self._buffers(cast(int, bucket), region + 1)
        if "ep_valid_rows" in inputs:
            inputs["ep_valid_rows"].fill_(tokens)
        for index, value in enumerate(values):
            target = inputs[str(index)]
            if tokens == bucket:
                target.copy_(value)
            else:
                target[:tokens].copy_(value)
                target[tokens:].zero_()
        output = self.runner._run_cuda_graph(self.mode, (bucket, region + 1), inputs)
        if tokens == bucket:
            return output
        if isinstance(spec.output, tuple):
            return tuple(value[:tokens] for value in output)
        return output[:tokens]

    @property
    def callbacks(
        self,
    ) -> tuple[Callable[..., torch.Tensor | tuple[torch.Tensor, ...]], ...]:
        return tuple(
            lambda *values, region=region: self.invoke(region, *values)
            for region in range(len(self.regions))
        )
