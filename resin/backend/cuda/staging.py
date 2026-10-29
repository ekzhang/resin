"""Pinned host and asynchronous device tensor staging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch


class StagingError(ValueError):
    """Host values do not fit the requested fixed device buffers."""


class TensorStager:
    """Own one mutable generation of pinned host tensors.

    Reusing a field requires its previous asynchronous copy to have finished.
    Use :class:`TensorStagerPool` when enqueuing successive device batches.
    """

    def __init__(self, device: str | torch.device) -> None:
        self.target_device = device
        self.host_tensors: dict[str, torch.Tensor] = {}

    def host(self, name: str, values: object, dtype: torch.dtype) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            values = np.asarray(values)
        tensor = self._host_buffer(name, values.shape, dtype)
        if isinstance(values, torch.Tensor):
            # CPU-to-CPU copying also supports dtypes NumPy cannot represent.
            tensor.copy_(values)
        else:
            np.copyto(tensor.numpy(), values, casting="unsafe")
        return tensor

    def _host_buffer(
        self, name: str, shape: Sequence[int], dtype: torch.dtype
    ) -> torch.Tensor:
        tensor = self.host_tensors.get(name)
        if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
            # These mutable buffers outlive the caller's inference-mode scope.
            with torch.inference_mode(False):
                tensor = torch.empty(
                    shape,
                    dtype=dtype,
                    device="cpu",
                    pin_memory=True,
                )
            self.host_tensors[name] = tensor
        return tensor

    def device(self, name: str, values: object, dtype: torch.dtype) -> torch.Tensor:
        return self.host(name, values, dtype).to(
            device=self.target_device, non_blocking=True
        )

    def output(self, name: str, source: torch.Tensor) -> torch.Tensor:
        tensor = self._host_buffer(name, source.shape, source.dtype)
        tensor.copy_(source, non_blocking=True)
        return tensor

    def tensors(
        self, fields: Mapping[str, tuple[object, torch.dtype]], *, device: bool
    ) -> dict[str, torch.Tensor]:
        stage = self.device if device else self.host
        return {
            name: stage(name, values, dtype) for name, (values, dtype) in fields.items()
        }

    def copy_into(
        self,
        targets: Mapping[str, torch.Tensor],
        values: Mapping[str, object],
    ) -> None:
        if targets is values:
            return
        if targets.keys() != values.keys():
            missing = targets.keys() - values.keys()
            extra = values.keys() - targets.keys()
            raise StagingError(
                f"staged inputs do not match buffers (missing={sorted(missing)}, "
                f"extra={sorted(extra)})"
            )
        staged = []
        try:
            for name, target in targets.items():
                value = values[name]
                if isinstance(value, getattr(torch, "Tensor", ())):
                    if value.shape != target.shape or value.dtype != target.dtype:
                        raise StagingError(
                            f"device input {name!r} has shape {tuple(value.shape)} and dtype {value.dtype}; "
                            f"expected shape {tuple(target.shape)} and dtype {target.dtype}"
                        )
                    source = (
                        self.host(name, value, target.dtype)
                        if value.device.type == "cpu" and target.device.type != "cpu"
                        else value
                    )
                else:
                    source = self.host(name, value, target.dtype).view(target.shape)
                staged.append((name, target, source))
        except StagingError:
            raise
        except Exception as error:
            raise StagingError(
                f"input {name!r} does not fit graph buffer with shape "
                f"{tuple(target.shape)} and dtype {target.dtype}: {error}"
            ) from error
        for name, target, source in staged:
            target.copy_(source, non_blocking=True)


@dataclass(slots=True)
class _PendingStaging:
    staging: TensorStager
    # None quarantines a generation if recording its completion event failed.
    event: torch.cuda.Event | None = None


class TensorStagerPool:
    """Reuse pinned inputs only after the GPU has consumed their generation.

    Each copy batch records one event after its H2D copies. Outstanding batches
    retain separate buffers at arbitrary queue depth; completed batches reuse
    their allocations without blocking the submitting CPU thread.
    """

    def __init__(self, device: str | torch.device) -> None:
        self.target_device = device
        self._available: list[TensorStager] = []
        self._pending: list[_PendingStaging] = []

    def copy_into(
        self,
        targets: Mapping[str, torch.Tensor],
        values: Mapping[str, object],
    ) -> None:
        if targets is values:
            return
        completed = []
        outstanding = []
        for item in self._pending:
            if item.event is not None and item.event.query():
                completed.append(item.staging)
            else:
                outstanding.append(item)
        self._pending = outstanding
        self._available.extend(completed)
        staging = (
            self._available.pop()
            if self._available
            else TensorStager(self.target_device)
        )
        try:
            staging.copy_into(targets, values)
        finally:
            if staging.host_tensors:
                # Retain the source even when a later copy or event record fails:
                # an earlier H2D operation may already be reading this generation.
                pending = _PendingStaging(staging)
                self._pending.append(pending)
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.target_device))
                pending.event = event
            else:
                self._available.append(staging)
