"""Fixed token capacities and coordinated buffer lifetime for distributed layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from contextlib import ExitStack, contextmanager

import torch
from torch import Tensor


class LayerCollective(ABC):
    """A distributed layer's fixed-capacity context and shared buffer lifetime."""

    def __init__(self) -> None:
        self._capacity: int | None = None
        self._valid_rows: Tensor | None = None

    @property
    @abstractmethod
    def allocated(self) -> bool:
        """Whether this operation owns shared buffers requiring coordinated close."""

    @abstractmethod
    def close(self) -> None:
        """Release shared buffers; repeated calls must be harmless."""

    @contextmanager
    def fixed_capacity(
        self, rows: int, *, valid_rows: Tensor | None = None
    ) -> Generator[None, None, None]:
        """Select common token capacity and optionally a device scalar of valid rows."""
        if type(rows) is not int or rows <= 0:
            raise ValueError("collective capacity must be a positive integer")
        if valid_rows is not None and (
            valid_rows.numel() != 1
            or valid_rows.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("valid collective rows must be one integer tensor value")
        previous, self._capacity = self._capacity, rows
        previous_valid, self._valid_rows = self._valid_rows, valid_rows
        try:
            yield
        finally:
            self._capacity = previous
            self._valid_rows = previous_valid


@contextmanager
def fixed_layer_capacity(
    model: torch.nn.Module, capacity: int, *, valid_rows: Tensor | None = None
) -> Generator[None, None, None]:
    """Select fixed-capacity layer collectives throughout a captured model."""

    with ExitStack() as contexts:
        for collective in layer_collectives(model):
            contexts.enter_context(
                collective.fixed_capacity(capacity, valid_rows=valid_rows)
            )
        yield


def layer_collectives(model: torch.nn.Module) -> Iterator[LayerCollective]:
    """Visit each layer collective once, including collectives that are modules."""
    seen = set()
    for module in model.modules():
        for value in (module, *vars(module).values()):
            if isinstance(value, LayerCollective) and id(value) not in seen:
                seen.add(id(value))
                yield value


def close_layer_collectives(model: torch.nn.Module) -> None:
    """Release shared workspaces after all ranks have destroyed their graphs."""
    for collective in layer_collectives(model):
        collective.close()
