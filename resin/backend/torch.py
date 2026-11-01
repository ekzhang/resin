"""Small PyTorch scopes shared by concrete model backends."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import torch


@contextmanager
def use_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


__all__ = ["use_default_dtype"]
