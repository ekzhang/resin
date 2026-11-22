from collections.abc import Sequence

import mlx.core

class CacheStorage:
    def __init__(self, byte_shape: Sequence[int]) -> None: ...
    def initial_epoch(self) -> mlx.core.array: ...
    def read_epoch(self, epoch: mlx.core.array) -> mlx.core.array: ...
    def scatter_rows(
        self,
        epoch: mlx.core.array,
        row_indices: mlx.core.array,
        updates: mlx.core.array,
        *,
        row_bytes: int,
        after: Sequence[mlx.core.array] = [],
    ) -> mlx.core.array: ...
    def write_slice(
        self,
        epoch: mlx.core.array,
        byte_start: int,
        updates: mlx.core.array,
        *,
        after: Sequence[mlx.core.array] = [],
    ) -> mlx.core.array: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def buffer_identity(self) -> int: ...

def _paged_attention_partials(
    queries: mlx.core.array,
    current_keys: mlx.core.array,
    current_values: mlx.core.array,
    units: mlx.core.array,
    pages: mlx.core.array,
    lengths: mlx.core.array,
    layer: int,
    scale: float,
) -> list[mlx.core.array]: ...
def _warm_paged_attention() -> None: ...
