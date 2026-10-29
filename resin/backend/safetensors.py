"""Validated indexed-safetensor access shared by checkpoint loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from torch import Tensor


def indexed_tensor_files(root: Path, *, require_index: bool = False) -> dict[str, Path]:
    """Resolve every tensor name to one validated local safetensor file."""

    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        from safetensors import safe_open

        if require_index:
            raise FileNotFoundError(f"missing checkpoint index: {index_path}")
        model_file = root / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(f"missing checkpoint weights: {model_file}")
        with safe_open(str(model_file), framework="pt", device="cpu") as handle:
            return {key: model_file for key in handle.keys()}

    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {index_path}: {error}") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path.name} must contain a non-empty weight_map")

    tensor_files = {}
    validated_files: set[Path] = set()
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError("checkpoint index contains an invalid tensor name")
        if not isinstance(filename, str):
            raise ValueError("checkpoint index contains an invalid shard name")
        relative = Path(filename)
        if (
            not filename
            or relative.is_absolute()
            or len(relative.parts) != 1
            or relative.suffix != ".safetensors"
        ):
            raise ValueError(f"invalid checkpoint shard name: {filename!r}")
        path = root / relative
        if path not in validated_files:
            if not path.is_file():
                raise FileNotFoundError(f"missing checkpoint shard: {path}")
            validated_files.add(path)
        tensor_files[key] = path
    return tensor_files


class SafetensorReader:
    """Keep at most one memory-mapped checkpoint shard open."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._context: Any | None = None
        self._handle: Any | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._path = None
        self._context = None
        self._handle = None

    def _open(self, path: Path):
        if path != self._path:
            from safetensors import safe_open

            self.close()
            self._context = safe_open(str(path), framework="pt", device="cpu")
            self._handle = self._context.__enter__()
            self._path = path
        assert self._handle is not None
        return self._handle

    def read(self, path: Path, key: str) -> Tensor:
        return self._open(path).get_tensor(key)

    def read_slice(
        self,
        path: Path,
        key: str,
        slices: tuple[slice, ...],
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Read bounded contiguous axis intervals without materializing the rest.

        ``shape`` describes the full checkpoint tensor, before slicing. Reject
        clipped or stepped intervals rather than silently changing shard geometry.
        The returned CPU tensor retains its storage after this reader changes files.
        """
        source = self._open(path).get_slice(key)
        actual = tuple(source.get_shape())
        if shape is not None and actual != tuple(shape):
            raise ValueError(f"{key}: expected {shape}, found {actual}")
        if not isinstance(slices, tuple) or len(slices) != len(actual):
            raise ValueError("slices must specify one interval per tensor axis")
        normalized = []
        for interval, size in zip(slices, actual, strict=True):
            if not isinstance(interval, slice) or interval.step not in (None, 1):
                raise ValueError("tensor slices must be contiguous intervals")
            start = 0 if interval.start is None else interval.start
            stop = size if interval.stop is None else interval.stop
            if (
                type(start) is not int
                or type(stop) is not int
                or not 0 <= start < stop <= size
            ):
                raise ValueError(f"slice [{start}:{stop}] is outside axis size {size}")
            normalized.append(slice(start, stop))
        return source[tuple(normalized)]


__all__ = ["SafetensorReader", "indexed_tensor_files"]
