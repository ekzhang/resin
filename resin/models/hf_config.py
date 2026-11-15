"""Small dataclass helpers for checkpoint geometry and its validation."""

from collections.abc import Iterator
from dataclasses import MISSING, fields
from functools import cache
from typing import Annotated, Any, TypeVar, cast, get_args, get_type_hints

Config = TypeVar("Config")


PositiveInt = Annotated[int, "positive"]
PositiveFloat = Annotated[float, "positive"]


@cache
def _positive_fields(cls: type) -> tuple[str, ...]:
    return tuple(
        name
        for name, kind in get_type_hints(cls, include_extras=True).items()
        if "positive" in get_args(kind)
    )


def invalid_dimensions(config: Any, *, strict: bool = False) -> Iterator[str]:
    for name in _positive_fields(type(config)):
        value = getattr(config, name)
        if (strict and type(value) is not int) or value <= 0:
            yield name


@cache
def _scalar_types(cls: type) -> dict[str, type]:
    return {
        name: kind
        for name, kind in get_type_hints(cls).items()
        if kind in (int, float, bool, str)
    }


def parse_hf_config(
    cls: type[Config], values: dict[str, Any], *, coerce: bool = False, **overrides: Any
) -> Config:
    """Read declared fields; callers supply aliases and nested/sequence values."""
    converters = _scalar_types(cls) if coerce else {}
    for item in fields(cast(Any, cls)):
        if item.name in overrides or (
            item.name not in values
            and (item.default is not MISSING or item.default_factory is not MISSING)
        ):
            continue
        value = values[item.name]
        if convert := converters.get(item.name):
            value = convert(value)
        overrides[item.name] = value
    return cls(**overrides)


def require_options(
    values: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    mismatched = {
        key: (values.get(key), value)
        for key, value in expected.items()
        if values.get(key) != value
    }
    if mismatched:
        raise ValueError(f"unsupported {label} options: {mismatched}")
