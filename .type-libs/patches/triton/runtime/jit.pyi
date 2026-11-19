# Triton 3.9 infers invariant wrapper self types that ty cannot specialize.
# Device calls can also pass bare literals to tl.constexpr parameters. Keep the
# decorated signature for launch proxies and the return type for device calls.
from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar, overload

_F = TypeVar("_F", bound=Callable[..., Any])
_F_co = TypeVar("_F_co", bound=Callable[..., Any], covariant=True)
_R = TypeVar("_R")

class KernelInterface(Generic[_F_co]):
    def __getitem__(self, grid: Any) -> _F_co: ...
    def run(self, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any: ...

class JITCallable:
    @property
    def fn(self) -> Callable[..., Any]: ...

class JITFunction(JITCallable, KernelInterface[_F_co]):
    device_caches: dict[int, tuple[dict[str, Any], *tuple[Any, ...]]]

    def __call__(
        self: JITFunction[Callable[..., _R]], *args: Any, **kwargs: Any
    ) -> _R: ...

class ConstexprFunction(JITCallable, Generic[_F_co]):
    def __call__(
        self: ConstexprFunction[Callable[..., _R]], *args: Any, **kwargs: Any
    ) -> _R: ...

@overload
def jit(fn: _F) -> JITFunction[_F]: ...
@overload
def jit(
    *,
    version: Any = ...,
    repr: Callable[..., Any] | None = ...,
    launch_metadata: Callable[..., Any] | None = ...,
    do_not_specialize: Iterable[int | str] | None = ...,
    do_not_specialize_on_alignment: Iterable[int | str] | None = ...,
    debug: bool | None = ...,
    noinline: bool | None = ...,
) -> Callable[[_F], JITFunction[_F]]: ...
def constexpr_function(fn: _F) -> ConstexprFunction[_F]: ...
