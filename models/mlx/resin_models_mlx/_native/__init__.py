"""Native Engine-private cache resource, built against the pinned MLX ABI.

``read_epoch`` is an internal transient view, never an immutable snapshot.
Pass all prior computed-reader outputs to the next writer's ``after=`` argument.
Use ``snapshot`` for independent values; subsequent writes automatically depend
on pending snapshots. Older epochs do not preserve unversioned private views.
Scatter rows must be unique;
out-of-range device indices are ignored. Operations do not support transforms.
"""

import mlx.core  # noqa: F401 -- register MLX types and load libmlx for editable builds

from resin_models_mlx._native_cache import CacheStorage

__all__ = ["CacheStorage"]
