"""Live DeltaNet state kept on the GPU between steps (see ``live_state.py``)."""

from __future__ import annotations

from resin_models_mlx.live_state import LiveStates as _LiveStates
from resin_models_mlx.qwen3_5.cache import STATE_GROUP, CacheArena
from resin_models_mlx.qwen3_5.model import Qwen35Model


class LiveStates(_LiveStates):
    """Per-request DeltaNet state, indexed into the batch arrays that hold it."""

    def __init__(self, arena: CacheArena, model: Qwen35Model) -> None:
        super().__init__(arena, model.initial_state, STATE_GROUP)


__all__ = ["LiveStates"]
