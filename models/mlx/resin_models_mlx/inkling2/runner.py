"""Inkling 2 target-only runner on Apple Silicon over Engine-owned MLX slabs."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from resin.backend.plans import PagedCachePlan
from resin.models.inkling2.config import Inkling2Config
from resin_models_mlx.cache_arena import KVHistory
from resin_models_mlx.inkling2.attention import decode_attention, paged_prefill
from resin_models_mlx.inkling2.cache import (
    GLOBAL_GROUP,
    LOCAL_GROUP,
    STATE_GROUP,
    CacheArena,
    CacheGeometry,
)
from resin_models_mlx.inkling2.checkpoint import load_model
from resin_models_mlx.inkling2.model import Inkling2Model, Inkling2Weights, logits
from resin_models_mlx.live_state import LiveStates
from resin_models_mlx.metal import device_array
from resin_models_mlx.runner import (
    DEFAULT_WORKSPACE_BYTES,
    MlxRunner,
    ModelStep,
    context_limit,
)

if TYPE_CHECKING:
    from resin import BatchPlan


class _StepStore:
    """Attention over one step's retained history plus its own new keys.

    The step's keys and values are kept for the runner to write after every
    reader of the arena view has been ordered.
    """

    def __init__(
        self,
        arena: CacheArena,
        plan: PagedCachePlan,
        rows: np.ndarray,
        contexts: np.ndarray,
        window: int,
    ) -> None:
        self.geometry = arena.geometry
        self.history: dict[int, KVHistory] = {}
        self.paged: dict[int, tuple[mx.array, mx.array, int]] = {}
        self.visits: dict[int, dict[int, tuple[mx.array, mx.array]]] = {
            GLOBAL_GROUP: {},
            LOCAL_GROUP: {},
        }
        for group in self.visits:
            firsts = (
                np.maximum(contexts - (window - 1), 0)
                if group == LOCAL_GROUP
                else np.zeros_like(contexts)
            )
            history = arena.retained_pages(plan, group, rows, contexts, earliest=firsts)
            self.history[group] = history
            meta = np.stack([contexts, firsts, history.starts], axis=1)
            self.paged[group] = (
                device_array(history.bases),
                device_array(meta),
                int(np.max(contexts + 1 - firsts)),
            )

    def attend(self, local, slot, q, k, v, bias, scale, window):
        group = LOCAL_GROUP if local else GLOBAL_GROUP
        self.visits[group][slot] = (k, v)
        history = self.history[group]
        pages, meta, longest = self.paged[group]
        layer = slot * history.layer_stride
        if q.shape[1] == 1:
            output, lse = decode_attention(
                q[:, 0],
                k[:, 0],
                v[:, 0],
                history.units,
                pages,
                meta,
                bias[:, 0],
                layer,
                scale,
                longest,
            )
            return output[:, None], lse[:, None]
        return paged_prefill(
            q,
            k,
            v,
            bias,
            history.units,
            pages,
            meta,
            layer,
            scale,
            window if local else None,
        )

    @property
    def writes(self):
        groups = {}
        for group, visits in self.visits.items():
            # Preserve slot order even for checkpoints that list local layers
            # out of execution order.
            ordered = (visits[slot] for slot in range(self.geometry.layers(group)))
            keys, values = zip(*ordered, strict=True)
            groups[group] = keys, values
        return groups


class Inkling2MlxRunner(MlxRunner[CacheArena]):
    """Inkling 2 target with paged global/local attention and live conv history."""

    model_name = "Inkling 2"

    def __init__(
        self,
        model: str | Path | tuple[Inkling2Config, Inkling2Weights],
        *,
        memory_utilization: float = 0.85,
        cache_slab_capacity: int | None = None,
        max_seq_len: int | None = None,
        max_batch_size: int = 8,
        prefill_chunk_size: int = 8192,
        workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    ) -> None:
        if workspace_bytes < 0:
            raise ValueError("workspace_bytes must be non-negative")
        if isinstance(model, tuple):
            config, weights = model
            self.model_path = None
        else:
            self.model_path = Path(model)
            config, weights = load_model(self.model_path)
        self.config = config
        self._model = Inkling2Model(config, weights)
        self.max_seq_len = context_limit(max_seq_len, config.max_position_embeddings)
        self.benchmark_metadata = {
            "quant": "checkpoint-nvfp4-mxfp8-experts",
            "weight_source": "prequantized_checkpoint",
            # Prefill chunks of 512 or more tokens instead run exactly
            # dequantized FP32 GEMMs per expert segment (model.SEGMENTED_ROUTES).
            "expert_backend": "mlx-gather-qmm",
            "graphs": {"prefill": "disabled", "decode": "disabled"},
        }
        # Logits cover the real vocabulary; embedding rows beyond it are padding.
        self.vocab_size = config.unpadded_vocab_size
        self.eos_token_id = config.eos_token_id
        self.cache_geometry = CacheGeometry.from_config(config, prefill_chunk_size)
        geometry = self.cache_geometry
        slab_capacity = self._slab_capacity(
            geometry,
            memory_utilization,
            workspace_bytes,
            max_batch_size,
            cache_slab_capacity,
        )
        # Room for every running request at full context, and as much again
        # for reusable prefixes; beyond that, memory is better left unwired.
        wanted = 2 * max_batch_size * geometry.sequence_slabs(self.max_seq_len) + 16
        slab_capacity = min(slab_capacity, wanted)
        if slab_capacity < geometry.sequence_slabs(self.max_seq_len):
            raise ValueError(
                f"the Inkling 2 cache holds {slab_capacity} slabs, fewer than one "
                f"{self.max_seq_len}-token request needs; lower max_seq_len"
            )
        cache = CacheArena(geometry.layout(slab_capacity), geometry)
        super().__init__(
            cache,
            LiveStates(cache, self._model.initial_state, STATE_GROUP),
            partial(logits, weights=weights, config=config),
            prefill_chunk_size,
        )

    def _step(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        tokens: np.ndarray,
        contexts: np.ndarray,
        token_ids: mx.array,
    ) -> ModelStep:
        store = _StepStore(
            self._cache,
            plan,
            rows,
            contexts,
            self.config.sliding_window_size,
        )
        hidden, state = self._model.hidden(
            token_ids,
            mx.array(plan.batch.positions[tokens].astype(np.int32)),
            self._states.gather(plan, rows),
            store,
        )
        return ModelStep(hidden, state, store.writes)

    def _cache_plan(self, plan: BatchPlan) -> PagedCachePlan:
        if any(media is not None for media in plan.media):
            raise ValueError("the Inkling 2 MLX runner serves text only")
        return super()._cache_plan(plan)


__all__ = ["Inkling2MlxRunner"]
