"""Qwen3.5 runner on Apple Silicon over Engine-owned MLX slabs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from resin import BatchPhase
from resin.backend.plans import PagedCachePlan
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_mlx.live_state import LiveStates
from resin_models_mlx.qwen3_5.cache import (
    KV_GROUP,
    STATE_GROUP,
    CacheArena,
    CacheGeometry,
)
from resin_models_mlx.qwen3_5.checkpoint import checkpoint_quantization, load_model
from resin_models_mlx.qwen3_5.model import Qwen35Model
from resin_models_mlx.qwen3_5.speculation import MtpDecode, SpeculativeFeedback
from resin_models_mlx.runner import (
    DEFAULT_WORKSPACE_BYTES,
    MlxCompletion,
    MlxRunner,
    ModelStep,
    context_limit,
)

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput
    from resin_models_mlx.runner import _Launch


class Qwen35MlxRunner(MlxRunner[CacheArena]):
    """Qwen3.5 runner with paged attention, DeltaNet state and optional vision."""

    model_name = "Qwen3.5"

    def __init__(
        self,
        model: str | Path | Qwen35Model,
        *,
        weight_bits: int | None = 8,
        group_size: int = 64,
        memory_utilization: float = 0.85,
        cache_slab_capacity: int | None = None,
        max_seq_len: int | None = None,
        max_batch_size: int = 16,
        prefill_chunk_size: int = 8192,
        workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
        include_vision: bool = True,
        compile_vision: bool = True,
        speculative_draft_tokens: int = 0,
    ) -> None:
        if type(prefill_chunk_size) is not int or prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be a positive integer")
        self._media = None
        if workspace_bytes < 0:
            raise ValueError("workspace_bytes must be non-negative")
        if type(speculative_draft_tokens) is not int or speculative_draft_tokens < 0:
            raise ValueError("speculative_draft_tokens must be a non-negative integer")
        self.speculative_draft_tokens = speculative_draft_tokens
        include_mtp = speculative_draft_tokens > 0
        if isinstance(model, Qwen35Model):
            if include_mtp and not model.has_mtp:
                raise ValueError("MTP decoding requires a model with its MTP head")
            self._model = model
            self.config: Qwen35Config = model.config
            self.model_path = None
            quantization = "preloaded"
            weight_source = "preloaded"
        else:
            self.model_path = Path(model)
            self._model, self.config, raw_config = load_model(
                self.model_path,
                bits=weight_bits,
                group_size=group_size,
                include_mtp=include_mtp,
            )
            packed = checkpoint_quantization(raw_config)
            bits, group = (
                (packed["bits"], packed["group_size"])
                if packed is not None
                else (weight_bits, group_size)
            )
            quantization = "bfloat16" if bits is None else f"mlx-int{bits}-g{group}"
            weight_source = (
                "bf16_checkpoint" if packed is None else "prequantized_checkpoint"
            )
            if include_vision and raw_config.get("vision_config"):
                from resin_models_mlx.qwen3_5.media import MediaEncoder
                from resin_models_mlx.qwen3_5.vision import load_vision_model

                visual = load_vision_model(
                    self.model_path, raw_config, compiled=compile_vision
                )
                self._media = MediaEncoder(
                    visual,
                    self.config.hidden_size,
                    patch_size=visual.config.patch_size,
                    merge_size=visual.config.spatial_merge_size,
                    temporal_patch_size=visual.config.temporal_patch_size,
                )
                if compile_vision:
                    side = visual.config.spatial_merge_size * 8
                    mx.eval(
                        visual(
                            mx.zeros(
                                (side * side, visual.config.patch_width),
                                dtype=mx.bfloat16,
                            ),
                            np.array([[1, side, side]], dtype=np.int64),
                        )
                    )
        self.max_seq_len = context_limit(
            max_seq_len, self.config.max_position_embeddings
        )
        # Describes how this runner executes for benchmark records.
        self.benchmark_metadata: dict[str, object] = {
            "quant": quantization,
            "weight_source": weight_source,
            "expert_backend": "mlx-gather-qmm" if self.config.is_moe else "none",
            "graphs": {"prefill": "disabled", "decode": "disabled"},
        }
        self.vocab_size = self.config.vocab_size
        self.eos_token_id = self.config.eos_token_id
        self.cache_geometry = CacheGeometry.from_config(
            self.config, include_mtp=include_mtp
        )
        if include_mtp:
            # The verified block holds its replayed state beside input and output.
            workspace_bytes += max_batch_size * self.cache_geometry.state_bytes
        slab_capacity = self._slab_capacity(
            self.cache_geometry,
            memory_utilization,
            workspace_bytes,
            max_batch_size,
            cache_slab_capacity,
        )
        cache = CacheArena(
            self.cache_geometry.layout(slab_capacity), self.cache_geometry
        )
        if self.config.head_dim == 256:
            from resin_models_mlx.paged_attention import _warm_fused_merge

            _warm_fused_merge()
        if (
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ) == (8, 2, 256):
            from resin_models_mlx._native_cache import _warm_paged_attention

            _warm_paged_attention()
        super().__init__(
            cache,
            LiveStates(cache, self._model.initial_state, STATE_GROUP),
            self._model.logits,
            prefill_chunk_size,
            max_accepted_per_step=speculative_draft_tokens + 1,
        )
        self._mtp_decode = None
        if include_mtp:
            self._mtp_decode = MtpDecode(
                self._model,
                self._cache,
                self._states,
                self._sampler,
                self._media,
                speculative_draft_tokens,
                self.max_seq_len,
            )
            self.benchmark_metadata["speculative"] = {
                "method": "mtp",
                "draft_tokens": speculative_draft_tokens,
            }

    def speculative_stats(self) -> dict[str, int | float]:
        return self._mtp_decode.stats.snapshot() if self._mtp_decode is not None else {}

    def release(self, sequence_ids: list[int]) -> None:
        super().release(sequence_ids)
        if self._mtp_decode is not None:
            self._mtp_decode.release(sequence_ids)

    def _write(self, launch: _Launch) -> None:
        super()._write(launch)
        if (
            self._mtp_decode is not None
            and launch.plan.batch.phase == BatchPhase.PREFILL
        ):
            self._mtp_decode.prefill(launch)

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        if self._mtp_decode is not None and plan.phase == BatchPhase.DECODE:
            return self._mtp_decode.forward(self._cache_plan(plan))
        return super().forward(plan)

    def enqueue(
        self,
        plan: BatchPlan,
        *,
        feedback: mx.array | SpeculativeFeedback | None = None,
    ) -> MlxCompletion | ForwardOutput:
        if self._mtp_decode is not None and plan.phase == BatchPhase.DECODE:
            if feedback is not None and not isinstance(feedback, SpeculativeFeedback):
                raise RuntimeError(
                    "a speculative continuation needs speculative feedback"
                )
            return self._mtp_decode.enqueue(self._cache_plan(plan), feedback)
        if isinstance(feedback, SpeculativeFeedback):
            raise RuntimeError("target-only decode received speculative feedback")
        return super().enqueue(plan, feedback=feedback)

    def _step(
        self,
        plan: PagedCachePlan,
        rows: np.ndarray,
        tokens: np.ndarray,
        contexts: np.ndarray,
        token_ids: mx.array,
    ) -> ModelStep:
        batch = plan.batch
        history = self._cache.history(plan, rows, contexts, decode=tokens.shape[1] == 1)
        embeddings = None
        rotary_positions = None
        if any(batch._plan.media[int(row)] is not None for row in rows):
            if self._media is None:
                raise ValueError("this Qwen3.5 runner did not load a vision tower")
            embeddings, rotary_positions = self._media.prepare(
                batch._plan,
                rows,
                batch.token_ids[tokens],
                contexts,
                self._model.embed_tokens(token_ids),
            )
        output = self._model(
            token_ids,
            mx.array(contexts.astype(np.int32)),
            self._states.gather(plan, rows),
            history,
            input_embeddings=embeddings,
            rotary_positions=rotary_positions,
        )
        return ModelStep(
            output.hidden, output.state, {KV_GROUP: (output.keys, output.values)}
        )


__all__ = ["Qwen35MlxRunner"]
