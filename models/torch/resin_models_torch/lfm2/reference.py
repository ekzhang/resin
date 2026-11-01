"""Readable tokenizer-free LFM2 loading and execution on PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from resin import RunnerConfig
from resin.backend.loading import (
    MODEL_FILES,
    load_safetensor_weights,
    read_config,
    resolve_dtype,
    resolve_model,
)
from resin.backend.plans import BatchView
from resin.backend.sampling import (
    PromptScores,
    TorchSampler,
    format_output,
    prompt_scores,
)
from resin.backend.topology import ParallelismCapabilities
from resin.protocols import Runner
from resin_models_torch.lfm2.config import Lfm2Config
from resin_models_torch.lfm2.model import Lfm2ForCausalLM, Lfm2State

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput


DEFAULT_MODEL = "LiquidAI/LFM2.5-230M"
DEFAULT_REVISION = "13a53837c4906b4f7405932532ba85d182bb013b"


def _download_model(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(MODEL_FILES),
        )
    )


def resolve_model_path(model: str | Path, revision: str | None) -> Path:
    return resolve_model(model, revision, _download_model)


def load_model(model_path: Path, *, device, dtype=None):
    config = read_config(model_path)
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError(
            "the PyTorch runner requires an unquantized LFM2 checkpoint; "
            "quantized safetensors use a different physical format"
        )
    args = Lfm2Config.from_dict(config)
    target = torch.device(device)
    load_target = target if target.type in {"cpu", "cuda"} else torch.device("cpu")
    with torch.device("meta"):
        model = Lfm2ForCausalLM(args)
    load_safetensor_weights(model_path, model, device=load_target)
    model.materialize_buffers(load_target)
    if target != load_target:
        model.to(target)
    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype is not None:
        if not resolved_dtype.is_floating_point:
            raise ValueError("model dtype must be floating point")
        model.to(dtype=resolved_dtype)
    # Restore computed RoPE buffers after Module.to casts floating-point buffers.
    model.materialize_buffers(target)
    model.eval()
    return model, config


class Lfm2TorchRunner(Runner):
    """LFM2 runner consuming Rust-built integer-token batches."""

    runner_config = RunnerConfig(continuation_prefill=True)
    parallelism_capabilities = ParallelismCapabilities()

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = DEFAULT_REVISION,
        device: str | torch.device,
        dtype=None,
    ) -> None:
        self.device = torch.device(device)
        self.model_path = resolve_model_path(model, revision)
        self._model, self.config = load_model(
            self.model_path, device=self.device, dtype=dtype
        )
        self.eos_token_id = self.config.get("eos_token_id")
        self.vocab_size = self.config["vocab_size"]
        self._sampler = TorchSampler()
        self._cache: dict[int, Lfm2State] = {}
        self.forward_calls = 0

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        logits_rows = []
        prompt_rows = []
        self.forward_calls += 1
        batch = BatchView.from_plan(plan)
        for index, sequence_id in enumerate(batch.sequence_ids):
            sequence_id = int(sequence_id)
            cache = self._cache.get(sequence_id)
            offset = 0 if cache is None else cache.length
            token_slice = batch.token_slice(index)
            positions = batch.positions[token_slice]
            if int(batch.context_lengths[index]) != offset:
                expected = np.arange(offset, offset + len(positions), dtype=np.uint32)
                raise ValueError(
                    f"non-contiguous positions for request {sequence_id}: "
                    f"expected {expected.tolist()}, got {positions.tolist()}"
                )
            if np.any(batch.token_ids[token_slice] >= self.vocab_size):
                raise ValueError(f"token id is out of range for request {sequence_id}")

        with torch.inference_mode():
            if batch.num_sequences > 1 and np.all(batch.query_lengths == 1):
                caches = []
                for sequence_id in batch.sequence_ids:
                    sequence_id = int(sequence_id)
                    cache = self._cache.get(sequence_id)
                    if cache is None:
                        cache = self._model.make_cache()
                        self._cache[sequence_id] = cache
                    caches.append(cache)
                inputs = torch.tensor(
                    batch.token_ids[:, None],
                    dtype=torch.long,
                    device=self.device,
                )
                position_tensor = torch.tensor(
                    batch.positions[:, None],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = self._model.forward_batch(inputs, caches, position_tensor)
                logits_rows.append(logits[:, -1, :])
                if plan.needs_prompt_logprobs:
                    prompt_rows.append(prompt_scores(plan, logits[:, 0]))
            else:
                self._forward_rows(batch, logits_rows, plan, prompt_rows)
        sampler = getattr(self, "_sampler", None)
        if sampler is None:
            sampler = self._sampler = TorchSampler()
        logits = torch.cat(logits_rows)
        scores = PromptScores.concat(cast(list[PromptScores], prompt_rows))
        return format_output(
            logits, sampler(logits, plan), plan, prompt_logprobs=scores
        )

    def _forward_rows(self, batch, logits_rows, plan, prompt_rows) -> None:
        for index, sequence_id in enumerate(batch.sequence_ids):
            sequence_id = int(sequence_id)
            token_slice = batch.token_slice(index)
            cache = self._cache.get(sequence_id)
            if cache is None:
                cache = self._model.make_cache()
                self._cache[sequence_id] = cache
            inputs = torch.tensor(
                batch.token_ids[token_slice][None], dtype=torch.long, device=self.device
            )
            position_tensor = torch.tensor(
                batch.positions[token_slice][None], dtype=torch.long, device=self.device
            )
            logits = self._model(
                inputs,
                cache,
                position_tensor,
                logits_to_keep=None if plan.needs_prompt_logprobs else 1,
                **({"return_hidden": True} if plan.needs_prompt_logprobs else {}),
            )
            if plan.needs_prompt_logprobs:
                prompt_rows.append(
                    prompt_scores(
                        plan,
                        logits[0],
                        self._model.logits,
                        token_start=token_slice.start,
                    )
                )
                logits = self._model.logits(logits[:, -1:])
            logits_rows.append(logits[:, -1, :])

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._cache.pop(request_id, None)

    @property
    def live_cache_count(self) -> int:
        return len(self._cache)
