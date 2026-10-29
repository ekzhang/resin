"""Readable tokenizer-free LFM2 loading and execution on PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from resin import RunnerConfig
from resin.backend.loading import (
    MODEL_FILES,
    materialize_safetensor_model,
    read_config,
    resolve_model,
)
from resin.backend.plans import plan_rows
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
    model = materialize_safetensor_model(
        model_path,
        device=device,
        dtype=dtype,
        build=lambda: Lfm2ForCausalLM(args),
    )
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
        rows = plan_rows(plan, self._cache, self.vocab_size)

        with torch.inference_mode():
            if len(rows) > 1 and all(len(row.token_ids) == 1 for row in rows):
                caches = []
                for row in rows:
                    cache = row.cache
                    if cache is None:
                        cache = self._model.make_cache()
                        self._cache[row.request_id] = cache
                    caches.append(cache)
                inputs = torch.tensor(
                    np.stack([row.token_ids for row in rows]),
                    dtype=torch.long,
                    device=self.device,
                )
                position_tensor = torch.tensor(
                    np.stack([row.positions for row in rows]),
                    dtype=torch.long,
                    device=self.device,
                )
                logits = self._model.forward_batch(inputs, caches, position_tensor)
                logits_rows.append(logits[:, -1, :])
                if plan.needs_prompt_logprobs:
                    prompt_rows.append(prompt_scores(plan, logits[:, 0]))
            else:
                self._forward_rows(rows, logits_rows, plan, prompt_rows)
        sampler = getattr(self, "_sampler", None)
        if sampler is None:
            sampler = self._sampler = TorchSampler()
        logits = torch.cat(logits_rows)
        scores = PromptScores.concat(cast(list[PromptScores], prompt_rows))
        return format_output(
            logits, sampler(logits, plan), plan, prompt_logprobs=scores
        )

    def _forward_rows(self, rows, logits_rows, plan, prompt_rows) -> None:
        token_start = 0
        for row in rows:
            cache = row.cache
            if cache is None:
                cache = self._model.make_cache()
                self._cache[row.request_id] = cache
            inputs = torch.tensor(
                row.token_ids[None], dtype=torch.long, device=self.device
            )
            position_tensor = torch.tensor(
                row.positions[None], dtype=torch.long, device=self.device
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
                        plan, logits[0], self._model.logits, token_start=token_start
                    )
                )
                logits = self._model.logits(logits[:, -1:])
            logits_rows.append(logits[:, -1, :])
            token_start += len(row.token_ids)

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._cache.pop(request_id, None)

    @property
    def live_cache_count(self) -> int:
        return len(self._cache)
