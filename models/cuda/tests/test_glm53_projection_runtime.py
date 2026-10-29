"""Execute runner phase boundaries on CPU; inspect real target head operator M."""

from types import SimpleNamespace

import pytest
import torch

from resin.models.glm5_3_flash.config import Glm53Config
from resin_models_cuda.glm5_3_flash import cuda as runner_module
from resin_models_cuda.glm5_3_flash.cuda import Glm53FlashCudaRunner


@pytest.mark.parametrize("drafts", [0, 1, 3, 5, 7])
@pytest.mark.parametrize("capacity", [None, 2, 8, 64])
def test_constructor_target_profile_is_shared_across_capacity_and_draft_width(
    monkeypatch, drafts, capacity
):
    # Keep actual capacity/profile/catalog calculation; replace only weights,
    # allocation and graph registration at the CPU boundary.
    monkeypatch.setattr(
        runner_module.CudaRunner,
        "__init__",
        lambda self, **kwargs: setattr(self, "device", torch.device("cpu")),
    )
    monkeypatch.setattr(runner_module, "resolve_model_path", lambda value: value)
    loads = []

    def load(*args, **kwargs):
        loads.append(kwargs)
        return SimpleNamespace(layers=(), mtp=None), Glm53Config(), {}

    monkeypatch.setattr(runner_module, "load_model", load)
    for name in ("_configure_parallel_graphs", "_configure_cache_binding"):
        monkeypatch.setattr(runner_module.CudaRunner, name, lambda *args: None)
    monkeypatch.setattr(
        runner_module.CudaRunner,
        "_configure_prefill_graphs",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        runner_module.CudaRunner,
        "_configure_cuda_graphs",
        lambda self, modes: setattr(self, "observed_modes", modes),
    )
    monkeypatch.setattr(
        runner_module,
        "CacheArena",
        lambda *args, **kwargs: SimpleNamespace(binding=None),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (128 * 1024**3,) * 2)
    options = {} if capacity is None else {"max_batch_size": capacity}
    runner = Glm53FlashCudaRunner(
        "unused",
        device="cpu",
        max_seq_len=4096,
        prefill_chunk_size=512,
        speculative_draft_tokens=drafts,
        **options,
    )
    batch = 8 if capacity is None else capacity
    assert runner.max_batch_size == batch
    assert runner.target_projection_rows == 64
    assert runner.max_seq_len == 4096
    assert runner.runner_config.max_accepted_per_step == drafts + 1
    assert runner.cache_layout.decode_write_extra_tokens == int(bool(drafts))
    assert loads[0]["max_tokens"] == max(512, batch * (drafts + 1))
    assert getattr(runner, "observed_modes")["decode"].shapes == tuple(
        (size, 4096) for size in (1, 2, 4, 8, 12, 16, 24, 32, 64) if size <= batch
    )
