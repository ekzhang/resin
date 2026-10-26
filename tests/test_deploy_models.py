"""Shared serving/benchmark configuration preserves runner and checkpoint contracts."""

import importlib
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from deploy.models import (
    MODEL_SPECS,
    create_runner,
    download_checkpoint,
    get_model,
    require_checkpoint,
    validate_gpu,
)


def test_mlx_factory_constructs_single_device_runner(monkeypatch):
    spec = get_model("ornith-1.0-35b")
    module = importlib.import_module("resin_models_mlx.qwen3_5_moe")
    calls = []
    monkeypatch.setattr(
        module,
        "Qwen3_5MoeMlxRunner",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    def create(**overrides):
        options: dict[str, Any] = dict(
            device="mps",
            context_length=8192,
            max_batch_size=8,
            prefill_chunk_size=4096,
            draft_tokens=0,
        )
        return create_runner(spec, **(options | overrides))

    create()
    assert calls == [((spec.path,), {"memory_utilization": 0.7})]
    with pytest.raises(ValueError, match="no MTP"):
        create(draft_tokens=2)
    with pytest.raises(ValueError, match="one Apple Silicon device"):
        create(groups=object())
    with pytest.raises(ValueError, match="4096-token chunks"):
        create(prefill_chunk_size=8192)


def make_checkpoint(path, required_files):
    path.mkdir(parents=True)
    for name in required_files:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")


@pytest.mark.parametrize("name", MODEL_SPECS)
def test_existing_complete_cache_adoption_never_downloads(monkeypatch, tmp_path, name):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    spec = replace(get_model(name), model_mount=str(tmp_path))
    make_checkpoint(spec.path, spec.checkpoint.required_files)
    if spec.assistant:
        make_checkpoint(spec.assistant_path, spec.assistant.required_files)

    def no_download(*args, **kwargs):
        raise AssertionError("unexpected download")

    monkeypatch.setattr("huggingface_hub.snapshot_download", no_download)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        require_checkpoint(spec)
    report = download_checkpoint(spec, metadata_only=True)
    assert report["cache_hit"] and report["metadata_only"]
    assert require_checkpoint(spec) == spec.path
    (spec.path / "model.safetensors").write_bytes(b"changed size")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        require_checkpoint(spec)


def test_local_checkpoint_override_needs_no_volume_marker(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    spec = get_model("gpt-oss-20b")
    make_checkpoint(tmp_path / "local", spec.checkpoint.required_files)
    assert require_checkpoint(spec, checkpoint=tmp_path / "local") == tmp_path / "local"


@pytest.mark.parametrize(
    "draft,groups",
    [(0, None), (5, SimpleNamespace(topology=SimpleNamespace(world_size=2)))],
)
def test_inkling_rejects_unsupported_runtime_options(draft, groups):
    with pytest.raises(ValueError, match="fixed five-token|single GPU"):
        create_runner(
            "inkling",
            device="cuda:0",
            groups=groups,
            context_length=1024,
            max_batch_size=1,
            prefill_chunk_size=4096,
            draft_tokens=draft,
        )


@pytest.mark.parametrize(
    "model,gpu",
    [
        ("gemma4", "B200"),
        ("qwen3_5", "B300"),
        ("inkling", "H200"),
        ("deepseek-v41", "H100"),
    ],
)
def test_gpu_override_cannot_reuse_incompatible_kernel_cache(model, gpu):
    with pytest.raises(ValueError, match="GPU-specific kernel configuration"):
        validate_gpu(model, gpu)
    assert validate_gpu(model) == get_model(model).gpu


@pytest.mark.parametrize("gpu", ["H100", "H200"])
def test_gpt_gpu_override_preserves_sm90_kernel_architecture(gpu):
    assert validate_gpu("gpt-oss-120b", gpu) == gpu


@pytest.mark.parametrize(
    "maximum,expected",
    [(3, (1, 2, 3)), (8, (1, 2, 4, 8)), (17, (1, 2, 4, 8, 12, 16, 17))],
)
def test_glm_default_target_and_mtp_catalogs_match_physical_batches(
    monkeypatch, maximum, expected
):
    module = importlib.import_module("resin_models_cuda.glm5_3_flash.cuda")
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module, "Glm53FlashCudaRunner", Runner)
    for drafts in (0, 1, 3, 7):
        create_runner(
            "glm-5.3-flash",
            device="cuda:0",
            context_length=4096,
            max_batch_size=maximum,
            prefill_chunk_size=512,
            draft_tokens=drafts,
        )
    shapes = tuple((batch, 4096) for batch in expected)
    assert all(call["cuda_graph_decode_shapes"] == shapes for call in calls)
    for active in range(1, maximum + 1):
        selected = [
            next(
                shape
                for shape in call["cuda_graph_decode_shapes"]
                if shape[0] >= active
            )
            for call in calls
        ]
        assert selected == [selected[0]] * len(calls)


@pytest.mark.parametrize(
    "policy,catalog,expected",
    [
        ("full", (3, 5, 8), ((3, 4096), (5, 4096), (8, 4096))),
        ("full", (), ()),
        ("disabled", None, ()),
    ],
)
def test_glm_explicit_catalog_and_disabled_override_remain_authoritative(
    monkeypatch, policy, catalog, expected
):
    module = importlib.import_module("resin_models_cuda.glm5_3_flash.cuda")
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs["cuda_graph_decode_shapes"])

    monkeypatch.setattr(module, "Glm53FlashCudaRunner", Runner)
    for drafts in (0, 3):
        create_runner(
            "glm-5.3-flash",
            device="cuda:0",
            context_length=4096,
            max_batch_size=8,
            prefill_chunk_size=512,
            draft_tokens=drafts,
            decode_graph=policy,
            decode_batch_sizes=catalog,
        )
    assert calls == [expected, expected]


@pytest.mark.parametrize("include_mtp", [False, True])
@pytest.mark.parametrize("checkpoint", ["serving", "full-window"])
def test_glm_serving_checkpoints_allow_real_engine_prefix_reuse(
    include_mtp, checkpoint
):
    from output_helpers import forward_output

    from resin import (
        Engine,
        RunnerConfig,
        SamplingParams,
        SchedulerConfig,
    )
    from resin.models.glm5_3_flash.config import Glm53Config
    from resin_models_cuda.glm5_3_flash.cache import CacheGeometry

    spec = get_model("glm-5.3-flash")
    chunk = spec.prefill_chunk_size if checkpoint == "serving" else spec.context_length
    geometry = CacheGeometry.from_config(
        Glm53Config(), state_tokens=chunk, include_mtp=include_mtp
    )

    class Runner:
        # Actual model geometry and native Engine ownership, without allocating
        # cache tensors or model weights. Only publication/lookup is tested.
        runner_config = RunnerConfig(
            cache_layout=geometry.layout(128),
            max_batch_tokens=chunk,
            prefill_chunk_size=chunk,
            continuation_prefill=True,
            max_accepted_per_step=4 if include_mtp else 1,
        )

        def __init__(self):
            self.plans = []

        def bind_resources(self, binding):
            self.binding = binding

        def forward(self, plan):
            self.plans.append(plan)
            output = forward_output(
                [int(plan.token_ids[end - 1]) + 1 for end in plan.token_indptr[1:]]
            )
            receipts = {}
            for group in plan.cache_groups:
                if group.state is None:
                    continue
                for reservation_id, row, prefix, _, _ in group.state.checkpoint_slots:
                    start, end = plan.token_range(row)
                    if int(plan.context_lengths[row]) + end - start == prefix:
                        receipts[reservation_id] = None
            return output.with_state_checkpoints(tuple(receipts))

    runner = Runner()
    with Engine(
        runner,
        context_length=spec.context_length,
        max_running_requests=spec.max_batch_size,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=chunk, prefill_chunk_size=chunk
        ),
    ) as engine:
        prompt = list(range(spec.prefill_chunk_size + 256))
        first = engine.submit(
            prompt, sampling_params=SamplingParams(max_output_tokens=1)
        )
        while engine.has_work():
            engine.step()
        assert first.take_result() == [len(prompt)]
        runner.plans.clear()
        second = engine.submit(
            prompt, sampling_params=SamplingParams(max_output_tokens=1)
        )
        expected = spec.prefill_chunk_size if checkpoint == "serving" else 0
        assert second.cached_tokens == expected
        while engine.has_work():
            engine.step()
        assert runner.plans[0].context_lengths.tolist() == [expected]
        assert sum(plan.num_tokens for plan in runner.plans) == len(prompt) - expected
        assert second.take_result() == [len(prompt)]
