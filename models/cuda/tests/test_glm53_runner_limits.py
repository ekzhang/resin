"""Public context limits and shared target/decode profiles."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resin import (
    BatchPhase,
    BatchPlan,
    Engine,
    ForwardOutput,
    Parallelism,
    RunnerConfig,
    SamplingParams,
)
from resin_models_cuda.glm5_3_flash.cache import CacheGeometry
from resin_models_cuda.glm5_3_flash.cuda import Glm53FlashCudaRunner


@pytest.fixture
def construct_runner(monkeypatch):
    import resin_models_cuda.glm5_3_flash.cuda as module
    from resin.models.glm5_3_flash.config import Glm53Config

    # Keep real profile/catalog calculation; replace loading and device work.
    monkeypatch.setattr(
        module.CudaRunner,
        "__init__",
        lambda self, **kwargs: setattr(self, "device", torch.device("cpu")),
    )
    monkeypatch.setattr(module, "resolve_model_path", lambda value: value)
    loads = []

    def load(*args, **kwargs):
        loads.append(kwargs)
        return SimpleNamespace(layers=(), mtp=None), Glm53Config(), {}

    monkeypatch.setattr(module, "load_model", load)
    for name in ("_configure_parallel_graphs", "_configure_cache_binding"):
        monkeypatch.setattr(module.CudaRunner, name, lambda *args: None)
    monkeypatch.setattr(
        module.CudaRunner, "_configure_prefill_graphs", lambda *args, **kwargs: ()
    )
    monkeypatch.setattr(
        module.CudaRunner,
        "_configure_cuda_graphs",
        lambda self, modes: setattr(self, "observed_modes", modes),
    )
    monkeypatch.setattr(
        module, "CacheArena", lambda *args, **kwargs: SimpleNamespace(binding=None)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (128 * 1024**3,) * 2)

    def construct(**options):
        model = Glm53FlashCudaRunner(
            "unused", device="cpu", max_seq_len=4096, **options
        )
        return model, loads[-1]

    return construct


@pytest.mark.parametrize(
    "drafts,ep,catalog,policy,expected",
    [
        (0, 1, None, "full", ((1, 4096), (2, 4096), (4, 4096), (8, 4096))),
        (7, 4, None, "full", ((1, 4096), (2, 4096), (4, 4096), (8, 4096))),
        (3, 2, ((3, 4096), (8, 4096)), "full", ((3, 4096), (8, 4096))),
        (1, 1, (), "full", ()),
        (3, 2, None, "disabled", ()),
    ],
)
def test_constructor_decode_catalog(
    construct_runner, drafts, ep, catalog, policy, expected
):
    model, _ = construct_runner(
        max_batch_size=8,
        speculative_draft_tokens=drafts,
        parallel_groups=SimpleNamespace(
            topology=Parallelism(world_size=4, attn_dp=4, ep=ep)
        ),
        cuda_graph_decode_shapes=catalog,
        cuda_graph_decode=policy,
    )
    assert getattr(model, "observed_modes")["decode"].shapes == expected
    assert model.prefill_chunk_size == model.runner_config.prefill_chunk_size == 8192
    assert model.cache_geometry.state_tokens == 8192


def test_constructor_reserves_large_verification_and_selects_projection_bucket(
    construct_runner,
):
    model, loaded = construct_runner(
        max_batch_size=128, prefill_chunk_size=512, speculative_draft_tokens=7
    )
    assert loaded["max_tokens"] == 1024
    assert model.runner_config.max_accepted_per_step == 8
    assert model.cache_layout.decode_write_extra_tokens == 1
    assert model._eager_projection_rows(1) == 64
    assert model._eager_projection_rows(9) == 96
    assert model._eager_projection_rows(128) == 1024


def runner(drafts):
    result = object.__new__(Glm53FlashCudaRunner)
    result.max_seq_len = 4096
    result.max_batch_size = 2
    result.speculative_draft_tokens = drafts
    result.target_projection_rows = 64
    result._projection_batch_buckets = (1, 2)
    return result


def plan(context, phase=BatchPhase.DECODE):
    return BatchPlan(
        sequence_ids=[1],
        context_lengths=[context],
        token_ids=[7],
        token_indptr=[0, 1],
        phase=phase,
    )


def test_eager_target_marks_execution_phase_before_forward(monkeypatch):
    import resin_models_cuda.glm5_3_flash.cuda as runner_module

    # Exercise the real eager orchestration without CUDA's pinned-host staging.
    monkeypatch.setattr(
        runner_module,
        "TensorStager",
        lambda device: SimpleNamespace(
            device=lambda name, values, dtype: torch.tensor(
                np.asarray(values), dtype=dtype, device="cpu"
            )
        ),
    )
    model = runner(0)
    model.device = torch.device("cpu")
    phases = []

    def forward(ids, positions, cache, *, hidden):
        phases.append((cache.is_decode, cache.projection_rows))
        return torch.ones((ids.numel(), 16))

    model._cache = SimpleNamespace(
        tensorize=lambda cache_plan, staging: SimpleNamespace(
            batch=lambda *args: SimpleNamespace(store=lambda hidden: None)
        )
    )
    model._model = SimpleNamespace(
        forward_hidden=forward, logits=lambda hidden, *, projection_rows=0: hidden
    )
    model._sampler = lambda logits, batch: torch.zeros(
        batch.num_sequences, dtype=torch.long
    )
    for phase in (BatchPhase.PREFILL, BatchPhase.DECODE):
        model._eager(plan(2, phase), object())
    assert phases == [(False, 0), (True, 64)]


@pytest.mark.parametrize("drafts", [0, 1, 3, 7])
def test_last_public_context_row_remains_legal_with_native_mtp(drafts):
    model = runner(drafts)
    assert model._validate_plan(plan(4095)) == ((0, 1),)
    assert model._validate_plan(plan(4095, BatchPhase.PREFILL)) == ((0, 1),)
    with pytest.raises(ValueError, match="max_seq_len"):
        model._validate_plan(plan(4096))


@pytest.mark.parametrize("drafts", [0, 1, 3, 7])
def test_graph_selection_uses_public_width_and_capture_reserves_verification(drafts):
    model = runner(drafts)
    required, allocated = [], []

    def select(mode, context, *, required_shape, eligible=True):
        assert eligible
        required.append(required_shape)
        return None

    def capture_inputs(batch, width):
        allocated.append((batch, width))
        return {}

    model._select_distributed_cuda_graph = select
    assert model._distributed_graph(plan(4095), object(), None) == (False, None)
    assert required == [(1, 4096)]
    model._cache = SimpleNamespace(graph_shape_capture_inputs=capture_inputs)
    model.vocab_size = 16
    model.parallel_groups = None
    model._decode_graph_capture_inputs((1, 4096))
    assert allocated == [(1, 4096 + drafts)]


@pytest.mark.parametrize("drafts", [0, 1, 3, 7])
@pytest.mark.parametrize("context", [63, 64, 510, 511, 4094])
def test_real_engine_leases_are_prepared_at_page_and_public_boundaries(drafts, context):
    model = runner(drafts)
    geometry = CacheGeometry(
        ("linear_attention", "deepseek_sparse_attention"),
        hidden_size=16,
        latent_dim=8,
        index_dim=4,
        heads=2,
        head_dim=8,
        conv_history=3,
        state_tokens=512,
        include_mtp=bool(drafts),
    )
    model.cache_layout = geometry.layout(128)
    model.runner_config = RunnerConfig(
        max_batch_tokens=512,
        prefill_chunk_size=512,
        continuation_prefill=True,
        max_accepted_per_step=drafts + 1,
        cache_layout=model.cache_layout,
    )
    prepared = []

    class PlanRunner:
        runner_config = model.runner_config

        def bind_resources(self, binding):
            pass

        def forward(self, batch):
            prepared.append(model._prepare(batch))
            return ForwardOutput(
                token_ids=np.full(batch.num_sequences, 7, dtype=np.uint32),
                token_indptr=np.arange(batch.num_sequences + 1, dtype=np.uint32),
            )

    # A target-only diagnostic forward may use fewer writes than its lease.
    # Its validation must still use the RunnerConfig installed in the Engine.
    model.speculative_draft_tokens = 0
    engine = Engine(PlanRunner(), context_length=4096, max_running_requests=1)
    request = engine.submit(
        [1] * context, sampling_params=SamplingParams(max_output_tokens=2)
    )
    while engine.has_work():
        engine.step()
    assert request.take_result() == [7, 7]
    decode = [item for item in prepared if item.phase == BatchPhase.DECODE]
    assert len(decode) == 1
    assert decode[0].batch.context_lengths.tolist() == [context]
    expected_end = (
        context
        + model.runner_config.max_accepted_per_step
        + model.cache_layout.decode_write_extra_tokens
    )
    assert len(decode[0].groups[0].block_slab_ids) == (expected_end + 63) // 64
    engine.close()
