from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from resin import BatchPhase, BatchPlan
from resin.backend.cuda.failures import CudaFailureGuard, PostAcceptCudaError
from resin.backend.cuda.graphs import default_decode_graph_shapes
from resin.backend.cuda.runner import CudaRunner
from resin.models.gemma4.config import SLIDING_ATTENTION
from resin.models.gemma4.plan import PAGE_SIZE, SLIDING_WINDOW, Gemma4BatchPlan
from resin_models_cuda.gemma4 import cuda as gemma4_runtime
from resin_models_cuda.gemma4.cuda import (
    Gemma4CudaRunner,
    _Gemma4KvArena,
    _select_nvfp4_backend,
    plan_gemma4_kv_capacity,
)


def _hidden_storage_runner() -> Gemma4CudaRunner:
    runner = object.__new__(Gemma4CudaRunner)
    runner.device = torch.device("cpu")
    runner.max_num_seqs = 2
    runner.config = SimpleNamespace(hidden_size=5)
    runner._target_hidden = {}
    runner._hidden_bank = None
    runner._hidden_free_slots = []
    runner._hidden_slot_generation = 0
    runner._hidden_index_cache = {}
    runner._hidden_input_buffers = {}
    runner._media_features = {}
    return runner


def test_hidden_storage_owns_rows_across_rebatch_growth_and_release() -> None:
    runner = _hidden_storage_runner()
    source = torch.arange(10, dtype=torch.bfloat16).reshape(2, 5)
    runner._set_target_hidden([10, 20], source)
    saved = runner._gather_target_hidden([20, 10], 4)
    assert saved.tolist() == [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4], [0] * 5, [0] * 5]
    output_pointer = saved.data_ptr()
    index_pointer = runner._hidden_index_cache[False][1].data_ptr()
    assert runner._gather_target_hidden([20, 10], 4).data_ptr() == output_pointer
    assert runner._hidden_index_cache[False][1].data_ptr() == index_pointer

    # The next replay can overwrite its result without corrupting saved state.
    source.fill_(-99)
    runner._set_target_hidden([30], torch.full((1, 5), 30, dtype=torch.bfloat16))
    released_slot = runner._target_hidden[20]
    runner.release([20])
    runner._set_target_hidden([40], torch.full((1, 5), 40, dtype=torch.bfloat16))
    assert runner._target_hidden[40] == released_slot
    assert runner._gather_target_hidden([30, 10, 40], 4).tolist() == [
        [30] * 5,
        [0, 1, 2, 3, 4],
        [40] * 5,
        [0] * 5,
    ]
    with pytest.raises(KeyError):
        runner._gather_target_hidden([20], 1)
    with pytest.raises(ValueError, match="distinct"):
        runner._set_target_hidden([99, 99], torch.zeros(2, 5, dtype=torch.bfloat16))


def test_hidden_storage_saves_mixed_acceptance_and_prefill_final_rows() -> None:
    from resin.backend.cuda.completion import DeviceAcceptance

    runner = _hidden_storage_runner()
    verification = torch.arange(60, dtype=torch.bfloat16).reshape(3, 4, 5)
    expected = torch.stack((verification[0, 0], verification[1, 1], verification[2, 3]))
    accepted = DeviceAcceptance.from_samples(
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        torch.tensor([[0, 1, 2, 3], [4, 0, 0, 0], [7, 8, 9, 10]]),
        torch.zeros(3, dtype=torch.int64),
    )
    assert accepted.accepted_drafts.tolist() == [0, 1, 3]
    runner._store_accepted_hidden([10, 30, 40], accepted.accepted_drafts, verification)
    verification.zero_()
    assert torch.equal(runner._gather_target_hidden([10, 30, 40], 3), expected)

    prefill = torch.arange(20, dtype=torch.bfloat16).reshape(4, 5)
    runner._store_last_hidden(
        cast(BatchPlan, SimpleNamespace(sequence_ids=[30, 10])),
        cast(Gemma4BatchPlan, SimpleNamespace(last_token_indices=[1, 3])),
        prefill,
    )
    assert torch.equal(runner._gather_target_hidden([30, 10], 2), prefill[[1, 3]])


def test_capacity_planner_accounts_for_fp8_shared_parent_bytes() -> None:
    total = 96 * 1024**3
    free = 63 * 1024**3
    plan = plan_gemma4_kv_capacity(total, free, 0.8)

    assert plan.bytes_per_parent == 25 * 1024**2
    assert plan.protected_free_bytes == math.ceil(total * 0.2)
    assert (
        plan.parent_capacity
        == (free - plan.protected_free_bytes - plan.overhead_bytes)
        // plan.bytes_per_parent
        - 1
    )
    assert plan.overhead_bytes == 13 * 256 * 1024**2
    assert plan.allocation_bytes == (plan.parent_capacity + 1) * plan.bytes_per_parent


def test_fp4_backend_falls_back_when_cudnn_is_too_old() -> None:
    assert _select_nvfp4_backend("auto", 92000) == "cute-dsl"
    assert _select_nvfp4_backend("auto", 92301) == "cudnn"


def test_local_and_global_cache_views_cover_raw_parent_child_rows() -> None:
    config = SimpleNamespace(
        local_layer_count=2,
        global_layer_count=1,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        history_blocks_per_parent=2,
    )
    arena = _Gemma4KvArena(
        torch.device("cpu"),
        config,
        parent_capacity=3,
        dtype=torch.float8_e4m3fn,
    )

    expected = 4 * 2 * 2 * PAGE_SIZE * 2 * 4
    assert arena.storage.numel() == expected
    assert arena.local_keys.untyped_storage().data_ptr() == arena.storage.data_ptr()
    assert arena.global_keys.untyped_storage().data_ptr() == arena.storage.data_ptr()
    assert arena.local_keys.shape == (2, 4, PAGE_SIZE, 2, 4)
    assert arena.global_keys.shape == (1, 8, PAGE_SIZE, 1, 8)

    # Packing is two in this fixture. Raw global page 2 is physical parent 1,
    # child 0; raw page 4 is physical parent 2, child 0. Parent zero retains
    # both null-page children, so there is no compact-ID shift to remember.
    local_parent_bytes = arena.storage[0].numel() * arena.storage.element_size()
    assert arena.local_keys[:, 1].data_ptr() - arena.storage.data_ptr() < (
        2 * local_parent_bytes
    )
    assert arena.global_keys[:, 2].data_ptr() - arena.storage.data_ptr() == (
        local_parent_bytes
    )
    assert arena.global_keys[:, 4].data_ptr() - arena.storage.data_ptr() == (
        2 * local_parent_bytes
    )


def test_bfloat16_cache_capacity_accounts_for_two_byte_elements() -> None:
    fp8 = plan_gemma4_kv_capacity(96 * 1024**3, 63 * 1024**3, 0.8)
    bf16 = plan_gemma4_kv_capacity(
        96 * 1024**3,
        63 * 1024**3,
        0.8,
        cache_element_bytes=2,
    )

    assert bf16.bytes_per_parent == 2 * fp8.bytes_per_parent
    assert bf16.parent_capacity < fp8.parent_capacity


@pytest.mark.parametrize("capacity,expected_width", [(2, 8), (32, 11)])
@pytest.mark.parametrize("speculative_draft_tokens", [0, 3])
def test_default_graph_history_uses_packed_cache_and_context_bounds(
    monkeypatch, capacity, expected_width, speculative_draft_tokens
) -> None:
    class ConfiguredGraphs(Exception):
        pass

    config = SimpleNamespace(
        max_position_embeddings=641,
        history_blocks_per_parent=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
        hidden_size=8,
        vocab_size=16,
    )
    model = SimpleNamespace(prepare_attention=lambda: None)
    assistant_config = SimpleNamespace(
        backbone_hidden_size=8,
        vocab_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=8,
    )
    received = []

    def initialize(runner, **options):
        runner.device = torch.device("cpu")
        runner.cuda_graph_decode = options["cuda_graph_decode"]
        runner.kv_cache_torch_dtype = torch.float8_e4m3fn

    def configure(runner, modes):
        received.append(modes["decode"].shapes)
        raise ConfiguredGraphs

    monkeypatch.setattr(CudaRunner, "__init__", initialize)
    monkeypatch.setattr(Gemma4CudaRunner, "_configure_cuda_graphs", configure)
    monkeypatch.setattr(gemma4_runtime, "resolve_model_path", lambda *a: "unused")
    monkeypatch.setattr(gemma4_runtime, "resolve_assistant_path", lambda *a: "unused")
    monkeypatch.setattr(
        gemma4_runtime, "load_model", lambda *a, **k: (model, config, {})
    )
    monkeypatch.setattr(
        gemma4_runtime, "load_assistant", lambda *a, **k: (object(), assistant_config)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (100 * 1024**3,) * 2)
    with pytest.raises(ConfiguredGraphs):
        Gemma4CudaRunner(
            device="cuda",
            cache_parent_capacity=capacity,
            speculative_draft_tokens=speculative_draft_tokens,
        )
    assert received == [
        default_decode_graph_shapes(
            expected_width,
            speculative=bool(speculative_draft_tokens),
            max_batch_size=128,
        )
    ]


def test_ordinary_graph_capture_rows_stay_inside_null_page() -> None:
    runner = object.__new__(Gemma4CudaRunner)
    runner.speculative_draft_tokens = 0

    inputs = runner._graph_capture_inputs((128, 4096), vocab_size=256)

    assert inputs["local_append_offsets"] == tuple(
        row % PAGE_SIZE for row in range(128)
    )
    assert inputs["global_append_offsets"] == inputs["local_append_offsets"]


def test_ordinary_graph_padding_rows_stay_inside_null_page() -> None:
    plan = SimpleNamespace(
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        page_indices=torch.tensor([7], dtype=torch.int32).numpy(),
        last_page_lengths=torch.tensor([1], dtype=torch.int32).numpy(),
        append_slots=torch.tensor([7 * PAGE_SIZE], dtype=torch.int64).numpy(),
    )

    inputs = Gemma4CudaRunner._graph_page_inputs(
        # Minimal page-table double covers only fields used by this adapter.
        "local",
        plan,  # ty: ignore[invalid-argument-type]
        active_rows=1,
        graph_rows=128,
        width=81,
    )

    offsets = inputs["local_append_offsets"]
    assert isinstance(offsets, np.ndarray)
    assert max(offsets) < PAGE_SIZE


@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
def test_prefill_regions_activate_after_media_and_attention_planning(phase) -> None:
    runner = object.__new__(Gemma4CudaRunner)
    runner.device = torch.device("cpu")
    runner.cuda_graph_prefill = "breakable"
    runner._prefill_graph_active = False
    setattr(
        runner,
        "_model",
        SimpleNamespace(embed_tokens=lambda ids: ids[:, None].bfloat16()),
    )
    events = []

    def record(name, result):
        def call(*_args):
            events.append((name, runner._prefill_graph_active))
            return result

        return call

    runner._merge_media_embeddings = record("media", {})
    runner._local_custom_mask = record("mask", None)
    runner._device_plan = record("cache", object())
    runner._plan_attention = record("attention", (object(), object()))
    expected = torch.ones(3, 1, dtype=torch.bfloat16)
    runner._run_transformer = record("transformer", expected)
    batch = SimpleNamespace(
        token_ids=(1, 2, 3),
        positions=(130, 131, 132),
        local=object(),
        global_=object(),
        phase=phase,
    )
    # This fixture isolates activation order from native plan construction.
    assert runner._eager_transformer(object(), batch) is expected  # ty: ignore[invalid-argument-type]
    assert events[:-1] == [
        (name, False) for name in ("media", "mask", "cache", "cache", "attention")
    ]
    assert events[-1] == ("transformer", phase == BatchPhase.PREFILL)
    assert not runner._prefill_graph_active


def test_prefill_feed_forward_can_reuse_output_storage_across_layers() -> None:
    runner = object.__new__(Gemma4CudaRunner)
    rows, width = 3, 4
    layers = [
        SimpleNamespace(
            input_layernorm=lambda hidden: hidden * 0.5,
            self_attn=SimpleNamespace(
                layer_type=SLIDING_ATTENTION,
                project_qkv=lambda hidden, *_: (hidden[:, None],) * 3,
                o_proj=lambda hidden: hidden * 0.25,
            ),
            post_attention_layernorm=lambda hidden: hidden + 0.125,
            pre_feedforward_layernorm=lambda hidden: hidden * 0.5,
            mlp=lambda hidden: hidden.square() + 0.25,
            post_feedforward_layernorm=lambda hidden: hidden * 0.25,
            layer_scalar=scale,
        )
        for scale in (0.5, 0.75)
    ]
    setattr(runner, "config", SimpleNamespace(layer_types=(SLIDING_ATTENTION,) * 2))
    setattr(
        runner,
        "_model",
        SimpleNamespace(
            language_model=SimpleNamespace(
                layers=layers,
                rotary_emb=lambda *_: (None, None),
                norm=lambda hidden: hidden * 0.5,
            )
        ),
    )
    setattr(
        runner,
        "_cache",
        SimpleNamespace(
            dtype=torch.bfloat16,
            local_keys=torch.zeros(2, 1, PAGE_SIZE, 1, width, dtype=torch.bfloat16),
            local_values=torch.zeros(2, 1, PAGE_SIZE, 1, width, dtype=torch.bfloat16),
        ),
    )
    plan = SimpleNamespace(
        append_pages=torch.zeros(rows, dtype=torch.int64),
        append_offsets=torch.arange(rows),
    )
    wrapper = SimpleNamespace(run=lambda query, _: query)
    hidden = torch.arange(rows * width, dtype=torch.bfloat16).reshape(rows, width) / 16

    def run():
        return runner._run_transformer(
            # Minimal device-plan doubles supply only append destinations.
            hidden,
            torch.arange(rows),
            plan,  # ty: ignore[invalid-argument-type]
            plan,  # ty: ignore[invalid-argument-type]
            wrapper,
            wrapper,
        )

    operations = tuple(
        lambda hidden, layer=layer: runner._feed_forward(layer, hidden)
        for layer in layers
    )
    runner._prefill_feed_forward = operations
    expected = run()
    shared = torch.empty_like(hidden)

    def replay(operation, hidden):
        shared.copy_(operation(hidden))
        return shared

    runner._prefill_feed_forward = tuple(
        lambda hidden, operation=operation: replay(operation, hidden)
        for operation in operations
    )
    torch.testing.assert_close(run(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("speculative_draft_tokens", [0, 3])
def test_graph_attention_contract_matches_capture_and_replay(
    monkeypatch, speculative_draft_tokens
) -> None:
    import flashinfer

    wrappers = []

    class Wrapper:
        def __init__(self, _workspace, _layout, **options):
            self.options = options
            self.calls = []
            wrappers.append(self)

        def plan(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    for name in (
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
    ):
        monkeypatch.setattr(flashinfer, name, Wrapper, raising=False)
    runner = object.__new__(Gemma4CudaRunner)
    runner.device = torch.device("cpu")
    setattr(
        runner,
        "config",
        SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=4,
            global_head_dim=8,
            hidden_size=8,
            vocab_size=16,
        ),
    )
    runner.speculative_draft_tokens = speculative_draft_tokens
    runner.attention_config = runner.config
    setattr(runner, "_assistant", object())
    setattr(
        runner,
        "_cache",
        SimpleNamespace(
            dtype=torch.float8_e4m3fn,
            local_keys=[None],
            local_values=[None],
            global_keys=[None],
            global_values=[None],
        ),
    )
    setattr(
        runner,
        "_sampler",
        SimpleNamespace(empty=lambda *_: None, graph_inputs=lambda _: {}),
    )
    setattr(runner, "_graph_attention_workspaces", (None, None))
    runner._graph_attention = {}
    runner._graph_spec_attention = {}
    runner._graph_verify_hidden = {}
    runner._graph_logits = {}
    runner._graph_acceptance = {}
    shape = (2, 3)
    step = runner._new_decode_graph_step(shape)
    inputs = runner._graph_capture_inputs(shape, vocab_size=16)
    for name, values in inputs.items():
        if name.endswith("_page_indices"):
            assert isinstance(values, tuple)
            inputs[name] = (17 if "local" in name else 23,) * len(values)
    runner._prepare_graph_attention(shape, inputs)

    assert step.inputs.keys() <= inputs.keys()
    assert len(wrappers) == (4 if speculative_draft_tokens else 2)
    for wrapper in wrappers:
        (initial, initial_options), (current, current_options) = wrapper.calls
        assert initial_options == current_options
        assert initial[-4:] == current[-4:]
        local = current[-2] == 4
        assert current[-4:] == (4, 2 if local else 1, 4 if local else 8, PAGE_SIZE)
        assert current_options["window_left"] == (SLIDING_WINDOW - 1 if local else -1)
        assert current_options["disable_split_kv"] is True
        verify = "qo_indptr_buf" in wrapper.options
        assert current_options.get("causal", False) is verify
        assert current[-6].tolist() == [17 if local else 23] * shape[0]
        if verify:
            assert current[0].tolist() == [0, 4, 8]
            assert current[-5].tolist() == [5, 5]
        assert wrapper.options["backend"] == (
            "auto" if speculative_draft_tokens else "fa2"
        )
        assert wrapper.options.get("use_tensor_cores", False) is (
            not speculative_draft_tokens
        )


@pytest.mark.parametrize("graph", [False, True])
def test_speculative_failure_after_state_update_is_not_retryable(
    monkeypatch, graph
) -> None:
    state = [0]
    synchronized = []
    runner = object.__new__(Gemma4CudaRunner)
    runner.speculative_draft_tokens = 1
    setattr(runner, "cache_layout", object())
    setattr(runner, "_check_cuda_graphs", lambda _phase: None)
    runner._cuda_failure = CudaFailureGuard(
        lambda: synchronized.append(True), label="Gemma speculative decode"
    )
    native = SimpleNamespace(phase=BatchPhase.DECODE)
    monkeypatch.setattr(
        gemma4_runtime, "build_gemma4_batch_plan", lambda *_args, **_kwargs: native
    )

    def fail_after_update(_plan, _batch):
        state[0] += 1
        raise ValueError("sampling failed after state update")

    setattr(
        runner,
        "_speculative_graph_forward",
        fail_after_update if graph else lambda _plan, _batch: None,
    )
    setattr(runner, "_speculative_decode", fail_after_update)

    with pytest.raises(PostAcceptCudaError, match="sampling failed"):
        # The plan double tests failure tagging after a simulated cache write.
        runner.forward(native)  # ty: ignore[invalid-argument-type]

    assert state == [1]
    assert synchronized == [True]


@pytest.fixture
def cached_edge_runner():
    from resin_models_cuda.gemma4.edge_cuda import Gemma4EdgeCudaRunner

    runner = object.__new__(Gemma4EdgeCudaRunner)
    runner.device = torch.device("cpu")
    runner.speculative_draft_tokens = 3
    shape = (2, 256)
    wrappers = [
        SimpleNamespace(_plan_info=object(), workspace_revision=0) for _ in range(4)
    ]
    runner._graph_spec_attention = {
        shape: (wrappers[0], wrappers[1], wrappers[2], wrappers[3])
    }
    calls, failure = [], [None]

    def plan(
        wrapper,
        indptr,
        indices,
        last,
        *,
        local,
        qo_indptr=None,
        assistant=False,
        non_blocking=False,
    ):
        calls.append(
            (wrapper, indptr.clone(), indices.clone(), last.clone(), qo_indptr)
        )
        wrapper.workspace_revision += 1
        if wrapper is failure[0]:
            failure[0] = None
            raise RuntimeError("planner failed after mutating workspace")
        wrapper._plan_info = object()

    setattr(runner, "_plan_graph_attention", plan)
    inputs = {"verify_qo_indptr": np.array([0, 4, 8], dtype=np.int32)}
    for prefix in ("assistant_", "verify_"):
        for kind in ("local", "global"):
            name = prefix + kind
            inputs[name + "_kv_indptr"] = np.array([0, 1, 3], dtype=np.int32)
            inputs[name + "_page_indices"] = np.array(
                [2, 5, 7, 0, 0, 0], dtype=np.int32
            )
            inputs[name + "_last_page_lengths"] = np.array([63, 2], dtype=np.int32)
    return runner, shape, wrappers, inputs, calls, failure


def test_fractional_page_length_and_physical_recycling_keep_schedule(
    cached_edge_runner,
):
    runner, shape, wrappers, inputs, calls, _ = cached_edge_runner
    runner._prepare_graph_attention(shape, inputs)
    assert len(calls) == 4
    for name, value in inputs.items():
        if name.endswith("page_indices"):
            value[:3] = [17, 18, 19]
        if name.endswith("last_page_lengths"):
            value[:] = [64, 3]
    runner._prepare_graph_attention(shape, inputs)
    assert len(calls) == 4
    assert runner.attention_plan_cache_stats == {"hits": 4, "misses": 4}
    assert runner.benchmark_counters["attention_plan_cache.hits"] == 4
    assert runner.benchmark_counters["attention_plan_cache.misses"] == 4
    assert all(w.workspace_revision == 1 for w in wrappers)


@pytest.mark.parametrize("prefix", ["assistant_local", "verify_global"])
def test_page_boundary_and_unequal_rows_replan_only_changed_wrapper(
    cached_edge_runner, prefix
):
    runner, shape, _, inputs, calls, _ = cached_edge_runner
    runner._prepare_graph_attention(shape, inputs)
    inputs[prefix + "_kv_indptr"][:] = [0, 2, 3]
    # Total pages are unchanged, but one row gained a page and another lost
    # one. The full CSR signature, not its final value, determines the plan.
    runner._prepare_graph_attention(shape, inputs)
    assert len(calls) == 5
    assert runner.attention_plan_cache_stats == {"hits": 3, "misses": 5}
    inputs[prefix + "_kv_indptr"][:] = [0, 2, 4]
    runner._prepare_graph_attention(shape, inputs)
    assert len(calls) == 6


def test_verifier_query_schedule_is_part_of_signature(cached_edge_runner):
    runner, shape, _, inputs, calls, _ = cached_edge_runner
    runner._prepare_graph_attention(shape, inputs)
    inputs["verify_qo_indptr"][:] = [0, 3, 8]
    runner._prepare_graph_attention(shape, inputs)
    assert len(calls) == 6
    assert runner.attention_plan_cache_stats == {"hits": 2, "misses": 6}


def test_external_plan_identity_invalidates_last_signature(cached_edge_runner):
    runner, shape, wrappers, inputs, calls, _ = cached_edge_runner
    runner._prepare_graph_attention(shape, inputs)
    wrappers[0]._plan_info = object()
    runner._prepare_graph_attention(shape, inputs)
    assert len(calls) == 5
    assert calls[-1][0] is wrappers[0]


def test_failed_replan_invalidates_previous_signature_before_retry(cached_edge_runner):
    runner, shape, wrappers, inputs, calls, failure = cached_edge_runner
    runner._prepare_graph_attention(shape, inputs)
    previous_plan = wrappers[0]._plan_info
    inputs["assistant_local_kv_indptr"][:] = [0, 2, 4]
    failure[0] = wrappers[0]
    with pytest.raises(RuntimeError, match="mutating workspace"):
        runner._prepare_graph_attention(shape, inputs)
    assert wrappers[0]._plan_info is previous_plan
    assert wrappers[0]._resin_edge_last_host_plan is None
    inputs["assistant_local_kv_indptr"][:] = [0, 1, 3]
    runner._prepare_graph_attention(shape, inputs)
    assert len(calls) == 6
    assert wrappers[0].workspace_revision == 3
    assert runner.attention_plan_cache_stats == {"hits": 3, "misses": 6}
