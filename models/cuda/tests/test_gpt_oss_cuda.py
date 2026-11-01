from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    CacheRetention,
    SamplingParams,
)
from resin.backend.cuda.graphs import default_decode_graph_shapes
from resin.backend.cuda.runner import CudaRunner
from resin.backend.cuda.sampling import FusedSampler
from resin.backend.resources import CachePoolBinding
from resin.backend.sampling import TorchSampler
from resin.models.gpt_oss.plan import (
    PAGE_SIZE,
    PREFILL_CHUNK_SIZE,
    SLIDING_WINDOW,
    WINDOW_RETENTION_TOKENS,
    WINDOW_TABLE_WIDTH,
)
from resin.models.gpt_oss.plan import (
    build_attention_batches as _build_attention_batches,
)
from resin_models_cuda.gpt_oss import cuda as gpt_oss_runtime
from resin_models_cuda.gpt_oss.cuda import (
    GptOssCudaRunner,
    _EmptyFa4AuxData,
    _route,
    _router_logits,
    plan_gpt_oss_kv_capacity,
)
from resin_models_cuda.gpt_oss.model import SinkAttention, TopKRouter
from resin_models_cuda.gpt_oss.paged_kv import paged_attention_host_inputs


def build_attention_batches(plan, *, parent_capacity: int = 2048):
    layout = CacheLayout(
        parent_capacity=parent_capacity,
        groups=[
            CacheGroupLayout(PAGE_SIZE),
            CacheGroupLayout(
                PAGE_SIZE,
                retention=CacheRetention.WINDOW,
                window_tokens=WINDOW_RETENTION_TOKENS,
                read_window_tokens=SLIDING_WINDOW,
            ),
        ],
    )
    return _build_attention_batches(plan, layout)


def _group_plan(group_index, rows, block_table_start_indices, positions, indptr):
    block_indptr = [0]
    block_slab_ids = []
    append_slab_ids = []
    append_block_offsets = []
    for sequence_index, row in enumerate(rows):
        block_slab_ids.extend(row)
        block_indptr.append(len(block_slab_ids))
        first = block_table_start_indices[sequence_index]
        token_start, token_end = indptr[sequence_index : sequence_index + 2]
        for position in positions[token_start:token_end]:
            append_slab_ids.append(row[position // 128 - first])
            append_block_offsets.append(position % 128)
    return CacheGroupPlan(
        group_index=group_index,
        blocks_per_slab=1,
        block_indptr=block_indptr,
        block_slab_ids=block_slab_ids,
        block_slotnums=[0] * len(block_slab_ids),
        block_table_start_indices=list(block_table_start_indices),
        append_slab_ids=append_slab_ids,
        append_slotnums=[0] * len(append_slab_ids),
        append_block_offsets=append_block_offsets,
    )


def _flat_plan(
    context_lengths,
    query_lengths,
    *,
    phase=BatchPhase.PREFILL,
    sampling_params=None,
):
    sequence_count = len(context_lengths)
    assert len(query_lengths) == sequence_count
    token_indptr = [0]
    positions = []
    history_rows = []
    window_rows = []
    window_first = []
    for sequence_index, (context, query) in enumerate(
        zip(context_lengths, query_lengths, strict=True)
    ):
        assert context >= 0 and query > 0
        positions.extend(range(context, context + query))
        token_indptr.append(len(positions))
        post_length = context + query
        last_block = (post_length - 1) // 128
        history_rows.append(
            tuple(
                1 + sequence_index * 128 + logical for logical in range(last_block + 1)
            )
        )
        first = max(0, last_block + 1 - WINDOW_TABLE_WIDTH)
        window_first.append(first)
        window_rows.append(
            tuple(
                1025 + sequence_index * 128 + logical % WINDOW_TABLE_WIDTH
                for logical in range(first, last_block + 1)
            )
        )

    history = _group_plan(
        0,
        history_rows,
        [0] * sequence_count,
        positions,
        token_indptr,
    )
    window = _group_plan(
        1,
        window_rows,
        window_first,
        positions,
        token_indptr,
    )
    return BatchPlan(
        sequence_ids=list(range(100, 100 + sequence_count)),
        context_lengths=list(context_lengths),
        token_ids=list(range(1, len(positions) + 1)),
        token_indptr=token_indptr,
        phase=phase,
        cache_groups=(history, window),
        sampling_params=sampling_params,
    )


def _replace_group(plan, index, **changes):
    groups = list(plan.cache_groups)
    fields = {
        name: getattr(groups[index], name)
        for name in (
            "group_index",
            "blocks_per_slab",
            "block_indptr",
            "block_slab_ids",
            "block_slotnums",
            "block_table_start_indices",
            "append_slab_ids",
            "append_slotnums",
            "append_block_offsets",
        )
    }
    groups[index] = CacheGroupPlan(**(fields | changes))
    return BatchPlan(
        sequence_ids=plan.sequence_ids,
        context_lengths=plan.context_lengths,
        token_ids=plan.token_ids,
        token_indptr=plan.token_indptr,
        phase=plan.phase,
        cache_groups=groups,
        sampling_params=plan.sampling_params,
    )


def _prefill_plan():
    return _flat_plan([0, 130], [128, 5])


def _decode_plan():
    return _flat_plan(
        [130, 260],
        [1, 1],
        phase=BatchPhase.DECODE,
    )


class TestGptOssCudaNativePlan:
    @pytest.mark.parametrize(
        "capacity,context_length,expected_width", [(128, 32769, 128), (512, 32769, 257)]
    )
    def test_default_graph_history_uses_cache_and_context_bounds(
        self, monkeypatch, capacity, context_length, expected_width
    ) -> None:
        class ConfiguredGraphs(Exception):
            pass

        config = {
            "model_type": "gpt_oss",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "head_dim": 4,
            "sliding_window": SLIDING_WINDOW,
            "max_position_embeddings": context_length,
        }
        model = SimpleNamespace(
            args=gpt_oss_runtime.GptOssConfig.from_dict(config),
            model=SimpleNamespace(layers=[], norm=torch.nn.Identity()),
        )
        received = []

        def initialize(runner, **options):
            runner.device = torch.device("cpu")
            runner.cuda_graph_decode = options["cuda_graph_decode"]

        def configure(runner, modes):
            received.append(modes["decode"].shapes)
            raise ConfiguredGraphs

        monkeypatch.setattr(CudaRunner, "__init__", initialize)
        monkeypatch.setattr(GptOssCudaRunner, "_configure_cuda_graphs", configure)
        monkeypatch.setattr(
            gpt_oss_runtime, "resolve_model_path", lambda *a, **k: "unused"
        )
        monkeypatch.setattr(
            gpt_oss_runtime, "load_model", lambda *a, **k: (model, config)
        )
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (100 * 1024**3,) * 2)
        with pytest.raises(ConfiguredGraphs):
            GptOssCudaRunner(
                device="cuda",
                cache_parent_capacity=capacity,
                kernel_attention=lambda: None,
            )
        assert received == [
            default_decode_graph_shapes(
                expected_width, speculative=False, max_batch_size=128
            )
        ]

    def test_empty_fa4_aux_data_is_a_zero_width_jit_argument(self) -> None:
        empty = SimpleNamespace(tensors=None, scalars=None)
        adapted = _EmptyFa4AuxData(empty)

        assert adapted.tensors is None
        assert adapted.scalars is None
        assert adapted.__c_pointers__() == []
        assert adapted.__get_mlir_types__() == []
        assert adapted.__new_from_mlir_values__([]) is adapted
        with pytest.raises(NotImplementedError, match="does not support"):
            _EmptyFa4AuxData(SimpleNamespace(tensors=[object()], scalars=None))

    @staticmethod
    def _binding_runner():
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        setattr(runner, "cache_layout", object())
        binding = CachePoolBinding(runner.cache_layout, resource="GPT-OSS KV")
        runner._cache_binding = binding
        return runner

    @staticmethod
    def _resources(runner, pool_id=41, layout=None):
        return SimpleNamespace(
            cache_pool_id=pool_id,
            cache_layout=runner.cache_layout if layout is None else layout,
        )

    def test_cuda_graphs_use_the_shared_static_step_lifecycle(self) -> None:
        parameters = inspect.signature(GptOssCudaRunner.__init__).parameters
        source = inspect.getsource(GptOssCudaRunner)
        assert "cuda_graph_decode_shapes" in parameters
        assert GptOssCudaRunner.__bases__ == (CudaRunner,)
        assert hasattr(GptOssCudaRunner, "cuda_graph_stats")
        assert hasattr(GptOssCudaRunner, "capture_cuda_graphs")
        assert "self._configure_cuda_graphs(" in source
        assert "def capture_cuda_graphs" not in source
        assert "CUDAGraph" not in source
        assert ".replay()" not in source

    def test_kernels_are_flat_explicit_constructor_arguments(self) -> None:
        parameters = inspect.signature(GptOssCudaRunner.__init__).parameters

        assert "kernel_attention" in parameters
        assert "kernel_moe" in parameters
        assert "kernels" not in parameters
        assert "moe_kernel" not in parameters
        with pytest.raises(TypeError, match="kernel_attention"):
            GptOssCudaRunner(device="cuda", kernel_attention=object())

    def test_graph_capture_inputs_use_only_the_reserved_null_page(self) -> None:
        inputs = GptOssCudaRunner._graph_capture_inputs((2, 4))

        assert inputs["token_indices"] == (0, 1)
        assert inputs["sequence_indices"] == (0, 1)
        assert inputs["full_page_table"] == (0,) * 8
        assert inputs["sliding_page_table"] == (0,) * (2 * WINDOW_TABLE_WIDTH)
        assert inputs["full_append_slots"] == (0, 1)
        assert inputs["sliding_append_slots"] == (0, 1)

    @pytest.mark.parametrize("rows,expected_bucket", [(3, 4), (4, 4), (6, 8)])
    def test_decode_graph_buckets_preserve_live_rows_and_isolate_padding(
        self, rows, expected_bucket
    ) -> None:
        plan = _flat_plan(
            [130 + 10 * row for row in range(rows)],
            [1] * rows,
            phase=BatchPhase.DECODE,
            sampling_params=[
                SamplingParams(temperature=0.7, top_p=0.9, top_k=5, seed=100 + row)
                for row in range(rows)
            ],
        )
        _, batch = build_attention_batches(plan)
        runner = object.__new__(GptOssCudaRunner)
        runner.parallel_groups = None
        runner.vocab_size = 16
        setattr(runner, "_sampler", FusedSampler)
        setattr(runner, "_cuda_graph_mode_enabled", lambda _: True)
        shapes = default_decode_graph_shapes(4, speculative=False, max_batch_size=128)
        runner._graph_logits = {shape: torch.empty(shape[0], 16) for shape in shapes}
        setattr(
            runner,
            "_select_cuda_graph",
            lambda _, match: next(shape for shape in shapes if match(shape)),
        )
        captured = []

        def replay(mode, shape, inputs):
            captured.append((mode, shape, inputs))
            return torch.arange(1000, 1000 + shape[0])

        setattr(runner, "_run_cuda_graph", replay)
        output = runner._graph_forward(plan, batch)
        assert output is not None
        assert output.token_ids.tolist() == list(range(1000, 1000 + rows))
        assert output.token_indptr.tolist() == list(range(rows + 1))
        mode, shape, inputs = captured.pop()
        assert mode == "decode"
        assert shape == (expected_bucket, 4)
        expected = paged_attention_host_inputs(
            batch, {"full": 4, "sliding": WINDOW_TABLE_WIDTH}
        )
        for name, values in expected.items():
            np.testing.assert_array_equal(inputs[name][: len(values)], values)
        for name, values in FusedSampler.host_inputs(plan, 16).items():
            assert inputs[name][:rows] == values
        for name in ("token_ids", "positions"):
            np.testing.assert_array_equal(inputs[name][:rows], getattr(plan, name))
            np.testing.assert_array_equal(inputs[name][rows:], 0)
        for name in ("token_indices", "sequence_indices"):
            np.testing.assert_array_equal(inputs[name], np.arange(expected_bucket))
        for name, width in (("full", 4), ("sliding", WINDOW_TABLE_WIDTH)):
            np.testing.assert_array_equal(
                inputs[f"{name}_page_table"][rows * width :], 0
            )
            np.testing.assert_array_equal(inputs[f"{name}_seqused_k"][rows:], 1)
            np.testing.assert_array_equal(
                inputs[f"{name}_append_slots"][rows:], np.arange(rows, expected_bucket)
            )
            assert all(
                slot < PAGE_SIZE for slot in inputs[f"{name}_append_slots"][rows:]
            )
        assert inputs["sampling_stochastic_mask"][rows:] == (False,) * (
            expected_bucket - rows
        )

    @pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
    def test_prefill_regions_are_inactive_during_decode_fallback(
        self, monkeypatch, phase
    ) -> None:
        plan = _flat_plan([130, 260], [1, 1], phase=phase)
        batches = build_attention_batches(plan)
        runner = object.__new__(GptOssCudaRunner)
        setattr(runner, "device", "cpu")
        runner.hidden_size = 2
        setattr(runner, "cache_layout", object())
        runner.cuda_graph_prefill = "breakable"
        runner._prefill_graph_active = False
        runner.parallel_groups = None
        setattr(runner, "_cuda_failure", nullcontext())
        setattr(runner, "_check_cuda_graphs", lambda _phase: None)
        setattr(runner, "_graph_forward", lambda *_: None)
        setattr(runner, "_model", SimpleNamespace(lm_head=lambda hidden: hidden))
        setattr(runner, "_sampler", lambda logits, _: logits.argmax(dim=-1))
        setattr(
            runner,
            "_tensorize",
            lambda host, *_: SimpleNamespace(
                sequence_indices=torch.tensor(host.sequence_indices)
            ),
        )
        events = []

        def forward(*_, **kwargs):
            events.append(runner._prefill_graph_active)
            return torch.tensor([[0, 1], [1, 0]], dtype=torch.bfloat16)

        setattr(runner, "_forward_batch", forward)
        monkeypatch.setattr(
            gpt_oss_runtime, "build_attention_batches", lambda *_: batches
        )
        monkeypatch.setattr(
            gpt_oss_runtime,
            "TensorStager",
            lambda _: SimpleNamespace(
                device=lambda _name, values, dtype: torch.tensor(values, dtype=dtype)
            ),
        )
        assert runner.forward(plan).token_ids.tolist() == [1, 0]
        assert events == [phase == BatchPhase.PREFILL]
        assert not runner._prefill_graph_active

    def test_kv_capacity_plan_uses_post_load_free_memory(self) -> None:
        gib = 1024**3
        plan = plan_gpt_oss_kv_capacity(
            100 * gib,
            45 * gib,
            0.75,
        )

        assert plan.protected_free_bytes == 25 * gib
        assert plan.cache_budget_bytes == 20 * gib
        assert plan.bytes_per_parent == 4_718_592
        assert plan.parent_capacity == 4_550
        assert plan.allocation_bytes <= plan.cache_budget_bytes

        tp4 = plan_gpt_oss_kv_capacity(
            100 * gib,
            45 * gib,
            0.75,
            tensor_parallel_size=4,
        )
        assert tp4.bytes_per_parent == plan.bytes_per_parent // 4
        assert tp4.parent_capacity > plan.parent_capacity

        with pytest.raises(ValueError, match="tensor_parallel_size"):
            plan_gpt_oss_kv_capacity(
                100 * gib,
                45 * gib,
                0.75,
                tensor_parallel_size=0,
            )

    def test_runner_exposes_only_the_flat_grouped_cache_path(self) -> None:
        source = inspect.getsource(GptOssCudaRunner)
        parameters = inspect.signature(GptOssCudaRunner.__init__).parameters
        assert "RunnerConfig(" in source
        assert "continuation_prefill=True" in source
        assert "prefill_chunk_size=self.prefill_chunk_size" in source
        assert "max_batch_tokens=self.prefill_chunk_size" in source
        assert "self._configure_cache_binding(" in source
        assert hasattr(GptOssCudaRunner, "bind_resources")
        assert "CacheGroupLayout(self.page_size)" in source
        assert "retention=CacheRetention.WINDOW" in source
        assert "build_attention_batches" in source
        assert "tensor_parallel" in parameters
        assert "fast_allreduce" in parameters
        assert source.count("self.tensor_parallel.sum(") == 1
        assert source.count("self._allreduce_residual_rmsnorm(") == 2

    def test_runner_owns_gpt_oss_router_policy_directly(self) -> None:
        source = inspect.getsource(GptOssCudaRunner)

        assert GptOssCudaRunner.__bases__ == (CudaRunner,)
        assert "_router_logits(" in source
        assert "_route(" in source

    def test_cache_pool_binding_is_exact_idempotent_and_permanent(self) -> None:
        runner = self._binding_runner()

        assert runner.bind_resources(self._resources(runner)) is None
        assert runner.bind_resources(self._resources(runner)) is None
        fresh = self._binding_runner()
        with pytest.raises(ValueError, match="frozen layout object"):
            fresh.bind_resources(self._resources(fresh, layout=object()))
        with pytest.raises(ValueError, match="different cache pool"):
            runner.bind_resources(self._resources(runner, pool_id=42))
        for invalid in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                fresh = self._binding_runner()
                fresh.bind_resources(self._resources(fresh, pool_id=invalid))

    def test_prefill_tables_are_derived_from_history_and_window_csr(self) -> None:
        batches, decode = build_attention_batches(_prefill_plan())

        assert decode is None
        assert len(batches) == 1
        batch = batches[0]
        assert batch.phase == BatchPhase.PREFILL
        assert batch.sequence_indices == (0, 1)
        assert batch.query_lengths == (128, 5)
        assert batch.token_indices[:3].tolist() == [0, 1, 2]
        assert batch.token_indices[-3:].tolist() == [130, 131, 132]
        assert batch.group("full").width == 2
        assert batch.group("full").page_table.tolist() == [1, 1, 129, 130]
        assert batch.group("full").lengths == (128, 135)
        assert batch.group("sliding").width == WINDOW_TABLE_WIDTH
        assert (
            batch.group("sliding").page_table[:WINDOW_TABLE_WIDTH].tolist()
            == [1025] * WINDOW_TABLE_WIDTH
        )
        assert batch.group("sliding").page_table[
            WINDOW_TABLE_WIDTH : WINDOW_TABLE_WIDTH + 2
        ].tolist() == [1153, 1154]
        assert batch.group("sliding").lengths == (128, 135)
        assert batch.group("full").append_slots[0] == 128
        assert batch.group("full").append_slots[-1] == 130 * 128 + 6
        assert batch.group("sliding").append_slots[-1] == 1154 * 128 + 6

    def test_one_token_continuation_remains_explicit_prefill(self) -> None:
        plan = _flat_plan([260], [1])
        prefill, decode = build_attention_batches(plan)

        assert decode is None
        assert len(prefill) == 1
        batch = prefill[0]
        assert batch.query_lengths == (1,)
        assert batch.token_indices.tolist() == [0]
        assert batch.group("full").page_table.tolist() == [1, 2, 3]
        assert batch.group("full").lengths == (261,)
        assert batch.group("sliding").page_table[:3].tolist() == [1025, 1026, 1027]
        assert batch.group("sliding").lengths == (261,)

    def test_graph_padding_repeats_only_pages_owned_by_each_row(self) -> None:
        _, batch = build_attention_batches(_decode_plan())
        assert batch is not None
        inputs = paged_attention_host_inputs(
            batch, {"full": 4, "sliding": WINDOW_TABLE_WIDTH}
        )
        assert inputs["full_page_table"].tolist() == [
            1,
            2,
            1,
            1,
            129,
            130,
            131,
            129,
        ]

    def test_decode_tables_and_append_slots_come_from_sparse_groups(self) -> None:
        _, batch = build_attention_batches(_decode_plan())

        assert batch is not None
        assert batch is not None
        assert batch.sequence_indices == (0, 1)
        assert batch.query_lengths == (1, 1)
        assert batch.group("full").width == 3
        assert batch.group("full").page_table.tolist() == [1, 2, 1, 129, 130, 131]
        assert batch.group("full").lengths == (131, 261)
        assert batch.group("sliding").page_table[:3].tolist() == [1025, 1026, 1025]
        assert batch.group("sliding").page_table[
            WINDOW_TABLE_WIDTH : WINDOW_TABLE_WIDTH + 3
        ].tolist() == [1153, 1154, 1155]
        assert batch.group("sliding").lengths == (131, 261)
        assert batch.group("sliding").append_slots.tolist() == [
            1026 * 128 + 2,
            1155 * 128 + 4,
        ]
        assert batch.group("full").append_slots.tolist() == [2 * 128 + 2, 131 * 128 + 4]

    def test_frozen_append_destinations_are_a_validation_oracle(self) -> None:
        plan = _prefill_plan()
        append = plan.cache_groups[1].append_slab_ids.copy()
        append[0] += 1
        with pytest.raises(ValueError, match="append target"):
            build_attention_batches(_replace_group(plan, 1, append_slab_ids=append))

    def test_group_order_suffix_and_parent_ownership_fail_closed(self) -> None:
        with pytest.raises(ValueError, match="ordered"):
            _replace_group(_prefill_plan(), 0, group_index=1)

        plan = _flat_plan([9000], [1])
        with pytest.raises(ValueError, match="active range"):
            build_attention_batches(
                _replace_group(plan, 1, block_table_start_indices=[0])
            )

        plan = _flat_plan([0], [1])
        with pytest.raises(ValueError, match="cannot address"):
            build_attention_batches(
                _replace_group(plan, 1, block_slab_ids=[1], append_slab_ids=[1])
            )

        with pytest.raises(ValueError, match="outside"):
            build_attention_batches(plan, parent_capacity=128)

    def test_multi_page_prefill_is_validated_before_device_work(self) -> None:
        prefill, decode = build_attention_batches(_flat_plan([0], [PREFILL_CHUNK_SIZE]))

        assert decode is None
        assert prefill[0].query_lengths == (PREFILL_CHUNK_SIZE,)
        assert prefill[0].group("full").width == PREFILL_CHUNK_SIZE // 128
        assert prefill[0].group("sliding").width == WINDOW_TABLE_WIDTH
        assert len(set(prefill[0].group("sliding").append_slots)) == PREFILL_CHUNK_SIZE

        with pytest.raises(ValueError, match="exceeds 8192"):
            build_attention_batches(_flat_plan([0], [PREFILL_CHUNK_SIZE + 1]))

    def test_cuda_attention_has_no_fp32_promotion_or_kv_head_expansion(self) -> None:
        source = "\n".join(
            inspect.getsource(method)
            for method in (
                GptOssCudaRunner._attention,
                GptOssCudaRunner._attention_project,
                GptOssCudaRunner._attention_core,
            )
        )
        assert ".float(" not in source
        assert "repeat_interleave" not in source
        assert "repeat(" not in source
        assert source.count("qkv_proj(") == 1
        assert "attention.q_proj(" not in source
        assert "attention.k_proj(" not in source
        assert "attention.v_proj(" not in source
        assert "index_copy_" in source

    def test_runner_has_no_python_request_cache_or_page_allocator(self) -> None:
        source = inspect.getsource(GptOssCudaRunner)
        assert "_request_pages" not in source
        assert "_request_lengths" not in source
        assert "_free_pages" not in source
        assert "_logical_pages" not in source
        assert "_sequence_cache" not in source

    def test_one_zeroed_parent_allocation_backs_both_group_views(self) -> None:
        from resin_models_cuda.gpt_oss.paged_kv import PagedKvArena

        source = inspect.getsource(GptOssCudaRunner.__init__)
        arena_source = inspect.getsource(PagedKvArena.__init__)
        assert "self._kv = PagedKvArena(" in source
        assert arena_source.count("torch.zeros(") == 1
        assert "self.capacity + 1" in arena_source
        assert "self.keys = self.storage[0]" in arena_source
        assert "self.values = self.storage[1]" in arena_source


class TestGptOssCudaTinyOracle:
    def test_attention_tp_projection_requests_fp32_output_from_bf16_gemm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        setattr(runner, "tensor_parallel", SimpleNamespace(enabled=True))
        projection = torch.nn.Linear(4, 3, bias=True, dtype=torch.bfloat16)
        hidden = torch.randn(2, 4, dtype=torch.bfloat16)
        calls = []
        original_mm = torch.mm

        def mm(left, right, *, out_dtype):
            calls.append((left.dtype, right.dtype, out_dtype))
            # CUDA supports this output-dtype overload; emulate its arithmetic
            # on CPU while asserting that input tensors stay BF16.
            return original_mm(left.float(), right.float())

        monkeypatch.setattr(gpt_oss_runtime.torch, "mm", mm)
        output = runner._attention_output_projection(projection, hidden)
        expected = torch.nn.functional.linear(
            hidden.float(), projection.weight.float(), projection.bias.float()
        )
        assert calls == [(torch.bfloat16, torch.bfloat16, torch.float32)]
        assert output.dtype == torch.float32
        torch.testing.assert_close(output, expected)

        setattr(runner, "tensor_parallel", SimpleNamespace(enabled=False))
        calls.clear()
        torch.testing.assert_close(
            runner._attention_output_projection(projection, hidden), projection(hidden)
        )
        assert calls == []

    def test_attention_tp_reduces_fp32_before_casting_for_residual_norm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        # The BF16-only fused all-reduce cannot preserve FP32 GEMM partials.
        monkeypatch.setattr(runner, "_allreduce_workspace", object(), raising=False)
        partial = torch.full((1, 2, 4), 1000.75, dtype=torch.float32)
        residual = torch.full((1, 2, 4), 0.5, dtype=torch.bfloat16)
        calls = []

        def reduce(tensor):
            assert tensor.dtype == torch.float32
            calls.append("reduce")
            return tensor + (-998.75)

        def norm(tensor, weight, *, residual, eps, prenorm):
            assert tensor.dtype == residual.dtype == torch.bfloat16
            torch.testing.assert_close(tensor, torch.full_like(residual, 2))
            calls.append("norm")
            return tensor, tensor + residual

        setattr(runner, "tensor_parallel", SimpleNamespace(sum=reduce))
        monkeypatch.setattr(gpt_oss_runtime, "rmsnorm", norm)
        normalized, combined = runner._allreduce_residual_rmsnorm(
            partial, residual, torch.ones(4, dtype=torch.bfloat16), 1e-5
        )
        assert calls == ["reduce", "norm"]
        torch.testing.assert_close(normalized, torch.full_like(residual, 2))
        torch.testing.assert_close(combined, torch.full_like(residual, 2.5))

    def test_flashinfer_allreduce_fuses_residual_and_rmsnorm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        monkeypatch.setattr(
            runner,
            "_allreduce_workspace",
            SimpleNamespace(backend="trtllm"),
            raising=False,
        )
        partial = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        residual = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        weight = torch.randn(8, dtype=torch.bfloat16)
        eps = 1e-5
        calls = []

        def fused(**kwargs):
            calls.append(kwargs)
            combined = kwargs["input"] + kwargs["residual_in"]
            kwargs["residual_out"].copy_(combined)
            normalized = combined.float()
            normalized *= torch.rsqrt(
                normalized.square().mean(-1, keepdim=True) + kwargs["rms_eps"]
            )
            kwargs["norm_out"].copy_((normalized * kwargs["rms_gamma"]).bfloat16())
            return kwargs["norm_out"]

        monkeypatch.setattr("resin_models_cuda.gpt_oss.cuda.allreduce_fusion", fused)
        normalized, combined = runner._allreduce_residual_rmsnorm(
            partial, residual, weight, eps
        )

        expected_combined = partial + residual
        expected_normalized = expected_combined.float()
        expected_normalized *= torch.rsqrt(
            expected_normalized.square().mean(-1, keepdim=True) + eps
        )
        expected_normalized = (expected_normalized * weight).bfloat16()
        torch.testing.assert_close(combined, expected_combined)
        torch.testing.assert_close(normalized, expected_normalized)
        assert "use_oneshot" not in calls[0]
        assert not calls[0]["launch_with_pdl"]
        assert calls[0]["trigger_completion_at_end"]

    def test_packed_qkv_matches_three_loaded_projections(self) -> None:
        import torch

        # This projection-only fixture supplies the checkpoint-loaded fields.
        attention = cast(
            SinkAttention,
            SimpleNamespace(
                q_proj=torch.nn.Linear(8, 12, bias=True),
                k_proj=torch.nn.Linear(8, 4, bias=True),
                v_proj=torch.nn.Linear(8, 4, bias=True),
            ),
        )
        hidden = torch.randn(7, 8)
        expected = torch.cat(
            (
                attention.q_proj(hidden),
                attention.k_proj(hidden),
                attention.v_proj(hidden),
            ),
            dim=-1,
        )

        SinkAttention.pack_qkv_projection(attention)

        torch.testing.assert_close(attention.qkv_proj(hidden), expected)
        assert not hasattr(attention, "q_proj")
        assert not hasattr(attention, "k_proj")
        assert not hasattr(attention, "v_proj")

    def test_router_uses_tinygemm_through_128_rows_then_linear(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        router = cast(
            TopKRouter,
            SimpleNamespace(
                weight=torch.randn(16, 64, dtype=torch.bfloat16),
                bias=torch.randn(16, dtype=torch.bfloat16),
            ),
        )
        calls = []

        def tinygemm(input, weight, output, bias, *, use_pdl):
            calls.append((input.shape[0], use_pdl))
            output.copy_(torch.nn.functional.linear(input, weight, bias))

        monkeypatch.setattr("resin_models_cuda.gpt_oss.cuda.tinygemm_bf16", tinygemm)
        for rows in (1, 2, 4, 8, 16, 32, 64, 128):
            hidden = torch.randn(rows, 64, dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(hidden, router.weight, router.bias)
            torch.testing.assert_close(
                _router_logits(router, hidden), expected, rtol=0, atol=0
            )

        hidden = torch.randn(129, 64, dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(hidden, router.weight, router.bias)
        torch.testing.assert_close(
            _router_logits(router, hidden), expected, rtol=0, atol=0
        )

        assert calls == [
            (1, False),
            (2, False),
            (4, False),
            (8, False),
            (16, False),
            (32, False),
            (64, False),
            (128, False),
        ]

    @pytest.mark.parametrize("experts", [32, 128])
    def test_official_top4_router_uses_cutile_and_other_layouts_fall_back(
        self, monkeypatch: pytest.MonkeyPatch, experts: int
    ) -> None:
        import torch

        official = cast(
            TopKRouter, SimpleNamespace(top_k=4, benchmark_balanced_experts=False)
        )
        logits = torch.randn(32, experts, dtype=torch.bfloat16)
        expected_values, expected_indices = torch.topk(logits, 4, dim=-1, sorted=True)
        expected_weights = torch.softmax(expected_values, dim=-1)
        calls = []

        def top4_softmax(actual_logits):
            calls.append(actual_logits.shape)
            return expected_indices.to(torch.int32), expected_weights.to(torch.float32)

        with monkeypatch.context() as context:
            context.setattr("resin_models_cuda.gpt_oss.cuda.top4_softmax", top4_softmax)
            indices, weights = _route(official, logits)

        assert calls == [torch.Size([32, experts])]
        torch.testing.assert_close(indices, expected_indices.to(torch.int32))
        torch.testing.assert_close(weights, expected_weights.to(torch.float32))

        generic = cast(
            TopKRouter, SimpleNamespace(top_k=2, benchmark_balanced_experts=False)
        )
        indices, weights = _route(generic, logits[:, :16])
        values, expected_indices = torch.topk(logits[:, :16], 2, dim=-1, sorted=True)
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights, torch.softmax(values, dim=-1))

        large_logits = torch.randn(8192, experts, dtype=torch.bfloat16)
        large_values, large_indices = torch.topk(large_logits, 4, dim=-1, sorted=True)
        fused_calls = []

        def fused(actual_logits):
            fused_calls.append(actual_logits)
            return large_indices.int(), torch.softmax(large_values, dim=-1)

        with monkeypatch.context() as context:
            context.setattr("resin_models_cuda.gpt_oss.cuda.top4_softmax", fused)
            _route(official, large_logits)
        assert len(fused_calls) == 1
        assert fused_calls[0] is large_logits

    def test_synthetic_routes_balance_ep_owners_in_both_router_paths(self):
        from resin.bench import BenchmarkRunner
        from resin.models.gpt_oss.config import GptOssConfig

        router = TopKRouter(
            cast(
                GptOssConfig,
                SimpleNamespace(
                    num_experts_per_tok=4,
                    num_local_experts=32,
                    hidden_size=8,
                ),
            )
        )
        with torch.no_grad():
            router.weight.zero_()
            router.bias.copy_(torch.arange(32))
        hidden = torch.zeros(1, 8)
        assert (router(hidden)[2] // 16 == 1).all()
        BenchmarkRunner(
            SimpleNamespace(
                _model=torch.nn.Sequential(router),
                parallel_groups=SimpleNamespace(
                    ep=SimpleNamespace(size=2),
                    attn_dp=SimpleNamespace(local_rank=1),
                ),
            ),
            balanced_experts=True,
        )
        logits, weights, indices = router(hidden)
        assert indices.tolist() == [[2, 18, 3, 19]]
        cuda_indices, cuda_weights = _route(router, logits)
        torch.testing.assert_close(cuda_indices, indices)
        torch.testing.assert_close(cuda_weights, weights)
        torch.testing.assert_close(weights, logits.gather(1, indices).softmax(-1))

    def test_static_decode_step_restores_rust_sequence_order_before_sampling(
        self,
    ) -> None:
        import torch

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        setattr(runner, "_sampler", TorchSampler())
        runner.parallel_groups = None
        setattr(runner, "device", "cpu")
        runner.hidden_size = 3
        runner.vocab_size = 3
        runner._graph_logits = {}
        setattr(runner, "_model", SimpleNamespace(lm_head=lambda hidden: hidden))
        setattr(
            runner,
            "_forward_batch",
            lambda *_args: torch.tensor(
                [[0.0, 5.0, 1.0], [4.0, 0.0, 1.0]], dtype=torch.bfloat16
            ),
        )
        step = runner._new_decode_graph_step((2, 1))
        step.inputs["sequence_indices"].copy_(torch.tensor([1, 0]))

        step.launch()

        assert step.output.tolist() == [0, 1]

    def test_rotary_path_matches_reference_formula_without_dtype_promotion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
        setattr(runner, "_torch", torch)
        hidden = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -3.0]],
                [[-2.0, 1.0, 0.25, 0.75], [4.0, 3.0, 2.0, 1.0]],
            ],
            dtype=torch.bfloat16,
        )
        cos = torch.tensor([[[1.0, 0.5], [0.25, 1.0]]], dtype=torch.bfloat16)
        sin = torch.tensor([[[0.0, 0.5], [1.0, 0.0]]], dtype=torch.bfloat16)

        def apply_rotary(hidden, cos, sin, *, interleaved, inplace):
            assert not interleaved
            assert inplace
            first, second = hidden.chunk(2, dim=-1)
            hidden.copy_(
                torch.cat(
                    (
                        first * cos.unsqueeze(0).unsqueeze(2)
                        - second * sin.unsqueeze(0).unsqueeze(2),
                        second * cos.unsqueeze(0).unsqueeze(2)
                        + first * sin.unsqueeze(0).unsqueeze(2),
                    ),
                    dim=-1,
                )
            )
            return hidden

        first, second = hidden.chunk(2, dim=-1)
        expected = torch.cat(
            (
                first * cos.squeeze(0).unsqueeze(1)
                - second * sin.squeeze(0).unsqueeze(1),
                second * cos.squeeze(0).unsqueeze(1)
                + first * sin.squeeze(0).unsqueeze(1),
            ),
            dim=-1,
        )

        monkeypatch.setattr("resin_models_cuda.gpt_oss.cuda.apply_rotary", apply_rotary)
        actual = runner._apply_yarn_rotary(hidden, cos, sin)

        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
