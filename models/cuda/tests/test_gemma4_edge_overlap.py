"""Cache ownership and real Engine overlap regressions for Gemma Edge."""

from __future__ import annotations

import gc
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def runtime():
    from resin_models_cuda.gemma4 import edge_cuda

    return edge_cuda


class _Event:
    def __init__(self):
        self.ready = True

    def record(self, _stream):
        pass

    def query(self):
        return self.ready


def test_hidden_feedback_pool_respects_owner_and_consumer_event(runtime, monkeypatch):
    from resin.backend.cuda.completion import CudaContinuation, DeviceFeedback

    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: None)
    pool = runtime._EdgeHiddenPool("cpu", 8)
    lease = pool.acquire(4)
    lease.tensor.copy_(torch.arange(32).reshape(4, 8))
    lease.record_use()
    original_pointer = lease.tensor.data_ptr()
    feedback = runtime._EdgeDeviceFeedback(
        DeviceFeedback(torch.tensor([3, 4, 5, 6]), torch.tensor([65, 66, 67, 68])),
        lease,
        CudaContinuation().begin((10, 11, 12), (4, 160), {}),
    ).restrict(3)
    del lease
    gc.collect()
    other = pool.acquire(4)
    assert other.tensor.data_ptr() != original_pointer
    assert feedback.inputs()["token_ids"].shape == (4,)
    assert feedback.positions.tolist() == [65, 66, 67, 1]
    assert feedback.target_hidden[-1].tolist() == [0] * 8
    feedback.hidden_lease.event.ready = False
    del feedback
    gc.collect()
    third = pool.acquire(4)
    assert third.tensor.data_ptr() != original_pointer
    pool.pending[0][1].ready = True
    reused = pool.acquire(4)
    assert reused.tensor.data_ptr() == original_pointer
    assert pool.allocations == 3


def test_hidden_pool_quarantines_failed_event_record(runtime, monkeypatch):
    class Broken(_Event):
        def record(self, _stream):
            raise RuntimeError("record failed")

    monkeypatch.setattr(torch.cuda, "Event", Broken)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: None)
    pool = runtime._EdgeHiddenPool("cpu", 8)
    lease = pool.acquire(2)
    pointer = lease.tensor.data_ptr()
    with pytest.raises(RuntimeError, match="record failed"):
        lease.record_use()
    del lease
    gc.collect()
    assert pool.pending[0][1] is None
    assert pool.acquire(2).tensor.data_ptr() != pointer


def test_graph_metadata_uses_exact_device_positions(runtime, monkeypatch):
    from resin.backend.cuda.graphs import CudaGraphStep
    from resin.backend.sampling import SAMPLING_OFFSET_STRIDE
    from resin.models.gemma4.plan import PagedAttentionPlan
    from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner

    assistant = PagedAttentionPlan(
        qo_indptr=np.array([0, 1], dtype=np.int32),
        append_slots=np.empty(0, dtype=np.int64),
        first_token_positions=(0,),
        kv_lengths=(64,),
        kv_indptr=np.array([0, 1], dtype=np.int32),
        page_indices=np.array([1], dtype=np.int32),
        last_page_lengths=np.array([64], dtype=np.int32),
    )
    verify = PagedAttentionPlan(
        first_token_positions=(0,),
        kv_lengths=(67,),
        kv_indptr=np.array([0, 2], dtype=np.int32),
        page_indices=np.array([1, 2], dtype=np.int32),
        last_page_lengths=np.array([3], dtype=np.int32),
        qo_indptr=np.array([0, 4], dtype=np.int32),
        append_slots=np.array([127, 128, 129, 130]),
    )
    packed = Gemma4CudaRunner._spec_graph_page_inputs(
        "local",
        assistant,
        verify,
        active_rows=1,
        graph_rows=4,
        page_width=4,
        query_width=4,
    )
    assert np.asarray(packed["assistant_local_last_page_lengths"]).tolist() == [
        64,
        1,
        1,
        1,
    ]
    assert np.asarray(packed["verify_local_last_page_lengths"]).tolist() == [3, 5, 5, 5]
    inputs = {
        "positions": torch.tensor([511, 512, 8703, 1]),
        "sampling_offsets": torch.empty(4, dtype=torch.int64),
    }
    for kind in ("local", "global"):
        for prefix in ("assistant_", "verify_"):
            inputs[f"{prefix}{kind}_last_page_lengths"] = torch.empty(
                4, dtype=torch.int32
            )
        inputs[f"verify_{kind}_append_offsets"] = torch.empty(16, dtype=torch.int64)
    monkeypatch.setattr(
        Gemma4CudaRunner,
        "_new_speculative_graph_step",
        lambda _self, _shape: CudaGraphStep(
            inputs=inputs, output=None, launch=lambda: None
        ),
    )
    runner = object.__new__(runtime.Gemma4EdgeCudaRunner)
    runner.device = torch.device("cpu")
    runner.speculative_draft_tokens = 3
    step = runner._new_speculative_graph_step((4, 160))
    for positions in ([511, 512, 8703, 1], [513, 515, 8705, 1]):
        inputs["positions"].copy_(torch.tensor(positions))
        step.launch()
        assert inputs["sampling_offsets"].tolist() == [
            (p + 1) * SAMPLING_OFFSET_STRIDE for p in positions
        ]
        for kind in ("local", "global"):
            assert inputs[f"assistant_{kind}_last_page_lengths"].tolist() == [
                (p - 1) % 64 + 1 for p in positions
            ]
            assert inputs[f"verify_{kind}_last_page_lengths"].tolist() == [
                (p + 3) % 64 + 1 for p in positions
            ]
            assert inputs[f"verify_{kind}_append_offsets"].tolist() == [
                (p + q) % 64 for p in positions for q in range(4)
            ]


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
weights = pytest.mark.skipif(
    not all(
        os.getenv(name)
        for name in ("RESIN_TEST_CHECKPOINT", "RESIN_TEST_ASSISTANT_CHECKPOINT")
    ),
    reason="requires pinned target and assistant checkpoints",
)


@cuda
@torch.inference_mode()
def test_hidden_bank_cuda_ownership_growth_permutation_and_reuse(runtime):
    from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner

    runner = object.__new__(Gemma4CudaRunner)
    runner.device = torch.device("cuda")
    runner.config = SimpleNamespace(hidden_size=8)
    runner.max_num_seqs = 2
    runner._target_hidden, runner._hidden_bank = {}, None
    runner._hidden_free_slots, runner._hidden_slot_generation = [], 0
    runner._hidden_index_cache, runner._hidden_input_buffers = {}, {}
    runner._media_features = {}
    source = torch.arange(16, dtype=torch.bfloat16, device="cuda").reshape(2, 8)
    expected = source.clone()
    runner._set_target_hidden([101, 202], source)
    source.zero_()
    actual = runner._gather_target_hidden([202, 101], 4)
    torch.testing.assert_close(actual[:2], expected.flip(0), rtol=0, atol=0)
    assert torch.count_nonzero(actual[2:]) == 0
    pointer = actual.data_ptr()
    runner._set_target_hidden(
        [303, 404, 505], torch.full((3, 8), 7, dtype=torch.bfloat16, device="cuda")
    )
    runner.release([202, 404])
    runner._set_target_hidden(
        [606], torch.full((1, 8), 9, dtype=torch.bfloat16, device="cuda")
    )
    again = runner._gather_target_hidden([606, 101, 505], 4)
    assert again.data_ptr() == pointer
    torch.testing.assert_close(again[0], torch.full_like(again[0], 9), rtol=0, atol=0)
    torch.testing.assert_close(again[1], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(again[2], torch.full_like(again[2], 7), rtol=0, atol=0)
    assert torch.count_nonzero(again[-1]) == 0


@pytest.fixture
def real_runner(runtime):
    runner = runtime.Gemma4EdgeCudaRunner(
        Path(os.environ["RESIN_TEST_CHECKPOINT"]),
        device="cuda",
        assistant_model=os.environ["RESIN_TEST_ASSISTANT_CHECKPOINT"],
        speculative_draft_tokens=3,
        cache_parent_capacity=1024,
        cuda_graph_prefill="breakable",
        cuda_graph_decode="full",
        cuda_graph_decode_shapes=[(4, 160)],
    )
    runner.capture_cuda_graphs()
    # A managed runner binds to one Engine cache pool for its lifetime. Keep
    # that owner fixed while comparing the two submission paths after draining.
    facade = _ModeRunner(runner)
    runner._edge_test_engine = _new_engine(facade)
    runner._edge_test_facade = facade
    try:
        yield runner
    finally:
        runner._edge_test_engine.close()
    runner.close_cuda_graphs()
    del runner._edge_test_engine
    del runner._edge_test_facade
    facade.runner = None
    del runner
    gc.collect()
    torch.cuda.empty_cache()


@lru_cache
def _prompt_seed():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["RESIN_TEST_CHECKPOINT"], local_files_only=True
    )
    assert tokenizer is not None
    return tokenizer.encode(
        "The following report summarizes a research experiment. We measured latency, throughput, and numerical correctness. ",
        add_special_tokens=True,
    )


def _prompt(length):
    seed = _prompt_seed()
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


class _ModeRunner:
    """Compare real forward/enqueue with one Engine/cache owner per test."""

    def __init__(self, runner):
        self.runner = runner
        self.overlap = self.crossed = False
        self.previous = self.previous_tables = None
        self.chain = self.longest_chain = self.renewals = 0

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def forward(self, plan):
        return self.runner.forward(plan)

    def enqueue(self, plan, *, feedback=None):
        if not self.overlap:
            assert feedback is None
            return self.runner.forward(plan)
        if feedback is None:
            tables = tuple(
                (
                    tuple(map(int, group.block_slab_ids)),
                    tuple(map(int, group.block_table_start_indices)),
                )
                for group in plan.cache_groups
            )
            self.renewals += (
                self.previous_tables is not None and tables != self.previous_tables
            )
            self.previous_tables = tables
        self.chain = self.chain + 1 if feedback is not None else 0
        self.longest_chain = max(self.longest_chain, self.chain)
        if feedback is not None and self.previous is not None:
            # Anchor63 crosses a physical page even at minimum acceptance.
            self.crossed |= any(position % 64 == 63 for position in self.previous)
        self.previous = tuple(map(int, plan.positions))
        return self.runner.enqueue(plan, feedback=feedback)


def _new_engine(runner, *, running=4):
    from resin import Engine, SchedulerConfig

    return Engine(
        runner,
        context_length=10000,
        max_running_requests=running,
        prefix_cache=False,
        enable_overlap=True,
        scheduler_config=SchedulerConfig(prefill_chunk_size=8192),
    )


def _engine(runner, overlap, running=4):
    assert running <= 4
    engine = runner._edge_test_engine
    assert not engine.has_work()
    runner._edge_test_facade.overlap = overlap
    return engine


def _admit(engine, length, count=96, **kwargs: Any):
    from resin import SamplingParams

    policy: dict[str, Any] = {
        "stop_token_ids": [],
        "temperature": 0.0,
        "seed": 41,
        **kwargs,
    }
    return engine.submit(
        _prompt(length),
        sampling_params=SamplingParams(max_output_tokens=count, **policy),
    )


@cuda
@weights
@pytest.mark.parametrize("sampled", [False, True])
@torch.inference_mode()
def test_real_overlap_matches_sync_across_pages_windows_padding_and_feedback(
    real_runner, sampled
):
    results = []
    contexts = (
        (63, 64, 65, 511, 512, 513, 8191, 8192, 8193, 8703, 8704, 8705)
        if not sampled
        else (65, 8192, 8704)
    )
    policy = {} if not sampled else {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    for overlap in (False, True):
        engine = _engine(real_runner, overlap)
        outputs = {}
        for length in contexts:
            leases = [
                _admit(engine, length, **policy),
                _admit(engine, length + 1, **policy),
            ]
            while engine.has_work():
                engine.step()
            outputs[length] = [lease.take_result() for lease in leases]
        if overlap:
            assert engine.overlap_stats[0] > 0 and engine.overlap_stats[1] > 0
            assert real_runner._edge_hidden_pool.allocations <= 4
        results.append(outputs)
        assert engine.cache_free_parent_count == real_runner.cache_parent_capacity
    assert results[0] == results[1]
    assert not real_runner._target_hidden


@cuda
@weights
@torch.inference_mode()
def test_real_overlap_stop_cancel_and_rebatch_preserve_survivor_and_hidden_slots(
    real_runner,
):
    lengths = (64, 65, 513, 8704)
    engine = _engine(real_runner, False)
    leases = [_admit(engine, length, 128) for length in lengths]
    while engine.has_work():
        engine.step()
    expected = {
        length: lease.take_result()
        for length, lease in zip(lengths, leases, strict=True)
    }
    stop = next(
        token
        for index, token in enumerate(expected[64])
        if index >= 8 and expected[64].index(token) == index
    )
    stop_index = expected[64].index(stop)
    engine = _engine(real_runner, True, running=3)
    stopped = _admit(engine, 64, 128, stop_token_ids=[stop])
    canceled = _admit(engine, 65, 128)
    survivor = _admit(engine, 513, 128)
    while engine.overlap_stats[1] == 0:
        assert engine.has_work()
        engine.step()
    assert engine.cancel(canceled)
    canceled_tokens = canceled.take_result()
    assert canceled_tokens == expected[65][: len(canceled_tokens)]
    # A new request reuses released sequence slots while a survivor remains.
    replacement = _admit(engine, 8704, 96)
    while engine.has_work():
        engine.step()
    assert stopped.take_result() == expected[64][: stop_index + 1]
    assert survivor.take_result() == expected[513]
    assert replacement.take_result() == expected[8704][:96]
    assert engine.cache_free_parent_count == real_runner.cache_parent_capacity
    assert not real_runner._target_hidden


@cuda
@weights
@torch.inference_mode()
def test_real_reserved_overlap_crosses_pages_with_unequal_rows_padding_and_rebatch():
    from resin import SamplingParams
    from resin_models_cuda.gemma4.edge_cuda import Gemma4EdgeCudaRunner

    runner = Gemma4EdgeCudaRunner(
        Path(os.environ["RESIN_TEST_CHECKPOINT"]),
        device="cuda",
        assistant_model=os.environ["RESIN_TEST_ASSISTANT_CHECKPOINT"],
        speculative_draft_tokens=3,
        cache_parent_capacity=8192,
        cuda_graph_prefill="breakable",
        cuda_graph_decode="full",
        cuda_graph_decode_shapes=[(32, 160)],
    )
    runner.capture_cuda_graphs()

    facade = _ModeRunner(runner)
    engine = _new_engine(facade, running=32)
    lengths = ([63, 64, 65, 511, 512, 513, 8191, 8192, 8193, 8703, 8704, 8705] * 3)[:31]
    seed = [2] + [1000 + (i * 17) % 29000 for i in range(10000)]

    def submit(length, policy, count=160):
        return engine.submit(
            seed[:length],
            sampling_params=SamplingParams(
                max_output_tokens=count,
                stop_token_ids=[],
                seed=41,
                **policy,
            ),
        )

    try:
        for policy in (
            {"temperature": 0.0},
            {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        ):
            outputs = []
            for overlap in (False, True):
                assert not engine.has_work()
                facade.overlap, facade.previous = overlap, None
                before = engine.overlap_stats[1]
                leases = [submit(length, policy) for length in lengths]
                while engine.has_work():
                    engine.step()
                outputs.append([lease.take_result() for lease in leases])
                assert engine.cache_free_parent_count == runner.cache_parent_capacity
                assert not runner._target_hidden
                if overlap:
                    assert engine.overlap_stats[1] > before
                    assert facade.crossed
                    assert facade.renewals > 0
                    assert facade.longest_chain >= 16
                    assert runner._edge_hidden_pool.allocations <= 4
            assert outputs[0] == outputs[1]
        # Reordered and newly admitted rows must gather owned bank state rather
        # than retaining row views into a previous graph's output storage.
        facade.overlap = True
        before = engine.overlap_stats[1]
        leases = [submit(length, {"temperature": 0.0}, count=48) for length in lengths]
        while engine.overlap_stats[1] == before:
            engine.step()
        for lease in leases[::7]:
            if not lease.done:
                assert engine.cancel(lease)
        replacement = submit(8704, {"temperature": 0.0}, count=48)
        while engine.has_work():
            engine.step()
        assert len(replacement.take_result()) == 48
        assert engine.cache_free_parent_count == runner.cache_parent_capacity
        assert not runner._target_hidden
    finally:
        engine.close()
        runner.close_cuda_graphs()
        del runner
        gc.collect()
        torch.cuda.empty_cache()
