"""Trace the actual Engine continuation path, including worst-case acceptance."""

from types import SimpleNamespace

import pytest
from output_helpers import forward_output

from resin import BatchPhase, CacheGroupLayout, CacheLayout, Engine, RunnerConfig
from resin.bench import BenchCase, BenchmarkRunner, decode_output_budget, run_sweep


class ImmediateDeviceRunner:
    """Async contract with immediate device work; scheduling is the real Engine."""

    vocab_size = 1024

    def __init__(self, drafts, extra_write_row):
        self.speculative_draft_tokens = drafts
        self.runner_config = RunnerConfig(
            cache_layout=CacheLayout(
                parent_capacity=16,
                groups=[CacheGroupLayout(4096)],
                decode_write_extra_tokens=extra_write_row,
            ),
            max_accepted_per_step=drafts + 1,
        )
        self.synchronous_decodes = 0
        self.enqueued = 0

    def bind_resources(self, resources):
        pass

    def output(self, plan, feedback=None):
        tokens = plan.token_ids.tolist() if feedback is None else feedback.tokens
        positions = (
            plan.context_lengths.tolist() if feedback is None else feedback.positions
        )
        rows = [
            [token + index + 1 for index in range(self.speculative_draft_tokens + 1)]
            for token in tokens
        ]
        return forward_output(rows), SimpleNamespace(
            tokens=[row[-1] for row in rows],
            positions=[pos + len(row) for pos, row in zip(positions, rows)],
        )

    def forward(self, plan):
        if plan.phase == BatchPhase.PREFILL:
            return forward_output([2] * plan.num_sequences)
        self.synchronous_decodes += 1
        return self.output(plan)[0]

    def enqueue(self, plan, *, feedback=None):
        self.enqueued += 1
        output, next_feedback = self.output(plan, feedback)
        return SimpleNamespace(feedback=next_feedback, result=lambda: output)


def exercise(*, drafts, extra_write_row, warmup, passes, declared=True):
    raw = ImmediateDeviceRunner(drafts, extra_write_row)
    observed = BenchmarkRunner(raw)
    # Exact public capacity from the admission helper, with no cache-page edge
    # anywhere in the window. Output-budget failures cannot hide behind pages.
    budget = decode_output_budget(
        warmup_steps=warmup,
        measure_steps=12,
        draft_tokens=drafts,
        overlap=True,
        trace_passes=passes,
        max_accepted_per_step=raw.runner_config.max_accepted_per_step,
    )
    engine = Engine(observed, context_length=64 + budget, max_running_requests=2)
    trace_stats = []

    def trace(case, run_pass):
        assert engine.overlap_pending and engine.overlap_ready
        for _ in range(passes):
            before = engine.overlap_stats
            value = run_pass()
            trace_stats.append(
                {
                    "delta": tuple(a - b for a, b in zip(engine.overlap_stats, before)),
                    "ready": engine.overlap_ready,
                    "pending": engine.overlap_pending,
                    "tokens": value["committed_tokens"],
                }
            )

    try:
        record = run_sweep(
            engine,
            [BenchCase(phase="decode", context_length=64, batch_size=2)],
            runner=observed,
            prompt_tokens=[1],
            warmup_steps=warmup,
            measure_steps=12,
            trace_step=trace,
            trace_decode_passes=passes if declared else None,
        )[0]
        assert not engine.has_work()
    finally:
        engine.close()
    return raw, record, trace_stats


@pytest.mark.parametrize("drafts,extra_write_row", [(0, 0), (3, 0), (3, 1), (5, 1)])
@pytest.mark.parametrize("warmup", [0, 3])
@pytest.mark.parametrize("passes", [1, 3, 5])
def test_every_trace_completion_retains_real_engine_overlap(
    drafts, extra_write_row, warmup, passes
):
    raw, record, traced = exercise(
        drafts=drafts, extra_write_row=extra_write_row, warmup=warmup, passes=passes
    )
    assert record["committed_tokens"] == 12 * 2 * (drafts + 1)
    assert record["overlap_counts"]["continuations"] == 12
    assert raw.synchronous_decodes == 0
    assert (
        traced
        == [dict(delta=(1, 1), ready=True, pending=True, tokens=2 * (drafts + 1))]
        * passes
    )


def test_plain_trace_callback_defaults_to_three_overlap_completions():
    raw, _, traced = exercise(
        drafts=0, extra_write_row=0, warmup=3, passes=3, declared=False
    )
    assert raw.synchronous_decodes == 0
    assert all(row["delta"] == (1, 1) for row in traced)


def test_historical_trace_budget_reproduces_the_terminal_drain(monkeypatch):
    # Independent negative control: the historical 19-token budget is adequate
    # to finish three trace calls, but insufficient to keep them overlapped.
    monkeypatch.setattr("resin.bench.decode_output_budget", lambda **kwargs: 19)
    raw, record, traced = exercise(drafts=0, extra_write_row=0, warmup=3, passes=3)
    assert record["overlap_counts"] == {"enqueues": 12, "continuations": 12}
    assert [row["delta"] for row in traced] == [(1, 1), (0, 0), (0, 0)]
    assert [row["ready"] for row in traced] == [False, True, True]
    assert raw.synchronous_decodes == 1


@pytest.mark.parametrize("drafts", [0, 3, 5])
@pytest.mark.parametrize("overlap", [False, True])
def test_untraced_admission_budget_is_unchanged(drafts, overlap):
    assert decode_output_budget(
        warmup_steps=3, measure_steps=12, draft_tokens=drafts, overlap=overlap
    ) == 1 + (3 + 12 + 1 + int(overlap)) * (drafts + 1)


@pytest.mark.parametrize("passes", [0, -1, True, 2.5])
def test_invalid_callback_count_fails_before_admission(passes):
    raw = ImmediateDeviceRunner(0, 0)
    runner = BenchmarkRunner(raw)
    engine = Engine(runner, context_length=128)
    try:
        with pytest.raises(ValueError, match="positive integer"):
            run_sweep(
                engine,
                [BenchCase(phase="decode", context_length=4, batch_size=1)],
                runner=runner,
                prompt_tokens=[1],
                trace_step=lambda case, step: None,
                trace_decode_passes=passes,
            )
        assert not engine.has_work() and raw.enqueued == 0
    finally:
        engine.close()


def test_callback_overrun_fails_before_an_unreserved_forward():
    raw = ImmediateDeviceRunner(0, 0)
    runner = BenchmarkRunner(raw)
    engine = Engine(runner, context_length=128)
    calls = []

    def trace(case, step):
        for _ in range(3):
            calls.append(raw.enqueued)
            step()

    try:
        with pytest.raises(ValueError, match="exceeded reserved decode completions"):
            run_sweep(
                engine,
                [BenchCase(phase="decode", context_length=4, batch_size=1)],
                runner=runner,
                prompt_tokens=[1],
                warmup_steps=0,
                measure_steps=1,
                trace_step=trace,
                trace_decode_passes=2,
            )
        assert len(calls) == 3 and raw.enqueued == calls[-1]
        assert not engine.has_work()
    finally:
        engine.close()


def test_advertised_callback_count_controls_reservation():
    raw = ImmediateDeviceRunner(0, 0)
    runner = BenchmarkRunner(raw)
    # Exactly the five-pass budget: one prefill + one measured + five traced
    # outputs + three reserved tokens. The old implicit three-pass budget fails.
    engine = Engine(runner, context_length=4 + 10)
    traced = []

    def trace(case, step):
        for _ in range(5):
            before = engine.overlap_stats
            step()
            traced.append(tuple(a - b for a, b in zip(engine.overlap_stats, before)))
            assert engine.overlap_pending and engine.overlap_ready

    setattr(trace, "decode_passes", 5)
    try:
        run_sweep(
            engine,
            [BenchCase(phase="decode", context_length=4, batch_size=1)],
            runner=runner,
            prompt_tokens=[1],
            warmup_steps=0,
            measure_steps=1,
            trace_step=trace,
        )
        assert traced == [(1, 1)] * 5
    finally:
        engine.close()


def test_explicit_count_cannot_override_a_different_advertised_count():
    raw = ImmediateDeviceRunner(0, 0)
    runner = BenchmarkRunner(raw)
    engine = Engine(runner, context_length=128)

    def trace(case, step):
        raise AssertionError("conflicting trace protocol must fail before work")

    setattr(trace, "decode_passes", 5)
    try:
        with pytest.raises(ValueError, match="disagrees with the trace callback"):
            run_sweep(
                engine,
                [BenchCase(phase="decode", context_length=4, batch_size=1)],
                runner=runner,
                prompt_tokens=[1],
                trace_step=trace,
                trace_decode_passes=3,
            )
        assert not engine.has_work() and raw.enqueued == 0
    finally:
        engine.close()
