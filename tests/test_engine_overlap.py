from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace

import pytest
from output_helpers import forward_output

from resin import CacheGroupLayout, CacheLayout, Engine, RunnerConfig, SamplingParams


@dataclass
class Feedback:
    tokens: list[int]
    positions: list[int]


class Completion:
    def __init__(self, output, feedback, *, prior_harvest=None, submitted=None):
        self.feedback = feedback
        self.output = output
        self.prior_harvest = prior_harvest
        self.submitted = submitted
        self.future: Future | None = None
        self.error: BaseException | None = None

    def result(self):
        if self.error is not None:
            raise self.error
        if self.submitted is not None:
            assert self.submitted.wait(2), (
                "next GPU submission must precede previous CPU harvest"
            )
        if self.prior_harvest is not None:
            self.prior_harvest.set()
        return self.future.result(timeout=2) if self.future else self.output


class AsyncRunner:
    runner_config = RunnerConfig(
        cache_layout=CacheLayout(16, [CacheGroupLayout(128)]),
        cache_write_lookahead_tokens=2,
    )

    def __init__(self):
        self.plans = []
        self.released = []
        self.pending = []
        self.completed_before_release = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.next_submitted = Event()
        self.prior_harvest = Event()
        self.require_overlap = False
        self.fail = False
        self.reject_enqueue = False

    def bind_resources(self, binding):
        self.binding = binding

    def forward(self, plan):
        self.plans.append(plan)
        return forward_output(
            [[int(plan.token_ids[int(end) - 1]) + 1] for end in plan.token_indptr[1:]]
        )

    def enqueue(self, plan, *, feedback=None):
        if self.reject_enqueue:
            self.reject_enqueue = False
            raise RuntimeError("enqueue rejected before launch")
        self.plans.append(plan)
        tokens = plan.token_ids.tolist() if feedback is None else feedback.tokens
        positions = plan.positions.tolist() if feedback is None else feedback.positions
        rows = [[int(token) + 1, int(token) + 2] for token in tokens]
        completion = Completion(
            forward_output(rows),
            Feedback(
                [row[-1] for row in rows], [position + 2 for position in positions]
            ),
        )
        if self.require_overlap and not self.pending:
            completion.submitted = self.next_submitted
            completion.prior_harvest = self.prior_harvest
        if self.require_overlap and feedback is not None:

            def device_work():
                self.next_submitted.set()
                assert self.prior_harvest.wait(2), (
                    "GPU work and previous CPU harvest must overlap"
                )
                return forward_output(rows)

            completion.future = self.executor.submit(device_work)
        if self.fail:
            completion.error = RuntimeError("asynchronous device failure")
        self.pending.append(completion)
        return completion

    def release(self, ids):
        self.completed_before_release.append(
            all(value.future is None or value.future.done() for value in self.pending)
        )
        self.released.extend(ids)

    def shutdown(self):
        self.executor.shutdown(wait=True)


def test_next_device_submission_precedes_previous_cpu_harvest():
    runner = AsyncRunner()
    runner.require_overlap = True
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=24))
    assert [event.token_id for event in engine.step()] == [2]
    assert engine.step() == []
    assert engine.overlap_pending
    assert engine.active_request_count == 1 and engine.ready_request_count == 0
    assert [event.token_id for event in engine.step()] == [3, 4]
    assert runner.next_submitted.is_set() and runner.prior_harvest.is_set()
    assert engine.overlap_stats == (2, 1)
    engine.close()
    assert request.cancelled
    assert request.take_result() == [2, 3, 4, 5, 6]
    assert all(runner.completed_before_release)
    runner.shutdown()


@pytest.mark.parametrize("overlap", [True, False])
def test_overlap_switch_preserves_single_step_for_synchronous_runners(overlap):
    class Synchronous:
        def forward(self, plan):
            return forward_output([int(token) + 1 for token in plan.token_ids])

    engine = Engine(Synchronous(), enable_overlap=overlap)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=8))
    assert [event.token_id for event in engine.step()] == [2]
    assert not engine.overlap_pending
    while engine.has_work():
        assert len(engine.step()) == 1
    assert request.take_result() == list(range(2, 10))


def test_cancel_drains_accepted_work_and_preserves_other_request_events():
    runner = AsyncRunner()
    engine = Engine(runner)
    first = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=24))
    second = engine.submit([10], sampling_params=SamplingParams(max_output_tokens=24))
    engine.step()
    assert engine.step() == []
    assert engine.cancel(first)
    assert first.cancelled and first.take_result() == [2, 3, 4]
    assert not engine.overlap_pending
    events = engine.step()
    assert [(event.request_id, event.token_id) for event in events[:2]] == [
        (second.id, 12),
        (second.id, 13),
    ]
    engine.close()
    assert all(runner.completed_before_release)
    runner.shutdown()


def test_async_failure_quarantines_the_original_writer_lease():
    runner = AsyncRunner()
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=24))
    engine.step()
    runner.fail = True
    engine.step()
    with pytest.raises(RuntimeError, match="asynchronous device failure"):
        engine.step(allow_overlap=False)
    assert engine.poisoned and not request.done
    assert runner.released == []
    assert engine.cache_occupied_block_count == 1
    with pytest.raises(RuntimeError, match="quarantined"):
        engine.close()
    runner.shutdown()


@pytest.mark.parametrize("invalid", [[3, 4], [[3, 4]]])
@pytest.mark.parametrize("continuation", [False, True])
def test_completion_sequence_result_quarantines_accepted_work(
    monkeypatch, invalid, continuation
):
    runner = AsyncRunner()
    original = runner.enqueue

    def enqueue(plan, *, feedback=None):
        completion = original(plan, feedback=feedback)
        if len(runner.pending) == (2 if continuation else 1):
            completion.output = invalid
        return completion

    monkeypatch.setattr(runner, "enqueue", enqueue)
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=24))
    assert [event.token_id for event in engine.step()] == [2]
    assert engine.step() == []
    if continuation:
        assert [event.token_id for event in engine.step()] == [3, 4]
    with pytest.raises(TypeError, match="runner must return ForwardOutput"):
        engine.step(allow_overlap=False)
    assert engine.poisoned and not request.done
    assert engine.cache_occupied_block_count == 1
    assert runner.released == []
    with pytest.raises(RuntimeError, match="quarantined"):
        engine.close()
    runner.shutdown()


@pytest.mark.parametrize("kind", ["flat", "nested", "noncallable"])
@pytest.mark.parametrize("continuation", [False, True])
def test_invalid_immediate_enqueue_result_quarantines_accepted_work(
    monkeypatch, kind, continuation
):
    runner = AsyncRunner()
    original = runner.enqueue
    calls = 0

    def enqueue(plan, *, feedback=None):
        nonlocal calls
        calls += 1
        if calls == (2 if continuation else 1):
            if kind == "noncallable":
                return SimpleNamespace(result=[])
            return [3, 4] if kind == "flat" else [[3, 4]]
        return original(plan, feedback=feedback)

    monkeypatch.setattr(runner, "enqueue", enqueue)
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=24))
    assert [event.token_id for event in engine.step()] == [2]
    if continuation:
        assert engine.step() == []
    with pytest.raises(
        TypeError, match=r"runner.enqueue\(\) must return ForwardOutput"
    ):
        engine.step()
    assert engine.poisoned and not request.done
    assert engine.cache_occupied_block_count == 1
    assert runner.released == []
    with pytest.raises(RuntimeError, match="quarantined"):
        engine.close()
    runner.shutdown()


def test_enqueue_accepts_immediately_completed_native_output(monkeypatch):
    runner = AsyncRunner()
    monkeypatch.setattr(runner, "enqueue", lambda plan: forward_output([[3, 4]]))
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=24))
    assert [event.token_id for event in engine.step()] == [2]
    assert [event.token_id for event in engine.step()] == [3, 4]
    assert not engine.overlap_pending and not engine.poisoned
    engine.close()
    assert request.cancelled and request.take_result() == [2, 3, 4]
    runner.shutdown()


def test_short_budgets_use_synchronous_reference():
    runner = AsyncRunner()
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=4))
    engine.step()
    assert engine.step()
    assert not runner.pending and not engine.overlap_pending
    engine.close()
    assert request.done
    runner.shutdown()


@pytest.mark.parametrize("stop", [3, 4])
def test_stop_result_precedes_successor_completion_but_release_waits(monkeypatch, stop):
    runner = AsyncRunner()
    original = runner.enqueue

    def enqueue(plan, *, feedback=None):
        completion = original(plan, feedback=feedback)
        if feedback is not None:
            completion.future = Future()
        return completion

    monkeypatch.setattr(runner, "enqueue", enqueue)
    engine = Engine(runner)
    request = engine.submit(
        [1], sampling_params=SamplingParams(max_output_tokens=24, stop_token_ids=[stop])
    )
    assert [event.token_id for event in engine.step()] == [2]
    assert engine.step() == []
    events = engine.step()
    assert [event.token_id for event in events] == list(range(3, stop + 1))
    assert events[-1].finished and events[-1].finish_reason == "stop"
    assert request.done and not request.cancelled
    assert request.take_result() == list(range(2, stop + 1))
    assert engine.overlap_stats == (2, 1)
    assert engine.overlap_pending and not engine.overlap_ready
    assert engine.active_request_count == 0 and engine.has_work()
    assert engine.cache_free_parent_count == 15
    assert runner.released == []
    successor = runner.pending[-1]
    assert not successor.future.done()

    successor.future.set_result(successor.output)
    assert engine.step() == []
    assert not engine.overlap_pending and not engine.has_work()
    assert engine.cache_free_parent_count == 16
    assert runner.released == [request.id]
    assert all(runner.completed_before_release)
    assert not engine.cancel(request)
    with pytest.raises(RuntimeError, match="already been consumed"):
        request.take_result()
    engine.close()
    assert runner.released == [request.id]
    runner.shutdown()


def test_cancel_survivor_drains_terminal_peer_without_losing_other_events():
    runner = AsyncRunner()
    engine = Engine(runner)
    stopped = engine.submit(
        [1], sampling_params=SamplingParams(max_output_tokens=24, stop_token_ids=[3])
    )
    canceled = engine.submit(
        [10], sampling_params=SamplingParams(max_output_tokens=24, stop_token_ids=[999])
    )
    survivor = engine.submit(
        [20], sampling_params=SamplingParams(max_output_tokens=24, stop_token_ids=[999])
    )
    engine.step()
    assert engine.step() == []
    events = engine.step()
    assert [event.token_id for event in events if event.request_id == stopped.id] == [3]
    assert stopped.take_result() == [2, 3]
    assert engine.overlap_pending and not engine.overlap_ready
    assert runner.released == []

    assert engine.cancel(canceled)
    assert canceled.cancelled and canceled.take_result() == [11, 12, 13, 14, 15]
    assert runner.released == [stopped.id, canceled.id]
    events = engine.step()
    assert [(event.request_id, event.token_id) for event in events] == [
        (survivor.id, 24),
        (survivor.id, 25),
    ]
    engine.close()
    assert survivor.take_result() == [21, 22, 23, 24, 25, 26, 27]
    assert all(runner.completed_before_release)
    assert engine.cache_free_parent_count == 16
    runner.shutdown()


def test_gpu_feedback_matches_reference_across_page_and_budget_drains():
    outputs = []
    for overlap in (False, True):
        runner = AsyncRunner()
        engine = Engine(runner, enable_overlap=overlap, context_length=1024)
        request = engine.submit(
            [1] * 123, sampling_params=SamplingParams(max_output_tokens=270)
        )
        events = []
        while engine.has_work():
            events.extend(engine.step())
        outputs.append([event.token_id for event in events])
        assert request.take_result() == outputs[-1]
        assert engine.cache_free_parent_count == 16
        assert (
            engine.overlap_stats[1] > 0 if overlap else engine.overlap_stats == (0, 0)
        )
        engine.close()
        runner.shutdown()
    assert outputs[0] == outputs[1] == list(range(2, 272))


@pytest.mark.parametrize("second_enqueue", [False, True])
def test_completion_inspection_failure_quarantines_accepted_work(
    monkeypatch, second_enqueue
):
    import gc
    import weakref

    class InvalidCompletion:
        @property
        def result(self):
            raise RuntimeError("completion descriptor failed")

    runner = AsyncRunner()
    original = runner.enqueue
    references = []
    calls = 0

    def enqueue(plan, *, feedback=None):
        nonlocal calls
        calls += 1
        if calls == (2 if second_enqueue else 1):
            result = InvalidCompletion()
            references.append(weakref.ref(result))
            return result
        return original(plan, feedback=feedback)

    monkeypatch.setattr(runner, "enqueue", enqueue)
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=24))
    engine.step()
    if second_enqueue:
        assert engine.step() == []
    with pytest.raises(RuntimeError, match="completion descriptor failed"):
        engine.step()
    gc.collect()
    assert references[0]() is not None  # Its GPU buffer owner stays quarantined.
    assert engine.poisoned and not request.done
    assert engine.cache_occupied_block_count == 1
    assert runner.released == []
    runner.shutdown()


def test_feedback_descriptor_failure_cannot_leave_a_healthy_pending_engine(monkeypatch):
    class InvalidFeedback:
        @property
        def feedback(self):
            raise RuntimeError("feedback descriptor failed")

        def result(self):
            return forward_output([[3, 4]])

    runner = AsyncRunner()
    monkeypatch.setattr(runner, "enqueue", lambda plan: InvalidFeedback())
    engine = Engine(runner)
    request = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=24))
    engine.step()
    assert engine.step() == []
    with pytest.raises(RuntimeError, match="feedback descriptor failed"):
        engine.step()
    assert engine.poisoned and not request.done
    assert engine.cache_occupied_block_count == 1
    assert runner.released == []
    runner.shutdown()


def test_retryable_rejection_keeps_unrelated_already_committed_stream_events():
    runner = AsyncRunner()
    engine = Engine(runner)
    first = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=24))
    second = engine.submit([10], sampling_params=SamplingParams(max_output_tokens=24))
    engine.step()
    assert engine.step() == []
    assert engine.cancel(first)
    runner.reject_enqueue = True
    with pytest.raises(RuntimeError, match="enqueue rejected before launch"):
        engine.step()
    assert not engine.poisoned
    events = engine.step()
    assert [(event.request_id, event.token_id) for event in events] == [
        (second.id, 12),
        (second.id, 13),
    ]
    engine.close()
    runner.shutdown()


def test_arriving_work_drains_the_pending_cohort_before_scheduler_admission():
    runner = AsyncRunner()
    engine = Engine(runner, max_running_requests=2)
    first = engine.submit([1], sampling_params=SamplingParams(max_output_tokens=40))
    engine.step()
    assert engine.step() == []
    enqueued = len(runner.pending)
    added = engine.submit(
        [20], sampling_params=SamplingParams(max_output_tokens=1), priority=100
    )
    assert not engine.overlap_ready
    assert [event.token_id for event in engine.step()] == [3, 4]
    assert not engine.overlap_pending
    assert len(runner.pending) == enqueued
    for _ in range(4):
        if added.done:
            break
        engine.step()
    assert added.take_result() == [21]
    engine.cancel(first)
    engine.close()
    runner.shutdown()
