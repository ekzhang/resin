from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from test_flat_cache import RecordingCachedRunner

from resin import CacheGroupLayout, CacheLayout, Engine, SamplingParams


class Operation(Future):
    def poll(self, wait=False):
        self.waited = wait
        return self.result(timeout=2) if wait or self.done() else None


@pytest.mark.parametrize("delay", ["lookup", "load", "store"])
def test_async_storage_progress_and_retirement(delay):
    pending = []

    def operation(kind, action, buffers=()):
        op = Operation()
        if kind == delay:
            pending.append((op, action, buffers))
        else:
            op.set_result(action())
        return op

    def lookup(tokens, maximum, alignment):
        length = 2 if tokens[0] == 1 and maximum >= 2 else 0
        return operation("lookup" if length else "miss", lambda: length)

    def load(tokens, groups, buffers):
        def copy():
            for regions in buffers:
                regions[0][:] = b"*"
            return delay != "load"

        return operation("load", copy, buffers)

    def store(tokens, groups, buffers):
        before = tuple(bytes(r[0]) for r in buffers)
        return operation(
            "store", lambda: before == tuple(bytes(r[0]) for r in buffers), buffers
        )

    layout = CacheLayout(parent_capacity=8, groups=[CacheGroupLayout(2)])
    runner = RecordingCachedRunner(layout)
    arena = np.full(9, 73, np.uint8)

    def buffers(groups):
        view = memoryview(arena)
        return tuple((view[slab : slab + 1],) for slab, _ in groups[0][1])

    setattr(runner, "cache_storage_buffers", buffers)
    storage = SimpleNamespace(lookup=lookup, load=load, store=store)
    engine = Engine(runner, prefix_cache=True, prefix_cache_storage=storage)
    params = SamplingParams(max_output_tokens=10)
    running = engine.submit([9, 9], sampling_params=params)
    engine.step()
    waiting = engine.submit([1, 2, 3], sampling_params=params)
    for _ in range(3):
        engine.step()
    assert any(running.id in plan.sequence_ids for plan in runner.plans[1:])
    op, _, buffers = pending[0]
    before = tuple(bytes(r[0]) for r in buffers)
    assert engine.cancel(running)
    if delay != "lookup":
        assert engine.cancel(waiting)
    engine.clear_prefix_cache()
    engine.submit([7] * 8, sampling_params=SamplingParams(max_output_tokens=1))
    engine.step()
    arena[runner.plans[-1].cache_groups[0].append_slab_ids] = 7
    assert before == tuple(bytes(r[0]) for r in buffers)
    with ThreadPoolExecutor(2) as workers:
        closer = workers.submit(engine.close)
        with pytest.raises(TimeoutError):
            closer.result(0.02)
        list(workers.map(lambda item: item[0].set_result(item[1]()), pending))
        closer.result(1)
    assert op.waited and op.result() == (2 if delay == "lookup" else delay != "load")
    if delay == "load":
        assert all(bytes(r[0]) == b"*" for r in buffers) and waiting.cached_tokens == 0
