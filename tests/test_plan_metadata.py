from __future__ import annotations

import numpy as np
import pytest
from output_helpers import forward_output

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheLayout,
    Engine,
    RunnerConfig,
    SamplingParams,
    SchedulerConfig,
)
from resin.backend.distributed import _runner_signature


def test_prefill_lengths_are_owned_readonly_views_with_legacy_endpoint_defaults():
    lengths = np.array([20, 30], dtype=np.uint32)
    plan = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[4, 8],
        token_ids=[10, 11, 12],
        token_indptr=[0, 2, 3],
        prefill_lengths=lengths,
    )
    lengths[:] = 99
    view = plan.prefill_lengths
    assert view.dtype == np.uint32 and not view.flags.writeable
    assert view.tolist() == [20, 30]
    assert np.shares_memory(view, plan.prefill_lengths)
    with pytest.raises(ValueError, match="read-only"):
        view[0] = 3
    del plan
    assert view.tolist() == [20, 30]

    legacy = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[4, 8],
        token_ids=[10, 11, 12],
        token_indptr=[0, 2, 3],
    )
    assert legacy.prefill_lengths.tolist() == [6, 9]
    with pytest.raises(ValueError, match="prefill_lengths.*one item per sequence"):
        BatchPlan(
            sequence_ids=[1, 2],
            context_lengths=[4, 8],
            token_ids=[10, 11, 12],
            token_indptr=[0, 2, 3],
            prefill_lengths=[20],
        )
    with pytest.raises(ValueError, match="prefill_lengths must cover"):
        BatchPlan(
            sequence_ids=[1, 2],
            context_lengths=[4, 8],
            token_ids=[10, 11, 12],
            token_indptr=[0, 2, 3],
            prefill_lengths=[5, 9],
        )
    edge = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[0, 2**32 - 2],
        token_ids=[10, 11, 12],
        token_indptr=[0, 2, 3],
        prefill_lengths=[20, 2**32 - 1],
    )
    assert edge.prefill_lengths.tolist() == [20, 2**32 - 1]


def test_native_plans_keep_full_prefill_lengths_through_chunks():
    class Runner:
        runner_config = RunnerConfig(prefill_chunk_size=2, continuation_prefill=True)

        def __init__(self):
            self.plans = []

        def forward(self, plan):
            self.plans.append(plan)
            return forward_output([9] * plan.num_sequences)

    runner = Runner()
    engine = Engine(runner, context_length=64)
    first = engine.submit(
        [1, 2, 3, 4, 5, 6, 7], sampling_params=SamplingParams(max_output_tokens=3)
    )
    second = engine.submit(
        [10, 11, 12], sampling_params=SamplingParams(max_output_tokens=2)
    )
    expected = {first.id: 7, second.id: 3}
    while engine.has_work():
        engine.step()
    assert runner.plans[0].prefill_lengths.tolist() == [7, 3]
    assert runner.plans[0].token_indptr.tolist() == [0, 2, 4]
    assert any(plan.phase == BatchPhase.DECODE for plan in runner.plans)
    for plan in runner.plans:
        if plan.phase == BatchPhase.PREFILL:
            assert plan.prefill_lengths.tolist() == [
                expected[int(i)] for i in plan.sequence_ids
            ]
    assert first.done and second.done


@pytest.mark.parametrize(
    "replay,expected_hit", [(0, 384), (128, 384), (140, 256), (512, 0)]
)
def test_prefix_replay_policy_caps_hits_and_preserves_shared_producer_pages(
    replay, expected_hit
):
    layout = CacheLayout(
        parent_capacity=20, groups=[CacheGroupLayout(128)], replay_suffix_tokens=replay
    )

    class Runner:
        runner_config = RunnerConfig(cache_layout=layout, continuation_prefill=True)

        def __init__(self):
            self.plans = []

        def bind_resources(self, binding):
            assert binding.cache_layout.replay_suffix_tokens == replay

        def forward(self, plan):
            self.plans.append(plan)
            return forward_output([9] * plan.num_sequences)

    runner = Runner()
    with Engine(
        runner,
        context_length=1024,
        prefix_cache=True,
        scheduler_config=SchedulerConfig(prefill_chunk_size=128),
    ) as engine:
        prompt = list(range(512))
        seed = engine.submit(
            prompt, sampling_params=SamplingParams(max_output_tokens=1)
        )
        while engine.has_work():
            engine.step()
        assert seed.done
        producer_pages = runner.plans[-1].cache_groups[0].block_slab_ids.copy()
        stats = engine.prefix_cache_stats
        assert stats is not None
        cached_entries = stats[0]
        second = engine.submit(
            prompt, sampling_params=SamplingParams(max_output_tokens=1)
        )
        assert second.cached_tokens == expected_hit
        engine.step()
        restored = runner.plans[-1]
        assert restored.context_lengths.tolist() == [expected_hit]
        assert restored.prefill_lengths.tolist() == [512]
        assert restored.cache_groups[0].block_slab_ids[
            : expected_hit // 128
        ].tolist() == (producer_pages[: expected_hit // 128].tolist())
        stats = engine.prefix_cache_stats
        assert stats is not None and stats[0] >= cached_entries
        while engine.has_work():
            engine.step()
        scored = engine.submit(
            prompt,
            sampling_params=SamplingParams(max_output_tokens=1, prompt_logprobs=True),
        )
        assert scored.cached_tokens == 0
        engine.cancel(scored)


def test_prefix_replay_policy_is_frozen_and_part_of_replica_configuration():
    groups = [CacheGroupLayout(128)]
    plain = CacheLayout(parent_capacity=20, groups=groups)
    replay = CacheLayout(parent_capacity=20, groups=groups, replay_suffix_tokens=140)
    assert plain.replay_suffix_tokens == 0
    assert replay.replay_suffix_tokens == 140
    assert "replay_suffix_tokens=140" in repr(replay)
    assert _runner_signature(RunnerConfig(cache_layout=plain)) != _runner_signature(
        RunnerConfig(cache_layout=replay)
    )
    with pytest.raises(AttributeError):
        setattr(replay, "replay_suffix_tokens", 1)
