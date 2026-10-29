from __future__ import annotations

import pytest
import torch
from output_helpers import forward_output

import resin
from resin import BatchPlan, Engine, SamplingParams
from resin.backend.cuda.sampling import FusedSampler
from resin.backend.sampling import TorchSampler, batch_sampling_inputs
from resin.backend.speculative import SpeculativeStats, accept_speculative_tokens


class RecordingRunner:
    def __init__(self) -> None:
        self.plans = []

    def forward(self, plan):
        self.plans.append(plan)
        return forward_output([0] * plan.num_sequences)


class TestSamplingContract:
    def test_speculative_acceptance_returns_prefix_plus_one_bonus(self) -> None:
        acceptance = accept_speculative_tokens(
            [[10, 11, 12], [20, 21, 22]],
            [[10, 11, 13, 14], [31, 30, 29, 28]],
        )

        assert acceptance.token_rows == [[10, 11, 13], [31]]
        assert acceptance.accepted_counts == [2, 0]
        assert acceptance.processed_counts == [3, 1]

    def test_speculative_stats_own_shared_runner_bookkeeping(self) -> None:
        stats = SpeculativeStats(3)
        acceptance = stats.accept([[10, 11, 12]], [[10, 21, 22, 23]])

        assert acceptance.token_rows == [[10, 21]]
        assert stats.snapshot(include_draft_tokens=True) == {
            "draft_tokens": 3,
            "blocks": 1,
            "drafts": 3,
            "accepted": 1,
            "emitted": 2,
            "acceptance_rate": 1 / 3,
        }

    def test_sampling_params_validate_defaults_and_preserve_extra_identity(
        self,
    ) -> None:
        marker = object()
        params = SamplingParams(extra=marker)

        assert params.max_output_tokens is None
        assert params.stop_token_ids == ()
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k is None
        assert params.seed is None
        assert params.extra is marker

        for kwargs in (
            {"temperature": -1.0},
            {"temperature": float("inf")},
            {"top_p": 0.0},
            {"top_p": 1.1},
            {"top_k": 0},
            {"seed": -1},
        ):
            with pytest.raises(ValueError):
                # Parametrized invalid values exercise native validation.
                SamplingParams(**kwargs)  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_output_tokens": -1},
            {"max_output_tokens": 2**32},
            {"stop_token_ids": [-1]},
            {"stop_token_ids": [2**32]},
        ],
    )
    def test_generation_policy_rejects_values_outside_native_token_range(
        self, kwargs
    ) -> None:
        with pytest.raises((OverflowError, ValueError)):
            SamplingParams(**kwargs)

    def test_sampling_params_rejects_retired_generation_limit_name(self) -> None:
        with pytest.raises(TypeError, match="max_new_tokens"):
            SamplingParams(max_new_tokens=1)  # ty: ignore[unknown-argument]

    def test_stop_tokens_are_snapshotted_before_submission(self) -> None:
        stops = [0]
        params = SamplingParams(max_output_tokens=3, stop_token_ids=stops)
        stops[0] = 1
        engine = Engine(RecordingRunner())
        request = engine.submit([7], sampling_params=params)

        events = engine.step()

        assert request.take_result() == [0]
        assert events[-1].finished and events[-1].finish_reason == "stop"

    def test_engine_freezes_mixed_row_annotations_and_advances_offsets(self) -> None:
        first_extra = {"processor": "first"}
        second_extra = object()
        runner = RecordingRunner()
        engine = Engine(runner, max_running_requests=2)
        first = engine.submit(
            [1],
            sampling_params=SamplingParams(
                max_output_tokens=2,
                temperature=0.7,
                top_p=0.9,
                top_k=32,
                seed=11,
                extra=first_extra,
            ),
        )
        second = engine.submit(
            [2],
            sampling_params=SamplingParams(
                max_output_tokens=2,
                temperature=0.0,
                seed=22,
                extra=second_extra,
            ),
        )

        engine.step()
        initial = runner.plans[0]
        assert [params.temperature for params in initial.sampling_params] == [0.7, 0.0]
        assert [params.top_p for params in initial.sampling_params] == [0.9, 1.0]
        assert [params.top_k for params in initial.sampling_params] == [32, None]
        assert initial.sampling_params[0].extra is first_extra
        assert initial.sampling_params[1].extra is second_extra
        assert [params.seed for params in initial.sampling_params] == [11, 22]
        assert batch_sampling_inputs(initial, 128)["offsets"] == (32, 32)

        engine.step()
        decode = runner.plans[1]
        assert batch_sampling_inputs(decode, 128)["offsets"] == (64, 64)
        assert first.take_result() == [0, 0]
        assert second.take_result() == [0, 0]

    def test_verified_block_advances_sampling_offset_per_committed_token(self) -> None:
        class BlockRunner:
            runner_config = resin.RunnerConfig(max_accepted_per_step=3)

            def __init__(self) -> None:
                self.plans = []

            def forward(self, plan):
                self.plans.append(plan)
                if plan.phase == resin.BatchPhase.PREFILL:
                    return forward_output([1])
                if len(self.plans) == 2:
                    return forward_output([[2, 3, 4]])
                return forward_output([[5]])

        runner = BlockRunner()
        engine = Engine(runner)
        request = engine.submit(
            [0], sampling_params=SamplingParams(max_output_tokens=5, seed=7)
        )

        while engine.has_work():
            engine.step()

        assert [
            batch_sampling_inputs(plan, 128)["offsets"] for plan in runner.plans
        ] == [
            (32,),
            (64,),
            (160,),
        ]
        assert request.take_result() == [1, 2, 3, 4, 5]

    def test_batch_plan_requires_sequence_aligned_sampling_annotations(self) -> None:
        params = SamplingParams()
        with pytest.raises(ValueError, match="one item per sequence"):
            BatchPlan(
                sequence_ids=[7, 8],
                context_lengths=[0, 0],
                token_ids=[1, 2],
                token_indptr=[0, 1, 2],
                sampling_params=[params],
            )


class TestSharedSampler:
    @staticmethod
    def _plan(params, *, sampling_mask=None):
        rows = len(params)
        return BatchPlan(
            sequence_ids=list(range(10, 10 + rows)),
            context_lengths=[0] * rows,
            token_ids=list(range(1, rows + 1)),
            token_indptr=list(range(rows + 1)),
            sampling_params=params,
            sampling_mask=sampling_mask,
        )

    def test_torch_sampler_unifies_greedy_top_k_and_unsampled_rows(self) -> None:
        sampler = TorchSampler()
        logits = torch.tensor([[0.0, 2.0, 1.0], [0.0, 2.0, 1.0], [0.0, 2.0, 1.0]])
        plan = self._plan(
            [
                SamplingParams(temperature=0.0),
                SamplingParams(temperature=1.0, top_k=1),
                SamplingParams(temperature=1.0),
            ],
            sampling_mask=[True, True, False],
        )

        assert sampler(logits, plan).tolist() == [1, 1, 1]

    @pytest.mark.parametrize(
        ("top_p", "top_k", "expected"),
        [
            (0.7, 2, [0.60 / 0.85, 0.25 / 0.85]),
            (0.7, 1, [1.0]),
            (0.5, 2, [1.0, 0.0]),
            (1.0, 2, [0.60 / 0.85, 0.25 / 0.85]),
        ],
    )
    def test_torch_sampler_applies_top_p_before_top_k(
        self, monkeypatch: pytest.MonkeyPatch, top_p, top_k, expected
    ) -> None:
        # At p=.7 the original .60/.25/.15 distribution needs two tokens.
        # Renormalizing its top two first would incorrectly keep only .60.
        logits = torch.tensor([[0.60, 0.25, 0.15]]).log()
        plan = self._plan(
            [SamplingParams(temperature=1.0, top_p=top_p, top_k=top_k, seed=7)]
        )
        observed = []

        def sample(probabilities, count, *, generator):
            observed.append(probabilities.clone())
            return probabilities.argmax().reshape(count)

        monkeypatch.setattr(torch, "multinomial", sample)
        assert TorchSampler()(logits, plan).tolist() == [0]
        assert len(observed) == 1
        torch.testing.assert_close(observed[0], torch.tensor(expected))

    def test_fused_sampler_receives_per_row_filter_and_rng_tensors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        received = []
        sampler = FusedSampler.__new__(FusedSampler)

        def sample_tensors(logits, tensors, *, out=None):
            received.append(tensors)
            return logits.argmax(-1)

        monkeypatch.setattr(sampler, "sample_tensors", sample_tensors)
        plan = BatchPlan(
            sequence_ids=[10, 11],
            context_lengths=[1, 2],
            token_ids=[1, 2],
            token_indptr=[0, 1, 2],
            sampling_params=[
                SamplingParams(temperature=0.5, top_p=0.8, top_k=2, seed=31),
                SamplingParams(temperature=0.0, seed=32),
            ],
        )
        logits = torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 1.0]])
        output = torch.empty(2, dtype=torch.int64)
        assert sampler(logits, plan, out=output) is output
        assert output.tolist() == [2, 1]
        tensors = received[0]
        torch.testing.assert_close(tensors.temperatures, torch.tensor([0.5, 1.0]))
        assert tensors.top_ks.tolist() == [2, 3]
        torch.testing.assert_close(tensors.top_ps, torch.tensor([0.8, 1.0]))
        assert tensors.seeds.tolist() == [31, 32]
        assert tensors.offsets.tolist() == [64, 96]
        assert tensors.stochastic_mask.tolist() == [True, False]
        shifted = sampler.host_inputs(plan, 3, offset_delta=64)
        assert shifted["sampling_offsets"] == (128, 160)

    def test_fused_sampler_prepares_one_policy_batch_for_verified_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sampler = FusedSampler.__new__(FusedSampler)
        received = []

        def sample_block(logits, tensors):
            received.append((logits, tensors))
            return logits.argmax(-1)

        monkeypatch.setattr(sampler, "sample_block", sample_block)
        plan = BatchPlan(
            sequence_ids=[10, 11],
            context_lengths=[1, 2],
            token_ids=[1, 2],
            token_indptr=[0, 1, 2],
            sampling_params=[SamplingParams(seed=31), SamplingParams(seed=32)],
        )
        logits = torch.randn(2, 4, 14)[:, :, 1::2]
        result = sampler(logits, plan, offset_delta=32)
        torch.testing.assert_close(result, logits.argmax(-1))
        assert len(received) == 1
        assert received[0][0] is logits
        assert received[0][1].offsets.tolist() == [96, 128]
        assert received[0][1].seeds.tolist() == [31, 32]
