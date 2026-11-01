"""One Engine admission limit governs direct and HTTP alternative requests."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch
from test_http_lifecycle import Recipe, running_server
from test_prompt_logprobs import TransitionRunner
from test_recipe import _byte_tokenizer

from resin import Engine, SamplingParams
from resin.backend.plan_transport import PlanTransport
from resin.recipe import Tokenizer


def _params(field, count, *, max_output_tokens=None):
    if field == "top_logprobs":
        return SamplingParams(max_output_tokens=max_output_tokens, top_logprobs=count)
    return SamplingParams(
        max_output_tokens=max_output_tokens, prompt_top_logprobs=count
    )


def _scored_runner():
    runner = TransitionRunner()
    runner.table = torch.zeros((4, 4))
    return runner


def test_default_logprob_limit_is_read_only():
    engine = Engine(_scored_runner())
    assert engine.max_top_logprobs == 32
    with pytest.raises(AttributeError):
        engine.max_top_logprobs = 100  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("field", ["top_logprobs", "prompt_top_logprobs"])
def test_default_limit_rejects_above_thirty_two_before_admission(field):
    runner = _scored_runner()
    engine = Engine(runner)
    with pytest.raises(ValueError, match=field):
        engine.submit([1, 2], sampling_params=_params(field, 33))
    assert engine.active_request_count == 0
    assert engine.ready_request_count == 0
    assert not engine.has_work() and not engine.poisoned
    assert runner.calls == []


@pytest.mark.parametrize("field", ["top_logprobs", "prompt_top_logprobs"])
def test_limit_rejection_preserves_admission_and_request_id(field):
    runner = _scored_runner()
    engine = Engine(runner, max_top_logprobs=3)
    accepted = _params(field, 3, max_output_tokens=1)
    first = engine.submit([1, 2], sampling_params=accepted)
    before = engine.active_request_count, engine.ready_request_count
    with pytest.raises(ValueError, match=field):
        engine.submit([1, 2], sampling_params=_params(field, 4))
    assert (engine.active_request_count, engine.ready_request_count) == before
    assert not engine.poisoned and runner.calls == []
    second = engine.submit([1, 2], sampling_params=accepted)
    assert second.id == first.id + 1
    while engine.has_work():
        engine.step()
    assert first.done and second.done


def test_sampling_parameters_are_reusable_across_engine_limits():
    params = SamplingParams(
        max_output_tokens=1, top_logprobs=21, prompt_top_logprobs=21
    )
    smaller = Engine(_scored_runner(), max_top_logprobs=20)
    with pytest.raises(ValueError, match="top_logprobs"):
        smaller.submit([1, 2], sampling_params=params)
    runner = _scored_runner()
    larger = Engine(runner, max_top_logprobs=32)
    request = larger.submit([1, 2], sampling_params=params)
    events = larger.step()
    assert request.done and len(events) == 1
    policy = runner.calls[0].sampling_params[0]
    assert policy.top_logprobs == params.top_logprobs == 21
    assert policy.prompt_top_logprobs == params.prompt_top_logprobs == 21
    assert request.prompt_logprobs is not None


def test_zero_limit_allows_selected_generated_and_prompt_scores():
    engine = Engine(_scored_runner(), max_top_logprobs=0)
    request = engine.submit(
        [1, 2],
        sampling_params=SamplingParams(
            max_output_tokens=1, logprobs=True, prompt_logprobs=True
        ),
    )
    event = engine.step()[0]
    assert event.logprobs is not None and event.logprobs.top_logprobs == []
    assert event.logprobs.logprob == pytest.approx(-np.log(4))
    assert request.prompt_logprobs is not None
    prompt_score = request.prompt_logprobs[1]
    assert prompt_score is not None and prompt_score.top_logprobs == []
    for field in ("top_logprobs", "prompt_top_logprobs"):
        with pytest.raises(ValueError, match=field):
            engine.submit([1, 2], sampling_params=_params(field, 1))


@pytest.mark.parametrize("count", [1500, 2**32 - 1])
def test_configured_counts_survive_live_plan_and_transport(count):
    runner = _scored_runner()
    engine = Engine(runner, max_top_logprobs=count)
    assert engine.max_top_logprobs == count
    request = engine.submit(
        [1, 2],
        sampling_params=SamplingParams(
            max_output_tokens=1, top_logprobs=count, prompt_top_logprobs=count
        ),
    )
    event = engine.step()[0]
    assert request.done and event.logprobs is not None
    assert len(event.logprobs.top_logprobs) == 4
    plan = runner.calls[0]
    restored = PlanTransport().restore(PlanTransport().prepare(plan))
    for candidate in (plan, restored):
        assert candidate.sampling_params[0].top_logprobs == count
        assert candidate.sampling_params[0].prompt_top_logprobs == count
        np.testing.assert_array_equal(candidate.prompt_score_top_counts, [count])


@pytest.mark.parametrize("limit", [-1, 2**32])
def test_engine_limit_requires_uint32(limit):
    with pytest.raises(OverflowError):
        Engine(_scored_runner(), max_top_logprobs=limit)


@pytest.mark.parametrize("field", ["top_logprobs", "prompt_top_logprobs"])
@pytest.mark.parametrize("count", [-1, 2**32])
def test_sampling_counts_require_uint32(field, count):
    with pytest.raises(OverflowError):
        _params(field, count)


def _scored_recipe():
    recipe = Recipe()
    recipe.tokenizer = Tokenizer(_byte_tokenizer())
    return recipe


def _request(base, api, count):
    body = {"model": "test", "top_logprobs": count}
    if api == "chat/completions":
        body.update(
            messages=[{"role": "user", "content": "hi"}], max_tokens=1, logprobs=True
        )
    else:
        body.update(
            input="hi", max_output_tokens=1, include=["message.output_text.logprobs"]
        )
    return urllib.request.urlopen(
        urllib.request.Request(
            f"{base}/v1/{api}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=5,
    )


@pytest.mark.parametrize("api", ["chat/completions", "responses"])
@pytest.mark.parametrize("limit", [None, 20, 32, 1500])
def test_http_uses_engine_limit_for_admission(api, limit):
    runner = _scored_runner()
    effective_limit = 32 if limit is None else limit
    with running_server(
        runner,
        _scored_recipe(),
        engine_options=None if limit is None else {"max_top_logprobs": limit},
    ) as (_, engine, _, errors, base, _):
        assert engine.max_top_logprobs == effective_limit
        with pytest.raises(urllib.error.HTTPError) as caught:
            _request(base, api, effective_limit + 1)
        with caught.value as response:
            assert response.code == 400
            message = json.load(response)["error"]["message"]
            assert "top_logprobs" in message and str(effective_limit) in message
        assert runner.calls == []
        with _request(base, api, effective_limit) as response:
            scores = json.load(response)["resin"]["generation_logprobs"]
        assert len(scores) == 1 and len(scores[0]["top_logprobs"]) == 4
        assert runner.calls[0].sampling_params[0].top_logprobs == effective_limit
        assert not engine.poisoned and errors == []


@pytest.mark.parametrize("api", ["chat/completions", "responses"])
def test_http_zero_limit_keeps_selected_scores(api):
    runner = _scored_runner()
    with running_server(
        runner, _scored_recipe(), engine_options={"max_top_logprobs": 0}
    ) as (_, engine, _, errors, base, _):
        with _request(base, api, 0) as response:
            scores = json.load(response)["resin"]["generation_logprobs"]
        assert len(scores) == 1 and scores[0]["top_logprobs"] == []
        assert scores[0]["logprob"] == pytest.approx(-np.log(4))
        with pytest.raises(urllib.error.HTTPError) as caught:
            _request(base, api, 1)
        with caught.value as response:
            assert response.code == 400
        assert len(runner.calls) == 1
        assert not engine.poisoned and errors == []
