from __future__ import annotations

from types import SimpleNamespace

from http_packets import decoded_events, ingress
from output_helpers import forward_output

import resin
from resin import CacheGroupLayout, CacheLayout, Engine, RunnerConfig
from resin._resin import _HttpSession


class RecordingCachedRunner:
    def __init__(self) -> None:
        layout = CacheLayout(parent_capacity=16, groups=[CacheGroupLayout(2)])
        self.cache_layout = layout
        self.runner_config = RunnerConfig(
            continuation_prefill=True,
            cache_layout=layout,
        )
        self.plans = []

    def bind_resources(self, binding) -> None:
        assert binding.cache_pool_id is not None
        assert binding.cache_layout is not None

    def forward(self, plan):
        self.plans.append(plan)
        return forward_output(
            [
                plan.token_ids[end - 1] + 1
                for _, end in map(plan.token_range, range(plan.num_sequences))
            ]
        )


class TokenRecipe:
    tokenizer = None
    stop_token_ids = []

    def render(self, messages, *, tools=(), options=None):
        label = messages[0].content[0].text
        tokens = [1, 2, 3, 4]
        if label == "long":
            tokens.append(5)
        return resin.recipe.RenderedPrompt(tokens)

    def create_parser(self, prompt):
        return SimpleNamespace(feed=lambda ids: [resin.recipe.TextDelta("ok")])


def admission(request_id, label, *, cacheable=None):
    request = {
        "id": request_id,
        "model": "test",
        "messages": [{"role": "user", "content": label}],
        "tools": [],
        "max_output_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "seed": None,
        "logprobs": False,
        "top_logprobs": 0,
        "prompt_logprobs": False,
        "routed_dp_rank": None,
        "stop": [],
        "reasoning_effort": None,
        "extra": {},
    }
    if cacheable is not None:
        request["cacheable"] = cacheable
    return ingress([request])


def drain(session, engine):
    events = []
    while engine.has_work():
        events.extend(decoded_events(session.process(engine.step())))
    return events


def finished_cached_tokens(events, request_id):
    finished = [
        event
        for event_id, event in events
        if event_id == request_id and event["type"] == "finished"
    ]
    assert len(finished) == 1, events
    return finished[0]["data"]["cached_tokens"]


def test_http_session_uncacheable_request_does_not_reuse_or_publish_prefixes() -> None:
    runner = RecordingCachedRunner()
    engine = Engine(runner, prefix_cache=True)
    session = _HttpSession(engine, TokenRecipe())
    session.admit(admission("private", "short", cacheable=False))
    events = drain(session, engine)
    assert finished_cached_tokens(events, "private") == 0
    assert engine.prefix_cache_stats == (0, 0, 0, 0)

    session.admit(admission("seed", "short", cacheable=True))
    events = drain(session, engine)
    assert finished_cached_tokens(events, "seed") == 0
    assert engine.prefix_cache_stats == (2, 0, 1, 0)

    session.admit(admission("skipped", "long", cacheable=False))
    events = decoded_events(session.process(engine.step()))
    assert runner.plans[-1].context_lengths.tolist() == [0]
    assert runner.plans[-1].token_ids[0] == 1
    events.extend(drain(session, engine))
    assert finished_cached_tokens(events, "skipped") == 0
    assert engine.prefix_cache_stats == (2, 0, 1, 0)

    session.admit(admission("reused", "long"))
    events = decoded_events(session.process(engine.step()))
    assert runner.plans[-1].token_ids.tolist() == [5]
    events.extend(drain(session, engine))
    assert finished_cached_tokens(events, "reused") == 4
    assert engine.prefix_cache_stats == (2, 1, 1, 0)
    engine.close()
