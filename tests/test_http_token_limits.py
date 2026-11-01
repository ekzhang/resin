"""HTTP token limits preserve the engine's context budget and explicit overrides."""

from __future__ import annotations

import json

import pytest
from test_http_lifecycle import Recipe, Runner, running_server
from test_http_stops import events, send

from resin.recipe import RenderedPrompt


class BudgetRecipe(Recipe):
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens

    def render(self, messages, *, tools=(), options=None):
        return RenderedPrompt(list(range(self.prompt_tokens)))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "api,prompt_tokens,limits,expected_tokens",
    [
        ("chat/completions", 3, {}, 6),
        ("chat/completions", 8, {}, 1),
        ("responses", 3, {}, 6),
        ("responses", 8, {}, 1),
        ("chat/completions", 3, {"max_tokens": 2}, 2),
        ("chat/completions", 3, {"max_completion_tokens": 3}, 3),
        (
            "chat/completions",
            3,
            {"max_tokens": 2, "max_completion_tokens": 4},
            4,
        ),
    ],
)
def test_http_generation_limits_use_remaining_context_or_explicit_override(
    api, prompt_tokens, limits, expected_tokens, stream
):
    # The last sampled token does not need another context position in the engine.
    runner = Runner()
    body = {"model": "test", "stream": stream, **limits}
    if api == "chat/completions":
        body["messages"] = [{"role": "user", "content": "hi"}]
        body["stream_options"] = {"include_usage": True}
    else:
        body["input"] = "hi"
    with running_server(
        runner,
        BudgetRecipe(prompt_tokens),
        engine_options={"context_length": 8},
    ) as (_, engine, _, errors, base, _):
        with send(base, "/v1/" + api, body) as response:
            if api == "chat/completions":
                if stream:
                    values = events(response)
                    choices = [
                        choice for value in values for choice in value["choices"]
                    ]
                    text = "".join(
                        choice["delta"].get("content") or "" for choice in choices
                    )
                    usage = next(
                        value["usage"] for value in values if value.get("usage")
                    )
                    finish_reason = choices[-1]["finish_reason"]
                else:
                    value = json.load(response)
                    choice = value["choices"][0]
                    text, usage = choice["message"]["content"], value["usage"]
                    finish_reason = choice["finish_reason"]
                assert finish_reason == "length"
                assert usage["prompt_tokens"] == prompt_tokens
                assert usage["completion_tokens"] == expected_tokens
            else:
                value = (
                    next(
                        item["response"]
                        for item in events(response)
                        if item["type"] == "response.incomplete"
                    )
                    if stream
                    else json.load(response)
                )
                text = "".join(
                    part["text"]
                    for item in value["output"]
                    if item["type"] == "message"
                    for part in item["content"]
                )
                assert value["usage"]["input_tokens"] == prompt_tokens
                assert value["usage"]["output_tokens"] == expected_tokens
                assert value["status"] == "incomplete"
                assert value["incomplete_details"] == {"reason": "max_output_tokens"}
        assert text == "ok" * expected_tokens
        assert sum(runner.counts.values()) == expected_tokens
        assert not engine.has_work()
        assert not engine.poisoned
        assert errors == []
