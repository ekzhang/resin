"""Protocol token counts use generation's recipe without submitting inference."""

from __future__ import annotations

import json
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_http_lifecycle import Runner, post, request_body, running_server
from test_http_stops import send
from test_recipe import _byte_tokenizer

from resin.recipe import RenderedPrompt, get_native_recipe


class CountingRecipe:
    tokenizer = None
    stop_token_ids = []

    def __init__(self):
        self.parsers = 0
        self.renders = []
        self.prompts = []

    def render(self, messages, *, tools=(), options=None):
        assert options is not None
        text = "|".join(part.text for message in messages for part in message.content)
        if text == "bad-render":
            raise ValueError("injected rendering failure")
        signature = (
            text,
            tuple(tool.name for tool in tools),
            options.reasoning_effort,
            options.thinking_mode,
        )
        self.renders.append(signature)
        count = len(repr(signature))
        prompt = RenderedPrompt([1] * count)
        self.prompts.append(prompt)
        return prompt

    def create_parser(self, prompt):
        assert any(prompt is rendered for rendered in self.prompts)
        self.parsers += 1
        return SimpleNamespace(feed=lambda ids: [])


CASES = [
    (
        "/v1/responses",
        "/v1/responses/input_tokens",
        {
            "input": "hello",
            "instructions": "Be concise",
            "reasoning": {"effort": "high"},
            "tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "parameters": {"type": "object"},
                }
            ],
        },
        {"max_output_tokens": 1},
    ),
    (
        "/v1/messages",
        "/v1/messages/count_tokens",
        {
            "messages": [{"role": "user", "content": "hello"}],
            "system": [{"type": "text", "text": "Be concise"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [{"name": "weather", "input_schema": {"type": "object"}}],
        },
        {"max_tokens": 1},
    ),
]


@pytest.mark.parametrize("generation_path,count_path,body,limits", CASES)
@pytest.mark.parametrize("native", [False, True])
def test_count_matches_generation_usage_without_inference(
    generation_path, count_path, body, limits, native
):
    recipe = CountingRecipe()
    if native:
        tokenizer = _byte_tokenizer()
        tokenizer.add_special_tokens(
            [
                "<｜User｜>",
                "<｜System｜>",
                "<｜Assistant｜>",
                "<｜begin▁of▁sentence｜>",
                "<｜end▁of▁sentence｜>",
                "<think>",
                "</think>",
                "｜DSML｜",
                "<|start|>",
                "<|end|>",
                "<|return|>",
                "<|call|>",
                "<|channel|>",
                "<|message|>",
                "<|constrain|>",
            ]
        )
        family = "gpt_oss" if generation_path == "/v1/responses" else "deepseek_v41"
        recipe = get_native_recipe(family, tokenizer)
    runner = Runner()
    with running_server(runner, recipe) as (_, engine, _, errors, base, _):
        with send(base, count_path, {"model": "test", **body}) as response:
            count = json.load(response)
        assert count["input_tokens"] > 0
        assert count.get("object") == (
            "response.input_tokens" if generation_path == "/v1/responses" else None
        )
        assert runner.counts == {}
        assert not engine.has_work()
        if isinstance(recipe, CountingRecipe):
            assert recipe.parsers == 0
        with send(
            base, generation_path, {"model": "test", **body, **limits}
        ) as response:
            generated = json.load(response)
        assert generated["usage"]["input_tokens"] == count["input_tokens"]
        assert runner.counts
        if isinstance(recipe, CountingRecipe):
            assert recipe.parsers == 1
            assert recipe.renders[0] == recipe.renders[1]
        assert errors == []


@pytest.mark.parametrize("_,count_path,body,limits", CASES)
def test_count_does_not_reserve_the_engine_context(_, count_path, body, limits):
    runner, recipe = Runner(), CountingRecipe()
    with running_server(runner, recipe, engine_options={"context_length": 2}) as (
        _,
        engine,
        _,
        errors,
        base,
        _,
    ):
        with send(base, count_path, {"model": "test", **body}) as response:
            assert json.load(response)["input_tokens"] > engine.context_length
        assert not engine.has_work()
        assert runner.counts == {}
        assert recipe.parsers == 0
        assert errors == []


@pytest.mark.parametrize("generation_path,count_path,body,limits", CASES)
def test_count_validation_and_render_failures(
    generation_path, count_path, body, limits
):
    recipe = CountingRecipe()
    runner = Runner()
    with running_server(runner, recipe) as (_, _, _, errors, base, _):
        bad_body = (
            {"model": "test", "input": "bad-render"}
            if generation_path == "/v1/responses"
            else {
                "model": "test",
                "messages": [{"role": "user", "content": "bad-render"}],
            }
        )
        with pytest.raises(urllib.error.HTTPError) as failure:
            send(base, count_path, bad_body)
        assert failure.value.code == 500
        assert (
            "injected rendering failure" in json.load(failure.value)["error"]["message"]
        )
        with pytest.raises(urllib.error.HTTPError) as failure:
            send(base, count_path, {"model": "wrong", **body})
        assert failure.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as failure:
            send(base, count_path, {"model": "test", **body, "routed_dp_rank": 1})
        assert failure.value.code == 400
        with send(base, count_path, {"model": "test", **body}) as response:
            assert json.load(response)["input_tokens"] > 0
        assert runner.counts == {}
        assert errors == []


def test_slow_count_render_does_not_block_inference():
    rendering, resume = threading.Event(), threading.Event()

    class SlowRecipe(CountingRecipe):
        def render(self, messages, *, tools=(), options=None):
            if messages[0].content[0].text == "slow":
                rendering.set()
                assert resume.wait(timeout=5)
            return super().render(messages, tools=tools, options=options)

    recipe, runner = SlowRecipe(), Runner()
    with running_server(runner, recipe, render_workers=2, max_active_requests=2) as (
        _,
        _,
        _,
        errors,
        base,
        _,
    ):

        def count():
            with send(
                base, "/v1/responses/input_tokens", {"model": "test", "input": "slow"}
            ) as response:
                return json.load(response)

        with ThreadPoolExecutor() as pool:
            pending = pool.submit(count)
            try:
                assert rendering.wait(timeout=2)
                with send(
                    base,
                    "/v1/responses",
                    {"model": "test", "input": "fast", "max_output_tokens": 1},
                ) as response:
                    assert json.load(response)["usage"]["output_tokens"] == 1
            finally:
                resume.set()
            assert pending.result(timeout=5)["input_tokens"] > 0
        assert recipe.parsers == 1
        assert errors == []


def test_counts_bypass_a_full_inference_admission_and_queue():
    forwarding, resume = threading.Event(), threading.Event()

    class BlockingRunner(Runner):
        def forward(self, plan):
            forwarding.set()
            assert resume.wait(timeout=5)
            return super().forward(plan)

    recipe = CountingRecipe()
    with running_server(
        BlockingRunner(),
        recipe,
        max_active_requests=1,
        request_queue_capacity=1,
        render_workers=1,
    ) as (_, _, _, errors, base, _):
        try:
            with post(base, request_body(stream=True)) as first:
                assert forwarding.wait(timeout=2)
                with post(base, request_body(stream=True)) as queued:
                    with pytest.raises(urllib.error.HTTPError) as failure:
                        post(base, request_body())
                    assert failure.value.code == 429
                    for _, path, body, _ in CASES:
                        with send(base, path, {"model": "test", **body}) as response:
                            assert json.load(response)["input_tokens"] > 0
                    assert recipe.parsers == 1
                    resume.set()
                    first.read()
                    queued.read()
            assert errors == []
        finally:
            resume.set()


def test_shutdown_does_not_join_a_blocked_render():
    rendering, resume = threading.Event(), threading.Event()

    class BlockingRecipe(CountingRecipe):
        def render(self, messages, *, tools=(), options=None):
            rendering.set()
            assert resume.wait(timeout=5)
            return super().render(messages, tools=tools, options=options)

    with running_server(Runner(), BlockingRecipe(), render_workers=1) as (
        server,
        _,
        thread,
        errors,
        base,
        _,
    ):
        try:
            with post(base, request_body(stream=True)):
                assert rendering.wait(timeout=2)
            server.shutdown()
            thread.join(timeout=2)
            assert not thread.is_alive(), "shutdown waited for prompt rendering"
            assert errors == []
        finally:
            resume.set()


@pytest.mark.parametrize("malformed", [False, True])
def test_count_rejects_invalid_recipe_output(malformed):
    class InvalidRecipe(CountingRecipe):
        def render(self, messages, *, tools=(), options=None):
            return object() if malformed else RenderedPrompt([])

    runner = Runner()
    with running_server(runner, InvalidRecipe()) as (_, engine, _, errors, base, _):
        with pytest.raises(urllib.error.HTTPError) as failure:
            send(base, "/v1/responses/input_tokens", {"model": "test", "input": "hi"})
        assert failure.value.code == 500
        assert not engine.has_work()
        assert runner.counts == {}
        assert errors == []


def test_messages_generation_still_requires_output_limit():
    with running_server(Runner(), CountingRecipe()) as (_, _, _, _, base, _):
        with pytest.raises(urllib.error.HTTPError) as failure:
            send(
                base,
                "/v1/messages",
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert failure.value.code == 400
