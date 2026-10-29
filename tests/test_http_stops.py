"""HTTP stops match decoded text, including inside and across generated tokens."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from output_helpers import forward_output
from test_http_lifecycle import running_server

import resin


class TextRecipe:
    tokenizer = None
    stop_token_ids = []

    def __init__(self, pieces):
        self.pieces = pieces

    def render(self, messages, *, tools=(), options=None):
        return resin.recipe.RenderedPrompt([1])

    def create_parser(self, prompt):
        pieces = self.pieces

        class Parser:
            def feed(self, ids):
                return [resin.recipe.TextDelta(pieces[token]) for token in ids]

        return Parser()


class PiecesRunner:
    def __init__(self, pieces, *, block=False):
        self.runner_config = resin.RunnerConfig(
            max_accepted_per_step=max(1, len(pieces))
        )
        self.pieces = pieces
        self.block = block
        self.position = 0
        self.released = []

    def forward(self, plan):
        assert plan.num_sequences == 1
        if self.block and self.position:
            self.position = len(self.pieces)
            return forward_output([list(range(1, len(self.pieces)))])
        token = self.position
        self.position += 1
        return forward_output([token])

    def release(self, ids):
        self.released.extend(ids)


def send(base, path, body):
    return urllib.request.urlopen(
        urllib.request.Request(
            base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=5,
    )


def events(response):
    return [
        json.loads(line[6:])
        for line in response.read().decode().splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "pieces,stops,expected,reason",
    [
        (["hello S", "T", "OPtail", "later"], ["STOP"], "hello ", "stop"),
        (["hello🦊", "E", "NDtail", "later"], ["🦊END"], "hello", "stop"),
        (["hello STOP tail", "later"], ["STOP"], "hello ", "stop"),
        (["a", "aa", "x"], ["aab", "abc"], "aaax", "length"),
        (["aST", "O"], ["STOP"], "aSTO", "length"),
        (["aENDbSTOP", "later"], ["STOP", "END"], "a", "stop"),
    ],
)
def test_chat_stops_inside_tokens_across_tokens_and_at_length_limit(
    stream, pieces, stops, expected, reason
):
    runner = PiecesRunner(pieces)
    with running_server(runner, TextRecipe(pieces)) as (_, engine, _, errors, base, _):
        with send(
            base,
            "/v1/chat/completions",
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": len(pieces),
                "stream": stream,
                "stop": stops,
            },
        ) as response:
            if stream:
                choices = [item["choices"][0] for item in events(response)]
                text = "".join(item["delta"].get("content") or "" for item in choices)
                finish = choices[-1]["finish_reason"]
            else:
                choice = json.load(response)["choices"][0]
                text, finish = choice["message"]["content"], choice["finish_reason"]
        assert text == expected
        assert finish == reason
        assert not engine.has_work()
        assert runner.released
        assert not engine.poisoned
        assert errors == []


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_stop_reports_matching_marker(stream):
    pieces = ["answer EN", "Dtail", "ignored"]
    with running_server(PiecesRunner(pieces), TextRecipe(pieces)) as (
        _,
        _,
        _,
        errors,
        base,
        _,
    ):
        with send(
            base,
            "/v1/messages",
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": len(pieces),
                "stream": stream,
                "stop_sequences": ["END"],
            },
        ) as response:
            if stream:
                values = events(response)
                text = "".join(
                    v["delta"].get("text", "")
                    for v in values
                    if v["type"] == "content_block_delta"
                )
                result = next(
                    v["delta"] for v in values if v["type"] == "message_delta"
                )
            else:
                result = json.load(response)
                text = "".join(part.get("text", "") for part in result["content"])
        assert text == "answer "
        assert result["stop_reason"] == "stop_sequence"
        assert result["stop_sequence"] == "END"
        assert errors == []


def test_empty_stop_is_rejected_before_inference():
    runner = PiecesRunner(["ignored"])
    with running_server(runner, TextRecipe(runner.pieces)) as (
        _,
        _,
        _,
        errors,
        base,
        _,
    ):
        with pytest.raises(urllib.error.HTTPError) as caught:
            send(
                base,
                "/v1/chat/completions",
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stop": [""],
                },
            )
        assert caught.value.code == 400
        assert runner.position == 0
        assert errors == []


@pytest.mark.parametrize("stream", [False, True])
def test_stop_discards_remaining_verified_block_and_reports_consumed_tokens(stream):
    pieces = ["hello S", "T", "OP tail", "ignored"]
    with running_server(PiecesRunner(pieces, block=True), TextRecipe(pieces)) as (
        _,
        _,
        _,
        errors,
        base,
        _,
    ):
        with send(
            base,
            "/v1/chat/completions",
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": len(pieces),
                "stream": stream,
                "stop": ["STOP"],
                "stream_options": {"include_usage": True},
            },
        ) as response:
            if stream:
                values = events(response)
                text = "".join(
                    v["choices"][0]["delta"].get("content", "")
                    for v in values
                    if v["choices"]
                )
                used = values[-1]["usage"]["completion_tokens"]
            else:
                value = json.load(response)
                text = value["choices"][0]["message"]["content"]
                used = value["usage"]["completion_tokens"]
        assert text == "hello "
        assert used == 3
        assert errors == []


@pytest.mark.parametrize("stream", [False, True])
def test_responses_stop_marker_is_hidden(stream):
    pieces = ["answer EN", "Dtail", "ignored"]
    with running_server(PiecesRunner(pieces), TextRecipe(pieces)) as (
        _,
        _,
        _,
        errors,
        base,
        _,
    ):
        with send(
            base,
            "/v1/responses",
            {
                "model": "test",
                "input": "hi",
                "max_output_tokens": len(pieces),
                "stream": stream,
                "stop": ["END"],
            },
        ) as response:
            if stream:
                values = events(response)
                text = "".join(
                    value["delta"]
                    for value in values
                    if value["type"] == "response.output_text.delta"
                )
            else:
                result = json.load(response)
                text = "".join(
                    part["text"]
                    for item in result["output"]
                    if item["type"] == "message"
                    for part in item["content"]
                )
        assert text == "answer "
        assert errors == []
