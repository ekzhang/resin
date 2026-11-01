"""Scores cross the real HTTP actor without losing token identity or bytes."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, cast

import numpy as np
import pytest
from recipe_fuzz.harness import Case, Part, normalize
from recipe_fuzz.test_differential import ADAPTERS
from test_http_lifecycle import running_server
from test_recipe import _byte_tokenizer

import resin
from resin import ForwardOutput, RunnerConfig
from resin.recipe import get_native_recipe


@pytest.fixture(scope="module", params=ADAPTERS)
def scored_recipe(request):
    return request.param, ADAPTERS[request.param]()


def request(base, path, body):
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


@pytest.mark.parametrize("metaspace", [False, True])
def test_byte_fallback_scores_preserve_raw_bytes_and_literal_metaspace(metaspace):
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteFallback, Fuse, Replace, Sequence
    from tokenizers.models import WordLevel

    backend = Tokenizer(
        WordLevel({"▁": 0, "<0xE2>": 1, "<0x82>": 2, "<0xAC>": 3, "<0x0000>": 4})
    )
    backend.decoder = Sequence(
        ([Replace("▁", " ")] if metaspace else []) + [ByteFallback(), Fuse()]
    )

    class Recipe:
        stop_token_ids = []

        tokenizer = resin.recipe.Tokenizer(backend)

        def render(self, messages, *, tools=(), options=None):
            return resin.recipe.RenderedPrompt([0])

        def create_parser(self, prompt):
            class Parser:
                def feed(self, _ids):
                    return []

            return Parser()

    class Runner:
        runner_config = RunnerConfig(max_accepted_per_step=4)
        offset = 0

        def forward(self, _plan):
            row = [0] if self.offset == 0 else [1, 2, 3, 4]
            self.offset += len(row)
            return ForwardOutput(
                token_ids=np.asarray(row, np.uint32),
                token_indptr=np.asarray([0, len(row)], np.uint32),
                logprobs=np.asarray([-np.inf] * len(row), np.float32),
            )

    with running_server(Runner(), Recipe()) as (_, _, _, errors, base, _):
        with request(
            base,
            "/v1/chat/completions",
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "logprobs": True,
            },
        ) as response:
            scores = json.load(response)["resin"]["generation_logprobs"]
        assert errors == []
    assert [score["bytes"] for score in scores] == [
        list((" " if metaspace else "▁").encode()),
        [0xE2],
        [0x82],
        [0xAC],
        list(b"<0x0000>"),
    ]
    assert all(score["logprob"] is None for score in scores)


def score_bytes(scores):
    return bytes(byte for score in scores for byte in score["bytes"])


def collect_scores(response, *, api, stream, partial_tokens):
    value: Any
    if not stream:
        value = json.load(response)
        raw = value["resin"]["generation_logprobs"]
    else:
        chunks = events(response)
        stream_errors = [chunk for chunk in chunks if "error" in chunk]
        assert not stream_errors, stream_errors
        if api == "chat/completions":
            choices = [chunk["choices"][0] for chunk in chunks if chunk["choices"]]
            raw = [
                token
                for chunk in chunks
                for token in chunk.get("resin", {}).get("generation_logprobs", [])
            ]
            deltas = [
                (
                    choice["delta"].get("content") or "",
                    (choice.get("logprobs") or {}).get("content", []),
                )
                for choice in choices
            ]
            tools = {}
            for choice in choices:
                for item in choice["delta"].get("tool_calls", []):
                    tool = tools.setdefault(
                        item["index"], {"function": {"name": "", "arguments": ""}}
                    )
                    for key in ("name", "arguments"):
                        tool["function"][key] += item.get("function", {}).get(key, "")
            value = {
                "choices": [
                    {
                        "message": {
                            "content": "".join(delta for delta, _ in deltas),
                            "reasoning_content": "".join(
                                choice["delta"].get("reasoning_content") or ""
                                for choice in choices
                            ),
                            "tool_calls": list(tools.values()),
                        },
                        "logprobs": {
                            "content": [
                                score for _, scores in deltas for score in scores
                            ]
                        },
                        "finish_reason": choices[-1]["finish_reason"],
                    }
                ]
            }
        else:
            raw = [
                token
                for chunk in chunks
                if chunk["type"] == "resin.generation_logprobs.delta"
                for token in chunk["logprobs"]
            ]
            deltas = [
                (chunk["delta"], chunk["logprobs"])
                for chunk in chunks
                if chunk["type"] == "response.output_text.delta"
            ]
            value = next(
                chunk["response"]
                for chunk in chunks
                if chunk["type"] in ("response.completed", "response.incomplete")
            )
            final_scores = [
                token
                for item in value["output"]
                if item["type"] == "message"
                for token in item["content"][0]["logprobs"]
            ]
            assert final_scores == [token for _, scores in deltas for token in scores]
            assert "".join(delta for delta, _ in deltas) == "".join(
                item["content"][0]["text"]
                for item in value["output"]
                if item["type"] == "message"
            )
        for delta, scores in deltas:
            # Scores must travel with their visible text, never later in an
            # empty delta. A cut source token deliberately stays raw-only.
            assert not scores or delta
            if delta and not partial_tokens:
                assert score_bytes(scores) == delta.encode(), (delta, scores)
    return scored_result(value, raw, api)


def scored_result(value: dict[str, Any], raw, api):
    if api == "chat/completions":
        choice = value["choices"][0]
        message = choice["message"]
        text, reasoning = message["content"], message.get("reasoning_content") or ""
        tools = [
            {
                "name": tool["function"]["name"],
                "args": json.loads(tool["function"]["arguments"]),
            }
            for tool in message.get("tool_calls", [])
        ]
        canonical, reason = choice["logprobs"]["content"], choice["finish_reason"]
    else:
        text = "".join(
            item["content"][0]["text"]
            for item in value["output"]
            if item["type"] == "message"
        )
        reasoning = "".join(
            part["text"]
            for item in value["output"]
            if item["type"] == "reasoning"
            for part in item["summary"]
        )
        tools = [
            {"name": item["name"], "args": json.loads(item["arguments"])}
            for item in value["output"]
            if item["type"] == "function_call"
        ]
        canonical = [
            token
            for item in value["output"]
            if item["type"] == "message"
            for token in item["content"][0]["logprobs"]
        ]
        reason = None
    return {
        "text": text,
        "reasoning": reasoning,
        "tools": tools,
        "raw": raw,
        "canonical": canonical,
        "reason": reason,
    }


def scored_exchange(
    recipe, ids, alternative, *, api, block, controls=None, partial_tokens=False
):
    class Runner:
        runner_config = RunnerConfig(max_accepted_per_step=max(1, len(ids)))
        offset = 0

        def forward(self, plan):
            params = plan.sampling_params[0]
            assert params.logprobs and params.top_logprobs == 2
            # Prefill returns one token, then a verified block can exercise
            # bytewise parsing and coalescing past the bounded client queue.
            end = len(ids) if block and self.offset else self.offset + 1
            row = ids[self.offset : end]
            scores = np.asarray(
                [-(index + 1) / 100 for index in range(self.offset, end)], np.float32
            )
            self.offset = end
            return ForwardOutput(
                token_ids=np.asarray(row, np.uint32),
                token_indptr=np.asarray([0, len(row)], np.uint32),
                logprobs=scores,
                logprob_indptr=np.asarray([0, len(row)], np.uint32),
                top_indptr=np.arange(0, len(row) * 2 + 1, 2, dtype=np.uint32),
                top_token_ids=np.asarray(
                    [token for actual in row for token in (actual, alternative)],
                    np.uint32,
                ),
                top_logprobs=np.asarray(
                    [score for actual in scores for score in (actual, -9)], np.float32
                ),
            )

    body = {
        "model": "test",
        "top_logprobs": 2,
        "max_completion_tokens": len(ids),
        "max_output_tokens": len(ids),
        "messages": [{"role": "user", "content": "hi"}],
        "input": "hi",
        **(controls or {}),
    }
    body.update(
        {"logprobs": True}
        if api == "chat/completions"
        else {"include": ["message.output_text.logprobs"]}
    )
    results = []
    for stream in (False, True):
        with running_server(Runner(), recipe) as (_, engine, _, errors, base, _):
            with request(base, "/v1/" + api, {**body, "stream": stream}) as response:
                result = collect_scores(
                    response, api=api, stream=stream, partial_tokens=partial_tokens
                )
            assert not engine.poisoned and not engine.has_work() and errors == []
        raw = result["raw"]
        assert [token["token_id"] for token in raw] == ids[: len(raw)]
        assert [token["token_index"] for token in raw] == list(range(len(raw)))
        assert [token["logprob"] for token in raw] == pytest.approx(
            [-(index + 1) / 100 for index in range(len(raw))]
        )
        for token in raw:
            assert token["top_logprobs"][0]["token_id"] == token["token_id"]
            assert token["top_logprobs"][0]["bytes"] == token["bytes"]
            assert token["top_logprobs"][0]["logprob"] == token["logprob"]
            assert token["top_logprobs"][1]["token_id"] == alternative
            assert token["top_logprobs"][1]["bytes"] == list(b"z")
            assert token["top_logprobs"][1]["logprob"] == -9
        indices = [token["token_index"] for token in result["canonical"]]
        assert indices == sorted(set(indices))
        for token in result["canonical"]:
            original = raw[token["token_index"]]
            for key in ("token_id", "bytes", "logprob", "top_logprobs"):
                assert token[key] == original[key]
        if not partial_tokens:
            assert score_bytes(result["canonical"]) == result["text"].encode()
        results.append(result)
    assert results[0] == results[1]
    return results[0]


def recipe_controls(name):
    return {"reasoning_effort": "high"} if name.startswith("deepseek_") else {}


@pytest.mark.parametrize("api", ["chat/completions", "responses"])
@pytest.mark.parametrize("block", [False, True])
def test_generation_scores_travel_with_text_for_every_stock_recipe(
    scored_recipe, api, block
):
    name, adapter = scored_recipe
    case = getattr(adapter, "case", lambda value: value)(
        Case(
            (
                Part("reasoning", "plan"),
                Part("text", "café \n 🧠" + (" stable" * 64 if block else "")),
                Part("tool", {"name": "weather", "args": {"n": 1}}),
            )
        )
    )
    ids = adapter.encode(case)
    expected = normalize(adapter.oracles[0][1](case))
    result = scored_exchange(
        adapter.recipe,
        ids,
        adapter.ordinary.encode("z").ids[0],
        api=api,
        block=block,
        controls=recipe_controls(name),
    )
    assert result["text"] == "".join(
        part.value
        for part in expected
        if part.kind == "text" and isinstance(part.value, str)
    )
    assert result["reasoning"] == "".join(
        part.value
        for part in expected
        if part.kind == "reasoning" and isinstance(part.value, str)
    )
    assert result["tools"] == [part.value for part in expected if part.kind == "tool"]
    assert len(result["raw"]) == len(ids)
    # Unicode scalars spanning multiple source tokens retain every score.
    assert any(token["token"] == "�" for token in result["canonical"])


@pytest.mark.parametrize("api", ["chat/completions", "responses"])
@pytest.mark.parametrize("stop_case", ["match", "mismatch", "length_prefix"])
def test_stock_recipe_stop_buffer_preserves_whole_token_scores(
    scored_recipe, api, stop_case
):
    name, adapter = scored_recipe
    text = {
        "match": "café 🧠 STOP tail",
        "mismatch": "café 🧠 ST? end",
        "length_prefix": "café 🧠 STO",
    }[stop_case]
    case = Case((Part("text", text),))
    ids = adapter.encode(case)
    expected = "".join(
        part.value
        for part in normalize(adapter.oracles[0][1](case))
        if part.kind == "text" and isinstance(part.value, str)
    )
    if stop_case == "length_prefix":
        # End after the held prefix, before model delimiters can resolve it.
        text_ids = adapter.ordinary.encode(text).ids
        at = next(
            index
            for index in range(len(ids))
            if ids[index : index + len(text_ids)] == text_ids
        )
        ids = ids[: at + len(text_ids)]
        expected = ("C:" if name.startswith("deepseek_") else "") + text
    elif stop_case == "match":
        expected = expected.split("STOP")[0]
    result = scored_exchange(
        adapter.recipe,
        ids,
        adapter.ordinary.encode("z").ids[0],
        api=api,
        block=True,
        controls={"stop": ["STOP"], **recipe_controls(name)},
    )
    assert result["text"] == expected
    if api == "chat/completions":
        assert result["reason"] == (
            "length" if stop_case == "length_prefix" else "stop"
        )
    if stop_case == "match":
        assert len(result["raw"]) < len(ids)
        assert result["raw"][-1]["bytes"] == list(b"P")
    else:
        assert len(result["raw"]) == len(ids)


def piece_recipe(name, pieces, *, terminal=True):
    """Replace byte tokens with a few deliberately awkward source tokens."""
    from tokenizers import Tokenizer

    adapter = ADAPTERS[name]()
    backend = Tokenizer.from_str(adapter.recipe.tokenizer.to_str())
    backend.add_tokens(pieces)
    piece_ids = [backend.token_to_id(piece) for piece in pieces]
    text_ids = adapter.ordinary.encode("".join(pieces)).ids
    ids = adapter.encode(Case((Part("text", "".join(pieces)),)))
    at = next(
        index
        for index in range(len(ids))
        if ids[index : index + len(text_ids)] == text_ids
    )
    tail = ids[at + len(text_ids) :] if terminal else []
    return (
        get_native_recipe("qwen3_8" if name == "qwen3_5" else name, backend),
        ids[:at] + piece_ids + tail,
        piece_ids,
    )


@pytest.mark.parametrize("api", ["chat/completions", "responses"])
@pytest.mark.parametrize(
    "pieces,expected,whole,terminal",
    [
        (["whole ", "prefixSTOPtail", "ignored"], "whole prefix", [0], True),
        (["whole ", "STO", "X", " end"], "whole STOX end", [0, 1, 2, 3], True),
        (["whole ", "STO"], "whole STO", [0, 1], False),
    ],
)
def test_stop_cutting_one_token_keeps_other_whole_token_scores(
    api, pieces, expected, whole, terminal
):
    recipe, ids, piece_ids = piece_recipe("inkling", pieces, terminal=terminal)
    result = scored_exchange(
        recipe,
        ids,
        recipe.tokenizer.encode("z")[0],
        api=api,
        block=True,
        controls={"stop": ["STOP"]},
        partial_tokens="STOP" in "".join(pieces),
    )
    assert result["text"] == expected
    assert [token["token_id"] for token in result["canonical"]] == [
        piece_ids[index] for index in whole
    ]
    if "STOP" in "".join(pieces):
        assert result["raw"][-1]["token_id"] == piece_ids[1]
        assert result["raw"][-1]["bytes"] == list(b"prefixSTOPtail")


@pytest.mark.parametrize("api", ["chat/completions", "responses"])
def test_qwen_trim_only_excludes_scores_for_partially_visible_tokens(api):
    pieces = ["  leading", " middle ", "trailing  "]
    recipe, ids, piece_ids = piece_recipe("qwen3_5", pieces)
    result = scored_exchange(
        recipe,
        ids,
        recipe.tokenizer.encode("z")[0],
        api=api,
        block=False,
        partial_tokens=True,
    )
    assert result["text"] == "leading middle trailing"
    assert [token["token_id"] for token in result["canonical"]] == [piece_ids[1]]
    assert score_bytes(result["canonical"]) == b" middle "


@pytest.mark.parametrize("api", ["chat/completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_prompt_scores_allow_zero_output_tokens(api, stream):
    backend = _byte_tokenizer()
    backend.add_special_tokens(
        [
            "<|message_system|>",
            "<|message_user|>",
            "<|message_tool|>",
            "<|content_xml|>",
        ]
    )
    recipe = get_native_recipe("inkling", backend)
    request_input = {"messages": [{"role": "user", "content": "score this"}]}
    prompt_ids = recipe.render(
        [
            resin.recipe.Message(
                role=resin.recipe.Role.USER, content=[resin.recipe.Text("score this")]
            )
        ]
    ).token_ids

    class Runner:
        runner_config = RunnerConfig(continuation_prefill=True)

        def forward(self, plan):
            assert plan.sampling_params[0].prompt_logprobs
            np.testing.assert_array_equal(plan.prompt_score_token_ids, prompt_ids[1:])
            assert not any(plan.sampling_mask)
            return ForwardOutput(
                # Non-sampling rows retain the runner's ordinary dummy token.
                token_ids=np.asarray([0], np.uint32),
                token_indptr=np.asarray([0, 1], np.uint32),
                prompt_logprobs=np.asarray(
                    [-index / 100 for index in range(1, len(prompt_ids))], np.float32
                ),
                prompt_logprob_indptr=np.asarray([0, len(prompt_ids) - 1], np.uint32),
            )

    body = {
        **request_input,
        "model": "test",
        "input": "score this",
        "max_completion_tokens": 0,
        "max_output_tokens": 0,
        "prompt_logprobs": True,
        "stream": stream,
    }
    with running_server(Runner(), recipe) as (_, engine, _, errors, base, _):
        with request(base, "/v1/" + api, body) as response:
            if stream:
                chunks = events(response)
                if api == "chat/completions":
                    scores = next(
                        chunk["resin"]["prompt_logprobs"]
                        for chunk in chunks
                        if "prompt_logprobs" in chunk.get("resin", {})
                    )
                else:
                    scores = next(
                        chunk["logprobs"]
                        for chunk in chunks
                        if chunk["type"] == "resin.prompt_logprobs"
                    )
            else:
                scores = json.load(response)["resin"]["prompt_logprobs"]
        assert errors == [] and not engine.poisoned
    assert scores[0] is None
    np.testing.assert_array_equal(
        [score["token_id"] for score in scores[1:]], prompt_ids[1:]
    )
    assert [score["logprob"] for score in scores[1:]] == pytest.approx(
        [-index / 100 for index in range(1, len(prompt_ids))]
    )


def custom_scored_recipe(*, with_spans, calls, invalid=None):
    backend = _byte_tokenizer()
    ids = backend.encode("ab").ids

    class Recipe:
        stop_token_ids = []

        tokenizer = resin.recipe.Tokenizer(backend)
        invalid_spans = invalid

        def render(self, messages, *, tools=(), options=None):
            return resin.recipe.RenderedPrompt([0])

        def create_parser(self, prompt):
            malformed = self.invalid_spans

            class Parser:
                count = 0

                def feed(self, token_ids):
                    calls.append("feed")
                    self.count += len(token_ids)
                    # The ordinary API preserves provenance while buffering,
                    # identically for scored and unscored requests.
                    if with_spans and self.count < 2:
                        return []
                    if malformed == "event_order":
                        return [
                            resin.recipe.TextDelta("b", token_spans=[(1, 0, 1, 0, 1)]),
                            resin.recipe.TextDelta("a", token_spans=[(0, 0, 1, 0, 1)]),
                        ]
                    spans: Any = [(index, 0, 1, index, index + 1) for index in range(2)]
                    if malformed == "payload":
                        spans = "token_spans must be a sequence of five-integer tuples"
                    elif malformed == "arity":
                        spans[-1] = (1, 0, 1, 1)
                    elif malformed == "negative":
                        spans[-1] = (-1, 0, 1, 1, 2)
                    elif malformed == "overflow":
                        spans[-1] = (2**32, 0, 1, 1, 2)
                    elif malformed == "future":
                        spans[-1] = (2, 0, 1, 1, 2)
                    elif malformed == "source":
                        spans[-1] = (1, 0, 2, 1, 2)
                    elif malformed == "text":
                        spans[-1] = (1, 0, 1, 1, 3)
                    elif malformed == "empty":
                        spans[-1] = (1, 0, 0, 1, 2)
                    elif malformed == "order":
                        spans.reverse()
                    return [
                        resin.recipe.TextDelta(
                            "ab" if with_spans else "ab"[self.count - 1],
                            token_spans=cast(
                                Any, spans if with_spans or malformed else ()
                            ),
                        )
                    ]

                def finish(self, _reason):
                    calls.append("finish")
                    return []

            return Parser()

    class Runner:
        def forward(self, plan):
            token = ids[0] if plan.positions[0] == 0 else ids[1]
            return ForwardOutput(
                token_ids=np.asarray([token], np.uint32),
                token_indptr=np.asarray([0, 1], np.uint32),
                logprobs=(
                    np.asarray([-1], np.float32)
                    if plan.sampling_params[0].logprobs
                    else None
                ),
            )

    return Recipe(), Runner(), ids


@pytest.mark.parametrize("with_spans", [False, True])
@pytest.mark.parametrize("scored", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_custom_parser_ordinary_feed_preserves_provenance_for_all_requests(
    with_spans, scored, stream
):
    calls = []
    recipe, runner, ids = custom_scored_recipe(with_spans=with_spans, calls=calls)
    with running_server(runner, recipe) as (_, engine, _, errors, base, _):
        with request(
            base,
            "/v1/chat/completions",
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
                "logprobs": scored,
                "stream": stream,
            },
        ) as response:
            if scored:
                result = collect_scores(
                    response,
                    api="chat/completions",
                    stream=stream,
                    partial_tokens=not with_spans,
                )
                assert result["text"] == "ab"
                assert [score["token_id"] for score in result["raw"]] == ids
                assert [score["token_id"] for score in result["canonical"]] == (
                    ids if with_spans else []
                )
            elif stream:
                assert (
                    "".join(
                        choice["delta"].get("content", "")
                        for chunk in events(response)
                        for choice in chunk["choices"]
                    )
                    == "ab"
                )
            else:
                assert json.load(response)["choices"][0]["message"]["content"] == "ab"
        assert not engine.poisoned and not engine.has_work() and errors == []
    assert calls == ["feed", "feed", "finish"]


@pytest.mark.parametrize(
    ("scored", "invalid"),
    [
        (scored, invalid)
        for scored in (False, True)
        for invalid in (
            "payload",
            "arity",
            "negative",
            "overflow",
            "future",
            "source",
            "text",
            "empty",
            "order",
            "event_order",
        )
        if scored or invalid != "source"
    ],
)
def test_invalid_source_spans_fail_every_request_without_poisoning_engine(
    scored, invalid
):
    from urllib.error import HTTPError

    calls = []
    recipe, runner, _ = custom_scored_recipe(
        with_spans=True, calls=calls, invalid=invalid
    )
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "logprobs": scored,
    }
    with running_server(runner, recipe) as (_, engine, _, errors, base, _):
        with pytest.raises(HTTPError) as caught:
            request(base, "/v1/chat/completions", body)
        assert caught.value.code == 500
        assert "parser" in caught.value.read().decode()
        recipe.invalid_spans = None
        with request(base, "/v1/chat/completions", body) as response:
            assert json.load(response)["choices"][0]["message"]["content"] == "ab"
        assert not engine.poisoned and not engine.has_work() and errors == []
