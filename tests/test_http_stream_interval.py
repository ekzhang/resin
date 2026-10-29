"""Stream batching changes delivery cadence, preserving parsing and score identity."""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from functools import partial

import numpy as np
import pytest
import test_http_lifecycle as lifecycle
from http_packets import decoded_events, ingress
from test_http_logprobs import collect_scores, events, request, score_bytes
from test_recipe import _byte_tokenizer

import resin
from resin import Engine, ForwardOutput
from resin._resin import _HttpSession
from resin.server import HttpServer


class Recipe:
    tokenizer = None
    stop_token_ids = []

    def __init__(self, pieces, *, eos=None, silent=()):
        backend = _byte_tokenizer()
        backend.add_tokens(pieces)
        self.tokenizer = resin.recipe.Tokenizer(backend)
        self.ids = [self.tokenizer.token_to_id(piece) for piece in pieces]
        self.pieces = dict(zip(self.ids, pieces, strict=True))
        self.eos = self.tokenizer.token_to_id(eos) if eos else None
        self.stop_token_ids = [] if self.eos is None else [self.eos]
        self.silent = silent

    def render(self, messages, *, tools=(), options=None):
        return resin.recipe.RenderedPrompt([0])

    def create_parser(self, prompt):
        pieces, eos, silent = self.pieces, self.eos, self.silent

        class Parser:
            position = 0

            def feed(self, ids):
                output = []
                for token in ids:
                    position, self.position = self.position, self.position + 1
                    if token != eos and pieces[token] not in silent:
                        text = pieces[token]
                        length = len(text.encode())
                        output.append(
                            resin.recipe.TextDelta(
                                text, token_spans=[(position, 0, length, 0, length)]
                            )
                        )
                return output

        return Parser()


class Runner:
    def __init__(self, script):
        self.runner_config = resin.RunnerConfig(
            max_accepted_per_step=max(map(len, script))
        )
        self.script = script
        self.steps, self.positions = {}, {}
        self.released = []

    def forward(self, plan):
        rows, values = [], []
        for row, request_id in enumerate(plan.sequence_ids):
            request_id = int(request_id)
            step = self.steps.get(request_id, 0)
            position = self.positions.get(request_id, 0)
            tokens = self.script[step]
            rows.append(tokens)
            if plan.sampling_params[row].logprobs:
                values.extend(
                    -(index + 1) / 10
                    for index in range(position, position + len(tokens))
                )
            self.steps[request_id], self.positions[request_id] = (
                step + 1,
                position + len(tokens),
            )
        offsets = np.asarray([0, *np.cumsum([len(row) for row in rows])], np.uint32)
        return ForwardOutput(
            token_ids=np.asarray([token for row in rows for token in row], np.uint32),
            token_indptr=offsets,
            logprobs=np.asarray(values, np.float32) if values else None,
        )

    def release(self, ids):
        self.released.extend(ids)


def admission(request_id, limit, *, stops=()):
    return ingress(
        [
            {
                "id": request_id,
                "model": "test",
                "messages": [],
                "tools": [],
                "max_output_tokens": limit,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "seed": None,
                "logprobs": True,
                "top_logprobs": 0,
                "prompt_logprobs": False,
                "routed_dp_rank": None,
                "stop": list(stops),
                "reasoning_effort": None,
                "extra": {},
            }
        ]
    )


@contextmanager
def session_case(pieces, interval, *, blocks=None, eos=None, silent=()):
    recipe = Recipe(pieces, eos=eos, silent=silent)
    script = (
        [[token] for token in recipe.ids]
        if blocks is None
        else [[recipe.ids[index] for index in block] for block in blocks]
    )
    runner = Runner(script)
    engine = Engine(runner)
    session = _HttpSession(engine, recipe, stream_interval=interval)
    try:
        yield session, engine, runner, recipe
    finally:
        session.close()
        engine.close()


def tick(session, engine):
    return decoded_events(session.process(engine.step()))


def text_events(events):
    return [
        (request_id, event["data"])
        for request_id, event in events
        if event["type"] == "text_delta"
    ]


def assert_scores(events, expected_ids):
    raw = [
        score
        for _, event in events
        if event["type"] == "logprobs"
        for score in event["data"]
    ]
    assert [score["token_id"] for score in raw] == expected_ids
    assert [score["token_index"] for score in raw] == list(range(len(raw)))
    assert [score["logprob"] for score in raw] == pytest.approx(
        [-(index + 1) / 10 for index in range(len(raw))]
    )
    for _, data in text_events(events):
        assert score_bytes(data["logprobs"]) == data["delta"].encode()


@pytest.mark.parametrize(
    "interval,expected",
    [(1, ["a", "b", "c", "d", "e", "f"]), (3, ["a", "", "", "bcd", "", "ef"])],
)
def test_first_output_is_immediate_then_token_interval_and_final_residual(
    interval, expected
):
    with session_case(list("abcdef"), interval) as (session, engine, _, recipe):
        session.admit(admission("one", 6))
        output = []
        for delta in expected:
            step = tick(session, engine)
            assert "".join(data["delta"] for _, data in text_events(step)) == delta
            output.extend(step)
        assert output[-1][1]["type"] == "finished"
        assert output[-1][1]["data"]["reason"] == "length"
        assert_scores(output, recipe.ids)
        assert session.active_count == 0 and not engine.has_work()


def test_verified_blocks_count_tokens_and_coalesce_without_waiting_for_more_steps():
    with session_case(list("abcdefg"), 3, blocks=[[0], [1, 2], [3, 4], [5, 6]]) as (
        session,
        engine,
        _,
        recipe,
    ):
        session.admit(admission("block", 7))
        output = []
        for expected in ("a", "", "bcde", "fg"):
            step = tick(session, engine)
            assert [data["delta"] for _, data in text_events(step)] == (
                [expected] if expected else []
            )
            output.extend(step)
        assert_scores(output, recipe.ids)


def test_raw_only_flushes_never_consume_the_first_semantic_output():
    with session_case(["~"] * 2 + ["a", "b"], 2, silent=("~",)) as (
        session,
        engine,
        _,
        recipe,
    ):
        session.admit(admission("headers", 4))
        output = []
        for index, expected in enumerate(("", "", "a", "b")):
            step = tick(session, engine)
            assert "".join(data["delta"] for _, data in text_events(step)) == expected
            if index == 1:
                assert (
                    sum(
                        len(event["data"])
                        for _, event in step
                        if event["type"] == "logprobs"
                    )
                    == 2
                )
            output.extend(step)
        assert_scores(output, recipe.ids)


def test_native_session_rejects_zero_interval():
    recipe = Recipe(["a"])
    engine = Engine(Runner([[recipe.ids[0]]]))
    try:
        with pytest.raises(ValueError, match="stream_interval"):
            _HttpSession(engine, recipe, stream_interval=0)
    finally:
        engine.close()


@pytest.mark.parametrize(
    "ending,pieces,limit,stops,expected,reason,consumed",
    [
        ("eos", ["a", "b", "<|content_model_end_sampling|>"], 9, [], "ab", "stop", 3),
        ("length", ["a", "b", "c"], 3, [], "abc", "length", 3),
        ("stop", ["a", "b", "ST", "OPtail", "ignored"], 9, ["STOP"], "ab", "stop", 4),
        ("held_prefix", ["a", "ST"], 2, ["STOP"], "aST", "length", 2),
    ],
)
def test_terminal_paths_flush_residual_and_stops_still_parse_every_step(
    ending, pieces, limit, stops, expected, reason, consumed
):
    eos = pieces[-1] if ending == "eos" else None
    with session_case(pieces, 8, eos=eos) as (session, engine, runner, recipe):
        session.admit(admission("terminal", limit, stops=stops))
        first = tick(session, engine)
        assert [data["delta"] for _, data in text_events(first)] == ["a"]
        output = first
        for index in range(1, consumed):
            step = tick(session, engine)
            if index + 1 < consumed:
                assert text_events(step) == []
            output.extend(step)
        assert "".join(data["delta"] for _, data in text_events(output)) == expected
        assert output[-1][1]["data"]["reason"] == reason
        assert_scores(output, recipe.ids[:consumed])
        assert runner.released and session.active_count == 0
        assert not engine.has_work() and not engine.poisoned


def test_intervals_belong_to_each_request_and_cancel_discards_only_its_pending_output():
    with session_case(list("abcdef"), 3) as (session, engine, runner, recipe):
        output = []

        def next_text():
            step = tick(session, engine)
            output.extend(step)
            return text_events(step)

        session.admit(admission("first", 6))
        assert [request_id for request_id, _ in next_text()] == ["first"]
        assert next_text() == []
        session.admit(admission("second", 6))
        assert [(name, data["delta"]) for name, data in next_text()] == [
            ("second", "a")
        ]
        # The new request's prefill ran separately from the older decode row.
        assert next_text() == []
        assert [(name, data["delta"]) for name, data in next_text()] == [
            ("first", "bcd")
        ]
        session.cancel(["second"])
        assert session.active_count == 1 and runner.released
        remaining = []
        while engine.has_work():
            remaining.extend(tick(session, engine))
        output.extend(remaining)
        assert all(name == "first" for name, _ in remaining)
        assert [(name, data["delta"]) for name, data in text_events(remaining)] == [
            ("first", "ef")
        ]
        assert session.active_count == 0
        assert_scores([event for event in output if event[0] == "first"], recipe.ids)
        assert_scores(
            [event for event in output if event[0] == "second"], recipe.ids[:1]
        )


@pytest.mark.parametrize("api", ["chat/completions", "responses", "messages"])
def test_actual_sse_delivery_batches_text_and_preserves_full_response(api, monkeypatch):
    recipe = Recipe(list("abcdef"))
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "input": "hi",
        "max_tokens": 6,
        "max_output_tokens": 6,
        "logprobs": True,
    }
    if api == "responses":
        body["include"] = ["message.output_text.logprobs"]
    results = []
    for interval in (1, 3):
        monkeypatch.setattr(
            lifecycle, "HttpServer", partial(HttpServer, stream_interval=interval)
        )
        for stream in (False, True):
            runner = Runner([[token] for token in recipe.ids])
            with lifecycle.running_server(runner, recipe) as (
                _,
                engine,
                _,
                errors,
                base,
                _,
            ):
                with request(
                    base, "/v1/" + api, {**body, "stream": stream}
                ) as response:
                    payload = response.read()
                if stream:
                    chunks = events(io.BytesIO(payload))
                    if api == "chat/completions":
                        deltas = [
                            choice["delta"].get("content")
                            for chunk in chunks
                            for choice in chunk["choices"]
                            if choice["delta"].get("content")
                        ]
                    elif api == "responses":
                        deltas = [
                            chunk["delta"]
                            for chunk in chunks
                            if chunk["type"] == "response.output_text.delta"
                        ]
                    else:
                        deltas = [
                            chunk["delta"]["text"]
                            for chunk in chunks
                            if chunk["type"] == "content_block_delta"
                            and chunk["delta"]["type"] == "text_delta"
                        ]
                    assert deltas == (
                        list("abcdef") if interval == 1 else ["a", "bcd", "ef"]
                    )
                if api != "messages":
                    result = collect_scores(
                        io.BytesIO(payload),
                        api=api,
                        stream=stream,
                        partial_tokens=False,
                    )
                    assert result["text"] == "abcdef"
                    results.append(result)
                elif not stream:
                    assert (
                        "".join(part["text"] for part in json.loads(payload)["content"])
                        == "abcdef"
                    )
                assert not engine.poisoned and errors == []
    if results:
        assert all(result == results[0] for result in results)
