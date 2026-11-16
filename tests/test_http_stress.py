"""The live load client must detect broken streams, not merely HTTP 200s."""

import asyncio
import json

import httpx
import pytest

from scripts.stress_http import (
    Case,
    Completion,
    chat_case,
    robustness_cases,
    run_case,
    run_stage,
)


def payload(*, stream=False, score_index=0, finish="length"):
    return {
        "id": "completion-id",
        "choices": [
            {
                "index": 0,
                "delta" if stream else "message": {"content": "hi"},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        "resin": {
            "generation_logprobs": [
                {
                    "token_index": score_index,
                    "token_id": 10,
                    "logprob": -0.25,
                    "top_logprobs": [{"token_id": 10, "logprob": -0.25}],
                }
            ]
        },
    }


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, data, *, stall=False):
        self.data, self.stall, self.closed = data, stall, False

    async def __aiter__(self):
        # Chunk boundaries deliberately bisect lines and JSON.
        for index in range(0, len(self.data), 7):
            yield self.data[index : index + 7]
        if self.stall:
            await asyncio.sleep(60)

    async def aclose(self):
        self.closed = True


def perform(response, *, stream=False, disconnect=False, timeout=1):
    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/",
            transport=httpx.MockTransport(lambda _: response),
        ) as client:
            return await run_case(
                client,
                Case(
                    name="test",
                    body={
                        "stream": stream,
                        "max_tokens": 4,
                        "logprobs": True,
                        "top_logprobs": 1,
                    },
                    disconnect=disconnect,
                ),
                timeout,
            )

    return asyncio.run(run())


def test_json_scores_and_usage():
    result = perform(httpx.Response(200, json=payload()))
    assert result["ok"]
    assert result["completion_tokens"] == result["raw_logprobs"] == 1
    assert result["ttft_ms"] is None


@pytest.mark.parametrize(
    "change,match",
    [
        ({"score_index": 1}, "noncontiguous"),
        ({"finish": None}, "finish reason"),
    ],
)
def test_bad_completion_fails_even_when_status_is_200(change, match):
    result = perform(httpx.Response(200, json=payload(**change)))
    assert not result["ok"]
    assert match in result["error"]


def test_chunked_multiline_sse_with_comments():
    data = (
        b": keepalive\r\n\r\n"
        + b"data: "
        + json.dumps(payload(stream=True), indent=2)
        .encode()
        .replace(b"\n", b"\ndata: ")
        + b"\r\n\r\ndata: [DONE]\r\n\r\n"
    )
    stream = ByteStream(data)
    result = perform(
        httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        ),
        stream=True,
    )
    assert result["ok"]
    assert result["ttft_ms"] is not None
    assert stream.closed


@pytest.mark.parametrize(
    "data,match",
    [
        (b'data: {"error":{"message":"GPU failed"}}\n\ndata: [DONE]\n\n', "GPU failed"),
        (b"data: [DONE]\n\n", "finish reason"),
        (b'data: {"id":', "inside an event"),
        (b"", "without [DONE]"),
    ],
)
def test_invalid_sse_fails(data, match):
    result = perform(
        httpx.Response(
            200, content=data, headers={"content-type": "text/event-stream"}
        ),
        stream=True,
    )
    assert not result["ok"]
    assert match in result["error"]


def test_disconnect_closes_stream_without_waiting_for_terminal_event():
    data = b"data: " + json.dumps(payload(stream=True, finish=None)).encode() + b"\n\n"
    stream = ByteStream(data, stall=True)
    result = perform(
        httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        ),
        stream=True,
        disconnect=True,
    )
    assert result["ok"] and result["disconnected"]
    assert stream.closed
    assert "completion_tokens" not in result  # Partial output is not throughput.


def test_timeout_closes_response():
    stream = ByteStream(b": started\n\n", stall=True)
    result = perform(
        httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        ),
        stream=True,
        timeout=0.01,
    )
    assert not result["ok"]
    assert "TimeoutError" in result["error"]
    assert stream.closed


def test_changed_response_id_is_rejected():
    tracker = Completion(scored=True, top_k=1)
    tracker.add(payload(stream=True), stream=True)
    with pytest.raises(ValueError, match="ID changed"):
        tracker.add({**payload(stream=True), "id": "other"}, stream=True)


def test_routes_cover_every_rank_in_both_forms():
    cases = [chat_case("model", i, 4, 16) for i in range(24)]
    assert {
        c.body["routed_dp_rank"] for c in cases if "routed_dp_rank" in c.body
    } == set(range(4))
    assert {int(c.headers["resin-routed-dp-rank"]) for c in cases if c.headers} == set(
        range(4)
    )
    disconnects = [c for c in robustness_cases("model", 4) if c.disconnect]
    assert {c.body["routed_dp_rank"] for c in disconnects} == set(range(4))


@pytest.mark.parametrize("stream", [False, True])
def test_client_against_real_resin_http_listener(stream):
    from test_http_lifecycle import Recipe, Runner, running_server

    async def run(base):
        async with httpx.AsyncClient(base_url=base + "/v1/") as client:
            return await run_case(
                client,
                Case(
                    name="live",
                    body={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 2,
                        "stream": stream,
                        "stream_options": {"include_usage": True},
                    },
                ),
                5,
            )

    with running_server(Runner(), Recipe()) as (_, _, _, errors, base, _):
        result = asyncio.run(run(base))
        assert result["ok"], result
        assert result["completion_tokens"] == 2
        assert errors == []


def test_invalid_top_logprobs_case_is_rejected_before_generation():
    from test_http_lifecycle import Recipe, Runner, running_server

    case = next(
        case for case in robustness_cases("test", 1) if case.name == "bad-top-logprobs"
    )

    async def run(base):
        async with httpx.AsyncClient(base_url=base + "/v1/") as client:
            return await run_case(client, case, 5)

    runner = Runner()
    with running_server(runner, Recipe()) as (_, _, _, errors, base, _):
        result = asyncio.run(run(base))
        assert result["ok"] and result["status"] in (400, 422), result
        assert not runner.counts and not errors


def test_each_routed_rank_gets_both_protocols_scoring_modes_and_prompt_lengths():
    cases = [chat_case("model", i, 4, 16) for i in range(192)]
    for rank in range(4):
        routed = [
            c
            for c in cases
            if c.body.get("routed_dp_rank") == rank
            or c.headers.get("resin-routed-dp-rank") == str(rank)
        ]
        assert {c.body["stream"] for c in routed} == {False, True}
        assert {c.body.get("logprobs", False) for c in routed} == {False, True}
        assert (
            len({c.body["messages"][0]["content"].count("observatory") for c in routed})
            == 4
        )
    assert all(c.body["stream"] for c in robustness_cases("model", 4) if c.disconnect)


def test_modal_result_redirect_preserves_completion_validation():
    methods = []

    def handle(request):
        methods.append(request.method)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(303, headers={"location": "/result/123"})
        return httpx.Response(200, json=payload())

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            return await run_case(
                client,
                Case(
                    name="redirect",
                    body={"max_tokens": 4, "logprobs": True, "top_logprobs": 1},
                ),
                1,
            )

    result = asyncio.run(run())
    assert result["ok"]
    assert methods == ["POST", "GET"]


def test_incremental_journal_survives_interrupted_stage(tmp_path):
    journal = tmp_path / "requests.jsonl"

    async def run():
        async def handle(_request):
            await asyncio.sleep(60)
            return httpx.Response(200, json=payload())

        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            task = asyncio.create_task(
                run_stage(
                    client,
                    "pending",
                    [Case(name="pending", body={"max_tokens": 1})],
                    1,
                    60,
                    event_path=journal,
                )
            )
            while not journal.exists():
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "start" and rows[0]["case"] == "pending"
    assert rows[0]["body"] == {"max_tokens": 1}


def test_incremental_journal_records_failures_before_stage_finishes(tmp_path):
    journal = tmp_path / "requests.jsonl"

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test/v1/",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(500, text="worker failed")
            ),
        ) as client:
            return await run_stage(
                client,
                "failure",
                [Case(name="failed", body={"max_tokens": 1})],
                1,
                1,
                event_path=journal,
            )

    result = asyncio.run(run())
    assert result["passed"] == 0
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "finish"]
    assert rows[-1]["status"] == 500 and not rows[-1]["ok"]
