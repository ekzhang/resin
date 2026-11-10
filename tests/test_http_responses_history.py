"""Feed emitted Responses items back through the native recipe and HTTP actor."""

import json

import pytest
from output_helpers import forward_output
from recipe_fuzz.gpt_oss import GptOssAdapter
from recipe_fuzz.harness import Case, Part
from test_http_lifecycle import running_server
from test_http_stops import send


@pytest.mark.parametrize("stream", [False, True])
def test_responses_tool_history_round_trip_and_token_count(stream):
    adapter = GptOssAdapter()

    class Runner:
        output = iter(())

        def forward(self, plan):
            return forward_output([next(self.output)])

    runner = Runner()
    runner.output = iter(
        adapter.encode(
            Case(
                (
                    Part("reasoning", "check weather"),
                    Part("tool", {"name": "weather", "args": {"city": "Paris"}}),
                )
            )
        )
    )
    with running_server(runner, adapter.recipe) as (_, engine, _, errors, base, _):
        history = [{"role": "user", "content": "weather?"}]
        with send(
            base,
            "/v1/responses",
            {
                "model": "test",
                "input": history,
                "max_output_tokens": 256,
                "stream": stream,
            },
        ) as response:
            if stream:
                events = [
                    json.loads(line[6:])
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: ")
                ]
                generated = next(
                    event["response"]
                    for event in events
                    if event["type"] == "response.completed"
                )
            else:
                generated = json.load(response)
        history.extend(generated["output"])
        call = next(
            item for item in generated["output"] if item["type"] == "function_call"
        )
        history.append(
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": "sunny",
            }
        )
        with send(
            base, "/v1/responses/input_tokens", {"model": "test", "input": history}
        ) as response:
            counted = json.load(response)["input_tokens"]
        runner.output = iter(adapter.encode(Case((Part("text", "It is sunny."),))))
        with send(
            base,
            "/v1/responses",
            {"model": "test", "input": history, "max_output_tokens": 256},
        ) as response:
            continued = json.load(response)
        assert continued["output"][0]["content"][0]["text"] == "It is sunny."
        assert continued["usage"]["input_tokens"] == counted
        assert not engine.poisoned and errors == []
