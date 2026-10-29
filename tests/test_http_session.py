"""The routed session retains the ordinary actor's bounded-stream lifecycle."""

from collections.abc import Sequence

import pytest
from http_packets import decoded_events, ingress
from output_helpers import forward_output

import resin
from resin import BatchPhase, Engine
from resin._resin import _HttpSession


@pytest.mark.parametrize("stream_interval", [1, 4])
def test_session_overflow_terminates_client_and_keeps_other_requests_usable(
    stream_interval,
):
    class Runner:
        runner_config = resin.RunnerConfig(max_accepted_per_step=40)

        def forward(self, plan):
            return forward_output(
                [1] if plan.phase == BatchPhase.PREFILL else [[2] * 40]
            )

    class Recipe:
        tokenizer = None
        stop_token_ids = []

        def render(self, messages, *, tools=(), options=None):
            return resin.recipe.RenderedPrompt([0])

        def create_parser(self, prompt):
            class Parser:
                def feed(self, ids):
                    return [
                        resin.recipe.ReasoningDelta("a"),
                        resin.recipe.TextDelta("b"),
                    ]

            return Parser()

    def admission(request_id, limit):
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
                    "logprobs": False,
                    "top_logprobs": 0,
                    "prompt_logprobs": False,
                    "routed_dp_rank": None,
                    "stop": [],
                    "reasoning_effort": None,
                    "extra": {},
                }
            ]
        )

    engine = Engine(Runner())
    session = _HttpSession(engine, Recipe(), stream_interval=stream_interval)
    session.admit(admission("overflow", 100))
    first = decoded_events(session.process(engine.step()))
    assert [event["type"] for _, event in first] == [
        "started",
        "reasoning_delta",
        "text_delta",
    ]
    # Forty verified tokens produce eighty noncoalescible semantic events.
    overflow = decoded_events(session.process(engine.step()))
    assert len(overflow) == 65  # 64 buffered events plus the terminal error.
    assert all(request_id == "overflow" for request_id, _ in overflow)
    assert overflow[-1][1] == {
        "type": "error",
        "data": {"message": "the inference stream ended before completion"},
    }
    assert session.active_count == 0
    assert not engine.has_work() and not engine.closed and not engine.poisoned
    assert session.process([]).to_bytes() == b""

    session.admit(admission("ordinary", 1))
    ordinary = decoded_events(session.process(engine.step()))
    assert [event["type"] for _, event in ordinary] == [
        "started",
        "reasoning_delta",
        "text_delta",
        "finished",
    ]
    assert all(request_id == "ordinary" for request_id, _ in ordinary)
    assert ordinary[-1][1]["data"]["reason"] == "length"
    assert session.active_count == 0 and session.process([]).to_bytes() == b""
    session.close()
    assert not engine.closed
    engine.close()


def test_http_admission_passes_typed_messages_tools_and_render_controls():
    seen = []
    expected_prompt = resin.recipe.RenderedPrompt([1, 2])

    class Recipe:
        tokenizer = None
        stop_token_ids = []

        def render(
            self,
            messages: Sequence[resin.recipe.Message],
            *,
            tools: Sequence[resin.recipe.Tool] = (),
            options: resin.recipe.RenderOptions | None = None,
        ):
            assert options is not None
            assert isinstance(messages[0].content[0], resin.recipe.Text)
            assert isinstance(messages[1].content[0], resin.recipe.Thinking)
            assert messages[0].role == resin.recipe.Role.SYSTEM
            assert messages[0].content[0].text == "Instructions"
            assert messages[0].tools[0].name == "local"
            assert messages[1].content[0].text == ""
            call = messages[1].content[1]
            assert isinstance(call, resin.recipe.ToolCall)
            assert call.arguments == '{ "x": 1 }'
            assert tools[0].name == "lookup"
            assert tools[0].parameters == {"type": "object"}
            assert options.reasoning_effort == "high"
            assert options.enable_thinking is False
            assert not hasattr(options, "temperature")
            seen.append(messages)
            return expected_prompt

        def create_parser(self, prompt):
            assert prompt is expected_prompt

            class Parser:
                def feed(self, ids):
                    return []

            return Parser()

    class Runner:
        def forward(self, plan):
            return forward_output([3])

    engine = Engine(Runner())
    session = _HttpSession(engine, Recipe())
    session.admit(
        ingress(
            [
                {
                    "id": "typed",
                    "model": "test",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Instructions",
                            "tools": [{"name": "local"}],
                        },
                        {
                            "role": "assistant",
                            "reasoning_content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{ "x": 1 }',
                                    },
                                }
                            ],
                        },
                        {"role": "user", "content": "Continue"},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "max_output_tokens": 1,
                    "temperature": 0.2,
                    "top_p": 1.0,
                    "top_k": None,
                    "seed": None,
                    "logprobs": False,
                    "top_logprobs": 0,
                    "prompt_logprobs": False,
                    "routed_dp_rank": None,
                    "stop": [],
                    "reasoning_effort": "high",
                    "extra": {"reasoning_effort": "low", "enable_thinking": False},
                }
            ]
        )
    )
    events = decoded_events(session.process(engine.step()))
    assert not [event for _, event in events if event["type"] == "error"]
    assert len(seen) == 1
    session.close()
    engine.close()


def test_http_session_snapshots_recipe_stops_before_rendering():
    class Recipe:
        tokenizer = None

        def __init__(self):
            self.stops = [3]
            self.stop_reads = 0

        @property
        def stop_token_ids(self):
            self.stop_reads += 1
            return self.stops

        def render(self, messages, *, tools=(), options=None):
            return resin.recipe.RenderedPrompt([1])

        def create_parser(self, prompt):
            class Parser:
                def feed(self, ids):
                    return [resin.recipe.TextDelta("".join(map(str, ids)))]

            return Parser()

    class Runner:
        def forward(self, plan):
            return forward_output([3] * plan.num_sequences)

    recipe = Recipe()
    engine = Engine(Runner())
    session = _HttpSession(engine, recipe)
    assert recipe.stop_reads == 1
    recipe.stops[:] = [4]
    recipe.stops = [5]
    try:
        for request_id in ("first", "second"):
            session.admit(
                ingress(
                    [
                        {
                            "id": request_id,
                            "model": "test",
                            "messages": [],
                            "tools": [],
                            "max_output_tokens": 3,
                            "temperature": 0,
                            "top_p": 1,
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
                    ]
                )
            )
            output = decoded_events(session.process(engine.step()))
            finished = [event for _, event in output if event["type"] == "finished"]
            assert len(finished) == 1
            assert finished[0]["data"]["reason"] == "stop"
            assert finished[0]["data"]["output_tokens"] == 1
            assert session.active_count == 0 and not engine.has_work()
        assert recipe.stop_reads == 1
    finally:
        session.close()
        engine.close()
