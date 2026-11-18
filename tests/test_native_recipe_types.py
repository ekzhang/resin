"""Ownership and state boundaries of the native conversation interface."""

from __future__ import annotations

import gc
import json
import urllib.error
import weakref

import numpy as np
import pytest
from output_helpers import forward_output
from test_http_lifecycle import post, running_server
from test_http_stops import send
from test_recipe import _byte_tokenizer

import resin
from resin.recipe import Inkling2Media, NativeRecipe, get_native_recipe


def tokenizer():
    backend = _byte_tokenizer()
    backend.add_special_tokens(
        [
            "<|start|>",
            "<|end|>",
            "<|return|>",
            "<|call|>",
            "<|channel|>",
            "<|message|>",
            "<|constrain|>",
            "<|message_user|>",
            "<|message_system|>",
            "<|message_tool|>",
            "<|content_image|>",
            "<|unused_200054|>",
        ]
    )
    return backend


def test_typed_inputs_own_json_values_and_copy_out_fields():
    parameters = {"type": "object", "properties": {"n": {"type": "integer"}}}
    tool = resin.recipe.Tool("lookup", parameters=parameters)
    call = resin.recipe.ToolCall("lookup", {"n": 3}, id="call-1")
    message = resin.recipe.Message(resin.recipe.Role.ASSISTANT, [call])
    options = resin.recipe.RenderOptions(response_format={"type": "json_object"})
    parameters["properties"]["n"]["type"] = "string"
    tool.parameters["properties"]["n"]["type"] = "boolean"
    call.arguments["n"] = 99
    options.response_format["type"] = "text"
    assert tool.parameters["properties"]["n"]["type"] == "integer"
    saved_call = message.content[0]
    assert isinstance(saved_call, resin.recipe.ToolCall)
    assert saved_call.arguments == {"n": 3}
    assert options.response_format == {"type": "json_object"}
    assert isinstance(message.content, tuple)
    with pytest.raises(AttributeError):
        setattr(message, "role", resin.recipe.Role.USER)
    with pytest.raises(AttributeError):
        setattr(saved_call, "name", "changed")


def test_native_prompt_is_owned_bound_to_one_recipe_and_parsers_are_fresh():
    backend = tokenizer()
    first = get_native_recipe("gpt_oss", backend)
    second = get_native_recipe("gpt_oss", backend)
    prompt = first.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])]
    )
    original = prompt.token_ids
    assert np.shares_memory(prompt.token_ids, original)
    with pytest.raises(ValueError, match="read-only"):
        original[0] = 99
    assert first.stop_token_ids
    with pytest.raises(ValueError, match="this recipe"):
        second.create_parser(prompt)
    with pytest.raises(ValueError, match="this recipe"):
        first.create_parser(resin.recipe.RenderedPrompt(original))
    ids = backend.encode("thought", add_special_tokens=False).ids
    a, b = first.create_parser(prompt), first.create_parser(prompt)
    events = a.feed(ids)
    assert events == b.feed(ids)
    assert events
    assert all(isinstance(event, resin.recipe.ReasoningDelta) for event in events)
    assert all(isinstance(event, resin.recipe.ParserEvent) for event in events)
    assert a is not b
    assert first.parse_response(ids, prompt=prompt)["reasoning_content"] == "thought"


def test_native_events_preserve_variants_values_and_owned_provenance():
    spans = [(0, 0, 2, 0, 2)]
    text = resin.recipe.TextDelta("é", token_spans=spans)
    reasoning = resin.recipe.ReasoningDelta("é", token_spans=spans)
    start = resin.recipe.ToolCallStart(2, "call-2", "lookup", token_spans=spans)
    arguments = resin.recipe.ToolCallArgumentsDelta(2, '{"x":', token_spans=spans)
    spans.clear()
    assert text.type == "text_delta" and text.delta == "é"
    assert reasoning.type == "reasoning_delta" and reasoning.delta == "é"
    assert start.type == "tool_call_start"
    assert (start.index, start.id, start.name) == (2, "call-2", "lookup")
    assert arguments.type == "tool_call_arguments_delta"
    assert (arguments.index, arguments.delta) == (2, '{"x":')
    assert text == resin.recipe.TextDelta("é", token_spans=[(0, 0, 2, 0, 2)])
    assert text != reasoning
    assert text != resin.recipe.TextDelta("é")
    assert text != {"type": "text_delta", "delta": "é"}
    for event in (text, reasoning, start, arguments):
        assert isinstance(event, resin.recipe.ParserEvent)
        assert event.token_spans == ((0, 0, 2, 0, 2),)
        assert isinstance(event.token_spans, tuple)
        with pytest.raises(AttributeError):
            setattr(event, "token_spans", ())
    with pytest.raises(AttributeError):
        setattr(text, "delta", "changed")
    with pytest.raises(AttributeError):
        setattr(start, "index", 0)
    assert resin.recipe.TextDelta("synthetic").token_spans == ()


@pytest.mark.parametrize(
    ("spans", "error"),
    [
        ([(0, 0, 1, 0)], ValueError),
        ([(-1, 0, 1, 0, 1)], OverflowError),
        ([(1 << 32, 0, 1, 0, 1)], OverflowError),
    ],
)
def test_native_event_constructor_validates_span_storage(spans, error):
    with pytest.raises(error):
        resin.recipe.TextDelta("x", token_spans=spans)


def test_parser_does_not_retain_media_or_the_rendered_prompt():
    native = get_native_recipe("inkling2", tokenizer())
    pixels = np.zeros((40, 40, 3), dtype=np.float32)
    source = weakref.ref(pixels)
    messages = [
        resin.recipe.Message(
            resin.recipe.Role.USER, [resin.recipe.Image(source=pixels)]
        )
    ]
    prompt = native.render(messages)
    assert isinstance(prompt.media, Inkling2Media)
    assert prompt.media.pixel_values is not None
    reference = weakref.ref(prompt.media.pixel_values)
    parser = native.create_parser(prompt)
    del pixels, messages, prompt
    gc.collect()
    assert source() is None
    assert reference() is None
    assert parser.feed([]) == []


def test_native_media_accounts_for_each_source():
    native = get_native_recipe("inkling2", tokenizer())
    valid = resin.recipe.Image(source=np.zeros((40, 40, 3), dtype=np.float32))
    prompt = native.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [valid, valid])]
    )
    assert isinstance(prompt.media, Inkling2Media)
    assert prompt.media.pixel_values is not None
    assert prompt.media.pixel_values.shape[0] == 2
    assert len(prompt.media.image_positions) == 2
    assert prompt.media.image_grid_thw is not None
    assert prompt.media.image_grid_thw.tolist() == [[1, 1, 1], [1, 1, 1]]
    with pytest.raises(ValueError, match="dtype float32"):
        native.render(
            [
                resin.recipe.Message(
                    resin.recipe.Role.USER, [valid, resin.recipe.Image(source=object())]
                )
            ]
        )


def test_public_order_and_per_message_controls_are_not_silently_dropped():
    native = get_native_recipe("gpt_oss", tokenizer())
    messages = [
        resin.recipe.Message(
            resin.recipe.Role.ASSISTANT,
            [
                resin.recipe.ToolCall("f", {}),
                resin.recipe.Text("later text"),
            ],
        )
    ]
    with pytest.raises(ValueError, match="block order"):
        native.debug_render_text(messages)
    with pytest.raises(ValueError, match="per-message"):
        native.debug_render_text(
            [
                resin.recipe.Message(
                    resin.recipe.Role.USER, [resin.recipe.Text("hi")], task="action"
                )
            ]
        )
    with pytest.raises(ValueError, match="initial"):
        native.debug_render_text(
            [
                resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")]),
                resin.recipe.Message(
                    resin.recipe.Role.SYSTEM, [], tools=[resin.recipe.Tool("f")]
                ),
            ]
        )
    with pytest.raises(ValueError, match="does not support rendering option"):
        native.debug_render_text(
            [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])],
            options=resin.recipe.RenderOptions(add_vision_id=True),
        )
    with pytest.raises(ValueError, match="latest_reminder"):
        native.debug_render_text(
            [
                resin.recipe.Message(
                    resin.recipe.Role.LATEST_REMINDER, [resin.recipe.Text("Remember")]
                )
            ]
        )


def test_raw_tool_arguments_survive_without_json_reserialization():
    native = get_native_recipe(
        "gpt_oss",
        tokenizer(),
        use_system_prompt=False,
    )
    raw = '{ "n": 1.00, "nested": [true, null] }'
    text = native.debug_render_text(
        [
            resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("run")]),
            resin.recipe.Message(
                resin.recipe.Role.ASSISTANT, [resin.recipe.ToolCall("f", raw)]
            ),
        ]
    )
    assert raw in text


def test_native_http_lowering_matches_public_typed_messages():
    backend = tokenizer()
    backend.add_special_tokens(
        [
            "<bos>",
            "<|turn>",
            "<turn|>",
            "<eos>",
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
            '<|"|>',
        ]
    )
    native = get_native_recipe("gemma4", backend)
    messages = [
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("Go")]),
        resin.recipe.Message(
            resin.recipe.Role.ASSISTANT,
            [
                resin.recipe.Thinking("Use the tool"),
                resin.recipe.ToolCall("f", {"n": 3}, id="c1"),
                resin.recipe.ToolResult({"answer": 4}, name="f"),
                resin.recipe.Text("Here"),
            ],
        ),
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("Next")]),
    ]
    reconstructed = [
        resin.recipe.Message(message.role, message.content) for message in messages
    ]
    expected = native.render(messages).token_ids
    np.testing.assert_array_equal(native.render(reconstructed).token_ids, expected)
    seen = []

    class Runner:
        def forward(self, plan):
            seen.extend(plan.token_ids.tolist())
            return forward_output([backend.token_to_id("<eos>")])

    wire_messages = [
        {"role": "user", "content": "Go"},
        {
            "role": "assistant",
            "content": "Here",
            "reasoning_content": "Use the tool",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": {"n": 3}},
                }
            ],
            "tool_responses": [{"name": "f", "response": {"answer": 4}}],
        },
        {"role": "user", "content": "Next"},
    ]
    with running_server(Runner(), native) as (_, _, _, errors, base, _):
        with post(
            base,
            json.dumps(
                {"model": "test", "messages": wire_messages, "max_tokens": 1}
            ).encode(),
        ) as response:
            assert response.status == 200
        assert not errors
    np.testing.assert_array_equal(seen, expected)


def test_rendered_prompt_token_view_is_owned_readonly_and_outlives_prompt():
    source = np.asarray([11, 22, 33], dtype=np.uint32)
    prompt = resin.recipe.RenderedPrompt(source)
    source[0] = 99
    tokens = prompt.token_ids
    assert tokens.dtype == np.uint32 and tokens.ndim == 1
    assert tokens.flags.c_contiguous and not tokens.flags.writeable
    assert tokens.base is prompt
    assert np.shares_memory(tokens, prompt.token_ids)
    np.testing.assert_array_equal(tokens, [11, 22, 33])
    with pytest.raises(ValueError, match="read-only"):
        tokens[0] = 7
    with pytest.raises(ValueError, match="WRITEABLE"):
        tokens.flags.writeable = True
    del prompt
    gc.collect()
    np.testing.assert_array_equal(tokens, [11, 22, 33])
    empty = resin.recipe.RenderedPrompt([]).token_ids
    assert empty.dtype == np.uint32 and empty.shape == (0,)
    assert not empty.flags.writeable


@pytest.mark.parametrize(
    ("recipe_name", "default_markers"),
    [
        ("gpt_oss", ["<|return|>", "<|call|>"]),
        ("deepseek_v41", ["<｜end▁of▁sentence｜>"]),
        ("inkling2", ["<|content_model_end_sampling|>"]),
        ("qwen3_5", ["<|im_end|>", "<|endoftext|>"]),
        ("qwen3_8", ["<|im_end|>", "<|endoftext|>"]),
        ("gemma4", ["<turn|>", "<eos>"]),
        ("lfm2", ["<|im_end|>"]),
    ],
)
def test_native_recipe_stop_defaults_and_overrides_share_one_contract(
    recipe_name, default_markers
):
    backend = tokenizer()
    backend.add_special_tokens(default_markers)
    expected = [backend.token_to_id(marker) for marker in default_markers]
    default = get_native_recipe(recipe_name, backend)
    assert type(default) is NativeRecipe and default.tokenizer.is_native
    assert isinstance(default.stop_token_ids, list)
    assert default.stop_token_ids == expected
    for override, wanted in [(None, expected), ([], [])]:
        assert (
            get_native_recipe(
                recipe_name, backend, stop_token_ids=override
            ).stop_token_ids
            == wanted
        )

    source = [901, 902]
    recipe = get_native_recipe(recipe_name, backend, stop_token_ids=source)
    source[0] = 999
    stops = recipe.stop_token_ids
    assert isinstance(stops, list)
    assert stops == [901, 902]
    stops[0] = 0
    stops.append(903)
    assert recipe.stop_token_ids == [901, 902]


def test_http_chat_template_kwargs_apply_and_respect_explicit_overrides():
    backend = tokenizer()
    backend.add_special_tokens(
        [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<|vision_start|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_end|>",
        ]
    )
    native = get_native_recipe("qwen3_5", backend, text_only=True)
    messages = [
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("First")]),
        resin.recipe.Message(
            resin.recipe.Role.ASSISTANT,
            [resin.recipe.Thinking("prior"), resin.recipe.Text("Done")],
        ),
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("Second")]),
    ]
    default_ids = native.render(messages).token_ids
    kwargs_ids = native.render(
        messages,
        options=resin.recipe.RenderOptions(
            enable_thinking=False, preserve_thinking=True
        ),
    ).token_ids
    override_ids = native.render(
        messages,
        options=resin.recipe.RenderOptions(
            enable_thinking=True, preserve_thinking=True
        ),
    ).token_ids
    assert not np.array_equal(kwargs_ids, default_ids)
    assert not np.array_equal(kwargs_ids, override_ids)

    seen: list[list[int]] = []
    eos = backend.token_to_id("<|im_end|>")

    class Runner:
        def forward(self, plan):
            seen.append(plan.token_ids.tolist())
            return forward_output([eos])

    wire_messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "reasoning_content": "prior", "content": "Done"},
        {"role": "user", "content": "Second"},
    ]
    with running_server(Runner(), native) as (_, _, _, errors, base, _):
        with post(
            base,
            json.dumps(
                {
                    "model": "test",
                    "messages": wire_messages,
                    "max_tokens": 1,
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                        "preserve_thinking": True,
                    },
                }
            ).encode(),
        ) as response:
            assert response.status == 200
        with post(
            base,
            json.dumps(
                {
                    "model": "test",
                    "messages": wire_messages,
                    "max_tokens": 1,
                    "enable_thinking": True,
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                        "preserve_thinking": True,
                    },
                }
            ).encode(),
        ) as response:
            assert response.status == 200
        with send(
            base,
            "/v1/responses/input_tokens",
            {
                "model": "test",
                "input": wire_messages,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": True,
                },
            },
        ) as response:
            assert json.load(response)["input_tokens"] == len(kwargs_ids)
        for bad_kwargs, expected_error in [
            ("bad", "chat_template_kwargs must be an object"),
            ({"unknown_option": True}, "unknown field"),
        ]:
            with pytest.raises(urllib.error.HTTPError) as failure:
                send(
                    base,
                    "/v1/responses/input_tokens",
                    {
                        "model": "test",
                        "input": wire_messages,
                        "chat_template_kwargs": bad_kwargs,
                    },
                )
            assert failure.value.code == 500
            assert expected_error in json.load(failure.value)["error"]["message"]
        assert not errors

    np.testing.assert_array_equal(seen[0], kwargs_ids)
    np.testing.assert_array_equal(seen[1], override_ids)
