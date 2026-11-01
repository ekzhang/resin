from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from tokenizers import Tokenizer as HuggingFaceTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

import resin
from resin.protocols import FinalizingResponseParser
from resin.recipe import (
    ParserEvent,
    Tokenizer,
    get_native_recipe,
)

MARKERS = (
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|content_tool_error|>",
    "<|content_model_end_sampling|>",
    "<|end_message|>",
    "<|message_system|>",
    "<|message_user|>",
    "<|message_tool|>",
)


def _byte_tokenizer() -> HuggingFaceTokenizer:
    vocab = {
        character: index for index, character in enumerate(sorted(ByteLevel.alphabet()))
    }
    tokenizer = HuggingFaceTokenizer(BPE(vocab, []))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens(list(MARKERS))
    return tokenizer


class _FastTokenizer:
    """Expose the same native tokenizer entry point as Transformers."""

    def __init__(self, backend_tokenizer: HuggingFaceTokenizer) -> None:
        self.backend_tokenizer = backend_tokenizer

    def decode(self, *_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("fast tokenizers must decode in Rust")


def _token_ids(tokenizer: HuggingFaceTokenizer) -> dict[str, int]:
    result = {}
    for marker in MARKERS:
        token_id = tokenizer.token_to_id(marker)
        assert token_id is not None
        result[marker] = token_id
    return result


def _inkling_parser(tokenizer, *, content: bool = False) -> FinalizingResponseParser:
    recipe = get_native_recipe("inkling", tokenizer)
    parser = recipe.create_parser(recipe.render([]))
    if content:
        assert parser.feed(recipe.tokenizer.encode("<|content_text|>")) == []
    return parser


def _coalesce(events: list[ParserEvent]) -> list[ParserEvent]:
    """Ignore chunk boundaries and provenance for this semantic comparison."""
    result: list[ParserEvent] = []
    for event in events:
        if isinstance(event, (resin.recipe.TextDelta, resin.recipe.ReasoningDelta)):
            previous = result[-1] if result else None
            if isinstance(
                previous, (resin.recipe.TextDelta, resin.recipe.ReasoningDelta)
            ) and type(previous) is type(event):
                result[-1] = type(event)(previous.delta + event.delta)
            else:
                result.append(type(event)(event.delta))
        elif isinstance(event, resin.recipe.ToolCallStart):
            result.append(resin.recipe.ToolCallStart(event.index, event.id, event.name))
        elif isinstance(event, resin.recipe.ToolCallArgumentsDelta):
            result.append(resin.recipe.ToolCallArgumentsDelta(event.index, event.delta))
        else:
            raise AssertionError(f"unexpected parser event: {event}")
    return result


def test_complete_response_uses_actual_generation_prefix() -> None:
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
        ]
    )
    model_recipe = get_native_recipe("gpt_oss", backend, use_system_prompt=False)
    messages = [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])]
    prompt = model_recipe.render(messages)
    output = backend.encode(
        "think<|end|><|start|>assistant<|channel|>final<|message|>answer<|return|>",
        add_special_tokens=False,
    ).ids
    expected = {"content": "answer", "reasoning_content": "think", "tool_calls": []}
    assert model_recipe.parse_response(output, prompt=prompt) == expected
    with pytest.raises(ValueError, match="prompt"):
        model_recipe.parse_response(
            output, prompt=resin.recipe.RenderedPrompt(prompt.token_ids)
        )
    # Parsing uses the actual prompt, without re-rendering a request.
    assert model_recipe.create_parser(prompt) is not model_recipe.create_parser(prompt)


def test_native_decode_is_equivalent_for_every_chunk_boundary() -> None:
    backend = _byte_tokenizer()
    tokenizer = Tokenizer(_FastTokenizer(backend))
    text = (
        "<|content_thinking|>想一想 🧠"
        "<|end_message|><|message_model|><|content_text|>Hi café! \ufffd"
        "<|end_message|><|message_model|><|content_invoke_tool_json|>"
        '{"name":"weather","args":{"city":"東京"}}'
        "<|end_message|><|content_model_end_sampling|>"
    )
    ids = backend.encode(text, add_special_tokens=False).ids

    def parse(chunks: Sequence[Sequence[int]]) -> list[ParserEvent]:
        parser = _inkling_parser(tokenizer)
        events: list[ParserEvent] = []
        for chunk in chunks:
            events.extend(parser.feed(chunk))
        events.extend(parser.finish("stop"))
        return _coalesce(events)

    expected: list[ParserEvent] = [
        resin.recipe.ReasoningDelta("想一想 🧠"),
        resin.recipe.TextDelta("Hi café! \ufffd"),
        resin.recipe.ToolCallStart(0, "call_0", "weather"),
        resin.recipe.ToolCallArgumentsDelta(0, '{"city":"東京"}'),
    ]
    assert parse([ids]) == expected
    assert parse([[token_id] for token_id in ids]) == expected
    for split in range(len(ids) + 1):
        assert parse([ids[:split], ids[split:]]) == expected


def test_native_decoder_buffers_partial_utf8_between_feeds() -> None:
    backend = _byte_tokenizer()
    ids = backend.encode("🦀", add_special_tokens=False).ids
    assert len(ids) == 4
    markers = _token_ids(backend)
    parser = _inkling_parser(Tokenizer(_FastTokenizer(backend)))

    assert parser.feed([markers["<|content_text|>"]]) == []
    for token_id in ids[:-1]:
        assert parser.feed([token_id]) == []
    assert parser.feed(ids[-1:]) == [
        resin.recipe.TextDelta(
            "🦀", token_spans=[(index, 0, 1, 0, 4) for index in range(1, 5)]
        )
    ]
    assert not hasattr(parser, "feed_attributed")
    assert not hasattr(parser, "finish_attributed")
    assert parser.finish("length") == []


def test_native_decoder_flushes_incomplete_utf8_at_finish() -> None:
    backend = _byte_tokenizer()
    parser = _inkling_parser(Tokenizer(_FastTokenizer(backend)), content=True)
    incomplete = backend.encode("🦀", add_special_tokens=False).ids[:2]

    assert parser.feed(incomplete) == []
    assert parser.finish("length") == [
        resin.recipe.TextDelta(
            "�", token_spans=[(index, 0, 1, 0, 3) for index in range(1, 3)]
        )
    ]
    assert parser.finish("length") == []


def test_native_tokenizer_is_a_snapshot_and_parsers_have_independent_state() -> None:
    backend = _byte_tokenizer()
    tokenizer = Tokenizer(_FastTokenizer(backend))
    recipe = get_native_recipe("inkling", tokenizer)
    prompt = recipe.render([])
    backend.decoder = None
    parsers = [recipe.create_parser(prompt) for _ in range(2)]
    for parser in parsers:
        assert parser.feed(tokenizer.encode("<|content_text|>")) == []
    ids = backend.encode("café", add_special_tokens=False).ids

    def expected(offset: int) -> list[ParserEvent]:
        return [
            resin.recipe.TextDelta(
                "café",
                token_spans=[
                    (offset, 0, 1, 0, 1),
                    (offset + 1, 0, 1, 1, 2),
                    (offset + 2, 0, 1, 2, 3),
                    (offset + 3, 0, 1, 3, 5),
                    (offset + 4, 0, 1, 3, 5),
                ],
            )
        ]

    assert parsers[0].feed(ids) == expected(1)
    assert parsers[1].feed(ids) == expected(1)
    assert parsers[0].finish("stop") == []
    assert parsers[1].feed(ids) == expected(6)


def test_native_parser_rejects_an_unsupported_decoder_chain() -> None:
    from tokenizers.decoders import WordPiece

    backend = _byte_tokenizer()
    backend.decoder = WordPiece()
    with pytest.raises(ValueError, match="decoder"):
        _inkling_parser(Tokenizer(backend))


def test_custom_tokenizer_with_no_native_backend_uses_decode() -> None:
    calls: list[list[int]] = []

    class CustomTokenizer(_CallbackTokenizer):
        backend_tokenizer = None

        def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
            assert not skip_special_tokens
            calls.append(list(ids))
            return "".join(chr(token_id) for token_id in ids)

    tokenizer = Tokenizer(CustomTokenizer())
    assert not tokenizer.is_native
    parser = _inkling_parser(tokenizer, content=True)

    assert parser.feed([ord("x")]) == [resin.recipe.TextDelta("x")]
    assert calls == [[ord("x")]]
    assert parser.finish("stop") == []


class _CallbackTokenizer:
    """A tokenizer with explicit ordinary encoding and reversible markers."""

    markers = {marker: index + 1000 for index, marker in enumerate(MARKERS)}

    def __init__(self):
        self.raw_encodes = []

    def encode(self, text, *, add_special_tokens=False):
        self.raw_encodes.append(text)
        return [self.markers[text]] if text in self.markers else list(text.encode())

    def encode_ordinary(self, text):
        return list(text.encode())

    def convert_tokens_to_ids(self, token):
        return self.markers.get(token, 0)

    def convert_ids_to_tokens(self, token_id):
        return next(
            (token for token, value in self.markers.items() if value == token_id),
            "<unk>",
        )

    def decode(self, ids, *, skip_special_tokens=False):
        return bytes(ids).decode(errors="replace")


def test_callback_tokenizer_keeps_ordinary_encoding_and_serving_stops_separate():
    tokenizer = _CallbackTokenizer()
    model_recipe = get_native_recipe("inkling", tokenizer)
    marker = "<|content_model_end_sampling|>"
    messages = [
        resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text(marker)])
    ]
    rendered = model_recipe.render(messages)
    assert tokenizer.markers[marker] not in rendered.token_ids
    assert model_recipe.stop_token_ids == [tokenizer.markers[marker]]
    assert tokenizer.raw_encodes == []


@pytest.mark.parametrize("invalid_id", [0, True])
def test_callback_tokenizer_rejects_unknown_or_boolean_control_ids(
    invalid_id, monkeypatch
):
    tokenizer = _CallbackTokenizer()
    monkeypatch.setattr(tokenizer, "convert_tokens_to_ids", lambda _: invalid_id)
    model_recipe = get_native_recipe("inkling", tokenizer)
    with pytest.raises(ValueError, match="control token"):
        model_recipe.stop_token_ids


def test_callback_decoder_is_captured_once_instead_of_refetched():
    class TokenizerWithProperty(_CallbackTokenizer):
        reads = 0

        @property
        def decode(self):
            self.reads += 1
            if self.reads > 1:
                raise ValueError("decode accessor cannot be reread")
            return super().decode

    tokenizer = TokenizerWithProperty()
    model_recipe = get_native_recipe("inkling", tokenizer)
    parser = model_recipe.create_parser(
        model_recipe.render(
            [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])]
        )
    )
    assert parser.feed([tokenizer.markers["<|content_text|>"]]) == []
    assert parser.feed([ord("x")]) == [resin.recipe.TextDelta("x")]
    assert parser.finish("stop") == []
    assert tokenizer.reads == 1


def test_deepseek_constructor_effort_matches_per_request_effort():
    backend = _byte_tokenizer()
    messages = [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])]
    configured = get_native_recipe(
        "deepseek_v41",
        backend,
        reasoning_effort="high",
    )
    per_request = get_native_recipe("deepseek_v41", backend)
    expected = per_request.debug_render_text(
        messages, options=resin.recipe.RenderOptions(reasoning_effort="high")
    )
    assert configured.debug_render_text(messages) == expected
    assert (
        configured.debug_render_text(messages, options=resin.recipe.RenderOptions())
        == expected
    )
    assert configured.debug_render_text(
        messages, options=resin.recipe.RenderOptions(reasoning_effort="low")
    ) == per_request.debug_render_text(
        messages, options=resin.recipe.RenderOptions(reasoning_effort="low")
    )


def test_inkling_tool_float_preserves_original_value() -> None:
    import json

    backend = _byte_tokenizer()
    parser = _inkling_parser(Tokenizer(backend))
    # TML 0.2.0 rounds this to 0.9999899864196776; the generated JSON and
    # native parser must retain the exact Python value.
    ids = backend.encode(
        '<|content_invoke_tool_json|>{"name":"weather","args":{"":0.9999899864196777}}<|end_message|>'
    ).ids
    events = parser.feed(ids)
    assert isinstance(events[-1], resin.recipe.ToolCallArgumentsDelta)
    assert json.loads(events[-1].delta) == {"": 0.9999899864196777}


def test_callback_recipe_stop_ids_are_static_after_first_resolution():
    tokenizer = _CallbackTokenizer()
    tokenizer.markers = dict(tokenizer.markers)
    marker = "<|content_model_end_sampling|>"
    expected = tokenizer.markers[marker]
    recipe = get_native_recipe("inkling", tokenizer)
    assert recipe.stop_token_ids == [expected]
    tokenizer.markers[marker] = expected + 1
    assert recipe.stop_token_ids == [expected]
