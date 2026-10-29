"""Differential grammar fuzzing and arbitrary-byte decoder fuzzing."""

from __future__ import annotations

import json
import os

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st
from tml_renderers import chat, tokenizers, v0
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from recipe_fuzz.deepseek import DeepSeekV41Adapter
from recipe_fuzz.gemma import GemmaAdapter
from recipe_fuzz.gpt_oss import GptOssAdapter
from recipe_fuzz.harness import (
    Case,
    Part,
    assert_chunk_invariance,
    assert_streams,
    normalize,
)
from recipe_fuzz.inputs import native_inputs
from recipe_fuzz.lfm2 import Lfm2Adapter
from recipe_fuzz.qwen import OrnithAdapter, QwenAdapter
from resin.recipe import InklingResponseParser, Tokenizer, get_native_recipe

MARKERS = (
    "message_system",
    "message_user",
    "message_tool",
    "content_xml",
    "message_model",
    "content_text",
    "content_thinking",
    "content_invoke_tool_json",
    "content_invoke_tool_text",
    "content_tool_error",
    "content_model_end_sampling",
    "end_message",
)
LIMIT = int(os.environ.get("RESIN_FUZZ_EXAMPLES", "40"))
FUZZ = settings(max_examples=LIMIT, deadline=None, derandomize=True)
TEXT = st.one_of(
    st.text(min_size=1, max_size=48),
    st.sampled_from(("🦀 café 東京", "\ufffd", "<|end_message|>", "\x00\n\t")),
)
JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(2**53), 2**53)
    | st.floats(allow_nan=False, allow_infinity=False, width=32)
    | st.text(max_size=24),
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(st.text(max_size=8), children, max_size=3)
    ),
    max_leaves=8,
)
TOOLS = st.builds(
    lambda name, args: Part("tool", {"name": name, "args": args}),
    st.sampled_from(("weather", "lookup", "f", "a_b")),
    st.dictionaries(st.text(max_size=8), JSON_VALUES, max_size=3),
)
PARTS = st.one_of(st.builds(Part, st.sampled_from(("text", "reasoning")), TEXT), TOOLS)
CUTS = st.lists(st.integers(0, 512), max_size=12)


class InklingAdapter:
    """Encode the same grammar independently for HF and TML tokenizers."""

    def __init__(self):
        vocab = {char: index for index, char in enumerate(sorted(ByteLevel.alphabet()))}
        self.ordinary = HFTokenizer(BPE(vocab, []))
        self.ordinary.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
        self.ordinary.decoder = ByteLevelDecoder()
        backend = HFTokenizer.from_str(self.ordinary.to_str())
        backend.add_special_tokens([f"<|{name}|>" for name in MARKERS])
        self.ids = {
            f"<|{name}|>": backend.token_to_id(f"<|{name}|>") for name in MARKERS
        }
        assert all(token_id is not None for token_id in self.ids.values())
        self.recipe = get_native_recipe("inkling", backend)
        self.tokenizer = Tokenizer(backend)
        assert self.tokenizer.is_native
        self.reference = tokenizers.O200kBaseChatTokenizer()
        _, self.reference_parser = v0.Renderer(self.reference).render_for_completion(
            chat.OpenAIMessage.from_oss_messages([{"role": "user", "content": "hi"}])
        )
        self.oracles = (
            ("generated-grammar", lambda case: normalize(case.parts)),
            ("tml-v0", self.oracle),
        )

    @staticmethod
    def prompt_case(case):
        messages = [
            {"role": "system", "content": "Helpful."},
            {"role": "user", "content": "hello"},
        ]
        for index, part in enumerate(case.parts):
            if part.kind == "tool":
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"c{index}",
                                "function": {
                                    "name": part.value["name"],
                                    "arguments": part.value["args"],
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {"role": "tool", "tool_call_id": f"c{index}", "content": "ok"}
                )
            elif part.kind == "reasoning":
                messages.append({"role": "assistant", "reasoning_content": part.value})
            else:
                messages.append({"role": "assistant", "content": part.value})
        messages.append({"role": "user", "content": "continue"})
        return {
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}},
                }
            ],
            "add_generation_prompt": True,
        }

    @staticmethod
    def prompt_oracle(request):
        from pathlib import Path

        from transformers.utils.chat_template_utils import _compile_jinja_template

        template = _compile_jinja_template(
            Path(__file__).with_name("templates").joinpath("inkling.jinja").read_text()
        )
        return template.render(**request)

    @staticmethod
    def fragments(case):
        for part in case.parts:
            yield True, "message_model"
            yield (
                True,
                {
                    "text": "content_text",
                    "reasoning": "content_thinking",
                    "tool": "content_invoke_tool_json",
                }[part.kind],
            )
            payload = (
                json.dumps(part.value, ensure_ascii=False)
                if part.kind == "tool"
                else part.value
            )
            yield False, payload
            yield True, "end_message"

    def encode(self, case, *, terminal=True):
        result = []
        for special, text in self.fragments(case):
            if special:
                result.append(self.ids[f"<|{text}|>"])
            else:
                # No added markers on this tokenizer: literal marker text
                # stays ordinary content rather than changing parser state.
                result.extend(self.ordinary.encode(text, add_special_tokens=False).ids)
        if terminal:
            result.append(self.ids["<|content_model_end_sampling|>"])
        return result

    def oracle(self, case):
        ids = []
        for special, text in self.fragments(case):
            ids.extend(
                [self.reference.encode_special(text)]
                if special
                else self.reference.encode_ordinary(text)
            )
        ids.append(self.reference.encode_special("content_model_end_sampling"))
        parts = []
        for message in self.reference_parser.parse_tokens(ids):
            content = message.content
            if isinstance(content, chat.Text):
                parts.append(Part("text", content.text))
            elif isinstance(content, chat.Thinking):
                parts.append(Part("reasoning", content.text))
            elif isinstance(content, chat.InvokeTool):
                parts.append(
                    Part(
                        "tool",
                        {
                            "name": content.name,
                            "args": {
                                arg.name: json.loads(arg.value) for arg in content.args
                            },
                        },
                    )
                )
            else:
                assert isinstance(content, chat.ModelEndSampling)
        return normalize(parts)

    def parser(self):
        return InklingResponseParser(self.tokenizer, self.ids)

    def tool_truncation(self, case):
        prefix = Case(case.parts[:-1])
        ids = self.encode(case, terminal=False)
        payload_start = len(self.encode(prefix, terminal=False)) + 2
        return ids, payload_start, len(ids) - 1

    def text_parser(self):
        return InklingResponseParser(
            self.tokenizer, self.ids, prompt_ids=[self.ids["<|content_text|>"]]
        )


# Each additional format supplies encoding, its real oracle, and parser setup;
# the case generator, semantic normalization, and chunk driver stay shared.
ADAPTERS = {
    "inkling": InklingAdapter,
    "gpt_oss": GptOssAdapter,
    "deepseek_v41": DeepSeekV41Adapter,
    "qwen3_5": QwenAdapter,
    "ornith": OrnithAdapter,
    "lfm2": Lfm2Adapter,
    "gemma4": GemmaAdapter,
}


@pytest.fixture(scope="module", params=ADAPTERS)
def adapter(request):
    return ADAPTERS[request.param]()


@FUZZ
@given(st.lists(PARTS, min_size=1, max_size=5), TOOLS, CUTS, st.data())
def test_reference_semantics_and_truncated_tools(adapter, parts, tool, cuts, data):
    make_case = getattr(adapter, "case", lambda value: value)
    case = make_case(Case(tuple([*parts, tool])))
    # A final tool may be cut at any byte of its JSON, including complete JSON
    # with its closing control token still absent. It must remain uncommitted.
    prefix = make_case(Case(tuple(parts)))
    ids, first, last = adapter.tool_truncation(case)
    cut = data.draw(st.integers(first, last), label="tool truncation")
    for oracle_index, (name, oracle) in enumerate(adapter.oracles):
        note(f"reference oracle: {name}")
        float_ulps = 1 if oracle_index > 0 and name.startswith("tml-") else 0
        assert_streams(
            adapter.parser,
            adapter.encode(case),
            oracle(case),
            cuts,
            float_ulps=float_ulps,
        )
        if getattr(adapter, "commits_tools_at_close", True):
            assert_streams(
                adapter.parser,
                ids[:cut],
                oracle(prefix),
                cuts,
                reason="length",
                float_ulps=float_ulps,
            )
    if not getattr(adapter, "commits_tools_at_close", True):
        assert_chunk_invariance(adapter.parser, ids[:cut], cuts)
    if hasattr(adapter, "prompt_oracle"):
        request = adapter.prompt_case(case)
        assert adapter.recipe.debug_render_text(
            **native_inputs(
                request, tools_before_text=isinstance(adapter, GemmaAdapter)
            )
        ) == adapter.prompt_oracle(request)


@FUZZ
@given(st.lists(st.integers(0, 255), max_size=192), CUTS)
def test_native_decoder_matches_full_reference_for_arbitrary_bytes(adapter, ids, cuts):
    # Every ordinary token represents one byte; generated inputs include
    # malformed UTF-8, incomplete suffixes, and literal replacement characters.
    expected = adapter.ordinary.decode(ids, skip_special_tokens=False)
    expected = getattr(adapter, "normalize_text", lambda value: value)(expected)
    assert_streams(
        adapter.text_parser,
        ids,
        [Part("text", expected)],
        cuts,
        reason="length",
    )
