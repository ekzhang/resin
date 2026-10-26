"""Qwen grammar projection, TML parser oracle, and pinned HF templates."""

from __future__ import annotations

import json
from pathlib import Path

from tml_renderers import chat, renderers
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers.utils.chat_template_utils import _compile_jinja_template

import resin
from recipe_fuzz.harness import Case, Part, normalize
from resin.recipe import get_native_recipe

MARKERS = (
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
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


class QwenAdapter:
    # Native recipe and pinned prompt template exercised by this adapter.
    variant = "qwen3_8"

    def __init__(self):
        vocab = {char: index for index, char in enumerate(sorted(ByteLevel.alphabet()))}
        self.ordinary = HFTokenizer(BPE(vocab, []))
        self.ordinary.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
        self.ordinary.decoder = ByteLevelDecoder()
        backend = HFTokenizer.from_str(self.ordinary.to_str())
        backend.add_special_tokens(list(MARKERS))
        self.ids = {marker: backend.token_to_id(marker) for marker in MARKERS}
        self.recipe = get_native_recipe(self.variant, backend)
        self.reference = renderers.Qwen35()
        _, self.reference_parser = self.reference.render_for_completion(
            chat.OpenAIMessage.from_oss_messages([{"role": "user", "content": "hi"}])
        )
        self.oracles = (
            ("generated-grammar", lambda case: normalize(case.parts)),
            ("tml-qwen35", self.oracle),
        )
        directory = Path(__file__).with_name("templates")
        self.templates = {
            variant: _compile_jinja_template(
                (directory / f"{variant}.jinja").read_text()
            )
            for variant in ("qwen3_5", "qwen3_8", "ornith")
        }

    @staticmethod
    def case(case):
        # Qwen's legal response grammar has one reasoning region, public
        # text, then calls. Project the shared parts generator onto it. XML
        # parameter names are identifiers; JSON *values* remain arbitrary.
        parts = []
        for kind in ("reasoning", "text"):
            text = "".join(
                part.value for part in case.parts if part.kind == kind
            ).strip()
            if text:
                parts.append(Part(kind, text))
        for part in case.parts:
            if part.kind == "tool":
                args = {
                    f"p{index}": value
                    for index, value in enumerate(part.value["args"].values())
                }
                parts.append(Part("tool", {"name": part.value["name"], "args": args}))
        return Case(tuple(parts))

    @staticmethod
    def fragments(case, *, terminal=True):
        for part in case.parts:
            if part.kind == "reasoning":
                yield False, part.value
        yield True, "</think>"
        for part in case.parts:
            if part.kind == "text":
                yield False, part.value
            elif part.kind == "tool":
                yield True, "<tool_call>"
                yield False, "\n<function=" + part.value["name"] + ">\n"
                for key, value in part.value["args"].items():
                    yield (
                        False,
                        "<parameter="
                        + key
                        + ">\n"
                        + json.dumps(value, ensure_ascii=False)
                        + "\n</parameter>\n",
                    )
                yield False, "</function>\n"
                yield True, "</tool_call>"
        if terminal:
            yield True, "<|im_end|>"

    def encode(self, case, *, terminal=True):
        result = []
        for special, value in self.fragments(case, terminal=terminal):
            result.extend(
                [self.ids[value]] if special else self.ordinary.encode(value).ids
            )
        return result

    def oracle(self, case):
        tokenizer = self.reference.tokenizer
        ids = []
        for special, value in self.fragments(case):
            ids.extend(
                [tokenizer.encode_special(value)]
                if special
                else tokenizer.encode_ordinary(value)
            )
        parts = []
        for message in self.reference_parser.parse_tokens(ids):
            match message.content:
                case chat.Text(text=text):
                    parts.append(Part("text", text.strip()))
                case chat.Thinking(text=text):
                    parts.append(Part("reasoning", text.strip()))
                case chat.InvokeTool(name=name, args=args):
                    parts.append(
                        Part(
                            "tool",
                            {
                                "name": name,
                                "args": {
                                    arg.name: json.loads(arg.value) for arg in args
                                },
                            },
                        )
                    )
                case chat.ModelEndSampling():
                    pass
                case other:
                    raise AssertionError(f"unexpected Qwen content {other}")
        return normalize(parts)

    def parser(self):
        return self.recipe.create_parser(
            self.recipe.render(
                [
                    resin.recipe.Message(
                        resin.recipe.Role.USER, [resin.recipe.Text("hi")]
                    )
                ],
                options=resin.recipe.RenderOptions(enable_thinking=True),
            )
        )

    def text_parser(self):
        return self.recipe.create_parser(
            self.recipe.render(
                [
                    resin.recipe.Message(
                        resin.recipe.Role.USER, [resin.recipe.Text("hi")]
                    )
                ],
                options=resin.recipe.RenderOptions(enable_thinking=False),
            )
        )

    @staticmethod
    def normalize_text(text):
        return text.strip()

    def tool_truncation(self, case):
        prefix = Case(case.parts[:-1])
        prefix_length = len(self.encode(prefix, terminal=False))
        final = list(self.fragments(Case(case.parts[-1:]), terminal=False))
        # The renderer has already opened thinking; output closes it before
        # the tool marker and its function header.
        header = final[2]
        header_length = 1 + len(self.ordinary.encode(header[1]).ids)
        ids = self.encode(case, terminal=False)
        return ids, prefix_length + header_length, len(ids) - 1

    @staticmethod
    def prompt_case(case):
        text = "".join(part.value for part in case.parts if part.kind != "tool")
        calls = [
            {"function": {"name": part.value["name"], "arguments": part.value["args"]}}
            for part in case.parts
            if part.kind == "tool"
        ]
        messages = [{"role": "user", "content": text}]
        if calls:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "reasoning_content": "Plan.",
                        "content": "",
                        "tool_calls": calls,
                    },
                    {"role": "tool", "content": "done"},
                ]
            )
        return {"messages": messages}

    def prompt_oracle(self, request):
        options = request.get("extra", {})
        variant = options.get("recipe_variant", self.variant)
        context = {key: value for key, value in request.items() if key != "extra"}
        context.update(options)
        context.setdefault("add_generation_prompt", True)
        return self.templates[variant].render(**context)


class OrnithAdapter(QwenAdapter):
    """Ornith's Qwen3.5 template: preserved reasoning and JSON tool arguments."""

    variant = "ornith"
