"""Typed conversations, native recipes, prepared media, and response parsers."""

from collections.abc import Sequence
from typing import Protocol

from resin._resin import (
    Audio,
    DeepSeekV41Image,
    DeepSeekV41Media,
    Gemma4Media,
    Glm53FlashMedia,
    Image,
    InklingMedia,
    Message,
    NativeRecipe,
    ParserEvent,
    Qwen35Media,
    ReasoningDelta,
    RenderedPrompt,
    RenderOptions,
    Role,
    Text,
    TextDelta,
    Thinking,
    Tool,
    ToolCall,
    ToolCallArgumentsDelta,
    ToolCallStart,
    ToolResult,
    Video,
    get_native_recipe,
)
from resin._resin import RecipeTokenizer as Tokenizer
from resin.protocols import (
    FinalizingResponseParser,
    MediaInput,
    MediaTokenSpan,
    ResponseParser,
    TokenSpan,
)

MediaBlock = Image | Video | Audio
ContentBlock = Text | Thinking | MediaBlock | ToolCall | ToolResult


class Recipe(Protocol):
    """Render requests for a model and construct an incremental response parser.

    Requests are given as a typed list of messages, standardized across model
    families. A recipe describes how to translate a conversation into tokens
    and media spans, similar to a Hugging Face "chat template."

    We recommend using one of the recipes in the `resin-recipe` crate, which
    includes optimized incremental parsers in Rust.
    """

    @property
    def stop_token_ids(self) -> Sequence[int]:
        """Static model stop IDs, independent of the rendered request."""
        ...

    @property
    def tokenizer(self) -> Tokenizer | None:
        """Prepared tokenizer for vocabulary metadata, or None when unavailable."""
        ...

    def render(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] = (),
        options: RenderOptions | None = None,
    ) -> RenderedPrompt:
        """Prepare media and render tokens with their final media positions."""
        ...

    def create_parser(self, prompt: RenderedPrompt) -> ResponseParser: ...


__all__ = [
    "Audio",
    "ContentBlock",
    "DeepSeekV41Image",
    "DeepSeekV41Media",
    "FinalizingResponseParser",
    "Gemma4Media",
    "Image",
    "InklingMedia",
    "MediaBlock",
    "MediaInput",
    "MediaTokenSpan",
    "Message",
    "NativeRecipe",
    "ParserEvent",
    "Qwen35Media",
    "Glm53FlashMedia",
    "Recipe",
    "ReasoningDelta",
    "RenderedPrompt",
    "RenderOptions",
    "ResponseParser",
    "Role",
    "Text",
    "TextDelta",
    "Thinking",
    "Tokenizer",
    "TokenSpan",
    "Tool",
    "ToolCall",
    "ToolCallArgumentsDelta",
    "ToolCallStart",
    "ToolResult",
    "Video",
    "get_native_recipe",
]
