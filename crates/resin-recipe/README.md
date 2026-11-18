# resin-recipe

Stock model conversation formats, independent of Python, HTTP and GPU execution.
Concrete recipes implement the `Recipe` trait for GPT-OSS, DeepSeek V4.1,
Inkling2, Qwen3.5/3.8, Gemma4, GLM-5.3-Flash or LFM2.5. Each family owns its
configuration, renderer, parser, message policy and stop markers.

`Recipe::from_config` resolves model defaults through a lazy provider for processor
metadata and dates. `render` validates request controls against those defaults;
parsers use the same recipe and rendered prefix. The recipe also declares input
modalities and prepares normalized RGB bitmaps and media placeholders through
`prepare_media`. Its private `media` module owns the family's settings and pixel
processing. Settings are parsed once; preparation borrows them and returns the
shared `Prepared` type. One Python adapter handles preparation caching, prompt
encoding and binding to immutable `Media` objects.

Model code describes format rules. Shared `Parser<G>` owns token batching,
decoder boundaries, EOS handling and finalization; each grammar handles only
text and control transitions. A small token declaration supplies spellings,
aliases, lookup and validation once. Renderers share request/tool conventions
and span operations while retaining their model-specific ordering and framing.

A renderer produces a `Prompt`: ordinary text plus explicit control-token spans.
Adjacent text coalesces before encoding, preserving BPE boundaries. A prepared
`TokenizerEncoder` shares the tokenizer and encodes ordinary marker spellings as
text. Native image/video processor expansions replace only explicit media
slots, then use the same encoding boundary. Each model's `media` module uses the
Rust `image` crate for resizing, padding and normalization. Runners derive patch
geometry from the bitmaps and pack their tensor layout before vision encoding.
Resize pixels can differ from HF filters; strict no-resize oracles check
normalization, padding and patch order.

`Media(span, image=...)` accepts a `float32` HWC RGB array and snapshots its
contents, including strided inputs. Video uses a tuple of frame arrays plus frame
spans and timing; audio uses samples and a sample rate. The arrays are read-only.
`RenderedPrompt.media` is an ordered tuple of these objects; token IDs remain in
`RenderedPrompt.token_ids` and are required separately for Engine submission.

`ResponseParser::feed` consumes only new token IDs. `finish(Stop | Length)` flushes
buffered text, including truncated UTF-8; finalization is idempotent. A shared
`TokenizerDecoder` retains the tokenizer's rolling decode context. Control IDs,
not printable text, drive model grammar. Tool behavior follows the format:
Inkling2/Harmony/Qwen/Gemma/LFM commit validated calls at their closing marker;
DeepSeek streams DSML calls and argument deltas as they arrive. Finalization does
not invent complete arguments for an unfinished call.

Python's `get_native_recipe(name, tokenizer_or_processor)` factory returns the
common `NativeRecipe`, implementing the structural `Recipe` protocol. See the
[Python API example](../../README.md#minimal-use) for Engine submission.

A native `RenderedPrompt` belongs to the recipe instance that rendered it.
Each `create_parser(prompt)` call creates independent response state without
retaining pixels. Submit with `token_ids=prompt.token_ids, media=prompt.media` and
pass `recipe.stop_token_ids` to `SamplingParams`; HTTP serving applies those stops
automatically.

These parsers use one common native handle; the HTTP actor consumes Rust events
directly. Python callers receive immutable `ParserEvent` variants: `TextDelta`,
`ReasoningDelta`, `ToolCallStart` and `ToolCallArgumentsDelta`. Their payload and
`token_spans` provenance are attributes. Applications can supply a custom recipe
with the same `render` and `create_parser` methods and parser `feed` (plus optional
`finish`), returning these native events. HTTP wire dictionaries stay in the HTTP
adapter, outside the native recipe API.

Text and reasoning events carry `delta`; tool starts carry `index`, `id`, and
`name`; tool argument deltas carry `index` and `delta`. Every variant exposes a
`type` discriminator and immutable `token_spans` for source-token attribution.
See [logprob attribution](../../docs/logprobs.md#http).

Fast Hugging Face tokenizers are snapshotted once and shared across requests.
Custom Python tokenizers can supply `encode`, `convert_tokens_to_ids` and
`decode`, with optional `encode_ordinary` for literal text. This compatibility
path retains Python callbacks and cumulative
decoding. Its special-token behavior is the custom tokenizer's responsibility.
The native path follows backend tokenizer semantics rather than Transformers'
additional `clean_up_tokenization_spaces` postprocessing.

References are pinned checkpoint templates/encoders for prompt parity. Harmony
and DeepSeek Recipe are primary parser oracles; TML supplies secondary checks
where available. All reference libraries and templates are development-only.
DeepSeek V4.1 supports its DSML grammar and image slots. LFM's
nested Python-literal strings use a fixed Python 3.13 / Unicode 15.1 printable
repertoire for deterministic rendering across deployment Python versions.

Run `cargo test -p resin-recipe` for native invariants,
`uv run pytest tests/test_recipe*.py tests/runtime/test_recipe*.py -q` for integration, and
`uv run pytest tests/recipe_fuzz -q` for shared differential fuzzing.
See the [fuzz README](../../tests/recipe_fuzz/README.md) for budgets and oracle
limitations.

GLM-5.3-Flash uses `get_native_recipe("glm5_3_flash", tokenizer)`, with the
NVIDIA NVFP4 checkpoint template pinned to
`da920bb0b9f4a06727223a349e55468e38352348`. It supports text conversations,
reasoning (`low`, `high`, or the template default `max`), `clear_thinking`, and
GLM XML tool calls. The three static stops are `<|endoftext|>`, `<|user|>` and
`<|observation|>`. The template always opens `<think>`; it does not implement an
`enable_thinking=False` mode. Images use a native processor with aspect-preserving
bicubic resize, a black-padded aligned canvas and CLIP normalization. The runner
packs the generic `Media` bitmaps in checkpoint temporal/spatial patch order;
their spans and content identities support distributed transport and partial
image prefill. Native defaults match the pinned checkpoint; `media_config` can
supply the checkpoint's `image_processor` object.
Video and audio input remain unsupported. The pinned text/format oracle is `tests/fixtures/glm_hf.json`;
`licenses/glm-MIT.txt` retains the upstream Z.AI notice.
