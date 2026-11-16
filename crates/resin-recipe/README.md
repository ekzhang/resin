# resin-recipe

Stock model conversation formats, independent of Python, HTTP and GPU execution.
Concrete recipes implement the `Recipe` trait for GPT-OSS, DeepSeek V4.1,
Inkling2, Qwen3.5/3.8, Gemma4, GLM-5.3-Flash or LFM2.5. Each family owns serde construction
parameters, such as `GptOssRecipeParams`, and its matching renderer, parser,
message policy and stop markers.

Rust callers construct a recipe once from its parameters. `render` merges and
validates each request's `extra` controls against those defaults. The matching
parser uses the same recipe parameters and the actual rendered prefix. Media
configuration is captured at construction and stays outside per-request
rendering controls.

Model code describes format rules. Shared `Parser<G>` owns token batching,
decoder boundaries, EOS handling and finalization; each grammar handles only
text and control transitions. A small token declaration supplies spellings,
aliases, lookup and validation once. Renderers share request/tool conventions
and span operations while retaining their model-specific ordering and framing.

A renderer produces a `Prompt`: ordinary text plus explicit control-token spans.
Adjacent text coalesces before encoding, preserving BPE boundaries. A prepared
`TokenizerEncoder` shares the tokenizer and encodes ordinary marker spellings as
text. Native image/video processor expansions replace only explicit media
slots, then use the same encoding boundary. Model-owned modules under `media/`
prepare pixels and model geometry using the Rust `image` crate. The Python
binding exposes owned NumPy payload arrays and cache identities; runners execute
the GPU vision encoder. Resize pixels can differ from HF filters; strict
no-resize oracles check normalization, padding and patch order.

Direct payload constructors accept NumPy arrays with their declared dtypes:
`float32` pixels/patches and `int64` geometry. They copy the data into owned
storage, including strided inputs. Runners perform any Torch/device conversion
when preparing the vision encoder inputs.

`ResponseParser::feed` consumes only new token IDs. `finish(Stop | Length)` flushes
buffered text, including truncated UTF-8; finalization is idempotent. A shared
`TokenizerDecoder` retains the tokenizer's rolling decode context. Control IDs,
not printable text, drive model grammar. Tool behavior follows the format:
Inkling2/Harmony/Qwen/Gemma/LFM commit validated calls at their closing marker;
DeepSeek streams DSML calls and argument deltas as they arrive. Finalization does
not invent complete arguments for an unfinished call.

Python's `get_native_recipe(name, tokenizer_or_processor)` factory returns the
common `NativeRecipe`, implementing the structural `Recipe` protocol. See the
[Python API example](../../README.md#modal-http-endpoints) for construction,
formatting defaults, typed messages, media preparation and Engine submission.

A native `RenderedPrompt` belongs to the recipe instance that rendered it.
Each `create_parser(prompt)` call creates independent response state without
retaining pixels. Pass `recipe.stop_token_ids` to `SamplingParams` for direct
Engine submissions; HTTP serving applies those stops automatically.

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
`uv run pytest tests/test_recipe*.py -q` for integration, and
`uv run pytest tests/recipe_fuzz -q` for shared differential fuzzing.
See the [fuzz README](../../tests/recipe_fuzz/README.md) for budgets and oracle
limitations and the [Python API example](../../README.md#modal-http-endpoints)
for native recipe construction.

GLM-5.3-Flash uses `get_native_recipe("glm5_3_flash", tokenizer)`, with the
NVIDIA NVFP4 checkpoint template pinned to
`da920bb0b9f4a06727223a349e55468e38352348`. It supports text conversations,
reasoning (`low`, `high`, or the template default `max`), `clear_thinking`, and
GLM XML tool calls. The three static stops are `<|endoftext|>`, `<|user|>` and
`<|observation|>`. The template always opens `<think>`; it does not implement an
`enable_thinking=False` mode. Images use a native processor with aspect-preserving
bicubic resize, a black-padded aligned canvas, CLIP normalization, and the
checkpoint's temporal/spatial patch order. `Glm53FlashMedia` owns pixel/grid
buffers, absolute image positions and pixel-dependent cache identity; it supports
distributed transport and partial image prefill. Native defaults match the pinned
checkpoint; `media_config` can supply the checkpoint's `image_processor` object.
Video and audio input remain unsupported. The pinned text/format oracle is `tests/fixtures/glm_hf.json`;
`licenses/glm-MIT.txt` retains the upstream Z.AI notice.
