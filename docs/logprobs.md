# Token logprobs

Request scores with `SamplingParams(logprobs=True)` or alternatives with
`SamplingParams(top_logprobs=5)`. Each returned `TokenEvent.logprobs` contains the
selected token's `logprob` and a list of `(token_id, logprob)` alternatives.
Scores describe the raw model distribution, before temperature, top-k, or top-p
sampling; requesting them does not change token selection.

`Engine(runner, max_top_logprobs=32)` sets the maximum number of top alternatives
per generated or prompt token. The default is 32. Direct submissions and HTTP
requests use the same Engine limit; raise it to allow more alternatives, including
counts above 1,000. Setting it to zero disables top alternatives while retaining
selected-token logprobs. `SamplingParams` holds request counts; the Engine rejects
counts above its configured limit before admission.

Request input scores independently with `SamplingParams(prompt_logprobs=True)`.
Use `prompt_top_logprobs` for top alternatives at each prompt position, or pass
`prompt_target_logprobs` to `Engine.submit` for specific token IDs at each position:

```python
lease = engine.submit(
    [10, 20, 30],
    sampling_params=SamplingParams(max_output_tokens=0, prompt_top_logprobs=3),
    prompt_target_logprobs=[[], [42, 43], [7, 7]],
)
```

This asks the distribution after token `10` to score targets `42` and `43`, and
the distribution after token `20` to score target `7` twice. The target matrix
must have one row per prompt token, with an empty first row. Target order and
duplicates are preserved. Targets are prompt-only; there is no generated-token
target parameter.

`RequestLease.prompt_logprobs` contains one entry per prompt token. The first is
`None` because it has no predecessor; other entries contain raw-model scores
after the predecessor token's prefill chunk commits. Unprocessed entries remain
`None`. Each scored entry has the actual prompt token's `logprob`, requested
`top_logprobs`, and requested `target_logprobs`; both alternative lists contain
`(token_id, logprob)` pairs. Any of the three prompt options enables input
scoring. `top_logprobs` and `prompt_top_logprobs` are independent. Set
`SamplingParams(max_output_tokens=0, prompt_logprobs=True)` to score only the input
and complete without output events.
All stock runners reuse the same prefill hidden states and project scores in
bounded chunks, with no second transformer pass. Cross-chunk targets are part
of the execution plan, so the final predecessor in a chunk scores the first
token of the next chunk.

Scoring requests bypass prefix restoration: KV cache entries do not retain
the predecessor distributions needed to score a cached prefix. They still
publish reusable KV prefixes for ordinary requests. Generated-token scoring,
including speculative acceptance, keeps its existing behavior.

## CUDA execution

CUDA runners share `resin_models_cuda.scoring`; reference runners use
`resin.backend.sampling`. Scoring keeps captured model decode enabled, including
speculative verification and overlap scheduling. A separate captured score tail
reads existing raw-logit storage and scores the actual accepted tokens.
Completion snapshots scores into owned host buffers before later replays can
overwrite them, then packs accepted positions and requested alternatives during
harvest.

The [CuTe DSL kernels](../models/cuda/resin_models_cuda/_scoring_cutedsl.py) reduce
raw logits into stable normalization statistics and gather
selected, top, and explicit target scores. They never materialize a full FP32
normalized vocabulary tensor. Unscored batches launch no scoring kernels; the
selected-only and common top-alternative paths skip unrequested rows and rejected
verification positions.

Counts through 32 use fused CuTe reduction and selection. Counts from 33 through
2,048 use FlashInfer radix selection with sparse CuTe normalization; larger
counts use repeated CuTe selection passes. Larger requests cost more work and
output storage. Prompt scoring eagerly projects and scores at most 256 requested
predecessor rows at a time, with no prompt graph cache or padded target capacity.

Compiled kernels specialize for vocabulary geometry, dtype, layout, and scoring
features. After a path's first warmup, changing batch size or requested top count
does not compile new kernels. Generated-token score graphs bind fixed buffers,
so a new storage shape or capacity can require graph capture. Persistent artifacts
use the [kernel cache](kernel-cache.md); reuse does not eliminate graph capture
or model warmup.

`deploy/modal_bench_scoring.py` runs checkpoint-free correctness, replay, compilation,
allocation, and timing checks on A100, H200, and B200. Reports go under the ignored
`profiles/scoring/` directory. Kernel timings measure scoring independently of
model execution, filtered sampling, and host submission overhead.

## Array boundary

Runners return a `ForwardOutput` with contiguous one-dimensional NumPy arrays:

| Column | Dtype | Meaning |
| --- | --- | --- |
| `token_ids` | uint32 | Packed output tokens |
| `token_indptr` | uint32 | Token row offsets, length batch + 1 |
| `logprobs` | float32 | Packed selected-token scores |
| `logprob_indptr` | uint32 | Score row offsets; defaults to token_indptr |
| `top_indptr` | uint32 | Alternative offsets, length scored tokens + 1 |
| `top_token_ids` | uint32 | Packed alternative token IDs |
| `top_logprobs` | float32 | Packed alternative scores |
| `prompt_logprobs` | float32 | Packed requested predecessor scores |
| `prompt_logprob_indptr` | uint32 | Prompt-score row offsets, length batch + 1 |
| `prompt_top_indptr` | uint32 | Top-alternative offsets, length prompt scores + 1 |
| `prompt_top_token_ids` | uint32 | Packed prompt top-alternative IDs |
| `prompt_top_logprobs` | float32 | Packed prompt top-alternative scores |
| `prompt_target_indptr` | uint32 | Target offsets, length prompt scores + 1 |
| `prompt_target_token_ids` | uint32 | Packed requested prompt target IDs |
| `prompt_target_logprobs` | float32 | Packed prompt target scores |

The three `top_*` columns are supplied together or omitted. An empty generated-score
row opts out; every other generated-score row must cover its complete raw token
row, including positions beyond EOS or the remaining output budget. All offsets start at zero,
are monotone, and end at their corresponding column length.

Construction retains the exact arrays without dtype conversion or copying. The
Rust adapter acquires read-only NumPy borrows once and retains their owners while
the result is validated and committed. Runners must not modify or recycle these
buffers during that interval, including from `release()`. An asynchronous
completion owns its staged score buffers independently of later graph replays.
Tensor-backed NumPy views keep their CPU tensors alive through NumPy's base
owner. The adapter rejects incorrect
dtypes and noncontiguous or unaligned storage instead of silently coercing them.

This removes Python scalar/tuple boxing and intermediate Python-to-Rust token
and score copies. GPU-to-host transfers still occur; only committed history and
events acquire Rust-owned storage. Token-only results use the same flat
`ForwardOutput` contract; plain lists are rejected.

Native Rust runners can expose the same borrowed columns through
`Runner::logprobs() -> Option<LogprobView<'_>>`. The core validates every raw score
before committing any request, including scores that EOS would otherwise hide.
Invalid outputs follow the existing execution quarantine path.

Prompt-score columns are supplied together, independently of generated-token
scores. `BatchPlan.prompt_score_indices` selects flattened prefill predecessor
rows, `prompt_score_token_ids` supplies their next prompt tokens, and
`prompt_score_indptr` partitions them by sequence. `prompt_score_top_counts`
supplies each predecessor's requested top count. `prompt_target_indptr` and
`prompt_target_token_ids` supply its requested targets in the same order as the
score rows. Each optional top/target column triplet is supplied together.
Runners return every requested score, including on intermediate prefill chunks
with no sampled output. The shared scorer normalizes each raw distribution once
for selected, top, and target scores, and checks target vocabulary bounds before
gathering. Native runners expose these columns through `Runner::prompt_logprobs()`;
invalid or missing scores quarantine the entire execution before progress becomes visible.

## HTTP

Chat Completions accepts `logprobs: true` and `top_logprobs` from zero through the
Engine's `max_top_logprobs`. Responses
accepts `include: ["message.output_text.logprobs"]` and `top_logprobs`; a positive
alternative count also enables selected-token scores. These flags reach the
same native `SamplingParams` and NumPy score columns used by direct Engine calls.

Every scored generated token appears in `resin.generation_logprobs`, with
`token_id`, zero-based `token_index`, display `token`, `bytes`, `logprob`, optional
semantic `channel`, and `top_logprobs`. Alternatives include their IDs, display
text, bytes, and scores. Byte-level and byte-fallback tokenizers retain the actual
bytes, including incomplete UTF-8; concatenating display strings can introduce
replacement characters, so use `bytes` to reconstruct text. Unsupported decoder
formats and custom recipes without a serialized tokenizer report `bytes: null`.
JSON represents a valid negative-infinite logprob as `null`.

The complete raw list includes reasoning, tool markup, headers, and model
terminators. Canonical Chat `choices[0].logprobs.content` and Responses
`output_text.logprobs` contain only tokens proven to belong wholly to visible
content. Native recipe parsers carry source-byte spans through grammar buffers,
UTF-8 decoding, trimming, and stop matching. Held text waits until a complete
source token can be associated with its score. If trimming or a stop string cuts
inside a token, its surviving text still appears, but that token's score remains
raw-only: a whole-token probability cannot describe a fraction of that token.
Transformed tool arguments have no fabricated text attribution. The raw list
remains complete through the token that matches the stop, with no records for
the remaining verified-block suffix.

Chat streams publish the raw records on chunks under
`resin.generation_logprobs`; their canonical content records accompany the text
they describe. Responses streams publish `resin.generation_logprobs.delta` events
with a `logprobs` array. Canonical records accompany each
`response.output_text.delta`, and accumulate on the completed text part and final
response. Concatenating raw score arrays yields the same records as the
non-streaming response, independently of semantic parser chunk boundaries.

Every recipe parser returns provenance from its ordinary `feed` and `finish`
methods, for scored and unscored requests. Custom parsers use the same
`ResponseParser`/`FinalizingResponseParser` protocols from `resin.protocols`.
They return native `resin.recipe.ParserEvent` objects: `TextDelta`,
`ReasoningDelta`, `ToolCallStart`, or `ToolCallArgumentsDelta`. Fields are read
as attributes; parser events do not use dictionaries. Each `token_spans` entry
is a five-integer tuple:

```python
resin.recipe.TextDelta("ab", token_spans=[(0, 0, 1, 0, 1), (1, 0, 1, 1, 2)])
```

The fields are `(token_index, source_start, source_end, text_start, text_end)`.
The example maps the full bytes of generated tokens 0 and 1 onto `a` and `b`.
Token indices count all generated tokens, including headers and controls.
Source ranges address the original vocabulary token's bytes; text ranges address
the event delta's UTF-8 bytes. Both are half-open, and every field fits uint32.
Several byte tokens can share one whole Unicode character's text range. Preserve
these compact spans in source-token order through buffering; the native representation uses five u32
values rather than six machine-sized offsets. Full-token byte lengths come from
the tokenizer vocabulary instead of being repeated in each span.

Empty spans, the constructor default, mean synthetic text or unknown provenance.
They still allow raw scores, with no guessed canonical association. Custom Python
tokenizers without a serialized backend can use this fallback; native parser
creation rejects unsupported serialized decoder chains. Malformed span values,
unseen or out-of-order source indices, empty/reversed
ranges, and invalid text bounds fail the request even when scores are off,
without poisoning the engine. Scored requests also check source byte extents
against the raw vocabulary bytes when those bytes are available.

Both APIs accept `prompt_logprobs: true`, independently of generated scores.
Prompt top alternatives and target scoring are currently engine-only.
The result's `resin.prompt_logprobs` follows the complete rendered prompt,
including model scaffold and control tokens. Its first entry is `null`; later
entries carry token IDs, text, bytes, and scores. Set `max_completion_tokens: 0`
(Chat) or `max_output_tokens: 0` (Responses) for input scoring without generation.
Chat streams send one `resin.prompt_logprobs` field; Responses streams send one
`resin.prompt_logprobs` event with a `logprobs` array before completion.

## Full-model validation limits

Kernel qualification does not establish model parity across prefill chunk sizes
or distributed topologies. GPT-OSS-20B unchunked scores matched an independent
output-head oracle within 1.91e-6, but a historical single-H100 run differed by up
to 5.15 nats between a 180-token chunk and 32-token chunks. Composed distributed
runs also retained nine failed strict comparisons despite matching greedy probes.
The cause remains unresolved; see [distributed validation](distributed.md) and
the [prompt-score contract tests](../models/cuda/tests/test_gpt_oss_distributed_runner.py).
