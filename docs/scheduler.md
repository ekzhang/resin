# Native scheduling and batch phases

Resin keeps ready-queue ordering and request progress in Rust. The scheduler
selects work; the runner executes the already-frozen token ranges.

```python
from resin import Engine, SamplingParams, SchedulerConfig

engine = Engine(
    runner,
    max_running_requests=32,
    context_length=32768,
    enable_overlap=True,  # default
    scheduler_config=SchedulerConfig(
        max_batch_tokens=4096,
        prefill_chunk_size=8192,
        schedule_conservativeness=1.0,
    ),
)

request = engine.submit(
    prompt.token_ids,
    priority=5,
    batch_key=7,
    cacheable=True,
    sampling_params=SamplingParams(
        max_output_tokens=64,  # optional hard stop, not a KV reservation
        stop_token_ids=recipe.stop_token_ids,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        seed=7,
    ),
)
```

`cacheable` defaults to true. False skips prefix lookup and publishing for that
request; see [unified paged cache](cache-pool.md).

`context_length` is an Engine-level model limit, not a batching preference. It
defaults to 32768 and rejects requests whose requested sequence would exceed
that bound. If `SamplingParams.max_output_tokens` is `None` (the default), generation
may use the remaining model context. The value is always a terminal generation
cap; Resin never reserves that entire lifetime at submission.

`prefill_chunk_size` defaults to 8192 tokens and is the scheduler's execution
unit for prompt or replay work. A runner may declare a smaller hard
`prefill_chunk_size`; the Engine uses the smaller bound and never silently
re-chunks an accepted plan. A runner must declare
`RunnerConfig(continuation_prefill=True)` when the selected range stops before the
end of a prompt; the Engine rejects such a range before runner acceptance when
the runner does not support continuation.

## Cohort selection

The scheduler distinguishes admitted requests from waiting requests. Admission
is sticky: successful survivors remain admitted until completion, cancellation,
or memory retraction. There is no FIFO/round-robin mode that rotates a decoder
back behind waiting work.

Selection is deterministic:

1. Keep admitted prefill chunks running and fill compatible vacant slots.
2. Otherwise, admit waiting prefills into vacant slots and run that prefill
   batch before resident decoders.
3. When no prefill can run, execute the resident decode batch.
4. Within one phase, choose the highest priority and oldest compatible
   `batch_key` cohort.
5. Never exceed `max_running_requests` admitted requests.

Waiting admission chooses the highest compatible priority, while already
admitted requests retain their slots. A batch may therefore contain admitted
rows of different priorities. It never mixes keys or phases. `batch_key=None`
is a real cohort and differs from zero. Priorities are signed 32-bit integers;
larger values are admitted first within the preferred phase.

The ready queue stays in age order. Taking a batch merges selected indices with
one queue traversal; returning survivors sorts only those survivors and merges
them into the queue. Neither operation repeatedly scans or sorts all waiting
requests for each selected row.

Prefill and decode never share a plan. `BatchPlan.phase` is the explicit
`BatchPhase.PREFILL` or `BatchPhase.DECODE` value for the complete batch. A
one-token continuation prefill therefore remains unambiguously prefill.

Prefill can pause resident decoders, but does not evict them: their KV and
admission remain live. A full admitted set is not displaced merely to accept a
new prefill.

Native turn ordinals never cross the Python boundary. A pre-accept rejection
restores the selected requests and their exact order.

## KV admission and retraction

Managed KV starts empty and grows transactionally to the end of each frozen
plan. Admission uses an estimated completion length, not the request's entire
generation cap. `schedule_conservativeness` scales that estimate:

- larger values reserve more admission headroom and reduce retractions;
- smaller values admit more concurrent work and may replay more often;
- `1.0` is the default, while `0.0` is maximally aggressive.

The estimated new-token ratio begins at
`min(0.7 * schedule_conservativeness, 1.0)`, decays during successful decode,
and is raised from observed progress after a retraction. `Engine.new_token_ratio`
exposes the current diagnostic value. This estimate controls admission only;
actual pages are allocated on demand, and allocation failure cannot corrupt an
existing request table.

If the next step cannot grow, Resin keeps at least one running request and
retracts an eligible request with the least completed output
(longer prompts lose ties). It asks the runner to release request-local state,
frees the request's KV, returns it to the waiting tail, and later replays its
prompt plus committed generated inputs. Replay prefill never samples a duplicate
token.

Retracted requests release all History, Window, and State pages together, then
replay from their token history when readmitted. A prefix-cache hit may avoid
recomputing the aligned shared prefix.

## Token-range planning

Without token controls, a prefill row consumes its remaining prompt or replay
input and a decode row consumes its one input token. With controls:

- `prefill_chunk_size` limits each selected prefill row;
- `max_batch_tokens` is consumed in row order across flattened plan tokens;
- decode rows always request one token and later rows are omitted when the
  decode batch exhausts its token budget;
- the scheduler keeps an oldest prefix of the selected cohort rather than
  skipping a row.

For every managed Window group, a step's write span must fit
`window_tokens - read_window_tokens + 1`. The Engine caps prefill by the smallest
of these spans and rejects incompatible speculative lookahead at construction.
If the logical read window is omitted, it defaults to physical retention, which
permits one write per step. Built-in runners reserve the read window plus a
prefill guard band so they can execute whole chunks safely, including after a
prefix-cache hit. Physical block boundaries alone do not determine safe chunks.

`BatchPlan.token_indptr` describes the final flattened range for each row.
`context_lengths` contains each row's pre-append logical length. Initial prefill
is sampled only when it reaches the end of the prompt; replay prefill is never
sampled; decode rows are always sampled. `sampling_mask` makes this event/commit
mask explicit. Prefill returns one token ID per row, including an ignored ID
for an unsampled continuation or replay row. Decode may return a verified token
block within the runner's declared speculative capacity.

`SamplingParams` is a frozen per-request policy that holds generation limits,
stop token IDs, sampling settings, and logprob options. `stop_token_ids` defaults
to an empty tuple, so raw token requests must supply any desired model stop
tokens explicitly. For recipe requests, pass `recipe.stop_token_ids`; HTTP
serving does this automatically. There is no Engine-level EOS fallback. Stop IDs are copied into
an immutable tuple at construction. Reuse a policy for requests with the same
settings, or construct a new one to change the limit or stops.

Sampling defaults to temperature `1.0`, top-p `1.0`, and no top-k limit.
Temperature `0` requests deterministic argmax. The frozen plan carries one policy
and effective seed per row; the runner derives random offsets from absolute
sequence progress so requests can share a batch without
sharing a random stream. The optional
`extra` value is an opaque Python object for custom runner code; Resin preserves
it but does not inspect it. Shared Torch and fused CUDA backend samplers consume
the standard fields.

## Step and completion

`step()` drives batch submission and completion and returns committed token
events. The default overlap scheduler can enqueue a decode batch and return an
empty list while it runs, then submit a safe continuation before harvesting the
previous result. Without overlap, a step selects at most one batch, calls the
runner synchronously, and commits its result before returning. An empty list
also occurs when there is no work or a continuation-prefill step advances cache
state without sampling; use `has_work()` to determine whether work remains.

On successful synchronous completion Resin:

1. validates every raw output row against the phase and leased write capacity;
2. applies stop and length limits and prepares progress updates and events;
3. asks the runner to release terminal request-local state, if supported;
4. releases cache access, publishes aligned prefixes, and retires terminal resources;
5. commits prefill/decode progress and completes terminal request handles;
6. returns survivors to the sticky admitted set;
7. emits events for sampled rows.

Stop token IDs and `SamplingParams.max_output_tokens` are terminal conditions.
Cancellation completes a ready request without running another model step.

## Overlap scheduling

`Engine(enable_overlap=True)` is the default. Runners can implement the optional
`enqueue(plan, *, feedback=None)` boundary and return an owned completion with
`result()` and device `feedback`. The Engine submits the next decode continuation
using that feedback before collecting and committing the previous result on the
CPU, so device execution overlaps host completion work. Synchronous runners
continue to work through `forward()`.

Overlap requires a stable decode cohort, enough remaining output and context
budget, and both steps' write spans inside the same owned blocks in every cache
group. The Engine retains the requests and a single writer lease across
continuations. New arrivals, cancellation, close, and cache
page boundaries drain pending work before changing ownership. Other steps,
including prefill, use the synchronous path. `enable_overlap=False` disables
overlap for the Engine; `step(allow_overlap=False)` drains pending work or runs
the next batch synchronously.

Generated-token logprobs and top alternatives can overlap when the runner's
completion returns the requested score columns. The shared CUDA completion
computes scores on the device and copies them into owned buffers before
the next graph replay. Score packing waits until the completion is collected;
token-only decoding does not compute or transfer scores.

Completions use the same output validation and logical commit rules as the
synchronous path. A stop token can finish a request while its successor is
already in flight: the result is published immediately, but its row, runner
state, and cache lease remain live until that successor completes. Outputs from
the successor for finished rows are discarded before resources are retired.

## Failure semantics

Failures before runner acceptance are transactional: temporary access is
released and selected requests return to their original positions. Once the
runner may have addressed cache or state storage, a protocol, cleanup, or commit
failure poisons the Engine. The accepted batch and its capabilities are
quarantined rather than retried against reusable memory.

This distinction is the scheduler's central safety rule: logical retry is only
allowed while Resin can prove that physical execution did not begin.

## Current limits

Scheduling does not choose CUDA graph buckets, kernel shapes, or sampling
kernels. Those remain runner concerns. Cache admission and request progress are
Engine concerns; sampling offsets are derived from absolute sequence progress.
Overlap submits safe decode continuations ahead of CPU completion; prefill and
decode still execute as separate phases and do not overlap each other.
