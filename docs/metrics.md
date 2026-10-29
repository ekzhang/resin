# Metrics

Metrics are enabled by default. The engine switches are explicit keyword arguments:

```python
from resin import Engine
from resin.server import HttpServer

with Engine(runner, enable_metrics=True, context_length=32768) as engine:
    HttpServer(engine, recipe=recipe, model="my-model").serve(port=8000)
```

The server automatically exposes `/metrics` when engine metrics are enabled.
Pass `enable_metrics=False` to disable collection and the endpoint.
`DistributedEngine(worker, enable_metrics=True)`
enables the same instrumentation across its replica ranks; the frontend serves
one endpoint for the complete engine.

The serving loop publishes cumulative snapshots approximately once per second,
at normal control boundaries. A long execution can delay publication. Scrapes
read the last complete snapshot and never invoke Python, enter inference
collectives, or synchronize the GPU. Distributed publication gathers rank
snapshots periodically through CPU communication; it adds no collective to each
token's output path.

Applications without HTTP can collect the same data directly:

```python
snapshot = engine.metrics_snapshot()
if snapshot is not None:
    print(snapshot.requests["running_requests"])
    print(snapshot.cache)
```

The inspector returns a frozen `resin.EngineMetrics` snapshot. Its `requests`
and optional `cache` properties expose dictionaries and lists; `runner` is an
optional native `RunnerMetrics`.
Snapshots and returned inspection views are independent of later execution.
Cache inventory scans physical allocation metadata, so poll periodically rather
than on every generated token.

Snapshot fields follow exported metric stems where possible, omitting the
counter suffix `_total`. Request-state fields remain explicit:
`completed_requests` maps to `resin:num_requests_total`, while
`decode_tokens` feeds `resin:realtime_tokens_total{mode="decode"}` and
includes prefill output and committed work from cancelled requests.
For throughput between snapshots, divide the change in `decode_tokens` by the
change in `elapsed_seconds`.

## Exponential histograms and scraping

Latency and token distributions use native base-2 exponential histograms rather
than configured bucket boundaries. They start at schema 4 (a bucket growth factor
of about 1.044) and lower their resolution when the bucket budget is exceeded.
Engine and DP-wait snapshots hold at most 160 positive buckets; HTTP observations
use the same initial schema and bucket limit through `prometheus-client`.
Bucket merging preserves counts and the separately accumulated sum. Snapshots
expose `schema`, `offset`, noncumulative `counts`, `zero_count`, `count`, and `sum`.

Enable native histograms and protobuf scraping in Prometheus:

```yaml
scrape_configs:
  - job_name: resin
    scrape_native_histograms: true
    scrape_protocols: [PrometheusProto]
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

`/metrics` negotiates its representation from the `Accept` header. Protobuf
scrapes receive complete native distributions. The default text view retains
counters, gauges, histogram counts and sums, and the `+Inf` bucket for inspection;
it has no finite buckets and cannot provide useful percentiles. The server
encodes native snapshots directly for publication; the Python inspector is
separate from this export path.

## Ownership and timing

Rust records request and scheduler observations after transactional commit.
`resin-core` owns observations and snapshots. `resin-http` owns the Prometheus
metric schema, descriptions, and encoding. `resin-py` serializes inspection
snapshots and adapts Python runner observations, without duplicating their Rust
schemas or defining exporter policy.
Rejected work, failed commits, speculative padding, stop-clipped output, and
discarded overlap successors do not become generated-token observations.
Zero-output prompt scoring completes normally but contributes no TTFT or TPOT.

Request latency starts at native submission, including scheduler waiting.
Queue latency measures the initial wait before execution; memory retractions
have a separate counter. Token latency uses commit timestamps before response
parsing and `stream_interval` buffering. A batch containing several accepted
tokens contributes its interval divided by its token count, with that many
observations. Request TPOT contributes one observation per completed request
with at least two output tokens: `(completion - first_token) / (output_tokens - 1)`.

HTTP observations use separate names and begin when the validated generation
request enters the frontend admission queue. They include recipe rendering and
native queueing. HTTP TTFT ends at the first raw committed token, before parser
buffering. It does not measure client network latency. Successful HTTP latency
includes response preparation through terminal completion. HTTP and engine
histograms must not be combined as observations of the same clock.

| Measurement                               | Metric                                                                                                         |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| TTFT                                      | `resin:time_to_first_token_seconds`                                                                            |
| Token-weighted TPOT                       | `resin:inter_token_latency_seconds`                                                                            |
| Request-weighted TPOT                     | `resin:request_time_per_output_token_seconds`                                                                  |
| Successful request latency                | `resin:e2e_request_latency_seconds`                                                                            |
| Initial scheduler queue latency           | `resin:queue_time_seconds`                                                                                     |
| Accepted / completed / cancelled requests | `resin:num_accepted_requests_total`, `resin:num_requests_total`, `resin:num_aborted_requests_total`            |
| Memory retractions                        | `resin:num_retracted_requests_total`                                                                           |
| Running / waiting requests                | `resin:num_running_reqs`, `resin:num_queue_reqs`                                                               |
| Completed prompt / cached / output tokens | `resin:prompt_tokens_total`, `resin:cached_tokens_total`, `resin:generation_tokens_total`                      |
| Live token work                           | `resin:realtime_tokens_total{mode="prefill_compute\|prefill_cache\|decode"}`                                   |
| Completed request length distributions    | `resin:prompt_tokens_histogram`, `resin:uncached_prompt_tokens_histogram`, `resin:generation_tokens_histogram` |
| HTTP TTFT / successful latency            | `resin:http_time_to_first_token_seconds`, `resin:http_request_latency_seconds`                                 |
| HTTP queue latency / depth                | `resin:http_queue_time_seconds`, `resin:http_num_queue_reqs`                                                   |

Completed-request token totals exclude cancellations. Live counters retain
committed work for requests later cancelled. `prefill_compute` includes replay
after retraction; `prefill_cache` counts reused prompt tokens at successful
submission; `decode` counts committed output tokens, including the first token
produced by prefill. Calculate generation throughput with the live output
counter's rate over the dashboard window of your choosing.

Metrics carry `model_name` and `engine_type="resin"`. Engine and runner series
also carry `rank` and `attn_dp_rank`. Logical request observations are emitted
once by each attention-DP driver. Physical execution observations are emitted
for every participating rank. Native cache inventory describes each replica's
logical cache pool; it must not be multiplied by TP or CP when counting token
capacity. HTTP observations are emitted once by the frontend, without rank
labels.

## GPU and distributed observations

The optional `resin.protocols.MetricsRunner` contract has two methods:

```python
def configure_metrics(self, *, enabled: bool = True) -> None: ...
def metrics_snapshot(self) -> RunnerMetrics: ...
```

The snapshot uses the same names across native and Python execution:

```python
from resin import RunnerMetrics

snapshot = RunnerMetrics(
    forward_execution_seconds={"decode": 1.25},
    cuda_graph_passes={"decode_cuda_graph": 40},
    spec_draft_tokens=120,
    spec_accepted_tokens=90,
    spec_verify_rows=40,
    dp_attention_sync_wait_seconds=None,  # Optional exponential histogram.
)
```

`resin.RunnerMetrics` is a frozen native snapshot with keyword-only fields for
phase durations, graph counts, speculation, and DP wait. It is also re-exported
from `resin.backend.metrics`. The Engine enables collection before
`bind_resources`, allowing shared graph machinery to capture timing boundaries.
A snapshot returns cumulative data without device synchronization or distributed
collectives. Ordinary runners need neither method. `forward`,
`enqueue`, and `ForwardOutput` are unchanged.

Shared CUDA instrumentation records owned event generations for each execution
and collects intervals only after their completion event is ready. Captured
phase boundaries use independent replay generations so an overlapped successor
cannot overwrite a pending measurement.

| Measurement                      | Metric                                                                                                   |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Device time by phase             | `resin:forward_execution_seconds_total{category="prefill\|target_verify\|draft\|decode\|idle"}`          |
| No-work engine time              | `resin:scheduler_idle_seconds_total`                                                                     |
| Blocking host DP coordination    | `resin:dp_attention_sync_wait_seconds`                                                                   |
| Graph decisions per forward pass | `resin:cuda_graph_passes_total{mode="decode_cuda_graph\|decode_none\|prefill_cuda_graph\|prefill_none"}` |

Phase durations are exclusive within an observed execution. Detailed draft and
verification time is subtracted from its enclosing decode interval. `idle`
means participation without local requests: a rank may still execute expert
work. `scheduler_idle_seconds_total` instead measures no queued requests and no
outstanding physical execution. Logical completion with an already-enqueued
successor remains busy until the successor retires.

A graph hit means the pass used a graph. Several prefill regions count as one
pass; this does not imply that every operation was captured. Runners without
timing or graph instrumentation omit those observations rather than fabricating
zero work. The shared timing path has been validated on a Modal H100 with overlapping
captured replays, independent event generations, and nonblocking collection.
A warmed two-matmul microbenchmark measured about 13 µs extra per pass for
whole-pass timing and 30 µs for captured draft/verify detail. These are
instrumentation measurements on a tiny graph, not end-to-end model benchmarks.
The [H100 validation run](https://modal.com/apps/ekzhang/main/ap-cdI1ckx429UUBtzuhQnldS)
passed three shared-runtime tests covering captured and eager timing. The
one-off Modal wrapper has been retired; run
`uv run pytest tests/test_cuda_metrics.py -q` on a CUDA host for the current
shared-runtime timing tests.
Graphs captured before metrics are enabled retain whole-pass timing only.
Enable metrics before the initial capture for draft/verify detail;
`capture_cuda_graphs()` is idempotent and does not rebuild existing graphs.

The unaccounted fraction of a wall-time chart is **unattributed time**, not a
measurement of CPU overhead. Untimed device work can occur outside the spans;
device intervals can contain waits; host DP waits can overlap device execution.
Do not add the DP wait histogram's time to the device-time stack.

## Shared cache and prefix LRU

`resin:cache_slabs{state="capacity|free|occupied|evictable"}` describes the shared
physical pool. Evictable is a subset of occupied. `resin:token_usage` is occupied
slabs divided by total slabs, including evictable prefixes.

`resin:cache_group_blocks{pool,group,state}` describes blocks in the `kv`, `swa`,
and `recurrent` groups. Occupied, cached, evictable, pinned, and free counts are
not disjoint: cached and pinned blocks can still be owned by a live request.
Free blocks are unoccupied slots inside slabs already assigned to that group.
`resin:cache_group_slabs{pool,group}` describes assigned slabs.

`resin:cache_group_tokens{pool,group,state="occupied|evictable"}` reports
block-rounded history token capacity. Recurrent checkpoints use block counts
instead of misleading token counts. Groups share fungible slabs; they do not
have independent fixed capacities.

Evictable capacity is what removing every cached prefix could actually free,
with active requests and device leases unchanged. Removing an LRU entry can
free no storage because another prefix or request shares it. The prefix metrics
are `resin:prefix_cache_entries`, `resin:prefix_cache_hits_total`,
`resin:prefix_cache_misses_total`, `resin:prefix_cache_evictions_total`,
`resin:prefix_cache_reclaimed_blocks_total`, and
`resin:prefix_cache_reclaimed_slabs_total`. Reclaimed counters report physical
frees; eviction counts include explicit clearing.

## Dashboard queries

Native histograms aggregate across replicas. For example, p99 native TTFT:

```promql
histogram_quantile(0.99,
  sum by (model_name) (
    rate(resin:time_to_first_token_seconds[5m])
  )
)
```

Use the same query with `generation_tokens_histogram` for output-length
percentiles or `request_time_per_output_token_seconds` for request TPOT.
Live generation throughput in tokens per second, including unfinished requests
and committed work from cancellations (adjust the window as needed):

```promql
sum by (model_name) (rate(resin:realtime_tokens_total{mode="decode"}[1m]))
```

Per-rank GPU phase percentages:

```promql
100 * sum by (model_name, rank, category) (
  rate(resin:forward_execution_seconds_total[1m])
)
```

Select one rank for an engine interactivity chart. Summing device seconds over
GPUs produces GPU-seconds and can exceed wall time. Decode graph hit rate:

```promql
sum by (model_name, rank) (
  rate(resin:cuda_graph_passes_total{mode="decode_cuda_graph"}[1m])
) / sum by (model_name, rank) (
  rate(resin:cuda_graph_passes_total{mode=~"decode_.*"}[1m])
)
```

Replace `decode` with `prefill` for prefill graph usage.

Speculation exposes cumulative gauges `resin:spec_accept_rate` and
`resin:spec_accept_length`, plus raw counters `resin:spec_draft_tokens_total`,
`resin:spec_accepted_tokens_total`, and `resin:spec_verify_rows_total`. Use raw
counters for an interval's acceptance rate and mean accepted length:

```promql
rate(resin:spec_accepted_tokens_total[1m])
  / rate(resin:spec_draft_tokens_total[1m])

1 + rate(resin:spec_accepted_tokens_total[1m])
  / rate(resin:spec_verify_rows_total[1m])
```

These are runner observations before request stop/length clipping, so they can
exceed the engine's committed output. Several physical ranks can observe the
same speculative rows; retain rank labels when comparing raw counts. Aggregate
ratios from counter sums rather than averaging acceptance gauges.

## OTLP via a Collector

An OpenTelemetry Collector can scrape the protobuf endpoint and convert native
histograms to OTLP exponential histograms without instrumenting model execution
again. A minimal Collector configuration, with target addresses resolved from
the Collector's network environment:

```yaml
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: resin
          scrape_interval: 5s
          scrape_native_histograms: true
          scrape_protocols: [PrometheusProto]
          static_configs:
            - targets: ["127.0.0.1:8000"]
processors:
  batch: {}
exporters:
  otlphttp:
    endpoint: ${env:OTEL_EXPORTER_OTLP_ENDPOINT}
service:
  pipelines:
    metrics:
      receivers: [prometheus]
      processors: [batch]
      exporters: [otlphttp]
```

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to the backend's OTLP base URL and configure
authentication as that backend requires. See the Collector's
[Prometheus receiver](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/receiver/prometheusreceiver/README.md)
and [pipeline configuration](https://opentelemetry.io/docs/collector/configuration/).
Native OTLP export is not required by the runner or the core metrics API.
