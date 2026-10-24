//! Export typed snapshots without introducing an exporter into the engine core.

use crate::{
    cache_metrics::CacheMetrics,
    metrics_types::{Histogram, RequestMetrics},
};
use prometheus::{TextEncoder, proto};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::BTreeMap;

/// An immutable logical-engine snapshot and optional physical-runner observations.
/// Reading this snapshot never synchronizes the accelerator. Cache inventory is
/// scanned, so collect periodically rather than once per token.
#[pyclass(frozen, get_all, module = "resin._resin", skip_from_py_object)]
pub(crate) struct EngineMetrics {
    pub(crate) requests: RequestMetrics,
    pub(crate) generation_throughput: f64,
    pub(crate) cache: Option<CacheMetrics>,
    pub(crate) runner: Option<Py<PyAny>>,
}

#[pymethods]
impl EngineMetrics {
    /// Encode one engine's cumulative observations in Prometheus text format.
    #[pyo3(signature = (*, model_name, rank = 0, attn_dp_rank = 0, include_runner = true))]
    pub(crate) fn prometheus(
        &self,
        py: Python<'_>,
        model_name: &str,
        rank: u32,
        attn_dp_rank: u32,
        include_runner: bool,
    ) -> PyResult<String> {
        let mut out = Exposition::new(model_name, rank, attn_dp_rank);
        let r = &self.requests;
        out.gauge("gen_throughput", "Committed output tokens per second in the last collection window (at least one second).", self.generation_throughput, &[]);
        for (name, help, value) in [
            (
                "num_accepted_requests_total",
                "Requests accepted by the native engine.",
                r.accepted_requests,
            ),
            (
                "num_requests_total",
                "Successfully completed requests.",
                r.completed_requests,
            ),
            (
                "num_aborted_requests_total",
                "Cancelled requests.",
                r.aborted_requests,
            ),
            (
                "num_retracted_requests_total",
                "Requests retracted for cache pressure.",
                r.retracted_requests,
            ),
            (
                "prompt_tokens_total",
                "Input tokens in completed requests.",
                r.prompt_tokens,
            ),
            (
                "generation_tokens_total",
                "Output tokens in completed requests.",
                r.generation_tokens,
            ),
        ] {
            out.counter(name, help, value as f64, &[]);
        }
        out.counter(
            "cached_tokens_total",
            "Cached input tokens in completed requests.",
            r.cached_tokens as f64,
            &[("cache_source", "device")],
        );
        for (mode, value) in [
            ("prefill_compute", r.prefill_compute_tokens),
            ("prefill_cache", r.prefill_cache_tokens),
            ("decode", r.decode_tokens),
        ] {
            out.counter(
                "realtime_tokens_total",
                "Live committed tokens; prefill compute includes replay after retraction.",
                value as f64,
                &[("mode", mode)],
            );
        }
        out.gauge(
            "num_running_reqs",
            "Admitted nonterminal requests, including in-flight work.",
            r.running_requests as f64,
            &[],
        );
        out.gauge(
            "num_queue_reqs",
            "Requests waiting for native scheduler admission.",
            r.waiting_requests as f64,
            &[],
        );
        out.counter(
            "scheduler_idle_seconds_total",
            "Wall time with no native engine work; distinct from GPU idle participation.",
            r.scheduler_idle_seconds,
            &[],
        );
        for (name, help, histogram) in [
            (
                "prompt_tokens_histogram",
                "Input length of completed requests.",
                &r.prompt_tokens_histogram,
            ),
            (
                "uncached_prompt_tokens_histogram",
                "Uncached input length of completed requests.",
                &r.uncached_prompt_tokens_histogram,
            ),
            (
                "generation_tokens_histogram",
                "Output length of completed requests.",
                &r.generation_tokens_histogram,
            ),
            (
                "queue_time_seconds",
                "Submission to initial scheduler admission.",
                &r.queue_time_seconds,
            ),
            (
                "time_to_first_token_seconds",
                "Native submission to first committed output token.",
                &r.time_to_first_token_seconds,
            ),
            (
                "inter_token_latency_seconds",
                "Inter-commit interval per output token, weighted by tokens.",
                &r.inter_token_latency_seconds,
            ),
            (
                "request_time_per_output_token_seconds",
                "Mean inter-token latency per completed request with at least two tokens.",
                &r.request_time_per_output_token_seconds,
            ),
            (
                "e2e_request_latency_seconds",
                "Native submission to successful completion; excludes HTTP rendering.",
                &r.e2e_request_latency_seconds,
            ),
        ] {
            out.histogram(name, help, histogram, &[]);
        }
        if let Some(cache) = &self.cache {
            out.cache(cache);
        }
        if include_runner && let Some(runner) = &self.runner {
            out.runner(runner.bind(py))?;
        }
        out.encode()
    }
}

struct Exposition {
    labels: Vec<proto::LabelPair>,
    families: BTreeMap<String, proto::MetricFamily>,
}

impl Exposition {
    fn new(model: &str, rank: u32, dp: u32) -> Self {
        Self {
            labels: [
                ("model_name", model.to_owned()),
                ("engine_type", "resin".into()),
                ("rank", rank.to_string()),
                ("attn_dp_rank", dp.to_string()),
            ]
            .into_iter()
            .map(|(name, value)| label(name, &value))
            .collect(),
            families: BTreeMap::new(),
        }
    }
    fn add(
        &mut self,
        name: &str,
        help: &str,
        kind: proto::MetricType,
        mut metric: proto::Metric,
        labels: &[(&str, &str)],
    ) {
        metric.set_label(
            self.labels
                .iter()
                .cloned()
                .chain(labels.iter().map(|(name, value)| label(name, value)))
                .collect(),
        );
        let family = self.families.entry(name.into()).or_insert_with(|| {
            let mut f = proto::MetricFamily::default();
            f.set_name(format!("resin:{name}"));
            f.set_help(help.into());
            f.set_field_type(kind);
            f
        });
        family.mut_metric().push(metric);
    }
    fn counter(&mut self, name: &str, help: &str, value: f64, labels: &[(&str, &str)]) {
        let mut counter = proto::Counter::default();
        counter.set_value(value);
        let mut metric = proto::Metric::default();
        metric.set_counter(counter);
        self.add(name, help, proto::MetricType::COUNTER, metric, labels);
    }
    fn gauge(&mut self, name: &str, help: &str, value: f64, labels: &[(&str, &str)]) {
        let mut gauge = proto::Gauge::default();
        gauge.set_value(value);
        let mut metric = proto::Metric::default();
        metric.set_gauge(gauge);
        self.add(name, help, proto::MetricType::GAUGE, metric, labels);
    }
    fn histogram(&mut self, name: &str, help: &str, value: &Histogram, labels: &[(&str, &str)]) {
        let mut histogram = proto::Histogram::default();
        histogram.set_sample_count(value.count);
        histogram.set_sample_sum(value.sum);
        histogram.set_bucket(
            value
                .bounds
                .iter()
                .zip(&value.counts)
                .map(|(&bound, &count)| {
                    let mut b = proto::Bucket::default();
                    b.set_upper_bound(bound);
                    b.set_cumulative_count(count);
                    b
                })
                .collect(),
        );
        let mut metric = proto::Metric::default();
        metric.set_histogram(histogram);
        self.add(name, help, proto::MetricType::HISTOGRAM, metric, labels);
    }
    fn cache(&mut self, c: &CacheMetrics) {
        for (state, value) in [
            ("capacity", c.capacity_slabs),
            ("free", c.free_slabs),
            ("occupied", c.occupied_slabs),
            ("evictable", c.reclaimable_slabs),
        ] {
            self.gauge(
                "cache_slabs",
                "Physical slabs in the shared pool; evictable is a subset of occupied.",
                value.into(),
                &[("state", state)],
            );
        }
        self.gauge(
            "token_usage",
            "Fraction of shared physical slabs occupied, including evictable prefixes.",
            if c.capacity_slabs == 0 {
                0.0
            } else {
                f64::from(c.occupied_slabs) / f64::from(c.capacity_slabs)
            },
            &[],
        );
        for g in &c.groups {
            let group = g.group.to_string();
            for (state, value) in [
                ("occupied", g.occupied_blocks),
                ("cached", g.cached_blocks),
                ("evictable", g.reclaimable_blocks),
                ("pinned", g.pinned_blocks),
                ("free", g.free_blocks),
            ] {
                self.gauge(
                    "cache_group_blocks",
                    "Physical block inventory; cached, evictable and pinned overlap occupied.",
                    value as f64,
                    &[("pool", g.pool), ("group", &group), ("state", state)],
                );
            }
            for (state, value) in [
                ("occupied", g.occupied_tokens),
                ("evictable", g.reclaimable_tokens),
            ] {
                if let Some(value) = value {
                    self.gauge("cache_group_tokens", "Block-rounded history token capacity; recurrent checkpoints use blocks instead.",value as f64,&[("pool",g.pool),("group",&group),("state",state)]);
                }
            }
            self.gauge(
                "cache_group_slabs",
                "Slabs assigned to a cache group in the shared pool.",
                g.occupied_slabs.into(),
                &[("pool", g.pool), ("group", &group)],
            );
        }
        if let Some(p) = &c.prefix {
            self.gauge(
                "prefix_cache_entries",
                "Entries in the prefix LRU.",
                p.entries as f64,
                &[],
            );
            for (name, help, value) in [
                ("prefix_cache_hits_total", "Prefix lookup hits.", p.hits),
                (
                    "prefix_cache_misses_total",
                    "Prefix lookup misses.",
                    p.misses,
                ),
                (
                    "prefix_cache_evictions_total",
                    "LRU entries evicted, including explicit clearing.",
                    p.evictions,
                ),
                (
                    "prefix_cache_reclaimed_blocks_total",
                    "Blocks actually freed during LRU eviction.",
                    p.reclaimed_blocks,
                ),
                (
                    "prefix_cache_reclaimed_slabs_total",
                    "Slabs actually freed during LRU eviction.",
                    p.reclaimed_slabs,
                ),
            ] {
                self.counter(name, help, value as f64, &[]);
            }
        }
    }
    fn runner(&mut self, r: &Bound<'_, PyAny>) -> PyResult<()> {
        let times: BTreeMap<String, f64> = r.getattr("forward_seconds")?.extract()?;
        for (category, value) in times {
            self.counter(
                "forward_execution_seconds_total",
                "Nonoverlapping GPU execution intervals; host-side collection never synchronizes.",
                value,
                &[("category", &category)],
            );
        }
        let graphs: BTreeMap<String, u64> = r.getattr("cuda_graph_passes")?.extract()?;
        for (mode, value) in graphs {
            self.counter(
                "cuda_graph_passes_total",
                "Top-level forward executions by graph outcome, not graph region count.",
                value as f64,
                &[("mode", &mode)],
            );
        }
        let drafts: u64 = r.getattr("speculative_drafts")?.extract()?;
        let accepted: u64 = r.getattr("speculative_accepted")?.extract()?;
        let rows: u64 = r.getattr("speculative_rows")?.extract()?;
        for (name, help, value) in [
            (
                "spec_draft_tokens_total",
                "Draft tokens proposed before request stop clipping.",
                drafts,
            ),
            (
                "spec_accepted_tokens_total",
                "Accepted draft tokens before request stop clipping.",
                accepted,
            ),
            (
                "spec_verify_rows_total",
                "Verified speculative request rows.",
                rows,
            ),
        ] {
            self.counter(name, help, value as f64, &[]);
        }
        if drafts > 0 {
            self.gauge(
                "spec_accept_rate",
                "Cumulative accepted draft fraction; use raw counters for interval rates.",
                accepted as f64 / drafts as f64,
                &[],
            );
        }
        if rows > 0 {
            self.gauge("spec_accept_length","Cumulative committed speculative length including one target token per row, before stop clipping.",(accepted+rows) as f64/rows as f64,&[]);
        }
        let wait = Histogram {
            bounds: r.getattr("dp_sync_bounds")?.extract()?,
            counts: r.getattr("dp_sync_buckets")?.extract()?,
            count: r.getattr("dp_sync_count")?.extract()?,
            sum: r.getattr("dp_sync_seconds")?.extract()?,
        };
        if wait.bounds.len() != wait.counts.len() {
            return Err(PyValueError::new_err(
                "runner DP histogram bucket count is invalid",
            ));
        }
        self.histogram("dp_attention_sync_wait_seconds","Blocking host coordination time, including collective latency; can overlap GPU execution.",&wait,&[]);
        Ok(())
    }
    fn encode(self) -> PyResult<String> {
        TextEncoder::new()
            .encode_to_string(&self.families.into_values().collect::<Vec<_>>())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
fn label(name: &str, value: &str) -> proto::LabelPair {
    let mut l = proto::LabelPair::default();
    l.set_name(name.into());
    l.set_value(value.into());
    l
}

/// Encode physical-rank observations separately from logical request ownership.
#[pyfunction]
#[pyo3(signature = (snapshot, *, model_name, rank, attn_dp_rank))]
pub(crate) fn _runner_metrics_prometheus(
    snapshot: &Bound<'_, PyAny>,
    model_name: &str,
    rank: u32,
    attn_dp_rank: u32,
) -> PyResult<String> {
    let mut out = Exposition::new(model_name, rank, attn_dp_rank);
    out.runner(snapshot)?;
    out.encode()
}
