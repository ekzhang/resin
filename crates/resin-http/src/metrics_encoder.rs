//! Prometheus names, descriptions, and encoding for exporter-independent snapshots.

use prometheus_client::encoding::prometheus_protobuf::prometheus_data_model as proto;
use resin_core::{
    metrics::{EngineMetricsSnapshot, Histogram, RunnerMetricsSnapshot},
    resources::paged_kv::CacheMetrics,
};
use std::{collections::BTreeMap, fmt};

/// A histogram count cannot be represented by Prometheus's signed bucket deltas.
#[derive(Debug)]
pub struct MetricsEncodeError;

impl fmt::Display for MetricsEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("histogram bucket count exceeds Prometheus integer range")
    }
}
impl std::error::Error for MetricsEncodeError {}

/// Build the metric families for one logical engine or physical rank.
pub struct MetricsEncoder {
    labels: Vec<proto::LabelPair>,
    families: BTreeMap<String, proto::MetricFamily>,
}

impl MetricsEncoder {
    pub fn new(model: &str, rank: u32, dp: u32) -> Self {
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
    /// Map logical-engine observations to the public Resin metric schema.
    pub fn engine(
        &mut self,
        r: &EngineMetricsSnapshot,
        cache: Option<&CacheMetrics>,
    ) -> Result<(), MetricsEncodeError> {
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
            self.counter(name, help, value as f64, &[]);
        }
        self.counter(
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
            self.counter(
                "realtime_tokens_total",
                "Live committed tokens; prefill compute includes replay after retraction.",
                value as f64,
                &[("mode", mode)],
            );
        }
        self.gauge(
            "num_running_reqs",
            "Admitted nonterminal requests, including in-flight work.",
            r.running_requests as f64,
            &[],
        );
        self.gauge(
            "num_queue_reqs",
            "Requests waiting for native scheduler admission.",
            r.waiting_requests as f64,
            &[],
        );
        self.counter(
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
            self.histogram(name, help, histogram, &[])?;
        }
        if let Some(cache) = cache {
            self.cache(cache);
        }
        Ok(())
    }

    pub fn finish(self) -> Vec<proto::MetricFamily> {
        self.families.into_values().collect()
    }

    fn add(
        &mut self,
        name: &str,
        help: &str,
        kind: proto::MetricType,
        mut metric: proto::Metric,
        labels: &[(&str, &str)],
    ) {
        metric.label = self
            .labels
            .iter()
            .cloned()
            .chain(labels.iter().map(|(name, value)| label(name, value)))
            .collect();
        let family = self
            .families
            .entry(name.into())
            .or_insert_with(|| proto::MetricFamily {
                name: format!("resin:{name}"),
                help: help.into(),
                r#type: kind as i32,
                ..Default::default()
            });
        family.metric.push(metric);
    }
    fn counter(&mut self, name: &str, help: &str, value: f64, labels: &[(&str, &str)]) {
        let metric = proto::Metric {
            counter: Some(proto::Counter {
                value,
                ..Default::default()
            }),
            ..Default::default()
        };
        self.add(name, help, proto::MetricType::Counter, metric, labels);
    }
    fn gauge(&mut self, name: &str, help: &str, value: f64, labels: &[(&str, &str)]) {
        let metric = proto::Metric {
            gauge: Some(proto::Gauge { value }),
            ..Default::default()
        };
        self.add(name, help, proto::MetricType::Gauge, metric, labels);
    }
    fn histogram(
        &mut self,
        name: &str,
        help: &str,
        value: &Histogram,
        labels: &[(&str, &str)],
    ) -> Result<(), MetricsEncodeError> {
        let mut previous = 0;
        let deltas = value
            .counts
            .iter()
            .map(|&count| {
                let count = i64::try_from(count).map_err(|_| MetricsEncodeError)?;
                let delta = count - previous;
                previous = count;
                Ok(delta)
            })
            .collect::<Result<Vec<_>, MetricsEncodeError>>()?;
        let histogram = proto::Histogram {
            sample_count: value.count,
            sample_sum: value.sum,
            schema: i32::from(value.schema),
            zero_count: value.zero_count,
            // A zero-length span identifies an empty native histogram on the wire.
            positive_span: vec![proto::BucketSpan {
                offset: if value.counts.is_empty() {
                    0
                } else {
                    value.offset
                },
                length: value.counts.len() as u32,
            }],
            positive_delta: deltas,
            ..Default::default()
        };
        self.add(
            name,
            help,
            proto::MetricType::Histogram,
            proto::Metric {
                histogram: Some(histogram),
                ..Default::default()
            },
            labels,
        );
        Ok(())
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
    pub fn runner(&mut self, r: &RunnerMetricsSnapshot) -> Result<(), MetricsEncodeError> {
        for (category, &value) in &r.forward_execution_seconds {
            self.counter(
                "forward_execution_seconds_total",
                "Nonoverlapping GPU execution intervals; host-side collection never synchronizes.",
                value,
                &[("category", category)],
            );
        }
        for (mode, &value) in &r.cuda_graph_passes {
            self.counter(
                "cuda_graph_passes_total",
                "Top-level forward executions by graph outcome, not graph region count.",
                value as f64,
                &[("mode", mode)],
            );
        }
        let drafts = r.spec_draft_tokens;
        let accepted = r.spec_accepted_tokens;
        let rows = r.spec_verify_rows;
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
        let empty = Histogram::default();
        let wait = r.dp_attention_sync_wait_seconds.as_ref().unwrap_or(&empty);
        self.histogram("dp_attention_sync_wait_seconds","Blocking host coordination time, including collective latency; can overlap GPU execution.",wait,&[])?;
        Ok(())
    }
}
fn label(name: &str, value: &str) -> proto::LabelPair {
    proto::LabelPair {
        name: name.into(),
        value: value.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_wire_preserves_weighted_zero_and_downscaled_observations() {
        let mut values = resin_core::metrics::Histogram::default();
        values.observe_weighted(0.0, 2);
        values.observe_weighted(0.000_001, 3);
        values.observe_weighted(1_000.0, 5);
        assert!(values.schema < 4);
        let expected = values.clone();
        let mut out = MetricsEncoder::new("test", 3, 1);
        out.histogram("latency_seconds", "Latency.", &values, &[])
            .unwrap();
        let wire = crate::encode_metrics(&out.families.into_values().collect::<Vec<_>>());
        let families = crate::decode_metrics(&wire).unwrap();
        let metric = &families[0].metric[0];
        assert!(metric.label.contains(&label("rank", "3")));
        let histogram = metric.histogram.as_ref().unwrap();
        assert_eq!(histogram.sample_count, 10);
        assert_eq!(histogram.sample_sum, expected.sum);
        assert_eq!(histogram.zero_count, 2);
        assert_eq!(histogram.schema, i32::from(expected.schema));
        assert!(histogram.bucket.is_empty());
        let span = &histogram.positive_span[0];
        assert_eq!(span.offset, expected.offset);
        assert_eq!(span.length as usize, expected.counts.len());
        let counts: Vec<u64> = histogram
            .positive_delta
            .iter()
            .scan(0_i64, |previous, delta| {
                *previous += delta;
                Some(*previous as u64)
            })
            .collect();
        assert_eq!(counts, expected.counts);
    }
}
