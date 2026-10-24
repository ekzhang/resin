//! Read-only snapshots; collection and exporter policy stay outside these types.

use pyo3::prelude::*;
use resin_core::metrics::{self, EngineMetricsSnapshot};

/// Cumulative finite buckets, observation count, and sum. `count` is the +Inf bucket.
#[pyclass(frozen, get_all, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct Histogram {
    pub(crate) bounds: Vec<f64>,
    pub(crate) counts: Vec<u64>,
    pub(crate) count: u64,
    pub(crate) sum: f64,
}

impl From<metrics::Histogram> for Histogram {
    fn from(value: metrics::Histogram) -> Self {
        Self {
            bounds: value.bounds.to_vec(),
            counts: value.counts,
            count: value.count,
            sum: value.sum,
        }
    }
}

/// Cumulative request observations and current scheduler state for one engine.
///
/// Completed-request totals exclude cancellations. Live token counters include
/// all committed work; prefill compute includes replay after retraction.
#[pyclass(frozen, get_all, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct RequestMetrics {
    pub(crate) accepted_requests: u64,
    pub(crate) completed_requests: u64,
    pub(crate) aborted_requests: u64,
    pub(crate) retracted_requests: u64,
    pub(crate) prompt_tokens: u64,
    pub(crate) cached_tokens: u64,
    pub(crate) generation_tokens: u64,
    pub(crate) prefill_compute_tokens: u64,
    pub(crate) prefill_cache_tokens: u64,
    pub(crate) decode_tokens: u64,
    pub(crate) waiting_requests: usize,
    pub(crate) running_requests: usize,
    pub(crate) elapsed_seconds: f64,
    pub(crate) scheduler_idle_seconds: f64,
    pub(crate) prompt_tokens_histogram: Histogram,
    pub(crate) uncached_prompt_tokens_histogram: Histogram,
    pub(crate) generation_tokens_histogram: Histogram,
    pub(crate) queue_time_seconds: Histogram,
    pub(crate) time_to_first_token_seconds: Histogram,
    pub(crate) inter_token_latency_seconds: Histogram,
    pub(crate) request_time_per_output_token_seconds: Histogram,
    pub(crate) e2e_request_latency_seconds: Histogram,
}

impl From<EngineMetricsSnapshot> for RequestMetrics {
    fn from(value: EngineMetricsSnapshot) -> Self {
        Self {
            accepted_requests: value.accepted_requests,
            completed_requests: value.completed_requests,
            aborted_requests: value.aborted_requests,
            retracted_requests: value.retracted_requests,
            prompt_tokens: value.prompt_tokens,
            cached_tokens: value.cached_tokens,
            generation_tokens: value.generation_tokens,
            prefill_compute_tokens: value.prefill_compute_tokens,
            prefill_cache_tokens: value.prefill_cache_tokens,
            decode_tokens: value.decode_tokens,
            waiting_requests: value.waiting_requests,
            running_requests: value.running_requests,
            elapsed_seconds: value.elapsed_seconds,
            scheduler_idle_seconds: value.scheduler_idle_seconds,
            prompt_tokens_histogram: value.prompt_tokens_histogram.into(),
            uncached_prompt_tokens_histogram: value.uncached_prompt_tokens_histogram.into(),
            generation_tokens_histogram: value.generation_tokens_histogram.into(),
            queue_time_seconds: value.queue_time_seconds.into(),
            time_to_first_token_seconds: value.time_to_first_token_seconds.into(),
            inter_token_latency_seconds: value.inter_token_latency_seconds.into(),
            request_time_per_output_token_seconds: value
                .request_time_per_output_token_seconds
                .into(),
            e2e_request_latency_seconds: value.e2e_request_latency_seconds.into(),
        }
    }
}
