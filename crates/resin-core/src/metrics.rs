//! Exporter-independent observations at committed engine lifecycle boundaries.

use std::{collections::BTreeMap, time::Instant};

mod histogram;
pub use histogram::Histogram;

/// Cumulative physical-rank observations supplied by an optional runner hook.
/// Collecting these values never requires a metric exporter in the runner.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct RunnerMetricsSnapshot {
    pub forward_execution_seconds: BTreeMap<String, f64>,
    pub cuda_graph_passes: BTreeMap<String, u64>,
    pub spec_draft_tokens: u64,
    pub spec_accepted_tokens: u64,
    pub spec_verify_rows: u64,
    pub dp_attention_sync_wait_seconds: Option<Histogram>,
}

/// Logical request metrics, counted once by the engine owning each request.
///
/// Completed-request counters exclude cancellations. Live counters include all
/// successfully committed work, including work for subsequently cancelled
/// requests. Prefill compute includes replay after memory retraction.
#[derive(Clone, Debug, serde::Serialize)]
pub struct EngineMetricsSnapshot {
    pub accepted_requests: u64,
    pub completed_requests: u64,
    pub aborted_requests: u64,
    pub retracted_requests: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub generation_tokens: u64,
    pub prefill_compute_tokens: u64,
    pub prefill_cache_tokens: u64,
    pub decode_tokens: u64,
    pub waiting_requests: usize,
    pub running_requests: usize,
    pub elapsed_seconds: f64,
    /// Wall time with no queued requests or outstanding physical execution.
    pub scheduler_idle_seconds: f64,
    pub prompt_tokens_histogram: Histogram,
    pub uncached_prompt_tokens_histogram: Histogram,
    pub generation_tokens_histogram: Histogram,
    /// One observation for each request's initial successful admission.
    pub queue_time_seconds: Histogram,
    pub time_to_first_token_seconds: Histogram,
    /// Inter-batch interval divided by committed tokens, weighted by that count.
    pub inter_token_latency_seconds: Histogram,
    /// One observation per completed request with at least two output tokens.
    pub request_time_per_output_token_seconds: Histogram,
    pub e2e_request_latency_seconds: Histogram,
}

impl Default for EngineMetricsSnapshot {
    fn default() -> Self {
        Self {
            accepted_requests: 0,
            completed_requests: 0,
            aborted_requests: 0,
            retracted_requests: 0,
            prompt_tokens: 0,
            cached_tokens: 0,
            generation_tokens: 0,
            prefill_compute_tokens: 0,
            prefill_cache_tokens: 0,
            decode_tokens: 0,
            waiting_requests: 0,
            running_requests: 0,
            elapsed_seconds: 0.0,
            scheduler_idle_seconds: 0.0,
            prompt_tokens_histogram: Histogram::default(),
            uncached_prompt_tokens_histogram: Histogram::default(),
            generation_tokens_histogram: Histogram::default(),
            queue_time_seconds: Histogram::default(),
            time_to_first_token_seconds: Histogram::default(),
            inter_token_latency_seconds: Histogram::default(),
            request_time_per_output_token_seconds: Histogram::default(),
            e2e_request_latency_seconds: Histogram::default(),
        }
    }
}

/// Request-local timestamps and counts retained until retirement.
#[derive(Debug)]
pub(crate) struct RequestMetricsState {
    submitted_at: Instant,
    queued_at: Option<Instant>,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    prompt_tokens: u32,
    cached_tokens: u32,
    output_tokens: u64,
}

/// Mutable engine accounting that produces owned metrics snapshots.
pub(crate) struct EngineMetricsState {
    started_at: Instant,
    stopped_at: Option<Instant>,
    idle_since: Option<Instant>,
    values: EngineMetricsSnapshot,
}

impl EngineMetricsState {
    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self {
            started_at: now,
            stopped_at: None,
            idle_since: Some(now),
            values: EngineMetricsSnapshot::default(),
        }
    }

    pub(crate) fn snapshot(&self) -> EngineMetricsSnapshot {
        let now = self.stopped_at.unwrap_or_else(Instant::now);
        let mut snapshot = self.values.clone();
        snapshot.elapsed_seconds = now.duration_since(self.started_at).as_secs_f64();
        if let Some(idle_since) = self.idle_since {
            snapshot.scheduler_idle_seconds += now.duration_since(idle_since).as_secs_f64();
        }
        snapshot
    }

    pub(crate) fn set_idle(&mut self, idle: bool, now: Instant) {
        if idle {
            self.idle_since.get_or_insert(now);
        } else if let Some(since) = self.idle_since.take() {
            self.values.scheduler_idle_seconds += now.duration_since(since).as_secs_f64();
        }
    }

    pub(crate) fn close(&mut self) {
        let now = Instant::now();
        self.set_idle(false, now);
        self.stopped_at = Some(now);
    }

    pub(crate) fn submit(
        &mut self,
        prompt_tokens: u32,
        cached_tokens: u32,
        now: Instant,
    ) -> RequestMetricsState {
        self.set_idle(false, now);
        self.values.accepted_requests += 1;
        self.values.prefill_cache_tokens += u64::from(cached_tokens);
        RequestMetricsState {
            submitted_at: now,
            queued_at: Some(now),
            first_token_at: None,
            last_token_at: None,
            prompt_tokens,
            cached_tokens,
            output_tokens: 0,
        }
    }

    pub(crate) fn commit(
        &mut self,
        request: &mut RequestMetricsState,
        prepared_at: Instant,
        now: Instant,
        prefill_tokens: u32,
        output_tokens: usize,
        finished: bool,
    ) {
        let values = &mut self.values;
        if let Some(queued_at) = request.queued_at.take() {
            values
                .queue_time_seconds
                .observe(prepared_at.duration_since(queued_at).as_secs_f64());
        }
        values.prefill_compute_tokens += u64::from(prefill_tokens);
        values.decode_tokens += output_tokens as u64;
        request.output_tokens += output_tokens as u64;
        if output_tokens > 0 {
            if let Some(previous) = request.last_token_at {
                let elapsed = now.duration_since(previous).as_secs_f64();
                values
                    .inter_token_latency_seconds
                    .observe_weighted(elapsed / output_tokens as f64, output_tokens as u64);
            } else {
                request.first_token_at = Some(now);
                values
                    .time_to_first_token_seconds
                    .observe(now.duration_since(request.submitted_at).as_secs_f64());
            }
            request.last_token_at = Some(now);
        }
        if finished {
            values.completed_requests += 1;
            values.prompt_tokens += u64::from(request.prompt_tokens);
            values.cached_tokens += u64::from(request.cached_tokens);
            values.generation_tokens += request.output_tokens;
            values
                .prompt_tokens_histogram
                .observe(f64::from(request.prompt_tokens));
            values
                .uncached_prompt_tokens_histogram
                .observe(f64::from(request.prompt_tokens - request.cached_tokens));
            values
                .generation_tokens_histogram
                .observe(request.output_tokens as f64);
            values
                .e2e_request_latency_seconds
                .observe(now.duration_since(request.submitted_at).as_secs_f64());
            if request.output_tokens > 1 {
                let first = request
                    .first_token_at
                    .expect("output tokens have a timestamp");
                values.request_time_per_output_token_seconds.observe(
                    now.duration_since(first).as_secs_f64() / (request.output_tokens - 1) as f64,
                );
            }
        }
    }

    pub(crate) fn retract(&mut self) {
        self.values.retracted_requests += 1;
    }

    pub(crate) fn abort(&mut self) {
        self.values.aborted_requests += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn token_weighted_intervals_and_request_tpot_have_distinct_denominators() {
        let mut metrics = EngineMetricsState::new();
        let start = metrics.started_at;
        let mut request = metrics.submit(12, 8, start);
        for (prepared, completed, prefill, output, finished) in
            [(20, 100, 4, 1, false), (100, 160, 0, 3, true)]
        {
            metrics.commit(
                &mut request,
                start + Duration::from_millis(prepared),
                start + Duration::from_millis(completed),
                prefill,
                output,
                finished,
            );
        }
        let values = metrics.snapshot();
        assert_eq!(values.inter_token_latency_seconds.count, 3);
        assert_eq!(values.inter_token_latency_seconds.sum, 0.06);
        assert_eq!(values.request_time_per_output_token_seconds.count, 1);
        assert_eq!(values.request_time_per_output_token_seconds.sum, 0.02);
        assert_eq!(values.time_to_first_token_seconds.sum, 0.1);
        assert_eq!(values.queue_time_seconds.sum, 0.02);
        assert_eq!(values.e2e_request_latency_seconds.sum, 0.16);
        metrics.stopped_at = Some(start + Duration::from_secs(2));
        let stopped = metrics.snapshot();
        assert_eq!(stopped.elapsed_seconds, 2.0);
    }
}
