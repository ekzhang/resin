//! Exporter-independent observations at committed engine lifecycle boundaries.

use std::time::Instant;

const LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    600.0,
];
const TOKEN_BUCKETS: &[f64] = &[
    1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0,
];

/// Cumulative finite buckets. The implicit positive-infinity bucket is `count`.
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    pub bounds: &'static [f64],
    pub counts: Vec<u64>,
    pub count: u64,
    pub sum: f64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            counts: vec![0; bounds.len()],
            count: 0,
            sum: 0.0,
        }
    }

    fn observe(&mut self, value: f64) {
        self.observe_weighted(value, 1);
    }

    fn observe_weighted(&mut self, value: f64, weight: u64) {
        for (bound, count) in self.bounds.iter().zip(&mut self.counts) {
            if value <= *bound {
                *count += weight;
            }
        }
        self.count += weight;
        self.sum += value * weight as f64;
    }
}

/// Logical request metrics, counted once by the engine owning each request.
///
/// Completed-request counters exclude cancellations. Live counters include all
/// successfully committed work, including work for subsequently cancelled
/// requests. Prefill compute includes replay after memory retraction.
#[derive(Clone, Debug)]
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
            prompt_tokens_histogram: Histogram::new(TOKEN_BUCKETS),
            uncached_prompt_tokens_histogram: Histogram::new(TOKEN_BUCKETS),
            generation_tokens_histogram: Histogram::new(TOKEN_BUCKETS),
            queue_time_seconds: Histogram::new(LATENCY_BUCKETS),
            time_to_first_token_seconds: Histogram::new(LATENCY_BUCKETS),
            inter_token_latency_seconds: Histogram::new(LATENCY_BUCKETS),
            request_time_per_output_token_seconds: Histogram::new(LATENCY_BUCKETS),
            e2e_request_latency_seconds: Histogram::new(LATENCY_BUCKETS),
        }
    }
}

#[derive(Debug)]
pub(crate) struct RequestMetrics {
    submitted_at: Instant,
    queued_at: Option<Instant>,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    prompt_tokens: u32,
    cached_tokens: u32,
    output_tokens: u64,
}

pub(crate) struct EngineMetrics {
    started_at: Instant,
    stopped_at: Option<Instant>,
    idle_since: Option<Instant>,
    values: EngineMetricsSnapshot,
}

impl EngineMetrics {
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
        let mut snapshot = self.values.clone();
        let now = self.stopped_at.unwrap_or_else(Instant::now);
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
    ) -> RequestMetrics {
        self.set_idle(false, now);
        self.values.accepted_requests += 1;
        self.values.prefill_cache_tokens += u64::from(cached_tokens);
        RequestMetrics {
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
        request: &mut RequestMetrics,
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
        let mut metrics = EngineMetrics::new();
        let mut request = metrics.submit(12, 8, Instant::now());
        let start = request.submitted_at;
        metrics.commit(
            &mut request,
            start + Duration::from_millis(20),
            start + Duration::from_millis(100),
            4,
            1,
            false,
        );
        metrics.commit(
            &mut request,
            start + Duration::from_millis(100),
            start + Duration::from_millis(160),
            0,
            3,
            true,
        );
        let values = metrics.snapshot();
        assert_eq!(values.inter_token_latency_seconds.count, 3);
        assert_eq!(values.inter_token_latency_seconds.sum, 0.06);
        assert_eq!(values.request_time_per_output_token_seconds.count, 1);
        assert_eq!(values.request_time_per_output_token_seconds.sum, 0.02);
        assert_eq!(values.time_to_first_token_seconds.sum, 0.1);
        assert_eq!(values.queue_time_seconds.sum, 0.02);
        assert_eq!(values.e2e_request_latency_seconds.sum, 0.16);
        assert_eq!(values.prefill_compute_tokens, 4);
        assert_eq!(values.prefill_cache_tokens, 8);
        assert_eq!(values.prompt_tokens, 12);
        assert_eq!(values.cached_tokens, 8);
        assert_eq!(values.decode_tokens, 4);
        assert_eq!(values.generation_tokens, 4);
        assert_eq!(values.uncached_prompt_tokens_histogram.sum, 4.0);
    }

    #[test]
    fn histogram_buckets_are_inclusive_and_cumulative() {
        let mut histogram = Histogram::new(&[1.0, 2.0, 4.0]);
        histogram.observe_weighted(2.0, 3);
        histogram.observe(9.0);
        assert_eq!(histogram.counts, [0, 3, 3]);
        assert_eq!(histogram.count, 4);
        assert_eq!(histogram.sum, 15.0);
    }
}
