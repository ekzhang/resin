//! Read-only snapshots; collection and exporter policy stay outside these types.

use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use resin_core::metrics::{self, EngineMetricsSnapshot};

/// Immutable exponential histogram. Bucket counts are noncumulative;
/// `offset` is the Prometheus bucket index of the first positive bucket.
#[pyclass(frozen, get_all, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct Histogram {
    pub(crate) schema: i8,
    pub(crate) zero_count: u64,
    pub(crate) offset: i32,
    pub(crate) counts: Vec<u64>,
    pub(crate) count: u64,
    pub(crate) sum: f64,
}

impl Histogram {
    pub(crate) fn native(&self) -> metrics::Histogram {
        metrics::Histogram {
            schema: self.schema,
            zero_count: self.zero_count,
            offset: self.offset,
            counts: self.counts.clone(),
            count: self.count,
            sum: self.sum,
        }
    }
}

#[pymethods]
impl Histogram {
    #[new]
    #[pyo3(signature = (*, schema = 4, zero_count = 0, offset = 0, counts = None, count = 0, sum = 0.0))]
    fn new(
        schema: i8,
        zero_count: u64,
        offset: i32,
        counts: Option<Vec<u64>>,
        count: u64,
        sum: f64,
    ) -> PyResult<Self> {
        let counts = counts.unwrap_or_default();
        if !(-4..=4).contains(&schema) || !sum.is_finite() || sum < 0.0 {
            return Err(PyValueError::new_err(
                "histogram requires schema -4..4 and a finite nonnegative sum",
            ));
        }
        if counts.len() > 160 {
            return Err(PyValueError::new_err("histogram exceeds 160 buckets"));
        }
        if counts
            .iter()
            .try_fold(zero_count, |total, value| total.checked_add(*value))
            != Some(count)
        {
            return Err(PyValueError::new_err(
                "histogram count must equal zero_count plus bucket counts",
            ));
        }
        let scale = 2.0_f64.powi(i32::from(schema));
        let minimum = (-1074.0 * scale).ceil() as i32;
        let maximum = (1024.0 * scale).ceil() as i32;
        let valid_indices = if counts.is_empty() {
            offset == 0
        } else {
            offset >= minimum
                && offset
                    .checked_add(counts.len() as i32 - 1)
                    .is_some_and(|last| last <= maximum)
        };
        if !valid_indices {
            return Err(PyValueError::new_err(
                "histogram bucket indices are outside the finite positive range",
            ));
        }
        Ok(Self {
            schema,
            zero_count,
            offset,
            counts,
            count,
            sum,
        })
    }

    /// Combine cumulative distributions without mutating either snapshot.
    fn merge(&self, other: &Self) -> Self {
        let mut merged = self.native();
        merged.merge(&other.native());
        merged.into()
    }

    /// Plain numeric fields for explicit JSON transport; no Python pickling.
    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let values = PyDict::new(py);
        values.set_item("schema", self.schema)?;
        values.set_item("zero_count", self.zero_count)?;
        values.set_item("offset", self.offset)?;
        values.set_item("counts", &self.counts)?;
        values.set_item("count", self.count)?;
        values.set_item("sum", self.sum)?;
        Ok(values)
    }
}

impl From<metrics::Histogram> for Histogram {
    fn from(value: metrics::Histogram) -> Self {
        Self {
            schema: value.schema,
            zero_count: value.zero_count,
            offset: value.offset,
            counts: value.counts,
            count: value.count,
            sum: value.sum,
        }
    }
}

/// Mutable native accumulator shared by host execution observations.
#[pyclass(name = "_HistogramAccumulator", module = "resin._resin")]
#[derive(Default)]
pub(crate) struct HistogramAccumulator {
    values: metrics::Histogram,
}

#[pymethods]
impl HistogramAccumulator {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[pyo3(signature = (value, *, weight = 1))]
    fn observe(&mut self, value: f64, weight: u64) -> PyResult<()> {
        if !value.is_finite() || value < 0.0 {
            return Err(PyValueError::new_err(
                "histogram observations must be finite and nonnegative",
            ));
        }
        self.values.observe_weighted(value, weight);
        Ok(())
    }

    fn snapshot(&self) -> Histogram {
        self.values.clone().into()
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
