//! Thin Python inspection and runner hooks over native core observations.

use std::collections::BTreeMap;

use crate::Engine;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use pythonize::{depythonize, pythonize};
use resin_core::{
    metrics::{EngineMetricsSnapshot, Histogram, RunnerMetricsSnapshot},
    resources::paged_kv,
};
use resin_http::MetricsEncoder;

/// Assemble core snapshots without duplicating their field definitions.
#[derive(serde::Serialize)]
pub(crate) struct EngineMetrics {
    requests: EngineMetricsSnapshot,
    generation_throughput: f64,
    cache: Option<paged_kv::CacheMetrics>,
    runner: Option<RunnerMetricsSnapshot>,
}

impl Engine {
    pub(crate) fn collect_metrics(
        &mut self,
        py: Python<'_>,
        include_runner: bool,
    ) -> PyResult<Option<EngineMetrics>> {
        let Some(requests) = self.core.metrics_snapshot() else {
            return Ok(None);
        };
        let elapsed = self.metrics_sample.0.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            self.metrics_sample = (
                std::time::Instant::now(),
                requests.decode_tokens,
                (requests.decode_tokens - self.metrics_sample.1) as f64 / elapsed,
            );
        }
        let runner = if include_runner {
            self.runner_metrics
                .as_ref()
                .map(|hook| -> PyResult<_> {
                    Ok(hook
                        .bind(py)
                        .call0()?
                        .extract::<PyRef<'_, RunnerMetrics>>()?
                        .0
                        .clone())
                })
                .transpose()?
        } else {
            None
        };
        Ok(Some(EngineMetrics {
            generation_throughput: self.metrics_sample.2,
            requests,
            cache: self.core.cache_metrics(),
            runner,
        }))
    }
}

impl EngineMetrics {
    pub(crate) fn encode_protobuf(
        &self,
        model_name: &str,
        rank: u32,
        attn_dp_rank: u32,
    ) -> PyResult<Vec<u8>> {
        let mut encoder = MetricsEncoder::new(model_name, rank, attn_dp_rank);
        encoder
            .engine(
                &self.requests,
                self.generation_throughput,
                self.cache.as_ref(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if let Some(runner) = &self.runner {
            encoder
                .runner(runner)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        Ok(resin_http::encode_metrics(&encoder.finish()))
    }
}

fn histogram(value: &Bound<'_, PyAny>) -> PyResult<Histogram> {
    let histogram: Histogram = depythonize(value)?;
    histogram.validate().map_err(PyValueError::new_err)?;
    Ok(histogram)
}

/// A nonblocking snapshot of cumulative observations for one physical rank.
///
/// Forward categories are exclusive: prefill, decode, target_verify, draft,
/// and idle. Graph counts describe executions, not regions; DP wait uses a
/// bounded exponential histogram. The core snapshot is stored directly.
#[pyclass(frozen, eq, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, PartialEq)]
pub(crate) struct RunnerMetrics(RunnerMetricsSnapshot);

#[pymethods]
impl RunnerMetrics {
    #[new]
    #[pyo3(signature = (*, forward_seconds = None, cuda_graph_passes = None,
        speculative_drafts = 0, speculative_accepted = 0, speculative_rows = 0, dp_sync = None))]
    fn new(
        forward_seconds: Option<BTreeMap<String, f64>>,
        cuda_graph_passes: Option<BTreeMap<String, u64>>,
        speculative_drafts: u64,
        speculative_accepted: u64,
        speculative_rows: u64,
        dp_sync: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self(RunnerMetricsSnapshot {
            forward_seconds: forward_seconds.unwrap_or_default(),
            cuda_graph_passes: cuda_graph_passes.unwrap_or_default(),
            speculative_drafts,
            speculative_accepted,
            speculative_rows,
            dp_sync: dp_sync.map(histogram).transpose()?,
        }))
    }

    #[getter]
    fn forward_seconds(&self) -> &BTreeMap<String, f64> {
        &self.0.forward_seconds
    }

    #[getter]
    fn cuda_graph_passes(&self) -> &BTreeMap<String, u64> {
        &self.0.cuda_graph_passes
    }

    #[getter]
    fn speculative_drafts(&self) -> u64 {
        self.0.speculative_drafts
    }

    #[getter]
    fn speculative_accepted(&self) -> u64 {
        self.0.speculative_accepted
    }

    #[getter]
    fn speculative_rows(&self) -> u64 {
        self.0.speculative_rows
    }

    #[getter]
    fn dp_sync<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize(py, &self.0.dp_sync)?)
    }

    /// Plain fields for inspection and explicit distributed transport.
    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize(py, &self.0)?)
    }
}

/// Encode physical-rank observations separately from logical request ownership.
#[pyfunction]
#[pyo3(signature = (snapshot, *, model_name, rank, attn_dp_rank))]
pub(crate) fn _runner_metrics_prometheus_protobuf<'py>(
    py: Python<'py>,
    snapshot: &RunnerMetrics,
    model_name: &str,
    rank: u32,
    attn_dp_rank: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut encoder = MetricsEncoder::new(model_name, rank, attn_dp_rank);
    encoder
        .runner(&snapshot.0)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(
        py,
        &resin_http::encode_metrics(&encoder.finish()),
    ))
}

/// Native accumulator; snapshots are plain Python dictionaries.
#[pyclass(name = "_HistogramAccumulator", module = "resin._resin")]
#[derive(Default)]
pub(crate) struct HistogramAccumulator(Histogram);

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
        self.0.observe_weighted(value, weight);
        Ok(())
    }

    /// Return a snapshot with this coordinator's wait observations included.
    fn attach(&self, snapshot: &RunnerMetrics) -> RunnerMetrics {
        let mut result = snapshot.clone();
        result
            .0
            .dp_sync
            .get_or_insert_with(Histogram::default)
            .merge(&self.0);
        result
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize(py, &self.0)?)
    }
}
