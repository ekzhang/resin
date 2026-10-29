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

/// An immutable engine snapshot with request, cache, and runner observations.
/// Request and cache properties return independent dictionary views.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
pub(crate) struct EngineMetrics {
    requests: EngineMetricsSnapshot,
    cache: Option<paged_kv::CacheMetrics>,
    runner: Option<RunnerMetricsSnapshot>,
}

#[pymethods]
impl EngineMetrics {
    #[getter]
    fn requests<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize(py, &self.requests)?)
    }

    #[getter]
    fn cache<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize(py, &self.cache)?)
    }

    #[getter]
    fn runner(&self) -> Option<RunnerMetrics> {
        self.runner.clone().map(RunnerMetrics)
    }
}

impl Engine {
    pub(crate) fn collect_metrics(
        &self,
        py: Python<'_>,
        include_runner: bool,
    ) -> PyResult<Option<EngineMetrics>> {
        let Some(requests) = self.core.metrics_snapshot() else {
            return Ok(None);
        };
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
            .engine(&self.requests, self.cache.as_ref())
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
    #[pyo3(signature = (*, forward_execution_seconds = None, cuda_graph_passes = None,
        spec_draft_tokens = 0, spec_accepted_tokens = 0, spec_verify_rows = 0,
        dp_attention_sync_wait_seconds: "dict[str, Any] | None" = None))]
    fn new(
        forward_execution_seconds: Option<BTreeMap<String, f64>>,
        cuda_graph_passes: Option<BTreeMap<String, u64>>,
        spec_draft_tokens: u64,
        spec_accepted_tokens: u64,
        spec_verify_rows: u64,
        dp_attention_sync_wait_seconds: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self(RunnerMetricsSnapshot {
            forward_execution_seconds: forward_execution_seconds.unwrap_or_default(),
            cuda_graph_passes: cuda_graph_passes.unwrap_or_default(),
            spec_draft_tokens,
            spec_accepted_tokens,
            spec_verify_rows,
            dp_attention_sync_wait_seconds: dp_attention_sync_wait_seconds
                .map(histogram)
                .transpose()?,
        }))
    }

    #[getter]
    fn forward_execution_seconds(&self) -> &BTreeMap<String, f64> {
        &self.0.forward_execution_seconds
    }

    #[getter]
    fn cuda_graph_passes(&self) -> &BTreeMap<String, u64> {
        &self.0.cuda_graph_passes
    }

    #[getter]
    fn spec_draft_tokens(&self) -> u64 {
        self.0.spec_draft_tokens
    }

    #[getter]
    fn spec_accepted_tokens(&self) -> u64 {
        self.0.spec_accepted_tokens
    }

    #[getter]
    fn spec_verify_rows(&self) -> u64 {
        self.0.spec_verify_rows
    }

    #[getter]
    fn dp_attention_sync_wait_seconds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize(py, &self.0.dp_attention_sync_wait_seconds)?)
    }

    /// Plain fields for inspection and explicit distributed transport.
    #[pyo3(signature = () -> "dict[str, Any]")]
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
            .dp_attention_sync_wait_seconds
            .get_or_insert_with(Histogram::default)
            .merge(&self.0);
        result
    }

    #[pyo3(signature = () -> "dict[str, Any]")]
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(pythonize(py, &self.0)?)
    }
}
