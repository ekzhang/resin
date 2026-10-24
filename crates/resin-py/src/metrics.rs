//! Python snapshot access and runner extraction; export policy lives in resin-http.

use crate::{
    cache_metrics::CacheMetrics,
    metrics_types::{Histogram, RequestMetrics},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use resin_core::{
    metrics::{EngineMetricsSnapshot, RunnerMetricsSnapshot},
    resources::paged_kv,
};
use resin_http::MetricsEncoder;

/// An immutable logical-engine snapshot and optional physical-runner observations.
/// Reading this snapshot never synchronizes the accelerator. Cache inventory is
/// scanned, so collect periodically rather than once per token.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
pub(crate) struct EngineMetrics {
    pub(crate) requests: EngineMetricsSnapshot,
    #[pyo3(get)]
    pub(crate) generation_throughput: f64,
    pub(crate) cache: Option<paged_kv::CacheMetrics>,
    #[pyo3(get)]
    pub(crate) runner: Option<Py<PyAny>>,
}

#[pymethods]
impl EngineMetrics {
    #[getter]
    fn requests(&self) -> RequestMetrics {
        self.requests.clone().into()
    }

    #[getter]
    fn cache(&self) -> Option<CacheMetrics> {
        self.cache.clone().map(Into::into)
    }

    /// Human-readable metrics. Histogram distributions require the protobuf export.
    #[pyo3(signature = (*, model_name, rank = 0, attn_dp_rank = 0, include_runner = true))]
    fn prometheus(
        &self,
        py: Python<'_>,
        model_name: &str,
        rank: u32,
        attn_dp_rank: u32,
        include_runner: bool,
    ) -> PyResult<String> {
        let families = self
            .encoder(py, model_name, rank, attn_dp_rank, include_runner)?
            .finish();
        resin_http::metrics_text(&families)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Length-delimited Prometheus protobuf, including native exponential histograms.
    #[pyo3(signature = (*, model_name, rank = 0, attn_dp_rank = 0, include_runner = true))]
    fn prometheus_protobuf<'py>(
        &self,
        py: Python<'py>,
        model_name: &str,
        rank: u32,
        attn_dp_rank: u32,
        include_runner: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            &self.encode_protobuf(py, model_name, rank, attn_dp_rank, include_runner)?,
        ))
    }
}

impl EngineMetrics {
    pub(crate) fn encode_protobuf(
        &self,
        py: Python<'_>,
        model_name: &str,
        rank: u32,
        attn_dp_rank: u32,
        include_runner: bool,
    ) -> PyResult<Vec<u8>> {
        let families = self
            .encoder(py, model_name, rank, attn_dp_rank, include_runner)?
            .finish();
        Ok(resin_http::encode_metrics(&families))
    }

    fn encoder(
        &self,
        py: Python<'_>,
        model_name: &str,
        rank: u32,
        attn_dp_rank: u32,
        include_runner: bool,
    ) -> PyResult<MetricsEncoder> {
        let mut encoder = MetricsEncoder::new(model_name, rank, attn_dp_rank);
        encoder
            .engine(
                &self.requests,
                self.generation_throughput,
                self.cache.as_ref(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if include_runner && let Some(runner) = &self.runner {
            encoder
                .runner(&runner_snapshot(runner.bind(py))?)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        Ok(encoder)
    }
}

fn runner_snapshot(snapshot: &Bound<'_, PyAny>) -> PyResult<RunnerMetricsSnapshot> {
    let wait = snapshot.getattr("dp_sync")?;
    Ok(RunnerMetricsSnapshot {
        forward_seconds: snapshot.getattr("forward_seconds")?.extract()?,
        cuda_graph_passes: snapshot.getattr("cuda_graph_passes")?.extract()?,
        speculative_drafts: snapshot.getattr("speculative_drafts")?.extract()?,
        speculative_accepted: snapshot.getattr("speculative_accepted")?.extract()?,
        speculative_rows: snapshot.getattr("speculative_rows")?.extract()?,
        dp_sync: if wait.is_none() {
            None
        } else {
            Some(wait.extract::<PyRef<'_, Histogram>>()?.native())
        },
    })
}

/// Encode physical-rank observations separately from logical request ownership.
#[pyfunction]
#[pyo3(signature = (snapshot, *, model_name, rank, attn_dp_rank))]
pub(crate) fn _runner_metrics_prometheus_protobuf<'py>(
    snapshot: &Bound<'py, PyAny>,
    model_name: &str,
    rank: u32,
    attn_dp_rank: u32,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut encoder = MetricsEncoder::new(model_name, rank, attn_dp_rank);
    encoder
        .runner(&runner_snapshot(snapshot)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(
        snapshot.py(),
        &resin_http::encode_metrics(&encoder.finish()),
    ))
}
