//! Thin Python inspection and runner hooks over native core observations.

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
                .map(|hook| runner_snapshot(&hook.bind(py).call0()?))
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
            Some(histogram(&wait)?)
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

    #[pyo3(signature = (*, additional = None))]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        additional: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut snapshot = self.0.clone();
        if let Some(additional) = additional {
            snapshot.merge(&histogram(additional)?);
        }
        Ok(pythonize(py, &snapshot)?)
    }
}
