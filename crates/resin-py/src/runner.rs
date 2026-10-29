use std::num::NonZeroU32;

use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::cache::CacheLayout;

fn positive_u32(name: &str, value: Option<u32>) -> PyResult<Option<NonZeroU32>> {
    value
        .map(|value| {
            NonZeroU32::new(value)
                .ok_or_else(|| PyValueError::new_err(format!("{name} must be positive")))
        })
        .transpose()
}

/// Frozen scheduling limits and persistent-resource declarations for a runner.
///
/// Runners without special requirements may omit `runner_config` entirely.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
pub(crate) struct RunnerConfig {
    pub(super) max_batch_tokens: Option<NonZeroU32>,
    pub(super) prefill_chunk_size: Option<NonZeroU32>,
    pub(super) continuation_prefill: bool,
    pub(super) max_accepted_per_step: u32,
    pub(super) cache_layout: Option<Py<CacheLayout>>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_batch_tokens: None,
            prefill_chunk_size: None,
            continuation_prefill: false,
            max_accepted_per_step: 1,
            cache_layout: None,
        }
    }
}

impl RunnerConfig {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            max_batch_tokens: self.max_batch_tokens,
            prefill_chunk_size: self.prefill_chunk_size,
            continuation_prefill: self.continuation_prefill,
            max_accepted_per_step: self.max_accepted_per_step,
            cache_layout: self
                .cache_layout
                .as_ref()
                .map(|layout| layout.clone_ref(py)),
        }
    }
}

#[pymethods]
impl RunnerConfig {
    #[new]
    #[pyo3(signature = (
        *,
        max_batch_tokens = None,
        prefill_chunk_size = None,
        continuation_prefill = false,
        max_accepted_per_step = 1,
        cache_layout = None
    ))]
    fn new(
        py: Python<'_>,
        max_batch_tokens: Option<u32>,
        prefill_chunk_size: Option<u32>,
        continuation_prefill: bool,
        max_accepted_per_step: u32,
        cache_layout: Option<Py<CacheLayout>>,
    ) -> PyResult<Self> {
        if max_accepted_per_step == 0 {
            return Err(PyValueError::new_err(
                "max_accepted_per_step must be positive",
            ));
        }
        let extra = cache_layout.as_ref().map_or(0, |layout| {
            layout.borrow(py).inner_native().decode_write_extra_tokens()
        });
        max_accepted_per_step
            .checked_add(extra)
            .ok_or_else(|| PyValueError::new_err("decode cache write footprint overflows u32"))?;
        Ok(Self {
            max_batch_tokens: positive_u32("max_batch_tokens", max_batch_tokens)?,
            prefill_chunk_size: positive_u32("prefill_chunk_size", prefill_chunk_size)?,
            continuation_prefill,
            max_accepted_per_step,
            cache_layout,
        })
    }

    #[getter]
    fn max_batch_tokens(&self) -> Option<u32> {
        self.max_batch_tokens.map(NonZeroU32::get)
    }

    #[getter]
    fn prefill_chunk_size(&self) -> Option<u32> {
        self.prefill_chunk_size.map(NonZeroU32::get)
    }

    #[getter]
    fn continuation_prefill(&self) -> bool {
        self.continuation_prefill
    }

    #[getter]
    fn max_accepted_per_step(&self) -> u32 {
        self.max_accepted_per_step
    }

    #[getter]
    fn cache_layout(&self, py: Python<'_>) -> Option<Py<CacheLayout>> {
        self.cache_layout
            .as_ref()
            .map(|layout| layout.clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "RunnerConfig(max_batch_tokens={:?}, prefill_chunk_size={:?}, continuation_prefill={}, max_accepted_per_step={}, cache={})",
            self.max_batch_tokens(),
            self.prefill_chunk_size(),
            self.continuation_prefill,
            self.max_accepted_per_step,
            self.cache_layout.is_some(),
        )
    }
}

/// Concrete Engine-owned resource identities passed to `bind_resources`.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
pub(crate) struct ResourceBinding {
    cache_pool_id: Option<u64>,
    cache_layout: Option<Py<CacheLayout>>,
}

#[pymethods]
impl ResourceBinding {
    #[getter]
    fn cache_pool_id(&self) -> Option<u64> {
        self.cache_pool_id
    }

    #[getter]
    fn cache_layout(&self, py: Python<'_>) -> Option<Py<CacheLayout>> {
        self.cache_layout
            .as_ref()
            .map(|layout| layout.clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!("ResourceBinding(cache_pool_id={:?})", self.cache_pool_id)
    }
}

pub(super) struct ResolvedRunnerConfig {
    pub(super) config: RunnerConfig,
    binder: Option<Py<PyAny>>,
}

pub(super) fn optional_runner_callable(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    match runner.getattr(name) {
        Ok(value) if value.is_callable() => Ok(Some(value.unbind())),
        Ok(_) => Err(PyTypeError::new_err(format!(
            "runner.{name} must be callable"
        ))),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) fn runner_error_is_post_accept(py: Python<'_>, error: &PyErr) -> bool {
    match error.value(py).getattr("_resin_post_accept") {
        Ok(marker) => marker.is(PyBool::new(py, true)),
        Err(marker_error) if marker_error.is_instance_of::<PyAttributeError>(py) => false,
        // Failure to inspect an explicitly raised exception cannot justify
        // returning native resources to allocators which may still be in use.
        Err(_) => true,
    }
}

pub(super) fn resolve_runner_contract(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyAny>, Option<Py<PyAny>>)> {
    let forward = optional_runner_callable(py, runner, "forward")?
        .ok_or_else(|| PyTypeError::new_err("runner must define callable forward"))?;
    Ok((forward, optional_runner_callable(py, runner, "release")?))
}

pub(super) fn resolve_runner_config(
    py: Python<'_>,
    runner: &Bound<'_, PyAny>,
) -> PyResult<ResolvedRunnerConfig> {
    let config = match runner.getattr("runner_config") {
        Ok(config) => config
            .cast::<RunnerConfig>()
            .map_err(|_| {
                PyTypeError::new_err("runner.runner_config must be exactly a RunnerConfig")
            })?
            .borrow()
            .clone_ref(py),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => RunnerConfig::default(),
        Err(error) => return Err(error),
    };
    let has_resources = config.cache_layout.is_some();
    let binder = has_resources
        .then(|| optional_runner_callable(py, runner, "bind_resources"))
        .transpose()?
        .flatten();
    if has_resources && binder.is_none() {
        return Err(PyTypeError::new_err(
            "runner.runner_config declares resources but runner.bind_resources is missing",
        ));
    }
    Ok(ResolvedRunnerConfig { config, binder })
}

pub(super) fn bind_runner_resources(
    py: Python<'_>,
    resolved: ResolvedRunnerConfig,
    cache_pool_id: Option<u64>,
) -> PyResult<RunnerConfig> {
    let ResolvedRunnerConfig { config, binder } = resolved;
    if config.cache_layout.is_some() != cache_pool_id.is_some() {
        return Err(PyRuntimeError::new_err(
            "runner cache declaration and native cache pool disagree",
        ));
    }
    let Some(binder) = binder else {
        debug_assert!(cache_pool_id.is_none());
        return Ok(config);
    };
    let binding = Py::new(
        py,
        ResourceBinding {
            cache_pool_id,
            cache_layout: config
                .cache_layout
                .as_ref()
                .map(|layout| layout.clone_ref(py)),
        },
    )?;
    let acknowledgement = binder.bind(py).call1((binding,))?;
    if !acknowledgement.is_none() {
        return Err(PyTypeError::new_err(
            "runner.bind_resources must return None",
        ));
    }
    Ok(config)
}
