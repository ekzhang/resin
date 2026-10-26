//! Optional Python transport coordination over native cache-transfer ownership.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use resin_core::engine::StorageImportStart;
use resin_core::metrics::CacheStorageMetricsSnapshot;
use resin_core::resources::paged_kv::PrefixTransfer;

use crate::{CacheLayout, RequestPayload, core, core_error};

type TransferGroup = (u32, u32, Vec<u32>, Vec<u32>);
type Transfer = (u32, Vec<TransferGroup>);
type Candidate = (u64, Vec<u32>, u32, u32);
type Completions = (Vec<u64>, Vec<(u64, u32)>, Vec<(u64, bool)>);

fn descriptor(transfer: PrefixTransfer) -> Transfer {
    (
        transfer.resident_tokens,
        transfer
            .groups
            .into_iter()
            .map(|group| {
                (
                    group.group_index,
                    group.first_block,
                    group.slab_ids,
                    group.slotnums,
                )
            })
            .collect(),
    )
}

pub(crate) struct StorageState {
    runtime: Py<PyAny>,
    completed_loads: Vec<(u64, bool)>,
}

impl StorageState {
    pub(crate) fn new(
        py: Python<'_>,
        provider: &Py<PyAny>,
        runner: &Py<PyAny>,
        layout: &Py<CacheLayout>,
        namespace: Option<&Py<PyAny>>,
        max_pending: usize,
        metrics: bool,
    ) -> PyResult<Self> {
        let runtime = PyModule::import(py, "resin._cache_storage")?
            .getattr("_StorageRuntime")?
            .call1((provider, runner, layout, namespace, max_pending, metrics))?
            .unbind();
        Ok(Self {
            runtime,
            completed_loads: Vec::new(),
        })
    }

    pub(crate) fn metrics(&self, py: Python<'_>) -> PyResult<CacheStorageMetricsSnapshot> {
        let value = self.runtime.bind(py).call_method0("metrics_snapshot")?;
        let snapshot: CacheStorageMetricsSnapshot = pythonize::depythonize(&value)?;
        snapshot
            .validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(snapshot)
    }

    pub(crate) fn provider(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.runtime.bind(py).getattr("storage")?.unbind())
    }

    pub(crate) fn tick(
        &mut self,
        py: Python<'_>,
        engine: &mut core::EngineCore<RequestPayload>,
        closing: bool,
    ) -> PyResult<()> {
        let runtime = self.runtime.bind(py);
        // Immediate providers can complete lookup and restoration in the same
        // step. Bound the work so a ready provider cannot starve model dispatch.
        for _ in 0..8 {
            let mut progress = false;
            for job in engine.take_storage_exports() {
                runtime.call_method1("export", (job.id, job.tokens, descriptor(job.transfer)))?;
                progress = true;
            }
            let (exports, lookups, loads): Completions =
                runtime.call_method1("poll", (closing,))?.extract()?;
            progress |= !exports.is_empty() || !lookups.is_empty() || !loads.is_empty();
            for id in exports {
                engine.finish_storage_export(id).map_err(core_error)?;
            }
            for (request_id, length) in lookups {
                if length == 0 || closing {
                    engine
                        .storage_lookup_miss(&[request_id])
                        .map_err(core_error)?;
                    runtime.call_method1("resolve", (request_id,))?;
                }
            }
            self.completed_loads.extend(loads);
            if !engine.has_in_flight_step() {
                for (id, success) in std::mem::take(&mut self.completed_loads) {
                    let restored = engine
                        .finish_storage_import(id, success && !closing)
                        .map_err(core_error)?;
                    if restored > 0 {
                        runtime.call_method1("restored", (restored,))?;
                    }
                    progress = true;
                }
            }
            let ready: Vec<(u64, u32)> = runtime.call_method0("ready_loads")?.extract()?;
            for (request_id, length) in ready {
                if closing {
                    engine
                        .storage_lookup_miss(&[request_id])
                        .map_err(core_error)?;
                    runtime.call_method1("resolve", (request_id,))?;
                    continue;
                }
                match engine
                    .begin_storage_import(request_id, length)
                    .map_err(core_error)?
                {
                    StorageImportStart::Pending => {}
                    StorageImportStart::Gone
                    | StorageImportStart::Resident(_)
                    | StorageImportStart::Miss => {
                        runtime.call_method1("resolve", (request_id,))?;
                        progress = true;
                    }
                    StorageImportStart::Transfer(job) => {
                        runtime
                            .call_method1("load", (request_id, job.id, descriptor(job.transfer)))?;
                        progress = true;
                    }
                }
            }
            if !closing && runtime.call_method0("can_lookup")?.extract::<bool>()? {
                let candidates: Vec<Candidate> = engine
                    .storage_candidates()
                    .map_err(core_error)?
                    .into_iter()
                    .map(|request| {
                        (
                            request.request_id,
                            request.tokens,
                            request.resident_tokens,
                            request.max_prefix_tokens,
                        )
                    })
                    .collect();
                if !candidates.is_empty() {
                    runtime.call_method1("candidates", (candidates,))?;
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }
        Ok(())
    }
}

impl crate::Engine {
    pub(crate) fn retain_storage_quarantine(&mut self) {
        if self.core.storage_has_work()
            && let Some(storage) = self.cache_storage_state.take()
        {
            // An unretired foreign operation can still address these buffers
            // after Engine destruction. Keep its runtime and arena owners
            // alive; dropping a failed operation is not a retirement fence.
            std::mem::forget(storage);
        }
    }

    pub(crate) fn poll_storage(&mut self, py: Python<'_>, closing: bool) -> PyResult<()> {
        self.core.raise_if_open().map_err(core_error)?;
        let Some(mut storage) = self.cache_storage_state.take() else {
            return Ok(());
        };
        let result = storage.tick(py, &mut self.core, closing);
        self.cache_storage_state = Some(storage);
        if let Err(error) = &result {
            // Exceptions are not retirement proof. Keep every native owner and
            // Python buffer generation quarantined instead of retrying memory.
            self.core
                .poison_message(format!("cache storage operation failed: {error}"));
        }
        result
    }

    pub(crate) fn storage_metrics(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<CacheStorageMetricsSnapshot>> {
        self.cache_storage_state
            .as_ref()
            .map(|storage| storage.metrics(py))
            .transpose()
    }
}
