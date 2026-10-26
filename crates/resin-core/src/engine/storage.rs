//! Transport-free ownership and scheduling for optional immutable cache storage.

use std::collections::BTreeMap;

use super::*;
use crate::resources::paged_kv::{PrefixExport, PrefixImport, PrefixTransfer};

#[derive(Clone, Debug)]
pub struct StorageCandidate {
    pub request_id: u64,
    pub tokens: Vec<u32>,
    pub resident_tokens: u32,
    pub max_prefix_tokens: u32,
}

#[derive(Clone, Debug)]
pub struct StorageExportJob {
    pub id: u64,
    pub tokens: Vec<u32>,
    pub transfer: PrefixTransfer,
}

#[derive(Clone, Debug)]
pub struct StorageImportJob {
    pub id: u64,
    pub transfer: PrefixTransfer,
}

#[derive(Debug)]
pub enum StorageImportStart {
    Pending,
    Gone,
    Resident(u32),
    Miss,
    Transfer(StorageImportJob),
}

struct PendingImport {
    request_id: u64,
    tokens: Vec<u32>,
    import: PrefixImport,
}

pub(super) struct StorageState {
    max_pending: usize,
    next_id: u64,
    exports: BTreeMap<u64, PrefixExport>,
    export_jobs: Vec<StorageExportJob>,
    imports: BTreeMap<u64, PendingImport>,
    quarantined_requests: Vec<RequestPages>,
}

impl StorageState {
    fn next_id(&mut self) -> Result<u64> {
        let id = self.next_id;
        self.next_id = id
            .checked_add(1)
            .ok_or_else(|| Error::internal("cache-storage job id exhausted"))?;
        Ok(id)
    }

    fn full(&self) -> bool {
        self.exports.len() + self.imports.len() >= self.max_pending
    }
}

impl<P: Clone> EngineCore<P> {
    pub fn enable_cache_storage(&mut self, max_pending: usize) -> Result<()> {
        self.raise_if_open()?;
        if max_pending == 0 {
            return Err(Error::invalid("cache-storage max_pending must be positive"));
        }
        if self.prefix_alignment().is_none() {
            return Err(Error::invalid(
                "cache storage requires managed prefix caching",
            ));
        }
        if self.storage.is_some() {
            return Err(Error::invalid("cache storage is already enabled"));
        }
        self.storage = Some(StorageState {
            max_pending,
            next_id: 1,
            exports: BTreeMap::new(),
            export_jobs: Vec::new(),
            imports: BTreeMap::new(),
            quarantined_requests: Vec::new(),
        });
        Ok(())
    }

    pub fn storage_has_work(&self) -> bool {
        self.storage.as_ref().is_some_and(|storage| {
            !storage.exports.is_empty()
                || !storage.imports.is_empty()
                || !storage.quarantined_requests.is_empty()
        })
    }

    pub fn take_storage_exports(&mut self) -> Vec<StorageExportJob> {
        self.storage
            .as_mut()
            .map_or_else(Vec::new, |storage| std::mem::take(&mut storage.export_jobs))
    }

    /// Completion proves that no provider still reads the source addresses.
    pub fn finish_storage_export(&mut self, id: u64) -> Result<()> {
        self.raise_if_open()?;
        let export = self
            .storage
            .as_mut()
            .and_then(|storage| storage.exports.remove(&id))
            .ok_or_else(|| Error::invalid("unknown cache-storage export id"))?;
        self.storage
            .as_mut()
            .expect("storage")
            .export_jobs
            .retain(|job| job.id != id);
        self.kv_resources
            .as_mut()
            .expect("storage has managed resources")
            .release_prefix_export(export);
        self.storage_update_idle();
        Ok(())
    }

    pub(super) fn queue_storage_export(
        &mut self,
        tokens: &[u32],
        resident_tokens: u32,
    ) -> Result<()> {
        let Some(storage) = &self.storage else {
            return Ok(());
        };
        let resources = self
            .kv_resources
            .as_mut()
            .expect("storage has managed resources");
        if storage.full() {
            return Ok(());
        }
        let Some(export) = resources.export_prefix(tokens, resident_tokens)? else {
            return Ok(());
        };
        let storage = self.storage.as_mut().expect("checked storage");
        let id = match storage.next_id() {
            Ok(id) => id,
            Err(error) => {
                resources.release_prefix_export(export);
                return Err(error);
            }
        };
        let job = StorageExportJob {
            id,
            tokens: tokens[..resident_tokens as usize].to_vec(),
            transfer: export.view().clone(),
        };
        storage.exports.insert(id, export);
        storage.export_jobs.push(job);
        Ok(())
    }

    /// Suspend only the next admitted cohort while the frontend probes storage.
    /// No destination memory or running slot is reserved for a lookup.
    pub fn storage_candidates(&mut self) -> Result<Vec<StorageCandidate>> {
        self.raise_if_open()?;
        if self.in_flight.is_some() || self.storage.is_none() {
            return Ok(Vec::new());
        }
        let waiting = self
            .scheduler
            .iter()
            .filter(|request| request.storage_waiting)
            .count();
        let budget = self.max_running_requests().saturating_sub(waiting);
        if budget == 0 {
            return Ok(Vec::new());
        }
        let ids = self
            .scheduler
            .selected()
            .0
            .into_iter()
            .filter(|request| {
                !request.running
                    && !request.storage_checked
                    && request.cacheable
                    && !request.prompt_logprobs
                    && request.generated.is_empty()
                    && request.phase() == Phase::Prefill
            })
            .take(budget)
            .map(|request| request.id)
            .collect::<Vec<_>>();
        let alignment = self
            .prefix_alignment()
            .expect("storage has prefix alignment");
        let suffix = self
            .runner_config
            .cache_layout
            .as_ref()
            .expect("storage has layout")
            .replay_suffix_tokens();
        let mut candidates = Vec::new();
        for id in ids {
            let index = self
                .scheduler
                .position(|request| request.id == id)
                .expect("selected request remains queued");
            let mut request = self.scheduler.remove(index);
            request.storage_checked = true;
            let length = request.prompt_length();
            let maximum = length.saturating_sub(1) / alignment * alignment;
            let maximum = maximum.min(length.saturating_sub(suffix) / alignment * alignment);
            if maximum > request.resident_tokens {
                request.storage_waiting = true;
                request.storage_base_tokens = request.resident_tokens;
                candidates.push(StorageCandidate {
                    request_id: id,
                    tokens: request
                        .cache_prompt
                        .as_ref()
                        .unwrap_or(&request.prompt)
                        .clone(),
                    resident_tokens: request.resident_tokens,
                    max_prefix_tokens: maximum,
                });
            }
            self.scheduler.restore(index, request);
        }
        Ok(candidates)
    }

    pub fn storage_lookup_miss(&mut self, request_ids: &[u64]) -> Result<()> {
        self.raise_if_open()?;
        for &id in request_ids {
            if self.storage.as_ref().is_some_and(|storage| {
                storage
                    .imports
                    .values()
                    .any(|pending| pending.request_id == id)
            }) {
                return Err(Error::invalid(
                    "cannot miss an accepted cache-storage import",
                ));
            }
            if let Some(index) = self.scheduler.position(|request| request.id == id) {
                let mut request = self.scheduler.remove(index);
                request.storage_checked = true;
                request.storage_waiting = false;
                self.scheduler.restore(index, request);
            }
        }
        Ok(())
    }

    pub fn begin_storage_import(
        &mut self,
        request_id: u64,
        length: u32,
    ) -> Result<StorageImportStart> {
        self.raise_if_open()?;
        if self.in_flight.is_some() {
            return Ok(StorageImportStart::Pending);
        }
        let Some(storage) = &self.storage else {
            return Err(Error::invalid("cache storage is disabled"));
        };
        let Some(index) = self.scheduler.position(|request| request.id == request_id) else {
            return Ok(StorageImportStart::Gone);
        };
        let request = self.scheduler.iter().nth(index).expect("request exists");
        let alignment = self.prefix_alignment().expect("storage has alignment");
        let suffix = self
            .runner_config
            .cache_layout
            .as_ref()
            .expect("storage has layout")
            .replay_suffix_tokens();
        let maximum = (request.prompt_length().saturating_sub(1) / alignment * alignment)
            .min(request.prompt_length().saturating_sub(suffix) / alignment * alignment);
        if !request.storage_waiting
            || !request.storage_checked
            || length == 0
            || length <= request.storage_base_tokens
            || !length.is_multiple_of(alignment)
            || length > maximum
        {
            return Err(Error::invalid(
                "invalid cache-storage restore boundary or request state",
            ));
        }
        if storage
            .imports
            .values()
            .any(|pending| pending.request_id == request_id)
        {
            return Ok(StorageImportStart::Pending);
        }
        let identity =
            request.cache_prompt.as_ref().unwrap_or(&request.prompt)[..length as usize].to_vec();
        if storage
            .imports
            .values()
            .any(|pending| pending.tokens == identity)
        {
            return Ok(StorageImportStart::Pending);
        }
        // Another request may have promoted this prefix while the lookup ran.
        // Reuse its physical pages before reserving another private copy.
        let mut request = self.scheduler.remove(index);
        let result = (|| {
            self.reuse_storage_prefix(&mut request)?;
            if request.resident_tokens >= length {
                request.storage_waiting = false;
                return Ok(StorageImportStart::Resident(request.resident_tokens));
            }
            let running = self
                .scheduler
                .iter()
                .filter(|request| request.running)
                .count();
            if running >= self.max_running_requests()
                || self.storage.as_ref().expect("storage").full()
            {
                return Ok(StorageImportStart::Pending);
            }
            if self.admission_prefix_len(&[&request])? == 0 {
                return Ok(StorageImportStart::Pending);
            }
            let id = self.storage.as_mut().expect("storage").next_id()?;
            let import = match self
                .kv_resources
                .as_mut()
                .expect("storage resources")
                .begin_prefix_import(core_request_id(request_id), length)
            {
                Ok(import) => import,
                Err(CachePoolError::InsufficientCapacity { .. }) => {
                    request.storage_waiting = false;
                    return Ok(StorageImportStart::Miss);
                }
                Err(error) => return Err(error.into()),
            };
            let storage = self.storage.as_mut().expect("storage");
            let job = StorageImportJob {
                id,
                transfer: import.view().clone(),
            };
            storage.imports.insert(
                id,
                PendingImport {
                    request_id,
                    tokens: identity,
                    import,
                },
            );
            request.running = true;
            Ok(StorageImportStart::Transfer(job))
        })();
        self.scheduler.restore(index, request);
        result
    }

    fn reuse_storage_prefix(&mut self, request: &mut Request<P>) -> Result<()> {
        let resources = self.kv_resources.as_mut().expect("storage has resources");
        let mut pages = resources.attach(core_request_id(request.id), 0)?;
        let restored = match resources.restore_prefix(
            &mut pages,
            request.cache_prompt.as_deref().unwrap_or(&request.prompt),
        ) {
            Ok(restored) => restored,
            Err(error) => {
                resources
                    .retire(pages)
                    .expect("failed restore keeps its owner valid");
                return Err(error.into());
            }
        };
        if restored <= request.resident_tokens {
            resources
                .retire(pages)
                .expect("temporary prefix references remain valid");
            return Ok(());
        }
        if let Some(old) = request.cache.replace(pages)
            && let Err(failure) = resources.retire(old)
        {
            self.storage
                .as_mut()
                .expect("storage")
                .quarantined_requests
                .push(failure.request);
            let error = Error::from(failure.error);
            self.poison("cache-storage prefix reuse failed", &error);
            return Err(error);
        }
        self.credit_storage_prefix(request, restored, false);
        Ok(())
    }

    fn credit_storage_prefix(
        &mut self,
        request: &mut Request<P>,
        resident_tokens: u32,
        from_storage: bool,
    ) {
        request.resident_tokens = resident_tokens;
        request
            .lease
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cached_tokens = resident_tokens;
        if let (Some(metrics), Some(state)) = (&mut self.metrics, &mut request.metrics) {
            metrics.credit_cached_prefix(state, resident_tokens, from_storage);
        }
    }

    fn storage_update_idle(&mut self) {
        let idle = !self.has_work();
        if let Some(metrics) = &mut self.metrics {
            metrics.set_idle(idle, Instant::now());
        }
    }

    /// Return the successfully restored resident token count (zero on abort).
    /// The frontend must establish destination quiescence before either outcome.
    pub fn finish_storage_import(&mut self, id: u64, success: bool) -> Result<u32> {
        self.raise_if_open()?;
        if self.in_flight.is_some() {
            return Err(Error::internal(
                "drain model execution before completing storage imports",
            ));
        }
        let pending = self
            .storage
            .as_mut()
            .and_then(|storage| storage.imports.remove(&id))
            .ok_or_else(|| Error::invalid("unknown cache-storage import id"))?;
        let PendingImport {
            request_id,
            tokens,
            import,
        } = pending;
        let index = self.scheduler.position(|request| request.id == request_id);
        let length = import.view().resident_tokens;
        let resources = self.kv_resources.as_mut().expect("storage has resources");
        if !success || index.is_none() {
            if let Err(failure) = resources.abort_prefix_import(import) {
                self.storage.as_mut().expect("storage").imports.insert(
                    id,
                    PendingImport {
                        request_id,
                        tokens,
                        import: failure.import,
                    },
                );
                return Err(failure.error.into());
            }
            if let Some(index) = index {
                let mut request = self.scheduler.remove(index);
                request.storage_waiting = false;
                request.running = false;
                self.scheduler.restore(index, request);
            }
            self.storage_update_idle();
            return Ok(0);
        }
        let pages = match resources.commit_prefix_import(import, &tokens) {
            Ok(pages) => pages,
            Err(failure) => {
                self.storage.as_mut().expect("storage").imports.insert(
                    id,
                    PendingImport {
                        request_id,
                        tokens,
                        import: failure.import,
                    },
                );
                return Err(failure.error.into());
            }
        };
        let index = index.expect("checked request");
        let mut request = self.scheduler.remove(index);
        if let Some(old) = request.cache.replace(pages)
            && let Err(failure) = self.kv_resources.as_mut().expect("resources").retire(old)
        {
            // This indicates native ownership corruption after transfer;
            // retain the old capability rather than silently dropping it.
            self.storage
                .as_mut()
                .expect("storage")
                .quarantined_requests
                .push(failure.request);
            self.scheduler.restore(index, request);
            let error = Error::from(failure.error);
            self.poison("cache-storage adoption failed", &error);
            return Err(error);
        }
        self.credit_storage_prefix(&mut request, length, true);
        request.storage_waiting = false;
        self.scheduler.restore(index, request);
        self.storage_update_idle();
        Ok(length)
    }
}
