use std::collections::{HashMap, HashSet};
use std::fmt;
use std::num::NonZeroU32;

use crate::{RequestId, SequenceStep};

use super::kv::{
    KvGroupLayout, KvGroupPlan, KvLayout, KvPlanError, KvRetention, StateCheckpointSlot, StateCopy,
    StatePlan,
};
use super::paged::{
    CachePool, CachePoolError, CachedPages, GroupDemand, GroupId, PageAccessClaim, PageHandle,
    RequestPages, RetireRequestFailure, SequenceId, WriterAccessLease,
};
use super::prefix::PrefixCache;
use super::{BatchAccess, RequestResourceRef};

/// A point-in-time inventory of the shared physical cache pool.
///
/// Slabs are fungible across groups; group block counts are not independent
/// capacity partitions. Reclaimable storage excludes live requests and all
/// outstanding device access, including retired overlapped executions.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct CacheMetrics {
    pub capacity_slabs: u32,
    pub free_slabs: u32,
    pub occupied_slabs: u32,
    pub reclaimable_slabs: u32,
    pub groups: Vec<CacheGroupMetrics>,
    pub prefix: Option<PrefixCacheMetrics>,
}

/// Physical block usage for an ordered runner-declared cache group.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct CacheGroupMetrics {
    pub group: u32,
    /// Full history (`kv`), sliding-window history (`swa`), or `recurrent`.
    pub pool: &'static str,
    pub block_tokens: u32,
    pub blocks_per_slab: u32,
    pub occupied_slabs: u32,
    pub reclaimable_slabs: u32,
    pub occupied_blocks: u64,
    /// Blocks held by at least one prefix segment, including live requests.
    pub cached_blocks: u64,
    /// Blocks freed by retiring every cached prefix, with requests unchanged.
    pub reclaimable_blocks: u64,
    /// Blocks with outstanding device read or write leases.
    pub pinned_blocks: u64,
    /// Unoccupied slots in slabs currently bound to this group.
    pub free_blocks: u64,
    /// Block-rounded token capacity; recurrent checkpoints are not tokens.
    pub occupied_tokens: Option<u64>,
    pub reclaimable_tokens: Option<u64>,
}

/// Prefix LRU counters. Reclaimed storage counts actual physical frees during
/// eviction; evicting an ancestor or a live request's prefix may free nothing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
pub struct PrefixCacheMetrics {
    pub entries: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub reclaimed_blocks: u64,
    pub reclaimed_slabs: u64,
}

pub type PrefixStorageGroup = (u32, Vec<(u32, u32)>);

/// Synchronous storage hooks for complete immutable checkpoints. Every buffer
/// access must retire before a callback returns or unwinds, including failures.
/// Groups follow layout order; each contains its first logical block and the
/// chronological (slab, slot) addresses. Providers never retain these addresses.
pub trait PrefixStorage: fmt::Debug + Send + Sync {
    fn lookup(&mut self, tokens: &[u32], maximum: u32, alignment: u32) -> u32;
    fn transfer(&mut self, tokens: &[u32], groups: &[PrefixStorageGroup], load: bool) -> bool;
}

#[derive(Debug)]
pub struct PagedKvResources {
    layout: KvLayout,
    pool: CachePool,
    prefix_cache: Option<PrefixCache<super::paged::CachedPages>>,
    pub(crate) storage: Option<Box<dyn PrefixStorage>>,
    prefix_reclaimed_blocks: u64,
    prefix_reclaimed_slabs: u64,
    batch_active: bool,
    state_resources: HashMap<u64, StateResources>,
    state_groups: Vec<bool>,
    state_demands: Vec<GroupDemand>,
    last_snapshot_id: u64,
}

#[derive(Debug)]
struct StateResources {
    pending: bool,
    restore: Option<RequestPages>,
    snapshots: Vec<StateSnapshotReservation>,
}

impl StateResources {
    fn new(restore: Option<RequestPages>) -> Self {
        Self {
            pending: true,
            restore,
            snapshots: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct StateSnapshotReservation {
    id: u64,
    prefix_length: u32,
    /// Every persistent write allowed by the frozen lease, including rejected
    /// speculative rows. A WINDOW ring cannot publish an older checkpoint
    /// after this envelope has reused its physical pages.
    history_write_end: u32,
    owner: RequestPages,
    ready: bool,
}

impl PagedKvResources {
    pub fn new(layout: KvLayout) -> Result<Self, CachePoolError> {
        Self::configured(layout, None)
    }

    pub fn with_prefix_cache(layout: KvLayout) -> Result<Self, CachePoolError> {
        let alignment = prefix_alignment(&layout)?;
        Self::configured(layout, Some(alignment))
    }

    fn configured(layout: KvLayout, prefix_alignment: Option<u32>) -> Result<Self, CachePoolError> {
        let pool = CachePool::new(
            layout.parent_capacity().get(),
            &layout.packings().collect::<Vec<_>>(),
        )?;
        let prefix_cache = prefix_alignment.map(PrefixCache::new);
        let state_groups: Vec<_> = layout
            .groups()
            .iter()
            .map(|group| group.is_recurrent())
            .collect();
        let state_demands = if state_groups.iter().any(|&state| state) {
            state_groups
                .iter()
                .enumerate()
                .map(|(index, &state)| {
                    GroupDemand::new(GroupId::new(index as u32), usize::from(state))
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            layout,
            pool,
            prefix_cache,
            storage: None,
            prefix_reclaimed_blocks: 0,
            prefix_reclaimed_slabs: 0,
            batch_active: false,
            state_resources: HashMap::new(),
            state_groups,
            state_demands,
            last_snapshot_id: 0,
        })
    }

    pub const fn pool(&self) -> &CachePool {
        &self.pool
    }

    pub fn parent_capacity(&self) -> u32 {
        self.layout.parent_capacity().get()
    }

    /// Collect a bounded-cadence snapshot, scanning physical block metadata.
    /// No GPU synchronization or token-history traversal is needed.
    pub fn metrics(&self) -> CacheMetrics {
        let groups = self
            .layout
            .groups()
            .iter()
            .zip(self.pool.group_occupancy())
            .enumerate()
            .map(|(group, (layout, occupancy))| {
                let pool = if layout.is_recurrent() {
                    "recurrent"
                } else if layout.retention().window_tokens().is_some() {
                    "swa"
                } else {
                    "kv"
                };
                let block_tokens = layout.block_tokens().get();
                CacheGroupMetrics {
                    group: group as u32,
                    pool,
                    block_tokens,
                    blocks_per_slab: layout.blocks_per_slab().get(),
                    occupied_slabs: occupancy.occupied_slabs,
                    reclaimable_slabs: occupancy.reclaimable_slabs,
                    occupied_blocks: occupancy.occupied_blocks,
                    cached_blocks: occupancy.cached_blocks,
                    reclaimable_blocks: occupancy.reclaimable_blocks,
                    pinned_blocks: occupancy.pinned_blocks,
                    free_blocks: occupancy.free_blocks,
                    occupied_tokens: (!layout.is_recurrent())
                        .then_some(occupancy.occupied_blocks * u64::from(block_tokens)),
                    reclaimable_tokens: (!layout.is_recurrent())
                        .then_some(occupancy.reclaimable_blocks * u64::from(block_tokens)),
                }
            })
            .collect::<Vec<_>>();
        let free_slabs = self.pool.free_parent_count() as u32;
        CacheMetrics {
            capacity_slabs: self.parent_capacity(),
            free_slabs,
            occupied_slabs: self.parent_capacity() - free_slabs,
            reclaimable_slabs: groups.iter().map(|group| group.reclaimable_slabs).sum(),
            groups,
            prefix: self.prefix_metrics(),
        }
    }

    /// A batch must preserve the earliest query's read window while writing
    /// every scheduled token (including speculative lookahead).
    pub fn max_write_tokens(&self) -> Option<NonZeroU32> {
        self.layout
            .groups()
            .iter()
            .copied()
            .filter_map(KvGroupLayout::max_write_tokens)
            .min()
    }

    fn validate_write_span(&self, token_start: u32, context: u32) -> Result<(), CachePoolError> {
        let requested = context
            .checked_sub(token_start)
            .ok_or(CachePoolError::DemandOverflow)?;
        if let Some(maximum) = self.max_write_tokens()
            && requested > maximum.get()
        {
            return Err(CachePoolError::InvalidWriteSpan {
                requested,
                maximum: maximum.get(),
            });
        }
        Ok(())
    }

    pub fn attach(
        &mut self,
        request_id: RequestId,
        maximum_context: u32,
    ) -> Result<RequestPages, CachePoolError> {
        let demands = self.demands(maximum_context);
        let pending = self
            .pool
            .reserve_request(SequenceId::new(request_id.get()), &demands)?;
        let state = self.pool.promote_request(pending);
        if !self.state_demands.is_empty() {
            self.state_resources
                .insert(request_id.get(), StateResources::new(None));
        }
        Ok(state)
    }

    /// Grow one request just enough to address `context` tokens.
    pub fn ensure_context(
        &mut self,
        state: &mut RequestPages,
        token_start: u32,
        context: u32,
    ) -> Result<(), CachePoolError> {
        self.validate_write_span(token_start, context)?;
        // A previous wide lease can end beyond this step's ordinary span.
        // Keep its idle spare pages; grow_request must never shrink owners.
        let demands = self
            .layout
            .groups()
            .iter()
            .zip(state.groups())
            .enumerate()
            .map(|(index, (group, table))| {
                GroupDemand::new(
                    GroupId::new(index as u32),
                    (group.active_blocks(context).1 as usize).max(table.pages().len()),
                )
            })
            .collect::<Vec<_>>();
        loop {
            match self.pool.grow_request(state, &demands) {
                Ok(()) => break,
                Err(error @ CachePoolError::InsufficientCapacity { .. }) => {
                    if !self.evict_one_prefix()? {
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }

        let mut replacements = vec![Vec::new(); self.layout.groups().len()];
        for (group_index, group) in self.layout.groups().iter().copied().enumerate() {
            let table = &state.groups()[group_index];
            let first = token_start / group.block_tokens().get();
            let count = context
                .div_ceil(group.block_tokens().get())
                .saturating_sub(first);
            // A cyclic table can be visited at most once. In particular, a
            // long prefill must not scan every token to discover COW pages.
            let count = if group.is_recurrent() || group.max_write_tokens().is_some() {
                count.min(table.pages().len() as u32)
            } else {
                count
            };
            for logical in first..first + count {
                let index = page_index(group, table.pages().len(), logical).ok_or(
                    CachePoolError::InsufficientCapacity {
                        group: GroupId::new(group_index as u32),
                        requested: logical as usize + 1,
                        available: table.pages().len(),
                    },
                )?;
                let page = &table.pages()[index];
                if !page.is_writable() {
                    if group.retention() == KvRetention::FullContext && !group.is_recurrent() {
                        return Err(CachePoolError::PageNotWritable);
                    }
                    replacements[group_index].push(index);
                }
            }
            replacements[group_index].sort_unstable();
            replacements[group_index].dedup();
        }
        // grow_request already authenticated the whole idle owner above. An
        // unchanged ring needs no second full-history validation or mutation.
        if replacements.iter().all(Vec::is_empty) {
            return Ok(());
        }
        loop {
            match self.pool.replace_shared_pages(state, &replacements) {
                Ok(()) => return Ok(()),
                Err(error @ CachePoolError::InsufficientCapacity { .. }) => {
                    if !self.evict_one_prefix()? {
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Check optional cohort growth/COW before changing any request. Prefix
    /// eviction is deliberately excluded: overlap is optional under pressure.
    pub(crate) fn can_ensure_contexts(
        &self,
        requests: &[(&RequestPages, u32, u32)],
    ) -> Result<bool, CachePoolError> {
        let mut allocations = Vec::with_capacity(requests.len());
        for &(state, start, context) in requests {
            self.validate_write_span(start, context)?;
            let mut additions = Vec::with_capacity(self.layout.groups().len());
            for (index, group) in self.layout.groups().iter().copied().enumerate() {
                let pages = state.groups()[index].pages();
                let (_, wanted) = group.active_blocks(context);
                let page_count = (wanted as usize).max(pages.len());
                let mut extra = page_count - pages.len();
                let first = start / group.block_tokens().get();
                let count = context
                    .div_ceil(group.block_tokens().get())
                    .saturating_sub(first)
                    .min(page_count as u32);
                for logical in first..first + count {
                    let address = page_index(group, page_count, logical)
                        .ok_or(CachePoolError::StaleHandle)?;
                    if pages.get(address).is_some_and(|page| !page.is_writable()) {
                        extra += 1;
                    }
                }
                additions.push(GroupDemand::new(GroupId::new(index as u32), extra));
            }
            allocations.push((state, additions));
        }
        self.pool.can_allocate_request_pages(&allocations)
    }

    /// Attach the longest aligned cached prefix and return its resident token
    /// count. A miss leaves the empty request untouched.
    pub fn restore_prefix(
        &mut self,
        state: &mut RequestPages,
        tokens: &[u32],
    ) -> Result<u32, CachePoolError> {
        let maximum = tokens
            .len()
            .saturating_sub(self.layout.replay_suffix_tokens() as usize);
        let (pool, cache) = (&mut self.pool, &mut self.prefix_cache);
        let Some((length, snapshot)) = cache.as_mut().and_then(|cache| cache.find(tokens, maximum))
        else {
            if self.restore_storage(state, tokens, maximum)? {
                return self.restore_prefix(state, tokens);
            }
            return Ok(0);
        };
        let mut private = if !self.state_demands.is_empty() {
            // Allocate the private destination before attaching any immutable
            // source pages. Capacity pressure is a cache miss, leaving the
            // request empty and eligible for ordinary fresh prefill.
            match pool.reserve_request(state.sequence(), &self.state_demands) {
                Ok(pending) => Some(pool.promote_request(pending)),
                Err(CachePoolError::InsufficientCapacity { .. }) => return Ok(0),
                Err(error) => return Err(error),
            }
        } else {
            None
        };
        if let Err(error) = pool.attach_shared_prefix(state, snapshot) {
            if let Some(private) = private.take() {
                pool.retire_request(private)
                    .map_err(|failure| failure.error)?;
            }
            return Err(error);
        }
        if let Some(mut restore) = private {
            if let Err(error) =
                pool.exchange_state_restore_pages(state, &mut restore, &self.state_groups)
            {
                pool.clear_request(state)?;
                pool.retire_request(restore)
                    .map_err(|failure| failure.error)?;
                return Err(error);
            }
            self.state_resources
                .insert(state.sequence().get(), StateResources::new(Some(restore)));
        }
        Ok(length)
    }

    fn restore_storage(
        &mut self,
        state: &mut RequestPages,
        tokens: &[u32],
        maximum: usize,
    ) -> Result<bool, CachePoolError> {
        let Some(storage) = self.storage.as_mut() else {
            return Ok(false);
        };
        let alignment = self.prefix_cache.as_ref().unwrap().alignment();
        let maximum = maximum.min(tokens.len().saturating_sub(1)) as u32;
        let length = storage.lookup(tokens, maximum, alignment);
        if length == 0 || length > maximum || !length.is_multiple_of(alignment) {
            return Ok(false);
        }
        let demands = self.demands(length);
        let pending = loop {
            match self.pool.reserve_request(state.sequence(), &demands) {
                Ok(pages) => break pages,
                Err(CachePoolError::InsufficientCapacity { .. }) => {
                    if !self.evict_one_prefix()? {
                        return Ok(false);
                    }
                }
                Err(error) => return Err(error),
            }
        };
        let mut imported = self.pool.promote_request(pending);
        let loaded = (|| {
            let tables = imported
                .groups()
                .iter()
                .map(|group| group.pages().iter().map(|page| page.handle()).collect())
                .collect();
            if !self.storage_transfer(&tokens[..length as usize], tables, true)? {
                return Ok(false);
            }
            let snapshot = self.pool.share_request_prefix(&mut imported, &demands)?;
            let cache = self.prefix_cache.as_mut().unwrap();
            let replaced = cache.insert(&tokens[..length as usize], snapshot);
            if let Some(replaced) = replaced {
                self.pool.retire_cached_prefix(replaced)?;
            }
            Ok(true)
        })();
        self.pool
            .retire_request(imported)
            .map_err(|failure| failure.error)?;
        loaded
    }

    /// Publish a completed aligned resident boundary. Returns false when the
    /// current progress cannot form a safe block-complete snapshot.
    pub fn publish_prefix(
        &mut self,
        state: &mut RequestPages,
        tokens: &[u32],
        resident_tokens: u32,
    ) -> Result<bool, CachePoolError> {
        if !self.state_demands.is_empty() {
            // Live state is never frozen in place. Only an authenticated,
            // separately materialized state snapshot may publish this prefix.
            return Ok(false);
        }
        let Some(cache) = &self.prefix_cache else {
            return Ok(false);
        };
        let length = resident_tokens as usize;
        if length == 0
            || length > tokens.len()
            || !resident_tokens.is_multiple_of(cache.alignment())
        {
            return Ok(false);
        }
        let demands = self.demands(resident_tokens);
        let pages = self.pool.share_request_prefix(state, &demands)?;
        if let Err(error) = self.store_prefix(&pages, &tokens[..length]) {
            self.pool.retire_cached_prefix(pages)?;
            return Err(error);
        }
        let replaced = self
            .prefix_cache
            .as_mut()
            .expect("cache enablement was checked")
            .insert(&tokens[..length], pages);
        if let Some(replaced) = replaced {
            self.pool.retire_cached_prefix(replaced)?;
        }
        Ok(true)
    }

    pub fn evict_one_prefix(&mut self) -> Result<bool, CachePoolError> {
        let Some(snapshot) = self.prefix_cache.as_mut().and_then(PrefixCache::evict_lru) else {
            return Ok(false);
        };
        let occupied = self.pool.occupied_block_count();
        let free = self.pool.free_parent_count();
        self.pool.retire_cached_prefix(snapshot)?;
        self.prefix_reclaimed_blocks = self
            .prefix_reclaimed_blocks
            .saturating_add((occupied - self.pool.occupied_block_count()) as u64);
        self.prefix_reclaimed_slabs = self
            .prefix_reclaimed_slabs
            .saturating_add((self.pool.free_parent_count() - free) as u64);
        Ok(true)
    }

    pub fn clear_prefix_cache(&mut self) -> Result<(), CachePoolError> {
        while self.evict_one_prefix()? {}
        Ok(())
    }

    pub fn prefix_metrics(&self) -> Option<PrefixCacheMetrics> {
        self.prefix_cache.as_ref().map(|cache| PrefixCacheMetrics {
            entries: cache.len() as u64,
            hits: cache.hits(),
            misses: cache.misses(),
            evictions: cache.evictions(),
            reclaimed_blocks: self.prefix_reclaimed_blocks,
            reclaimed_slabs: self.prefix_reclaimed_slabs,
        })
    }

    pub fn prefix_alignment(&self) -> Option<u32> {
        self.prefix_cache.as_ref().map(PrefixCache::alignment)
    }

    /// Drop resident KV while retaining the request's allocator identity.
    pub fn reset(&mut self, state: &mut RequestPages) -> Result<(), CachePoolError> {
        self.pool.validate_request_idle(state)?;
        self.retire_state_auxiliary(state.sequence().get())?;
        if !self.state_demands.is_empty() {
            self.state_resources
                .insert(state.sequence().get(), StateResources::new(None));
        }
        self.pool.clear_request(state)
    }

    /// Physical-parent budget for a set of estimated request contexts.
    pub fn estimated_parent_count(&self, contexts: &[u32]) -> Result<u64, CachePoolError> {
        self.layout
            .estimated_parent_count(contexts)
            .map_err(|_| CachePoolError::DemandOverflow)
    }

    /// Accumulate admission demand, crediting only pages physically shared by
    /// the selected requests. Merely equal token prefixes receive no credit.
    pub(crate) fn admission_estimate(&self) -> AdmissionEstimate<'_> {
        AdmissionEstimate {
            layout: &self.layout,
            blocks: vec![0; self.layout.groups().len()],
            occupied_parents: vec![HashSet::new(); self.layout.groups().len()],
            shared: HashSet::new(),
        }
    }

    pub fn retire(&mut self, state: RequestPages) -> Result<(), RetireRequestFailure> {
        if let Err(error) = self
            .pool
            .validate_request(&state)
            .and_then(|()| self.retire_state_auxiliary(state.sequence().get()))
        {
            return Err(RetireRequestFailure {
                error,
                request: state,
            });
        }
        self.pool.retire_request(state)
    }

    fn retire_state_auxiliary(&mut self, request_id: u64) -> Result<(), CachePoolError> {
        if let Some(state) = self.state_resources.remove(&request_id) {
            for owner in state
                .restore
                .into_iter()
                .chain(state.snapshots.into_iter().map(|slot| slot.owner))
            {
                self.pool
                    .retire_request(owner)
                    .map_err(|failure| failure.error)?;
            }
        }
        Ok(())
    }

    fn clear_drained_snapshots(&mut self) -> Result<(), CachePoolError> {
        for state in self.state_resources.values_mut() {
            for snapshot in state.snapshots.drain(..) {
                self.pool
                    .retire_request(snapshot.owner)
                    .map_err(|failure| failure.error)?;
            }
            if !state.pending
                && let Some(restore) = state.restore.take()
            {
                self.pool
                    .retire_request(restore)
                    .map_err(|failure| failure.error)?;
            }
        }
        Ok(())
    }

    pub(crate) fn finish_state_initialization(&mut self, request_ids: &[u64]) {
        for id in request_ids {
            if let Some(state) = self.state_resources.get_mut(id) {
                state.pending = false;
            }
        }
    }

    pub(crate) fn accept_state_checkpoints(
        &mut self,
        progress: &[(u64, u32, u32, bool)],
        checkpoints: &[u64],
    ) -> Result<(), PagedKvError> {
        if checkpoints.is_empty() {
            return Ok(());
        }
        let ids: HashSet<_> = checkpoints.iter().copied().collect();
        if ids.len() != checkpoints.len() {
            return Err(Self::invalid("duplicate state checkpoint receipt"));
        }
        let mut accepted = Vec::with_capacity(ids.len());
        for &(request_id, physical, visible, terminal) in progress {
            if let Some(state) = self.state_resources.get(&request_id) {
                for (index, snapshot) in state.snapshots.iter().enumerate() {
                    if !ids.contains(&snapshot.id) {
                        continue;
                    }
                    if snapshot.ready || (!terminal && snapshot.prefix_length != physical) {
                        return Err(Self::invalid(
                            "state checkpoint does not match the accepted cache prefix",
                        ));
                    }
                    accepted.push((request_id, index, !terminal && physical == visible));
                }
            }
        }
        if accepted.len() != ids.len() {
            return Err(Self::invalid(
                "unknown state checkpoint reservation or different cohort",
            ));
        }
        // Authenticate the complete report before publishing any ready flag.
        for (request_id, index, ready) in accepted {
            self.state_resources.get_mut(&request_id).unwrap().snapshots[index].ready = ready;
        }
        Ok(())
    }

    pub(crate) fn ready_state_checkpoints(&self, request_id: u64) -> Vec<(u64, u32)> {
        self.state_resources
            .get(&request_id)
            .into_iter()
            .flat_map(|state| &state.snapshots)
            .filter(|slot| slot.ready)
            .map(|slot| (slot.id, slot.prefix_length))
            .collect()
    }

    pub(crate) fn publish_state_prefix(
        &mut self,
        state: &mut RequestPages,
        reservation_id: u64,
        tokens: &[u32],
    ) -> Result<(), CachePoolError> {
        let snapshot = self
            .state_resources
            .get_mut(&state.sequence().get())
            .and_then(|state| {
                state
                    .snapshots
                    .iter_mut()
                    .find(|slot| slot.id == reservation_id && slot.ready)
            })
            .ok_or(CachePoolError::StaleHandle)?;
        if tokens.len() != snapshot.prefix_length as usize {
            return Err(CachePoolError::StaleHandle);
        }
        if snapshot.history_write_end > snapshot.prefix_length
            && self
                .layout
                .groups()
                .iter()
                .any(|group| group.retention().window_tokens().is_some())
        {
            // The copied STATE is immutable, but the accompanying WINDOW ring
            // remained writable for the whole continuation lease. Without a
            // separately pinned historical ring, an older prefix is optional
            // cache data we must discard rather than mislabel newer KV.
            snapshot.ready = false;
            return Ok(());
        }
        let history_demands: Vec<_> = self
            .layout
            .groups()
            .iter()
            .enumerate()
            .map(|(index, group)| {
                GroupDemand::new(
                    GroupId::new(index as u32),
                    if group.is_recurrent() {
                        0
                    } else {
                        group.active_blocks(snapshot.prefix_length).1 as usize
                    },
                )
            })
            .collect();
        let history = self.pool.share_request_prefix(state, &history_demands)?;
        let copied = self
            .pool
            .share_request_prefix(&mut snapshot.owner, &self.state_demands)?;
        let pages = self
            .pool
            .combine_state_snapshots(history, copied, &self.state_groups)?;
        snapshot.ready = false;
        if let Err(error) = self.store_prefix(&pages, tokens) {
            self.pool.retire_cached_prefix(pages)?;
            return Err(error);
        }
        let replaced = self
            .prefix_cache
            .as_mut()
            .ok_or(CachePoolError::StaleHandle)?
            .insert(tokens, pages);
        if let Some(replaced) = replaced {
            self.pool.retire_cached_prefix(replaced)?;
        }
        Ok(())
    }

    fn store_prefix(&mut self, pages: &CachedPages, tokens: &[u32]) -> Result<(), CachePoolError> {
        if self.storage.is_some() {
            let tables = pages
                .groups()
                .iter()
                .map(|group| {
                    let mut table = group
                        .segments()
                        .flat_map(|(_, pages)| pages.iter().rev().map(|page| page.handle()))
                        .collect::<Vec<_>>();
                    table.reverse();
                    table
                })
                .collect();
            self.storage_transfer(tokens, tables, false)?;
        }
        Ok(())
    }

    fn storage_transfer(
        &mut self,
        tokens: &[u32],
        tables: Vec<Vec<PageHandle>>,
        load: bool,
    ) -> Result<bool, CachePoolError> {
        let mut groups = Vec::new();
        for (group, table) in self.layout.groups().iter().zip(tables) {
            let (first, count) = group.active_blocks(tokens.len() as u32);
            let mut addresses = Vec::new();
            for logical in first..first + count {
                let page = page_for(*group, &table, logical).ok_or(CachePoolError::StaleHandle)?;
                addresses.push(self.pool.physical_address(*page)?);
            }
            groups.push((first, addresses));
        }
        let storage = self.storage.as_mut().unwrap();
        Ok(storage.transfer(tokens, &groups, load))
    }

    fn demands(&self, context: u32) -> Vec<GroupDemand> {
        self.layout
            .groups()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, group)| {
                let (_, blocks) = group.active_blocks(context);
                GroupDemand::new(GroupId::new(index as u32), blocks as usize)
            })
            .collect()
    }

    pub fn release(&mut self, lease: WriterAccessLease) {
        assert!(self.batch_active, "released a nonexistent KV batch");
        self.pool.release_writer_access(lease);
        self.batch_active = false;
    }

    fn invalid(message: impl Into<String>) -> PagedKvError {
        PagedKvError::Invalid(message.into())
    }

    pub fn acquire(
        &mut self,
        requests: &[RequestResourceRef<'_, RequestPages>],
        steps: &[SequenceStep],
        decode_write_lookahead: u32,
    ) -> Result<BatchAccess<PagedKvBatchView, WriterAccessLease>, PagedKvError> {
        if self.batch_active {
            return Err(Self::invalid(
                "cannot acquire a KV batch before the current batch is released",
            ));
        }
        self.clear_drained_snapshots()?;
        if requests.is_empty() || requests.len() != steps.len() {
            return Err(Self::invalid(
                "KV requests and steps must have equal nonzero lengths",
            ));
        }
        for step in steps {
            let post_length = step
                .token_end()
                .ok_or_else(|| Self::invalid("scheduled token range overflows u32"))?;
            let leased_length = post_length
                .checked_add(if step.phase == crate::Phase::Decode {
                    decode_write_lookahead
                } else {
                    0
                })
                .ok_or_else(|| Self::invalid("decode cache write lookahead overflows u32"))?;
            self.validate_write_span(step.token_start(), leased_length)?;
        }
        for request in requests {
            self.pool.validate_request(request.state)?;
            if request.state.sequence().get() != request.request_id.get() {
                return Err(Self::invalid(format!(
                    "request {} cache pages belong to sequence {}",
                    request.request_id.get(),
                    request.state.sequence().get()
                )));
            }
            if request.state.groups().len() != self.layout.groups().len() {
                return Err(Self::invalid(format!(
                    "request {} has the wrong cache group count",
                    request.request_id.get()
                )));
            }
            for (index, table) in request.state.groups().iter().enumerate() {
                if table.group() != GroupId::new(index as u32) {
                    return Err(Self::invalid(format!(
                        "request {} cache groups are out of order",
                        request.request_id.get()
                    )));
                }
            }
        }
        self.reserve_state_snapshots(requests, steps, decode_write_lookahead)?;

        let append_count = steps
            .iter()
            .map(|step| step.token_count().get() as usize)
            .sum();
        let mut groups = Vec::with_capacity(self.layout.groups().len());
        let mut claims = Vec::<PageAccessClaim>::new();
        let mut shared_reads = HashSet::new();
        for (group_index, group) in self.layout.groups().iter().copied().enumerate() {
            let mut indptr = Vec::with_capacity(requests.len() + 1);
            let mut block_slab_ids = Vec::new();
            let mut block_slotnums = Vec::new();
            let mut first_blocks = Vec::with_capacity(requests.len());
            let mut append_slab_ids = Vec::with_capacity(append_count);
            let mut append_slotnums = Vec::with_capacity(append_count);
            let mut append_offsets = Vec::with_capacity(append_count);
            let mut state_plan = group.is_recurrent().then(|| StatePlan {
                restore_copies: Vec::new(),
                checkpoint_slots: Vec::new(),
            });
            indptr.push(0);

            for (row, (request, step)) in requests.iter().zip(steps).enumerate() {
                let table = &request.state.groups()[group_index];
                let post_length = step
                    .token_end()
                    .ok_or_else(|| Self::invalid("scheduled token range overflows u32"))?;
                let leased_length = if step.phase == crate::Phase::Decode {
                    post_length
                        .checked_add(decode_write_lookahead)
                        .ok_or_else(|| {
                            Self::invalid("decode cache write lookahead overflows u32")
                        })?
                } else {
                    post_length
                };
                let (first, count) = group.active_blocks(leased_length);
                if count as usize > table.pages().len() {
                    return Err(Self::invalid(format!(
                        "request {} cache group {group_index} needs {count} blocks but reserved {}",
                        request.request_id.get(),
                        table.pages().len()
                    )));
                }
                first_blocks.push(first);
                for logical in first..first + count {
                    let page = page_for(group, table.pages(), logical).ok_or_else(|| {
                        Self::invalid(format!(
                            "request {} cache group {group_index} is missing logical block {logical}",
                            request.request_id.get()
                        ))
                    })?;
                    let (slab_id, slotnum) = self.pool.physical_address(page.handle())?;
                    block_slab_ids.push(slab_id);
                    block_slotnums.push(slotnum);
                    // Every request was authenticated above. A shared prefix
                    // needs one read pin for the batch, even when hundreds of
                    // rows expose that same history in their device tables.
                    // Keep every write claim so conflicting writers still fail
                    // prepare_writer_access's transactional duplicate check.
                    if page.is_writable() || shared_reads.insert((slab_id, slotnum)) {
                        claims.push(request.state.access_claim(page));
                    }
                }
                indptr.push(
                    u32::try_from(block_slab_ids.len()).map_err(|_| {
                        Self::invalid("cache block table exceeds the u32 CSR range")
                    })?,
                );
                if let Some(state) = &mut state_plan {
                    let resources = &self.state_resources[&request.request_id.get()];
                    if resources.pending {
                        let source = if let Some(owner) = &resources.restore {
                            let page = &owner.groups()[group_index].pages()[0];
                            let source = self.pool.physical_address(page.handle())?;
                            if shared_reads.insert(source) {
                                claims.push(owner.access_claim(page));
                            }
                            source
                        } else {
                            (0, 0)
                        };
                        state.restore_copies.push(StateCopy {
                            row: row as u32,
                            source_slab_id: source.0,
                            source_slotnum: source.1,
                        });
                    }
                    for snapshot in &resources.snapshots {
                        let page = &snapshot.owner.groups()[group_index].pages()[0];
                        let (destination_slab_id, destination_slotnum) =
                            self.pool.physical_address(page.handle())?;
                        claims.push(snapshot.owner.access_claim(page));
                        state.checkpoint_slots.push(StateCheckpointSlot {
                            reservation_id: snapshot.id,
                            row: row as u32,
                            prefix_length: snapshot.prefix_length,
                            destination_slab_id,
                            destination_slotnum,
                        });
                    }
                }

                for position in step.token_start()..post_length {
                    let logical = position / group.block_tokens().get();
                    let page = page_for(group, table.pages(), logical).ok_or_else(|| {
                        Self::invalid(format!(
                            "request {} cache group {group_index} is missing logical block {logical}",
                            request.request_id.get()
                        ))
                    })?;
                    let (slab_id, slotnum) = self.pool.physical_address(page.handle())?;
                    append_slab_ids.push(slab_id);
                    append_slotnums.push(slotnum);
                    append_offsets.push(if group.is_recurrent() {
                        0
                    } else {
                        position % group.block_tokens().get()
                    });
                }
            }
            let mut plan = KvGroupPlan::new(
                group_index as u32,
                indptr,
                block_slab_ids,
                block_slotnums,
                first_blocks,
                append_slab_ids,
                append_slotnums,
                append_offsets,
                group.blocks_per_slab(),
            )?;
            plan.state = state_plan;
            plan.validate()?;
            groups.push(plan);
        }

        let prepared = self.pool.prepare_writer_access(claims)?;
        let lease = self.pool.acquire_writer_access(prepared)?;
        self.batch_active = true;
        Ok(BatchAccess {
            view: PagedKvBatchView {
                pool_id: self.pool.id(),
                groups,
            },
            lease,
        })
    }

    fn reserve_state_snapshots(
        &mut self,
        requests: &[RequestResourceRef<'_, RequestPages>],
        steps: &[SequenceStep],
        lookahead: u32,
    ) -> Result<(), PagedKvError> {
        let Some(alignment) = self.prefix_alignment() else {
            return Ok(());
        };
        if self.state_demands.is_empty() {
            return Ok(());
        }
        for (request, step) in requests.iter().zip(steps) {
            let end = step
                .token_end()
                .ok_or_else(|| Self::invalid("state checkpoint range overflow"))?;
            let history_write_end = end
                .checked_add(if step.phase == crate::Phase::Decode {
                    lookahead
                } else {
                    0
                })
                .ok_or_else(|| Self::invalid("state checkpoint range overflow"))?;
            let first = if step.phase == crate::Phase::Prefill {
                if !end.is_multiple_of(alignment) {
                    continue;
                }
                u64::from(end)
            } else {
                (u64::from(step.token_start()) / u64::from(alignment) + 1) * u64::from(alignment)
            };
            let snapshots = &mut self
                .state_resources
                .get_mut(&request.request_id.get())
                .unwrap()
                .snapshots;
            for prefix_length in (0..2)
                .map(|index| first + index * u64::from(alignment))
                .take_while(|&length| length <= u64::from(history_write_end))
            {
                let Some(id) = self.last_snapshot_id.checked_add(1) else {
                    break;
                };
                let pending = match self
                    .pool
                    .reserve_request(request.state.sequence(), &self.state_demands)
                {
                    Ok(owner) => owner,
                    // Snapshot caching is optional and cannot evict live work.
                    Err(
                        CachePoolError::InsufficientCapacity { .. }
                        | CachePoolError::MetadataAllocation
                        | CachePoolError::WriterIdExhausted,
                    ) => break,
                    Err(error) => return Err(error.into()),
                };
                self.last_snapshot_id = id;
                snapshots.push(StateSnapshotReservation {
                    id,
                    prefix_length: prefix_length as u32,
                    history_write_end,
                    owner: self.pool.promote_request(pending),
                    ready: false,
                });
            }
        }
        Ok(())
    }
}

pub(crate) struct AdmissionEstimate<'a> {
    layout: &'a KvLayout,
    blocks: Vec<u64>,
    occupied_parents: Vec<HashSet<u32>>,
    shared: HashSet<PageHandle>,
}

impl AdmissionEstimate<'_> {
    pub(crate) fn include(
        &mut self,
        pages: Option<&RequestPages>,
        context: u32,
    ) -> Result<u64, CachePoolError> {
        let mut parents = 0u64;
        for (index, group) in self.layout.groups().iter().copied().enumerate() {
            let count = group.active_blocks(context).1;
            let mut added = u64::from(count);
            if let Some(pages) = pages {
                for page in pages.groups()[index].pages().iter().take(count as usize) {
                    self.occupied_parents[index].insert(page.handle().parent());
                    if group.retention() == KvRetention::FullContext
                        && !group.is_recurrent()
                        && !page.is_writable()
                        && !self.shared.insert(page.handle())
                    {
                        added -= 1;
                    }
                }
            }
            self.blocks[index] = self.blocks[index]
                .checked_add(added)
                .ok_or(CachePoolError::DemandOverflow)?;
            // Live pages cannot relocate to achieve the ideal packing. Their
            // parent identities provide a floor even for fragmented prefixes.
            let group_parents = self.blocks[index]
                .div_ceil(u64::from(group.blocks_per_slab().get()))
                .max(self.occupied_parents[index].len() as u64);
            parents = parents
                .checked_add(group_parents)
                .ok_or(CachePoolError::DemandOverflow)?;
        }
        Ok(parents)
    }
}

fn prefix_alignment(layout: &KvLayout) -> Result<u32, CachePoolError> {
    fn gcd(mut left: u32, mut right: u32) -> u32 {
        while right != 0 {
            (left, right) = (right, left % right);
        }
        left
    }
    layout.groups().iter().try_fold(1u32, |alignment, group| {
        alignment
            .checked_mul(group.block_tokens().get() / gcd(alignment, group.block_tokens().get()))
            .ok_or(CachePoolError::DemandOverflow)
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagedKvBatchView {
    pub pool_id: u64,
    pub groups: Vec<KvGroupPlan>,
}

#[derive(Debug)]
pub enum PagedKvError {
    Cache(CachePoolError),
    Invalid(String),
}

impl fmt::Display for PagedKvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cache(error) => error.fmt(formatter),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for PagedKvError {}

impl From<CachePoolError> for PagedKvError {
    fn from(error: CachePoolError) -> Self {
        Self::Cache(error)
    }
}

impl From<KvPlanError> for PagedKvError {
    fn from(error: KvPlanError) -> Self {
        Self::Invalid(error.to_string())
    }
}

fn page_for<R>(group: KvGroupLayout, pages: &[R], logical: u32) -> Option<&R> {
    page_index(group, pages.len(), logical).and_then(|index| pages.get(index))
}

fn page_index(group: KvGroupLayout, page_count: usize, logical: u32) -> Option<usize> {
    if page_count == 0 {
        return None;
    }
    if group.is_recurrent() {
        return Some(0);
    }
    let logical = logical as usize;
    let index =
        if group.is_recurrent() || matches!(group.retention(), KvRetention::SlidingWindow { .. }) {
            logical % page_count
        } else {
            logical
        };
    (index < page_count).then_some(index)
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroU64};
    use std::sync::{Arc, Mutex};

    use crate::{Phase, TokenRange};

    use super::*;

    type TestArena = HashMap<(usize, u32, u32), [u8; 8]>;
    type TestObjects = HashMap<Vec<u32>, Vec<Vec<[u8; 8]>>>;

    #[derive(Clone, Debug, Default)]
    struct ByteStorage {
        arena: Arc<Mutex<TestArena>>,
        objects: Arc<Mutex<TestObjects>>,
    }

    impl PrefixStorage for ByteStorage {
        fn lookup(&mut self, tokens: &[u32], maximum: u32, alignment: u32) -> u32 {
            self.objects
                .lock()
                .unwrap()
                .keys()
                .filter(|key| {
                    tokens.starts_with(key)
                        && key.len() <= maximum as usize
                        && key.len().is_multiple_of(alignment as usize)
                })
                .map(|key| key.len() as u32)
                .max()
                .unwrap_or(0)
        }

        fn transfer(&mut self, tokens: &[u32], groups: &[PrefixStorageGroup], load: bool) -> bool {
            let mut arena = self.arena.lock().unwrap();
            let mut objects = self.objects.lock().unwrap();
            if load {
                let Some(object) = objects.get(tokens) else {
                    return false;
                };
                for (index, ((_, addresses), blocks)) in groups.iter().zip(object).enumerate() {
                    assert_eq!(addresses.len(), blocks.len());
                    for (&(slab, slot), &bytes) in addresses.iter().zip(blocks) {
                        arena.insert((index, slab, slot), bytes);
                    }
                }
            } else {
                objects.insert(
                    tokens.to_vec(),
                    groups
                        .iter()
                        .enumerate()
                        .map(|(index, (_, addresses))| {
                            addresses
                                .iter()
                                .map(|&(slab, slot)| arena[&(index, slab, slot)])
                                .collect()
                        })
                        .collect(),
                );
            }
            true
        }
    }

    fn storage_roundtrip(recurrent: bool) {
        let mut groups = vec![KvGroupLayout::new(2, 2, KvRetention::FullContext).unwrap()];
        if recurrent {
            groups.push(
                KvGroupLayout::new_with_family(
                    2,
                    2,
                    KvRetention::FullContext,
                    crate::resources::kv::KvGroupFamily::State,
                )
                .unwrap(),
            );
        }
        let mut resources =
            PagedKvResources::with_prefix_cache(KvLayout::new(8, groups).unwrap()).unwrap();
        let storage = ByteStorage::default();
        resources.storage = Some(Box::new(storage.clone()));
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut seed = resources.attach(id, 0).unwrap();
        resources.ensure_context(&mut seed, 0, 4).unwrap();
        for (index, group) in seed.groups().iter().enumerate() {
            for (block, page) in group.pages().iter().enumerate() {
                let (slab, slot) = resources.pool.physical_address(page.handle()).unwrap();
                storage
                    .arena
                    .lock()
                    .unwrap()
                    .insert((index, slab, slot), [(index * 16 + block + 1) as u8; 8]);
            }
        }
        if recurrent {
            let access = resources
                .acquire(
                    &[RequestResourceRef {
                        request_id: id,
                        state: &seed,
                    }],
                    &[SequenceStep {
                        phase: Phase::Prefill,
                        tokens: TokenRange {
                            start: 0,
                            count: NonZeroU32::new(4).unwrap(),
                        },
                        sample: true,
                    }],
                    0,
                )
                .unwrap();
            let slot = access.view.groups[1]
                .state
                .as_ref()
                .unwrap()
                .checkpoint_slots
                .iter()
                .find(|slot| slot.prefix_length == 4)
                .unwrap();
            // The immutable snapshot deliberately differs from live state.
            storage.arena.lock().unwrap().insert(
                (1, slot.destination_slab_id, slot.destination_slotnum),
                [0xa5; 8],
            );
            let receipt = slot.reservation_id;
            resources
                .accept_state_checkpoints(&[(id.get(), 4, 4, false)], &[receipt])
                .unwrap();
            resources.finish_state_initialization(&[id.get()]);
            resources.release(access.lease);
            resources
                .publish_state_prefix(&mut seed, receipt, &[1, 2, 3, 4])
                .unwrap();
        } else {
            assert!(
                resources
                    .publish_prefix(&mut seed, &[1, 2, 3, 4], 4)
                    .unwrap()
            );
        }
        let expected = storage.objects.lock().unwrap()[&vec![1, 2, 3, 4]].clone();
        resources.retire(seed).unwrap();
        resources.clear_prefix_cache().unwrap();
        assert_eq!(resources.pool.free_parent_count(), 8);
        storage.arena.lock().unwrap().clear();

        let id = RequestId::new(NonZeroU64::new(2).unwrap());
        let mut restored = resources.attach(id, 0).unwrap();
        assert_eq!(
            resources
                .restore_prefix(&mut restored, &[1, 2, 3, 4, 5])
                .unwrap(),
            4
        );
        for (index, group) in restored.groups().iter().enumerate() {
            let pages = if recurrent && index == 1 {
                assert!(group.pages()[0].is_writable());
                resources.state_resources[&id.get()]
                    .restore
                    .as_ref()
                    .unwrap()
                    .groups()[index]
                    .pages()
            } else {
                group.pages()
            };
            for (block, page) in pages.iter().enumerate() {
                let (slab, slot) = resources.pool.physical_address(page.handle()).unwrap();
                assert_eq!(
                    storage.arena.lock().unwrap()[&(index, slab, slot)],
                    expected[index][block]
                );
            }
        }
        resources.retire(restored).unwrap();
        resources.clear_prefix_cache().unwrap();
        assert_eq!(resources.pool.free_parent_count(), 8);
    }

    #[test]
    fn storage_restores_history_bytes_after_all_device_pages_are_recycled() {
        storage_roundtrip(false);
    }

    #[test]
    fn storage_restores_recurrent_snapshot_bytes_with_private_live_state() {
        storage_roundtrip(true);
    }

    #[test]
    fn ancestor_eviction_counts_no_reclaimed_storage_until_the_last_checkpoint() {
        let layout = KvLayout::new(
            2,
            vec![KvGroupLayout::new(2, 2, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut request = resources.attach(id, 0).unwrap();
        resources.ensure_context(&mut request, 0, 2).unwrap();
        resources.publish_prefix(&mut request, &[1, 2], 2).unwrap();
        resources.ensure_context(&mut request, 2, 4).unwrap();
        resources
            .publish_prefix(&mut request, &[1, 2, 3, 4], 4)
            .unwrap();
        assert_eq!(resources.metrics().groups[0].reclaimable_blocks, 0);
        resources.retire(request).unwrap();

        let cached = resources.metrics();
        assert_eq!(cached.reclaimable_slabs, 1);
        assert_eq!(cached.groups[0].reclaimable_blocks, 2);
        assert!(resources.evict_one_prefix().unwrap());
        let ancestor = resources.metrics();
        assert_eq!(ancestor.groups[0].occupied_blocks, 2);
        assert_eq!(ancestor.prefix.unwrap().reclaimed_blocks, 0);

        assert!(resources.evict_one_prefix().unwrap());
        let empty = resources.metrics();
        assert_eq!(empty.free_slabs, 2);
        assert_eq!(empty.prefix.unwrap().reclaimed_blocks, 2);
        assert_eq!(empty.prefix.unwrap().reclaimed_slabs, 1);
    }

    #[test]
    fn layout_estimate_matches_fresh_pool_allocation() {
        let layout = KvLayout::new(
            20,
            vec![
                KvGroupLayout::new(4, 2, KvRetention::FullContext).unwrap(),
                KvGroupLayout::new(
                    4,
                    4,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(5).unwrap(),
                    },
                )
                .unwrap(),
                KvGroupLayout::new_with_family(
                    8,
                    2,
                    KvRetention::FullContext,
                    super::super::kv::KvGroupFamily::State,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let contexts = [0, 1, 4, 5, 17];
        let expected = layout.estimated_parent_count(&contexts).unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        assert_eq!(
            resources.estimated_parent_count(&contexts).unwrap(),
            expected
        );
        let states: Vec<_> = contexts
            .into_iter()
            .enumerate()
            .map(|(index, context)| {
                resources
                    .attach(
                        RequestId::new(NonZeroU64::new(index as u64 + 1).unwrap()),
                        context,
                    )
                    .unwrap()
            })
            .collect();
        assert_eq!(
            u64::from(resources.parent_capacity()) - resources.pool().free_parent_count() as u64,
            expected
        );
        for state in states {
            resources.retire(state).unwrap();
        }
        assert_eq!(resources.pool().free_parent_count(), 20);
    }

    #[test]
    fn admission_preserves_fragmented_shared_prefix_parents() {
        let layout = KvLayout::new(
            8,
            vec![
                KvGroupLayout::new(1, 8, KvRetention::FullContext).unwrap(),
                KvGroupLayout::new(8, 1, KvRetention::FullContext).unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = |value| RequestId::new(NonZeroU64::new(value).unwrap());
        let demands = |blocks| {
            [
                GroupDemand::new(GroupId::new(0), blocks),
                GroupDemand::new(GroupId::new(1), 0),
            ]
        };
        let mut seed = resources.attach(id(1), 0).unwrap();
        let mut fillers = Vec::new();
        for blocks in 1..=3 {
            resources
                .pool
                .grow_request(&mut seed, &demands(blocks))
                .unwrap();
            // Fill each parent before growing the prefix into another parent.
            let mut filler = resources.attach(id(10 + blocks as u64), 0).unwrap();
            resources
                .pool
                .grow_request(&mut filler, &demands(7))
                .unwrap();
            fillers.push(filler);
        }
        let prefix = resources
            .pool
            .share_request_prefix(&mut seed, &demands(3))
            .unwrap();
        for filler in fillers {
            resources.retire(filler).unwrap();
        }
        resources.retire(seed).unwrap();

        let mut requests = Vec::new();
        for index in 0..2 {
            let mut request = resources.attach(id(20 + index), 0).unwrap();
            resources
                .pool
                .attach_shared_prefix(&mut request, &prefix)
                .unwrap();
            resources.ensure_context(&mut request, 3, 4).unwrap();
            requests.push(request);
        }
        let mut estimate = resources.admission_estimate();
        assert_eq!(estimate.include(Some(&requests[0]), 4).unwrap(), 4);
        assert_eq!(estimate.include(Some(&requests[1]), 4).unwrap(), 5);
        // Three immutable child handles count once, but pin three parents.
        // Each request also owns one private child and a second-group parent.
        assert_eq!(estimate.blocks, [5, 2]);
        assert_eq!(resources.estimated_parent_count(&[4, 4]).unwrap(), 3);
        assert_eq!(8 - resources.pool().free_parent_count(), 5);
        drop(estimate);
        for request in requests {
            resources.retire(request).unwrap();
        }
        resources.pool.retire_cached_prefix(prefix).unwrap();
        assert_eq!(resources.pool().free_parent_count(), 8);
    }

    #[test]
    fn full_and_sliding_groups_share_one_current_batch_lease() {
        let layout = KvLayout::new(
            8,
            vec![
                KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap(),
                KvGroupLayout::new(
                    2,
                    2,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(7).unwrap(),
                    },
                )
                .unwrap()
                .with_read_window_tokens(4)
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let state = resources.attach(id, 12).unwrap();
        let step = SequenceStep {
            phase: Phase::Prefill,
            tokens: TokenRange {
                start: 0,
                count: NonZeroU32::new(4).unwrap(),
            },
            sample: false,
        };
        let requests = [RequestResourceRef {
            request_id: id,
            state: &state,
        }];
        let access = resources.acquire(&requests, &[step], 0).unwrap();
        assert_eq!(access.view.groups.len(), 2);
        assert!(resources.acquire(&requests, &[step], 0).is_err());
        resources.release(access.lease);
        resources.retire(state).unwrap();
    }

    #[test]
    fn request_pages_grow_on_demand_and_reset_for_replay() {
        let layout = KvLayout::new(
            2,
            vec![KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut state = resources.attach(id, 0).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 0);

        resources.ensure_context(&mut state, 0, 5).unwrap();
        assert_eq!(state.groups()[0].pages().len(), 2);
        assert_eq!(resources.pool().occupied_block_count(), 2);
        assert!(matches!(
            resources.ensure_context(&mut state, 5, 9),
            Err(CachePoolError::InsufficientCapacity { .. })
        ));
        assert_eq!(state.groups()[0].pages().len(), 2);

        resources.reset(&mut state).unwrap();
        assert!(state.groups()[0].pages().is_empty());
        assert_eq!(resources.pool().occupied_block_count(), 0);
        resources.ensure_context(&mut state, 0, 4).unwrap();
        assert_eq!(state.groups()[0].pages().len(), 1);
        resources.retire(state).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 0);
    }

    #[test]
    fn decode_write_lookahead_is_leased_but_not_reported_as_input_append() {
        let layout = KvLayout::new(
            2,
            vec![KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut state = resources.attach(id, 0).unwrap();
        resources.ensure_context(&mut state, 4, 8).unwrap();
        let step = SequenceStep {
            phase: Phase::Decode,
            tokens: TokenRange {
                start: 4,
                count: NonZeroU32::new(1).unwrap(),
            },
            sample: true,
        };
        let access = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: id,
                    state: &state,
                }],
                &[step],
                3,
            )
            .unwrap();
        let group = &access.view.groups[0];
        assert_eq!(group.block_slab_ids.len(), 2);
        assert_eq!(group.block_slotnums.len(), 2);
        assert_eq!(group.append_slab_ids, vec![group.block_slab_ids[1]]);
        assert_eq!(group.append_slotnums, vec![group.block_slotnums[1]]);
        assert_eq!(group.append_block_offsets, vec![0]);
        resources.release(access.lease);
        resources.retire(state).unwrap();
    }

    #[test]
    fn immutable_history_prefix_is_shared_and_suffix_stays_writable() {
        let layout = KvLayout::new(
            4,
            vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let first_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut first = resources.attach(first_id, 0).unwrap();
        resources.ensure_context(&mut first, 0, 4).unwrap();
        let write = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: first_id,
                    state: &first,
                }],
                &[SequenceStep {
                    phase: Phase::Prefill,
                    tokens: TokenRange {
                        start: 0,
                        count: NonZeroU32::new(4).unwrap(),
                    },
                    sample: false,
                }],
                0,
            )
            .unwrap();
        resources.release(write.lease);
        assert!(
            resources
                .publish_prefix(&mut first, &[10, 11, 12, 13], 4)
                .unwrap()
        );

        let second_id = RequestId::new(NonZeroU64::new(2).unwrap());
        let mut second = resources.attach(second_id, 0).unwrap();
        assert_eq!(
            resources
                .restore_prefix(&mut second, &[10, 11, 12, 13, 14])
                .unwrap(),
            4
        );
        assert_eq!(resources.pool().occupied_block_count(), 2);
        resources.ensure_context(&mut second, 4, 5).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 3);
        let access = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: second_id,
                    state: &second,
                }],
                &[SequenceStep {
                    phase: Phase::Prefill,
                    tokens: TokenRange {
                        start: 4,
                        count: NonZeroU32::new(1).unwrap(),
                    },
                    sample: true,
                }],
                0,
            )
            .unwrap();
        assert_eq!(access.view.groups[0].block_slab_ids.len(), 3);
        resources.release(access.lease);
        assert_eq!(
            resources.prefix_metrics(),
            Some(PrefixCacheMetrics {
                entries: 1,
                hits: 1,
                ..PrefixCacheMetrics::default()
            })
        );
        resources.retire(second).unwrap();
        resources.retire(first).unwrap();
    }

    #[test]
    fn shared_batch_read_claims_keep_all_tables_and_pin_retired_owners() {
        let layout = KvLayout::new(
            4,
            vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let seed_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut seed = resources.attach(seed_id, 0).unwrap();
        resources.ensure_context(&mut seed, 0, 4).unwrap();
        resources
            .publish_prefix(&mut seed, &[10, 11, 12, 13], 4)
            .unwrap();
        resources.retire(seed).unwrap();

        let mut owners = Vec::new();
        for value in 2..=3 {
            let id = RequestId::new(NonZeroU64::new(value).unwrap());
            let mut state = resources.attach(id, 0).unwrap();
            assert_eq!(
                resources
                    .restore_prefix(&mut state, &[10, 11, 12, 13, 20])
                    .unwrap(),
                4
            );
            resources.ensure_context(&mut state, 4, 5).unwrap();
            owners.push((id, state));
        }
        let steps = [SequenceStep {
            phase: Phase::Decode,
            tokens: TokenRange {
                start: 4,
                count: NonZeroU32::new(1).unwrap(),
            },
            sample: true,
        }; 2];
        let requests = owners
            .iter()
            .map(|(id, state)| RequestResourceRef {
                request_id: *id,
                state,
            })
            .collect::<Vec<_>>();
        let access = resources.acquire(&requests, &steps, 0).unwrap();
        assert_eq!(access.view.groups[0].block_indptr, [0, 3, 6]);
        let addresses = &access.view.groups[0].block_slab_ids;
        assert_eq!(&addresses[..2], &addresses[3..5]);
        assert_ne!(addresses[2], addresses[5]);
        assert_eq!(access.lease.len(), 4); // Two shared reads, two distinct writers.
        drop(requests);
        for (_, state) in owners {
            resources.retire(state).unwrap();
        }
        let pinned = resources.metrics();
        assert_eq!(pinned.reclaimable_slabs, 0);
        assert_eq!(pinned.groups[0].cached_blocks, 2);
        assert_eq!(pinned.groups[0].reclaimable_blocks, 0);
        assert_eq!(pinned.groups[0].pinned_blocks, 4);
        resources.clear_prefix_cache().unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 4);
        let evicted = resources.metrics();
        assert_eq!(evicted.groups[0].cached_blocks, 0);
        assert_eq!(evicted.groups[0].pinned_blocks, 4);
        assert_eq!(evicted.prefix.unwrap().reclaimed_blocks, 0);
        assert_eq!(evicted.prefix.unwrap().reclaimed_slabs, 0);
        resources.release(access.lease);
        assert_eq!(resources.pool().free_parent_count(), 4);
        assert_eq!(resources.metrics().groups[0].pinned_blocks, 0);
    }

    #[test]
    fn batch_shared_read_deduplication_never_hides_duplicate_writers() {
        let layout = KvLayout::new(
            1,
            vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let state = resources.attach(id, 2).unwrap();
        let requests = [
            RequestResourceRef {
                request_id: id,
                state: &state,
            },
            RequestResourceRef {
                request_id: id,
                state: &state,
            },
        ];
        let steps = [SequenceStep {
            phase: Phase::Decode,
            tokens: TokenRange {
                start: 0,
                count: NonZeroU32::new(1).unwrap(),
            },
            sample: true,
        }; 2];
        assert!(matches!(
            resources.acquire(&requests, &steps, 0),
            Err(PagedKvError::Cache(CachePoolError::DuplicateAccessClaim))
        ));
        assert!(!resources.batch_active);
        resources.retire(state).unwrap();
        assert_eq!(resources.pool().free_parent_count(), 1);
    }

    #[test]
    fn shared_sliding_snapshot_is_copy_on_write_when_ring_advances() {
        let layout = KvLayout::new(
            5,
            vec![
                KvGroupLayout::new(
                    2,
                    1,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(4).unwrap(),
                    },
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let first_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut first = resources.attach(first_id, 0).unwrap();
        for position in 0..6 {
            resources
                .ensure_context(&mut first, position, position + 1)
                .unwrap();
        }
        let write = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: first_id,
                    state: &first,
                }],
                &[SequenceStep {
                    phase: Phase::Decode,
                    tokens: TokenRange {
                        start: 5,
                        count: NonZeroU32::new(1).unwrap(),
                    },
                    sample: false,
                }],
                0,
            )
            .unwrap();
        resources.release(write.lease);
        resources
            .publish_prefix(&mut first, &[1, 2, 3, 4, 5, 6], 6)
            .unwrap();

        let second_id = RequestId::new(NonZeroU64::new(2).unwrap());
        let mut second = resources.attach(second_id, 0).unwrap();
        assert_eq!(
            resources
                .restore_prefix(&mut second, &[1, 2, 3, 4, 5, 6, 7])
                .unwrap(),
            6
        );
        resources.ensure_context(&mut second, 6, 7).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 4);
        assert!(second.groups()[0].pages()[0].is_writable());
        resources.retire(second).unwrap();
        resources.retire(first).unwrap();
    }

    #[test]
    fn unsafe_window_write_is_rejected_before_copy_on_write() {
        let layout = KvLayout::new(
            20,
            vec![
                KvGroupLayout::new(
                    2,
                    1,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(4).unwrap(),
                    },
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut state = resources.attach(id, 0).unwrap();
        for position in 0..6 {
            resources
                .ensure_context(&mut state, position, position + 1)
                .unwrap();
        }
        resources
            .publish_prefix(&mut state, &(0..6).collect::<Vec<_>>(), 6)
            .unwrap();
        let original = state.groups()[0]
            .pages()
            .iter()
            .map(|page| page.handle())
            .collect::<Vec<_>>();
        assert_eq!(
            resources.ensure_context(&mut state, 6, 12),
            Err(CachePoolError::InvalidWriteSpan {
                requested: 6,
                maximum: 1
            })
        );
        assert_eq!(
            state.groups()[0]
                .pages()
                .iter()
                .map(|page| page.handle())
                .collect::<Vec<_>>(),
            original
        );
        assert_eq!(resources.pool().occupied_block_count(), 3);
        resources.retire(state).unwrap();
    }

    #[test]
    fn chunked_window_prefill_keeps_early_query_history_after_prefix_cow() {
        let group = KvGroupLayout::new(
            2,
            1,
            KvRetention::SlidingWindow {
                tokens: NonZeroU32::new(9).unwrap(),
            },
        )
        .unwrap()
        .with_read_window_tokens(4)
        .unwrap();
        let mut resources =
            PagedKvResources::with_prefix_cache(KvLayout::new(20, vec![group]).unwrap()).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut state = resources.attach(id, 0).unwrap();
        resources.ensure_context(&mut state, 0, 6).unwrap();
        let history = state.groups()[0]
            .pages()
            .iter()
            .map(|page| resources.pool().physical_address(page.handle()).unwrap())
            .collect::<Vec<_>>();
        resources
            .publish_prefix(&mut state, &(0..6).collect::<Vec<_>>(), 6)
            .unwrap();
        resources.ensure_context(&mut state, 6, 12).unwrap();
        let access = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: id,
                    state: &state,
                }],
                &[SequenceStep {
                    phase: Phase::Prefill,
                    tokens: TokenRange {
                        start: 6,
                        count: NonZeroU32::new(6).unwrap(),
                    },
                    sample: false,
                }],
                0,
            )
            .unwrap();
        let plan = &access.view.groups[0];
        assert_eq!(plan.block_table_start_indices, vec![1]);
        let addresses = plan
            .block_slab_ids
            .iter()
            .copied()
            .zip(plan.block_slotnums.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(&addresses[..2], &history[1..]);
        assert_eq!(addresses.len(), 5);
        for (offset, (&slab, &slot)) in plan
            .append_slab_ids
            .iter()
            .zip(&plan.append_slotnums)
            .enumerate()
        {
            assert_eq!((slab, slot), addresses[(6 + offset) / 2 - 1]);
            assert!(!history[1..].contains(&(slab, slot)));
        }
        resources.release(access.lease);
        resources.retire(state).unwrap();
    }

    #[test]
    fn live_allocation_evicts_lru_prefix_pages_under_pressure() {
        let layout = KvLayout::new(
            2,
            vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let cached_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut cached = resources.attach(cached_id, 0).unwrap();
        resources.ensure_context(&mut cached, 0, 4).unwrap();
        resources
            .publish_prefix(&mut cached, &[1, 2, 3, 4], 4)
            .unwrap();
        resources.retire(cached).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 2);

        let live_id = RequestId::new(NonZeroU64::new(2).unwrap());
        let mut live = resources.attach(live_id, 0).unwrap();
        resources.ensure_context(&mut live, 0, 4).unwrap();
        let prefix_metrics = resources.prefix_metrics().unwrap();
        assert_eq!(prefix_metrics.entries, 0);
        assert_eq!(prefix_metrics.evictions, 1);
        assert_eq!(resources.pool().occupied_block_count(), 2);
        resources.retire(live).unwrap();
    }

    #[test]
    fn recurrent_prefix_keeps_input_shared_and_allocates_a_writable_output() {
        let layout = KvLayout::new(
            10,
            vec![
                KvGroupLayout::new(16, 1, KvRetention::FullContext).unwrap(),
                KvGroupLayout::new_with_family(
                    128,
                    6,
                    KvRetention::FullContext,
                    crate::resources::kv::KvGroupFamily::State,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let first_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut first = resources.attach(first_id, 0).unwrap();
        resources.ensure_context(&mut first, 0, 128).unwrap();
        // Native owns a distinct snapshot destination; acknowledgement never
        // freezes the live mutable state page.
        let access = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: first_id,
                    state: &first,
                }],
                &[SequenceStep {
                    phase: Phase::Prefill,
                    tokens: TokenRange {
                        start: 0,
                        count: NonZeroU32::new(128).unwrap(),
                    },
                    sample: true,
                }],
                0,
            )
            .unwrap();
        let state_plan = access.view.groups[1].state.as_ref().unwrap();
        assert_eq!(state_plan.restore_copies[0].source_slab_id, 0);
        let live = (
            access.view.groups[1].block_slab_ids[0],
            access.view.groups[1].block_slotnums[0],
        );
        let slot = &state_plan.checkpoint_slots[0];
        let snapshot_address = (slot.destination_slab_id, slot.destination_slotnum);
        assert_ne!(live, snapshot_address);
        let receipt = slot.reservation_id;
        let progress = [(first_id.get(), 128, 128, false)];
        assert!(
            resources
                .accept_state_checkpoints(&progress, &[receipt + 1])
                .is_err()
        );
        assert!(
            resources
                .accept_state_checkpoints(&[(first_id.get(), 127, 128, false)], &[receipt])
                .is_err()
        );
        assert!(
            resources
                .accept_state_checkpoints(&progress, &[receipt, receipt])
                .is_err()
        );
        assert!(
            resources
                .accept_state_checkpoints(&[(first_id.get() + 1, 128, 128, false)], &[receipt])
                .is_err()
        );
        assert!(
            resources
                .accept_state_checkpoints(&progress, &[receipt, receipt + 1])
                .is_err()
        );
        assert!(resources.ready_state_checkpoints(first_id.get()).is_empty());
        resources
            .accept_state_checkpoints(&progress, &[receipt])
            .unwrap();
        resources.finish_state_initialization(&[first_id.get()]);
        resources.release(access.lease);
        resources
            .publish_state_prefix(&mut first, receipt, &(0..128).collect::<Vec<_>>())
            .unwrap();
        assert!(first.groups()[1].pages()[0].is_writable());
        assert_eq!(
            resources
                .pool
                .physical_address(first.groups()[1].pages()[0].handle())
                .unwrap(),
            live
        );
        resources.retire(first).unwrap();

        let second_id = RequestId::new(NonZeroU64::new(2).unwrap());
        let mut second = resources.attach(second_id, 0).unwrap();
        assert_eq!(
            resources
                .restore_prefix(&mut second, &(0..129).collect::<Vec<_>>())
                .unwrap(),
            128
        );
        resources.ensure_context(&mut second, 128, 129).unwrap();
        let state_pages = second.groups()[1].pages();
        assert_eq!(state_pages.len(), 1);
        assert!(state_pages[0].is_writable());

        let access = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: second_id,
                    state: &second,
                }],
                &[SequenceStep {
                    phase: Phase::Prefill,
                    tokens: TokenRange {
                        start: 128,
                        count: NonZeroU32::new(1).unwrap(),
                    },
                    sample: true,
                }],
                0,
            )
            .unwrap();
        let state = &access.view.groups[1];
        assert_eq!(state.block_slab_ids.len(), 1);
        assert_eq!(state.block_slotnums.len(), 1);
        assert_eq!(state.append_slab_ids, vec![state.block_slab_ids[0]]);
        assert_eq!(state.append_slotnums, vec![state.block_slotnums[0]]);
        let copy = &state.state.as_ref().unwrap().restore_copies[0];
        assert_eq!((copy.source_slab_id, copy.source_slotnum), snapshot_address);
        assert_ne!(
            (state.block_slab_ids[0], state.block_slotnums[0]),
            snapshot_address
        );
        resources.release(access.lease);
        resources.retire(second).unwrap();
    }

    #[test]
    fn private_live_state_is_stable_across_checkpoint_boundaries() {
        let layout = KvLayout::new(
            2,
            vec![
                KvGroupLayout::new_with_family(
                    128,
                    6,
                    KvRetention::FullContext,
                    crate::resources::kv::KvGroupFamily::State,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut state = resources.attach(id, 0).unwrap();

        resources.ensure_context(&mut state, 0, 257).unwrap();

        assert_eq!(state.groups()[0].pages().len(), 1);
        let live = state.groups()[0].pages()[0].handle();
        resources.ensure_context(&mut state, 257, 385).unwrap();
        assert_eq!(state.groups()[0].pages()[0].handle(), live);
        assert!(
            state.groups()[0]
                .pages()
                .iter()
                .all(|page| page.is_writable())
        );
        resources.retire(state).unwrap();
    }

    #[test]
    fn state_initialization_retries_until_a_successful_completion() {
        let layout = KvLayout::new(
            1,
            vec![
                KvGroupLayout::new_with_family(
                    4,
                    1,
                    KvRetention::FullContext,
                    super::super::kv::KvGroupFamily::State,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::new(layout).unwrap();
        let id = RequestId::new(NonZeroU64::new(1).unwrap());
        let state = resources.attach(id, 1).unwrap();
        for attempt in 0..3 {
            let access = resources
                .acquire(
                    &[RequestResourceRef {
                        request_id: id,
                        state: &state,
                    }],
                    &[SequenceStep {
                        phase: Phase::Prefill,
                        tokens: TokenRange {
                            start: 0,
                            count: NonZeroU32::new(1).unwrap(),
                        },
                        sample: true,
                    }],
                    0,
                )
                .unwrap();
            let copies = &access.view.groups[0].state.as_ref().unwrap().restore_copies;
            assert_eq!(copies.len(), usize::from(attempt < 2));
            if attempt == 1 {
                resources.finish_state_initialization(&[id.get()]);
            }
            // Attempt0 models a pre-accept rejection, which must not consume
            // the zero/restore instruction. Attempt1 actually commits it.
            resources.release(access.lease);
        }
        resources.retire(state).unwrap();
        assert_eq!(resources.pool().free_parent_count(), 1);
    }

    #[test]
    fn private_restore_capacity_miss_keeps_the_request_empty() {
        let layout = KvLayout::new(
            2,
            vec![
                KvGroupLayout::new_with_family(
                    2,
                    1,
                    KvRetention::FullContext,
                    super::super::kv::KvGroupFamily::State,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
        let first_id = RequestId::new(NonZeroU64::new(1).unwrap());
        let mut first = resources.attach(first_id, 2).unwrap();
        let access = resources
            .acquire(
                &[RequestResourceRef {
                    request_id: first_id,
                    state: &first,
                }],
                &[SequenceStep {
                    phase: Phase::Prefill,
                    tokens: TokenRange {
                        start: 0,
                        count: NonZeroU32::new(2).unwrap(),
                    },
                    sample: true,
                }],
                0,
            )
            .unwrap();
        let slot = &access.view.groups[0]
            .state
            .as_ref()
            .unwrap()
            .checkpoint_slots[0];
        let receipt = slot.reservation_id;
        resources
            .accept_state_checkpoints(&[(first_id.get(), 2, 2, false)], &[receipt])
            .unwrap();
        resources.finish_state_initialization(&[first_id.get()]);
        resources.release(access.lease);
        resources
            .publish_state_prefix(&mut first, receipt, &[1, 2])
            .unwrap();
        resources.retire(first).unwrap();

        let blocker = resources
            .attach(RequestId::new(NonZeroU64::new(2).unwrap()), 1)
            .unwrap();
        let id = RequestId::new(NonZeroU64::new(3).unwrap());
        let mut request = resources.attach(id, 0).unwrap();
        assert_eq!(
            resources.restore_prefix(&mut request, &[1, 2, 3]).unwrap(),
            0
        );
        assert!(request.groups()[0].pages().is_empty());
        assert!(resources.state_resources[&id.get()].restore.is_none());
        assert_eq!(resources.prefix_metrics().unwrap().entries, 1);
        resources.retire(blocker).unwrap();
        resources.ensure_context(&mut request, 0, 1).unwrap();
        assert!(request.groups()[0].pages()[0].is_writable());
        resources.retire(request).unwrap();
        resources.clear_prefix_cache().unwrap();
        assert_eq!(resources.pool().free_parent_count(), 2);
    }

    #[test]
    fn delayed_window_state_checkpoint_cannot_publish_a_reused_ring() {
        for delayed in [false, true] {
            let layout = KvLayout::new(
                6,
                vec![
                    KvGroupLayout::new(
                        2,
                        1,
                        KvRetention::SlidingWindow {
                            tokens: NonZeroU32::new(5).unwrap(),
                        },
                    )
                    .unwrap()
                    .with_read_window_tokens(2)
                    .unwrap(),
                    KvGroupLayout::new_with_family(
                        4,
                        1,
                        KvRetention::FullContext,
                        super::super::kv::KvGroupFamily::State,
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
            let mut resources = PagedKvResources::with_prefix_cache(layout).unwrap();
            let id = RequestId::new(NonZeroU64::new(1).unwrap());
            let mut request = resources.attach(id, 0).unwrap();
            let (phase, start, count, lookahead) = if delayed {
                // Checkpoint4 is reached inside a lease permitted to write
                // through6. Its logical block3 reuses ring block0 afterward.
                resources.ensure_context(&mut request, 3, 7).unwrap();
                (Phase::Decode, 3, 1, 3)
            } else {
                resources.ensure_context(&mut request, 0, 4).unwrap();
                (Phase::Prefill, 0, 4, 0)
            };
            let access = resources
                .acquire(
                    &[RequestResourceRef {
                        request_id: id,
                        state: &request,
                    }],
                    &[SequenceStep {
                        phase,
                        tokens: TokenRange {
                            start,
                            count: NonZeroU32::new(count).unwrap(),
                        },
                        sample: true,
                    }],
                    lookahead,
                )
                .unwrap();
            let slot = &access.view.groups[1]
                .state
                .as_ref()
                .unwrap()
                .checkpoint_slots[0];
            let receipt = slot.reservation_id;
            // Reset cannot drop pending copies/snapshots while the lease is live.
            assert!(matches!(
                resources.reset(&mut request),
                Err(CachePoolError::AccessConflict)
            ));
            assert_eq!(resources.state_resources[&id.get()].snapshots.len(), 1);
            resources
                .accept_state_checkpoints(&[(id.get(), 4, 4, false)], &[receipt])
                .unwrap();
            resources.finish_state_initialization(&[id.get()]);
            resources.release(access.lease);
            resources
                .publish_state_prefix(&mut request, receipt, &[1, 2, 3, 4])
                .unwrap();
            assert_eq!(
                resources.prefix_metrics().unwrap().entries,
                u64::from(!delayed)
            );
            assert!(request.groups()[1].pages()[0].is_writable());
            resources.retire(request).unwrap();
            resources.clear_prefix_cache().unwrap();
            assert_eq!(resources.pool().free_parent_count(), 6);
        }
    }
}
