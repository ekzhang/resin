use std::collections::HashSet;
use std::fmt;
use std::num::NonZeroU32;

use crate::{RequestId, SequenceStep};

use super::kv::{KvGroupLayout, KvGroupPlan, KvLayout, KvPlanError, KvRetention};
use super::paged::{
    CachePool, CachePoolError, GroupDemand, GroupId, PageAccessClaim, PageHandle, RequestPages,
    RetireRequestFailure, SequenceId, WriterAccessLease,
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

#[derive(Debug)]
pub struct PagedKvResources {
    layout: KvLayout,
    pool: CachePool,
    prefix_cache: Option<PrefixCache<super::paged::CachedPages>>,
    prefix_reclaimed_blocks: u64,
    prefix_reclaimed_slabs: u64,
    batch_active: bool,
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
        Ok(Self {
            layout,
            pool,
            prefix_cache,
            prefix_reclaimed_blocks: 0,
            prefix_reclaimed_slabs: 0,
            batch_active: false,
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
        Ok(self.pool.promote_request(pending))
    }

    /// Grow one request just enough to address `context` tokens.
    pub fn ensure_context(
        &mut self,
        state: &mut RequestPages,
        token_start: u32,
        context: u32,
    ) -> Result<(), CachePoolError> {
        self.validate_write_span(token_start, context)?;
        let demands = self.demands(context);
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
            return Ok(0);
        };
        pool.attach_shared_prefix(state, snapshot)?;
        Ok(length)
    }

    /// Publish a completed aligned resident boundary. Returns false when the
    /// current progress cannot form a safe block-complete snapshot.
    pub fn publish_prefix(
        &mut self,
        state: &mut RequestPages,
        tokens: &[u32],
        resident_tokens: u32,
    ) -> Result<bool, CachePoolError> {
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
        self.pool.retire_request(state)
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
            indptr.push(0);

            for (request, step) in requests.iter().zip(steps) {
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
                    append_offsets.push(position % group.block_tokens().get());
                }
            }
            groups.push(KvGroupPlan::new(
                group_index as u32,
                indptr,
                block_slab_ids,
                block_slotnums,
                first_blocks,
                append_slab_ids,
                append_slotnums,
                append_offsets,
                group.blocks_per_slab(),
            )?);
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

    use crate::{Phase, TokenRange};

    use super::*;

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
        assert!(
            resources
                .publish_prefix(&mut first, &(0..128).collect::<Vec<_>>(), 128)
                .unwrap()
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
        assert_eq!(state_pages.len(), 2);
        assert!(!state_pages[0].is_writable());
        assert!(state_pages[1].is_writable());

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
        assert_eq!(state.block_slab_ids.len(), 2);
        assert_eq!(state.block_slotnums.len(), 2);
        assert_eq!(state.append_slab_ids, vec![state.block_slab_ids[1]]);
        assert_eq!(state.append_slotnums, vec![state.block_slotnums[1]]);
        resources.release(access.lease);
        resources.retire(second).unwrap();
    }

    #[test]
    fn recurrent_context_wraps_across_the_third_logical_block() {
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

        assert_eq!(state.groups()[0].pages().len(), 2);
        assert!(
            state.groups()[0]
                .pages()
                .iter()
                .all(|page| page.is_writable())
        );
        resources.retire(state).unwrap();
    }
}
