//! Immutable Python snapshots of native cache ownership and prefix LRU state.

use pyo3::prelude::*;
use resin_core::resources::paged_kv;

/// Shared physical cache capacity. Reclaimable slabs exclude live requests
/// and outstanding device access, including retired overlapped executions.
#[pyclass(frozen, get_all, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct CacheMetrics {
    pub(crate) capacity_slabs: u32,
    pub(crate) free_slabs: u32,
    pub(crate) occupied_slabs: u32,
    pub(crate) reclaimable_slabs: u32,
    pub(crate) groups: Vec<CacheGroupMetrics>,
    pub(crate) prefix: Option<PrefixCacheMetrics>,
}

impl From<paged_kv::CacheMetrics> for CacheMetrics {
    fn from(value: paged_kv::CacheMetrics) -> Self {
        Self {
            capacity_slabs: value.capacity_slabs,
            free_slabs: value.free_slabs,
            occupied_slabs: value.occupied_slabs,
            reclaimable_slabs: value.reclaimable_slabs,
            groups: value.groups.into_iter().map(Into::into).collect(),
            prefix: value.prefix.map(Into::into),
        }
    }
}

/// Physical usage for one runner-declared group in the unified cache pool.
/// Groups share slab capacity; they are not separate capacity partitions.
#[pyclass(frozen, get_all, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct CacheGroupMetrics {
    pub(crate) group: u32,
    /// ``kv``, ``swa``, or ``recurrent``.
    pub(crate) pool: &'static str,
    pub(crate) block_tokens: u32,
    pub(crate) blocks_per_slab: u32,
    pub(crate) occupied_slabs: u32,
    pub(crate) reclaimable_slabs: u32,
    pub(crate) occupied_blocks: u64,
    /// Blocks held by cached prefixes, including those shared with requests.
    pub(crate) cached_blocks: u64,
    /// Blocks freed by evicting every cached prefix with requests unchanged.
    pub(crate) reclaimable_blocks: u64,
    /// Blocks with outstanding device read or write leases.
    pub(crate) pinned_blocks: u64,
    /// Unoccupied slots in slabs already bound to this group.
    pub(crate) free_blocks: u64,
    /// Block-rounded token capacity, or ``None`` for recurrent checkpoints.
    pub(crate) occupied_tokens: Option<u64>,
    pub(crate) reclaimable_tokens: Option<u64>,
}

impl From<paged_kv::CacheGroupMetrics> for CacheGroupMetrics {
    fn from(value: paged_kv::CacheGroupMetrics) -> Self {
        Self {
            group: value.group,
            pool: value.pool,
            block_tokens: value.block_tokens,
            blocks_per_slab: value.blocks_per_slab,
            occupied_slabs: value.occupied_slabs,
            reclaimable_slabs: value.reclaimable_slabs,
            occupied_blocks: value.occupied_blocks,
            cached_blocks: value.cached_blocks,
            reclaimable_blocks: value.reclaimable_blocks,
            pinned_blocks: value.pinned_blocks,
            free_blocks: value.free_blocks,
            occupied_tokens: value.occupied_tokens,
            reclaimable_tokens: value.reclaimable_tokens,
        }
    }
}

/// Prefix LRU statistics. Reclaimed counters measure physical frees during
/// eviction; removing a shared ancestor or a live request's prefix can free zero.
#[pyclass(frozen, get_all, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefixCacheMetrics {
    pub(crate) entries: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) evictions: u64,
    pub(crate) reclaimed_blocks: u64,
    pub(crate) reclaimed_slabs: u64,
}

impl From<paged_kv::PrefixCacheMetrics> for PrefixCacheMetrics {
    fn from(value: paged_kv::PrefixCacheMetrics) -> Self {
        Self {
            entries: value.entries,
            hits: value.hits,
            misses: value.misses,
            evictions: value.evictions,
            reclaimed_blocks: value.reclaimed_blocks,
            reclaimed_slabs: value.reclaimed_slabs,
        }
    }
}
