//! Fungible physical placement for heterogeneous paged-token resources.
//!
//! The pool owns equal-sized physical parents.  An empty parent may serve any
//! cache group; while occupied, it is bound to one group and subdivided into
//! that group's configured number of child blocks.  Token retention, request
//! page-table policy, prefix identity, and device byte layout deliberately live
//! outside this module.

use std::collections::VecDeque;
use std::fmt;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// Ordered cache-group identity.  The runner declaration and native layout use
/// the same index; there is no process-global group registry.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GroupId(u32);

impl GroupId {
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Stable identity of one live child block. This is a copyable capability, not
/// ownership; request ownership stays in `PendingRequestPages`, `RequestPages`,
/// or an active writer lease.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PageHandle {
    pool_id: u64,
    group: GroupId,
    parent: u32,
    slot: u32,
    generation: u64,
}

/// Pool-local identity of the one sequence allowed to mutate private pages.
///
/// The Engine may use its already-unique request id. Keeping the type distinct
/// prevents a page handle or group index from being substituted accidentally.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SequenceId(u64);

impl SequenceId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Pool-minted, non-reusing identity of one request reservation. Logical
/// sequence ids are caller metadata and are not trusted as write authority.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct WriterId(u64);

impl PageHandle {
    /// One-based physical parent id.  Parent zero is reserved for the device
    /// null page and is never allocated.
    pub(crate) const fn parent(self) -> u32 {
        self.parent
    }

    #[cfg(test)]
    pub const fn slot(self) -> u32 {
        self.slot
    }

    #[cfg(test)]
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupDemand {
    group: GroupId,
    blocks: usize,
}

impl GroupDemand {
    pub const fn new(group: GroupId, blocks: usize) -> Self {
        Self { group, blocks }
    }
}

/// One ordered physical table for one input demand. Duplicate group demands
/// retain distinct tables even though placement shares partially filled
/// parents.
#[derive(Debug)]
pub struct GroupTable<R> {
    group: GroupId,
    pages: Vec<R>,
    /// A non-owning hint for sharing an unchanged published prefix. Ownership
    /// stays exclusively with cache snapshots, so eviction can reclaim it.
    snapshot: Weak<CachedSegment>,
}

impl<R> GroupTable<R> {
    pub const fn group(&self) -> GroupId {
        self.group
    }

    pub fn pages(&self) -> &[R] {
        &self.pages
    }
}

/// Move-only ownership of one request page. A raw handle may be copied for
/// metadata, but only this token accounts for the request reference.
#[derive(Debug)]
pub struct PageRef {
    handle: PageHandle,
    writable: bool,
}

impl PageRef {
    pub const fn handle(&self) -> PageHandle {
        self.handle
    }

    pub const fn is_writable(&self) -> bool {
        self.writable
    }
}

/// Move-only ownership of an immutable prefix snapshot. Pages may be shared by
/// any number of requests, but the issuing pool remains responsible for every
/// reference and for eviction.
#[derive(Debug)]
#[must_use = "cached pages must be retired through their issuing pool"]
pub struct CachedPages {
    pool_id: u64,
    groups: Vec<CachedGroupTable>,
}

impl CachedPages {
    pub fn groups(&self) -> &[CachedGroupTable] {
        &self.groups
    }
}

/// An immutable page table whose earlier segments may belong to several
/// checkpoints. A physical cache reference belongs to a segment, not to every
/// snapshot of the growing table.
#[derive(Debug)]
pub struct CachedGroupTable {
    group: GroupId,
    tail: Option<Arc<CachedSegment>>,
}

impl CachedGroupTable {
    pub const fn group(&self) -> GroupId {
        self.group
    }

    pub fn len(&self) -> usize {
        self.tail.as_ref().map_or(0, |tail| tail.len())
    }

    pub fn is_empty(&self) -> bool {
        self.tail.is_none()
    }

    /// Segments in reverse chronological order, with their absolute table
    /// offsets. No temporary flattened table is allocated.
    pub fn segments(&self) -> impl Iterator<Item = (usize, &[PageRef])> {
        cached_segments(self.tail.as_deref())
            .map(|segment| (segment.start, segment.pages.as_slice()))
    }

    fn pages_rev(&self) -> impl Iterator<Item = &PageRef> {
        self.segments().flat_map(|(_, pages)| pages.iter().rev())
    }
}

struct CachedSegment {
    start: usize,
    pages: Vec<PageRef>,
    previous: Option<Arc<CachedSegment>>,
}

impl fmt::Debug for CachedSegment {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CachedSegment")
            .field("start", &self.start)
            .field("pages", &self.pages)
            .field(
                "previous_len",
                &self.previous.as_ref().map(|previous| previous.len()),
            )
            .finish()
    }
}

impl CachedSegment {
    fn len(&self) -> usize {
        self.start + self.pages.len()
    }
}

fn cached_segments(tail: Option<&CachedSegment>) -> impl Iterator<Item = &CachedSegment> {
    std::iter::successors(tail, |segment| segment.previous.as_deref())
}

impl Drop for CachedSegment {
    fn drop(&mut self) {
        // Explicit pool retirement handles physical refs. Even when an owner
        // is abandoned or the entire engine is dropped, free metadata without
        // recursing through a potentially long checkpoint chain.
        let mut previous = self.previous.take();
        while let Some(segment) = previous {
            match Arc::try_unwrap(segment) {
                Ok(mut segment) => previous = segment.previous.take(),
                Err(_) => break,
            }
        }
    }
}

/// Provisional ownership returned by one all-or-nothing request reservation.
#[derive(Debug)]
#[must_use = "pending request pages must be promoted or aborted"]
pub struct PendingRequestPages {
    pool_id: u64,
    sequence: SequenceId,
    writer: WriterId,
    groups: Vec<GroupTable<PageRef>>,
}

impl PendingRequestPages {
    #[cfg(test)]
    pub const fn sequence(&self) -> SequenceId {
        self.sequence
    }

    #[cfg(test)]
    pub fn groups(&self) -> &[GroupTable<PageRef>] {
        &self.groups
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.groups.iter().map(|group| group.pages.len()).sum()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Move-only ownership of all physical pages belonging to one live request.
#[derive(Debug)]
#[must_use = "request pages must be retired through their issuing pool"]
pub struct RequestPages {
    pool_id: u64,
    sequence: SequenceId,
    writer: WriterId,
    groups: Vec<GroupTable<PageRef>>,
}

impl RequestPages {
    pub const fn sequence(&self) -> SequenceId {
        self.sequence
    }

    pub fn groups(&self) -> &[GroupTable<PageRef>] {
        &self.groups
    }

    /// Construct an exact-writer claim. Passing a page from another request is
    /// harmless: pool validation rejects the substituted writer atomically.
    pub const fn access_claim(&self, page: &PageRef) -> PageAccessClaim {
        PageAccessClaim {
            pool_id: self.pool_id,
            writer: self.writer,
            handle: page.handle,
            write: page.writable,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageAccessClaim {
    pool_id: u64,
    writer: WriterId,
    handle: PageHandle,
    write: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedWriterClaim {
    claim: PageAccessClaim,
    expected_write_epoch: u64,
}

#[derive(Debug)]
#[must_use = "prepared writer access must be acquired or discarded"]
pub struct PreparedWriterAccess {
    pool_id: u64,
    claims: Vec<PreparedWriterClaim>,
}

#[derive(Debug)]
#[must_use = "writer access must be released or retained as quarantine"]
pub struct WriterAccessLease {
    pool_id: u64,
    claims: Vec<PreparedWriterClaim>,
}

impl WriterAccessLease {
    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CachePoolError {
    InvalidParentCapacity,
    EmptyGroupLayout,
    TooManyGroups,
    PoolIdExhausted,
    WriterIdExhausted,
    MetadataAllocation,
    UnknownGroup {
        group: GroupId,
    },
    DemandOverflow,
    InsufficientCapacity {
        group: GroupId,
        requested: usize,
        available: usize,
    },
    InvalidWriteSpan {
        requested: u32,
        maximum: u32,
    },
    ParentGenerationExhausted {
        parent: u32,
    },
    ForeignPool,
    StaleHandle,
    ParentGroupMismatch {
        handle_group: GroupId,
        bound_group: GroupId,
    },
    DuplicateHandle,
    ForeignAccess,
    WrongWriter,
    PageNotWritable,
    DuplicateAccessClaim,
    AccessConflict,
    WriteEpochExhausted,
    WriteEpochChanged {
        expected: u64,
        actual: u64,
    },
}

impl fmt::Display for CachePoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParentCapacity => {
                formatter.write_str("cache pool parent capacity must be positive")
            }
            Self::EmptyGroupLayout => {
                formatter.write_str("cache pool requires at least one cache group")
            }
            Self::TooManyGroups => formatter.write_str("cache pool has too many groups"),
            Self::PoolIdExhausted => formatter.write_str("cache pool id space is exhausted"),
            Self::WriterIdExhausted => {
                formatter.write_str("cache pool writer id space is exhausted")
            }
            Self::MetadataAllocation => {
                formatter.write_str("could not allocate cache-pool metadata")
            }
            Self::UnknownGroup { group } => {
                write!(formatter, "cache group {} is not declared", group.index())
            }
            Self::DemandOverflow => formatter.write_str("cache block demand overflows usize"),
            Self::InsufficientCapacity {
                group,
                requested,
                available,
            } => write!(
                formatter,
                "cache group {} requested {requested} blocks but only {available} fit",
                group.index()
            ),
            Self::InvalidWriteSpan { requested, maximum } => write!(
                formatter,
                "cache write span {requested} exceeds window-safe maximum {maximum}"
            ),
            Self::ParentGenerationExhausted { parent } => {
                write!(formatter, "cache parent {parent} generation is exhausted")
            }
            Self::ForeignPool => {
                formatter.write_str("cache allocation belongs to a different pool")
            }
            Self::StaleHandle => formatter.write_str("cache page handle is stale"),
            Self::ParentGroupMismatch {
                handle_group,
                bound_group,
            } => write!(
                formatter,
                "cache page names group {}, but its parent is bound to group {}",
                handle_group.index(),
                bound_group.index()
            ),
            Self::DuplicateHandle => {
                formatter.write_str("cache allocation contains a duplicate page handle")
            }
            Self::ForeignAccess => formatter.write_str("writer access belongs to another pool"),
            Self::WrongWriter => {
                formatter.write_str("cache page belongs to a different request writer")
            }
            Self::PageNotWritable => {
                formatter.write_str("cache page has no exclusive mutable request owner")
            }
            Self::DuplicateAccessClaim => {
                formatter.write_str("one cache page has incompatible duplicate access claims")
            }
            Self::AccessConflict => {
                formatter.write_str("cache page already has an incompatible access lease")
            }
            Self::WriteEpochExhausted => {
                formatter.write_str("cache page write completion epoch is exhausted")
            }
            Self::WriteEpochChanged { expected, actual } => write!(
                formatter,
                "cache page write epoch changed from {expected} to {actual} after access preparation"
            ),
        }
    }
}

impl std::error::Error for CachePoolError {}

/// A failed retirement returns the exact move-only request token unchanged.
#[derive(Debug)]
pub struct RetireRequestFailure {
    pub error: CachePoolError,
    pub request: RequestPages,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PageMode {
    Reserved { writer: WriterId },
    Mutable { writer: WriterId },
    Shared,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PageSlot {
    generation: u64,
    mode: Option<PageMode>,
    request_refs: u32,
    /// Immutable segment ownership is separate from live request ownership.
    /// Several checkpoints can share one segment reference.
    cache_refs: u32,
    reader_leases: u32,
    writer_lease: bool,
    write_epoch: u64,
}

impl PageSlot {
    fn is_live(self) -> bool {
        matches!(self.mode, Some(PageMode::Reserved { .. }))
            || self.request_refs != 0
            || self.cache_refs != 0
            || self.reader_leases != 0
            || self.writer_lease
    }

    fn reserved_request(generation: u64, writer: WriterId) -> Self {
        Self {
            generation,
            mode: Some(PageMode::Reserved { writer }),
            ..Self::default()
        }
    }

    fn clear_lifetime(&mut self) {
        debug_assert!(!self.is_live());
        self.mode = None;
        self.request_refs = 0;
        self.cache_refs = 0;
        self.reader_leases = 0;
        self.writer_lease = false;
        self.write_epoch = 0;
    }
}

/// Physical inventory for one group. Reclaimable means that retiring every
/// cache reference would free the block now, with no request or device lease
/// still holding it. Free blocks are holes in already bound slabs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GroupOccupancy {
    pub occupied_slabs: u32,
    pub reclaimable_slabs: u32,
    pub occupied_blocks: u64,
    pub cached_blocks: u64,
    pub reclaimable_blocks: u64,
    pub pinned_blocks: u64,
    pub free_blocks: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Parent {
    binding: Option<GroupId>,
    slots: Vec<PageSlot>,
    occupied_slots: u32,
    /// Last generation issued anywhere in this parent.  It survives slot
    /// reuse, slot-vector clearing, and cross-group rebinding.
    next_generation: u64,
    /// Position in the pool's compact list of partially filled parents.
    /// Full and empty parents are absent.
    partial_index: Option<usize>,
}

#[derive(Debug)]
struct PlannedSlot {
    slot: u32,
    generation: u64,
    writer: WriterId,
}

#[derive(Debug)]
enum PlannedStorage {
    /// Only touched holes are recorded for an already-bound parent.
    Existing { updates: Vec<PlannedSlot> },
    /// A newly bound parent's complete slot vector is prepared off-pool and
    /// moved into place during the infallible commit.
    New { slots: Vec<PageSlot> },
}

#[derive(Debug)]
struct PlannedParent {
    parent: u32,
    group: GroupId,
    next_generation: u64,
    occupied_after: u32,
    next_slot: usize,
    storage: PlannedStorage,
}

impl PlannedParent {
    fn existing(parent_id: u32, parent: &Parent, group: GroupId) -> Self {
        debug_assert_eq!(parent.binding, Some(group));
        debug_assert!(parent.partial_index.is_some());
        Self {
            parent: parent_id,
            group,
            next_generation: parent.next_generation,
            occupied_after: parent.occupied_slots,
            next_slot: 0,
            storage: PlannedStorage::Existing {
                updates: Vec::new(),
            },
        }
    }

    fn new(
        parent_id: u32,
        parent: &Parent,
        group: GroupId,
        packing: u32,
    ) -> Result<Self, CachePoolError> {
        debug_assert!(parent.binding.is_none());
        debug_assert_eq!(parent.occupied_slots, 0);
        debug_assert!(parent.slots.is_empty());
        debug_assert!(parent.partial_index.is_none());

        let mut slots = Vec::new();
        slots
            .try_reserve_exact(packing as usize)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        slots.resize(packing as usize, PageSlot::default());
        Ok(Self {
            parent: parent_id,
            group,
            next_generation: parent.next_generation,
            occupied_after: 0,
            next_slot: 0,
            storage: PlannedStorage::New { slots },
        })
    }

    fn is_full(&self, actual: &Parent) -> bool {
        self.occupied_after as usize == self.slot_count(actual)
    }

    fn slot_count(&self, actual: &Parent) -> usize {
        match &self.storage {
            PlannedStorage::Existing { .. } => actual.slots.len(),
            PlannedStorage::New { slots } => slots.len(),
        }
    }

    fn allocate_up_to(
        &mut self,
        actual: &Parent,
        pool_id: u64,
        writer: WriterId,
        requested: usize,
        pages: &mut Vec<PageRef>,
    ) -> Result<usize, CachePoolError> {
        debug_assert_eq!(self.group, actual.binding.unwrap_or(self.group));
        let available = self
            .slot_count(actual)
            .saturating_sub(self.occupied_after as usize);
        let target = requested.min(available);
        if target == 0 {
            return Ok(0);
        }

        if let PlannedStorage::Existing { updates } = &mut self.storage {
            updates
                .try_reserve(target)
                .map_err(|_| CachePoolError::MetadataAllocation)?;
        }

        let mut allocated = 0;
        while allocated != target {
            let slot_index = self.next_slot;
            let is_live = match &self.storage {
                PlannedStorage::Existing { .. } => actual.slots[slot_index].is_live(),
                PlannedStorage::New { slots } => slots[slot_index].is_live(),
            };
            self.next_slot += 1;
            if is_live {
                continue;
            }

            let generation = self.next_generation.checked_add(1).ok_or(
                CachePoolError::ParentGenerationExhausted {
                    parent: self.parent,
                },
            )?;
            self.next_generation = generation;
            let reserved = PageSlot::reserved_request(generation, writer);
            match &mut self.storage {
                PlannedStorage::Existing { updates } => updates.push(PlannedSlot {
                    slot: slot_index as u32,
                    generation,
                    writer,
                }),
                PlannedStorage::New { slots } => slots[slot_index] = reserved,
            }
            self.occupied_after += 1;
            allocated += 1;
            pages.push(PageRef {
                handle: PageHandle {
                    pool_id,
                    group: self.group,
                    parent: self.parent,
                    slot: slot_index as u32,
                    generation,
                },
                writable: true,
            });
        }
        Ok(allocated)
    }
}

#[derive(Debug)]
struct GroupPlanState {
    /// Next entry in the pool's compact partial-parent list to inspect.  Each
    /// demanded group filters that list at most once; with D demanded groups
    /// and P partial parents, this part of admission is O(D * P).
    next_partial_index: usize,
    /// The one touched parent which still has room.  Packing always fills a
    /// parent before advancing, so one cursor is sufficient per group.
    current: Option<usize>,
}

#[derive(Debug)]
struct PlacementPlan {
    parents: Vec<PlannedParent>,
    group_states: Vec<Option<GroupPlanState>>,
    new_parent_ids: Vec<u32>,
    block_count: usize,
}

impl PlacementPlan {
    fn new(
        group_count: usize,
        total_blocks: usize,
        parent_capacity: usize,
    ) -> Result<Self, CachePoolError> {
        let touched_bound = total_blocks.min(parent_capacity);
        let mut parents = Vec::new();
        parents
            .try_reserve(touched_bound)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        let mut group_states = Vec::new();
        group_states
            .try_reserve_exact(group_count)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        group_states.resize_with(group_count, || None);
        let mut new_parent_ids = Vec::new();
        new_parent_ids
            .try_reserve(touched_bound)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        Ok(Self {
            parents,
            group_states,
            new_parent_ids,
            block_count: 0,
        })
    }
}

/// One flat pool of fungible, equal-sized physical parents.
#[derive(Debug)]
pub struct CachePool {
    id: u64,
    /// Last writer identity committed to a successful reservation. It is
    /// monotonic for this pool and never rewound by abort or retirement.
    last_writer_id: u64,
    group_packings: Box<[NonZeroU32]>,
    parents: Vec<Parent>,
    free_parents: VecDeque<u32>,
    /// Compact unordered membership. Removal uses `swap_remove`, so placement
    /// promises dense packing but deliberately no stable partial-parent
    /// priority. Capacity is fixed up front for allocation-free teardown.
    partial_parents: Vec<u32>,
    occupied_blocks: usize,
    /// Reused for linear ownership validation. Capacity grows before request
    /// tables grow, so validation and retirement never allocate. The mutex
    /// preserves read-only validation through a shared pool reference.
    validation_handles: Mutex<Vec<PageHandle>>,
}

impl CachePool {
    pub fn new(
        parent_capacity: u32,
        group_packings: &[NonZeroU32],
    ) -> Result<Self, CachePoolError> {
        if parent_capacity == 0 {
            return Err(CachePoolError::InvalidParentCapacity);
        }
        if group_packings.is_empty() {
            return Err(CachePoolError::EmptyGroupLayout);
        }
        if group_packings.len() > u32::MAX as usize {
            return Err(CachePoolError::TooManyGroups);
        }
        let id = NEXT_POOL_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| CachePoolError::PoolIdExhausted)?;

        let parent_capacity =
            usize::try_from(parent_capacity).map_err(|_| CachePoolError::MetadataAllocation)?;
        let mut packings = Vec::new();
        packings
            .try_reserve_exact(group_packings.len())
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        packings.extend_from_slice(group_packings);

        let mut parents = Vec::new();
        parents
            .try_reserve_exact(parent_capacity)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        parents.resize_with(parent_capacity, Parent::default);

        let mut free_parents = VecDeque::new();
        free_parents
            .try_reserve_exact(parent_capacity)
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        free_parents.extend(1..=parent_capacity as u32);

        let mut partial_parents = Vec::new();
        partial_parents
            .try_reserve_exact(parent_capacity)
            .map_err(|_| CachePoolError::MetadataAllocation)?;

        Ok(Self {
            id,
            last_writer_id: 0,
            group_packings: packings.into_boxed_slice(),
            parents,
            free_parents,
            partial_parents,
            occupied_blocks: 0,
            validation_handles: Mutex::new(Vec::new()),
        })
    }

    pub const fn id(&self) -> u64 {
        self.id
    }

    pub fn blocks_per_slab(&self, group: GroupId) -> Result<NonZeroU32, CachePoolError> {
        self.group_packings
            .get(group.index() as usize)
            .copied()
            .ok_or(CachePoolError::UnknownGroup { group })
    }

    pub fn free_parent_count(&self) -> usize {
        self.free_parents.len()
    }

    pub const fn occupied_block_count(&self) -> usize {
        self.occupied_blocks
    }

    /// Inspect physical ownership only when publishing a metrics snapshot.
    /// This scans allocated metadata, never request token histories or the
    /// potentially shared prefix checkpoint chains.
    pub(crate) fn group_occupancy(&self) -> Vec<GroupOccupancy> {
        let mut groups = vec![GroupOccupancy::default(); self.group_packings.len()];
        for parent in &self.parents {
            let Some(group) = parent.binding else {
                continue;
            };
            let metrics = &mut groups[group.index() as usize];
            metrics.occupied_slabs += 1;
            metrics.occupied_blocks += u64::from(parent.occupied_slots);
            metrics.free_blocks += parent.slots.len() as u64 - u64::from(parent.occupied_slots);
            let mut reclaimable = 0;
            for slot in &parent.slots {
                metrics.cached_blocks += u64::from(slot.cache_refs != 0);
                let pinned = slot.reader_leases != 0 || slot.writer_lease;
                metrics.pinned_blocks += u64::from(pinned);
                if slot.cache_refs != 0 && slot.request_refs == 0 && !pinned {
                    reclaimable += 1;
                }
            }
            metrics.reclaimable_blocks += u64::from(reclaimable);
            metrics.reclaimable_slabs += u32::from(reclaimable == parent.occupied_slots);
        }
        groups
    }

    /// Resolve a live page capability into its device-facing physical address.
    /// Parent zero remains reserved for each runner's null page.
    pub fn physical_address(&self, handle: PageHandle) -> Result<(u32, u32), CachePoolError> {
        self.validate_live_handle(handle)?;
        Ok((handle.parent, handle.slot))
    }

    #[cfg(test)]
    pub fn bound_group(&self, parent: u32) -> Option<GroupId> {
        self.parents
            .get(parent.checked_sub(1)? as usize)
            .and_then(|parent| parent.binding)
    }

    /// Reserve every ordered group demand for one request atomically. Logical
    /// state is unchanged on every error; output tables preserve input order
    /// and multiplicity.
    pub fn reserve_request(
        &mut self,
        sequence: SequenceId,
        demands: &[GroupDemand],
    ) -> Result<PendingRequestPages, CachePoolError> {
        let writer = WriterId(
            self.last_writer_id
                .checked_add(1)
                .ok_or(CachePoolError::WriterIdExhausted)?,
        );
        let mut total_blocks = 0usize;
        for demand in demands {
            self.blocks_per_slab(demand.group)?;
            total_blocks = total_blocks
                .checked_add(demand.blocks)
                .ok_or(CachePoolError::DemandOverflow)?;
        }
        self.occupied_blocks
            .checked_add(total_blocks)
            .ok_or(CachePoolError::DemandOverflow)?;
        self.prepare_validation_capacity(total_blocks)?;

        let mut groups = Vec::new();
        groups
            .try_reserve_exact(demands.len())
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        for demand in demands {
            let mut pages = Vec::new();
            pages
                .try_reserve_exact(demand.blocks)
                .map_err(|_| CachePoolError::MetadataAllocation)?;
            groups.push(GroupTable {
                group: demand.group,
                pages,
                snapshot: Weak::new(),
            });
        }

        let mut plan =
            PlacementPlan::new(self.group_packings.len(), total_blocks, self.parents.len())?;
        for (demand, group) in demands.iter().zip(&mut groups) {
            self.plan_partition(&mut plan, writer, *demand, &mut group.pages)?;
        }
        debug_assert_eq!(
            groups.iter().map(|group| group.pages.len()).sum::<usize>(),
            total_blocks
        );
        debug_assert_eq!(plan.block_count, total_blocks);

        // Every fallible operation completed in prepared metadata.
        // Publishing touches only planned parents and performs no allocation.
        self.commit_plan(plan);
        self.last_writer_id = writer.0;
        Ok(PendingRequestPages {
            pool_id: self.id,
            sequence,
            writer,
            groups,
        })
    }

    /// Make a fully reserved request executable. Private token provenance makes
    /// this transition infallible.
    pub fn promote_request(&mut self, pending: PendingRequestPages) -> RequestPages {
        self.assert_pending_request(&pending);
        for page in pending.groups.iter().flat_map(|group| &group.pages) {
            let slot = self.slot_mut(page.handle);
            debug_assert_eq!(
                slot.mode,
                Some(PageMode::Reserved {
                    writer: pending.writer
                })
            );
            debug_assert_eq!(slot.request_refs, 0);
            slot.mode = Some(PageMode::Mutable {
                writer: pending.writer,
            });
            slot.request_refs = 1;
        }
        RequestPages {
            pool_id: pending.pool_id,
            sequence: pending.sequence,
            writer: pending.writer,
            groups: pending.groups,
        }
    }

    /// Grow an executable request to the supplied total block counts.
    ///
    /// Placement is all-or-nothing. Existing pages and request identity stay
    /// unchanged on failure.
    pub fn grow_request(
        &mut self,
        request: &mut RequestPages,
        demands: &[GroupDemand],
    ) -> Result<(), CachePoolError> {
        self.validate_request_idle(request)?;
        if demands.len() != request.groups.len()
            || demands
                .iter()
                .zip(&request.groups)
                .any(|(demand, group)| demand.group != group.group)
        {
            return Err(CachePoolError::StaleHandle);
        }
        let request_blocks = demands.iter().try_fold(0usize, |total, demand| {
            total
                .checked_add(demand.blocks)
                .ok_or(CachePoolError::DemandOverflow)
        })?;
        self.prepare_validation_capacity(request_blocks)?;
        let mut total_blocks = 0usize;
        let mut additions = Vec::new();
        additions
            .try_reserve_exact(demands.len())
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        for (demand, group) in demands.iter().zip(&mut request.groups) {
            let additional = demand
                .blocks
                .checked_sub(group.pages.len())
                .ok_or(CachePoolError::StaleHandle)?;
            group
                .pages
                .try_reserve_exact(additional)
                .map_err(|_| CachePoolError::MetadataAllocation)?;
            total_blocks = total_blocks
                .checked_add(additional)
                .ok_or(CachePoolError::DemandOverflow)?;
            let mut pages = Vec::new();
            pages
                .try_reserve_exact(additional)
                .map_err(|_| CachePoolError::MetadataAllocation)?;
            additions.push(GroupTable {
                group: demand.group,
                pages,
                snapshot: Weak::new(),
            });
        }
        if total_blocks == 0 {
            return Ok(());
        }
        self.occupied_blocks
            .checked_add(total_blocks)
            .ok_or(CachePoolError::DemandOverflow)?;

        let mut plan =
            PlacementPlan::new(self.group_packings.len(), total_blocks, self.parents.len())?;
        for ((addition, demand), group) in additions.iter_mut().zip(demands).zip(&request.groups) {
            let additional = demand.blocks - group.pages.len();
            self.plan_partition(
                &mut plan,
                request.writer,
                GroupDemand::new(addition.group, additional),
                &mut addition.pages,
            )?;
        }
        debug_assert_eq!(plan.block_count, total_blocks);

        self.commit_plan(plan);
        for page in additions.iter().flat_map(|group| &group.pages) {
            let slot = self.slot_mut(page.handle);
            debug_assert_eq!(
                slot.mode,
                Some(PageMode::Reserved {
                    writer: request.writer
                })
            );
            debug_assert_eq!(slot.request_refs, 0);
            slot.mode = Some(PageMode::Mutable {
                writer: request.writer,
            });
            slot.request_refs = 1;
        }
        for (group, mut addition) in request.groups.iter_mut().zip(additions) {
            group.pages.append(&mut addition.pages);
        }
        Ok(())
    }

    /// Freeze an ordered subset of an idle request's pages and retain one
    /// immutable cache reference to each page. The request keeps its own read
    /// reference, so publishing a prefix never invalidates its current table.
    pub fn share_request_prefix(
        &mut self,
        request: &mut RequestPages,
        demands: &[GroupDemand],
    ) -> Result<CachedPages, CachePoolError> {
        self.validate_request_idle(request)?;
        if demands.len() != request.groups.len()
            || demands.iter().zip(&request.groups).any(|(demand, group)| {
                demand.group != group.group || demand.blocks > group.pages.len()
            })
        {
            return Err(CachePoolError::StaleHandle);
        }
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(demands.len())
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        let mut shared_lengths = Vec::new();
        shared_lengths
            .try_reserve_exact(demands.len())
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        for (demand, source) in demands.iter().zip(&request.groups) {
            // Append-only growth preserves this prefix. Replacement and reset
            // invalidate the private hint before changing existing positions,
            // so publishing never has to compare the entire ancestor chain.
            let previous = source
                .snapshot
                .upgrade()
                .filter(|tail| tail.len() <= demand.blocks);
            let shared = previous.as_ref().map_or(0, |tail| tail.len());
            for page in &source.pages[shared..demand.blocks] {
                if self.validate_live_handle(page.handle)?.cache_refs == u32::MAX {
                    return Err(CachePoolError::DemandOverflow);
                }
            }
            let tail = if shared == demand.blocks {
                previous
            } else {
                let mut pages = Vec::new();
                pages
                    .try_reserve_exact(demand.blocks - shared)
                    .map_err(|_| CachePoolError::MetadataAllocation)?;
                pages.extend(
                    source.pages[shared..demand.blocks]
                        .iter()
                        .map(|page| PageRef {
                            handle: page.handle,
                            writable: false,
                        }),
                );
                Some(Arc::new(CachedSegment {
                    start: shared,
                    pages,
                    previous,
                }))
            };
            shared_lengths.push(shared);
            groups.push(CachedGroupTable {
                group: demand.group,
                tail,
            });
        }

        for (((demand, source), shared), cached) in demands
            .iter()
            .zip(&mut request.groups)
            .zip(shared_lengths)
            .zip(&groups)
        {
            for page in &mut source.pages[shared..demand.blocks] {
                let slot = self.slot_mut(page.handle);
                match slot.mode {
                    Some(PageMode::Mutable { writer }) => {
                        debug_assert_eq!(writer, request.writer);
                        debug_assert!(page.writable);
                        debug_assert_eq!(slot.request_refs, 1);
                        slot.mode = Some(PageMode::Shared);
                        page.writable = false;
                    }
                    Some(PageMode::Shared) => debug_assert!(!page.writable),
                    _ => unreachable!("validated request pages remain mutable or shared"),
                }
                slot.cache_refs += 1;
            }
            source.snapshot = cached.tail.as_ref().map(Arc::downgrade).unwrap_or_default();
        }
        Ok(CachedPages {
            pool_id: self.id,
            groups,
        })
    }

    /// Attach an immutable cached prefix to an otherwise empty request table.
    pub fn attach_shared_prefix(
        &mut self,
        request: &mut RequestPages,
        cached: &CachedPages,
    ) -> Result<(), CachePoolError> {
        self.validate_request_idle(request)?;
        if cached.pool_id != self.id || cached.groups.len() != request.groups.len() {
            return Err(CachePoolError::ForeignPool);
        }
        for (target, source) in request.groups.iter().zip(&cached.groups) {
            if target.group != source.group || !target.pages.is_empty() {
                return Err(CachePoolError::StaleHandle);
            }
            for page in source.pages_rev() {
                let slot = self.validate_live_handle(page.handle)?;
                if slot.mode != Some(PageMode::Shared) || page.writable {
                    return Err(CachePoolError::StaleHandle);
                }
                if slot.request_refs == u32::MAX {
                    return Err(CachePoolError::DemandOverflow);
                }
            }
        }
        for (target, source) in request.groups.iter_mut().zip(&cached.groups) {
            target
                .pages
                .try_reserve_exact(source.len())
                .map_err(|_| CachePoolError::MetadataAllocation)?;
        }
        let request_blocks = cached.groups.iter().try_fold(0usize, |total, group| {
            total
                .checked_add(group.len())
                .ok_or(CachePoolError::DemandOverflow)
        })?;
        self.prepare_validation_capacity(request_blocks)?;
        for (target, source) in request.groups.iter_mut().zip(&cached.groups) {
            for page in source.pages_rev() {
                self.slot_mut(page.handle).request_refs += 1;
                target.pages.push(PageRef {
                    handle: page.handle,
                    writable: false,
                });
            }
            target.pages.reverse();
            target.snapshot = source.tail.as_ref().map(Arc::downgrade).unwrap_or_default();
        }
        Ok(())
    }

    pub fn retire_cached_prefix(&mut self, cached: CachedPages) -> Result<(), CachePoolError> {
        if cached.pool_id != self.id {
            return Err(CachePoolError::ForeignPool);
        }
        for group in &cached.groups {
            let mut current = group.tail.as_ref();
            while let Some(segment) = current {
                if Arc::strong_count(segment) != 1 {
                    break;
                }
                for page in &segment.pages {
                    let slot = self.validate_live_handle(page.handle)?;
                    if slot.mode != Some(PageMode::Shared) || page.writable || slot.cache_refs == 0
                    {
                        return Err(CachePoolError::StaleHandle);
                    }
                }
                current = segment.previous.as_ref();
            }
        }
        for group in cached.groups.into_iter().rev() {
            let mut current = group.tail;
            while let Some(segment) = current {
                match Arc::try_unwrap(segment) {
                    Ok(mut segment) => {
                        for page in segment.pages.iter().rev() {
                            self.slot_mut(page.handle).cache_refs -= 1;
                            self.retire_slot_if_unused(page.handle);
                        }
                        current = segment.previous.take();
                    }
                    Err(_) => break,
                }
            }
        }
        Ok(())
    }

    /// Copy-on-write selected immutable pages for a request which is about to
    /// recycle sliding-window slots. Replacement placement is transactional;
    /// the old shared references remain untouched on failure.
    pub fn replace_shared_pages(
        &mut self,
        request: &mut RequestPages,
        replacements: &[Vec<usize>],
    ) -> Result<(), CachePoolError> {
        self.validate_request_idle(request)?;
        if replacements.len() != request.groups.len() {
            return Err(CachePoolError::StaleHandle);
        }
        let mut total = 0usize;
        for (indices, group) in replacements.iter().zip(&request.groups) {
            if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(CachePoolError::DuplicateHandle);
            }
            for &index in indices {
                let page = group.pages.get(index).ok_or(CachePoolError::StaleHandle)?;
                if page.writable
                    || self.validate_live_handle(page.handle)?.mode != Some(PageMode::Shared)
                {
                    return Err(CachePoolError::StaleHandle);
                }
            }
            total = total
                .checked_add(indices.len())
                .ok_or(CachePoolError::DemandOverflow)?;
        }
        if total == 0 {
            return Ok(());
        }
        self.occupied_blocks
            .checked_add(total)
            .ok_or(CachePoolError::DemandOverflow)?;

        let mut additions = Vec::new();
        additions
            .try_reserve_exact(request.groups.len())
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        for (indices, group) in replacements.iter().zip(&request.groups) {
            let mut pages = Vec::new();
            pages
                .try_reserve_exact(indices.len())
                .map_err(|_| CachePoolError::MetadataAllocation)?;
            additions.push(GroupTable {
                group: group.group,
                pages,
                snapshot: Weak::new(),
            });
        }

        let mut plan = PlacementPlan::new(self.group_packings.len(), total, self.parents.len())?;
        for (indices, addition) in replacements.iter().zip(&mut additions) {
            self.plan_partition(
                &mut plan,
                request.writer,
                GroupDemand::new(addition.group, indices.len()),
                &mut addition.pages,
            )?;
        }
        debug_assert_eq!(plan.block_count, total);
        self.commit_plan(plan);
        for page in additions.iter().flat_map(|group| &group.pages) {
            let slot = self.slot_mut(page.handle);
            debug_assert_eq!(
                slot.mode,
                Some(PageMode::Reserved {
                    writer: request.writer
                })
            );
            slot.mode = Some(PageMode::Mutable {
                writer: request.writer,
            });
            slot.request_refs = 1;
        }
        for ((indices, group), additions) in
            replacements.iter().zip(&mut request.groups).zip(additions)
        {
            if !indices.is_empty() {
                group.snapshot = Weak::new();
            }
            for (&index, replacement) in indices.iter().zip(additions.pages) {
                let old = std::mem::replace(&mut group.pages[index], replacement);
                debug_assert!(!old.writable);
                self.slot_mut(old.handle).request_refs -= 1;
                self.retire_slot_if_unused(old.handle);
            }
        }
        Ok(())
    }

    /// Release every page while retaining the request's sequence and writer.
    pub fn clear_request(&mut self, request: &mut RequestPages) -> Result<(), CachePoolError> {
        self.validate_request_idle(request)?;
        for group in request.groups.iter_mut().rev() {
            group.snapshot = Weak::new();
            for page in group.pages.drain(..).rev() {
                let slot = self.slot_mut(page.handle);
                debug_assert_ne!(slot.request_refs, 0);
                slot.request_refs -= 1;
                self.retire_slot_if_unused(page.handle);
            }
        }
        Ok(())
    }

    /// Discard a fresh request reservation. Like promotion, this is infallible
    /// after successful reservation.
    pub fn abort_request(&mut self, pending: PendingRequestPages) {
        self.assert_pending_request(&pending);
        for group in pending.groups.iter().rev() {
            for page in group.pages.iter().rev() {
                let slot = self.slot_mut(page.handle);
                debug_assert_eq!(slot.request_refs, 0);
                slot.mode = None;
                self.retire_slot_if_unused(page.handle);
            }
        }
    }

    /// Retire one move-only request owner. Validation completes before the first
    /// mutation; failure returns the exact token unchanged. Active leases keep
    /// their pages physically live until runner completion or quarantine.
    pub fn retire_request(&mut self, request: RequestPages) -> Result<(), RetireRequestFailure> {
        if let Err(error) = self.validate_request(&request) {
            return Err(RetireRequestFailure { error, request });
        }
        for group in request.groups.iter().rev() {
            for page in group.pages.iter().rev() {
                let slot = self.slot_mut(page.handle);
                debug_assert_ne!(slot.request_refs, 0);
                slot.request_refs -= 1;
                self.retire_slot_if_unused(page.handle);
            }
        }
        Ok(())
    }

    /// Authenticate and canonicalize an exact set of mutable request pages
    /// before runner acceptance. No lease state changes here.
    pub fn prepare_writer_access(
        &self,
        mut claims: Vec<PageAccessClaim>,
    ) -> Result<PreparedWriterAccess, CachePoolError> {
        for claim in &claims {
            self.validate_writer_claim(*claim)?;
        }
        claims.sort_unstable_by_key(|claim| claim.handle);
        if claims
            .windows(2)
            .any(|pair| pair[0].handle == pair[1].handle && (pair[0].write || pair[1].write))
        {
            return Err(CachePoolError::DuplicateAccessClaim);
        }
        claims.dedup_by_key(|claim| claim.handle);

        let mut prepared_claims = Vec::new();
        prepared_claims
            .try_reserve_exact(claims.len())
            .map_err(|_| CachePoolError::MetadataAllocation)?;
        for claim in &claims {
            let slot = self.validate_writer_available(*claim)?;
            prepared_claims.push(PreparedWriterClaim {
                claim: *claim,
                expected_write_epoch: slot.write_epoch,
            });
        }
        Ok(PreparedWriterAccess {
            pool_id: self.id,
            claims: prepared_claims,
        })
    }

    /// Acquire every prepared writer lease atomically. Revalidation catches an
    /// intervening retirement or competing lease without partially acquiring.
    pub fn acquire_writer_access(
        &mut self,
        prepared: PreparedWriterAccess,
    ) -> Result<WriterAccessLease, CachePoolError> {
        if prepared.pool_id != self.id {
            return Err(CachePoolError::ForeignAccess);
        }
        for prepared_claim in &prepared.claims {
            let claim = prepared_claim.claim;
            self.validate_writer_claim(claim)?;
            let slot = self.validate_writer_available(claim)?;
            if slot.write_epoch != prepared_claim.expected_write_epoch {
                return Err(CachePoolError::WriteEpochChanged {
                    expected: prepared_claim.expected_write_epoch,
                    actual: slot.write_epoch,
                });
            }
        }
        for prepared_claim in &prepared.claims {
            let slot = self.slot_mut(prepared_claim.claim.handle);
            if prepared_claim.claim.write {
                slot.writer_lease = true;
            } else {
                slot.reader_leases += 1;
            }
        }
        Ok(WriterAccessLease {
            pool_id: prepared.pool_id,
            claims: prepared.claims,
        })
    }

    /// Release a known-live writer lease. This path allocates nothing. A caller
    /// which cannot prove runner completion retains the lease as quarantine.
    pub fn release_writer_access(&mut self, lease: WriterAccessLease) {
        assert_eq!(
            lease.pool_id, self.id,
            "writer access can only return to its issuing cache pool"
        );
        for prepared_claim in &lease.claims {
            let claim = prepared_claim.claim;
            let slot = self
                .validate_live_handle(claim.handle)
                .expect("writer lease keeps its page live");
            if claim.write {
                assert!(slot.writer_lease);
            } else {
                assert_ne!(slot.reader_leases, 0);
            }
            assert_eq!(slot.write_epoch, prepared_claim.expected_write_epoch);
            if claim.write {
                assert_ne!(slot.write_epoch, u64::MAX);
            }
        }
        for prepared_claim in lease.claims {
            let claim = prepared_claim.claim;
            let slot = self.slot_mut(claim.handle);
            if claim.write {
                slot.writer_lease = false;
            } else {
                slot.reader_leases -= 1;
            }
            if claim.write {
                slot.write_epoch += 1;
            }
            self.retire_slot_if_unused(claim.handle);
        }
    }

    fn plan_partition(
        &self,
        plan: &mut PlacementPlan,
        writer: WriterId,
        demand: GroupDemand,
        pages: &mut Vec<PageRef>,
    ) -> Result<(), CachePoolError> {
        let packing = self.blocks_per_slab(demand.group)?.get();
        let PlacementPlan {
            parents,
            group_states,
            new_parent_ids,
            block_count,
        } = plan;
        let state = group_states[demand.group.index() as usize].get_or_insert(GroupPlanState {
            next_partial_index: 0,
            current: None,
        });
        let mut remaining = demand.blocks;

        while remaining != 0 {
            if let Some(plan_index) = state.current {
                let parent_id = parents[plan_index].parent;
                let actual = &self.parents[(parent_id - 1) as usize];
                let allocated = parents[plan_index]
                    .allocate_up_to(actual, self.id, writer, remaining, pages)?;
                remaining -= allocated;
                *block_count += allocated;
                if parents[plan_index].is_full(actual) {
                    state.current = None;
                }
                continue;
            }

            let next_existing = loop {
                let Some(&parent_id) = self.partial_parents.get(state.next_partial_index) else {
                    break None;
                };
                state.next_partial_index += 1;
                let actual = &self.parents[(parent_id - 1) as usize];
                if actual.binding == Some(demand.group) {
                    break Some(parent_id);
                }
            };
            if let Some(parent_id) = next_existing {
                let actual = &self.parents[(parent_id - 1) as usize];
                debug_assert!(actual.partial_index.is_some());
                let plan_index = parents.len();
                parents.push(PlannedParent::existing(parent_id, actual, demand.group));
                state.current = Some(plan_index);
                continue;
            }

            let Some(parent_id) = self.free_parents.get(new_parent_ids.len()).copied() else {
                return Err(CachePoolError::InsufficientCapacity {
                    group: demand.group,
                    requested: demand.blocks,
                    available: pages.len(),
                });
            };
            let actual = &self.parents[(parent_id - 1) as usize];
            let planned = PlannedParent::new(parent_id, actual, demand.group, packing)?;
            let plan_index = parents.len();
            parents.push(planned);
            new_parent_ids.push(parent_id);
            state.current = Some(plan_index);
        }
        Ok(())
    }

    fn commit_plan(&mut self, plan: PlacementPlan) {
        let PlacementPlan {
            parents,
            group_states: _,
            new_parent_ids,
            block_count,
        } = plan;
        let mut expected_new_parents = new_parent_ids.into_iter();

        for planned in parents {
            let PlannedParent {
                parent: parent_id,
                group,
                next_generation,
                occupied_after,
                next_slot: _,
                storage,
            } = planned;
            let parent_index = (parent_id - 1) as usize;
            match storage {
                PlannedStorage::Existing { updates } => {
                    let became_full = {
                        let parent = &mut self.parents[parent_index];
                        debug_assert_eq!(parent.binding, Some(group));
                        debug_assert!(parent.partial_index.is_some());
                        for update in updates {
                            let slot = &mut parent.slots[update.slot as usize];
                            debug_assert!(!slot.is_live());
                            *slot = PageSlot::reserved_request(update.generation, update.writer);
                        }
                        parent.occupied_slots = occupied_after;
                        parent.next_generation = next_generation;
                        occupied_after as usize == parent.slots.len()
                    };
                    if became_full {
                        self.remove_partial(parent_id);
                    }
                }
                PlannedStorage::New { slots } => {
                    debug_assert_eq!(expected_new_parents.next(), Some(parent_id));
                    assert_eq!(
                        self.free_parents.pop_front(),
                        Some(parent_id),
                        "cache-pool free-parent order changed during reservation"
                    );
                    let is_partial = {
                        let parent = &mut self.parents[parent_index];
                        debug_assert!(parent.binding.is_none());
                        debug_assert_eq!(parent.occupied_slots, 0);
                        debug_assert!(parent.slots.is_empty());
                        debug_assert!(parent.partial_index.is_none());
                        parent.binding = Some(group);
                        parent.slots = slots;
                        parent.occupied_slots = occupied_after;
                        parent.next_generation = next_generation;
                        occupied_after as usize != parent.slots.len()
                    };
                    if is_partial {
                        self.insert_partial(parent_id);
                    }
                }
            }
        }
        debug_assert_eq!(expected_new_parents.next(), None);
        self.occupied_blocks += block_count;
    }

    /// Preflight a request retirement without changing ownership or placement.
    pub fn validate_request(&self, request: &RequestPages) -> Result<(), CachePoolError> {
        self.validate_request_access(request, false)
    }

    fn validate_request_access(
        &self,
        request: &RequestPages,
        require_idle: bool,
    ) -> Result<(), CachePoolError> {
        if request.pool_id != self.id {
            return Err(CachePoolError::ForeignPool);
        }
        if self.has_duplicate_request_handle(request)? {
            return Err(CachePoolError::DuplicateHandle);
        }
        let mut active = false;
        for group in &request.groups {
            for page in &group.pages {
                if group.group != page.handle.group {
                    return Err(CachePoolError::StaleHandle);
                }
                let slot = self.validate_live_handle(page.handle)?;
                match (slot.mode, page.writable) {
                    (Some(PageMode::Mutable { writer }), true) => {
                        if writer != request.writer {
                            return Err(CachePoolError::WrongWriter);
                        }
                        if slot.request_refs != 1 {
                            return Err(CachePoolError::StaleHandle);
                        }
                    }
                    (Some(PageMode::Shared), false) if slot.request_refs != 0 => {}
                    _ => return Err(CachePoolError::StaleHandle),
                }
                active |= require_idle && (slot.writer_lease || slot.reader_leases != 0);
            }
        }
        // Preserve ownership-error precedence even when an earlier page is
        // busy, without looking up every historical handle for a second pass.
        if active {
            Err(CachePoolError::AccessConflict)
        } else {
            Ok(())
        }
    }

    fn validate_request_idle(&self, request: &RequestPages) -> Result<(), CachePoolError> {
        self.validate_request_access(request, true)
    }

    fn prepare_validation_capacity(&mut self, count: usize) -> Result<(), CachePoolError> {
        let handles = self
            .validation_handles
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        handles.clear();
        handles
            .try_reserve(count)
            .map_err(|_| CachePoolError::MetadataAllocation)
    }

    fn has_duplicate_request_handle(&self, request: &RequestPages) -> Result<bool, CachePoolError> {
        let mut handles = self
            .validation_handles
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        handles.clear();
        for page in request.groups.iter().flat_map(|group| &group.pages) {
            // An allocator-issued request always fits the preallocated table.
            // Reject internally forged excess owners without allocating here.
            if handles.len() == handles.capacity() {
                handles.sort_unstable();
                return if handles.windows(2).any(|pair| pair[0] == pair[1])
                    || handles.binary_search(&page.handle).is_ok()
                {
                    Ok(true)
                } else {
                    Err(CachePoolError::StaleHandle)
                };
            }
            handles.push(page.handle);
        }
        // Tables emitted by the allocator are already ordered except for a
        // small cyclic suffix. Sorting reusable scratch avoids hashing every
        // shared historical handle on each decode, without trusting callers
        // or allocating on a retirement/error path.
        handles.sort_unstable();
        Ok(handles.windows(2).any(|pair| pair[0] == pair[1]))
    }

    fn validate_writer_claim(&self, claim: PageAccessClaim) -> Result<(), CachePoolError> {
        if claim.pool_id != self.id {
            return Err(CachePoolError::ForeignAccess);
        }
        let slot = self.validate_live_handle(claim.handle)?;
        if claim.write {
            let Some(PageMode::Mutable { writer }) = slot.mode else {
                return Err(CachePoolError::PageNotWritable);
            };
            if writer != claim.writer {
                return Err(CachePoolError::WrongWriter);
            }
            if slot.request_refs != 1 {
                return Err(CachePoolError::PageNotWritable);
            }
        } else if slot.mode != Some(PageMode::Shared) || slot.request_refs == 0 {
            return Err(CachePoolError::StaleHandle);
        }
        Ok(())
    }

    fn validate_writer_available(
        &self,
        claim: PageAccessClaim,
    ) -> Result<&PageSlot, CachePoolError> {
        let slot = self.validate_live_handle(claim.handle)?;
        if claim.write && slot.write_epoch == u64::MAX {
            return Err(CachePoolError::WriteEpochExhausted);
        }
        if slot.writer_lease || (claim.write && slot.reader_leases != 0) {
            return Err(CachePoolError::AccessConflict);
        }
        Ok(slot)
    }

    fn validate_live_handle(&self, handle: PageHandle) -> Result<&PageSlot, CachePoolError> {
        if handle.pool_id != self.id {
            return Err(CachePoolError::ForeignPool);
        }
        let parent = self
            .parents
            .get(
                handle
                    .parent
                    .checked_sub(1)
                    .ok_or(CachePoolError::StaleHandle)? as usize,
            )
            .ok_or(CachePoolError::StaleHandle)?;
        let bound_group = parent.binding.ok_or(CachePoolError::StaleHandle)?;
        if bound_group != handle.group {
            return Err(CachePoolError::ParentGroupMismatch {
                handle_group: handle.group,
                bound_group,
            });
        }
        let slot = parent
            .slots
            .get(handle.slot as usize)
            .ok_or(CachePoolError::StaleHandle)?;
        if !slot.is_live() || slot.generation != handle.generation {
            return Err(CachePoolError::StaleHandle);
        }
        Ok(slot)
    }

    fn assert_pending_request(&self, pending: &PendingRequestPages) {
        assert_eq!(
            pending.pool_id, self.id,
            "pending request pages belong to a different cache pool"
        );
        for group in &pending.groups {
            for page in &group.pages {
                assert_eq!(group.group, page.handle.group);
                let slot = self
                    .validate_live_handle(page.handle)
                    .expect("fresh pending request page remains live");
                assert_eq!(
                    slot.mode,
                    Some(PageMode::Reserved {
                        writer: pending.writer
                    })
                );
                assert_eq!(slot.request_refs, 0);
                assert_eq!(slot.reader_leases, 0);
                assert!(!slot.writer_lease);
                assert_eq!(slot.write_epoch, 0);
            }
        }
    }

    fn slot_mut(&mut self, handle: PageHandle) -> &mut PageSlot {
        &mut self.parents[(handle.parent - 1) as usize].slots[handle.slot as usize]
    }

    fn insert_partial(&mut self, parent_id: u32) {
        let parent_index = (parent_id - 1) as usize;
        {
            let parent = &self.parents[parent_index];
            debug_assert!(parent.binding.is_some());
            debug_assert_ne!(parent.occupied_slots, 0);
            debug_assert!((parent.occupied_slots as usize) < parent.slots.len());
            debug_assert!(parent.partial_index.is_none());
        }
        debug_assert!(self.partial_parents.len() < self.parents.len());
        let partial_index = self.partial_parents.len();
        self.partial_parents.push(parent_id);
        self.parents[parent_index].partial_index = Some(partial_index);
    }

    fn remove_partial(&mut self, parent_id: u32) {
        let parent_index = (parent_id - 1) as usize;
        let partial_index = self.parents[parent_index]
            .partial_index
            .expect("only a listed partial parent may be removed");
        let removed = self.partial_parents.swap_remove(partial_index);
        debug_assert_eq!(removed, parent_id);
        self.parents[parent_index].partial_index = None;
        if let Some(&moved_parent_id) = self.partial_parents.get(partial_index) {
            self.parents[(moved_parent_id - 1) as usize].partial_index = Some(partial_index);
        }
    }

    fn retire_slot_if_unused(&mut self, handle: PageHandle) {
        let parent_index = (handle.parent - 1) as usize;
        let (became_empty, became_partial, was_partial) = {
            let parent = &mut self.parents[parent_index];
            let slot = &mut parent.slots[handle.slot as usize];
            if slot.is_live() {
                return;
            }
            slot.clear_lifetime();
            let was_full = parent.occupied_slots as usize == parent.slots.len();
            parent.occupied_slots -= 1;
            (
                parent.occupied_slots == 0,
                was_full && parent.occupied_slots != 0,
                parent.partial_index.is_some(),
            )
        };
        self.occupied_blocks -= 1;

        if became_empty {
            if was_partial {
                self.remove_partial(handle.parent);
            }
            let parent = &mut self.parents[parent_index];
            parent.binding = None;
            parent.slots.clear();
            self.free_parents.push_back(handle.parent);
        } else if became_partial {
            debug_assert!(!was_partial);
            self.insert_partial(handle.parent);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn nz(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).unwrap()
    }

    fn sequence(value: u64) -> SequenceId {
        SequenceId::new(value)
    }

    fn request_handles(request: &RequestPages) -> Vec<PageHandle> {
        request
            .groups()
            .iter()
            .flat_map(|group| group.pages().iter().map(PageRef::handle))
            .collect()
    }

    fn writer_claims(request: &RequestPages) -> Vec<PageAccessClaim> {
        request
            .groups()
            .iter()
            .flat_map(|group| group.pages().iter().map(|page| request.access_claim(page)))
            .collect()
    }

    #[test]
    fn growing_checkpoints_share_linear_metadata_and_reclaim_after_last_owner() {
        let mut pool = CachePool::new(512, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 0)])
            .unwrap();
        let mut request = pool.promote_request(pending);
        let mut snapshots = Vec::new();
        for count in 1..=512 {
            let demands = [GroupDemand::new(GroupId::new(0), count)];
            pool.grow_request(&mut request, &demands).unwrap();
            snapshots.push(pool.share_request_prefix(&mut request, &demands).unwrap());
        }
        let mut segments = HashSet::new();
        let mut stored_pages = 0;
        for snapshot in &snapshots {
            for segment in cached_segments(snapshot.groups[0].tail.as_deref()) {
                if segments.insert(std::ptr::from_ref(segment)) {
                    stored_pages += segment.pages.len();
                }
            }
        }
        assert_eq!(segments.len(), 512);
        assert_eq!(stored_pages, 512);
        for page in &request.groups[0].pages {
            let slot = pool.validate_live_handle(page.handle).unwrap();
            assert_eq!((slot.request_refs, slot.cache_refs), (1, 1));
        }
        let expected = request_handles(&request);
        let validation_capacity = pool.validation_handles.lock().unwrap().capacity();
        let last = snapshots.pop().unwrap();
        pool.retire_request(request).unwrap();
        for snapshot in snapshots {
            pool.retire_cached_prefix(snapshot).unwrap();
            assert_eq!(pool.occupied_block_count(), 512);
        }

        let pending = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 0)])
            .unwrap();
        let mut restored = pool.promote_request(pending);
        pool.attach_shared_prefix(&mut restored, &last).unwrap();
        assert_eq!(request_handles(&restored), expected);
        pool.retire_cached_prefix(last).unwrap();
        assert_eq!(pool.occupied_block_count(), 512);
        pool.retire_request(restored).unwrap();
        assert_eq!(pool.free_parent_count(), 512);
        assert_eq!(
            pool.validation_handles.lock().unwrap().capacity(),
            validation_capacity
        );
    }

    #[test]
    fn branched_checkpoints_keep_shared_ancestors_until_both_branches_retire() {
        let mut pool = CachePool::new(10, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 2)])
            .unwrap();
        let mut base = pool.promote_request(pending);
        let snapshot = pool
            .share_request_prefix(&mut base, &[GroupDemand::new(GroupId::new(0), 2)])
            .unwrap();
        let mut branches = Vec::new();
        for id in 2..=3 {
            let pending = pool
                .reserve_request(sequence(id), &[GroupDemand::new(GroupId::new(0), 0)])
                .unwrap();
            let mut request = pool.promote_request(pending);
            pool.attach_shared_prefix(&mut request, &snapshot).unwrap();
            pool.grow_request(&mut request, &[GroupDemand::new(GroupId::new(0), 3)])
                .unwrap();
            let cached = pool
                .share_request_prefix(&mut request, &[GroupDemand::new(GroupId::new(0), 3)])
                .unwrap();
            branches.push((request, cached));
        }
        pool.retire_request(base).unwrap();
        pool.retire_cached_prefix(snapshot).unwrap();
        assert_eq!(pool.occupied_block_count(), 4);
        let (request, cached) = branches.pop().unwrap();
        pool.retire_request(request).unwrap();
        pool.retire_cached_prefix(cached).unwrap();
        assert_eq!(pool.occupied_block_count(), 3);
        let (request, cached) = branches.pop().unwrap();
        pool.retire_cached_prefix(cached).unwrap();
        assert_eq!(pool.occupied_block_count(), 3);
        pool.retire_request(request).unwrap();
        assert_eq!(pool.occupied_block_count(), 0);
    }

    #[test]
    fn cyclic_replacement_starts_a_new_snapshot_without_reusing_stale_prefixes() {
        let mut pool = CachePool::new(8, &[nz(1)]).unwrap();
        let demands = [GroupDemand::new(GroupId::new(0), 3)];
        let pending = pool.reserve_request(sequence(1), &demands).unwrap();
        let mut request = pool.promote_request(pending);
        let old = pool.share_request_prefix(&mut request, &demands).unwrap();
        let old_handles = request_handles(&request);
        pool.replace_shared_pages(&mut request, &[vec![0]]).unwrap();
        assert!(request.groups[0].snapshot.upgrade().is_none());
        let new = pool.share_request_prefix(&mut request, &demands).unwrap();
        assert!(new.groups[0].tail.as_ref().unwrap().previous.is_none());
        assert_ne!(request_handles(&request)[0], old_handles[0]);
        pool.retire_cached_prefix(old).unwrap();
        assert_eq!(pool.occupied_block_count(), 3);
        pool.retire_request(request).unwrap();
        pool.retire_cached_prefix(new).unwrap();
        assert_eq!(pool.occupied_block_count(), 0);
    }

    #[test]
    fn evicted_snapshot_hint_cannot_resurrect_retired_ownership() {
        let mut pool = CachePool::new(3, &[nz(1)]).unwrap();
        let demands = [GroupDemand::new(GroupId::new(0), 2)];
        let pending = pool.reserve_request(sequence(1), &demands).unwrap();
        let mut request = pool.promote_request(pending);
        let snapshot = pool.share_request_prefix(&mut request, &demands).unwrap();
        pool.retire_cached_prefix(snapshot).unwrap();
        assert!(request.groups[0].snapshot.upgrade().is_none());
        let replacement = pool.share_request_prefix(&mut request, &demands).unwrap();
        for page in &request.groups[0].pages {
            let slot = pool.validate_live_handle(page.handle).unwrap();
            assert_eq!((slot.request_refs, slot.cache_refs), (1, 1));
        }
        pool.retire_request(request).unwrap();
        pool.retire_cached_prefix(replacement).unwrap();
        assert_eq!(pool.free_parent_count(), 3);
    }

    #[test]
    fn republishing_the_same_checkpoint_reuses_its_segment_ownership() {
        let mut pool = CachePool::new(2, &[nz(1)]).unwrap();
        let demands = [GroupDemand::new(GroupId::new(0), 2)];
        let pending = pool.reserve_request(sequence(1), &demands).unwrap();
        let mut request = pool.promote_request(pending);
        let first = pool.share_request_prefix(&mut request, &demands).unwrap();
        let second = pool.share_request_prefix(&mut request, &demands).unwrap();
        assert!(Arc::ptr_eq(
            first.groups[0].tail.as_ref().unwrap(),
            second.groups[0].tail.as_ref().unwrap()
        ));
        for page in &request.groups[0].pages {
            let slot = pool.validate_live_handle(page.handle).unwrap();
            assert_eq!((slot.request_refs, slot.cache_refs), (1, 1));
        }
        pool.retire_cached_prefix(first).unwrap();
        pool.retire_request(request).unwrap();
        assert_eq!(pool.occupied_block_count(), 2);
        pool.retire_cached_prefix(second).unwrap();
        assert_eq!(pool.occupied_block_count(), 0);
    }

    #[test]
    fn dropping_a_long_segment_chain_is_iterative_and_does_not_return_physical_refs() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let mut request = pool.promote_request(pending);
        let snapshot = pool
            .share_request_prefix(&mut request, &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        pool.retire_request(request).unwrap();
        drop(snapshot);
        // Explicit pool retirement is required; abandoning a token must never
        // make possibly addressed physical storage reusable.
        assert_eq!(pool.occupied_block_count(), 1);

        let mut tail = None;
        for _ in 0..50_000 {
            tail = Some(Arc::new(CachedSegment {
                start: 0,
                pages: Vec::new(),
                previous: tail,
            }));
        }
        drop(tail);
    }

    #[test]
    fn empty_parent_is_fungible_across_groups_and_generation_survives_rebinding() {
        let mut pool = CachePool::new(1, &[nz(2), nz(1)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first_writer = first.writer;
        let first_handle = first.groups()[0].pages()[0].handle();
        assert_eq!(pool.bound_group(1), Some(GroupId::new(0)));
        pool.abort_request(first);
        assert_eq!(pool.bound_group(1), None);

        let second = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(1), 1)])
            .unwrap();
        assert!(second.writer > first_writer);
        let second_handle = second.groups()[0].pages()[0].handle();
        assert_eq!(second_handle.parent(), first_handle.parent());
        assert!(second_handle.generation() > first_handle.generation());
        assert_eq!(pool.bound_group(1), Some(GroupId::new(1)));
        pool.abort_request(second);
    }

    #[test]
    fn group_packing_exposes_physical_parent_and_child_slots() {
        let mut pool = CachePool::new(2, &[nz(3)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 4)])
            .unwrap();
        let addresses = pending.groups()[0]
            .pages()
            .iter()
            .map(|page| pool.physical_address(page.handle()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(addresses, vec![(1, 0), (1, 1), (1, 2), (2, 0)]);
        assert_eq!(pool.free_parent_count(), 0);
        assert_eq!(pool.occupied_block_count(), 4);
        pool.abort_request(pending);
    }

    #[test]
    fn multi_group_reservation_is_transactional_on_capacity_failure() {
        let mut pool = CachePool::new(2, &[nz(2), nz(1)]).unwrap();
        let before_parents = pool.parents.clone();
        let before_free = pool.free_parents.clone();
        let before_partial = pool.partial_parents.clone();
        let before_occupied = pool.occupied_blocks;

        let error = pool
            .reserve_request(
                sequence(1),
                &[
                    GroupDemand::new(GroupId::new(0), 3),
                    GroupDemand::new(GroupId::new(1), 1),
                ],
            )
            .unwrap_err();
        assert_eq!(
            error,
            CachePoolError::InsufficientCapacity {
                group: GroupId::new(1),
                requested: 1,
                available: 0,
            }
        );
        assert_eq!(pool.parents, before_parents);
        assert_eq!(pool.free_parents, before_free);
        assert_eq!(pool.partial_parents, before_partial);
        assert_eq!(pool.occupied_blocks, before_occupied);
    }

    #[test]
    fn duplicate_demands_share_placement_but_preserve_ordered_group_tables() {
        let mut pool = CachePool::new(2, &[nz(2), nz(1)]).unwrap();
        let pending = pool
            .reserve_request(
                sequence(1),
                &[
                    GroupDemand::new(GroupId::new(0), 1),
                    GroupDemand::new(GroupId::new(1), 1),
                    GroupDemand::new(GroupId::new(0), 1),
                ],
            )
            .unwrap();
        assert_eq!(pending.groups().len(), 3);
        assert_eq!(pending.groups()[0].group(), GroupId::new(0));
        assert_eq!(pending.groups()[1].group(), GroupId::new(1));
        assert_eq!(pending.groups()[2].group(), GroupId::new(0));
        let first = pending.groups()[0].pages()[0].handle();
        let third = pending.groups()[2].pages()[0].handle();
        assert_eq!(first.parent(), third.parent());
        assert_ne!(first.slot(), third.slot());
        assert_eq!(pool.free_parent_count(), 0);
        pool.abort_request(pending);
    }

    #[test]
    fn promotion_installs_exact_request_refs_and_retirement_reclaims_parent() {
        let mut pool = CachePool::new(1, &[nz(2)]).unwrap();
        let pending = pool
            .reserve_request(sequence(7), &[GroupDemand::new(GroupId::new(0), 2)])
            .unwrap();
        assert_eq!(pending.sequence(), sequence(7));
        let request = pool.promote_request(pending);
        assert_eq!(request.sequence(), sequence(7));
        let writer = request.writer;
        for handle in request_handles(&request) {
            let slot = pool.validate_live_handle(handle).unwrap();
            assert_eq!(slot.mode, Some(PageMode::Mutable { writer }));
            assert_eq!(slot.request_refs, 1);
        }
        assert_eq!(pool.occupied_block_count(), 2);
        pool.retire_request(request).unwrap();
        assert_eq!(pool.occupied_block_count(), 0);
        assert_eq!(pool.free_parent_count(), 1);
    }

    #[test]
    fn terminal_return_paths_use_pre_reserved_parent_metadata() {
        let mut pool = CachePool::new(4, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 3)])
            .unwrap();
        let request = pool.promote_request(pending);

        // Admission does not replace or shrink either terminal-path buffer.
        let empty = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 0)])
            .unwrap();
        pool.abort_request(empty);
        assert!(pool.free_parents.capacity() >= pool.parents.len());
        assert!(pool.partial_parents.capacity() >= pool.parents.len());
        pool.retire_request(request).unwrap();
        assert_eq!(pool.free_parent_count(), 4);
    }

    #[test]
    fn partial_parent_swap_removal_updates_the_moved_parent_index() {
        let mut pool = CachePool::new(3, &[nz(2), nz(2), nz(2)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first = pool.promote_request(first);
        let middle = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(1), 1)])
            .unwrap();
        let middle = pool.promote_request(middle);
        let last = pool
            .reserve_request(sequence(3), &[GroupDemand::new(GroupId::new(2), 1)])
            .unwrap();
        let last = pool.promote_request(last);
        assert_eq!(pool.partial_parents, vec![1, 2, 3]);

        pool.retire_request(middle).unwrap();
        assert_eq!(pool.partial_parents, vec![1, 3]);
        assert_eq!(pool.parents[2].partial_index, Some(1));

        let fill_last = pool
            .reserve_request(sequence(4), &[GroupDemand::new(GroupId::new(2), 1)])
            .unwrap();
        assert_eq!(fill_last.groups()[0].pages()[0].handle().parent(), 3);
        assert_eq!(pool.partial_parents, vec![1]);
        assert_eq!(pool.parents[2].partial_index, None);
        pool.abort_request(fill_last);
        pool.retire_request(last).unwrap();
        pool.retire_request(first).unwrap();
        assert!(pool.partial_parents.is_empty());
    }

    #[test]
    fn generation_exhaustion_before_commit_leaves_the_pool_unchanged() {
        let mut pool = CachePool::new(2, &[nz(1), nz(1)]).unwrap();
        pool.parents[1].next_generation = u64::MAX;
        let before_parents = pool.parents.clone();
        let before_free = pool.free_parents.clone();

        let error = pool
            .reserve_request(
                sequence(1),
                &[
                    GroupDemand::new(GroupId::new(0), 1),
                    GroupDemand::new(GroupId::new(1), 1),
                ],
            )
            .unwrap_err();
        assert_eq!(
            error,
            CachePoolError::ParentGenerationExhausted { parent: 2 }
        );
        assert_eq!(pool.parents, before_parents);
        assert_eq!(pool.free_parents, before_free);
        assert!(pool.partial_parents.is_empty());
        assert_eq!(pool.occupied_block_count(), 0);
    }

    #[test]
    fn writer_id_exhaustion_leaves_the_pool_unchanged() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        pool.last_writer_id = u64::MAX;
        let before_parents = pool.parents.clone();
        let before_free = pool.free_parents.clone();

        assert_eq!(
            pool.reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
                .unwrap_err(),
            CachePoolError::WriterIdExhausted
        );
        assert_eq!(pool.parents, before_parents);
        assert_eq!(pool.free_parents, before_free);
        assert!(pool.partial_parents.is_empty());
        assert_eq!(pool.occupied_block_count(), 0);
        assert_eq!(pool.last_writer_id, u64::MAX);
    }

    #[test]
    fn late_failure_rolls_back_planned_holes_in_an_existing_parent() {
        let mut pool = CachePool::new(2, &[nz(3), nz(1)]).unwrap();
        let seed = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let seed = pool.promote_request(seed);
        pool.parents[1].next_generation = u64::MAX;
        let before_parents = pool.parents.clone();
        let before_free = pool.free_parents.clone();
        let before_partial = pool.partial_parents.clone();

        let error = pool
            .reserve_request(
                sequence(2),
                &[
                    GroupDemand::new(GroupId::new(0), 1),
                    GroupDemand::new(GroupId::new(1), 1),
                ],
            )
            .unwrap_err();
        assert_eq!(
            error,
            CachePoolError::ParentGenerationExhausted { parent: 2 }
        );
        assert_eq!(pool.parents, before_parents);
        assert_eq!(pool.free_parents, before_free);
        assert_eq!(pool.partial_parents, before_partial);
        assert_eq!(pool.occupied_block_count(), 1);
        pool.retire_request(seed).unwrap();
    }

    #[test]
    fn same_slot_reuse_bumps_generation_while_its_parent_stays_bound() {
        let mut pool = CachePool::new(1, &[nz(2)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first = pool.promote_request(first);
        let stale = first.groups()[0].pages()[0].handle();
        let sibling = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let sibling = pool.promote_request(sibling);
        pool.retire_request(first).unwrap();

        let replacement = pool
            .reserve_request(sequence(3), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let fresh = replacement.groups()[0].pages()[0].handle();
        assert_eq!(fresh.parent(), stale.parent());
        assert_eq!(fresh.slot(), stale.slot());
        assert!(fresh.generation() > stale.generation());
        assert_eq!(
            pool.physical_address(stale).unwrap_err(),
            CachePoolError::StaleHandle
        );
        pool.abort_request(replacement);
        pool.retire_request(sibling).unwrap();
    }

    #[test]
    fn interleaved_pending_allocations_share_a_parent_in_either_finish_order() {
        let mut pool = CachePool::new(1, &[nz(2)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let second = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        assert_eq!(
            first.groups()[0].pages()[0].handle().parent(),
            second.groups()[0].pages()[0].handle().parent()
        );
        pool.abort_request(first);
        let second = pool.promote_request(second);
        pool.retire_request(second).unwrap();
        assert_eq!(pool.free_parent_count(), 1);

        let first = pool
            .reserve_request(sequence(3), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let second = pool
            .reserve_request(sequence(4), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first = pool.promote_request(first);
        pool.abort_request(second);
        pool.retire_request(first).unwrap();
        assert_eq!(pool.free_parent_count(), 1);
        assert!(pool.partial_parents.is_empty());
    }

    #[test]
    fn forged_duplicate_retirement_is_rejected_without_consuming_the_token() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let mut request = pool.promote_request(pending);
        let handle = request.groups[0].pages[0].handle;
        request.groups[0].pages.push(PageRef {
            handle,
            writable: true,
        });

        let mut failure = pool.retire_request(request).unwrap_err();
        assert_eq!(failure.error, CachePoolError::DuplicateHandle);
        assert_eq!(
            failure.request.groups[0]
                .pages
                .iter()
                .map(PageRef::handle)
                .collect::<Vec<_>>(),
            vec![handle, handle]
        );
        failure.request.groups[0].pages.pop();
        pool.retire_request(failure.request).unwrap();
    }

    #[test]
    fn idle_validation_keeps_ownership_errors_ahead_of_active_leases() {
        let mut pool = CachePool::new(3, &[nz(1)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 2)])
            .unwrap();
        let mut first = pool.promote_request(first);
        let second = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let second = pool.promote_request(second);
        let claims = vec![first.access_claim(&first.groups[0].pages[0])];
        let prepared = pool.prepare_writer_access(claims).unwrap();
        let lease = pool.acquire_writer_access(prepared).unwrap();
        let original = first.groups[0].pages[1].handle;
        first.groups[0].pages[1].handle = second.groups[0].pages[0].handle;
        let demands = [GroupDemand::new(GroupId::new(0), 2)];
        assert_eq!(
            pool.grow_request(&mut first, &demands),
            Err(CachePoolError::WrongWriter)
        );
        first.groups[0].pages[1].handle = original;
        assert_eq!(
            pool.grow_request(&mut first, &demands),
            Err(CachePoolError::AccessConflict)
        );
        pool.release_writer_access(lease);
        pool.grow_request(&mut first, &demands).unwrap();
        pool.retire_request(first).unwrap();
        pool.retire_request(second).unwrap();
        assert_eq!(pool.free_parent_count(), 3);
    }

    #[test]
    fn duplicate_validation_uses_preallocated_scratch_for_forged_excess_pages() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        let request = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let mut request = pool.promote_request(request);
        let capacity = pool.validation_handles.lock().unwrap().capacity();
        let handle = request.groups[0].pages[0].handle;
        for _ in 0..=capacity {
            request.groups[0].pages.push(PageRef {
                handle,
                writable: true,
            });
        }
        assert_eq!(
            pool.validate_request(&request),
            Err(CachePoolError::DuplicateHandle)
        );
        assert_eq!(pool.validation_handles.lock().unwrap().capacity(), capacity);
        request.groups[0].pages.truncate(1);
        pool.retire_request(request).unwrap();
    }

    #[test]
    fn failed_retirement_returns_the_unchanged_request_token() {
        let mut owner_pool = CachePool::new(1, &[nz(1)]).unwrap();
        let mut foreign_pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = owner_pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let request = owner_pool.promote_request(pending);
        let expected = request_handles(&request);
        let failure = foreign_pool.retire_request(request).unwrap_err();
        assert_eq!(failure.error, CachePoolError::ForeignPool);
        assert_eq!(request_handles(&failure.request), expected);
        owner_pool.retire_request(failure.request).unwrap();
    }

    #[test]
    fn stale_foreign_and_wrong_group_handles_are_rejected() {
        let mut pool = CachePool::new(1, &[nz(1), nz(1)]).unwrap();
        let foreign = CachePool::new(1, &[nz(1), nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let live = pending.groups()[0].pages()[0].handle();

        let mut foreign_handle = live;
        foreign_handle.pool_id = foreign.id();
        assert_eq!(
            foreign.physical_address(live).unwrap_err(),
            CachePoolError::ForeignPool
        );
        assert_eq!(
            pool.physical_address(foreign_handle).unwrap_err(),
            CachePoolError::ForeignPool
        );

        let mut wrong_group = live;
        wrong_group.group = GroupId::new(1);
        assert_eq!(
            pool.physical_address(wrong_group).unwrap_err(),
            CachePoolError::ParentGroupMismatch {
                handle_group: GroupId::new(1),
                bound_group: GroupId::new(0),
            }
        );
        pool.abort_request(pending);

        let replacement = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        assert_eq!(
            pool.physical_address(live).unwrap_err(),
            CachePoolError::StaleHandle
        );
        pool.abort_request(replacement);
    }

    #[test]
    fn zero_block_demands_still_preserve_output_order() {
        let mut pool = CachePool::new(1, &[nz(1), nz(2)]).unwrap();
        let pending = pool
            .reserve_request(
                sequence(1),
                &[
                    GroupDemand::new(GroupId::new(1), 0),
                    GroupDemand::new(GroupId::new(0), 0),
                ],
            )
            .unwrap();
        assert_eq!(pending.groups().len(), 2);
        assert_eq!(pending.groups()[0].group(), GroupId::new(1));
        assert_eq!(pending.groups()[1].group(), GroupId::new(0));
        assert!(pending.is_empty());
        pool.abort_request(pending);
        assert_eq!(pool.free_parent_count(), 1);
    }

    #[test]
    fn pending_request_pages_are_not_writer_accessible_before_promotion() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(9), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let page = &pending.groups()[0].pages()[0];
        let claim = PageAccessClaim {
            pool_id: pool.id(),
            writer: pending.writer,
            handle: page.handle(),
            write: true,
        };
        assert_eq!(
            pool.prepare_writer_access(vec![claim]).unwrap_err(),
            CachePoolError::PageNotWritable
        );
        pool.abort_request(pending);
    }

    #[test]
    fn duplicate_sequence_id_cannot_substitute_request_writer() {
        let mut pool = CachePool::new(2, &[nz(1), nz(1)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first = pool.promote_request(first);
        let second = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(1), 1)])
            .unwrap();
        let second = pool.promote_request(second);
        assert_eq!(first.sequence(), second.sequence());
        assert_ne!(first.writer, second.writer);
        let claims = vec![
            first.access_claim(&first.groups()[0].pages()[0]),
            first.access_claim(&second.groups()[0].pages()[0]),
        ];
        assert_eq!(
            pool.prepare_writer_access(claims).unwrap_err(),
            CachePoolError::WrongWriter
        );
        for handle in request_handles(&first)
            .into_iter()
            .chain(request_handles(&second))
        {
            let slot = pool.validate_live_handle(handle).unwrap();
            assert!(!slot.writer_lease && slot.reader_leases == 0);
        }
        pool.retire_request(first).unwrap();
        pool.retire_request(second).unwrap();
    }

    #[test]
    fn duplicate_and_competing_writer_leases_are_rejected_atomically() {
        let mut pool = CachePool::new(1, &[nz(2)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 2)])
            .unwrap();
        let request = pool.promote_request(pending);
        let first_claim = request.access_claim(&request.groups()[0].pages()[0]);
        assert_eq!(
            pool.prepare_writer_access(vec![first_claim, first_claim])
                .unwrap_err(),
            CachePoolError::DuplicateAccessClaim
        );

        let first = pool
            .acquire_writer_access(pool.prepare_writer_access(vec![first_claim]).unwrap())
            .unwrap();
        let all = writer_claims(&request);
        assert_eq!(
            pool.prepare_writer_access(all).unwrap_err(),
            CachePoolError::AccessConflict
        );
        let second_handle = request.groups()[0].pages()[1].handle();
        let slot = pool.validate_live_handle(second_handle).unwrap();
        assert!(!slot.writer_lease && slot.reader_leases == 0);
        pool.release_writer_access(first);
        pool.retire_request(request).unwrap();
    }

    #[test]
    fn prepared_multi_page_acquisition_revalidates_without_partial_leasing() {
        let mut pool = CachePool::new(2, &[nz(1)]).unwrap();
        let first = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let first = pool.promote_request(first);
        let second = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let second = pool.promote_request(second);
        let first_handle = first.groups()[0].pages()[0].handle();
        let prepared = pool
            .prepare_writer_access(vec![
                first.access_claim(&first.groups()[0].pages()[0]),
                second.access_claim(&second.groups()[0].pages()[0]),
            ])
            .unwrap();
        pool.retire_request(second).unwrap();
        assert_eq!(
            pool.acquire_writer_access(prepared).unwrap_err(),
            CachePoolError::StaleHandle
        );
        let slot = pool.validate_live_handle(first_handle).unwrap();
        assert!(!slot.writer_lease && slot.reader_leases == 0);
        pool.retire_request(first).unwrap();
    }

    #[test]
    fn writer_release_advances_epoch_and_epoch_exhaustion_is_non_mutating() {
        let mut pool = CachePool::new(1, &[nz(2)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 2)])
            .unwrap();
        let request = pool.promote_request(pending);
        let claims = writer_claims(&request);
        let lease = pool
            .acquire_writer_access(pool.prepare_writer_access(claims).unwrap())
            .unwrap();
        assert_eq!(lease.len(), 2);
        pool.release_writer_access(lease);
        let handles = request_handles(&request);
        assert!(
            handles
                .iter()
                .all(|handle| pool.validate_live_handle(*handle).unwrap().write_epoch == 1)
        );

        pool.slot_mut(handles[1]).write_epoch = u64::MAX;
        assert_eq!(
            pool.prepare_writer_access(writer_claims(&request))
                .unwrap_err(),
            CachePoolError::WriteEpochExhausted
        );
        assert!(handles.iter().all(|handle| {
            let slot = pool.validate_live_handle(*handle).unwrap();
            !slot.writer_lease && slot.reader_leases == 0
        }));
        pool.slot_mut(handles[1]).write_epoch = 1;
        pool.retire_request(request).unwrap();
    }

    #[test]
    fn prepared_writer_access_rejects_an_intervening_completed_write() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let request = pool.promote_request(pending);
        let claim = request.access_claim(&request.groups()[0].pages()[0]);
        let stale = pool.prepare_writer_access(vec![claim]).unwrap();
        let newer = pool.prepare_writer_access(vec![claim]).unwrap();

        let newer = pool.acquire_writer_access(newer).unwrap();
        pool.release_writer_access(newer);
        assert_eq!(
            pool.acquire_writer_access(stale).unwrap_err(),
            CachePoolError::WriteEpochChanged {
                expected: 0,
                actual: 1,
            }
        );
        let slot = pool.validate_live_handle(claim.handle).unwrap();
        assert!(!slot.writer_lease && slot.reader_leases == 0);
        pool.retire_request(request).unwrap();
    }

    #[test]
    fn active_writer_lease_is_the_quarantine_lifetime_after_request_retirement() {
        let mut pool = CachePool::new(1, &[nz(1)]).unwrap();
        let pending = pool
            .reserve_request(sequence(1), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let request = pool.promote_request(pending);
        let handle = request.groups()[0].pages()[0].handle();
        let lease = pool
            .acquire_writer_access(pool.prepare_writer_access(writer_claims(&request)).unwrap())
            .unwrap();
        pool.retire_request(request).unwrap();
        let slot = pool.validate_live_handle(handle).unwrap();
        assert_eq!(slot.request_refs, 0);
        assert!(slot.writer_lease);
        assert_eq!(pool.occupied_block_count(), 1);
        assert_eq!(
            pool.reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)],)
                .unwrap_err(),
            CachePoolError::InsufficientCapacity {
                group: GroupId::new(0),
                requested: 1,
                available: 0,
            }
        );

        pool.release_writer_access(lease);
        assert_eq!(pool.occupied_block_count(), 0);
        let replacement = pool
            .reserve_request(sequence(2), &[GroupDemand::new(GroupId::new(0), 1)])
            .unwrap();
        let fresh = replacement.groups()[0].pages()[0].handle();
        assert!(fresh.generation() > handle.generation());
        pool.abort_request(replacement);
    }
}
