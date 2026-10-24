use std::fmt;
use std::num::NonZeroU32;

pub const MAX_KERNEL_VALUE: u32 = i32::MAX as u32;

/// How a cache group advances through a sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvGroupFamily {
    /// Token-addressed attention or rolling model state.
    History,
    /// A page-aligned recurrent checkpoint ring.
    State,
}

/// Logical history retained by a token-addressed cache group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvRetention {
    /// Full attention: every logical block remains resident for the request.
    FullContext,
    /// Sliding-window attention: only the chronological suffix remains resident.
    SlidingWindow { tokens: NonZeroU32 },
}

impl KvRetention {
    pub const fn window_tokens(self) -> Option<NonZeroU32> {
        match self {
            Self::FullContext => None,
            Self::SlidingWindow { tokens } => Some(tokens),
        }
    }
}

/// Static geometry and retention policy for one ordered cache group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvGroupLayout {
    block_tokens: NonZeroU32,
    blocks_per_slab: NonZeroU32,
    retention: KvRetention,
    read_window_tokens: Option<NonZeroU32>,
    family: KvGroupFamily,
}

impl KvGroupLayout {
    pub fn new(
        block_tokens: u32,
        blocks_per_slab: u32,
        retention: KvRetention,
    ) -> Result<Self, KvLayoutError> {
        Self::new_with_family(
            block_tokens,
            blocks_per_slab,
            retention,
            KvGroupFamily::History,
        )
    }

    pub fn new_with_family(
        block_tokens: u32,
        blocks_per_slab: u32,
        retention: KvRetention,
        family: KvGroupFamily,
    ) -> Result<Self, KvLayoutError> {
        let block_tokens = NonZeroU32::new(block_tokens)
            .ok_or_else(|| KvLayoutError::new("block_tokens must be positive"))?;
        if block_tokens.get() > MAX_KERNEL_VALUE {
            return Err(KvLayoutError::new(
                "block_tokens exceeds the signed int32 kernel metadata range",
            ));
        }
        let blocks_per_slab = NonZeroU32::new(blocks_per_slab)
            .ok_or_else(|| KvLayoutError::new("blocks_per_slab must be positive"))?;
        if retention
            .window_tokens()
            .is_some_and(|tokens| tokens.get() > MAX_KERNEL_VALUE)
        {
            return Err(KvLayoutError::new(
                "window_tokens exceeds the signed int32 kernel metadata range",
            ));
        }
        if matches!(family, KvGroupFamily::State)
            && matches!(retention, KvRetention::SlidingWindow { .. })
        {
            return Err(KvLayoutError::new(
                "recurrent state groups do not accept window retention",
            ));
        }
        Ok(Self {
            block_tokens,
            blocks_per_slab,
            retention,
            read_window_tokens: retention.window_tokens(),
            family,
        })
    }

    /// Declare the logical read window separately from its physical guard band.
    /// The window includes the current token; all writes in one batch must fit
    /// beside the preceding `read_window_tokens - 1` tokens without aliasing.
    pub fn with_read_window_tokens(mut self, tokens: u32) -> Result<Self, KvLayoutError> {
        let retained = self.retention.window_tokens().ok_or_else(|| {
            KvLayoutError::new("history retention does not accept read_window_tokens")
        })?;
        let tokens = NonZeroU32::new(tokens)
            .filter(|tokens| tokens.get() <= retained.get())
            .ok_or_else(|| {
                KvLayoutError::new(
                    "read_window_tokens must be positive and no larger than window_tokens",
                )
            })?;
        self.read_window_tokens = Some(tokens);
        Ok(self)
    }

    pub const fn read_window_tokens(self) -> Option<NonZeroU32> {
        self.read_window_tokens
    }

    /// Maximum simultaneous writes whose earliest query retains its full read
    /// window. Block rounding may provide extra capacity, but is not a promise.
    pub fn max_write_tokens(self) -> Option<NonZeroU32> {
        self.retention.window_tokens().map(|retained| {
            NonZeroU32::new(retained.get() - self.read_window_tokens.unwrap().get() + 1)
                .expect("validated read window fits retention")
        })
    }

    pub const fn block_tokens(self) -> NonZeroU32 {
        self.block_tokens
    }

    pub const fn blocks_per_slab(self) -> NonZeroU32 {
        self.blocks_per_slab
    }

    pub const fn retention(self) -> KvRetention {
        self.retention
    }

    pub const fn family(self) -> KvGroupFamily {
        self.family
    }

    /// Whether this group stores a page-aligned recurrent-state snapshot.
    pub const fn is_recurrent(self) -> bool {
        matches!(self.family, KvGroupFamily::State)
    }

    /// Maximum request-owned suffix width after a sliding window saturates.
    pub const fn window_resident_blocks(self) -> Option<NonZeroU32> {
        match self.retention {
            KvRetention::SlidingWindow { tokens } => {
                let blocks = 1 + (tokens.get() - 1).div_ceil(self.block_tokens.get());
                NonZeroU32::new(blocks)
            }
            KvRetention::FullContext => None,
        }
    }

    pub(crate) fn active_blocks(self, context: u32) -> (u32, u32) {
        let total = context.div_ceil(self.block_tokens.get());
        let count = if self.is_recurrent() {
            total.min(2)
        } else {
            self.window_resident_blocks()
                .map_or(total, |maximum| total.min(maximum.get()))
        };
        (total - count, count)
    }
}

/// Frozen geometry for every cache group sharing one parent pool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvLayout {
    parent_capacity: NonZeroU32,
    groups: Box<[KvGroupLayout]>,
    replay_suffix_tokens: u32,
}

impl KvLayout {
    pub fn new(parent_capacity: u32, groups: Vec<KvGroupLayout>) -> Result<Self, KvLayoutError> {
        let parent_capacity = NonZeroU32::new(parent_capacity)
            .ok_or_else(|| KvLayoutError::new("parent_capacity must be positive"))?;
        if groups.is_empty() {
            return Err(KvLayoutError::new(
                "groups must contain at least one cache group",
            ));
        }
        if groups.len() > u32::MAX as usize {
            return Err(KvLayoutError::new(
                "group count exceeds the u32 group-index range",
            ));
        }
        for (group_index, group) in groups.iter().copied().enumerate() {
            let kernel_page_capacity =
                u64::from(parent_capacity.get()) * u64::from(group.blocks_per_slab().get());
            if kernel_page_capacity > u64::from(MAX_KERNEL_VALUE) {
                return Err(KvLayoutError::new(format!(
                    "cache group {group_index} page capacity exceeds the signed int32 kernel page-id range"
                )));
            }
            let kernel_page_count = kernel_page_capacity.checked_add(1).ok_or_else(|| {
                KvLayoutError::new(format!(
                    "cache group {group_index} physical slab row count overflows u64"
                ))
            })?;
            usize::try_from(kernel_page_count).map_err(|_| {
                KvLayoutError::new(format!(
                    "cache group {group_index} physical slab row count exceeds usize"
                ))
            })?;
            if group
                .window_resident_blocks()
                .is_some_and(|blocks| u64::from(blocks.get()) > kernel_page_capacity)
            {
                return Err(KvLayoutError::new(format!(
                    "cache group {group_index} retention requires more blocks than its physical page capacity"
                )));
            }
        }
        Ok(Self {
            parent_capacity,
            groups: groups.into_boxed_slice(),
            replay_suffix_tokens: 0,
        })
    }

    /// Require rebuilding this many final prompt tokens after a prefix hit.
    /// Models must not read omitted prefix state before rebuilding this suffix;
    /// snapshots may retain those opaque bytes without making them valid inputs.
    pub fn with_replay_suffix_tokens(mut self, tokens: u32) -> Self {
        self.replay_suffix_tokens = tokens;
        self
    }

    pub const fn replay_suffix_tokens(&self) -> u32 {
        self.replay_suffix_tokens
    }

    pub const fn parent_capacity(&self) -> NonZeroU32 {
        self.parent_capacity
    }

    pub fn groups(&self) -> &[KvGroupLayout] {
        &self.groups
    }

    pub fn group(&self, index: u32) -> Option<&KvGroupLayout> {
        self.groups.get(index as usize)
    }

    pub fn packings(&self) -> impl ExactSizeIterator<Item = NonZeroU32> + '_ {
        self.groups
            .iter()
            .copied()
            .map(KvGroupLayout::blocks_per_slab)
    }

    pub fn kernel_page_capacity(&self, group_index: u32) -> Option<u32> {
        let packing = self.group(group_index)?.blocks_per_slab().get();
        Some(
            self.parent_capacity
                .get()
                .checked_mul(packing)
                .expect("layout validation bounds kernel page capacity"),
        )
    }

    /// Physical rows required by a group, including null page zero.
    pub fn kernel_page_count(&self, group_index: u32) -> Option<u64> {
        self.kernel_page_capacity(group_index)
            .map(|capacity| u64::from(capacity) + 1)
    }

    /// Physical-parent demand for independent request contexts in an empty pool.
    pub fn estimated_parent_count(&self, context_lengths: &[u32]) -> Result<u64, KvLayoutError> {
        self.estimated_parent_count_with_shared_prefix(context_lengths, 0)
    }

    /// Estimate contexts sharing one block-complete prefix. Full history pages
    /// count once; window and recurrent pages retain private write capacity.
    /// This describes a workload, not a promise that a prefix cache hit exists.
    pub fn estimated_parent_count_with_shared_prefix(
        &self,
        context_lengths: &[u32],
        shared_prefix_tokens: u32,
    ) -> Result<u64, KvLayoutError> {
        if context_lengths
            .iter()
            .any(|context| *context < shared_prefix_tokens)
        {
            return Err(KvLayoutError::new(
                "shared prefix exceeds a request context",
            ));
        }
        let overflow = || KvLayoutError::new("estimated cache demand overflows u64");
        self.groups.iter().try_fold(0u64, |parents, group| {
            let mut blocks = context_lengths.iter().try_fold(0u64, |total, context| {
                total
                    .checked_add(u64::from(group.active_blocks(*context).1))
                    .ok_or_else(overflow)
            })?;
            if group.retention == KvRetention::FullContext && !group.is_recurrent() {
                if !shared_prefix_tokens.is_multiple_of(group.block_tokens.get()) {
                    return Err(KvLayoutError::new(
                        "shared prefix must align to full-history blocks",
                    ));
                }
                let shared_blocks = u64::from(shared_prefix_tokens / group.block_tokens.get());
                let extra_copies = u64::try_from(context_lengths.len().saturating_sub(1))
                    .map_err(|_| overflow())?;
                blocks -= shared_blocks
                    .checked_mul(extra_copies)
                    .ok_or_else(overflow)?;
            }
            parents
                .checked_add(blocks.div_ceil(u64::from(group.blocks_per_slab.get())))
                .ok_or_else(overflow)
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvLayoutError(String);

impl KvLayoutError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for KvLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for KvLayoutError {}

/// Executor-facing block table and append locations for one cache group.
#[derive(Clone, Debug, Eq, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct KvGroupPlan {
    pub group_index: u32,
    pub blocks_per_slab: u32,
    pub block_indptr: Vec<u32>,
    pub block_slab_ids: Vec<u32>,
    pub block_slotnums: Vec<u32>,
    pub block_table_start_indices: Vec<u32>,
    pub append_slab_ids: Vec<u32>,
    pub append_slotnums: Vec<u32>,
    pub append_block_offsets: Vec<u32>,
}

impl KvGroupPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        group_index: u32,
        block_indptr: Vec<u32>,
        block_slab_ids: Vec<u32>,
        block_slotnums: Vec<u32>,
        block_table_start_indices: Vec<u32>,
        append_slab_ids: Vec<u32>,
        append_slotnums: Vec<u32>,
        append_block_offsets: Vec<u32>,
        blocks_per_slab: NonZeroU32,
    ) -> Result<Self, KvPlanError> {
        let packing = blocks_per_slab.get();
        let plan = Self {
            group_index,
            blocks_per_slab: packing,
            block_indptr,
            block_slab_ids,
            block_slotnums,
            block_table_start_indices,
            append_slab_ids,
            append_slotnums,
            append_block_offsets,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), KvPlanError> {
        if self.blocks_per_slab == 0 {
            return Err(KvPlanError("blocks_per_slab must be positive"));
        }
        if self.block_slab_ids.len() > u32::MAX as usize {
            return Err(KvPlanError(
                "block address length exceeds the u32 CSR range",
            ));
        }
        if self.block_indptr.len() != self.block_table_start_indices.len() + 1 {
            return Err(KvPlanError(
                "block_indptr must contain one more item than block_table_start_indices",
            ));
        }
        if self.block_indptr.first() != Some(&0) {
            return Err(KvPlanError("block_indptr must start at zero"));
        }
        if self.block_indptr.last().copied() != Some(self.block_slab_ids.len() as u32) {
            return Err(KvPlanError(
                "block_indptr must end at the block address length",
            ));
        }
        if self.block_indptr.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(KvPlanError(
                "block_indptr must be monotonically nondecreasing",
            ));
        }
        if self.block_slab_ids.len() != self.block_slotnums.len()
            || self
                .block_slab_ids
                .iter()
                .zip(&self.block_slotnums)
                .any(|(&slab, &slotnum)| slab == 0 || slotnum >= self.blocks_per_slab)
        {
            return Err(KvPlanError("block physical addresses are malformed"));
        }
        if self.append_slab_ids.len() != self.append_block_offsets.len() {
            return Err(KvPlanError(
                "append addresses and append_block_offsets must have equal length",
            ));
        }
        if self.append_slab_ids.len() != self.append_slotnums.len()
            || self
                .append_slab_ids
                .iter()
                .zip(&self.append_slotnums)
                .any(|(&slab, &slotnum)| slab == 0 || slotnum >= self.blocks_per_slab)
        {
            return Err(KvPlanError("append physical addresses are malformed"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvPlanError(&'static str);

impl fmt::Display for KvPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for KvPlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(block_tokens: u32, packing: u32) -> KvGroupLayout {
        KvGroupLayout::new(block_tokens, packing, KvRetention::FullContext).unwrap()
    }

    #[test]
    fn full_and_sliding_retention_are_visibly_distinct() {
        let full = full(128, 1);
        let sliding = KvGroupLayout::new(
            64,
            2,
            KvRetention::SlidingWindow {
                tokens: NonZeroU32::new(193).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(full.window_resident_blocks(), None);
        assert_eq!(sliding.window_resident_blocks().unwrap().get(), 4);
        assert!(
            KvGroupLayout::new_with_family(
                128,
                1,
                KvRetention::SlidingWindow {
                    tokens: NonZeroU32::new(128).unwrap(),
                },
                KvGroupFamily::State,
            )
            .is_err()
        );
    }

    #[test]
    fn replay_suffix_is_frozen_into_layout_identity() {
        let layout = KvLayout::new(4, vec![full(128, 1)]).unwrap();
        let replay = layout.clone().with_replay_suffix_tokens(140);
        assert_eq!(layout.replay_suffix_tokens(), 0);
        assert_eq!(replay.replay_suffix_tokens(), 140);
        assert_eq!(replay.clone(), replay);
        assert_ne!(layout, replay);
    }

    #[test]
    fn layout_freezes_order_and_bounds_kernel_geometry() {
        let layout = KvLayout::new(8, vec![full(128, 1), full(32, 4)]).unwrap();
        assert_eq!(layout.groups().len(), 2);
        assert_eq!(
            layout.packings().collect::<Vec<_>>(),
            [NonZeroU32::new(1).unwrap(), NonZeroU32::new(4).unwrap()]
        );
        assert_eq!(layout.kernel_page_capacity(0), Some(8));
        assert_eq!(layout.kernel_page_count(1), Some(33));
        assert!(KvLayout::new(MAX_KERNEL_VALUE, vec![full(1, 2)]).is_err());
    }

    #[test]
    fn parent_estimate_uses_retention_and_packs_across_requests() {
        let window = KvGroupLayout::new(
            4,
            4,
            KvRetention::SlidingWindow {
                tokens: NonZeroU32::new(5).unwrap(),
            },
        )
        .unwrap()
        .with_read_window_tokens(2)
        .unwrap();
        let state =
            KvGroupLayout::new_with_family(8, 2, KvRetention::FullContext, KvGroupFamily::State)
                .unwrap();
        let layout = KvLayout::new(1, vec![full(4, 2), window, state]).unwrap();
        assert_eq!(layout.estimated_parent_count(&[]).unwrap(), 0);
        assert_eq!(layout.estimated_parent_count(&[0, 0]).unwrap(), 0);
        // 9 full blocks / 2 + 6 window blocks / 4 + 5 state blocks / 2,
        // rounding each group up after summing across requests.
        assert_eq!(
            layout.estimated_parent_count(&[0, 1, 4, 5, 17]).unwrap(),
            10
        );
        assert_eq!(window.active_blocks(17), (3, 2));
        assert_eq!(state.active_blocks(17), (1, 2));
        let large = KvLayout::new(1, vec![full(1, 1)]).unwrap();
        assert_eq!(
            large.estimated_parent_count(&[u32::MAX, u32::MAX]).unwrap(),
            2 * u64::from(u32::MAX)
        );
    }

    #[test]
    fn shared_prefix_estimate_preserves_private_window_and_state_capacity() {
        let window = KvGroupLayout::new(
            4,
            2,
            KvRetention::SlidingWindow {
                tokens: NonZeroU32::new(5).unwrap(),
            },
        )
        .unwrap();
        let state =
            KvGroupLayout::new_with_family(4, 2, KvRetention::FullContext, KvGroupFamily::State)
                .unwrap();
        let layout = KvLayout::new(1, vec![full(4, 2), window, state]).unwrap();
        // Shared 16-token history plus two private blocks per request: five
        // parents. Window and recurrent state each reserve three more.
        assert_eq!(
            layout
                .estimated_parent_count_with_shared_prefix(&[24; 3], 16)
                .unwrap(),
            11
        );
        assert_eq!(layout.estimated_parent_count(&[24; 3]).unwrap(), 15);
        assert!(
            layout
                .estimated_parent_count_with_shared_prefix(&[24; 3], 15)
                .is_err()
        );
        assert!(
            layout
                .estimated_parent_count_with_shared_prefix(&[24; 3], 28)
                .is_err()
        );
        assert_eq!(
            layout
                .estimated_parent_count_with_shared_prefix(&[], 16)
                .unwrap(),
            0
        );
    }

    #[test]
    fn window_guard_band_declares_its_safe_write_span() {
        let group = KvGroupLayout::new(
            2,
            1,
            KvRetention::SlidingWindow {
                tokens: NonZeroU32::new(9).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(group.read_window_tokens().unwrap().get(), 9);
        assert_eq!(group.max_write_tokens().unwrap().get(), 1);
        let group = group.with_read_window_tokens(4).unwrap();
        assert_eq!(group.max_write_tokens().unwrap().get(), 6);
        assert_eq!(group.window_resident_blocks().unwrap().get(), 5);
        assert!(group.with_read_window_tokens(0).is_err());
        assert!(group.with_read_window_tokens(10).is_err());
        assert!(full(2, 1).with_read_window_tokens(1).is_err());
    }

    #[test]
    fn group_plan_validates_csr_and_append_metadata() {
        let plan = KvGroupPlan::new(
            1,
            vec![0, 2, 3],
            vec![7, 8, 11],
            vec![0, 0, 0],
            vec![0, 5],
            vec![8, 11],
            vec![0, 0],
            vec![3, 0],
            NonZeroU32::new(1).unwrap(),
        )
        .unwrap();
        assert_eq!(plan.group_index, 1);
        assert!(
            KvGroupPlan::new(
                0,
                vec![0, 1],
                vec![0],
                vec![0],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                NonZeroU32::new(1).unwrap(),
            )
            .is_err()
        );
    }
}
