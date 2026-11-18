//! Python adapters for the pure Rust cache descriptors in `resin-core`.

use std::num::NonZeroU32;
use std::sync::Arc;

use numpy::PyArrayLike1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use resin_core::resources::kv as core_kv;
use resin_core::resources::kv::{
    KvBlockTable, KvGroupFamily, KvGroupLayout, KvGroupPlan, KvLayout, KvLayoutError, KvPlanError,
    KvRetention,
};

fn layout_error(error: KvLayoutError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn plan_error(error: KvPlanError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Cache-block lifetime policy.
///
/// For history groups, `HISTORY` retains every logical block and `WINDOW`
/// retains the suffix selected by `CacheGroupLayout.window_tokens`. Recurrent
/// state groups use `HISTORY` with one private live slot per request.
#[pyclass(
    frozen,
    eq,
    hash,
    rename_all = "SCREAMING_SNAKE_CASE",
    from_py_object,
    module = "resin._resin"
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CacheRetention {
    History,
    Window,
}

/// Whether a group stores token history or private live recurrent state.
#[pyclass(
    frozen,
    eq,
    hash,
    rename_all = "SCREAMING_SNAKE_CASE",
    from_py_object,
    module = "resin._resin"
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CacheGroupFamily {
    History,
    State,
}

fn group_repr(group: KvGroupLayout) -> String {
    let family = match group.family() {
        KvGroupFamily::History => "",
        KvGroupFamily::State => ", family=CacheGroupFamily.STATE",
    };
    let read_window = match group.read_window_tokens() {
        Some(tokens) if Some(tokens) != group.retention().window_tokens() => {
            format!(", read_window_tokens={tokens}")
        }
        _ => String::new(),
    };
    match group.retention() {
        KvRetention::FullContext => format!(
            "CacheGroupLayout(block_tokens={}, blocks_per_slab={}, retention=CacheRetention.HISTORY{family})",
            group.block_tokens(),
            group.blocks_per_slab()
        ),
        KvRetention::SlidingWindow { tokens } => format!(
            "CacheGroupLayout(block_tokens={}, blocks_per_slab={}, retention=CacheRetention.WINDOW, window_tokens={tokens}{read_window}{family})",
            group.block_tokens(),
            group.blocks_per_slab()
        ),
    }
}

/// Geometry and retention policy for one type of cache.
///
/// Cache groups share a pool of equal-sized slabs. `blocks_per_slab` is how
/// many of this group's blocks fit in one slab.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CacheGroupLayout {
    inner: KvGroupLayout,
}

#[pymethods]
impl CacheGroupLayout {
    #[new]
    #[pyo3(signature = (
        block_tokens,
        *,
        blocks_per_slab = 1,
        retention = CacheRetention::History,
        window_tokens = None,
        family = CacheGroupFamily::History,
        read_window_tokens = None
    ))]
    fn new(
        block_tokens: u32,
        blocks_per_slab: u32,
        retention: CacheRetention,
        window_tokens: Option<u32>,
        family: CacheGroupFamily,
        read_window_tokens: Option<u32>,
    ) -> PyResult<Self> {
        let retention = match retention {
            CacheRetention::History => {
                if window_tokens.is_some() {
                    return Err(PyValueError::new_err(
                        "history retention does not accept window_tokens",
                    ));
                }
                KvRetention::FullContext
            }
            CacheRetention::Window => {
                let tokens = window_tokens.ok_or_else(|| {
                    PyValueError::new_err("window retention requires window_tokens")
                })?;
                let tokens = NonZeroU32::new(tokens)
                    .ok_or_else(|| PyValueError::new_err("window_tokens must be positive"))?;
                KvRetention::SlidingWindow { tokens }
            }
        };
        let family = match family {
            CacheGroupFamily::History => KvGroupFamily::History,
            CacheGroupFamily::State => KvGroupFamily::State,
        };
        let inner =
            KvGroupLayout::new_with_family(block_tokens, blocks_per_slab, retention, family)
                .map_err(layout_error)?;
        let inner = match read_window_tokens {
            Some(tokens) => inner
                .with_read_window_tokens(tokens)
                .map_err(layout_error)?,
            None => inner,
        };
        Ok(Self { inner })
    }

    /// Number of token positions represented by one logical block.
    #[getter]
    fn block_tokens(&self) -> u32 {
        self.inner.block_tokens().get()
    }

    /// Number of this group's blocks packed into one slab.
    #[getter]
    fn blocks_per_slab(&self) -> u32 {
        self.inner.blocks_per_slab().get()
    }

    /// Lifetime policy for logical blocks in this group.
    #[getter]
    fn retention(&self) -> CacheRetention {
        match self.inner.retention() {
            KvRetention::FullContext => CacheRetention::History,
            KvRetention::SlidingWindow { .. } => CacheRetention::Window,
        }
    }

    /// Whether this group stores token history or recurrent checkpoints.
    #[getter]
    fn family(&self) -> CacheGroupFamily {
        match self.inner.family() {
            KvGroupFamily::History => CacheGroupFamily::History,
            KvGroupFamily::State => CacheGroupFamily::State,
        }
    }

    /// Retained suffix length for `WINDOW` groups, otherwise `None`.
    #[getter]
    fn window_tokens(&self) -> Option<u32> {
        self.inner.retention().window_tokens().map(NonZeroU32::get)
    }

    /// Logical read window including the current token. Defaults to the full
    /// retained window; reserve extra `window_tokens` for simultaneous writes.
    #[getter]
    fn read_window_tokens(&self) -> Option<u32> {
        self.inner.read_window_tokens().map(NonZeroU32::get)
    }

    /// Maximum resident logical blocks per sequence for a window group.
    #[getter]
    fn window_resident_block_count(&self) -> Option<u32> {
        self.inner.window_resident_blocks().map(NonZeroU32::get)
    }

    fn __repr__(&self) -> String {
        group_repr(self.inner)
    }
}

/// Immutable grouped-cache geometry owned by one Engine.
///
/// - `parent_capacity`: Number of physical slabs shared by all cache groups.
/// - `groups`: Cache-group layouts in the stable order used by `BatchPlan`.
/// - `replay_suffix_tokens`: Final prompt tokens rebuilt after a prefix-cache
///   hit. Models must rebuild omitted cache state before reading it.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheLayout {
    inner: KvLayout,
}

impl CacheLayout {
    pub(crate) const fn inner_native(&self) -> &KvLayout {
        &self.inner
    }

    fn repr_string(&self) -> String {
        let groups = self
            .inner
            .groups()
            .iter()
            .copied()
            .map(group_repr)
            .collect::<Vec<_>>()
            .join(", ");
        let replay = match self.inner.replay_suffix_tokens() {
            0 => String::new(),
            tokens => format!(", replay_suffix_tokens={tokens}"),
        };
        let writes = match self.inner.decode_write_extra_tokens() {
            0 => String::new(),
            tokens => format!(", decode_write_extra_tokens={tokens}"),
        };
        format!(
            "CacheLayout(parent_capacity={}, groups=[{groups}]{replay}{writes})",
            self.inner.parent_capacity()
        )
    }
}

#[pymethods]
impl CacheLayout {
    #[new]
    #[pyo3(signature = (*, parent_capacity, groups, replay_suffix_tokens=0, decode_write_extra_tokens=0))]
    fn new(
        py: Python<'_>,
        parent_capacity: u32,
        groups: Vec<Py<CacheGroupLayout>>,
        replay_suffix_tokens: u32,
        decode_write_extra_tokens: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: KvLayout::new(
                parent_capacity,
                groups.iter().map(|group| group.borrow(py).inner).collect(),
            )
            .map_err(layout_error)?
            .with_replay_suffix_tokens(replay_suffix_tokens)
            .with_decode_write_extra_tokens(decode_write_extra_tokens),
        })
    }

    /// Minimum final prompt suffix rebuilt after every prefix-cache hit.
    #[getter]
    fn replay_suffix_tokens(&self) -> u32 {
        self.inner.replay_suffix_tokens()
    }

    /// Additional persistent writes from the resolved decode operation.
    #[getter]
    fn decode_write_extra_tokens(&self) -> u32 {
        self.inner.decode_write_extra_tokens()
    }

    /// Number of fungible physical slabs in the pool.
    #[getter]
    fn parent_capacity(&self) -> u32 {
        self.inner.parent_capacity().get()
    }

    /// Number of logical cache groups.
    #[getter]
    fn group_count(&self) -> usize {
        self.inner.groups().len()
    }

    /// Ordered immutable cache-group descriptors.
    #[getter]
    fn groups(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.inner
                .groups()
                .iter()
                .map(|&inner| CacheGroupLayout { inner }),
        )?
        .unbind())
    }

    fn group_kernel_page_capacity(&self, group_index: u32) -> PyResult<u32> {
        self.inner.kernel_page_capacity(group_index).ok_or_else(|| {
            PyValueError::new_err("group_index is outside the cache layout group range")
        })
    }

    fn group_kernel_page_count(&self, group_index: u32) -> PyResult<u64> {
        self.inner.kernel_page_count(group_index).ok_or_else(|| {
            PyValueError::new_err("group_index is outside the cache layout group range")
        })
    }

    /// Physical slabs needed for contexts in an empty pool.
    /// Applies retention and packing across requests without allocating pages;
    /// an aligned shared prefix may be declared, and demand may exceed capacity.
    #[pyo3(signature = (context_lengths, *, shared_prefix_tokens=0))]
    fn estimated_parent_count(
        &self,
        context_lengths: Vec<u32>,
        shared_prefix_tokens: u32,
    ) -> PyResult<u64> {
        self.inner
            .estimated_parent_count_with_shared_prefix(&context_lengths, shared_prefix_tokens)
            .map_err(layout_error)
    }

    fn __repr__(&self) -> String {
        self.repr_string()
    }
}

/// Optional restore sources and snapshot destinations for one STATE group.
/// Its one-slot CSR owns the live destination for each row.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
pub(crate) struct CacheStatePlan {
    inner: core_kv::StatePlan,
}

#[pymethods]
impl CacheStatePlan {
    #[new]
    #[pyo3(signature = (*, restore_copies=Vec::new(), checkpoint_slots=Vec::new()))]
    fn new(
        restore_copies: Vec<(u32, u32, u32)>,
        checkpoint_slots: Vec<(u64, u32, u32, u32, u32)>,
    ) -> Self {
        Self {
            inner: core_kv::StatePlan {
                restore_copies: restore_copies
                    .into_iter()
                    .map(|(row, source_slab_id, source_slotnum)| core_kv::StateCopy {
                        row,
                        source_slab_id,
                        source_slotnum,
                    })
                    .collect(),
                checkpoint_slots: checkpoint_slots
                    .into_iter()
                    .map(
                        |(
                            reservation_id,
                            row,
                            prefix_length,
                            destination_slab_id,
                            destination_slotnum,
                        )| core_kv::StateCheckpointSlot {
                            reservation_id,
                            row,
                            prefix_length,
                            destination_slab_id,
                            destination_slotnum,
                        },
                    )
                    .collect(),
            },
        }
    }

    #[getter]
    fn restore_copies(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.inner
                .restore_copies
                .iter()
                .map(|copy| (copy.row, copy.source_slab_id, copy.source_slotnum)),
        )?
        .unbind())
    }

    #[getter]
    fn checkpoint_slots(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.inner.checkpoint_slots.iter().map(|slot| {
                (
                    slot.reservation_id,
                    slot.row,
                    slot.prefix_length,
                    slot.destination_slab_id,
                    slot.destination_slotnum,
                )
            }),
        )?
        .unbind())
    }
}

/// Plan for one type of cache in a BatchPlan.
///
/// Each `CacheGroupLayout` describes full-attention history, sliding-window
/// history, or recurrent state. This plan maps every batch sequence to that
/// group's physical blocks and token write locations.
///
/// - `group_index`: Zero-based index of the matching `CacheGroupLayout`.
/// - `blocks_per_slab`: Number of this group's blocks packed into each slab.
/// - `block_indptr`: CSR row pointer selecting each sequence's chronological
///   block-table entries.
/// - `block_slab_ids`: One-based slab IDs for the flattened block table.
/// - `block_slotnums`: Zero-based block slots within those slabs.
/// - `block_table_start_indices`: Logical index of each row's first block;
///   zero for full history and private live state, advancing for window rows. Block
///   `b` starts at token position `b * CacheGroupLayout.block_tokens`.
/// - `append_slab_ids`: Destination slab for each flattened `BatchPlan` token.
/// - `append_slotnums`: Destination block slot for each token.
/// - `append_block_offsets`: Token offset within each destination block.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Debug)]
pub(crate) struct CacheGroupPlan {
    pub(crate) groups: Arc<[KvGroupPlan]>,
    pub(crate) index: usize,
}

impl std::ops::Deref for CacheGroupPlan {
    type Target = KvGroupPlan;

    fn deref(&self) -> &KvGroupPlan {
        &self.groups[self.index]
    }
}

impl CacheGroupPlan {
    fn repr_string(&self) -> String {
        format!(
            "CacheGroupPlan(group_index={}, blocks_per_slab={}, sequences={}, blocks={}, appends={})",
            self.group_index,
            self.blocks_per_slab,
            self.block_table_start_indices.len(),
            self.block_slab_ids.len(),
            self.append_slab_ids.len()
        )
    }
}

crate::arrays::array_methods! {
    impl CacheGroupPlan {
        arrays {
            /// CSR row pointer into the block-address arrays. Its length is one more
            /// than the batch sequence count.
            block_indptr: u32 => blocks.block_indptr;
            /// One-based slab ID for each flattened block-table entry. Pair with
            /// `block_slotnums`; slab zero is reserved as null.
            block_slab_ids: u32 => blocks.block_slab_ids;
            /// Zero-based slot number for each flattened block-table entry. Pair with
            /// `block_slab_ids`.
            block_slotnums: u32 => blocks.block_slotnums;
            /// Absolute logical-block index at which each sequence's table row starts.
            /// This is zero for full-attention history and advances for sliding-window
            /// rows. Private live state always uses zero. It has one entry per sequence and is unrelated to
            /// the physical slab ID.
            block_table_start_indices: u32 => blocks.block_table_start_indices;
            /// One-based slab ID of each token's write target. Entry `k`
            /// corresponds to `BatchPlan.token_ids[k]`.
            append_slab_ids: u32 => append_slab_ids;
            /// Zero-based slot number of each token's write target. Entry `k`
            /// corresponds to `BatchPlan.token_ids[k]`.
            append_slotnums: u32 => append_slotnums;
            /// Token offset within each token's destination block. Entry `k`
            /// corresponds to `BatchPlan.token_ids[k]` and is less than the matching
            /// `CacheGroupLayout.block_tokens`.
            append_block_offsets: u32 => append_block_offsets;
        }

        #[new]
        #[pyo3(signature = (
            *,
            group_index,
            block_indptr: "Sequence[int] | npt.NDArray[np.uint32]",
            block_slab_ids: "Sequence[int] | npt.NDArray[np.uint32]",
            block_slotnums: "Sequence[int] | npt.NDArray[np.uint32]",
            block_table_start_indices: "Sequence[int] | npt.NDArray[np.uint32]",
            append_slab_ids: "Sequence[int] | npt.NDArray[np.uint32]",
            append_slotnums: "Sequence[int] | npt.NDArray[np.uint32]",
            append_block_offsets: "Sequence[int] | npt.NDArray[np.uint32]",
            blocks_per_slab = 1,
            state = None
        ))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            py: Python<'_>,
            group_index: u32,
            block_indptr: PyArrayLike1<'_, u32>,
            block_slab_ids: PyArrayLike1<'_, u32>,
            block_slotnums: PyArrayLike1<'_, u32>,
            block_table_start_indices: PyArrayLike1<'_, u32>,
            append_slab_ids: PyArrayLike1<'_, u32>,
            append_slotnums: PyArrayLike1<'_, u32>,
            append_block_offsets: PyArrayLike1<'_, u32>,
            blocks_per_slab: u32,
            state: Option<Py<CacheStatePlan>>,
        ) -> PyResult<Self> {
            let plan = KvGroupPlan {
                group_index,
                blocks_per_slab,
                blocks: Arc::new(KvBlockTable {
                    block_indptr: crate::arrays::owned_vec(block_indptr),
                    block_slab_ids: crate::arrays::owned_vec(block_slab_ids),
                    block_slotnums: crate::arrays::owned_vec(block_slotnums),
                    block_table_start_indices: crate::arrays::owned_vec(block_table_start_indices),
                }),
                append_slab_ids: crate::arrays::owned_vec(append_slab_ids),
                append_slotnums: crate::arrays::owned_vec(append_slotnums),
                append_block_offsets: crate::arrays::owned_vec(append_block_offsets),
                state: state.as_ref().map(|state| state.borrow(py).inner.clone()),
            };
            plan.validate().map_err(plan_error)?;
            Ok(Self { groups: Arc::new([plan]), index: 0 })
        }

        /// Zero-based index of this plan's matching CacheGroupLayout. BatchPlan
        /// orders group plans by this index and rejects gaps or reordering.
        #[getter]
        fn group_index(&self) -> u32 {
            self.group_index
        }

        /// Number of this group's blocks packed into one slab. Valid slot numbers
        /// are `0 <= slotnum < blocks_per_slab`.
        #[getter]
        fn blocks_per_slab(&self) -> u32 {
            self.blocks_per_slab
        }

        #[getter]
        fn state(&self) -> Option<CacheStatePlan> {
            self.state
                .clone()
                .map(|inner| CacheStatePlan { inner })
        }








        fn __repr__(&self) -> String {
            self.repr_string()
        }
    }
}
