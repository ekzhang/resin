#![expect(
    clippy::clone_on_copy,
    reason = "PyO3 0.29 from_py_object emits Clone::clone for Copy enums"
)]

use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::{Arc, OnceLock};

use numpy::{PyArray1, PyArrayLike1, PyReadonlyArray1};
use pyo3::exceptions::{
    PyAttributeError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

mod arrays;
mod cache;
mod cache_storage;
mod http;
mod http_distributed;
mod http_logprobs;
mod http_output;
mod http_session;
mod http_stop;
mod media_payload;
mod media_preprocess;
mod metrics;
mod numa;
mod output;
mod parallelism;
mod recipe;
mod recipe_events;
mod recipe_types;
mod runner;

use cache::{
    CacheGroupFamily, CacheGroupLayout, CacheGroupPlan, CacheLayout, CacheRetention, CacheStatePlan,
};
use output::{BorrowedOutput, BorrowedScores, ForwardOutput};
use parallelism::Parallelism;
use resin_core::engine::{self as core, CompletionReason};
use resin_core::logprobs::DEFAULT_MAX_TOP_LOGPROBS;
use resin_core::resources::paged_kv::PrefixStoragePolicy;
use resin_core::{BatchPlan as NativeBatchPlan, Phase};
use runner::{
    ResourceBinding, RunnerConfig, bind_runner_resources, optional_runner_callable,
    resolve_runner_config, resolve_runner_contract, runner_error_is_post_accept,
};

fn default_sampling_seed(engine_id: u64, request_id: u64) -> i64 {
    // SplitMix64 gives stable, well-distributed row seeds without adding a
    // control-plane RNG or making batch composition part of randomness.
    let mut value = engine_id ^ request_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) & i64::MAX as u64) as i64
}

/// Native admission and token-budget policy for an Engine.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct SchedulerConfig {
    max_batch_tokens: Option<NonZeroU32>,
    prefill_chunk_size: NonZeroU32,
    #[pyo3(get)]
    schedule_conservativeness: f64,
}

const DEFAULT_PREFILL_CHUNK_SIZE: NonZeroU32 = NonZeroU32::new(8192).unwrap();

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_tokens: None,
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
            schedule_conservativeness: 1.0,
        }
    }
}

#[pymethods]
impl SchedulerConfig {
    #[new]
    #[pyo3(signature = (
        *,
        max_batch_tokens = None,
        prefill_chunk_size = 8192,
        schedule_conservativeness = 1.0
    ))]
    fn new(
        max_batch_tokens: Option<u32>,
        prefill_chunk_size: u32,
        schedule_conservativeness: f64,
    ) -> PyResult<Self> {
        if !schedule_conservativeness.is_finite() || schedule_conservativeness < 0.0 {
            return Err(PyValueError::new_err(
                "schedule_conservativeness must be finite and non-negative",
            ));
        }
        Ok(Self {
            max_batch_tokens: max_batch_tokens
                .map(|value| {
                    NonZeroU32::new(value)
                        .ok_or_else(|| PyValueError::new_err("max_batch_tokens must be positive"))
                })
                .transpose()?,
            prefill_chunk_size: NonZeroU32::new(prefill_chunk_size)
                .ok_or_else(|| PyValueError::new_err("prefill_chunk_size must be positive"))?,
            schedule_conservativeness,
        })
    }

    #[getter]
    fn max_batch_tokens(&self) -> Option<u32> {
        self.max_batch_tokens.map(NonZeroU32::get)
    }

    #[getter]
    fn prefill_chunk_size(&self) -> u32 {
        self.prefill_chunk_size.get()
    }

    fn __repr__(&self) -> String {
        format!(
            "SchedulerConfig(max_batch_tokens={}, prefill_chunk_size={}, schedule_conservativeness={:?})",
            self.max_batch_tokens()
                .map_or_else(|| "None".to_owned(), |value| value.to_string()),
            self.prefill_chunk_size(),
            self.schedule_conservativeness,
        )
    }
}

#[pyclass(
    frozen,
    eq,
    hash,
    rename_all = "SCREAMING_SNAKE_CASE",
    from_py_object,
    module = "resin._resin"
)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum BatchPhase {
    Prefill,
    Decode,
}

impl BatchPhase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "PREFILL",
            Self::Decode => "DECODE",
        }
    }

    const fn core(self) -> resin_core::Phase {
        match self {
            Self::Prefill => resin_core::Phase::Prefill,
            Self::Decode => resin_core::Phase::Decode,
        }
    }

    const fn from_core(phase: Phase) -> Self {
        match phase {
            Phase::Prefill => Self::Prefill,
            Phase::Decode => Self::Decode,
        }
    }
}

/// Per-request generation policy, including sampling, length, and stop tokens.
///
/// A zero temperature selects the maximum-logit token. Positive temperatures
/// enable stochastic sampling after optional top-k and top-p filtering.
/// An omitted `max_output_tokens` uses the remaining context budget. Stop tokens
/// are explicit: pass a recipe's `stop_token_ids` for recipe stopping.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug)]
struct SamplingParams {
    #[pyo3(get)]
    max_output_tokens: Option<u32>,
    stop_token_ids: Arc<[u32]>,
    #[pyo3(get)]
    temperature: f64,
    #[pyo3(get)]
    top_p: f64,
    #[pyo3(get)]
    top_k: Option<u32>,
    #[pyo3(get)]
    seed: Option<i64>,
    #[pyo3(get)]
    logprobs: bool,
    #[pyo3(get)]
    top_logprobs: u32,
    #[pyo3(get)]
    prompt_logprobs: bool,
    #[pyo3(get)]
    prompt_top_logprobs: u32,
    extra: Option<Arc<Py<PyAny>>>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_output_tokens: None,
            stop_token_ids: Arc::default(),
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            seed: None,
            logprobs: false,
            top_logprobs: 0,
            prompt_logprobs: false,
            prompt_top_logprobs: 0,
            extra: None,
        }
    }
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (*, max_output_tokens = None, stop_token_ids = Vec::new(), temperature = 1.0, top_p = 1.0, top_k = None, seed = None, logprobs = false, top_logprobs = 0, prompt_logprobs = false, prompt_top_logprobs = 0, extra = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_output_tokens: Option<u32>,
        stop_token_ids: Vec<u32>,
        temperature: f64,
        top_p: f64,
        top_k: Option<u32>,
        seed: Option<i64>,
        logprobs: bool,
        top_logprobs: u32,
        prompt_logprobs: bool,
        prompt_top_logprobs: u32,
        extra: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(PyValueError::new_err(
                "temperature must be finite and non-negative",
            ));
        }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
            return Err(PyValueError::new_err(
                "top_p must be finite and in the interval (0, 1]",
            ));
        }
        if top_k == Some(0) {
            return Err(PyValueError::new_err(
                "top_k must be positive when supplied",
            ));
        }
        if seed.is_some_and(|seed| seed < 0) {
            return Err(PyValueError::new_err(
                "seed must be non-negative when supplied",
            ));
        }
        Ok(Self {
            max_output_tokens,
            stop_token_ids: stop_token_ids.into(),
            temperature,
            top_p,
            top_k,
            seed,
            logprobs: logprobs || top_logprobs > 0,
            top_logprobs,
            prompt_logprobs: prompt_logprobs || prompt_top_logprobs > 0,
            prompt_top_logprobs,
            extra: extra.map(Arc::new),
        })
    }

    #[getter]
    fn stop_token_ids(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.stop_token_ids.iter().copied())?.unbind())
    }

    #[getter]
    fn extra(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.extra
            .as_ref()
            .map(|extra| extra.as_ref().clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingParams(max_output_tokens={:?}, stop_token_ids={:?}, temperature={:?}, top_p={:?}, top_k={:?}, seed={:?}, logprobs={}, top_logprobs={}, extra={})",
            self.max_output_tokens,
            self.stop_token_ids,
            self.temperature,
            self.top_p,
            self.top_k,
            self.seed,
            self.logprobs,
            self.top_logprobs,
            if self.extra.is_some() {
                "<object>"
            } else {
                "None"
            }
        )
    }
}

/// Immutable execution plan for one prefill or decode step.
///
/// This runs one forward pass on a batch of sequences. `token_ids` contains
/// the newly scheduled input tokens, packed by sequence into one flat array;
/// `token_indptr` is the CSR row pointer, so sequence `i` owns
/// `token_ids[token_indptr[i]:token_indptr[i + 1]]`.
///
/// - `context_lengths`: Tokens already processed per sequence; new token
///   positions start here.
/// - `prefill_lengths`: Prefill endpoints, fixed across chunks, including
///   retraction replay. Defaults to each row's scheduled end.
/// - `cache_groups`: One placement plan per `CacheLayout` group, covering
///   every sequence.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Debug)]
struct BatchPlan {
    batch: NativeBatchPlan,
    payloads: Arc<Vec<RequestPayload>>,
    positions: OnceLock<Vec<u32>>,
}

impl std::ops::Deref for BatchPlan {
    type Target = NativeBatchPlan;

    fn deref(&self) -> &NativeBatchPlan {
        &self.batch
    }
}

impl From<core::ExecutionPlan<RequestPayload>> for BatchPlan {
    fn from(plan: core::ExecutionPlan<RequestPayload>) -> Self {
        Self {
            batch: plan.batch,
            payloads: Arc::new(plan.payloads),
            positions: OnceLock::new(),
        }
    }
}

impl BatchPlan {
    fn restore_requests(
        batch: NativeBatchPlan,
        requests: &Bound<'_, PyDict>,
        pending: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let mut payloads = Vec::with_capacity(batch.sequence_ids.len());
        for &id in &batch.sequence_ids {
            let request = match pending.get_item(id)? {
                Some(request) => request,
                None => requests.get_item(id)?.ok_or_else(|| {
                    PyValueError::new_err(format!("request {id} has not been registered"))
                })?,
            };
            let sampling_params = request
                .getattr("sampling_params")?
                .extract::<PyRef<'_, SamplingParams>>()?
                .clone();
            let media = request.getattr("media")?;
            let media = if media.is_none() {
                None
            } else {
                Some(Arc::new(media.cast_into::<PyTuple>()?.unbind()))
            };
            payloads.push(RequestPayload {
                sampling_params,
                media,
            });
        }
        Self::checked(Self {
            batch,
            payloads: Arc::new(payloads),
            positions: OnceLock::new(),
        })
    }

    fn checked(plan: Self) -> PyResult<Self> {
        plan.batch.validate().map_err(core_error)?;
        if plan
            .payloads
            .iter()
            .any(|payload| payload.sampling_params.seed.is_none())
        {
            return Err(PyValueError::new_err(
                "sampling_params must contain an effective seed for every sequence",
            ));
        }
        Ok(plan)
    }
}
crate::arrays::array_methods! {
    impl BatchPlan {
        arrays {
            /// Stable request identifier for each sequence.
            sequence_ids: u64 => batch.sequence_ids;
            /// Number of tokens already processed for each sequence.
            context_lengths: u32 => batch.context_lengths;
            /// Final history endpoint for each prefill, including retracted generated inputs.
            /// Only meaningful during prefill; constant across that history's chunks.
            prefill_lengths: u32 => batch.prefill_lengths;
            /// Flattened scheduled token IDs, grouped by sequence.
            token_ids: u32 => batch.token_ids;
            /// Offsets into `token_ids` for each sequence.
            token_indptr: u32 => batch.token_indptr;
            /// Which sequences should generate a sampled token. Used for chunked
            /// prefill, when some sequences only advance the cache.
            sampling_mask: bool => batch.sampling_mask;
            prompt_score_indices: u32 => batch.prompt_score_indices;
            prompt_score_token_ids: u32 => batch.prompt_score_token_ids;
            prompt_score_indptr: u32 => batch.prompt_score_indptr;
            prompt_score_top_counts: u32 => batch.prompt_score_top_counts;
            prompt_target_indptr: u32 => batch.prompt_target_indptr;
            prompt_target_token_ids: u32 => batch.prompt_target_token_ids;
        }

        #[pyo3(signature = () -> "npt.NDArray[np.uint8]")]
        fn _to_wire<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
            Ok(PyArray1::from_vec(py, self.batch.encode_wire().map_err(core_error)?))
        }

        #[staticmethod]
        #[pyo3(signature = (
            data: "npt.NDArray[np.uint8]",
            requests: "dict[int, Any]",
            pending: "dict[int, Any]"
        ))]
        fn _from_wire(
            data: PyReadonlyArray1<'_, u8>,
            requests: &Bound<'_, PyDict>,
            pending: &Bound<'_, PyDict>,
        ) -> PyResult<Self> {
            let bytes = data.as_slice().map_err(|_| {
                PyValueError::new_err("plan wire data must be a contiguous uint8 array")
            })?;
            Self::restore_requests(NativeBatchPlan::decode_wire(bytes).map_err(core_error)?, requests, pending)
        }

        fn _validate_cache_layout(&self, layout: &CacheLayout, decode_lookahead: u64) -> PyResult<()> {
            self.batch.validate_cache_layout(layout.inner_native(), decode_lookahead).map_err(core_error)
        }

        #[new]
        #[pyo3(signature = (
            *,
            sequence_ids: "Sequence[int] | npt.NDArray[np.uint64]",
            context_lengths: "Sequence[int] | npt.NDArray[np.uint32]",
            token_ids: "Sequence[int] | npt.NDArray[np.uint32]",
            token_indptr: "Sequence[int] | npt.NDArray[np.uint32]",
            prefill_lengths: "Sequence[int] | npt.NDArray[np.uint32] | None" = None,
            phase = BatchPhase::Prefill,
            batch_id = 0,
            batch_key = None,
            cache_groups = None,
            sampling_mask: "Sequence[bool] | npt.NDArray[np.bool_] | None" = None,
            sampling_params = None,
            media: "Sequence[tuple[Media, ...] | None] | None" = None,
            prompt_score_indices: "Sequence[int] | npt.NDArray[np.uint32] | None" = None,
            prompt_score_token_ids: "Sequence[int] | npt.NDArray[np.uint32] | None" = None,
            prompt_score_indptr: "Sequence[int] | npt.NDArray[np.uint32] | None" = None,
            prompt_score_top_counts: "Sequence[int] | npt.NDArray[np.uint32] | None" = None,
            prompt_target_indptr: "Sequence[int] | npt.NDArray[np.uint32] | None" = None,
            prompt_target_token_ids: "Sequence[int] | npt.NDArray[np.uint32] | None" = None
        ))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            py: Python<'_>,
            sequence_ids: PyArrayLike1<'_, u64>,
            context_lengths: PyArrayLike1<'_, u32>,
            token_ids: PyArrayLike1<'_, u32>,
            token_indptr: PyArrayLike1<'_, u32>,
            prefill_lengths: Option<PyArrayLike1<'_, u32>>,
            phase: BatchPhase,
            batch_id: u64,
            batch_key: Option<u64>,
            cache_groups: Option<Vec<Py<CacheGroupPlan>>>,
            sampling_mask: Option<PyArrayLike1<'_, bool>>,
            sampling_params: Option<Vec<Py<SamplingParams>>>,
            media: Option<Vec<Option<Py<PyTuple>>>>,
            prompt_score_indices: Option<PyArrayLike1<'_, u32>>,
            prompt_score_token_ids: Option<PyArrayLike1<'_, u32>>,
            prompt_score_indptr: Option<PyArrayLike1<'_, u32>>,
            prompt_score_top_counts: Option<PyArrayLike1<'_, u32>>,
            prompt_target_indptr: Option<PyArrayLike1<'_, u32>>,
            prompt_target_token_ids: Option<PyArrayLike1<'_, u32>>,
        ) -> PyResult<Self> {
            let sequence_ids = arrays::owned_vec(sequence_ids);
            let context_lengths = arrays::owned_vec(context_lengths);
            let token_ids = arrays::owned_vec(token_ids);
            let token_indptr = arrays::owned_vec(token_indptr);
            // Malformed CSR inputs are rejected by checked() below; deriving the
            // default prefill endpoint must not underflow before that validation.
            let prefill_lengths = prefill_lengths.map(arrays::owned_vec).unwrap_or_else(|| {
                context_lengths
                    .iter()
                    .zip(token_indptr.windows(2))
                    .map(|(context, range)| context.saturating_add(range[1].saturating_sub(range[0])))
                    .collect()
            });
            let sequence_count = sequence_ids.len();
            let cache_groups = cache_groups
                .unwrap_or_default()
                .into_iter()
                .map(|group| (**group.get()).clone())
                .collect();
            let sampling_mask = sampling_mask
                .map(arrays::owned_vec)
                .unwrap_or_else(|| vec![true; sequence_count]);
            let mut sampling_params = sampling_params
                .map(|params| {
                    params
                        .into_iter()
                        .map(|params| params.borrow(py).clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![SamplingParams::default(); sequence_count]);
            for (params, sequence_id) in sampling_params.iter_mut().zip(&sequence_ids) {
                params
                    .seed
                    .get_or_insert_with(|| default_sampling_seed(0, *sequence_id));
            }
            if sampling_params.len() != sequence_count {
                return Err(PyValueError::new_err("sampling_params must contain one item per sequence"));
            }
            if media.as_ref().is_some_and(|media| media.len() != sequence_count) {
                return Err(PyValueError::new_err("media must contain one item per sequence"));
            }
            let media = media
                .unwrap_or_else(|| (0..sequence_count).map(|_| None).collect())
                .into_iter()
                .map(|items| items.filter(|v| !v.bind(py).is_empty()).map(Arc::new));
            let payloads = sampling_params.into_iter().zip(media)
                .map(|(sampling_params, media)| RequestPayload { sampling_params, media })
                .collect();
            let prompt_score_indices = prompt_score_indices
                .map(arrays::owned_vec)
                .unwrap_or_default();
            let predecessor_count = prompt_score_indices.len();
            Self::checked(Self {
                batch: NativeBatchPlan {
                    batch_id,
                    batch_key,
                    phase: phase.core(),
                    sequence_ids,
                    context_lengths,
                    prefill_lengths,
                    token_ids,
                    token_indptr,
                    sampling_mask,
                    cache_groups,
                    prompt_score_indices,
                    prompt_score_top_counts: prompt_score_top_counts
                        .map(arrays::owned_vec)
                        .unwrap_or_else(|| vec![0; predecessor_count]),
                    prompt_target_indptr: prompt_target_indptr.map(arrays::owned_vec).unwrap_or_else(
                        || {
                            if predecessor_count == 0 {
                                Vec::new()
                            } else {
                                vec![0; predecessor_count + 1]
                            }
                        },
                    ),
                    prompt_target_token_ids: prompt_target_token_ids
                        .map(arrays::owned_vec)
                        .unwrap_or_default(),
                    prompt_score_token_ids: prompt_score_token_ids
                        .map(arrays::owned_vec)
                        .unwrap_or_default(),
                    prompt_score_indptr: prompt_score_indptr
                        .map(arrays::owned_vec)
                        .unwrap_or_else(|| vec![0; sequence_count + 1]),
                },
                payloads: Arc::new(payloads),
                positions: OnceLock::new(),
            })
        }






        /// Absolute position of every flattened scheduled token.
        #[getter]
        fn positions<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
            let plan = this.get();
            let positions = plan.positions.get_or_init(|| {
                let mut positions = Vec::with_capacity(plan.token_ids.len());
                for (context, range) in plan
                    .context_lengths
                    .iter()
                    .zip(plan.token_indptr.windows(2))
                {
                    positions.extend(*context..*context + (range[1] - range[0]));
                }
                positions
            });
            arrays::readonly_view(this.clone().into_any(), positions)
        }

        /// Ordered physical placement plans corresponding to CacheLayout groups.
        /// There is one plan per layout group, not one plan per sequence; each plan
        /// contains one block-table row for every sequence in this batch.
        #[getter]
        fn cache_groups(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            let groups = (0..self.cache_groups.len()).map(|index| CacheGroupPlan {
                groups: self.cache_groups.clone(),
                index,
            });
            Ok(PyTuple::new(py, groups)?.unbind())
        }


        /// Check natively so token-only graph replay does not box per-row policies.
        #[getter]
        fn needs_logprobs(&self) -> bool {
            self.sampling_mask
                .iter()
                .zip(self.payloads.iter())
                .any(|(sample, payload)| *sample && payload.sampling_params.logprobs)
        }
        #[getter]
        fn needs_prompt_logprobs(&self) -> bool {
            !self.prompt_score_indices.is_empty()
        }

        /// Effective sampling parameters, one per sequence.
        #[getter]
        fn sampling_params(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            Ok(PyTuple::new(py, self.payloads.iter().map(|row| row.sampling_params.clone()))?.unbind())
        }

        /// Opaque runner inputs, one per sequence. The scheduler never inspects them.
        #[getter]
        fn media(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            let payloads = self
                .payloads
                .iter()
                .map(|payload| {
                    payload.media
                        .as_ref()
                        .map_or_else(|| py.None(), |payload| payload.clone_ref(py).into_any())
                })
                .collect::<Vec<_>>();
            Ok(PyTuple::new(py, payloads)?.unbind())
        }

        /// Whether this is a prefill or decode batch.
        #[getter]
        fn phase(&self) -> BatchPhase { BatchPhase::from_core(self.batch.phase) }

        /// Monotonic identifier for tracing and correlating runner steps.
        #[getter]
        fn batch_id(&self) -> u64 { self.batch.batch_id }

        /// Optional key that prevents incompatible requests from sharing a batch.
        #[getter]
        fn batch_key(&self) -> Option<u64> { self.batch.batch_key }

        /// Number of sequences (batch size).
        #[getter]
        fn num_sequences(&self) -> usize {
            self.sequence_ids.len()
        }

        /// Number of flattened scheduled tokens.
        #[getter]
        fn num_tokens(&self) -> usize {
            self.token_ids.len()
        }

        /// Return the half-open range in `token_ids` for one sequence.
        fn token_range(&self, sequence_index: usize) -> PyResult<(usize, usize)> {
            if sequence_index >= self.sequence_ids.len() {
                return Err(PyValueError::new_err("sequence index is out of range"));
            }
            Ok((
                self.token_indptr[sequence_index] as usize,
                self.token_indptr[sequence_index + 1] as usize,
            ))
        }

        fn __repr__(&self) -> String {
            format!(
                "BatchPlan(batch_id={}, phase=BatchPhase.{}, sequences={}, tokens={}, cache_groups={})",
                self.batch_id,
                self.phase().as_str(),
                self.sequence_ids.len(),
                self.token_ids.len(),
                self.cache_groups.len()
            )
        }
    }
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug)]
struct TokenEvent {
    #[pyo3(get)]
    request_id: u64,
    #[pyo3(get)]
    token_id: u32,
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    finished: bool,
    finish_reason: Option<CompletionReason>,
    logprobs: Option<core::TokenLogprobs>,
}

#[pymethods]
impl TokenEvent {
    #[getter]
    fn logprobs(&self) -> Option<TokenLogprobs> {
        self.logprobs.clone().map(|inner| TokenLogprobs { inner })
    }
    #[getter]
    fn finish_reason(&self) -> Option<&'static str> {
        self.finish_reason.map(CompletionReason::as_str)
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenEvent(request_id={}, token_id={}, index={}, finished={}, finish_reason={:?})",
            self.request_id,
            self.token_id,
            self.index,
            self.finished,
            self.finish_reason.map(CompletionReason::as_str)
        )
    }
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone)]
struct TokenLogprobs {
    inner: core::TokenLogprobs,
}

#[pymethods]
impl TokenLogprobs {
    #[new]
    #[pyo3(signature = (logprob, top_logprobs = Vec::new(), target_logprobs = Vec::new()))]
    fn new(logprob: f32, top_logprobs: Vec<(u32, f32)>, target_logprobs: Vec<(u32, f32)>) -> Self {
        Self {
            inner: core::TokenLogprobs {
                logprob,
                top_logprobs,
                target_logprobs,
            },
        }
    }
    #[getter]
    fn logprob(&self) -> f32 {
        self.inner.logprob
    }
    #[getter]
    fn top_logprobs(&self) -> Vec<(u32, f32)> {
        self.inner.top_logprobs.clone()
    }
    #[getter]
    fn target_logprobs(&self) -> Vec<(u32, f32)> {
        self.inner.target_logprobs.clone()
    }
}

/// A caller-owned lifetime handle for one submitted request.
///
/// The Engine keeps the shared state only while the request is active. Once
/// the request retires, its generated tokens are moved into this lease. If
/// the caller drops the lease without taking the result, the token buffer is
/// released when the last internal reference disappears.
#[pyclass(module = "resin._resin")]
struct RequestLease {
    inner: core::RequestHandle,
}
#[pymethods]
impl RequestLease {
    /// Raw input-token logprobs; the first token has no predecessor and is None.
    #[getter]
    fn prompt_logprobs(&self) -> Option<Vec<Option<TokenLogprobs>>> {
        self.inner.prompt_logprobs().map(|scores| {
            scores
                .into_iter()
                .map(|score| score.map(|inner| TokenLogprobs { inner }))
                .collect()
        })
    }
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id()
    }
    #[getter]
    fn request_id(&self) -> u64 {
        self.inner.id()
    }
    #[getter]
    fn done(&self) -> bool {
        self.inner.done()
    }
    #[getter]
    fn cancelled(&self) -> bool {
        self.inner.cancelled()
    }
    /// Number of prompt tokens restored from the prefix cache at admission.
    #[getter]
    fn cached_tokens(&self) -> u32 {
        self.inner.cached_tokens()
    }
    /// Consume the completed token buffer exactly once.
    fn take_result(&self) -> PyResult<Vec<u32>> {
        self.inner.take_result().map_err(core_error)
    }
    fn __int__(&self) -> u64 {
        self.inner.id()
    }
    fn __index__(&self) -> u64 {
        self.inner.id()
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.extract::<PyRef<'_, RequestLease>>() {
            return Ok(self.inner.id() == other.inner.id()
                && self.inner.engine_id() == other.inner.engine_id());
        }
        Ok(other.extract::<u64>().is_ok_and(|id| id == self.inner.id()))
    }
    fn __repr__(&self) -> String {
        format!(
            "RequestLease(id={}, done={}, cancelled={})",
            self.id(),
            self.done(),
            self.cancelled()
        )
    }
}

#[derive(Clone, Debug)]
struct RequestPayload {
    sampling_params: SamplingParams,
    media: Option<Arc<Py<PyTuple>>>,
}

fn core_error(error: core::Error) -> PyErr {
    match error.kind {
        core::ErrorKind::InvalidInput => PyValueError::new_err(error.to_string()),
        core::ErrorKind::Memory => PyMemoryError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

/// Only language conversion and callback invocation belong in this adapter.
struct PythonRunner<'a, 'py> {
    py: Python<'py>,
    release: Option<&'a Py<PyAny>>,
    error: Option<PyErr>,
    scores: Option<BorrowedScores<'py>>,
    state_checkpoints: Vec<u64>,
}
impl<'a, 'py> PythonRunner<'a, 'py> {
    fn new(py: Python<'py>, release: Option<&'a Py<PyAny>>) -> Self {
        Self {
            py,
            release,
            error: None,
            scores: None,
            state_checkpoints: Vec::new(),
        }
    }

    fn failed(&mut self, error: PyErr, accepted: bool) -> core::Error {
        let message = error.to_string();
        self.error = Some(error);
        if accepted {
            core::Error::execution(message)
        } else {
            core::Error::rejected(message)
        }
    }
    fn result<T>(&mut self, result: core::Result<T>) -> PyResult<T> {
        result.map_err(|error| self.error.take().unwrap_or_else(|| core_error(error)))
    }
}
impl<'py> core::Runner<RequestPayload> for PythonRunner<'_, 'py> {
    fn forward(
        &mut self,
        _plan: core::ExecutionPlan<RequestPayload>,
    ) -> core::Result<core::OwnedTokenRows> {
        // Python execution uses prepare/complete so NumPy output stays borrowed.
        Err(core::Error::internal(
            "Python Engine invokes its runner directly",
        ))
    }
    fn logprobs(&self) -> Option<core::LogprobView<'_>> {
        self.scores.as_ref().and_then(BorrowedScores::logprobs)
    }
    fn prompt_logprobs(&self) -> Option<core::LogprobView<'_>> {
        self.scores
            .as_ref()
            .and_then(BorrowedScores::prompt_logprobs)
    }
    fn state_checkpoints(&self) -> &[u64] {
        &self.state_checkpoints
    }
    fn release(&mut self, ids: &[u64]) -> core::Result<()> {
        if let Some(release) = self.release {
            release
                .bind(self.py)
                .call1((ids.to_vec(),))
                .map_err(|error| self.failed(error, true))?;
        }
        Ok(())
    }
}

#[pyclass(module = "resin._resin")]
struct Engine {
    core: core::EngineCore<RequestPayload>,
    runner_forward: Py<PyAny>,
    runner_release: Option<Py<PyAny>>,
    runner_config: RunnerConfig,
    runner_enqueue: Option<Py<PyAny>>,
    overlap: bool,
    pending: Option<PendingExecution>,
    deferred_events: Vec<Py<TokenEvent>>,
    failed_executions: Vec<Py<PyAny>>,
    overlap_enqueues: u64,
    overlap_continuations: u64,
    runner_metrics: Option<Py<PyAny>>,
}

struct PendingExecution {
    plan: Py<BatchPlan>,
    completion: Py<PyAny>,
}

impl Engine {
    fn step_once(
        &mut self,
        py: Python<'_>,
        phase: Option<BatchPhase>,
        allow_overlap: bool,
    ) -> PyResult<Vec<Py<TokenEvent>>> {
        self.core.raise_if_open().map_err(core_error)?;
        self.core.poll_storage(false).map_err(core_error)?;
        if self.pending.is_some() {
            let feedback = if self.overlap
                && allow_overlap
                && phase.is_none_or(|phase| matches!(phase, BatchPhase::Decode))
                && self.core.has_continuation()
            {
                let previous = self.pending.as_ref().expect("pending execution");
                match previous.completion.bind(py).getattr("feedback") {
                    Ok(feedback) if feedback.is_none() => None,
                    Ok(feedback) => Some(feedback),
                    Err(error) => {
                        self.core.poison_message(format!(
                            "accepted completion feedback failed: {error}"
                        ));
                        return Err(error);
                    }
                }
            } else {
                None
            };
            // A runner may explicitly drain this completion when its runtime
            // cannot safely launch the next reserved position (for example a
            // small-batch attention plan). Inspect before consuming a batch ID
            // or constructing conservative continuation metadata.
            if let Some(feedback) = feedback
                && let Some(batch) = self.core.continuation_batch().map_err(core_error)?
            {
                let previous = self.pending.as_ref().expect("pending continuation");
                let plan = Py::new(
                    py,
                    BatchPlan {
                        batch,
                        payloads: previous.plan.get().payloads.clone(),
                        positions: OnceLock::new(),
                    },
                )?;
                return self.enqueue(py, plan, Some(feedback));
            }
            return self.harvest(py, false);
        }

        let mut runner = PythonRunner::new(py, self.runner_release.as_ref());
        let overlap_admission = self.overlap && allow_overlap && self.runner_enqueue.is_some();
        let prepared = if overlap_admission {
            self.core
                .prepare_overlap_step(&mut runner, phase.map(BatchPhase::core))
        } else {
            self.core
                .prepare_step(&mut runner, phase.map(BatchPhase::core))
        };
        let Some(plan) = runner.result(prepared)? else {
            return Ok(Vec::new());
        };
        drop(runner);
        let can_enqueue = overlap_admission
            && plan.phase == resin_core::Phase::Decode
            && self.core.has_continuation();
        let plan = match Py::new(py, BatchPlan::from(plan)) {
            Ok(plan) => plan,
            Err(error) => return Err(self.fail_output(py, error, false)),
        };
        if can_enqueue {
            return self.enqueue(py, plan, None);
        }
        match self.runner_forward.bind(py).call1((&plan,)) {
            Ok(output) => self.consume_output(py, &plan, &output, false),
            Err(error) => Err(self.fail_output(py, error, false)),
        }
    }

    fn enqueue(
        &mut self,
        py: Python<'_>,
        plan: Py<BatchPlan>,
        feedback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<TokenEvent>>> {
        let continuing = feedback.is_some();
        let enqueue = self
            .runner_enqueue
            .as_ref()
            .expect("checked enqueue")
            .bind(py);
        let result = if let Some(feedback) = feedback {
            let kwargs = PyDict::new(py);
            kwargs.set_item("feedback", feedback)?;
            enqueue.call((&plan,), Some(&kwargs))
        } else {
            enqueue.call1((&plan,))
        }
        .map_err(|error| {
            if continuing {
                self.core
                    .poison_message(format!("overlapped enqueue failed: {error}"));
                error
            } else {
                self.fail_output(py, error, false)
            }
        })?;
        let asynchronous = match runner_result_is_async(&result) {
            Ok(asynchronous) => asynchronous,
            Err(error) => {
                self.failed_executions.push(result.unbind());
                if continuing {
                    self.core
                        .poison_message(format!("accepted completion inspection failed: {error}"));
                    return Err(error);
                }
                return Err(self.fail_output(py, error, true));
            }
        };

        // A continuation has already used the previous step's lease, even
        // when the runner completed synchronously. Commit in launch order.
        let mut events = Vec::new();
        if continuing {
            match self.harvest(py, true) {
                Ok(previous) => events = previous,
                Err(error) => {
                    self.failed_executions.push(result.unbind());
                    return Err(error);
                }
            }
        }
        if asynchronous {
            self.pending = Some(PendingExecution {
                plan,
                completion: result.unbind(),
            });
            self.overlap_enqueues += 1;
            self.overlap_continuations += u64::from(continuing);
        } else {
            match self.consume_output(py, &plan, &result, false) {
                Ok(completed) => events.extend(completed),
                Err(error) => {
                    self.failed_executions.push(result.unbind());
                    return Err(error);
                }
            }
        }
        Ok(events)
    }

    fn consume_output(
        &mut self,
        py: Python<'_>,
        plan: &Py<BatchPlan>,
        output: &Bound<'_, PyAny>,
        continuing: bool,
    ) -> PyResult<Vec<Py<TokenEvent>>> {
        let mut runner = PythonRunner::new(py, self.runner_release.as_ref());
        let tokens;
        let output = match extract_runner_output(output) {
            Ok(output) => {
                tokens = output.tokens;
                runner.scores = Some(output.scores);
                runner.state_checkpoints = output.state_checkpoints;
                tokens
                    .view()
                    .and_then(|tokens| {
                        validate_requested_logprobs(
                            tokens,
                            runner.scores.as_ref().and_then(BorrowedScores::logprobs),
                            &plan.borrow(py),
                        )?;
                        Ok(tokens)
                    })
                    .map_err(|error| runner.failed(error, true))
            }
            Err(error) => Err(runner.failed(error, true)),
        };
        let mut event_error = None;
        let make_event = |event: core::TokenEvent| {
            Py::new(
                py,
                TokenEvent {
                    request_id: event.request_id,
                    token_id: event.token_id,
                    index: event.index,
                    finished: event.finished,
                    finish_reason: event.finish_reason,
                    logprobs: event.logprobs,
                },
            )
            .map_err(|error| {
                event_error = Some(error);
                core::Error::memory("could not allocate token event")
            })
        };
        let result = if continuing {
            self.core
                .complete_continuing_step_with(&mut runner, output, make_event)
        } else {
            self.core
                .complete_step_with(&mut runner, output, make_event)
        };
        result.map_err(|error| {
            runner
                .error
                .or(event_error)
                .unwrap_or_else(|| core_error(error))
        })
    }

    fn fail_output(&mut self, py: Python<'_>, error: PyErr, accepted: bool) -> PyErr {
        let mut runner = PythonRunner::new(py, self.runner_release.as_ref());
        let accepted = accepted || runner_error_is_post_accept(py, &error);
        let failure = runner.failed(error, accepted);
        let result = self.core.complete_step_with(&mut runner, Err(failure), Ok);
        result.err();
        runner.error.expect("preserved runner exception")
    }

    fn harvest(&mut self, py: Python<'_>, continuing: bool) -> PyResult<Vec<Py<TokenEvent>>> {
        let pending = self.pending.take().expect("harvest pending execution");
        match pending.completion.bind(py).call_method0("result") {
            Ok(output) => match self.consume_output(py, &pending.plan, &output, continuing) {
                Ok(events) => Ok(events),
                Err(error) => {
                    self.failed_executions.push(pending.completion);
                    Err(error)
                }
            },
            Err(error) => {
                self.failed_executions.push(pending.completion);
                Err(self.fail_output(py, error, true))
            }
        }
    }
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (runner: "Runner", *, max_running_requests = 32, context_length = 32768, max_top_logprobs = DEFAULT_MAX_TOP_LOGPROBS, scheduler_config = None, prefix_cache = false, state_tracking_interval = 256, prefix_cache_checkpoint_interval = 0, prefix_cache_storage: "CacheStorage | None" = None, prefix_cache_max_inflight = None, prefix_cache_storage_policy: r#"Literal["write_through", "write_through_selective"]"# = "write_through_selective", enable_overlap = true, enable_metrics = true))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python constructor exposes keyword configuration"
    )]
    fn new(
        py: Python<'_>,
        runner: Py<PyAny>,
        max_running_requests: usize,
        context_length: u32,
        max_top_logprobs: u32,
        scheduler_config: Option<Py<SchedulerConfig>>,
        prefix_cache: bool,
        state_tracking_interval: u32,
        prefix_cache_checkpoint_interval: u32,
        prefix_cache_storage: Option<Py<PyAny>>,
        prefix_cache_max_inflight: Option<usize>,
        prefix_cache_storage_policy: &str,
        enable_overlap: bool,
        enable_metrics: bool,
    ) -> PyResult<Self> {
        let prefix_cache_max_inflight = NonZeroUsize::new(
            prefix_cache_max_inflight.unwrap_or_else(|| max_running_requests.max(2)),
        )
        .ok_or_else(|| PyValueError::new_err("prefix_cache_max_inflight must be positive"))?;
        let storage_policy = match prefix_cache_storage_policy {
            "write_through" => PrefixStoragePolicy::WriteThrough,
            "write_through_selective" => PrefixStoragePolicy::WriteThroughSelective,
            _ => {
                return Err(PyValueError::new_err(
                    "prefix_cache_storage_policy must be 'write_through' or 'write_through_selective'",
                ));
            }
        };
        let runner_enqueue = optional_runner_callable(py, runner.bind(py), "enqueue")?;
        let (runner_forward, runner_release) = resolve_runner_contract(py, runner.bind(py))?;
        let resolved = resolve_runner_config(py, runner.bind(py))?;
        let config = &resolved.config;
        let scheduler = scheduler_config
            .as_ref()
            .map(|config| *config.borrow(py))
            .unwrap_or_default();
        let mut core = core::EngineCore::new(
            core::RunnerConfig {
                max_batch_tokens: config.max_batch_tokens,
                prefill_chunk_size: config.prefill_chunk_size,
                continuation_prefill: config.continuation_prefill,
                max_accepted_per_step: config.max_accepted_per_step,
                cache_layout: config
                    .cache_layout
                    .as_ref()
                    .map(|layout| layout.get().inner_native().clone()),
            },
            max_running_requests,
            context_length,
            core::SchedulerConfig {
                max_batch_tokens: scheduler.max_batch_tokens,
                prefill_chunk_size: scheduler.prefill_chunk_size,
                schedule_conservativeness: scheduler.schedule_conservativeness,
            },
            prefix_cache,
        )
        .map_err(core_error)?;
        core.set_prefix_cache_policy(state_tracking_interval, prefix_cache_checkpoint_interval)
            .map_err(core_error)?;
        core.set_max_top_logprobs(max_top_logprobs);
        if enable_metrics {
            core.enable_metrics().map_err(core_error)?;
        }
        if let Some(configure) = optional_runner_callable(py, runner.bind(py), "configure_metrics")?
        {
            let kwargs = PyDict::new(py);
            kwargs.set_item("enabled", enable_metrics)?;
            configure.bind(py).call((), Some(&kwargs))?;
        }
        let runner_metrics = if enable_metrics {
            optional_runner_callable(py, runner.bind(py), "metrics_snapshot")?
        } else {
            None
        };
        if let Some(provider) = prefix_cache_storage {
            let adapter = PyModule::import(py, "resin._cache_storage")?
                .getattr("_PrefixStorage")?
                .call1((provider, runner, &config.cache_layout))?
                .unbind();
            core.set_prefix_storage(
                Box::new(cache_storage::PythonPrefixStorage(adapter)),
                prefix_cache_max_inflight,
                storage_policy,
            )
            .map_err(core_error)?;
        }
        let runner_config = bind_runner_resources(py, resolved, core.cache_pool_id())?;
        Ok(Self {
            core,
            runner_forward,
            runner_release,
            runner_config,
            runner_enqueue,
            overlap: enable_overlap,
            pending: None,
            deferred_events: Vec::new(),
            failed_executions: Vec::new(),
            overlap_enqueues: 0,
            overlap_continuations: 0,
            runner_metrics,
        })
    }
    #[pyo3(signature = (
        token_ids: "Sequence[int] | npt.NDArray[np.uint32]",
        *,
        priority = 0,
        batch_key = None,
        sampling_params = None,
        media: "tuple[Media, ...] | None" = None,
        prompt_target_logprobs = None,
        cache_salt = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn submit(
        &mut self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        priority: i32,
        batch_key: Option<u64>,
        sampling_params: Option<Py<SamplingParams>>,
        media: Option<Py<PyTuple>>,
        prompt_target_logprobs: Option<Vec<Vec<u32>>>,
        cache_salt: Option<String>,
    ) -> PyResult<Py<RequestLease>> {
        self.core.raise_if_open().map_err(core_error)?;
        if token_ids.is_empty() {
            return Err(PyValueError::new_err("token_ids must be non-empty"));
        }
        let media = media.filter(|items| !items.bind(py).is_empty());
        let (cache_prompt, atomic_prefill_spans) = match media.as_ref() {
            Some(items) => {
                let (identity, spans) = media_payload::admission(items.bind(py), &token_ids)?;
                (Some(identity), spans)
            }
            None => (None, Vec::new()),
        };
        let mut sampling_params = sampling_params
            .map(|params| params.borrow(py).clone())
            .unwrap_or_default();
        let max_top_logprobs = self.core.max_top_logprobs();
        for (name, count) in [
            ("top_logprobs", sampling_params.top_logprobs),
            ("prompt_top_logprobs", sampling_params.prompt_top_logprobs),
        ] {
            if count > max_top_logprobs {
                return Err(PyValueError::new_err(format!(
                    "{name} must be between 0 and {max_top_logprobs}"
                )));
            }
        }
        sampling_params.seed.get_or_insert_with(|| {
            default_sampling_seed(self.core.id(), self.core.next_request_id())
        });
        let input = core::Submission {
            prompt_logprobs: sampling_params.prompt_logprobs,
            prompt_top_logprobs: sampling_params.prompt_top_logprobs,
            prompt_target_logprobs,
            token_ids,
            max_output_tokens: sampling_params
                .max_output_tokens
                .map(|value| value as usize),
            priority,
            batch_key,
            stop_token_ids: sampling_params.stop_token_ids.to_vec(),
            cache_key: cache_prompt,
            cache_salt,
            atomic_prefill_spans,
            payload: RequestPayload {
                sampling_params,
                media: media.map(Arc::new),
            },
        };
        let mut handle_error = None;
        self.core
            .submit_with(input, |inner| {
                Py::new(py, RequestLease { inner }).map_err(|error| {
                    handle_error = Some(error);
                    core::Error::memory("could not allocate request handle")
                })
            })
            .map_err(|error| handle_error.unwrap_or_else(|| core_error(error)))
    }

    #[pyo3(signature = (*, phase = None, allow_overlap = true))]
    fn step(
        &mut self,
        py: Python<'_>,
        phase: Option<BatchPhase>,
        allow_overlap: bool,
    ) -> PyResult<Vec<Py<TokenEvent>>> {
        let mut current = self.step_once(py, phase, allow_overlap)?;
        if self.deferred_events.is_empty() {
            return Ok(current);
        }
        let mut events = std::mem::take(&mut self.deferred_events);
        events.append(&mut current);
        Ok(events)
    }

    /// Collect a read-only native snapshot of engine and runner observations.
    /// Call periodically: cache inventory requires a scan of allocated blocks.
    fn metrics_snapshot(&self, py: Python<'_>) -> PyResult<Option<metrics::EngineMetrics>> {
        self.collect_metrics(py, true)
    }

    /// Native serving export; avoids constructing Python inspector dictionaries.
    #[pyo3(signature = (*, model_name, rank = 0, attn_dp_rank = 0, include_runner = true))]
    fn _metrics_prometheus_protobuf<'py>(
        &self,
        py: Python<'py>,
        model_name: &str,
        rank: u32,
        attn_dp_rank: u32,
        include_runner: bool,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyBytes>>> {
        self.collect_metrics(py, include_runner)?
            .map(|snapshot| {
                Ok(pyo3::types::PyBytes::new(
                    py,
                    &snapshot.encode_protobuf(model_name, rank, attn_dp_rank)?,
                ))
            })
            .transpose()
    }

    #[getter]
    fn metrics_enabled(&self) -> bool {
        self.core.metrics_enabled()
    }

    #[getter]
    fn overlap(&self) -> bool {
        self.overlap
    }

    #[getter]
    fn overlap_pending(&self) -> bool {
        self.pending.is_some()
    }

    #[getter]
    fn overlap_ready(&self, py: Python<'_>) -> bool {
        let Some(pending) = &self.pending else {
            return true;
        };
        self.core.has_continuation()
            && pending
                .completion
                .bind(py)
                .getattr("feedback")
                // Only an explicit None is a normal drain. Attribute errors
                // must reach step_once's existing fail-closed path instead of
                // silently bypassing the completion contract.
                .map_or(true, |feedback| !feedback.is_none())
    }

    #[getter]
    fn overlap_stats(&self) -> (u64, u64) {
        (self.overlap_enqueues, self.overlap_continuations)
    }

    /// Poll storage completions before proposing the next runnable phase.
    fn next_phase(&mut self) -> PyResult<Option<BatchPhase>> {
        self.core.poll_storage(false).map_err(core_error)?;
        self.core
            .next_phase()
            .map(|phase| {
                phase.map(|phase| match phase {
                    resin_core::Phase::Prefill => BatchPhase::Prefill,
                    resin_core::Phase::Decode => BatchPhase::Decode,
                })
            })
            .map_err(core_error)
    }

    /// Contain a failure reported by a peer without releasing local resources.
    fn poison(&mut self, reason: &str) {
        self.core.poison_message(reason);
    }

    fn cancel(&mut self, py: Python<'_>, request: PyRef<'_, RequestLease>) -> PyResult<bool> {
        self.core.raise_if_open().map_err(core_error)?;
        if request.inner.engine_id() != self.core.id() {
            return Err(PyValueError::new_err(
                "RequestLease belongs to a different Engine",
            ));
        }
        if self.pending.is_some() {
            let events = self.harvest(py, false)?;
            self.deferred_events.extend(
                events
                    .into_iter()
                    .filter(|event| event.borrow(py).request_id != request.inner.id()),
            );
        }
        self.deferred_events
            .retain(|event| event.borrow(py).request_id != request.inner.id());
        let mut runner = PythonRunner::new(py, self.runner_release.as_ref());
        let result = self.core.cancel(&mut runner, &request.inner);
        runner.result(result)
    }
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.pending.is_some() {
            self.harvest(py, false)?;
        }
        self.deferred_events.clear();
        let mut runner = PythonRunner::new(py, self.runner_release.as_ref());
        let result = self.core.close(&mut runner);
        runner.result(result)
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        slf.core.raise_if_open().map_err(core_error)?;
        Ok(slf)
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }

    fn has_work(&self) -> bool {
        self.core.has_work() || !self.deferred_events.is_empty()
    }

    #[getter]
    fn active_request_count(&self) -> usize {
        self.core.active_request_count()
    }

    #[getter]
    fn ready_request_count(&self) -> usize {
        self.core.ready_request_count()
    }

    #[getter]
    fn max_running_requests(&self) -> usize {
        self.core.max_running_requests()
    }

    /// Maximum top alternatives per generated or prompt token.
    #[getter]
    fn max_top_logprobs(&self) -> u32 {
        self.core.max_top_logprobs()
    }

    #[getter]
    fn poisoned(&self) -> bool {
        self.core.poisoned()
    }

    #[getter]
    fn closed(&self) -> bool {
        self.core.closed()
    }

    #[getter]
    fn poison_error(&self) -> Option<&str> {
        self.core.poison_error()
    }

    #[getter]
    fn scheduler_config(&self, py: Python<'_>) -> PyResult<Py<SchedulerConfig>> {
        {
            let config = self.core.scheduler_config();
            Py::new(
                py,
                SchedulerConfig {
                    max_batch_tokens: config.max_batch_tokens,
                    prefill_chunk_size: config.prefill_chunk_size,
                    schedule_conservativeness: config.schedule_conservativeness,
                },
            )
        }
    }

    #[getter]
    fn new_token_ratio(&self) -> f64 {
        self.core.new_token_ratio()
    }

    #[getter]
    fn context_length(&self) -> u32 {
        self.core.context_length()
    }

    #[getter]
    fn cache_layout(&self, py: Python<'_>) -> Option<Py<CacheLayout>> {
        self.runner_config
            .cache_layout
            .as_ref()
            .map(|layout| layout.clone_ref(py))
    }

    #[getter]
    fn cache_pool_id(&self) -> Option<u64> {
        self.core.cache_pool_id()
    }

    #[getter]
    fn cache_parent_capacity(&self) -> Option<u32> {
        self.core.cache_parent_capacity()
    }

    #[getter]
    fn cache_free_parent_count(&self) -> Option<usize> {
        self.core.cache_free_parent_count()
    }

    #[getter]
    fn cache_occupied_block_count(&self) -> Option<usize> {
        self.core.cache_occupied_block_count()
    }

    /// `(entries, hits, misses, evictions)` when prefix caching is enabled.
    #[getter]
    fn prefix_cache_stats(&self) -> Option<(usize, u64, u64, u64)> {
        self.core.prefix_cache_stats()
    }

    /// Drop all reusable cache prefixes.
    fn clear_prefix_cache(&mut self) -> PyResult<()> {
        self.core.clear_prefix_cache().map_err(core_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "Engine(ready_requests={}, max_running_requests={}, context_length={}, closed={}, poisoned={}, cache_blocks={})",
            self.ready_request_count(),
            self.max_running_requests(),
            self.context_length(),
            self.core.closed(),
            self.poisoned(),
            self.core.cache_occupied_block_count().unwrap_or(0)
        )
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if self.core.closed() || self.core.poisoned() {
            return;
        }

        match Python::try_attach(|py| self.close(py)) {
            Some(Ok(())) => {}
            Some(Err(error)) => {
                self.core
                    .poison_message(format!("Engine destructor cleanup failed: {error}"));
            }
            None => self.core.poison_message(
                "Engine destructor could not attach to Python; resources remain quarantined",
            ),
        }
    }
}

fn runner_result_is_async(output: &Bound<'_, PyAny>) -> PyResult<bool> {
    if output.is_instance_of::<ForwardOutput>() {
        return Ok(false);
    }
    match output.getattr("result") {
        Ok(method) if method.is_callable() => Ok(true),
        Err(error) if !error.is_instance_of::<PyAttributeError>(output.py()) => Err(error),
        _ => Err(PyTypeError::new_err(
            "runner.enqueue() must return ForwardOutput or a completion with callable result()",
        )),
    }
}

fn extract_runner_output<'py>(output: &Bound<'py, PyAny>) -> PyResult<BorrowedOutput<'py>> {
    let output = output
        .cast::<ForwardOutput>()
        .map_err(|_| PyTypeError::new_err("runner must return ForwardOutput"))?;
    output.borrow().borrow(output.py())
}

fn validate_requested_logprobs(
    tokens: core::TokenRows<'_>,
    view: Option<core::LogprobView<'_>>,
    plan: &BatchPlan,
) -> PyResult<()> {
    if tokens.len() != plan.sequence_ids.len() {
        return Err(PyRuntimeError::new_err(format!(
            "runner returned {} token rows for {} sequences",
            tokens.len(),
            plan.sequence_ids.len()
        )));
    }
    for (row, payload) in plan.payloads.iter().enumerate() {
        let params = &payload.sampling_params;
        if !params.logprobs || !plan.sampling_mask[row] {
            continue;
        }
        let range = view.and_then(|scores| {
            Some(*scores.indptr.get(row)? as usize..*scores.indptr.get(row + 1)? as usize)
        });
        if !range.as_ref().is_some_and(|range| {
            !range.is_empty()
                && range.len() == tokens.row(row).len()
                && view.unwrap().values.get(range.clone()).is_some()
        }) {
            return Err(PyRuntimeError::new_err(
                "runner did not return requested token logprobs",
            ));
        }
        if params.top_logprobs > 0 {
            let top = view.and_then(|scores| scores.top);
            for index in range.unwrap() {
                let count = top.and_then(|top| {
                    let start = *top.indptr.get(index)?;
                    let end = *top.indptr.get(index + 1)?;
                    end.checked_sub(start)
                });
                if !count.is_some_and(|count| count > 0 && count <= params.top_logprobs) {
                    return Err(PyRuntimeError::new_err(
                        "runner did not return requested top logprobs",
                    ));
                }
            }
        }
    }
    Ok(())
}

#[pymodule(name = "_resin")]
mod _resin {
    #[pymodule_export]
    use super::{
        BatchPhase, BatchPlan, CacheGroupFamily, CacheGroupLayout, CacheGroupPlan, CacheLayout,
        CacheRetention, CacheStatePlan, Engine, ForwardOutput, Parallelism, RequestLease,
        ResourceBinding, RunnerConfig, SamplingParams, SchedulerConfig, TokenEvent, TokenLogprobs,
    };

    #[pymodule_export]
    use crate::metrics::{
        _runner_metrics_prometheus_protobuf, EngineMetrics, HistogramAccumulator, RunnerMetrics,
    };

    #[pymodule_export]
    use crate::http::NativeHttpServer;

    #[pymodule_export]
    use crate::http_distributed::{HttpEvents, HttpIngress};

    #[pymodule_export]
    use crate::http_session::NativeHttpSession;

    #[pymodule_export]
    use crate::numa::{_numa_get_policy, _numa_set_policy};

    #[pymodule_export]
    use crate::recipe::{
        NativeRecipe, NativeResponseParser, RecipeTokenizer, RenderedPrompt, get_native_recipe,
    };

    #[pymodule_export]
    use crate::recipe_events::{
        ParserEvent, ReasoningDelta, TextDelta, ToolCallArgumentsDelta, ToolCallStart,
    };

    #[pymodule_export]
    use crate::media_payload::{_media_from_buffers, _media_to_buffers, Media};

    #[pymodule_export]
    use crate::recipe_types::{
        Audio, Image, Message, RenderOptions, Role, Text, Thinking, Tool, ToolCall, ToolResult,
        Video,
    };

    #[pymodule_export]
    #[allow(non_upper_case_globals)]
    const __version__: &str = env!("CARGO_PKG_VERSION");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire_plan(group_index: u32) -> BatchPlan {
        let group = resin_core::resources::kv::KvGroupPlan::new(
            group_index,
            vec![0, 1],
            vec![7],
            vec![0],
            vec![0],
            vec![7, 7],
            vec![0, 0],
            vec![0, 1],
            NonZeroU32::new(1).unwrap(),
        )
        .unwrap();
        BatchPlan {
            batch: NativeBatchPlan {
                batch_id: 1,
                batch_key: None,
                phase: Phase::Prefill,
                sequence_ids: vec![1],
                context_lengths: vec![0],
                prefill_lengths: vec![2],
                token_ids: vec![1, 2],
                token_indptr: vec![0, 2],
                sampling_mask: vec![true],
                cache_groups: Arc::new([group]),
                prompt_score_indices: Vec::new(),
                prompt_score_token_ids: Vec::new(),
                prompt_score_indptr: vec![0, 0],
                prompt_score_top_counts: Vec::new(),
                prompt_target_indptr: vec![0],
                prompt_target_token_ids: Vec::new(),
            },
            payloads: Arc::new(vec![RequestPayload {
                sampling_params: SamplingParams {
                    seed: Some(1),
                    ..SamplingParams::default()
                },
                media: None,
            }]),
            positions: OnceLock::new(),
        }
    }

    #[test]
    fn batch_plan_rejects_noncanonical_cache_group_order() {
        assert!(BatchPlan::checked(wire_plan(1)).is_err());
    }

    #[test]
    fn plan_archive_skips_request_objects_and_derived_positions() {
        let plan = wire_plan(0);
        let bytes = plan.batch.encode_wire().unwrap();
        plan.positions.set(vec![0, 1]).unwrap();
        assert_eq!(bytes, plan.batch.encode_wire().unwrap());
        let restored = NativeBatchPlan::decode_wire(&bytes).unwrap();
        assert_eq!(restored.sequence_ids, plan.sequence_ids);
        assert_eq!(restored.token_ids, plan.token_ids);
        assert_eq!(restored.cache_groups, plan.cache_groups);
    }

    #[test]
    fn structurally_valid_plan_archives_still_require_semantic_validation() {
        Python::initialize();
        Python::attach(|py| {
            let requests = PyDict::new(py);
            let pending = PyDict::new(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "sampling_params",
                    Py::new(
                        py,
                        SamplingParams {
                            seed: Some(1),
                            ..SamplingParams::default()
                        },
                    )
                    .unwrap(),
                )
                .unwrap();
            kwargs.set_item("media", py.None()).unwrap();
            let request = py
                .import("types")
                .unwrap()
                .getattr("SimpleNamespace")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();
            pending.set_item(1, request).unwrap();
            let valid =
                NativeBatchPlan::decode_wire(&wire_plan(0).batch.encode_wire().unwrap()).unwrap();
            assert!(BatchPlan::restore_requests(valid, &requests, &pending).is_ok());

            for (case, expected) in [
                (0, "token_indptr must start at zero"),
                (1, "prefill_lengths must cover"),
                (2, "blocks_per_slab must be positive"),
                (3, "block_indptr must end"),
                (4, "contiguous group_index"),
            ] {
                let mut plan = wire_plan(0).batch;
                match case {
                    0 => plan.token_indptr[0] = 1,
                    1 => plan.prefill_lengths[0] = 1,
                    _ => {
                        let group = &mut Arc::make_mut(&mut plan.cache_groups)[0];
                        match case {
                            2 => group.blocks_per_slab = 0,
                            3 => Arc::make_mut(&mut group.blocks).block_indptr[1] = 0,
                            _ => group.group_index = 1,
                        }
                    }
                }
                // These bytes pass rkyv's memory/representation checks. Only
                // BatchPlan::checked can reject their inconsistent semantics.
                let decoded = NativeBatchPlan::decode_wire(&plan.encode_wire().unwrap()).unwrap();
                let error = BatchPlan::restore_requests(decoded, &requests, &pending).unwrap_err();
                assert!(error.to_string().contains(expected), "{error}");
            }
        });
    }
}
