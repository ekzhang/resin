#![expect(
    clippy::clone_on_copy,
    reason = "PyO3 0.29 from_py_object emits Clone::clone for Copy enums"
)]

use std::num::NonZeroU32;
use std::sync::{Arc, OnceLock};

use numpy::{PyArray1, PyArrayLike1, PyReadonlyArray1};
use pyo3::exceptions::{
    PyAttributeError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

mod arrays;
mod cache;
mod http;
mod http_distributed;
mod http_logprobs;
mod http_output;
mod http_session;
mod http_stop;
mod media_payload;
mod media_preprocess;
mod metrics;
mod output;
mod parallelism;
mod planning;
mod recipe;
mod recipe_events;
mod recipe_media;
mod recipe_types;
mod runner;

use cache::{CacheGroupFamily, CacheGroupLayout, CacheGroupPlan, CacheLayout, CacheRetention};
use output::{BorrowedLogprobs, BorrowedOutput, ForwardOutput};
use parallelism::Parallelism;
use resin_core::engine::{self as core, CompletionReason};
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

    #[getter]
    fn schedule_conservativeness(&self) -> f64 {
        self.schedule_conservativeness
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
#[derive(
    Clone, Copy, Debug, Eq, Hash, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
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
}

/// Per-request generation policy, including sampling, length, and stop tokens.
///
/// A zero temperature selects the maximum-logit token. Positive temperatures
/// enable stochastic sampling after optional top-k and top-p filtering.
/// An omitted `max_output_tokens` uses the remaining context budget. Stop tokens
/// are explicit: pass a rendered prompt's `stop_token_ids` for recipe stopping.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug)]
struct SamplingParams {
    max_output_tokens: Option<u32>,
    stop_token_ids: Arc<[u32]>,
    temperature: f64,
    top_p: f64,
    top_k: Option<u32>,
    seed: Option<i64>,
    logprobs: bool,
    top_logprobs: usize,
    prompt_logprobs: bool,
    prompt_top_logprobs: usize,
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
        top_logprobs: usize,
        prompt_logprobs: bool,
        prompt_top_logprobs: usize,
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
        if top_logprobs > 1000 {
            return Err(PyValueError::new_err(
                "top_logprobs must be between 0 and 1000",
            ));
        }
        if prompt_top_logprobs > 1000 {
            return Err(PyValueError::new_err(
                "prompt_top_logprobs must be between 0 and 1000",
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
    fn max_output_tokens(&self) -> Option<u32> {
        self.max_output_tokens
    }

    #[getter]
    fn stop_token_ids(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.stop_token_ids.iter().copied())?.unbind())
    }

    #[getter]
    fn temperature(&self) -> f64 {
        self.temperature
    }

    #[getter]
    fn top_p(&self) -> f64 {
        self.top_p
    }

    #[getter]
    fn top_k(&self) -> Option<u32> {
        self.top_k
    }

    #[getter]
    fn seed(&self) -> Option<i64> {
        self.seed
    }

    #[getter]
    fn logprobs(&self) -> bool {
        self.logprobs
    }
    #[getter]
    fn prompt_logprobs(&self) -> bool {
        self.prompt_logprobs
    }
    #[getter]
    fn prompt_top_logprobs(&self) -> usize {
        self.prompt_top_logprobs
    }

    #[getter]
    fn top_logprobs(&self) -> usize {
        self.top_logprobs
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
/// `context_lengths[i]` is the number of tokens processed before this step,
/// so that sequence's new tokens occupy consecutive absolute positions
/// beginning there. `cache_groups` contains one placement plan per declared
/// `CacheLayout` group, and every group plan covers every sequence.
/// `prefill_lengths` records the final endpoint of each prefill history,
/// including generated inputs being reconstructed after retraction. It stays
/// fixed across those prefill chunks and is only meaningful during prefill.
/// Public constructors that omit it infer each endpoint from its scheduled end.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct BatchPlan {
    prompt_score_indices: Vec<u32>,
    prompt_score_token_ids: Vec<u32>,
    prompt_score_indptr: Vec<u32>,
    prompt_score_top_counts: Vec<u32>,
    prompt_target_indptr: Vec<u32>,
    prompt_target_token_ids: Vec<u32>,
    sequence_ids: Vec<u64>,
    context_lengths: Vec<u32>,
    prefill_lengths: Vec<u32>,
    token_ids: Vec<u32>,
    token_indptr: Vec<u32>,
    phase: BatchPhase,
    batch_id: u64,
    batch_key: Option<u64>,
    cache_groups: Vec<CacheGroupPlan>,
    sampling_mask: Vec<bool>,
    #[rkyv(with = rkyv::with::Skip)]
    sampling_params: Vec<SamplingParams>,
    #[rkyv(with = rkyv::with::Skip)]
    media: Vec<Option<Arc<Py<PyAny>>>>,
    #[rkyv(with = rkyv::with::Skip)]
    positions: OnceLock<Vec<u32>>,
}

const PLAN_WIRE_MAGIC: &[u8; 8] = b"RSNPLAN\0";
// Private same-build protocol: bump when the archived BatchPlan, BatchPhase,
// CacheGroupPlan or KvGroupPlan schema, or rkyv format flags, change.
const PLAN_WIRE_VERSION: u32 = 1;
const PLAN_WIRE_HEADER: usize = 20;

impl BatchPlan {
    fn encode_wire(&self) -> PyResult<Vec<u8>> {
        // Serialize directly into the eventual NumPy owner. The envelope is
        // separate from the derived schema and rejects version/length mistakes.
        let mut bytes =
            rkyv::api::high::to_bytes_in::<_, rkyv::rancor::Error>(self, vec![0; PLAN_WIRE_HEADER])
                .map_err(|error| PyValueError::new_err(format!("cannot encode plan: {error}")))?;
        let length = (bytes.len() - PLAN_WIRE_HEADER) as u64;
        bytes[..8].copy_from_slice(PLAN_WIRE_MAGIC);
        bytes[8..12].copy_from_slice(&PLAN_WIRE_VERSION.to_le_bytes());
        bytes[12..PLAN_WIRE_HEADER].copy_from_slice(&length.to_le_bytes());
        Ok(bytes)
    }

    fn decode_wire(bytes: &[u8]) -> PyResult<Self> {
        if bytes.len() < PLAN_WIRE_HEADER || &bytes[..8] != PLAN_WIRE_MAGIC {
            return Err(PyValueError::new_err("invalid plan wire format"));
        }
        if u32::from_le_bytes(bytes[8..12].try_into().unwrap()) != PLAN_WIRE_VERSION {
            return Err(PyValueError::new_err("unsupported plan wire version"));
        }
        let length = u64::from_le_bytes(bytes[12..PLAN_WIRE_HEADER].try_into().unwrap());
        if length != (bytes.len() - PLAN_WIRE_HEADER) as u64 {
            return Err(PyValueError::new_err("plan wire length mismatch"));
        }
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(&bytes[PLAN_WIRE_HEADER..])
            .map_err(|error| PyValueError::new_err(format!("invalid plan archive: {error}")))
    }

    fn restore_requests(
        mut self,
        requests: &Bound<'_, PyDict>,
        pending: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        self.sampling_params.reserve(self.sequence_ids.len());
        self.media.reserve(self.sequence_ids.len());
        for &id in &self.sequence_ids {
            let request = match pending.get_item(id)? {
                Some(request) => request,
                None => requests.get_item(id)?.ok_or_else(|| {
                    PyValueError::new_err(format!("request {id} has not been registered"))
                })?,
            };
            self.sampling_params.push(
                request
                    .getattr("sampling_params")?
                    .extract::<PyRef<'_, SamplingParams>>()?
                    .clone(),
            );
            let media = request.getattr("media")?;
            self.media.push(if media.is_none() {
                None
            } else {
                Some(Arc::new(media.unbind()))
            });
        }
        Self::checked(self)
    }

    fn checked(plan: Self) -> PyResult<Self> {
        let Self {
            token_ids,
            sequence_ids,
            token_indptr,
            context_lengths,
            prefill_lengths,
            sampling_params,
            sampling_mask,
            media,
            ..
        } = &plan;
        if token_ids.is_empty() {
            return Err(PyValueError::new_err(
                "a batch must contain at least one token",
            ));
        }
        if token_ids.len() > u32::MAX as usize {
            return Err(PyValueError::new_err(
                "batch token count exceeds the u32 metadata range",
            ));
        }
        if sequence_ids.is_empty() {
            return Err(PyValueError::new_err(
                "a batch must contain at least one sequence",
            ));
        }
        if token_indptr.len() != sequence_ids.len() + 1 {
            return Err(PyValueError::new_err(
                "token_indptr must contain one more item than sequence_ids",
            ));
        }
        if context_lengths.len() != sequence_ids.len() {
            return Err(PyValueError::new_err(
                "context_lengths must contain one item per sequence",
            ));
        }
        if prefill_lengths.len() != sequence_ids.len() {
            return Err(PyValueError::new_err(
                "prefill_lengths must contain one item per sequence",
            ));
        }
        if sampling_params.len() != sequence_ids.len() {
            return Err(PyValueError::new_err(
                "sampling_params must contain one item per sequence",
            ));
        }
        if sampling_params.iter().any(|params| params.seed.is_none()) {
            return Err(PyValueError::new_err(
                "sampling_params must contain an effective seed for every sequence",
            ));
        }
        if sampling_mask.len() != sequence_ids.len() {
            return Err(PyValueError::new_err(
                "sampling_mask must contain one item per sequence",
            ));
        }
        if media.len() != sequence_ids.len() {
            return Err(PyValueError::new_err(
                "media must contain one item per sequence",
            ));
        }
        if token_indptr.first() != Some(&0) {
            return Err(PyValueError::new_err("token_indptr must start at zero"));
        }
        if token_indptr.last().copied() != Some(token_ids.len() as u32) {
            return Err(PyValueError::new_err(
                "token_indptr must end at the token count",
            ));
        }
        if token_indptr.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PyValueError::new_err(
                "each sequence must contain at least one token",
            ));
        }
        if context_lengths
            .iter()
            .zip(token_indptr.windows(2))
            .any(|(context, range)| context.checked_add(range[1] - range[0]).is_none())
        {
            return Err(PyValueError::new_err(
                "scheduled positions exceed the u32 metadata range",
            ));
        }
        if plan.phase == BatchPhase::Prefill
            && context_lengths
                .iter()
                .zip(token_indptr.windows(2))
                .zip(prefill_lengths)
                .any(|((context, range), length)| *context + (range[1] - range[0]) > *length)
        {
            return Err(PyValueError::new_err(
                "prefill_lengths must cover every scheduled prefill row",
            ));
        }
        let mut sorted_ids = sequence_ids.clone();
        sorted_ids.sort_unstable();
        if sorted_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(PyValueError::new_err(
                "sequence ids must be unique within a batch",
            ));
        }

        plan.validate_cache()?;
        resin_core::logprobs::validate_indptr(
            &plan.prompt_score_indptr,
            sequence_ids.len(),
            plan.prompt_score_indices.len(),
        )
        .map_err(core_error)?;
        if !plan.prompt_score_indices.is_empty()
            || !plan.prompt_target_indptr.is_empty()
            || !plan.prompt_target_token_ids.is_empty()
        {
            resin_core::logprobs::validate_indptr(
                &plan.prompt_target_indptr,
                plan.prompt_score_indices.len(),
                plan.prompt_target_token_ids.len(),
            )
            .map_err(core_error)?;
        }
        if plan.prompt_score_token_ids.len() != plan.prompt_score_indices.len()
            || plan.prompt_score_top_counts.len() != plan.prompt_score_indices.len()
            || plan
                .prompt_score_top_counts
                .iter()
                .any(|&count| count > 1000)
            || plan
                .prompt_score_indptr
                .windows(2)
                .enumerate()
                .any(|(row, range)| {
                    plan.prompt_score_indices[range[0] as usize..range[1] as usize]
                        .iter()
                        .any(|&index| index < token_indptr[row] || index >= token_indptr[row + 1])
                })
            || plan.phase != BatchPhase::Prefill && !plan.prompt_score_indices.is_empty()
        {
            return Err(PyValueError::new_err(
                "prompt score indices and targets must align with their prefill sequence",
            ));
        }

        Ok(plan)
    }

    fn validate_cache(&self) -> PyResult<()> {
        if self.cache_groups.is_empty() {
            return Ok(());
        }

        for (group_index, group) in self.cache_groups.iter().enumerate() {
            group.validate_local()?;
            let group = group.inner_native();
            if group.group_index != group_index as u32 {
                return Err(PyValueError::new_err(
                    "cache_groups must be ordered by contiguous group_index",
                ));
            }
            if group.block_table_start_indices.len() != self.sequence_ids.len() {
                return Err(PyValueError::new_err(
                    "each cache group must contain one block-table segment per sequence",
                ));
            }
            if group.append_slab_ids.len() != self.token_ids.len()
                || group.append_slotnums.len() != self.token_ids.len()
            {
                return Err(PyValueError::new_err(
                    "each cache group must contain one append target per batch token",
                ));
            }
            for sequence_index in 0..self.sequence_ids.len() {
                let block_start = group.block_indptr[sequence_index] as usize;
                let block_end = group.block_indptr[sequence_index + 1] as usize;
                let row_slabs = &group.block_slab_ids[block_start..block_end];
                let row_slotnums = &group.block_slotnums[block_start..block_end];
                let token_start = self.token_indptr[sequence_index] as usize;
                let token_end = self.token_indptr[sequence_index + 1] as usize;
                if group.append_slab_ids[token_start..token_end]
                    .iter()
                    .zip(&group.append_slotnums[token_start..token_end])
                    .any(|(append_slab, append_slotnum)| {
                        !row_slabs
                            .iter()
                            .zip(row_slotnums)
                            .any(|(slab, slotnum)| slab == append_slab && slotnum == append_slotnum)
                    })
                {
                    return Err(PyValueError::new_err(
                        "cache append targets must belong to their sequence's block table",
                    ));
                }
            }
        }
        Ok(())
    }
}
#[pymethods]
impl BatchPlan {
    fn _to_wire<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        Ok(PyArray1::from_vec(py, self.encode_wire()?))
    }

    #[staticmethod]
    fn _from_wire(
        data: PyReadonlyArray1<'_, u8>,
        requests: &Bound<'_, PyDict>,
        pending: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let bytes = data.as_slice().map_err(|_| {
            PyValueError::new_err("plan wire data must be a contiguous uint8 array")
        })?;
        Self::decode_wire(bytes)?.restore_requests(requests, pending)
    }

    #[new]
    #[pyo3(signature = (
        sequence_ids,
        context_lengths,
        token_ids,
        token_indptr,
        *,
        prefill_lengths = None,
        phase = BatchPhase::Prefill,
        batch_id = 0,
        batch_key = None,
        cache_groups = None,
        sampling_mask = None,
        sampling_params = None,
        media = None,
        prompt_score_indices = None,
        prompt_score_token_ids = None,
        prompt_score_indptr = None,
        prompt_score_top_counts = None,
        prompt_target_indptr = None,
        prompt_target_token_ids = None
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
        media: Option<Vec<Option<Py<PyAny>>>>,
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
            .map(|group| group.borrow(py).clone())
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
        let media = media
            .unwrap_or_else(|| (0..sequence_count).map(|_| None).collect())
            .into_iter()
            .map(|payload| payload.map(Arc::new))
            .collect();
        let prompt_score_indices = prompt_score_indices
            .map(arrays::owned_vec)
            .unwrap_or_default();
        let predecessor_count = prompt_score_indices.len();
        Self::checked(Self {
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
            sequence_ids,
            context_lengths,
            prefill_lengths,
            token_ids,
            token_indptr,
            phase,
            batch_id,
            batch_key,
            cache_groups,
            sampling_mask,
            sampling_params,
            media,
            positions: OnceLock::new(),
        })
    }

    /// Stable request identifier for each sequence.
    #[getter]
    fn sequence_ids<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u64>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().sequence_ids)
    }

    /// Number of tokens already processed for each sequence.
    #[getter]
    fn context_lengths<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().context_lengths)
    }

    /// Final history endpoint for each prefill, including retracted generated inputs.
    /// Only meaningful during prefill; constant across that history's chunks.
    #[getter]
    fn prefill_lengths<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().prefill_lengths)
    }

    /// Flattened scheduled token IDs, grouped by sequence.
    #[getter]
    fn token_ids<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().token_ids)
    }

    /// Offsets into `token_ids` for each sequence.
    #[getter]
    fn token_indptr<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().token_indptr)
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
                positions.extend(*context..*context + range[1] - range[0]);
            }
            positions
        });
        arrays::readonly_view(this.clone().into_any(), positions)
    }

    /// Whether this is a prefill or decode batch. Mixed-chunk is not supported.
    #[getter]
    fn phase(&self) -> BatchPhase {
        self.phase
    }

    /// Monotonic identifier for tracing and correlating runner steps.
    #[getter]
    fn batch_id(&self) -> u64 {
        self.batch_id
    }

    /// Optional key that prevents incompatible requests from sharing a batch.
    #[getter]
    fn batch_key(&self) -> Option<u64> {
        self.batch_key
    }

    /// Ordered physical placement plans corresponding to CacheLayout groups.
    /// There is one plan per layout group, not one plan per sequence; each plan
    /// contains one block-table row for every sequence in this batch.
    #[getter]
    fn cache_groups(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let groups = self
            .cache_groups
            .iter()
            .cloned()
            .map(|group| Py::new(py, group))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, groups)?.unbind())
    }

    /// Which sequences should generate a sampled token. Used for chunked
    /// prefill, when some sequences only advance the cache.
    #[getter]
    fn sampling_mask<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<bool>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().sampling_mask)
    }

    /// Check natively so token-only graph replay does not box per-row policies.
    #[getter]
    fn needs_logprobs(&self) -> bool {
        self.sampling_mask
            .iter()
            .zip(&self.sampling_params)
            .any(|(sample, params)| *sample && params.logprobs)
    }
    #[getter]
    fn needs_prompt_logprobs(&self) -> bool {
        !self.prompt_score_indices.is_empty()
    }
    #[getter]
    fn prompt_score_indices<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().prompt_score_indices)
    }
    #[getter]
    fn prompt_score_token_ids<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().prompt_score_token_ids)
    }
    #[getter]
    fn prompt_score_indptr<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().prompt_score_indptr)
    }
    #[getter]
    fn prompt_score_top_counts<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().prompt_score_top_counts)
    }
    #[getter]
    fn prompt_target_indptr<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().prompt_target_indptr)
    }
    #[getter]
    fn prompt_target_token_ids<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        arrays::readonly_view(this.clone().into_any(), &this.get().prompt_target_token_ids)
    }

    /// Effective sampling parameters, one per sequence.
    #[getter]
    fn sampling_params(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let params = self
            .sampling_params
            .iter()
            .cloned()
            .map(|params| Py::new(py, params))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, params)?.unbind())
    }

    /// Opaque runner inputs, one per sequence. The scheduler never inspects them.
    #[getter]
    fn media(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let payloads = self
            .media
            .iter()
            .map(|payload| {
                payload
                    .as_ref()
                    .map_or_else(|| py.None(), |payload| payload.clone_ref(py))
            })
            .collect::<Vec<_>>();
        Ok(PyTuple::new(py, payloads)?.unbind())
    }

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
            self.phase.as_str(),
            self.sequence_ids.len(),
            self.token_ids.len(),
            self.cache_groups.len()
        )
    }
}

#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
#[derive(Clone, Debug)]
struct TokenEvent {
    request_id: u64,
    token_id: u32,
    index: usize,
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
    fn request_id(&self) -> u64 {
        self.request_id
    }

    #[getter]
    fn token_id(&self) -> u32 {
        self.token_id
    }

    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    #[getter]
    fn finished(&self) -> bool {
        self.finished
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
    media: Option<Arc<Py<PyAny>>>,
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
    forward: &'a Py<PyAny>,
    release: Option<&'a Py<PyAny>>,
    error: Option<PyErr>,
    logprobs: Option<BorrowedLogprobs<'py>>,
    prompt_logprobs: Option<BorrowedLogprobs<'py>>,
}
impl PythonRunner<'_, '_> {
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
        plan: core::ExecutionPlan<RequestPayload>,
    ) -> core::Result<core::OwnedTokenRows> {
        let plan = Py::new(self.py, planning::from_core(plan))
            .map_err(|error| self.failed(error, false))?;
        let output = self
            .forward
            .bind(self.py)
            .call1((&plan,))
            .map_err(|error| {
                let accepted = runner_error_is_post_accept(self.py, &error);
                self.failed(error, accepted)
            })?;
        let plan = plan.borrow(self.py);
        let output = extract_runner_output(&output).map_err(|error| self.failed(error, true))?;
        let tokens = output
            .tokens
            .view()
            .map_err(|error| self.failed(error, true))?;
        validate_requested_logprobs(tokens, output.logprobs.as_ref(), &plan)
            .map_err(|error| self.failed(error, true))?;
        let tokens = tokens.to_owned();
        self.logprobs = output.logprobs;
        self.prompt_logprobs = output.prompt_logprobs;
        Ok(tokens)
    }
    fn logprobs(&self) -> Option<core::LogprobView<'_>> {
        self.logprobs.as_ref().map(BorrowedLogprobs::view)
    }
    fn prompt_logprobs(&self) -> Option<core::LogprobView<'_>> {
        self.prompt_logprobs.as_ref().map(BorrowedLogprobs::view)
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
        let mut events = Vec::new();
        if self.pending.is_some() {
            let next = if self.overlap
                && allow_overlap
                && phase.is_none_or(|phase| matches!(phase, BatchPhase::Decode))
            {
                self.core.continuation_plan().map_err(core_error)?
            } else {
                None
            };
            if let Some(next) = next {
                let next_plan = Py::new(py, planning::from_core(next))?;
                let previous = self.pending.as_ref().expect("pending execution");
                let feedback = match previous.completion.bind(py).getattr("feedback") {
                    Ok(feedback) => feedback,
                    Err(error) => {
                        self.core.poison_message(format!(
                            "accepted completion feedback failed: {error}"
                        ));
                        return Err(error);
                    }
                };
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("feedback", feedback)?;
                let enqueue = self
                    .runner_enqueue
                    .as_ref()
                    .expect("pending runner can enqueue");
                let result = match enqueue.bind(py).call((&next_plan,), Some(&kwargs)) {
                    Ok(result) => result.unbind(),
                    Err(error) => {
                        self.core
                            .poison_message(format!("overlapped enqueue failed: {error}"));
                        return Err(error);
                    }
                };
                let asynchronous = match runner_result_is_async(result.bind(py)) {
                    Ok(asynchronous) => asynchronous,
                    Err(error) => {
                        self.failed_executions.push(result);
                        self.core.poison_message(format!(
                            "accepted completion inspection failed: {error}"
                        ));
                        return Err(error);
                    }
                };
                if !asynchronous {
                    // Distributed peers may collectively choose a synchronous
                    // fallback. It still executed the future step, so harvest
                    // and commit both results without releasing pages between.
                    let previous_events = match self.harvest(py, true) {
                        Ok(events) => events,
                        Err(error) => {
                            self.failed_executions.push(result);
                            return Err(error);
                        }
                    };
                    events.extend(previous_events);
                    events.extend(self.consume_output(py, &next_plan, result.bind(py), false)?);
                    return Ok(events);
                }
                let previous_events = match self.harvest(py, true) {
                    Ok(events) => events,
                    Err(error) => {
                        self.failed_executions.push(result);
                        return Err(error);
                    }
                };
                self.pending = Some(PendingExecution {
                    plan: next_plan,
                    completion: result,
                });
                self.overlap_enqueues += 1;
                self.overlap_continuations += 1;
                events.extend(previous_events);
            } else {
                events.extend(self.harvest(py, false)?);
            }
            return Ok(events);
        }

        let forward = self.runner_forward.clone_ref(py);
        let release = self
            .runner_release
            .as_ref()
            .map(|value| value.clone_ref(py));
        let mut runner = PythonRunner {
            py,
            forward: &forward,
            release: release.as_ref(),
            error: None,
            logprobs: None,
            prompt_logprobs: None,
        };
        let Some(plan) = self
            .core
            .prepare_step(&mut runner, phase.map(BatchPhase::core))
            .map_err(|error| runner.error.take().unwrap_or_else(|| core_error(error)))?
        else {
            return Ok(events);
        };
        let can_enqueue = self.overlap
            && allow_overlap
            && self.runner_enqueue.is_some()
            && plan.phase == resin_core::Phase::Decode
            && plan
                .payloads
                .iter()
                .all(|payload| !payload.sampling_params.logprobs)
            && self.core.retain_continuation(&plan);
        let plan = match Py::new(py, planning::from_core(plan)) {
            Ok(plan) => plan,
            Err(error) => return Err(self.fail_output(py, error, false)),
        };
        if can_enqueue {
            let enqueue = self.runner_enqueue.as_ref().expect("checked enqueue");
            match enqueue.bind(py).call1((&plan,)) {
                Ok(output) => {
                    let asynchronous = match runner_result_is_async(&output) {
                        Ok(asynchronous) => asynchronous,
                        Err(error) => {
                            self.failed_executions.push(output.unbind());
                            return Err(self.fail_output(py, error, true));
                        }
                    };
                    if asynchronous {
                        self.pending = Some(PendingExecution {
                            plan,
                            completion: output.unbind(),
                        });
                        self.overlap_enqueues += 1;
                        return Ok(events);
                    }
                    match self.consume_output(py, &plan, &output, false) {
                        Ok(completed) => events.extend(completed),
                        Err(error) => {
                            self.failed_executions.push(output.unbind());
                            return Err(error);
                        }
                    }
                }
                Err(error) => return Err(self.fail_output(py, error, false)),
            }
        } else {
            match self.runner_forward.bind(py).call1((&plan,)) {
                Ok(output) => events.extend(self.consume_output(py, &plan, &output, false)?),
                Err(error) => return Err(self.fail_output(py, error, false)),
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
        let forward = self.runner_forward.clone_ref(py);
        let release = self
            .runner_release
            .as_ref()
            .map(|value| value.clone_ref(py));
        let mut runner = PythonRunner {
            py,
            forward: &forward,
            release: release.as_ref(),
            error: None,
            logprobs: None,
            prompt_logprobs: None,
        };
        let tokens;
        let output = match extract_runner_output(output) {
            Ok(output) => {
                tokens = output.tokens;
                runner.logprobs = output.logprobs;
                runner.prompt_logprobs = output.prompt_logprobs;
                tokens
                    .view()
                    .and_then(|tokens| {
                        validate_requested_logprobs(
                            tokens,
                            runner.logprobs.as_ref(),
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
        let forward = self.runner_forward.clone_ref(py);
        let release = self
            .runner_release
            .as_ref()
            .map(|value| value.clone_ref(py));
        let mut runner = PythonRunner {
            py,
            forward: &forward,
            release: release.as_ref(),
            error: None,
            logprobs: None,
            prompt_logprobs: None,
        };
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
    #[pyo3(signature = (runner, *, max_running_requests = 32, context_length = 32768, scheduler_config = None, prefix_cache = false, enable_overlap = true, enable_metrics = true))]
    #[allow(
        clippy::too_many_arguments,
        reason = "Python constructor exposes keyword configuration"
    )]
    fn new(
        py: Python<'_>,
        runner: Py<PyAny>,
        max_running_requests: usize,
        context_length: u32,
        scheduler_config: Option<Py<SchedulerConfig>>,
        prefix_cache: bool,
        enable_overlap: bool,
        enable_metrics: bool,
    ) -> PyResult<Self> {
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
                cache_write_lookahead_tokens: config.cache_write_lookahead_tokens,
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
        token_ids = None,
        *,
        priority = 0,
        batch_key = None,
        sampling_params = None,
        media = None,
        prompt_target_logprobs = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn submit(
        &mut self,
        py: Python<'_>,
        token_ids: Option<Vec<u32>>,
        priority: i32,
        batch_key: Option<u64>,
        sampling_params: Option<Py<SamplingParams>>,
        media: Option<Py<PyAny>>,
        prompt_target_logprobs: Option<Vec<Vec<u32>>>,
    ) -> PyResult<Py<RequestLease>> {
        self.core.raise_if_open().map_err(core_error)?;
        let token_ids = match token_ids {
            Some(token_ids) => token_ids,
            None => media
                .as_ref()
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "token_ids may be omitted only when media supplies token_ids",
                    )
                })?
                .getattr(py, "token_ids")?
                .extract(py)?,
        };
        if token_ids.is_empty() {
            return Err(PyValueError::new_err("token_ids must be non-empty"));
        }
        let cache_prompt: Option<Vec<u32>> = match media.as_ref() {
            Some(media) => Some(media.getattr(py, "cache_key")?.extract::<Vec<u32>>(py)?),
            None => None,
        };
        if cache_prompt
            .as_ref()
            .is_some_and(|identity| identity.len() != token_ids.len())
        {
            return Err(PyValueError::new_err(
                "media cache_key must match token_ids length",
            ));
        }
        let atomic_prefill_spans = match media.as_ref() {
            Some(media) => media
                .getattr(py, "token_spans")?
                .extract::<Vec<(u32, u32)>>(py)?
                .into_iter()
                .map(|(position, length)| (position, position.saturating_add(length)))
                .collect(),
            None => Vec::new(),
        };
        let mut sampling_params = sampling_params
            .map(|params| params.borrow(py).clone())
            .unwrap_or_default();
        sampling_params.seed.get_or_insert_with(|| {
            default_sampling_seed(self.core.id(), self.core.next_request_id())
        });
        let input = core::Submission {
            prompt_logprobs: sampling_params.prompt_logprobs,
            prompt_top_logprobs: sampling_params.prompt_top_logprobs as u32,
            prompt_target_logprobs,
            token_ids,
            max_output_tokens: sampling_params
                .max_output_tokens
                .map(|value| value as usize),
            priority,
            batch_key,
            stop_token_ids: sampling_params.stop_token_ids.to_vec(),
            cache_key: cache_prompt,
            cacheable: true,
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
    fn overlap_ready(&self) -> bool {
        !self.pending.is_some() || self.core.has_continuation()
    }

    #[getter]
    fn overlap_stats(&self) -> (u64, u64) {
        (self.overlap_enqueues, self.overlap_continuations)
    }

    /// Read-only admission-aware proposal; reserves no requests or resources.
    fn next_phase(&self) -> PyResult<Option<BatchPhase>> {
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
        let mut runner = PythonRunner {
            py,
            forward: &self.runner_forward,
            release: self.runner_release.as_ref(),
            error: None,
            logprobs: None,
            prompt_logprobs: None,
        };
        let result = self.core.cancel(&mut runner, &request.inner);
        runner.result(result)
    }
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.pending.is_some() {
            self.harvest(py, false)?;
        }
        self.deferred_events.clear();
        let mut runner = PythonRunner {
            py,
            forward: &self.runner_forward,
            release: self.runner_release.as_ref(),
            error: None,
            logprobs: None,
            prompt_logprobs: None,
        };
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
    logprobs: Option<&BorrowedLogprobs<'_>>,
    plan: &BatchPlan,
) -> PyResult<()> {
    if tokens.len() != plan.sequence_ids.len() {
        return Err(PyRuntimeError::new_err(format!(
            "runner returned {} token rows for {} sequences",
            tokens.len(),
            plan.sequence_ids.len()
        )));
    }
    let view = logprobs.map(BorrowedLogprobs::view);
    for (row, params) in plan.sampling_params.iter().enumerate() {
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
                    end.checked_sub(start).map(|count| count as usize)
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
        CacheRetention, Engine, ForwardOutput, Parallelism, RequestLease, ResourceBinding,
        RunnerConfig, SamplingParams, SchedulerConfig, TokenEvent, TokenLogprobs,
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
    use crate::recipe::{
        DeepSeekV4Recipe, DeepSeekV41Recipe, Gemma4Recipe, GptOssRecipe, InklingRecipe, Lfm2Recipe,
        NativeInklingParser, NativeRecipe, NativeResponseParser, Qwen35Recipe, Qwen38Recipe,
        RecipeTokenizer, RenderedPrompt,
    };

    #[pymodule_export]
    use crate::recipe_events::{
        ParserEvent, ReasoningDelta, TextDelta, ToolCallArgumentsDelta, ToolCallStart,
    };

    #[pymodule_export]
    use crate::media_payload::{
        _media_from_buffers, _media_to_buffers, DeepSeekV41Image, DeepSeekV41Media, Gemma4Media,
        InklingMedia, Qwen35Media,
    };

    #[pymodule_export]
    use crate::media_preprocess::image_token_types;

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

    #[test]
    fn runner_results_require_native_output_or_callable_completion() {
        Python::initialize();
        Python::attach(|py| {
            for expression in [c"[1]", c"[[1]]", c"None", c"object()"] {
                let value = py.eval(expression, None, None).unwrap();
                let error = extract_runner_output(&value)
                    .err()
                    .expect("non-native output must be rejected");
                assert!(error.is_instance_of::<PyTypeError>(py));
                assert!(
                    error
                        .to_string()
                        .contains("runner must return ForwardOutput")
                );
                let error = runner_result_is_async(&value).unwrap_err();
                assert!(error.is_instance_of::<PyTypeError>(py));
            }

            let noncallable = py
                .eval(c"type('Invalid', (), {'result': []})()", None, None)
                .unwrap();
            assert!(runner_result_is_async(&noncallable).is_err());

            let completion = py
                .eval(
                    c"type('Completion', (), {'result': lambda self: None})()",
                    None,
                    None,
                )
                .unwrap();
            assert!(runner_result_is_async(&completion).unwrap());

            let invalid_descriptor = py
                .eval(
                    c"type('Invalid', (), {'result': property(lambda self: 1 / 0)})()",
                    None,
                    None,
                )
                .unwrap();
            let error = runner_result_is_async(&invalid_descriptor).unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyZeroDivisionError>(py));
        });
    }

    fn wire_plan(group_index: u32) -> BatchPlan {
        let group = CacheGroupPlan::from_parts(
            group_index,
            vec![0, 1],
            vec![7],
            vec![0],
            vec![0],
            vec![7, 7],
            vec![0, 0],
            vec![0, 1],
            1,
        )
        .unwrap();
        BatchPlan {
            prompt_score_indices: Vec::new(),
            prompt_score_token_ids: Vec::new(),
            prompt_score_indptr: vec![0, 0],
            prompt_score_top_counts: Vec::new(),
            prompt_target_indptr: vec![0],
            prompt_target_token_ids: Vec::new(),
            sequence_ids: vec![1],
            context_lengths: vec![0],
            prefill_lengths: vec![2],
            token_ids: vec![1, 2],
            token_indptr: vec![0, 2],
            phase: BatchPhase::Prefill,
            batch_id: 1,
            batch_key: None,
            cache_groups: vec![group],
            sampling_mask: vec![true],
            sampling_params: vec![SamplingParams {
                seed: Some(1),
                ..SamplingParams::default()
            }],
            media: vec![None],
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
        let bytes = plan.encode_wire().unwrap();
        plan.positions.set(vec![0, 1]).unwrap();
        assert_eq!(bytes, plan.encode_wire().unwrap());
        let restored = BatchPlan::decode_wire(&bytes).unwrap();
        assert_eq!(restored.sequence_ids, plan.sequence_ids);
        assert_eq!(restored.token_ids, plan.token_ids);
        assert_eq!(restored.cache_groups, plan.cache_groups);
        assert!(restored.sampling_params.is_empty());
        assert!(restored.media.is_empty());
        assert!(restored.positions.get().is_none());
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
            let valid = BatchPlan::decode_wire(&wire_plan(0).encode_wire().unwrap()).unwrap();
            assert!(valid.restore_requests(&requests, &pending).is_ok());

            for (case, expected) in [
                (0, "token_indptr must start at zero"),
                (1, "prefill_lengths must cover"),
                (2, "blocks_per_slab must be positive"),
                (3, "block_indptr must end"),
                (4, "contiguous group_index"),
            ] {
                let mut plan = wire_plan(0);
                match case {
                    0 => plan.token_indptr[0] = 1,
                    1 => plan.prefill_lengths[0] = 1,
                    _ => {
                        let mut group = plan.cache_groups[0].inner_native().clone();
                        match case {
                            2 => group.blocks_per_slab = 0,
                            3 => group.block_indptr[1] = 0,
                            _ => group.group_index = 1,
                        }
                        plan.cache_groups[0] = CacheGroupPlan::from_inner(group);
                    }
                }
                // These bytes pass rkyv's memory/representation checks. Only
                // BatchPlan::checked can reject their inconsistent semantics.
                let decoded = BatchPlan::decode_wire(&plan.encode_wire().unwrap()).unwrap();
                let error = decoded.restore_requests(&requests, &pending).unwrap_err();
                assert!(error.to_string().contains(expected), "{error}");
            }
        });
    }
}
