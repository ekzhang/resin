//! Inference transactions, independent of Python and device runtimes.
//!
//! The engine owns requests and cache leases; a runner only executes a frozen
//! plan and releases model-local state. Opaque payloads carry frontend metadata.

pub use crate::logprobs::{LogprobView, TokenLogprobs};
use crate::metrics::{EngineMetricsSnapshot, EngineMetricsState, RequestMetricsState};
pub use crate::output::{OwnedTokenRows, TokenRows};
use crate::resources::RequestResourceRef;
use crate::resources::kv::{KvGroupPlan, KvLayout};
use crate::resources::paged::{CachePoolError, RequestPages, WriterAccessLease};
use crate::resources::paged_kv::{CacheMetrics, PagedKvBatchView, PagedKvError, PagedKvResources};
use crate::{
    Phase, PlanningConfig, PlanningError, RequestId, RequestProgress, Schedulable, Scheduler,
    Selection, SequenceStep, TokenRange, schedule_ready_prefix,
};
use std::cmp::Reverse;
use std::fmt;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    InvalidInput,
    Memory,
    Internal,
    Rejected,
    Execution,
}

#[derive(Clone, Debug)]
pub struct Error {
    pub kind: ErrorKind,
    message: String,
}
impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidInput,
            message: message.into(),
        }
    }
    pub fn memory(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Memory,
            message: message.into(),
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Internal,
            message: message.into(),
        }
    }
    /// Execution did not change persistent state, or all changes were rolled back.
    pub fn rejected(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Rejected,
            message: message.into(),
        }
    }
    /// Execution changed state or its effects are unknown; resources are quarantined.
    pub fn execution(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Execution,
            message: message.into(),
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}
impl From<CachePoolError> for Error {
    fn from(error: CachePoolError) -> Self {
        match error {
            CachePoolError::MetadataAllocation | CachePoolError::InsufficientCapacity { .. } => {
                Self::memory(error.to_string())
            }
            _ => Self::internal(error.to_string()),
        }
    }
}
impl From<PagedKvError> for Error {
    fn from(error: PagedKvError) -> Self {
        match error {
            PagedKvError::Cache(error) => error.into(),
            PagedKvError::Invalid(_) => Self::internal(error.to_string()),
        }
    }
}
impl From<PlanningError> for Error {
    fn from(error: PlanningError) -> Self {
        match error {
            PlanningError::TokenRangeOverflow | PlanningError::AtomicPrefillSpanExceedsBudget => {
                Self::invalid(error.to_string())
            }
            _ => Self::internal(error.to_string()),
        }
    }
}

static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(1);
fn mint_engine_id() -> Result<u64> {
    NEXT_ENGINE_ID
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| Error::internal("engine id space is exhausted"))
}
fn core_request_id(id: u64) -> RequestId {
    RequestId::new(NonZeroU64::new(id).expect("positive request id"))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchedulerConfig {
    pub max_batch_tokens: Option<NonZeroU32>,
    pub prefill_chunk_size: NonZeroU32,
    pub schedule_conservativeness: f64,
}
const DEFAULT_PREFILL_CHUNK_SIZE: NonZeroU32 = NonZeroU32::new(8192).unwrap();
const INITIAL_NEW_TOKEN_RATIO: f64 = 0.7;
const MIN_NEW_TOKEN_RATIO_FACTOR: f64 = 0.14;
const NEW_TOKEN_RATIO_DECAY_STEPS: f64 = 600.0;
const RETRACTION_DECODE_HEADROOM: usize = 20;
impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_tokens: None,
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK_SIZE,
            schedule_conservativeness: 1.0,
        }
    }
}
impl SchedulerConfig {
    fn initial_new_token_ratio(self) -> f64 {
        (INITIAL_NEW_TOKEN_RATIO * self.schedule_conservativeness).min(1.0)
    }
}

#[derive(Clone, Debug, Default)]
pub struct RunnerConfig {
    pub max_batch_tokens: Option<NonZeroU32>,
    pub prefill_chunk_size: Option<NonZeroU32>,
    pub continuation_prefill: bool,
    /// Maximum additional committed outputs from one decode invocation.
    pub cache_write_lookahead_tokens: u32,
    /// Preferred decode write span reserved before dispatch, including the input
    /// token. Zero retains the single-step reservation and same-page overlap
    /// contract. A nonzero span opts into immutable, cross-page continuation
    /// tables; the runner must resolve actual GPU positions within those tables.
    /// Cohorts too close to an output/context limit or recurrent checkpoint
    /// boundary reserve one step instead.
    pub cache_write_reservation_tokens: u32,
    pub cache_layout: Option<KvLayout>,
}

impl RunnerConfig {
    fn decode_reservation_lookahead(&self) -> u32 {
        self.cache_write_reservation_tokens
            .checked_sub(1)
            .unwrap_or(self.cache_write_lookahead_tokens)
    }

    fn decode_reservation_fits_state(&self, start: u32) -> bool {
        let first = u64::from(start);
        let last = first + u64::from(self.decode_reservation_lookahead());
        self.cache_layout.as_ref().is_none_or(|layout| {
            layout.groups().iter().all(|group| {
                let block = u64::from(group.block_tokens().get());
                !group.is_recurrent() || first / block == last / block
            })
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionReason {
    Stop,
    Length,
}
impl CompletionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}
#[derive(Clone, Debug)]
pub struct TokenEvent {
    pub request_id: u64,
    pub token_id: u32,
    pub index: usize,
    pub finished: bool,
    pub finish_reason: Option<CompletionReason>,
    pub logprobs: Option<TokenLogprobs>,
}

#[derive(Debug, Default)]
struct RequestLeaseState {
    done: bool,
    cancelled: bool,
    cached_tokens: u32,
    result: Option<Vec<u32>>,
    prompt_logprobs: Option<Vec<Option<TokenLogprobs>>>,
}
#[derive(Clone, Debug)]
pub struct RequestHandle {
    engine_id: u64,
    request_id: u64,
    state: Arc<Mutex<RequestLeaseState>>,
}
impl RequestHandle {
    pub fn prompt_logprobs(&self) -> Option<Vec<Option<TokenLogprobs>>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .prompt_logprobs
            .clone()
    }
    pub fn id(&self) -> u64 {
        self.request_id
    }
    pub fn engine_id(&self) -> u64 {
        self.engine_id
    }
    pub fn done(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .done
    }
    pub fn cancelled(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cancelled
    }
    pub fn cached_tokens(&self) -> u32 {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cached_tokens
    }
    pub fn take_result(&self) -> Result<Vec<u32>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.done {
            return Err(Error::internal("request has not completed yet"));
        }
        state
            .result
            .take()
            .ok_or_else(|| Error::internal("request result has already been consumed"))
    }
}
fn complete_request_lease(
    state: &Arc<Mutex<RequestLeaseState>>,
    result: Vec<u32>,
    cancelled: bool,
) {
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.done = true;
    state.cancelled = cancelled;
    state.result = Some(result);
}

/// Frontends normalize media and sampling into token controls and an opaque payload.
#[derive(Clone, Debug)]
pub struct Submission<P> {
    pub prompt_logprobs: bool,
    pub prompt_top_logprobs: u32,
    pub prompt_target_logprobs: Option<Vec<Vec<u32>>>,
    pub token_ids: Vec<u32>,
    pub max_output_tokens: Option<usize>,
    pub priority: i32,
    pub batch_key: Option<u64>,
    pub stop_token_ids: Vec<u32>,
    pub cache_key: Option<Vec<u32>>,
    pub cacheable: bool,
    pub atomic_prefill_spans: Vec<(u32, u32)>,
    pub payload: P,
}
impl<P> Submission<P> {
    pub fn new(token_ids: Vec<u32>, payload: P) -> Self {
        Self {
            prompt_logprobs: false,
            prompt_top_logprobs: 0,
            prompt_target_logprobs: None,
            token_ids,
            payload,
            max_output_tokens: None,
            priority: 0,
            batch_key: None,
            stop_token_ids: Vec::new(),
            cache_key: None,
            cacheable: true,
            atomic_prefill_spans: Vec::new(),
        }
    }
}

/// Owned numeric plan; the runner may move its arrays into a language adapter.
#[derive(Clone, Debug)]
pub struct ExecutionPlan<P> {
    /// Flattened predecessor rows and their next prompt tokens, partitioned by sequence.
    pub prompt_score_indices: Vec<u32>,
    pub prompt_score_token_ids: Vec<u32>,
    pub prompt_score_indptr: Vec<u32>,
    pub prompt_score_top_counts: Vec<u32>,
    pub prompt_target_indptr: Vec<u32>,
    pub prompt_target_token_ids: Vec<u32>,
    pub batch_id: u64,
    pub batch_key: Option<u64>,
    pub phase: Phase,
    pub sequence_ids: Vec<u64>,
    pub context_lengths: Vec<u32>,
    /// Final prefill-history endpoints, including generated inputs on retraction.
    /// Only meaningful for prefill; constant across that history's chunks.
    pub prefill_lengths: Vec<u32>,
    pub token_ids: Vec<u32>,
    pub token_indptr: Vec<u32>,
    pub sampling_mask: Vec<bool>,
    pub cache_groups: Vec<KvGroupPlan>,
    pub payloads: Vec<P>,
}

// Foreign/user callbacks can unwind as well as return errors. Once a callback
// may have changed model state, either failure must take the quarantine path.
fn execution_call<T>(stage: &str, call: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(call))
        .unwrap_or_else(|_| Err(Error::execution(format!("runner panicked during {stage}"))))
}

pub trait Runner<P> {
    fn forward(&mut self, plan: ExecutionPlan<P>) -> Result<OwnedTokenRows>;
    fn logprobs(&self) -> Option<LogprobView<'_>> {
        None
    }
    fn prompt_logprobs(&self) -> Option<LogprobView<'_>> {
        None
    }
    fn release(&mut self, _request_ids: &[u64]) -> Result<()> {
        Ok(())
    }
}
#[derive(Debug)]
struct Request<P> {
    metrics: Option<RequestMetricsState>,
    prompt_logprobs: bool,
    prompt_top_logprobs: u32,
    prompt_target_logprobs: Option<Vec<Vec<u32>>>,
    id: u64,
    lease: Arc<Mutex<RequestLeaseState>>,
    priority: i32,
    batch_key: Option<u64>,
    turn_ordinal: u128,
    running: bool,
    prompt: Vec<u32>,
    cache_prompt: Option<Vec<u32>>,
    atomic_prefill_spans: Vec<(u32, u32)>,
    resident_tokens: u32,
    generated: Vec<u32>,
    max_output_tokens: usize,
    stop_token_ids: Vec<u32>,
    payload: P,
    cacheable: bool,
    cache: Option<RequestPages>,
}

impl<P> Request<P> {
    fn prompt_score_range(&self, row: SequenceStep) -> std::ops::Range<usize> {
        if !self.prompt_logprobs || row.phase != Phase::Prefill {
            return 0..0;
        }
        let start = row.token_start() as usize;
        start..(row.token_end().unwrap() as usize).min(self.prompt.len().saturating_sub(1))
    }

    fn prompt_targets(&self, predecessor: usize) -> &[u32] {
        self.prompt_target_logprobs
            .as_ref()
            .map_or(&[], |rows| &rows[predecessor + 1])
    }

    fn prompt_length(&self) -> u32 {
        u32::try_from(self.prompt.len()).expect("submit bounds prompt length to u32")
    }

    fn prefill_length(&self) -> Result<u32> {
        let generated_inputs =
            u32::try_from(self.generated.len().saturating_sub(1)).map_err(|_| {
                Error::invalid("generated sequence length exceeds the u32 metadata range")
            })?;
        self.prompt_length()
            .checked_add(generated_inputs)
            .ok_or_else(|| Error::invalid("sequence position exceeds the u32 metadata range"))
    }

    fn logical_context(&self) -> Result<u32> {
        let prefill_length = self.prefill_length()?;
        if self.resident_tokens > prefill_length
            || (self.resident_tokens == prefill_length && self.generated.is_empty())
        {
            return Err(Error::internal(format!(
                "request {} has invalid resident progress {} for input length {prefill_length}",
                self.id, self.resident_tokens
            )));
        }
        Ok(self.resident_tokens)
    }

    fn sample_at_prefill_end(&self) -> bool {
        self.max_output_tokens > 0 && self.generated.is_empty()
    }

    /// Check the frozen row both before execution and before committing progress.
    fn validate_step(&self, scheduled: ScheduledRow) -> Result<u32> {
        if scheduled.phase != self.phase() || scheduled.token_start() != self.logical_context()? {
            return Err(Error::internal(format!(
                "request {} progress changed after its row was scheduled",
                self.id
            )));
        }
        let token_end = scheduled
            .token_end()
            .ok_or_else(|| Error::invalid("scheduled token range overflows u32"))?;
        let expected_sample = match scheduled.phase {
            Phase::Prefill => {
                let prefill_length = self.prefill_length()?;
                if token_end > prefill_length {
                    return Err(Error::internal(
                        "scheduled prefill range exceeds the request replay input",
                    ));
                }
                self.sample_at_prefill_end() && token_end == prefill_length
            }
            Phase::Decode => {
                if scheduled.token_count().get() != 1 {
                    return Err(Error::internal(
                        "scheduled decode rows must contain exactly one token",
                    ));
                }
                true
            }
        };
        if scheduled.sample != expected_sample {
            return Err(Error::internal(
                "scheduled row sampling does not match its token range",
            ));
        }
        Ok(token_end)
    }
}

impl<P> Schedulable for Request<P> {
    fn priority(&self) -> i32 {
        self.priority
    }

    fn batch_key(&self) -> Option<u64> {
        self.batch_key
    }

    fn phase(&self) -> Phase {
        let input_length = self.prompt.len() + self.generated.len().saturating_sub(1);
        if (self.resident_tokens as usize) < input_length {
            Phase::Prefill
        } else {
            Phase::Decode
        }
    }

    fn is_running(&self) -> bool {
        self.running
    }

    fn set_running(&mut self, running: bool) {
        self.running = running;
    }

    fn turn_ordinal(&self) -> u128 {
        self.turn_ordinal
    }

    fn set_turn_ordinal(&mut self, turn: u128) {
        self.turn_ordinal = turn;
    }
}

type ScheduledRow = SequenceStep;

struct BatchRow<P> {
    request: Request<P>,
    scheduled: ScheduledRow,
    // Logical completion may precede the already-enqueued step's retirement.
    // Keep its physical row and cache ownership until that step is harvested.
    terminal: bool,
}

struct PreparedProgress {
    resident_tokens: u32,
    sampled_tokens: Vec<u32>,
}

struct Batch<P> {
    rows: Vec<BatchRow<P>>,
    resources: Option<WriterAccessLease>,
    prepared_at: Option<Instant>,
}

pub struct EngineCore<P> {
    metrics: Option<EngineMetricsState>,
    in_flight: Option<Batch<P>>,
    continuation: Option<ExecutionPlan<P>>,
    quarantined: Option<Batch<P>>,
    runner_config: RunnerConfig,
    scheduler: Scheduler<Request<P>>,
    scheduler_config: SchedulerConfig,
    new_token_ratio: f64,
    minimum_new_token_ratio: f64,
    new_token_ratio_decay: f64,
    context_length: NonZeroU32,
    engine_id: u64,
    next_request_id: u64,
    next_batch_id: u64,
    kv_resources: Option<PagedKvResources>,
    poisoned_error: Option<String>,
    closed: bool,
}

impl<P: Clone> EngineCore<P> {
    pub fn new(
        runner_config: RunnerConfig,
        max_running_requests: usize,
        context_length: u32,
        scheduler_config: SchedulerConfig,
        prefix_cache: bool,
    ) -> Result<Self> {
        let max_running_requests = NonZeroUsize::new(max_running_requests)
            .ok_or_else(|| Error::invalid("max_running_requests must be positive"))?;
        let context_length = NonZeroU32::new(context_length)
            .ok_or_else(|| Error::invalid("context_length must be positive"))?;
        if !scheduler_config.schedule_conservativeness.is_finite()
            || scheduler_config.schedule_conservativeness < 0.0
        {
            return Err(Error::invalid(
                "schedule_conservativeness must be finite and non-negative",
            ));
        }
        if runner_config.cache_write_lookahead_tokens > 0 && runner_config.cache_layout.is_none() {
            return Err(Error::invalid(
                "cache_write_lookahead_tokens requires cache_layout",
            ));
        }
        if runner_config.cache_write_reservation_tokens > 0 {
            if runner_config.cache_layout.is_none() {
                return Err(Error::invalid(
                    "cache_write_reservation_tokens requires cache_layout",
                ));
            }
            if u64::from(runner_config.cache_write_reservation_tokens)
                < 2 * (u64::from(runner_config.cache_write_lookahead_tokens) + 1)
            {
                return Err(Error::invalid(
                    "cache_write_reservation_tokens must cover two maximum decode steps",
                ));
            }
        }
        let kv_resources = runner_config
            .cache_layout
            .as_ref()
            .map(|layout| {
                if prefix_cache {
                    PagedKvResources::with_prefix_cache(layout.clone())
                } else {
                    PagedKvResources::new(layout.clone())
                }
            })
            .transpose()?;
        if let Some(limit) = kv_resources
            .as_ref()
            .and_then(PagedKvResources::max_write_tokens)
            && u64::from(runner_config.decode_reservation_lookahead()) + 1 > u64::from(limit.get())
        {
            return Err(Error::invalid(
                "decode cache write reservation exceeds the Window read/write capacity",
            ));
        }
        let new_token_ratio = scheduler_config.initial_new_token_ratio();
        let minimum_new_token_ratio = new_token_ratio * MIN_NEW_TOKEN_RATIO_FACTOR;
        let new_token_ratio_decay =
            (new_token_ratio - minimum_new_token_ratio) / NEW_TOKEN_RATIO_DECAY_STEPS;
        Ok(Self {
            metrics: None,
            in_flight: None,
            continuation: None,
            quarantined: None,
            runner_config,
            scheduler: Scheduler::new(max_running_requests),
            scheduler_config,
            new_token_ratio,
            minimum_new_token_ratio,
            new_token_ratio_decay,
            context_length,
            engine_id: mint_engine_id()?,
            next_request_id: 1,
            next_batch_id: 1,
            kv_resources,
            poisoned_error: None,
            closed: false,
        })
    }

    pub fn id(&self) -> u64 {
        self.engine_id
    }

    /// Enable native accounting before submission, preserving cumulative totals.
    pub fn enable_metrics(&mut self) -> Result<()> {
        if self.metrics.is_some() {
            return Ok(());
        }
        self.raise_if_open()?;
        if self.next_request_id != 1 {
            return Err(Error::invalid(
                "metrics must be enabled before submitting requests",
            ));
        }
        self.metrics = Some(EngineMetricsState::new());
        Ok(())
    }

    pub fn metrics_enabled(&self) -> bool {
        self.metrics.is_some()
    }

    /// Copy observations without device synchronization or runner callbacks.
    pub fn metrics_snapshot(&self) -> Option<EngineMetricsSnapshot> {
        let mut snapshot = self.metrics.as_ref()?.snapshot();
        snapshot.waiting_requests = self
            .scheduler
            .iter()
            .filter(|request| !request.running)
            .count();
        snapshot.running_requests = self
            .scheduler
            .iter()
            .filter(|request| request.running)
            .count()
            + self
                .in_flight
                .iter()
                .chain(&self.quarantined)
                .flat_map(|batch| &batch.rows)
                .filter(|row| !row.terminal)
                .count();
        Some(snapshot)
    }
    pub fn next_request_id(&self) -> u64 {
        self.next_request_id
    }
    pub fn scheduler_config(&self) -> SchedulerConfig {
        self.scheduler_config
    }
    pub fn new_token_ratio(&self) -> f64 {
        self.new_token_ratio
    }
    pub fn context_length(&self) -> u32 {
        self.context_length.get()
    }
    pub fn has_work(&self) -> bool {
        self.in_flight.is_some() || !self.scheduler.is_empty()
    }
    pub fn active_request_count(&self) -> usize {
        self.scheduler.len()
            + self.in_flight.as_ref().map_or(0, |batch| {
                batch.rows.iter().filter(|row| !row.terminal).count()
            })
    }

    pub fn ready_request_count(&self) -> usize {
        self.scheduler.len()
    }
    pub fn max_running_requests(&self) -> usize {
        self.scheduler.max_running_requests()
    }
    pub fn poisoned(&self) -> bool {
        self.poisoned_error.is_some()
    }
    pub fn closed(&self) -> bool {
        self.closed
    }
    pub fn poison_error(&self) -> Option<&str> {
        self.poisoned_error.as_deref()
    }
    pub fn cache_pool_id(&self) -> Option<u64> {
        self.kv_resources.as_ref().map(|r| r.pool().id())
    }
    pub fn cache_parent_capacity(&self) -> Option<u32> {
        self.kv_resources.as_ref().map(|r| r.parent_capacity())
    }
    pub fn cache_free_parent_count(&self) -> Option<usize> {
        self.kv_resources
            .as_ref()
            .map(|r| r.pool().free_parent_count())
    }
    pub fn cache_occupied_block_count(&self) -> Option<usize> {
        self.kv_resources
            .as_ref()
            .map(|r| r.pool().occupied_block_count())
    }
    pub fn prefix_cache_stats(&self) -> Option<(usize, u64, u64, u64)> {
        self.kv_resources
            .as_ref()
            .and_then(PagedKvResources::prefix_metrics)
            .map(|stats| {
                (
                    stats.entries as usize,
                    stats.hits,
                    stats.misses,
                    stats.evictions,
                )
            })
    }
    pub fn cache_metrics(&self) -> Option<CacheMetrics> {
        self.kv_resources.as_ref().map(PagedKvResources::metrics)
    }
    pub fn clear_prefix_cache(&mut self) -> Result<()> {
        self.raise_if_open()?;
        if let Some(resources) = self.kv_resources.as_mut() {
            resources.clear_prefix_cache()?;
        }
        Ok(())
    }
    fn planning_config(&self) -> PlanningConfig {
        let min_limit = |a: Option<NonZeroU32>, b: Option<NonZeroU32>| match (a, b) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let prefill_limit = min_limit(
            self.runner_config.prefill_chunk_size,
            self.kv_resources
                .as_ref()
                .and_then(PagedKvResources::max_write_tokens),
        );
        PlanningConfig {
            max_batch_tokens: min_limit(
                self.scheduler_config.max_batch_tokens,
                self.runner_config.max_batch_tokens,
            ),
            prefill_chunk_size: prefill_limit
                .map_or(self.scheduler_config.prefill_chunk_size, |limit| {
                    limit.min(self.scheduler_config.prefill_chunk_size)
                }),
            checkpoint_alignment: self.prefix_alignment().and_then(NonZeroU32::new),
        }
    }

    pub fn submit(&mut self, input: Submission<P>) -> Result<RequestHandle> {
        self.submit_with(input, Ok)
    }
    /// Allocate the frontend handle before publishing a request into native state.
    pub fn submit_with<H>(
        &mut self,
        input: Submission<P>,
        make_handle: impl FnOnce(RequestHandle) -> Result<H>,
    ) -> Result<H> {
        self.raise_if_open()?;
        let submitted_at = self.metrics.as_ref().map(|_| Instant::now());
        let Submission {
            prompt_logprobs,
            prompt_top_logprobs,
            prompt_target_logprobs,
            token_ids,
            max_output_tokens,
            priority,
            batch_key,
            mut stop_token_ids,
            cache_key: cache_prompt,
            cacheable,
            atomic_prefill_spans,
            payload,
        } = input;
        if token_ids.is_empty() {
            return Err(Error::invalid("token_ids must be non-empty"));
        }
        if prompt_top_logprobs > 1000 {
            return Err(Error::invalid(
                "prompt_top_logprobs must be between 0 and 1000",
            ));
        }
        if let Some(targets) = &prompt_target_logprobs {
            if targets.len() != token_ids.len() || !targets[0].is_empty() {
                return Err(Error::invalid(
                    "prompt_target_logprobs must match prompt length and begin with an empty row",
                ));
            }
            targets.iter().try_fold(0_u32, |total, row| {
                total
                    .checked_add(u32::try_from(row.len()).map_err(|_| {
                        Error::invalid("prompt target count exceeds u32 metadata range")
                    })?)
                    .ok_or_else(|| Error::invalid("prompt target count exceeds u32 metadata range"))
            })?;
        }
        let prompt_logprobs =
            prompt_logprobs || prompt_top_logprobs > 0 || prompt_target_logprobs.is_some();
        if cache_prompt
            .as_ref()
            .is_some_and(|key| key.len() != token_ids.len())
        {
            return Err(Error::invalid(
                "media cache_key must match token_ids length",
            ));
        }
        let mut previous_end = 0;
        let effective_limits = self.planning_config();
        let atomic_limit = effective_limits
            .max_batch_tokens
            .map_or(effective_limits.prefill_chunk_size, |budget| {
                budget.min(effective_limits.prefill_chunk_size)
            })
            .get();
        for &(start, end) in &atomic_prefill_spans {
            if start >= end
                || end as usize > token_ids.len()
                || start < previous_end
                || end - start > atomic_limit
            {
                return Err(Error::invalid(
                    "media token_spans must be sorted, disjoint, within the prompt, and fit one prefill row",
                ));
            }
            previous_end = end;
        }
        if max_output_tokens == Some(0) && !prompt_logprobs {
            return Err(Error::invalid(
                "max_output_tokens must be positive unless prompt_logprobs is enabled",
            ));
        }
        let max_output_tokens = max_output_tokens.unwrap_or_else(|| {
            (self.context_length.get() as usize).saturating_sub(token_ids.len()) + 1
        });
        let max_input_tokens = token_ids
            .len()
            .checked_add(max_output_tokens.saturating_sub(1))
            .filter(|length| *length <= u32::MAX as usize)
            .ok_or_else(|| {
                Error::invalid("maximum request input length exceeds the u32 metadata range")
            })? as u32;
        if max_input_tokens > self.context_length.get() {
            return Err(Error::invalid(format!(
                "request context length {max_input_tokens} exceeds Engine context_length {}",
                self.context_length.get()
            )));
        }

        let request_id = self.next_request_id;
        let next_request_id = request_id
            .checked_add(1)
            .ok_or_else(|| Error::internal("request id space is exhausted"))?;
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();
        // Create the caller-owned lease before any native admission side
        // effect. A Python allocation failure therefore leaves submission
        // fully transactional.
        let lease_state = Arc::new(Mutex::new(RequestLeaseState {
            prompt_logprobs: prompt_logprobs.then(|| vec![None; token_ids.len()]),
            ..RequestLeaseState::default()
        }));
        let lease = make_handle(RequestHandle {
            engine_id: self.engine_id,
            request_id,
            state: Arc::clone(&lease_state),
        })?;

        let mut pages = self
            .kv_resources
            .as_mut()
            .map(|resources| {
                resources
                    .attach(core_request_id(request_id), 0)
                    .map_err(Error::from)
            })
            .transpose()?;
        // Media may provide a same-length, digest-derived identity stream. A
        // media object without one stays out of the token-only prefix cache.
        let cache_identity = cache_prompt
            .as_deref()
            .or_else(|| cacheable.then_some(token_ids.as_slice()));
        // KV entries do not retain predecessor distributions. Scoring must run
        // the prompt once, even when an ordinary request could reuse its KV.
        let cache_identity = cache_identity.filter(|_| !prompt_logprobs);
        let resident_tokens = match (&mut self.kv_resources, &mut pages, cache_identity) {
            (Some(resources), Some(pages), Some(identity)) => resources
                .restore_prefix(pages, identity)
                .map_err(Error::from)?,
            _ => 0,
        };
        lease_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cached_tokens = resident_tokens;
        self.next_request_id = next_request_id;
        let metrics = self.metrics.as_mut().map(|metrics| {
            metrics.submit(
                token_ids.len() as u32,
                resident_tokens,
                submitted_at.expect("metrics-enabled submission has a timestamp"),
            )
        });
        self.scheduler.enqueue(Request {
            metrics,
            prompt_logprobs,
            prompt_top_logprobs,
            prompt_target_logprobs,
            id: request_id,
            lease: lease_state,
            priority,
            batch_key,
            turn_ordinal: 0,
            running: false,
            prompt: token_ids,
            cache_prompt,
            atomic_prefill_spans,
            resident_tokens,
            generated: Vec::new(),
            max_output_tokens,
            stop_token_ids,
            payload,
            cacheable,
            cache: pages,
        });
        Ok(lease)
    }
    pub fn step(&mut self, runner: &mut impl Runner<P>) -> Result<Vec<TokenEvent>> {
        self.step_with(runner, Ok)
    }

    /// Convert events before committing, so a frontend allocation failure is atomic.
    pub fn step_with<E>(
        &mut self,
        runner: &mut impl Runner<P>,
        make_event: impl FnMut(TokenEvent) -> Result<E>,
    ) -> Result<Vec<E>> {
        self.step_phase_with(runner, None, make_event)
    }

    /// A coordinator may gate dispatch without reserving or exposing a pending batch.
    pub fn step_phase_with<E>(
        &mut self,
        runner: &mut impl Runner<P>,
        phase: Option<Phase>,
        make_event: impl FnMut(TokenEvent) -> Result<E>,
    ) -> Result<Vec<E>> {
        let Some(plan) = self.prepare_step(runner, phase)? else {
            return Ok(Vec::new());
        };
        let output = execution_call("forward", || runner.forward(plan));
        let output = output
            .as_ref()
            .map(OwnedTokenRows::view)
            .map_err(Clone::clone);
        self.complete_step_with(runner, output, make_event)
    }

    /// Freeze a plan and retain its requests and writer lease until completion.
    ///
    /// A device runtime may enqueue work and return immediately. Neither a
    /// second preparation nor retirement may reuse its pages before the exact
    /// accepted transaction is completed or rejected.
    pub fn prepare_step(
        &mut self,
        runner: &mut impl Runner<P>,
        phase: Option<Phase>,
    ) -> Result<Option<ExecutionPlan<P>>> {
        self.raise_if_open()?;
        self.require_completed_step()?;
        let Some((batch, plan)) = self.prepare_batch(runner, phase)? else {
            return Ok(None);
        };
        self.in_flight = Some(batch);
        Ok(Some(plan))
    }

    pub fn has_in_flight_step(&self) -> bool {
        self.in_flight.is_some()
    }

    pub fn has_continuation(&self) -> bool {
        self.continuation.is_some() && self.scheduler.is_empty()
    }

    /// Harvest an enqueued result using the same atomic commit as synchronous execution.
    pub fn complete_step_with<E>(
        &mut self,
        runner: &mut impl Runner<P>,
        output: Result<TokenRows<'_>>,
        make_event: impl FnMut(TokenEvent) -> Result<E>,
    ) -> Result<Vec<E>> {
        self.raise_if_open()?;
        self.continuation = None;
        let Some(mut batch) = self.in_flight.take() else {
            return Err(Error::internal("Engine has no in-flight step to complete"));
        };
        // A rejection promises that persistent state is unchanged. Every other
        // execution failure keeps the exact requests and writer lease quarantined.
        let output = match output {
            Err(error)
                if error.kind == ErrorKind::Rejected
                    && !batch.rows.iter().any(|row| row.terminal) =>
            {
                self.release_kv_access(batch.resources.take());
                self.scheduler
                    .restore_rejected(batch.rows.into_iter().map(|row| row.request).collect());
                return Err(error);
            }
            Err(error) => {
                return self.quarantine(batch, "runner forward failed after accepting work", error);
            }
            Ok(output) => output,
        };
        self.complete_batch(runner, batch, output, make_event, None)
    }

    /// Retain a decode view only when another maximum-sized step cannot
    /// allocate, rotate, replace, or publish any of its leased pages. Stops
    /// complete logically while physical retirement waits for the next step.
    /// The caller separately checks that its device runtime supports feedback.
    pub fn retain_continuation(&mut self, plan: &ExecutionPlan<P>) -> bool {
        let maximum = u64::from(self.runner_config.cache_write_lookahead_tokens) + 1;
        let Some(batch) = &self.in_flight else {
            return false;
        };
        let allowed = self.scheduler.is_empty()
            && plan.phase == Phase::Decode
            && batch.rows.len() == plan.sequence_ids.len()
            && batch.rows.iter().all(|row| {
                !row.terminal
                    && row
                        .request
                        .max_output_tokens
                        .saturating_sub(row.request.generated.len()) as u64
                        > 2 * maximum
                    && u64::from(row.scheduled.token_start()) + 2 * maximum
                        <= u64::from(self.context_length.get())
            })
            && self
                .runner_config
                .cache_layout
                .as_ref()
                .is_some_and(|layout| {
                    layout.groups().iter().enumerate().all(|(index, group)| {
                        plan.context_lengths
                            .iter()
                            .enumerate()
                            .all(|(row, &start)| {
                                let block = u64::from(group.block_tokens().get());
                                if self.runner_config.cache_write_reservation_tokens == 0
                                    || group.is_recurrent()
                                {
                                    // State checkpoints keep their ordinary ring
                                    // ownership: no continuation may cross a
                                    // checkpoint boundary, even with wider KV leases.
                                    u64::from(start) / block
                                        == (u64::from(start) + 2 * maximum - 1) / block
                                } else {
                                    // The page table is the frozen lease envelope,
                                    // not a promise that new pages may be allocated.
                                    // Full-page leases permit further continuations
                                    // only until the last already-pinned page ends.
                                    let table = &plan.cache_groups[index];
                                    let first = u64::from(table.block_table_start_indices[row]);
                                    let count = u64::from(
                                        table.block_indptr[row + 1] - table.block_indptr[row],
                                    );
                                    let earliest_read = group
                                        .read_window_tokens()
                                        .map_or(0, |window| start.saturating_sub(window.get() - 1));
                                    let lease_end = (first + count) * block;
                                    u64::from(earliest_read) / block >= first
                                        && u64::from(start) + 2 * maximum <= lease_end
                                        && group.max_write_tokens().is_none_or(|limit| {
                                            lease_end - u64::from(start) <= u64::from(limit.get())
                                        })
                                }
                            })
                    })
                });
        self.continuation = allowed.then(|| plan.clone());
        allowed
    }

    /// A conservative host plan reserves the longest possible previous output;
    /// the runtime substitutes exact GPU token/position feedback before launch.
    pub fn continuation_plan(&mut self) -> Result<Option<ExecutionPlan<P>>> {
        self.raise_if_open()?;
        if !self.scheduler.is_empty() {
            return Ok(None);
        }
        let Some(template) = &self.continuation else {
            return Ok(None);
        };
        let mut plan = template.clone();
        plan.batch_id = self.next_batch_id;
        self.next_batch_id = self
            .next_batch_id
            .checked_add(1)
            .ok_or_else(|| Error::internal("batch id space is exhausted"))?;
        let maximum = self.runner_config.cache_write_lookahead_tokens + 1;
        for context in &mut plan.context_lengths {
            *context += maximum;
        }
        for (group, layout) in plan.cache_groups.iter_mut().zip(
            self.runner_config
                .cache_layout
                .as_ref()
                .expect("continuation keeps resources")
                .groups(),
        ) {
            for (row, &position) in plan.context_lengths.iter().enumerate() {
                group.append_block_offsets[row] = position % layout.block_tokens().get();
                if self.runner_config.cache_write_reservation_tokens > 0 {
                    let logical = position / layout.block_tokens().get();
                    let relative = logical
                        .checked_sub(group.block_table_start_indices[row])
                        .ok_or_else(|| Error::internal("continuation precedes its KV lease"))?;
                    let address = group.block_indptr[row] + relative;
                    if address >= group.block_indptr[row + 1] {
                        return Err(Error::internal("continuation exceeds its KV lease"));
                    }
                    group.append_slab_ids[row] = group.block_slab_ids[address as usize];
                    group.append_slotnums[row] = group.block_slotnums[address as usize];
                }
            }
        }
        Ok(Some(plan))
    }

    /// Commit the previous output while the next device step uses the same
    /// page lease. Terminal rows keep their original physical slots and cache
    /// ownership until that next step completes; its outputs are discarded.
    pub fn complete_continuing_step_with<E>(
        &mut self,
        runner: &mut impl Runner<P>,
        output: Result<TokenRows<'_>>,
        make_event: impl FnMut(TokenEvent) -> Result<E>,
    ) -> Result<Vec<E>> {
        self.raise_if_open()?;
        let Some(template) = self.continuation.take() else {
            return Err(Error::internal("Engine has no safe decode continuation"));
        };
        let batch = self
            .in_flight
            .take()
            .ok_or_else(|| Error::internal("Engine has no in-flight step to continue"))?;
        let output = match output {
            // A next forward was already accepted: even a previous-step
            // rejection cannot justify returning its shared writer lease.
            Err(error) => {
                return self.quarantine(batch, "overlapped execution failed after enqueue", error);
            }
            Ok(output) => output,
        };
        self.complete_batch(runner, batch, output, make_event, Some(template))
    }

    fn selected_cohort(&self) -> Result<(Vec<&Request<P>>, Selection)> {
        let (mut requests, mut selection) = self.scheduler.selected();
        let admitted = self.admission_prefix_len(&requests)?;
        if admitted == 0 {
            (requests, selection) = self.scheduler.selected_running();
        } else {
            requests.truncate(admitted);
            selection.truncate(admitted);
        }
        Ok((requests, selection))
    }

    /// Read-only proposal using the same admission fallback as actual dispatch.
    pub fn next_phase(&self) -> Result<Option<Phase>> {
        self.raise_if_open()?;
        if let Some(batch) = &self.in_flight {
            return Ok(batch.rows.first().map(|row| row.scheduled.phase));
        }
        Ok(self
            .selected_cohort()?
            .0
            .first()
            .map(|request| request.phase()))
    }

    fn prepare_batch(
        &mut self,
        runner: &mut impl Runner<P>,
        phase: Option<Phase>,
    ) -> Result<Option<(Batch<P>, ExecutionPlan<P>)>> {
        self.raise_if_open()?;
        if self.scheduler.is_empty() {
            return Ok(None);
        }

        let (scheduled_rows, selection, mut decode_reservation_lookahead) = {
            let (selected_requests, mut selection) = self.selected_cohort()?;
            if selected_requests.is_empty() {
                return Ok(None);
            }
            if phase.is_some_and(|phase| selected_requests[0].phase() != phase) {
                return Ok(None);
            }
            let scheduled_rows = schedule_requests(&selected_requests, self.planning_config())?;
            if scheduled_rows.is_empty() {
                return Err(Error::internal(
                    "scheduler produced no token-bearing rows for a nonempty cohort",
                ));
            }
            if scheduled_rows
                .iter()
                .zip(&selected_requests)
                .any(|(row, request)| {
                    row.phase == Phase::Prefill
                        && !row.sample
                        && (row.token_start() != 0
                            || row.token_end() != Some(request.prompt_length()))
                })
                && !self.runner_config.continuation_prefill
            {
                return Err(Error::invalid(
                    "prefill_chunk_size creates continuation rows but runner_config.continuation_prefill is false",
                ));
            }
            selection.truncate(scheduled_rows.len());
            let maximum = u64::from(self.runner_config.cache_write_lookahead_tokens) + 1;
            let reserve_continuation = self.runner_config.cache_write_reservation_tokens > 0
                && scheduled_rows
                    .iter()
                    .zip(&selected_requests)
                    .all(|(row, request)| {
                        row.phase == Phase::Decode
                            && request
                                .max_output_tokens
                                .saturating_sub(request.generated.len())
                                as u64
                                > 2 * maximum
                            && u64::from(row.token_start())
                                + u64::from(self.runner_config.cache_write_reservation_tokens)
                                <= u64::from(self.context_length.get())
                            // Within one checkpoint block, the wider context
                            // has exactly the ordinary state-ring active range.
                            // Otherwise fall back for the whole cohort before
                            // growth, COW, or writer-lease acquisition.
                            && self
                                .runner_config
                                .decode_reservation_fits_state(row.token_start())
                    });
            let lookahead = if reserve_continuation {
                self.runner_config.decode_reservation_lookahead()
            } else {
                self.runner_config.cache_write_lookahead_tokens
            };
            (scheduled_rows, selection, lookahead)
        };
        let batch_id = self.next_batch_id;
        self.next_batch_id = self
            .next_batch_id
            .checked_add(1)
            .ok_or_else(|| Error::internal("batch id space is exhausted"))?;
        for row in &scheduled_rows {
            let end = row
                .token_end()
                .ok_or_else(|| Error::invalid("scheduled token range overflows u32"))?;
            if row.phase == Phase::Decode && end.checked_add(decode_reservation_lookahead).is_none()
            {
                return Err(Error::invalid("decode cache write lookahead overflows u32"));
            }
        }
        let mut scheduled = self.scheduler.take_selected(selection);

        if decode_reservation_lookahead > self.runner_config.cache_write_lookahead_tokens {
            let resources = self.kv_resources.as_ref().expect("reservation has a cache");
            let requests = scheduled
                .iter()
                .zip(&scheduled_rows)
                .map(|(request, row)| {
                    (
                        request
                            .cache
                            .as_ref()
                            .expect("managed request has cache state"),
                        row.token_start(),
                        row.token_end().expect("validated scheduled range")
                            + decode_reservation_lookahead,
                    )
                })
                .collect::<Vec<_>>();
            match resources.can_ensure_contexts(&requests) {
                Ok(true) => {}
                Ok(false) => {
                    decode_reservation_lookahead = self.runner_config.cache_write_lookahead_tokens
                }
                Err(error) => {
                    self.scheduler.restore_rejected(scheduled);
                    return Err(Error::from(error));
                }
            }
        }

        let growth_error = if let Some(resources) = self.kv_resources.as_mut() {
            let mut growth_error = None;
            for (request, row) in scheduled.iter_mut().zip(&scheduled_rows) {
                let scheduled_end = row.token_end().expect("validated scheduled range");
                let context = if row.phase == Phase::Decode {
                    scheduled_end
                        .checked_add(decode_reservation_lookahead)
                        .expect("validated decode lookahead")
                } else {
                    scheduled_end
                };
                let pages = request
                    .cache
                    .as_mut()
                    .expect("managed requests keep their cache state");
                if let Err(error) = resources.ensure_context(pages, row.token_start(), context) {
                    growth_error = Some(error);
                    break;
                }
            }
            growth_error
        } else {
            None
        };
        if let Some(error) = growth_error {
            self.scheduler.restore_rejected(scheduled);
            if matches!(error, CachePoolError::InsufficientCapacity { .. })
                && self.retract_one(runner)?
            {
                return self.prepare_batch(runner, phase);
            }
            return Err(Error::from(error));
        }

        let mut acquired_kv = match self.kv_resources.as_mut() {
            Some(resources) => {
                let requests = scheduled
                    .iter()
                    .map(|request| RequestResourceRef {
                        request_id: core_request_id(request.id),
                        state: request
                            .cache
                            .as_ref()
                            .expect("managed requests keep their cache state"),
                    })
                    .collect::<Vec<_>>();
                match resources.acquire(&requests, &scheduled_rows, decode_reservation_lookahead) {
                    Ok(acquired) => Some(acquired),
                    Err(error) => {
                        self.scheduler.restore_rejected(scheduled);
                        return Err(Error::from(error));
                    }
                }
            }
            None => {
                if scheduled.iter().any(|request| request.cache.is_some()) {
                    self.scheduler.restore_rejected(scheduled);
                    return Err(Error::internal(
                        "unmanaged Engine request unexpectedly owns KV resource state",
                    ));
                }
                None
            }
        };

        // Payload cloning is caller code too. It runs before runner execution,
        // so an unwind must release access and restore the selected requests.
        let plan = catch_unwind(AssertUnwindSafe(|| {
            build_plan(
                batch_id,
                &scheduled,
                &scheduled_rows,
                acquired_kv.as_mut().map(|acquired| &mut acquired.view),
            )
        }))
        .unwrap_or_else(|_| {
            Err(Error::rejected(
                "payload cloning panicked before runner execution",
            ))
        });
        let plan = match plan {
            Ok(plan) => plan,
            Err(error) => {
                self.release_kv_access(acquired_kv.take().map(|access| access.lease));
                self.scheduler.restore_rejected(scheduled);
                return Err(error);
            }
        };
        let batch = Batch {
            prepared_at: self.metrics.as_ref().map(|_| Instant::now()),
            rows: scheduled
                .into_iter()
                .zip(scheduled_rows)
                .map(|(request, scheduled)| BatchRow {
                    request,
                    scheduled,
                    terminal: false,
                })
                .collect(),
            resources: acquired_kv.map(|access| access.lease),
        };

        Ok(Some((batch, plan)))
    }

    fn complete_batch<E>(
        &mut self,
        runner: &mut impl Runner<P>,
        mut batch: Batch<P>,
        output_token_ids: TokenRows<'_>,
        mut make_event: impl FnMut(TokenEvent) -> Result<E>,
        continuation: Option<ExecutionPlan<P>>,
    ) -> Result<Vec<E>> {
        // Validate the complete execution result before terminal output limits
        // can hide writes beyond the leased capacity or extra prefill tokens.
        let logprobs = match execution_call("logprob access", || Ok(runner.logprobs())) {
            Ok(scores) => scores,
            Err(error) => return self.quarantine(batch, "runner logprob access failed", error),
        };
        if let Err(error) = self
            .validate_runner_output(&batch.rows, output_token_ids)
            .and_then(|()| logprobs.map_or(Ok(()), |scores| scores.validate(output_token_ids)))
        {
            return self.quarantine(batch, "runner output exceeds its execution contract", error);
        }
        let decoded_batch = batch
            .rows
            .first()
            .is_some_and(|row| row.scheduled.phase == Phase::Decode);

        let prompt_logprobs =
            match execution_call("prompt logprob access", || Ok(runner.prompt_logprobs())) {
                Ok(scores) => scores,
                Err(error) => {
                    return self.quarantine(batch, "runner prompt logprob access failed", error);
                }
            };
        let prompt_counts = batch
            .rows
            .iter()
            .map(|row| row.request.prompt_score_range(row.scheduled).len())
            .collect::<Vec<_>>();
        let mut prompt_top_counts = Vec::new();
        let mut prompt_target_indptr = Vec::new();
        let mut prompt_target_ids = Vec::new();
        for row in &batch.rows {
            for position in row.request.prompt_score_range(row.scheduled) {
                if prompt_target_indptr.is_empty() {
                    prompt_target_indptr.push(0);
                }
                prompt_top_counts.push(row.request.prompt_top_logprobs);
                prompt_target_ids.extend_from_slice(row.request.prompt_targets(position));
                prompt_target_indptr.push(prompt_target_ids.len() as u32);
            }
        }
        if let Err(error) = match prompt_logprobs {
            Some(scores) => scores.validate_prompt(
                &prompt_counts,
                &prompt_top_counts,
                &prompt_target_indptr,
                &prompt_target_ids,
            ),
            None if prompt_counts.iter().any(|&count| count > 0) => Err(Error::execution(
                "runner did not return requested prompt logprobs",
            )),
            None => Ok(()),
        } {
            return self.quarantine(batch, "invalid prompt scores", error);
        }
        // Prepare owned scores before cleanup; publish them with logical progress.
        let prompt_updates = prompt_counts
            .iter()
            .enumerate()
            .map(|(row, &count)| {
                (0..count)
                    .map(|column| prompt_logprobs.unwrap().token(row, column))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut retired = Vec::with_capacity(batch.rows.len());
        let mut retired_ids = Vec::new();
        let event_capacity = output_token_ids.token_count();
        let mut events = Vec::with_capacity(event_capacity);
        let mut committed_token_ids = Vec::with_capacity(batch.rows.len());
        for (row_index, (row, token_ids)) in
            batch.rows.iter().zip(output_token_ids.iter()).enumerate()
        {
            let request = &row.request;
            if row.terminal {
                // This physical row ran after its previous result reached a
                // stop token. Validate its execution above, then ignore it.
                retired.push(true);
                retired_ids.push(request.id);
                committed_token_ids.push(Vec::new());
                continue;
            }
            if !row.scheduled.sample {
                let finished = request.max_output_tokens == 0
                    && row.scheduled.token_end() == Some(request.prompt_length());
                retired.push(finished);
                if finished {
                    retired_ids.push(request.id);
                }
                committed_token_ids.push(token_ids.to_vec());
                continue;
            }
            let remaining = request
                .max_output_tokens
                .saturating_sub(request.generated.len());
            let mut committed = Vec::with_capacity(token_ids.len().min(remaining));
            let mut finished = false;
            for (token_index, &token_id) in token_ids.iter().take(remaining).enumerate() {
                let finish_reason = if request.stop_token_ids.binary_search(&token_id).is_ok() {
                    Some(CompletionReason::Stop)
                } else if request.generated.len() + committed.len() + 1 >= request.max_output_tokens
                {
                    Some(CompletionReason::Length)
                } else {
                    None
                };
                finished = finish_reason.is_some();
                let event = match execution_call("event conversion", || {
                    make_event(TokenEvent {
                        request_id: request.id,
                        token_id,
                        index: request.generated.len() + committed.len(),
                        finished,
                        finish_reason,
                        logprobs: logprobs.and_then(|scores| scores.token(row_index, token_index)),
                    })
                }) {
                    Ok(event) => event,
                    Err(error) => {
                        return self.quarantine(batch, "token event construction failed", error);
                    }
                };
                events.push(event);
                committed.push(token_id);
                if finished {
                    break;
                }
            }
            retired.push(finished);
            if finished {
                retired_ids.push(request.id);
            }
            committed_token_ids.push(committed);
        }

        let prepared_progress =
            match Self::prepare_scheduled_progress(&mut batch.rows, &committed_token_ids) {
                Ok(prepared) => prepared,
                Err(error) => {
                    return self.quarantine(batch, "native progress commit failed", error);
                }
            };

        // Validate every physical retirement and reserve all destination
        // storage before asking the runner or any allocator to release state.
        for (row, retire) in batch.rows.iter().zip(&retired) {
            if !*retire || continuation.is_some() {
                continue;
            }
            if let Err(error) = self.validate_request_cache(&row.request) {
                return self.quarantine(batch, "native cache retirement failed", error);
            }
        }
        let mut survivors = Vec::new();
        if continuation.is_none()
            && survivors
                .try_reserve_exact(batch.rows.len().saturating_sub(retired_ids.len()))
                .is_err()
        {
            let error = Error::memory("could not reserve surviving request storage");
            return self.quarantine(batch, "native survivor storage failed", error);
        }

        if continuation.is_none()
            && !retired_ids.is_empty()
            && let Err(error) = execution_call("release", || runner.release(&retired_ids))
        {
            return self.quarantine(batch, "runner release failed", error);
        }
        if continuation.is_none() {
            // Runner cleanup proves physical execution is terminal. Only now
            // may its writer lease return and cache pages become reusable.
            self.release_kv_access(batch.resources.take());
            if let Err(error) =
                self.publish_batch_prefixes(&mut batch, &prepared_progress, output_token_ids)
            {
                return self.quarantine(batch, "native prefix publication failed", error);
            }
        }

        for (row, retire) in batch.rows.iter_mut().zip(&retired) {
            if !*retire || continuation.is_some() {
                continue;
            }
            if let Err(error) = self.retire_request_cache(&mut row.request) {
                return self.quarantine(
                    batch,
                    "native cache retirement failed after preflight",
                    error,
                );
            }
        }

        // Logical progress becomes visible only after every fallible physical
        // cleanup step has completed.
        let committed_at = self.metrics.as_ref().map(|_| Instant::now());
        if let Some(metrics) = &mut self.metrics {
            let now = committed_at.expect("metrics-enabled commit has a timestamp");
            let prepared_at = batch
                .prepared_at
                .expect("metrics-enabled batch has a timestamp");
            for ((row, progress), &finished) in
                batch.rows.iter_mut().zip(&prepared_progress).zip(&retired)
            {
                if row.terminal {
                    continue;
                }
                metrics.commit(
                    row.request
                        .metrics
                        .as_mut()
                        .expect("metrics-enabled request"),
                    prepared_at,
                    now,
                    if row.scheduled.phase == Phase::Prefill {
                        row.scheduled.token_count().get()
                    } else {
                        0
                    },
                    progress.sampled_tokens.len(),
                    finished,
                );
            }
        }
        Self::apply_scheduled_progress(&mut batch.rows, prepared_progress);
        for (row, updates) in batch.rows.iter().zip(prompt_updates) {
            if updates.is_empty() {
                continue;
            }
            let mut state = row
                .request
                .lease
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let scores = state
                .prompt_logprobs
                .as_mut()
                .expect("scored request owns score storage");
            let start = row.scheduled.token_start() as usize + 1;
            scores[start..start + updates.len()].clone_from_slice(&updates);
        }
        if let Some(mut template) = continuation {
            for (slot, (row, retire)) in batch.rows.iter_mut().zip(retired).enumerate() {
                if retire {
                    complete_request_lease(
                        &row.request.lease,
                        std::mem::take(&mut row.request.generated),
                        false,
                    );
                    row.terminal = true;
                } else {
                    let position = row.request.resident_tokens;
                    row.scheduled.tokens = TokenRange {
                        start: position,
                        count: NonZeroU32::MIN,
                    };
                    template.context_lengths[slot] = position;
                    template.token_ids[slot] = *row
                        .request
                        .generated
                        .last()
                        .expect("decode continuation has generated tokens");
                }
            }
            self.in_flight = Some(batch);
            self.retain_continuation(&template);
        } else {
            for (row, retire) in batch.rows.drain(..).zip(retired) {
                let request = row.request;
                if retire {
                    if !row.terminal {
                        complete_request_lease(&request.lease, request.generated, false);
                    }
                } else {
                    survivors.push(request);
                }
            }
            self.scheduler.requeue_survivors(survivors);
        }
        if decoded_batch {
            self.decay_new_token_ratio();
        }
        let idle = !self.has_work();
        if let Some(metrics) = &mut self.metrics {
            metrics.set_idle(
                idle,
                committed_at.expect("metrics-enabled commit has a timestamp"),
            );
        }
        Ok(events)
    }

    pub fn cancel(&mut self, runner: &mut impl Runner<P>, request: &RequestHandle) -> Result<bool> {
        self.raise_if_open()?;
        self.require_completed_step()?;
        if request.engine_id != self.engine_id {
            return Err(Error::invalid("RequestLease belongs to a different Engine"));
        }
        let request_id = request.request_id;
        if let Some(index) = self.scheduler.position(|request| request.id == request_id) {
            let mut request = self.scheduler.remove(index);
            if let Err(error) = self.validate_request_cache(&request) {
                self.scheduler.restore(index, request);
                self.poison("native cache cancellation retirement failed", &error);
                return Err(error);
            }
            if let Err(error) = execution_call("release", || runner.release(&[request_id])) {
                self.scheduler.restore(index, request);
                self.poison("runner release failed during ready cancellation", &error);
                return Err(error);
            }
            if let Err(error) = self.retire_request_cache(&mut request) {
                self.scheduler.restore(index, request);
                self.poison(
                    "native cache cancellation retirement failed after preflight",
                    &error,
                );
                return Err(error);
            }
            let generated = std::mem::take(&mut request.generated);
            complete_request_lease(&request.lease, generated, true);
            let idle = !self.has_work();
            if let Some(metrics) = &mut self.metrics {
                metrics.abort();
                metrics.set_idle(idle, Instant::now());
            }
            return Ok(true);
        }

        Ok(false)
    }

    pub fn close(&mut self, runner: &mut impl Runner<P>) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.raise_if_poisoned()?;
        self.require_completed_step()?;
        self.close_healthy(runner)
    }
}

impl<P: Clone> EngineCore<P> {
    fn prefix_alignment(&self) -> Option<u32> {
        self.kv_resources
            .as_ref()
            .and_then(PagedKvResources::prefix_alignment)
    }

    fn publish_batch_prefixes(
        &mut self,
        batch: &mut Batch<P>,
        progress: &[PreparedProgress],
        physical_output: TokenRows<'_>,
    ) -> Result<()> {
        let Some(alignment) = self.prefix_alignment() else {
            return Ok(());
        };
        let has_recurrent_state = self
            .runner_config
            .cache_layout
            .as_ref()
            .is_some_and(|layout| layout.groups().iter().any(|group| group.is_recurrent()));
        for ((row, progress), physical_tokens) in batch
            .rows
            .iter_mut()
            .zip(progress)
            .zip(physical_output.iter())
        {
            if row.terminal
                || !row.request.cacheable
                || progress.resident_tokens == 0
                || progress.resident_tokens % alignment != 0
            {
                continue;
            }
            // EOS/output limits can shorten a speculative result after the
            // runner committed the whole accepted block. STATE groups need
            // only store its final checkpoint, so the shortened aligned
            // history has no guaranteed matching recurrent state. History-only
            // caches still contain valid token-addressed entries for it.
            if has_recurrent_state
                && row.scheduled.phase == Phase::Decode
                && progress.sampled_tokens.len() < physical_tokens.len()
            {
                continue;
            }
            let resident = progress.resident_tokens as usize;
            let mut tokens = Vec::new();
            tokens
                .try_reserve_exact(
                    row.request
                        .prompt
                        .len()
                        .saturating_add(row.request.generated.len())
                        .saturating_add(progress.sampled_tokens.len()),
                )
                .map_err(|_| Error::memory("could not publish prefix token identity"))?;
            tokens.extend_from_slice(
                row.request
                    .cache_prompt
                    .as_deref()
                    .unwrap_or(&row.request.prompt),
            );
            tokens.extend_from_slice(&row.request.generated);
            tokens.extend_from_slice(&progress.sampled_tokens);
            if resident > tokens.len() {
                return Err(Error::internal(
                    "resident KV progress exceeds the request token history",
                ));
            }
            tokens.truncate(resident);
            let resources = self
                .kv_resources
                .as_mut()
                .expect("prefix alignment requires managed cache resources");
            let pages = row
                .request
                .cache
                .as_mut()
                .expect("prefix-enabled requests own managed cache pages");
            resources
                .publish_prefix(pages, &tokens, progress.resident_tokens)
                .map_err(Error::from)?;
        }
        Ok(())
    }

    fn estimated_context(&self, request: &Request<P>) -> u32 {
        let future_inputs = request.max_output_tokens.saturating_sub(1);
        let estimated_future = ((future_inputs as f64) * self.new_token_ratio).ceil() as usize;
        let estimate = request
            .prompt
            .len()
            .saturating_add(estimated_future)
            .max(
                request
                    .prefill_length()
                    .expect("active request length is valid") as usize,
            )
            .min(self.context_length.get() as usize);
        estimate as u32
    }

    fn admission_prefix_len(&self, selected: &[&Request<P>]) -> Result<usize> {
        let Some(resources) = &self.kv_resources else {
            return Ok(selected.len());
        };
        if selected.iter().all(|request| request.is_running()) {
            return Ok(selected.len());
        }

        let mut estimate = resources.admission_estimate();
        let running = self
            .scheduler
            .iter()
            .filter(|request| request.is_running())
            .collect::<Vec<_>>();
        for request in &running {
            estimate.include(request.cache.as_ref(), self.estimated_context(request))?;
        }
        let running_count = running.len();
        let selected_running = selected
            .iter()
            .take_while(|request| request.is_running())
            .count();
        let mut admitted = selected_running;
        for request in &selected[selected_running..] {
            let parents = estimate
                .include(request.cache.as_ref(), self.estimated_context(request))
                .map_err(Error::from)?;
            if parents > u64::from(resources.parent_capacity())
                && (running_count != 0 || admitted != 0)
            {
                break;
            }
            admitted += 1;
        }
        Ok(admitted.max(usize::from(running_count == 0 && !selected.is_empty())))
    }

    fn decay_new_token_ratio(&mut self) {
        self.new_token_ratio =
            (self.new_token_ratio - self.new_token_ratio_decay).max(self.minimum_new_token_ratio);
    }

    fn update_new_token_ratio_after_retraction(&mut self) {
        let running = self.scheduler.iter().filter(|request| request.is_running());
        let (count, generated, limits) = running.fold(
            (0usize, 0usize, 0usize),
            |(count, generated, limits), request| {
                (
                    count + 1,
                    generated.saturating_add(request.generated.len()),
                    limits.saturating_add(request.max_output_tokens),
                )
            },
        );
        self.new_token_ratio = if count == 0 {
            self.scheduler_config.initial_new_token_ratio()
        } else {
            ((generated.saturating_add(RETRACTION_DECODE_HEADROOM * count)) as f64
                / limits.saturating_add(1) as f64)
                .min(1.0)
        };
    }

    fn retract_one(&mut self, runner: &mut impl Runner<P>) -> Result<bool> {
        let running_count = self
            .scheduler
            .iter()
            .filter(|request| request.is_running())
            .count();
        if running_count <= 1 {
            return Ok(false);
        }
        let Some(request_id) = self
            .scheduler
            .iter()
            .filter(|request| request.is_running() && request.cache.is_some())
            .min_by_key(|request| (request.generated.len(), Reverse(request.prompt.len())))
            .map(|request| request.id)
        else {
            return Ok(false);
        };
        let index = self
            .scheduler
            .position(|request| request.id == request_id)
            .expect("selected retraction victim remains ready");
        let mut request = self.scheduler.remove(index);

        if let Err(error) = execution_call("release", || runner.release(&[request.id])) {
            self.scheduler.restore(index, request);
            self.poison("runner release failed during request retraction", &error);
            return Err(error);
        }
        let pages = request
            .cache
            .as_mut()
            .expect("retraction victim owns managed cache state");
        if let Err(error) = self
            .kv_resources
            .as_mut()
            .expect("managed cache state keeps its allocator")
            .reset(pages)
        {
            self.scheduler.restore(index, request);
            let error = Error::from(error);
            self.poison(
                "native cache reset failed during request retraction",
                &error,
            );
            return Err(error);
        }
        request.resident_tokens = 0;
        if let Some(metrics) = &mut self.metrics {
            metrics.retract();
        }
        self.scheduler.requeue_retracted(request);
        self.update_new_token_ratio_after_retraction();
        Ok(true)
    }

    pub fn raise_if_open(&self) -> Result<()> {
        if self.closed {
            return Err(Error::internal("Engine is closed"));
        }
        self.raise_if_poisoned()
    }

    fn require_completed_step(&self) -> Result<()> {
        if self.in_flight.is_some() {
            return Err(Error::internal(
                "Engine must drain its in-flight step first",
            ));
        }
        Ok(())
    }

    fn raise_if_poisoned(&self) -> Result<()> {
        if let Some(error) = &self.poisoned_error {
            return Err(Error::internal(format!(
                "Engine is poisoned and its resources are quarantined: {error}"
            )));
        }
        Ok(())
    }

    fn poison(&mut self, stage: &str, error: &Error) {
        if self.poisoned_error.is_none() {
            self.poisoned_error = Some(format!("{stage}: {error}"));
        }
    }

    pub fn poison_message(&mut self, message: impl Into<String>) {
        if self.poisoned_error.is_none() {
            self.poisoned_error = Some(message.into());
        }
    }

    fn quarantine<T>(&mut self, batch: Batch<P>, stage: &str, error: Error) -> Result<T> {
        self.quarantined = Some(batch);
        self.poison(stage, &error);
        Err(error)
    }

    fn close_healthy(&mut self, runner: &mut impl Runner<P>) -> Result<()> {
        self.retire_ready(runner)?;
        if let Some(resources) = self.kv_resources.as_mut() {
            resources.clear_prefix_cache().map_err(Error::from)?;
        }
        self.closed = true;
        if let Some(metrics) = &mut self.metrics {
            metrics.close();
        }
        Ok(())
    }

    fn retire_ready(&mut self, runner: &mut impl Runner<P>) -> Result<()> {
        if self.scheduler.is_empty() {
            return Ok(());
        }
        let request_ids = self
            .scheduler
            .iter()
            .map(|request| request.id)
            .collect::<Vec<_>>();
        let retirement_validation = self
            .scheduler
            .iter()
            .try_for_each(|request| self.validate_request_cache(request));
        if let Err(error) = retirement_validation {
            self.poison("native cache retirement failed during Engine close", &error);
            return Err(error);
        }
        let mut retired = Vec::new();
        retired
            .try_reserve_exact(self.scheduler.len())
            .map_err(|_| Error::memory("could not reserve ready request retirement"))?;

        if let Err(error) = execution_call("release", || runner.release(&request_ids)) {
            self.poison("runner release failed during Engine close", &error);
            return Err(error);
        }
        retired.extend(self.scheduler.drain());
        for mut request in retired {
            if let Err(error) = self.retire_request_cache(&mut request) {
                self.poison(
                    "native cache retirement failed after close preflight",
                    &error,
                );
                return Err(error);
            }
            let generated = std::mem::take(&mut request.generated);
            complete_request_lease(&request.lease, generated, true);
            if let Some(metrics) = &mut self.metrics {
                metrics.abort();
            }
        }
        Ok(())
    }

    fn validate_request_cache(&self, request: &Request<P>) -> Result<()> {
        match (self.kv_resources.as_ref(), request.cache.as_ref()) {
            (Some(resources), Some(pages)) => resources
                .pool()
                .validate_request(pages)
                .map_err(Error::from),
            (None, None) => Ok(()),
            _ => Err(Error::internal(
                "request and Engine cache ownership disagree",
            )),
        }
    }

    fn retire_request_cache(&mut self, request: &mut Request<P>) -> Result<()> {
        if let Some(pages) = request.cache.take()
            && let Err(failure) = self
                .kv_resources
                .as_mut()
                .expect("request KV pages keep their resources")
                .retire(pages)
        {
            request.cache = Some(failure.request);
            return Err(Error::from(failure.error));
        }
        Ok(())
    }

    fn prepare_scheduled_progress(
        rows: &mut [BatchRow<P>],
        output_token_ids: &[Vec<u32>],
    ) -> Result<Vec<PreparedProgress>> {
        if rows.len() != output_token_ids.len() {
            return Err(Error::internal(
                "dense runner output does not cover every scheduled row",
            ));
        }
        let mut prepared = Vec::with_capacity(rows.len());
        for (row, token_ids) in rows.iter_mut().zip(output_token_ids) {
            let request = &mut row.request;
            if row.terminal {
                prepared.push(PreparedProgress {
                    resident_tokens: request.resident_tokens,
                    sampled_tokens: Vec::new(),
                });
                continue;
            }
            let scheduled = row.scheduled;
            let token_end = request.validate_step(scheduled)?;
            if token_ids.is_empty() {
                return Err(Error::internal(
                    "sampled runner output rows must contain at least one token",
                ));
            }
            if scheduled.sample {
                request
                    .generated
                    .try_reserve(token_ids.len())
                    .map_err(|_| Error::memory("could not reserve generated-token storage"))?;
            }
            let resident_tokens = if scheduled.phase == Phase::Decode {
                let additional = u32::try_from(token_ids.len() - 1)
                    .map_err(|_| Error::invalid("speculative token row is too long"))?;
                token_end
                    .checked_add(additional)
                    .ok_or_else(|| Error::invalid("request token position space is exhausted"))?
            } else {
                token_end
            };
            prepared.push(PreparedProgress {
                resident_tokens,
                sampled_tokens: if scheduled.sample {
                    token_ids.clone()
                } else {
                    Default::default()
                },
            });
        }
        Ok(prepared)
    }

    fn apply_scheduled_progress(rows: &mut [BatchRow<P>], prepared: Vec<PreparedProgress>) {
        assert_eq!(rows.len(), prepared.len());
        for (row, progress) in rows.iter_mut().zip(prepared) {
            row.request.resident_tokens = progress.resident_tokens;
            row.request.generated.extend(progress.sampled_tokens);
        }
    }

    fn validate_runner_output(
        &self,
        rows: &[BatchRow<P>],
        token_rows: TokenRows<'_>,
    ) -> Result<()> {
        if rows.len() != token_rows.len() {
            return Err(Error::internal(format!(
                "runner returned {} token rows for {} sequences",
                token_rows.len(),
                rows.len()
            )));
        }
        for (row, tokens) in rows.iter().zip(token_rows.iter()) {
            if tokens.is_empty() {
                return Err(Error::internal("runner returned an empty token row"));
            }
            if row.scheduled.phase == Phase::Prefill && tokens.len() != 1 {
                return Err(Error::internal(
                    "speculative token rows are only supported during decode",
                ));
            }
            let maximum = u64::from(self.runner_config.cache_write_lookahead_tokens) + 1;
            if row.request.cache.is_some() && tokens.len() as u64 > maximum {
                let limit = if self.runner_config.cache_write_reservation_tokens > 0 {
                    format!("declared output maximum is {maximum}")
                } else {
                    format!("leased cache space for at most {maximum}")
                };
                return Err(Error::internal(format!(
                    "runner returned {} decode tokens but {limit}",
                    tokens.len(),
                )));
            }
        }
        Ok(())
    }

    fn release_kv_access(&mut self, access: Option<WriterAccessLease>) {
        if let Some(access) = access {
            self.kv_resources
                .as_mut()
                .expect("writer access keeps its KV resources")
                .release(access);
        }
    }
}
fn build_plan<P: Clone>(
    batch_id: u64,
    requests: &[Request<P>],
    scheduled_rows: &[ScheduledRow],
    kv_view: Option<&mut PagedKvBatchView>,
) -> Result<ExecutionPlan<P>> {
    if requests.len() != scheduled_rows.len() {
        return Err(Error::internal(
            "scheduled row metadata does not cover every request",
        ));
    }
    if requests.is_empty() {
        return Err(Error::invalid("cannot build an empty batch"));
    }
    if requests
        .iter()
        .any(|request| request.cache.is_some() != kv_view.is_some())
    {
        return Err(Error::internal(
            "scheduled request and KV resource view ownership disagree",
        ));
    }
    let batch_key = requests.first().and_then(|request| request.batch_key);
    if requests
        .iter()
        .any(|request| request.batch_key != batch_key)
    {
        return Err(Error::internal(
            "scheduler selected requests with different batch keys",
        ));
    }
    let phase = scheduled_rows[0].phase;
    if scheduled_rows.iter().any(|row| row.phase != phase) {
        return Err(Error::internal(
            "scheduler selected requests from different phases",
        ));
    }

    let mut token_ids = Vec::new();
    let mut sequence_ids = Vec::with_capacity(requests.len());
    let mut token_indptr = Vec::with_capacity(requests.len() + 1);
    let mut context_lengths = Vec::with_capacity(requests.len());
    let mut prefill_lengths = Vec::with_capacity(requests.len());
    let mut payloads = Vec::with_capacity(requests.len());
    let mut sampling_mask = Vec::with_capacity(requests.len());
    let mut prompt_score_indices = Vec::new();
    let mut prompt_score_token_ids = Vec::new();
    let mut prompt_score_indptr = vec![0];
    let mut prompt_score_top_counts = Vec::new();
    let mut prompt_target_indptr = Vec::new();
    let mut prompt_target_token_ids = Vec::new();
    token_indptr.push(0);

    for (request, scheduled) in requests.iter().zip(scheduled_rows) {
        sequence_ids.push(request.id);
        payloads.push(request.payload.clone());
        let token_end = request.validate_step(*scheduled)?;
        for position in request.prompt_score_range(*scheduled) {
            if prompt_target_indptr.is_empty() {
                prompt_target_indptr.push(0);
            }
            prompt_score_indices
                .push((token_ids.len() + position - scheduled.token_start() as usize) as u32);
            prompt_score_token_ids.push(request.prompt[position + 1]);
            prompt_score_top_counts.push(request.prompt_top_logprobs);
            prompt_target_token_ids.extend_from_slice(request.prompt_targets(position));
            prompt_target_indptr.push(u32::try_from(prompt_target_token_ids.len()).map_err(
                |_| Error::invalid("batch prompt target count exceeds u32 metadata range"),
            )?);
        }
        prompt_score_indptr.push(prompt_score_indices.len() as u32);
        match scheduled.phase {
            Phase::Prefill => {
                let start = scheduled.token_start() as usize;
                let end = token_end as usize;
                let prompt_end = end.min(request.prompt.len());
                if start < prompt_end {
                    token_ids.extend_from_slice(&request.prompt[start..prompt_end]);
                }
                if end > request.prompt.len() {
                    let generated_start = start.saturating_sub(request.prompt.len());
                    let generated_end = end - request.prompt.len();
                    token_ids.extend_from_slice(&request.generated[generated_start..generated_end]);
                }
            }
            Phase::Decode => {
                token_ids.push(*request.generated.last().ok_or_else(|| {
                    Error::internal("scheduled decode request has no generated token")
                })?);
            }
        }
        context_lengths.push(scheduled.token_start());
        prefill_lengths.push(request.prefill_length()?);
        sampling_mask.push(scheduled.sample);
        token_indptr.push(
            u32::try_from(token_ids.len())
                .map_err(|_| Error::invalid("batch token count exceeds the u32 CSR range"))?,
        );
    }

    // Transfer the already-frozen vectors after every fallible validation and
    // payload clone. The batch keeps the independent lease through execution.
    let cache_groups = kv_view.map_or_else(Vec::new, |view| std::mem::take(&mut view.groups));
    Ok(ExecutionPlan {
        prompt_score_indices,
        prompt_score_token_ids,
        prompt_score_indptr,
        prompt_score_top_counts,
        prompt_target_indptr,
        prompt_target_token_ids,
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
        payloads,
    })
}

fn schedule_requests<P>(
    requests: &[&Request<P>],
    config: PlanningConfig,
) -> Result<Vec<SequenceStep>> {
    let progress = requests
        .iter()
        .map(|request| {
            Ok(RequestProgress {
                phase: request.phase(),
                prefill_length: request.prefill_length()?,
                logical_context: request.logical_context()?,
                sample_at_prefill_end: request.sample_at_prefill_end(),
                atomic_prefill_spans: request.atomic_prefill_spans.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(schedule_ready_prefix(&progress, config)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::kv::{KvGroupLayout, KvRetention};

    fn request(
        id: u64,
        prompt: &[u32],
        generated: &[u32],
        pages: Option<RequestPages>,
    ) -> Request<()> {
        Request {
            metrics: None,
            prompt_logprobs: false,
            prompt_top_logprobs: 0,
            prompt_target_logprobs: None,
            id,
            lease: Arc::new(Mutex::new(RequestLeaseState::default())),
            priority: 0,
            batch_key: None,
            turn_ordinal: 0,
            running: false,
            prompt: prompt.to_vec(),
            cache_prompt: None,
            atomic_prefill_spans: Vec::new(),
            resident_tokens: if generated.is_empty() {
                0
            } else {
                (prompt.len() + generated.len() - 1) as u32
            },
            generated: generated.to_vec(),
            max_output_tokens: 8,
            stop_token_ids: Vec::new(),
            payload: (),
            cacheable: true,
            cache: pages,
        }
    }

    fn history_layout() -> KvLayout {
        KvLayout::new(
            8,
            vec![
                KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap(),
                KvGroupLayout::new(2, 2, KvRetention::FullContext).unwrap(),
            ],
        )
        .unwrap()
    }
    fn window_layout() -> KvLayout {
        KvLayout::new(
            4,
            vec![
                KvGroupLayout::new(
                    4,
                    1,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(4).unwrap(),
                    },
                )
                .unwrap()
                .with_read_window_tokens(1)
                .unwrap(),
            ],
        )
        .unwrap()
    }
    fn planning(config: SchedulerConfig) -> PlanningConfig {
        PlanningConfig {
            max_batch_tokens: config.max_batch_tokens,
            prefill_chunk_size: config.prefill_chunk_size,
            checkpoint_alignment: None,
        }
    }
    #[test]
    fn heterogeneous_phase_schedule_is_rejected() {
        let prefill = request(1, &[10, 11, 12], &[], None);
        let decode = request(2, &[20, 21], &[99], None);
        let scheduled = [&prefill, &decode];
        let error =
            schedule_requests(&scheduled, planning(SchedulerConfig::default())).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("scheduler selected requests from different phases")
        );
    }

    #[test]
    fn retracted_history_endpoint_is_known_before_the_first_replay_chunk() {
        let mut replay = request(1, &[10, 11, 12], &[20, 21, 22, 23], None);
        replay.resident_tokens = 0;
        let mut config = planning(SchedulerConfig::default());
        config.prefill_chunk_size = NonZeroU32::new(2).unwrap();
        for (start, expected_tokens) in [(0, vec![10, 11]), (2, vec![12, 20]), (4, vec![21, 22])] {
            let rows = schedule_requests(&[&replay], config).unwrap();
            let plan = build_plan(1, std::slice::from_ref(&replay), &rows, None).unwrap();
            assert_eq!(plan.context_lengths, vec![start]);
            assert_eq!(plan.prefill_lengths, vec![6]);
            assert_eq!(plan.token_ids, expected_tokens);
            assert_eq!(plan.sampling_mask, vec![false]);
            replay.resident_tokens += 2;
        }
        assert_eq!(replay.phase(), Phase::Decode);
    }

    #[test]
    fn malformed_rows_are_rejected_before_execution_and_progress_commit() {
        for (decode, phase, start, count, sample) in [
            (false, Phase::Prefill, 1, 2, true),
            (false, Phase::Decode, 0, 1, true),
            (false, Phase::Prefill, 0, 4, true),
            (false, Phase::Prefill, 0, 1, true),
            (true, Phase::Decode, 3, 2, true),
            (true, Phase::Decode, 3, 1, false),
        ] {
            let request = request(1, &[1, 2, 3], if decode { &[9] } else { &[] }, None);
            let scheduled = SequenceStep {
                phase,
                tokens: crate::TokenRange {
                    start,
                    count: NonZeroU32::new(count).unwrap(),
                },
                sample,
            };
            assert!(build_plan(1, std::slice::from_ref(&request), &[scheduled], None).is_err());
            let mut rows = [BatchRow {
                request,
                scheduled,
                terminal: false,
            }];
            assert!(EngineCore::<()>::prepare_scheduled_progress(&mut rows, &[vec![9]]).is_err());
        }
    }

    #[test]
    fn history_plan_is_sparse_and_writer_claims_match_active_pages() {
        let layout = history_layout();
        let mut resources = PagedKvResources::new(layout.clone()).unwrap();
        let pages = resources.attach(core_request_id(41), 10).unwrap();
        let mut managed = request(41, &[1, 2, 3, 4, 5, 6], &[], Some(pages));
        managed.resident_tokens = 4;
        let scheduled = [&managed];
        let rows = schedule_requests(&scheduled, planning(SchedulerConfig::default())).unwrap();
        let resource_refs = [RequestResourceRef {
            request_id: core_request_id(41),
            state: managed.cache.as_ref().unwrap(),
        }];
        let mut acquired = resources.acquire(&resource_refs, &rows, 0).unwrap();
        let groups_ptr = acquired.view.groups.as_ptr();
        let block_ids_ptr = acquired.view.groups[0].block_slab_ids.as_ptr();
        let plan = build_plan(
            9,
            std::slice::from_ref(&managed),
            &rows,
            Some(&mut acquired.view),
        )
        .unwrap();

        assert!(acquired.view.groups.is_empty());
        assert_eq!(plan.cache_groups.as_ptr(), groups_ptr);
        assert_eq!(plan.cache_groups[0].block_slab_ids.as_ptr(), block_ids_ptr);

        assert_eq!(plan.token_ids, vec![5, 6]);
        assert_eq!(plan.context_lengths, vec![4]);
        assert_eq!(plan.prefill_lengths, vec![6]);
        assert_eq!(plan.cache_groups.len(), 2);
        let full = &plan.cache_groups[0];
        assert_eq!(full.block_indptr, [0, 2]);
        assert_eq!(full.block_slab_ids.len(), 2);
        assert_eq!(full.block_table_start_indices, [0]);
        assert_eq!(full.append_block_offsets, [0, 1]);
        let fine = &plan.cache_groups[1];
        assert_eq!(fine.block_indptr, [0, 3]);
        assert_eq!(fine.block_slab_ids.len(), 3);
        assert_eq!(fine.append_block_offsets, [0, 1]);

        assert_eq!(acquired.lease.len(), 5);
        resources.release(acquired.lease);

        let pages = managed.cache.take().unwrap();
        resources.pool().validate_request(&pages).unwrap();
        resources.retire(pages).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 0);
        assert_eq!(resources.pool().free_parent_count(), 8);
    }

    #[test]
    fn window_plan_is_chronological_and_reuses_only_an_evicted_block() {
        let layout = window_layout();
        let mut resources = PagedKvResources::new(layout.clone()).unwrap();
        let pages = resources.attach(core_request_id(51), 12).unwrap();
        let physical = pages.groups()[0]
            .pages()
            .iter()
            .map(|page| resources.pool().physical_address(page.handle()).unwrap())
            .collect::<Vec<_>>();
        let mut managed = request(
            51,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            &[],
            Some(pages),
        );
        managed.resident_tokens = 8;
        let scheduled = [&managed];
        let rows = schedule_requests(&scheduled, planning(SchedulerConfig::default())).unwrap();
        assert_eq!(rows[0].token_start(), 8);
        assert_eq!(rows[0].token_count().get(), 4);

        let resource_refs = [RequestResourceRef {
            request_id: core_request_id(51),
            state: managed.cache.as_ref().unwrap(),
        }];
        let mut acquired = resources.acquire(&resource_refs, &rows, 0).unwrap();
        let plan = build_plan(
            10,
            std::slice::from_ref(&managed),
            &rows,
            Some(&mut acquired.view),
        )
        .unwrap();
        let window = &plan.cache_groups[0];
        assert_eq!(window.block_table_start_indices, [1]);
        assert_eq!(window.block_indptr, [0, 2]);
        assert_eq!(window.block_slab_ids, [physical[1].0, physical[0].0]);
        assert_eq!(window.block_slotnums, [physical[1].1, physical[0].1]);
        assert_eq!(
            window.append_slab_ids,
            [physical[0].0, physical[0].0, physical[0].0, physical[0].0]
        );
        assert_eq!(
            window.append_slotnums,
            [physical[0].1, physical[0].1, physical[0].1, physical[0].1]
        );
        assert_eq!(window.append_block_offsets, [0, 1, 2, 3]);

        assert_eq!(acquired.lease.len(), 2);
        resources.release(acquired.lease);

        resources.retire(managed.cache.take().unwrap()).unwrap();
        assert_eq!(resources.pool().occupied_block_count(), 0);
    }

    #[test]
    fn window_prefill_can_cross_logical_block_boundaries() {
        let mut first = request(1, &[1, 2, 3, 4, 5, 6], &[], None);
        first.resident_tokens = 3;
        let second = request(2, &[7, 8, 9, 10, 11, 12], &[], None);
        let scheduled = [&first, &second];
        let rows = schedule_requests(&scheduled, planning(SchedulerConfig::default())).unwrap();
        assert_eq!(rows[0].token_count().get(), 3);
        assert_eq!(rows[1].token_count().get(), 6);
    }

    #[test]
    fn token_budget_and_prefill_chunk_size_still_compose() {
        let first = request(1, &[1, 2, 3, 4], &[], None);
        let second = request(2, &[5, 6, 7, 8], &[], None);
        let scheduled = [&first, &second];
        let config = SchedulerConfig {
            max_batch_tokens: NonZeroU32::new(3),
            prefill_chunk_size: NonZeroU32::new(2).expect("nonzero test chunk"),
            ..SchedulerConfig::default()
        };
        let rows = schedule_requests(&scheduled, planning(config)).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].token_count().get(), 2);
        assert_eq!(rows[1].token_count().get(), 1);
        assert!(!rows[0].sample);
        assert!(!rows[1].sample);
    }
}
