//! Shared request lifecycle for local serving and routed attention-DP drivers.
use crate::http_distributed::{HttpEvents, HttpIngress};
use crate::http_logprobs::Vocabulary;
use crate::http_output::OutputScores;
use crate::http_stop::StopFilter;
use crate::recipe::{self, NativeRecipe, NativeResponseParser, RenderedPrompt, SharedParser};
use crate::recipe_events;
use crate::recipe_types::{Message, RenderOptions, Tool};
use crate::{CompletionReason, Engine, RequestLease, SamplingParams, TokenEvent};
use pyo3::{
    exceptions::{PyAttributeError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyTuple},
};
use pythonize::pythonize;
use rayon::{ThreadPool, ThreadPoolBuilder};
use resin_http::{
    ConversationRequest, EVENT_QUEUE_CAPACITY, FinishReason, GeneratedTokenLogprob,
    GenerationEvent, GenerationJob,
};
use resin_recipe::{AttributedEvent, Event as ParsedEvent, FinishReason as ParserFinishReason};
use std::{
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc as std_mpsc,
    },
    thread,
    time::Instant,
};
use tokio::sync::mpsc;

fn into_http(event: ParsedEvent) -> GenerationEvent {
    match event {
        ParsedEvent::ReasoningDelta { delta } => GenerationEvent::ReasoningDelta(delta),
        ParsedEvent::TextDelta { delta } => GenerationEvent::TextDelta {
            delta,
            logprobs: Vec::new(),
        },
        ParsedEvent::ToolCallStart { index, id, name } => {
            GenerationEvent::ToolCallStart { index, id, name }
        }
        ParsedEvent::ToolCallArgumentsDelta { index, delta } => {
            GenerationEvent::ToolCallArgumentsDelta { index, delta }
        }
    }
}

enum Parser {
    Native(SharedParser),
    Python {
        object: Py<PyAny>,
        finish: Option<Py<PyAny>>,
    },
}

impl Parser {
    fn new(py: Python<'_>, object: Py<PyAny>) -> PyResult<Self> {
        if let Ok(native) = object.bind(py).cast::<NativeResponseParser>() {
            return Ok(Self::Native(native.borrow().inner.clone()));
        }
        let finish = optional_method(py, &object, "finish")?;
        Ok(Self::Python { object, finish })
    }

    fn token_channel(&self, id: u32) -> Option<&'static str> {
        match self {
            Self::Native(parser) => parser
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .token_channel(id),
            Self::Python { .. } => None,
        }
    }

    fn feed(&self, ids: &[u32]) -> PyResult<Vec<AttributedEvent>> {
        match self {
            Self::Native(parser) => recipe::feed(parser, ids),
            Self::Python { object, .. } => Python::attach(|py| {
                recipe_events::from_python(&object.bind(py).call_method1("feed", (ids,))?)
            }),
        }
    }

    fn finish(&self, reason: ParserFinishReason) -> PyResult<Vec<AttributedEvent>> {
        match self {
            Self::Native(parser) => recipe::finish(parser, reason),
            Self::Python { finish, .. } => Python::attach(|py| {
                let Some(method) = finish else {
                    return Ok(Vec::new());
                };
                let reason = match reason {
                    ParserFinishReason::Stop => "stop",
                    ParserFinishReason::Length => "length",
                };
                recipe_events::from_python(&method.bind(py).call1((reason,))?)
            }),
        }
    }
}

fn optional_method(py: Python<'_>, object: &Py<PyAny>, name: &str) -> PyResult<Option<Py<PyAny>>> {
    match object.bind(py).getattr(name) {
        Ok(method) if method.is_callable() => Ok(Some(method.unbind())),
        Ok(_) => Err(PyValueError::new_err(format!(
            "parser.{name} must be callable"
        ))),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Coalesce protocol-neutral output across steps, without delaying parsing or stops.
#[derive(Default)]
struct StreamOutput {
    events: Vec<GenerationEvent>,
    scores: Vec<GeneratedTokenLogprob>,
    tokens: u32,
    sent_output: bool,
}

impl StreamOutput {
    fn has_output(&self) -> bool {
        self.events.iter().any(|event| match event {
            GenerationEvent::TextDelta { delta, .. }
            | GenerationEvent::ReasoningDelta(delta)
            | GenerationEvent::ToolCallArgumentsDelta { delta, .. } => !delta.is_empty(),
            GenerationEvent::ToolCallStart { .. } => true,
            _ => false,
        })
    }

    fn ready(&self, interval: u32, finished: bool) -> bool {
        finished
            || self.tokens >= interval
            || (!self.sent_output && self.has_output())
            // Bound noncoalescible lanes even with a large configured interval.
            || self.events.len() >= EVENT_QUEUE_CAPACITY / 2
    }

    fn flush(&mut self, mut send: impl FnMut(GenerationEvent) -> bool) -> bool {
        if self.events.is_empty() && self.scores.is_empty() {
            return true;
        }
        let has_output = self.has_output();
        let delivered = self.events.drain(..).all(&mut send)
            && (self.scores.is_empty()
                || send(GenerationEvent::Logprobs(std::mem::take(&mut self.scores))));
        self.sent_output |= has_output && delivered;
        self.tokens = 0;
        delivered
    }
}

struct ActiveRequest {
    http_id: String,
    lease: Py<RequestLease>,
    parser: Parser,
    stops: StopFilter,
    scores: Option<OutputScores>,
    prompt_score_ids: Option<Vec<u32>>,
    sender: Option<mpsc::Sender<GenerationEvent>>,
    started: Instant,
    first_token_at: Option<Instant>,
    input_tokens: usize,
    cached_tokens: usize,
    output_tokens: usize,
    last_source: u32,
    stream: StreamOutput,
}

impl ActiveRequest {
    fn emit(&mut self, event: AttributedEvent, output: &mut Vec<GenerationEvent>) {
        if let Some(scores) = &mut self.scores {
            for event in scores.push(event) {
                push_event(output, event);
            }
        } else {
            push_event(output, into_http(event.event));
        }
    }

    fn parsed(
        &mut self,
        event: AttributedEvent,
        output: &mut Vec<GenerationEvent>,
    ) -> PyResult<()> {
        validate_spans(&event, self.output_tokens, self.scores.as_ref())?;
        if event
            .spans
            .first()
            .is_some_and(|span| span.token < self.last_source)
        {
            return Err(PyValueError::new_err(
                "response-parser token spans moved backwards",
            ));
        }
        if let Some(last) = event.spans.last() {
            self.last_source = last.token;
        }
        for event in self.stops.push(event) {
            self.emit(event, output);
        }
        Ok(())
    }

    fn flush_stops(&mut self, output: &mut Vec<GenerationEvent>) {
        for event in self.stops.finish() {
            self.emit(event, output);
        }
    }
}

type PacketEvents = HashMap<String, Vec<GenerationEvent>>;

fn send_event(
    sender: Option<&mpsc::Sender<GenerationEvent>>,
    packet: &mut PacketEvents,
    id: &str,
    event: GenerationEvent,
) -> bool {
    if let Some(sender) = sender {
        return sender.try_send(event).is_ok();
    }
    let Some(events) = packet.get_mut(id) else {
        packet.insert(id.to_owned(), vec![event]);
        return true;
    };
    if events.len() < EVENT_QUEUE_CAPACITY {
        events.push(event);
        return true;
    }
    if events.len() == EVENT_QUEUE_CAPACITY {
        events.push(GenerationEvent::Error {
            message: "the inference stream ended before completion".to_owned(),
        });
    }
    false
}

struct PendingRender {
    request: ConversationRequest,
    sender: Option<mpsc::Sender<GenerationEvent>>,
    enqueued_at: Instant,
}

pub(crate) fn render_worker_count(requested: Option<u32>) -> usize {
    requested.map_or_else(
        || {
            thread::available_parallelism()
                .map_or(1, usize::from)
                .min(16)
        },
        |value| value as usize,
    )
}

pub(crate) fn render_pool(workers: usize) -> Result<Arc<ThreadPool>, String> {
    ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|worker| format!("resin-render-{worker}"))
        .build()
        .map(Arc::new)
        .map_err(|error| format!("failed to create render pool: {error}"))
}

pub(crate) struct Session {
    engine: Py<Engine>,
    recipe: Py<PyAny>,
    stop_token_ids: Vec<u32>,
    vocabulary: Vocabulary,
    active: HashMap<u64, ActiveRequest>,
    renders: Arc<ThreadPool>,
    rendering: HashMap<String, Arc<AtomicBool>>,
    completed: std_mpsc::Sender<(PendingRender, RenderedJob)>,
    ready: std_mpsc::Receiver<(PendingRender, RenderedJob)>,
    packet: PacketEvents,
    released: Vec<String>,
    stream_interval: u32,
    pub(crate) metrics: Option<resin_http::MetricsEndpoint>,
    emit_first_token: bool,
}
impl Session {
    pub fn new(
        py: Python<'_>,
        engine: Py<Engine>,
        recipe: Py<PyAny>,
        stream_interval: u32,
        renders: Arc<ThreadPool>,
    ) -> PyResult<Self> {
        if stream_interval == 0 {
            return Err(PyValueError::new_err(
                "stream_interval must be a positive integer",
            ));
        }
        let state = engine.bind(py).borrow();
        if state.closed() || state.poisoned() {
            return Err(PyRuntimeError::new_err(
                "the inference engine is closed or poisoned",
            ));
        }
        drop(state);
        let vocabulary = Vocabulary::from_recipe(py, &recipe)?;
        let stop_token_ids = if let Ok(native) = recipe.bind(py).cast::<NativeRecipe>() {
            native.borrow().resolved_stop_token_ids()?.to_vec()
        } else {
            recipe.bind(py).getattr("stop_token_ids")?.extract()?
        };
        let (completed, ready) = std_mpsc::channel();
        Ok(Self {
            engine,
            recipe,
            stop_token_ids,
            vocabulary,
            active: HashMap::new(),
            renders,
            rendering: HashMap::new(),
            completed,
            ready,
            packet: HashMap::new(),
            released: Vec::new(),
            stream_interval,
            metrics: None,
            emit_first_token: false,
        })
    }
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
    pub fn admitted_count(&self) -> usize {
        self.active.len() + self.rendering.len()
    }

    pub fn schedule(&mut self, job: GenerationJob) -> PyResult<()> {
        self.enqueue(PendingRender {
            request: job.request,
            sender: Some(job.events),
            enqueued_at: job.enqueued_at,
        })
    }

    fn enqueue(&mut self, job: PendingRender) -> PyResult<()> {
        let id = &job.request.id;
        if self.rendering.contains_key(id)
            || (job.sender.is_none() && self.active.values().any(|active| &active.http_id == id))
        {
            return Err(PyValueError::new_err("duplicate HTTP request id"));
        }
        let canceled = Arc::new(AtomicBool::new(false));
        self.rendering.insert(id.clone(), canceled.clone());
        let recipe = Python::attach(|py| self.recipe.clone_ref(py));
        let completed = self.completed.clone();
        self.renders.spawn_fifo(move || {
            let rendered = catch_unwind(AssertUnwindSafe(|| {
                if canceled.load(Ordering::Acquire)
                    || job.sender.as_ref().is_some_and(mpsc::Sender::is_closed)
                {
                    return Err(PyRuntimeError::new_err("prompt rendering was cancelled"));
                }
                render_job(&recipe, &job.request)
            }))
            .unwrap_or_else(|_| Err(PyRuntimeError::new_err("prompt rendering panicked")));
            let _ = completed.send((job, rendered));
        });
        Ok(())
    }

    pub fn admit_ready(&mut self, limit: usize) -> usize {
        let mut ready = 0;
        while ready < limit
            && let Ok((job, rendered)) = self.ready.try_recv()
        {
            let Some(canceled) = self.rendering.remove(&job.request.id) else {
                continue;
            };
            if canceled.load(Ordering::Acquire) {
                self.released.push(job.request.id);
            } else {
                self.admit_rendered(job, rendered);
            }
            ready += 1;
        }
        ready
    }

    fn admit_rendered(&mut self, job: PendingRender, rendered: RenderedJob) {
        if let Some(metrics) = &self.metrics {
            metrics.observe_queue_time(job.enqueued_at.elapsed().as_secs_f64());
        }
        if job.sender.as_ref().is_some_and(mpsc::Sender::is_closed) {
            return;
        }
        let PendingRender {
            request,
            sender,
            enqueued_at,
        } = job;
        let result = Python::attach(|py| {
            let (token_ids, media, parser) = rendered?;
            if token_ids.is_empty() {
                return Err(PyValueError::new_err(
                    "recipe.render returned an empty token_ids sequence",
                ));
            }
            let parser = Parser::new(py, parser)?;
            let scores = request.logprobs.then(OutputScores::default);
            let extra = pythonize(py, &request.extra)?.unbind();
            let stops = StopFilter::new(request.stop.clone())?;
            let prompt_score_ids = request.prompt_logprobs.then(|| token_ids.clone());
            let sampling_params = Py::new(
                py,
                SamplingParams::new(
                    request
                        .max_output_tokens
                        .map(u32::try_from)
                        .transpose()
                        .map_err(|_| {
                            PyValueError::new_err("max_output_tokens must fit in a uint32")
                        })?,
                    self.stop_token_ids.to_vec(),
                    request.temperature,
                    request.top_p,
                    request.top_k,
                    request.seed,
                    request.logprobs,
                    request.top_logprobs,
                    request.prompt_logprobs,
                    0,
                    Some(extra),
                )?,
            )?;
            let input_tokens = token_ids.len();
            let lease = self.engine.bind(py).borrow_mut().submit(
                py,
                token_ids,
                request.priority.unwrap_or_default(),
                None,
                Some(sampling_params),
                media,
                None,
                request.cache_salt.clone(),
            )?;
            let lease_ref = lease.bind(py).borrow();
            let request_id = lease_ref.request_id();
            let cached_tokens = lease_ref.cached_tokens() as usize;
            drop(lease_ref);
            Ok::<_, PyErr>((
                request_id,
                ActiveRequest {
                    http_id: request.id.clone(),
                    lease,
                    parser,
                    stops,
                    scores,
                    prompt_score_ids,
                    sender: sender.clone(),
                    started: enqueued_at,
                    first_token_at: None,
                    input_tokens,
                    cached_tokens,
                    output_tokens: 0,
                    last_source: 0,
                    stream: StreamOutput::default(),
                },
            ))
        });
        let event = match result {
            Ok((id, active)) => {
                let input_tokens = active.input_tokens;
                self.active.insert(id, active);
                GenerationEvent::Started { input_tokens }
            }
            Err(error) => GenerationEvent::Error {
                message: format_python_error("request admission failed", error),
            },
        };
        send_event(sender.as_ref(), &mut self.packet, &request.id, event);
    }
    pub fn cancel_disconnected(&mut self) -> Result<(), String> {
        self.cancel_matching(|request| request.sender.as_ref().is_some_and(mpsc::Sender::is_closed))
            .map_err(|error| format_python_error("request cancellation failed", error))
    }
    pub fn process(&mut self, token_events: Vec<TokenEvent>) -> Result<(), String> {
        let mut grouped = HashMap::<u64, Vec<crate::TokenEvent>>::new();
        for event in token_events {
            grouped.entry(event.request_id).or_default().push(event);
        }
        Python::attach(|py| {
            for (&id, request) in self.active.iter() {
                if request.lease.bind(py).borrow().done() {
                    grouped.entry(id).or_default();
                }
            }
        });
        if grouped.is_empty() {
            return Ok(());
        }
        for (request_id, tokens) in grouped {
            let Some(request) = self.active.get_mut(&request_id) else {
                continue;
            };
            let first_token = !tokens.is_empty() && request.first_token_at.is_none();
            if first_token {
                request.first_token_at = Some(Instant::now());
                if let Some(metrics) = &self.metrics {
                    metrics.observe_first_token(request.started.elapsed().as_secs_f64());
                }
            }
            let mut finished = tokens.is_empty().then_some(CompletionReason::Length);
            let mut scores = std::mem::take(&mut request.stream.scores);
            let parsed = (|| -> PyResult<Vec<GenerationEvent>> {
                let mut output = std::mem::take(&mut request.stream.events);
                if tokens.is_empty() {
                    for event in request.parser.finish(ParserFinishReason::Length)? {
                        request.parsed(event, &mut output)?;
                    }
                    request.flush_stops(&mut output);
                }
                for token in tokens {
                    request.output_tokens += 1;
                    request.stream.tokens = request.stream.tokens.saturating_add(1);
                    let channel = if token.logprobs.is_none() {
                        None
                    } else {
                        request.parser.token_channel(token.token_id)
                    };
                    if let Some(logprobs) = &token.logprobs {
                        let score = self.vocabulary.generated(
                            token.token_id,
                            token.index,
                            channel,
                            logprobs,
                        );
                        if score.token.bytes.is_some()
                            && !matches!(channel, Some("control" | "reasoning" | "tool"))
                            && let Some(output_scores) = &mut request.scores
                        {
                            output_scores.insert(score.clone());
                        }
                        scores.push(score);
                    }
                    let events = request.parser.feed(&[token.token_id])?;
                    for event in events {
                        request.parsed(event, &mut output)?;
                        if request.stops.stopped() {
                            break;
                        }
                    }
                    if request.stops.stopped() {
                        finished = Some(CompletionReason::Stop);
                        Python::attach(|py| {
                            self.engine
                                .bind(py)
                                .borrow_mut()
                                .cancel(py, request.lease.bind(py).borrow())
                        })?;
                        break;
                    }
                    if let Some(reason) = token.finish_reason {
                        finished = Some(reason);
                        for event in request.parser.finish(match reason {
                            CompletionReason::Stop => ParserFinishReason::Stop,
                            CompletionReason::Length => ParserFinishReason::Length,
                        })? {
                            request.parsed(event, &mut output)?;
                        }
                        request.flush_stops(&mut output);
                        if request.stops.stopped() {
                            finished = Some(CompletionReason::Stop);
                        }
                        break;
                    }
                }
                if finished.is_some()
                    && let Some(scores) = &mut request.scores
                {
                    for event in scores.finish() {
                        push_event(&mut output, event);
                    }
                }
                Ok(output)
            })();
            let mut send = |event| {
                send_event(
                    request.sender.as_ref(),
                    &mut self.packet,
                    &request.http_id,
                    event,
                )
            };
            if first_token && self.emit_first_token {
                send(GenerationEvent::FirstToken);
            }
            let delivered = match parsed {
                Ok(events) => {
                    request.stream.events = events;
                    request.stream.scores = scores;
                    !request
                        .stream
                        .ready(self.stream_interval, finished.is_some())
                        || request.stream.flush(&mut send)
                }
                Err(error) => {
                    send(GenerationEvent::Error {
                        message: format_python_error("response parser failed", error),
                    });
                    false
                }
            };
            if !delivered {
                // A full output queue belongs to this client only. Never wait
                // for it from the one inference actor and stall every request.
                Python::attach(|py| {
                    self.engine
                        .bind(py)
                        .borrow_mut()
                        .cancel(py, request.lease.bind(py).borrow())
                })
                .map_err(|error| format_python_error("request cancellation failed", error))?;
                self.active.remove(&request_id);
                continue;
            }
            if let Some(reason) = finished {
                if let Some(sequence) = request.stops.matched() {
                    send(GenerationEvent::StopSequence(sequence.to_owned()));
                }
                if let Some(prompt_ids) = &request.prompt_score_ids {
                    let scores =
                        Python::attach(|py| request.lease.bind(py).borrow().prompt_logprobs())
                            .unwrap_or_default()
                            .into_iter()
                            .zip(prompt_ids)
                            .map(|(score, &id)| {
                                score.map(|score| self.vocabulary.token(id, score.inner.logprob))
                            })
                            .collect();
                    send(GenerationEvent::PromptLogprobs(scores));
                }
                let reason = match reason {
                    CompletionReason::Stop => FinishReason::Stop,
                    CompletionReason::Length => FinishReason::Length,
                };
                if let Some(metrics) = &self.metrics {
                    metrics.observe_completed(request.started.elapsed().as_secs_f64());
                }
                send(GenerationEvent::Finished {
                    reason,
                    input_tokens: request.input_tokens,
                    cached_tokens: request.cached_tokens,
                    output_tokens: request.output_tokens,
                    elapsed_ms: request.started.elapsed().as_secs_f64() * 1000.0,
                    generation_elapsed_ms: request
                        .first_token_at
                        .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                    time_to_first_token_ms: request.first_token_at.map_or(0.0, |first| {
                        first.duration_since(request.started).as_secs_f64() * 1000.0
                    }),
                });
                self.active.remove(&request_id);
            }
        }
        Ok(())
    }
    pub fn cancel(&mut self, ids: &[String]) -> PyResult<()> {
        for id in ids {
            if let Some(canceled) = self.rendering.get(id) {
                canceled.store(true, Ordering::Release);
            }
            self.packet.remove(id);
        }
        self.cancel_matching(|request| ids.contains(&request.http_id))
    }
    pub fn close(&mut self) -> PyResult<()> {
        self.cancel_renders();
        (self.completed, self.ready) = std_mpsc::channel();
        self.packet.clear();
        let result = self.cancel_matching(|_| true);
        self.released.clear();
        result
    }
    fn cancel_renders(&mut self) {
        for (_, canceled) in self.rendering.drain() {
            canceled.store(true, Ordering::Release);
        }
    }

    fn cancel_matching(&mut self, matches: impl Fn(&ActiveRequest) -> bool) -> PyResult<()> {
        let ids = self
            .active
            .iter()
            .filter_map(|(&id, request)| matches(request).then_some(id))
            .collect::<Vec<_>>();
        for id in ids {
            let request = self.active.remove(&id).expect("active id");
            Python::attach(|py| {
                self.engine
                    .bind(py)
                    .borrow_mut()
                    .cancel(py, request.lease.bind(py).borrow())
            })?;
            if request.sender.is_none() {
                self.released.push(request.http_id);
            }
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.cancel_renders();
    }
}
/// Generation input always includes its response parser.
type RenderedJob = PyResult<(Vec<u32>, Option<Py<PyTuple>>, Py<PyAny>)>;

struct RenderInput {
    messages: Vec<Py<Message>>,
    tools: Vec<Py<Tool>>,
    options: RenderOptions,
}

fn render_input(
    py: Python<'_>,
    recipe: &Py<PyAny>,
    request: &ConversationRequest,
) -> PyResult<RenderInput> {
    // JSON is an HTTP concern; both native and custom recipes receive the
    // same typed conversation. Native recipes stay inside Rust here.
    let messages = request
        .messages
        .iter()
        .map(|value| {
            let message = if let Ok(native) = recipe.bind(py).cast::<NativeRecipe>() {
                native.borrow().message_from_wire(py, value)?
            } else {
                Message::from_wire(py, value)?
            };
            Py::new(py, message)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let tools = request
        .tools
        .iter()
        .map(|value| Py::new(py, Tool::from_wire(value)?))
        .collect::<PyResult<Vec<_>>>()?;
    let options = RenderOptions::from_wire(&serde_json::json!({
        "reasoning_effort": request.reasoning_effort,
        "extra": request.extra,
    }))?;
    Ok(RenderInput {
        messages,
        tools,
        options,
    })
}

fn render_custom<'py>(
    py: Python<'py>,
    recipe: &Bound<'py, PyAny>,
    input: RenderInput,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("tools", input.tools)?;
    kwargs.set_item("options", Py::new(py, input.options)?)?;
    recipe.call_method("render", (input.messages,), Some(&kwargs))
}

/// Render prompt/media and create its parser off the engine thread.
fn render_job(recipe: &Py<PyAny>, request: &ConversationRequest) -> RenderedJob {
    Python::attach(|py| {
        let input = render_input(py, recipe, request)?;
        if let Ok(native) = recipe.bind(py).cast::<NativeRecipe>() {
            let native = native.borrow();
            let rendered = native.render(py, input.messages, input.tools, Some(&input.options))?;
            let parser = Py::new(py, native.create_parser(py, &rendered)?)?.into_any();
            Ok((rendered.token_ids, rendered.media, parser))
        } else {
            let rendered = render_custom(py, recipe.bind(py), input)?;
            let parser = recipe
                .bind(py)
                .call_method1("create_parser", (&rendered,))?
                .unbind();
            let prompt = rendered.cast::<RenderedPrompt>()?.borrow();
            Ok((
                prompt.token_ids.clone(),
                prompt.media.as_ref().map(|value| value.clone_ref(py)),
                parser,
            ))
        }
    })
}

/// Count the same rendered prompt without creating a parser or Engine request.
pub(crate) fn count_tokens(recipe: &Py<PyAny>, request: &ConversationRequest) -> PyResult<usize> {
    let count = Python::attach(|py| -> PyResult<usize> {
        let input = render_input(py, recipe, request)?;
        if let Ok(native) = recipe.bind(py).cast::<NativeRecipe>() {
            let prompt =
                native
                    .borrow()
                    .render(py, input.messages, input.tools, Some(&input.options))?;
            Ok(prompt.token_ids.len())
        } else {
            let rendered = render_custom(py, recipe.bind(py), input)?;
            Ok(rendered.cast::<RenderedPrompt>()?.borrow().token_ids.len())
        }
    })?;
    if count == 0 {
        return Err(PyValueError::new_err(
            "recipe.render returned an empty token_ids sequence",
        ));
    }
    Ok(count)
}

/// The same native lifecycle used by the ordinary actor, attached to one DP driver.
#[pyclass(name = "_HttpSession", module = "resin._resin")]
pub(crate) struct NativeHttpSession {
    inner: Mutex<Session>,
    admissions_per_step: usize,
}
#[pymethods]
impl NativeHttpSession {
    #[new]
    #[pyo3(signature = (engine, recipe: "Recipe", *, stream_interval = 1, emit_first_token = false,
        render_workers = None, admissions_per_step = 32))]
    fn new(
        py: Python<'_>,
        engine: Py<Engine>,
        recipe: Py<PyAny>,
        stream_interval: u32,
        emit_first_token: bool,
        render_workers: Option<u32>,
        admissions_per_step: usize,
    ) -> PyResult<Self> {
        if render_workers == Some(0) || admissions_per_step == 0 {
            return Err(PyValueError::new_err(
                "render and admission limits must be positive",
            ));
        }
        let renders =
            render_pool(render_worker_count(render_workers)).map_err(PyRuntimeError::new_err)?;
        let mut session = Session::new(py, engine, recipe, stream_interval, renders)?;
        session.emit_first_token = emit_first_token;
        Ok(Self {
            inner: Mutex::new(session),
            admissions_per_step,
        })
    }
    #[pyo3(signature = (ingress, *, rank = 0))]
    fn admit(&self, ingress: &mut HttpIngress, rank: usize) -> PyResult<usize> {
        let mut session = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        // Each replica owns its decoded ingress. Move large prompt and media
        // values into the render worker after the coordinator's broadcast.
        for request in std::mem::take(&mut ingress.requests)
            .into_iter()
            .filter(|request| request.routed_dp_rank.unwrap_or(0) == rank)
        {
            session.enqueue(PendingRender {
                request,
                sender: None,
                enqueued_at: Instant::now(),
            })?;
        }
        session.cancel(&ingress.canceled)?;
        Ok(session.admit_ready(self.admissions_per_step))
    }
    fn process(&self, py: Python<'_>, events: Vec<Py<TokenEvent>>) -> PyResult<HttpEvents> {
        let events = events
            .into_iter()
            .map(|event| event.bind(py).borrow().clone())
            .collect();
        let mut session = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        session.process(events).map_err(PyRuntimeError::new_err)?;
        Ok(HttpEvents {
            events: std::mem::take(&mut session.packet)
                .into_iter()
                .flat_map(|(id, events)| events.into_iter().map(move |event| (id.clone(), event)))
                .collect(),
            released: std::mem::take(&mut session.released),
            ..HttpEvents::default()
        })
    }
    fn close(&self) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .close()
    }
}

fn format_python_error(context: &str, error: PyErr) -> String {
    format!("{context}: {error}")
}

fn validate_spans(
    event: &AttributedEvent,
    tokens: usize,
    scores: Option<&OutputScores>,
) -> PyResult<()> {
    let length = match &event.event {
        ParsedEvent::TextDelta { delta }
        | ParsedEvent::ReasoningDelta { delta }
        | ParsedEvent::ToolCallArgumentsDelta { delta, .. } => delta.len(),
        ParsedEvent::ToolCallStart { .. } => 0,
    };
    if event.spans.iter().any(|span| {
        span.token as usize >= tokens
            || span.source.start >= span.source.end
            || span.text.start >= span.text.end
            || span.text.end as usize > length
            || scores
                .and_then(|scores| scores.source_len(span.token))
                .is_some_and(|length| span.source.end as usize > length)
    }) || event
        .spans
        .windows(2)
        .any(|pair| pair[0].token > pair[1].token)
    {
        return Err(PyValueError::new_err("invalid response-parser token spans"));
    }
    Ok(())
}

fn push_event(events: &mut Vec<GenerationEvent>, event: GenerationEvent) {
    match (events.last_mut(), event) {
        (
            Some(GenerationEvent::TextDelta { delta, logprobs }),
            GenerationEvent::TextDelta {
                delta: next,
                logprobs: scores,
            },
        ) => {
            delta.push_str(&next);
            logprobs.extend(scores);
        }
        (Some(GenerationEvent::ReasoningDelta(delta)), GenerationEvent::ReasoningDelta(next)) => {
            delta.push_str(&next);
        }
        (
            Some(GenerationEvent::ToolCallArgumentsDelta { index, delta }),
            GenerationEvent::ToolCallArgumentsDelta {
                index: next_index,
                delta: next,
            },
        ) if *index == next_index => delta.push_str(&next),
        (_, event) => events.push(event),
    }
}
