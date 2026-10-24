//! Shared request lifecycle for local serving and routed attention-DP drivers.
use crate::http_distributed::{HttpEvents, HttpIngress};
use crate::http_logprobs::Vocabulary;
use crate::http_output::OutputScores;
use crate::http_stop::StopFilter;
use crate::recipe::{
    self, NativeInklingParser, NativeRecipe, NativeResponseParser, RenderedPrompt, SharedParser,
};
use crate::recipe_events;
use crate::recipe_types::{Message, RenderOptions, Tool};
use crate::{CompletionReason, Engine, RequestLease, SamplingParams, TokenEvent};
use pyo3::{
    exceptions::{PyAttributeError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use pythonize::pythonize;
use resin_http::{
    EVENT_QUEUE_CAPACITY, FinishReason, GeneratedTokenLogprob, GenerationEvent, GenerationJob,
};
use resin_recipe::{AttributedEvent, Event as ParsedEvent, FinishReason as ParserFinishReason};
use std::{collections::HashMap, sync::Mutex, time::Instant};
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
        if let Ok(native) = object.bind(py).cast::<NativeInklingParser>() {
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

    fn flush(&mut self, sender: &mpsc::Sender<GenerationEvent>) -> bool {
        if self.events.is_empty() && self.scores.is_empty() {
            return true;
        }
        let has_output = self.has_output();
        let delivered = self
            .events
            .drain(..)
            .all(|event| sender.try_send(event).is_ok())
            && (self.scores.is_empty()
                || sender
                    .try_send(GenerationEvent::Logprobs(std::mem::take(&mut self.scores)))
                    .is_ok());
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
    prompt_logprobs: bool,
    prompt_ids: Vec<u32>,
    events: mpsc::Sender<GenerationEvent>,
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

pub(crate) struct Session {
    engine: Py<Engine>,
    recipe: Py<PyAny>,
    vocabulary: Vocabulary,
    active: HashMap<u64, ActiveRequest>,
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
        Ok(Self {
            engine,
            recipe,
            vocabulary,
            active: HashMap::new(),
            stream_interval,
            metrics: None,
            emit_first_token: false,
        })
    }
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
    pub fn admit(&mut self, job: GenerationJob) {
        if let Some(metrics) = &self.metrics {
            metrics.observe_queue_time(job.enqueued_at.elapsed().as_secs_f64());
        }
        admit(&self.engine, &self.recipe, job, &mut self.active);
    }
    pub fn cancel_disconnected(&mut self) -> Result<(), String> {
        cancel_disconnected(&self.engine, &mut self.active)
    }
    pub fn process(&mut self, tokens: Vec<TokenEvent>) -> Result<(), String> {
        process(
            &self.engine,
            &self.vocabulary,
            &mut self.active,
            self.stream_interval,
            self.metrics.as_ref(),
            self.emit_first_token,
            tokens,
        )
    }
    pub fn cancel(&mut self, ids: &[String]) -> PyResult<()> {
        let remove = self
            .active
            .iter()
            .filter_map(|(&id, request)| ids.contains(&request.http_id).then_some(id))
            .collect::<Vec<_>>();
        for id in remove {
            let request = self.active.remove(&id).expect("active id");
            Python::attach(|py| {
                self.engine
                    .bind(py)
                    .borrow_mut()
                    .cancel(py, request.lease.bind(py).borrow())
            })?;
        }
        Ok(())
    }
    pub fn close(&mut self) -> PyResult<()> {
        let ids = self
            .active
            .values()
            .map(|request| request.http_id.clone())
            .collect::<Vec<_>>();
        self.cancel(&ids)
    }
}
fn process(
    engine: &Py<Engine>,
    vocabulary: &Vocabulary,
    active: &mut HashMap<u64, ActiveRequest>,
    stream_interval: u32,
    metrics: Option<&resin_http::MetricsEndpoint>,
    emit_first_token: bool,
    token_events: Vec<TokenEvent>,
) -> Result<(), String> {
    let mut grouped = HashMap::<u64, Vec<crate::TokenEvent>>::new();
    for event in token_events {
        grouped.entry(event.request_id).or_default().push(event);
    }
    Python::attach(|py| {
        for (&id, request) in active.iter() {
            if request.lease.bind(py).borrow().done() {
                grouped.entry(id).or_default();
            }
        }
    });
    if grouped.is_empty() {
        return Ok(());
    }
    for (request_id, tokens) in grouped {
        let Some(request) = active.get_mut(&request_id) else {
            continue;
        };
        if !tokens.is_empty() && request.first_token_at.is_none() {
            request.first_token_at = Some(Instant::now());
            if let Some(metrics) = metrics {
                metrics.observe_first_token(request.started.elapsed().as_secs_f64());
            } else if emit_first_token {
                let _ = request.events.try_send(GenerationEvent::FirstToken);
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
                    let score =
                        vocabulary.generated(token.token_id, token.index, channel, logprobs);
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
                        engine
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
        let delivered = match parsed {
            Ok(events) => {
                request.stream.events = events;
                request.stream.scores = scores;
                !request.stream.ready(stream_interval, finished.is_some())
                    || request.stream.flush(&request.events)
            }
            Err(error) => {
                let _ = request.events.try_send(GenerationEvent::Error {
                    message: format_python_error("response parser failed", error),
                });
                false
            }
        };
        if !delivered {
            // A full output queue belongs to this client only. Never wait
            // for it from the one inference actor and stall every request.
            Python::attach(|py| {
                engine
                    .bind(py)
                    .borrow_mut()
                    .cancel(py, request.lease.bind(py).borrow())
            })
            .map_err(|error| format_python_error("request cancellation failed", error))?;
            active.remove(&request_id);
            continue;
        }
        if let Some(reason) = finished {
            if let Some(sequence) = request.stops.matched() {
                let _ = request
                    .events
                    .try_send(GenerationEvent::StopSequence(sequence.to_owned()));
            }
            if request.prompt_logprobs {
                let scores = Python::attach(|py| request.lease.bind(py).borrow().prompt_logprobs())
                    .unwrap_or_default()
                    .into_iter()
                    .zip(&request.prompt_ids)
                    .map(|(score, &id)| {
                        score.map(|score| vocabulary.token(id, score.inner.logprob))
                    })
                    .collect();
                let _ = request
                    .events
                    .try_send(GenerationEvent::PromptLogprobs(scores));
            }
            let reason = match reason {
                CompletionReason::Stop => FinishReason::Stop,
                CompletionReason::Length => FinishReason::Length,
            };
            if let Some(metrics) = metrics {
                metrics.observe_completed(request.started.elapsed().as_secs_f64());
            }
            let _ = request.events.try_send(GenerationEvent::Finished {
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
            active.remove(&request_id);
        }
    }
    Ok(())
}

fn admit(
    engine: &Py<Engine>,
    recipe: &Py<PyAny>,
    job: GenerationJob,
    active: &mut HashMap<u64, ActiveRequest>,
) {
    if job.events.is_closed() {
        return;
    }
    let result = Python::attach(|py| {
        // JSON is an HTTP concern; both native and custom recipes receive the
        // same typed conversation. Native recipes stay inside Rust here.
        let messages = job
            .request
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
        let tools = job
            .request
            .tools
            .iter()
            .map(|value| Py::new(py, Tool::from_wire(value)?))
            .collect::<PyResult<Vec<_>>>()?;
        let options = RenderOptions::from_wire(&serde_json::json!({
            "reasoning_effort": job.request.reasoning_effort,
            "extra": job.request.extra,
        }))?;
        let (token_ids, stop_token_ids, media, parser) =
            if let Ok(native) = recipe.bind(py).cast::<NativeRecipe>() {
                let native = native.borrow();
                let rendered = native.render(py, messages, tools, Some(&options))?;
                let parser = Py::new(py, native.create_parser(py, &rendered)?)?.into_any();
                (
                    rendered.token_ids,
                    rendered.stop_token_ids,
                    rendered.media,
                    parser,
                )
            } else {
                let kwargs = PyDict::new(py);
                kwargs.set_item("tools", tools)?;
                kwargs.set_item("options", Py::new(py, options)?)?;
                let rendered = recipe
                    .bind(py)
                    .call_method("render", (messages,), Some(&kwargs))?;
                let parser = recipe
                    .bind(py)
                    .call_method1("create_parser", (&rendered,))?
                    .unbind();
                let prompt = rendered.cast::<RenderedPrompt>()?.borrow();
                // Custom recipes also return the native prompt container, so media
                // and token ownership cross one well-defined admission boundary.
                (
                    prompt.token_ids.clone(),
                    prompt.stop_token_ids.clone(),
                    prompt.media.as_ref().map(|value| value.clone_ref(py)),
                    parser,
                )
            };
        if token_ids.is_empty() {
            return Err(PyValueError::new_err(
                "recipe.render returned an empty token_ids sequence",
            ));
        }
        let parser = Parser::new(py, parser)?;
        let scores = job.request.logprobs.then(OutputScores::default);
        let extra = pythonize(py, &job.request.extra)?.unbind();
        let stops = StopFilter::new(job.request.stop.clone())?;
        let prompt_ids = token_ids.clone();
        let sampling_params = Py::new(
            py,
            SamplingParams::new(
                Some(u32::try_from(job.request.max_output_tokens).map_err(|_| {
                    PyValueError::new_err("max_output_tokens must fit in a uint32")
                })?),
                stop_token_ids,
                job.request.temperature,
                job.request.top_p,
                job.request.top_k,
                job.request.seed,
                job.request.logprobs,
                job.request.top_logprobs,
                job.request.prompt_logprobs,
                0,
                Some(extra),
            )?,
        )?;
        let input_tokens = token_ids.len();
        let lease = engine.bind(py).borrow_mut().submit(
            py,
            Some(token_ids),
            0,
            None,
            Some(sampling_params),
            media,
            None,
        )?;
        let lease_ref = lease.bind(py).borrow();
        let request_id = lease_ref.request_id();
        let cached_tokens = lease_ref.cached_tokens() as usize;
        drop(lease_ref);
        Ok::<_, PyErr>((
            request_id,
            lease,
            parser,
            scores,
            stops,
            prompt_ids,
            input_tokens,
            cached_tokens,
        ))
    });
    match result {
        Ok((request_id, lease, parser, scores, stops, prompt_ids, input_tokens, cached_tokens)) => {
            let _ = job
                .events
                .try_send(GenerationEvent::Started { input_tokens });
            active.insert(
                request_id,
                ActiveRequest {
                    http_id: job.request.id.clone(),
                    lease,
                    parser,
                    stops,
                    scores,
                    prompt_logprobs: job.request.prompt_logprobs,
                    prompt_ids,
                    events: job.events,
                    started: job.enqueued_at,
                    first_token_at: None,
                    input_tokens,
                    cached_tokens,
                    output_tokens: 0,
                    last_source: 0,
                    stream: StreamOutput::default(),
                },
            );
        }
        Err(error) => {
            let _ = job.events.try_send(GenerationEvent::Error {
                message: format_python_error("request admission failed", error),
            });
        }
    }
}

fn cancel_disconnected(
    engine: &Py<Engine>,
    active: &mut HashMap<u64, ActiveRequest>,
) -> Result<(), String> {
    let disconnected = active
        .iter()
        .filter_map(|(request_id, request)| request.events.is_closed().then_some(*request_id))
        .collect::<Vec<_>>();
    for request_id in disconnected {
        let request = active
            .remove(&request_id)
            .expect("request id came from the active map");
        Python::attach(|py| {
            engine
                .bind(py)
                .borrow_mut()
                .cancel(py, request.lease.bind(py).borrow())
        })
        .map_err(|error| format_python_error("request cancellation failed", error))?;
    }
    Ok(())
}

/// The same native lifecycle used by the ordinary actor, attached to one DP driver.
#[pyclass(name = "_HttpSession", module = "resin._resin")]
pub(crate) struct NativeHttpSession {
    inner: Mutex<(Session, HashMap<String, mpsc::Receiver<GenerationEvent>>)>,
}
#[pymethods]
impl NativeHttpSession {
    #[new]
    #[pyo3(signature = (engine, recipe, *, stream_interval = 1, emit_first_token = false))]
    fn new(
        py: Python<'_>,
        engine: Py<Engine>,
        recipe: Py<PyAny>,
        stream_interval: u32,
        emit_first_token: bool,
    ) -> PyResult<Self> {
        let mut session = Session::new(py, engine, recipe, stream_interval)?;
        session.emit_first_token = emit_first_token;
        Ok(Self {
            inner: Mutex::new((session, HashMap::new())),
        })
    }
    #[pyo3(signature = (ingress, *, rank = 0))]
    fn admit(&self, ingress: &HttpIngress, rank: usize) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        let (session, outputs) = &mut *inner;
        for request in ingress
            .requests
            .iter()
            .filter(|request| request.routed_dp_rank.unwrap_or(0) == rank)
        {
            if outputs.contains_key(&request.id) {
                return Err(PyValueError::new_err("duplicate HTTP request id"));
            }
            let (events, receiver) = mpsc::channel(EVENT_QUEUE_CAPACITY);
            outputs.insert(request.id.clone(), receiver);
            session.admit(GenerationJob {
                request: request.clone(),
                events,
                enqueued_at: Instant::now(),
            });
        }
        session.cancel(&ingress.canceled)?;
        for id in &ingress.canceled {
            outputs.remove(id);
        }
        Ok(())
    }
    fn cancel(&self, ids: Vec<String>) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        inner.0.cancel(&ids)?;
        for id in ids {
            inner.1.remove(&id);
        }
        Ok(())
    }
    fn process(&self, py: Python<'_>, events: Vec<Py<TokenEvent>>) -> PyResult<HttpEvents> {
        let events = events
            .into_iter()
            .map(|event| event.bind(py).borrow().clone())
            .collect();
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        inner.0.process(events).map_err(PyRuntimeError::new_err)?;
        let mut output = Vec::new();
        inner.1.retain(|id, receiver| {
            let mut finished = false;
            while let Ok(event) = receiver.try_recv() {
                finished |= matches!(
                    event,
                    GenerationEvent::Finished { .. } | GenerationEvent::Error { .. }
                );
                output.push((id.clone(), event));
            }
            if !finished && receiver.is_closed() {
                // A verified block can exceed the bounded per-request queue.
                // The session cancels that lease; expose the same terminal
                // failure as an ordinary HTTP receiver reaching EOF.
                output.push((
                    id.clone(),
                    GenerationEvent::Error {
                        message: "the inference stream ended before completion".to_owned(),
                    },
                ));
                finished = true;
            }
            !finished
        });
        Ok(HttpEvents {
            events: output,
            ..HttpEvents::default()
        })
    }
    fn close(&self) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap_or_else(|error| error.into_inner());
        inner.0.close()?;
        inner.1.clear();
        Ok(())
    }
    #[getter]
    fn active_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .0
            .active_count()
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
