use std::{
    collections::BTreeMap,
    convert::Infallible,
    time::{SystemTime, UNIX_EPOCH},
};

use async_stream::stream;
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        Html, IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::{
    AppState,
    protocol::{ConversationRequest, GenerationEvent, ResponseAccumulator, json_error},
    ui::INDEX_HTML,
};

pub(crate) type RequestResult<T> = Result<T, Box<Response>>;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrManyStrings {
    One(String),
    Many(Vec<String>),
}

impl OneOrManyStrings {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    #[serde(default)]
    tools: Vec<Value>,
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_p")]
    top_p: f64,
    top_k: Option<u32>,
    seed: Option<i64>,
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
    #[serde(default)]
    prompt_logprobs: bool,
    routed_dp_rank: Option<Value>,
    stop: Option<OneOrManyStrings>,
    reasoning_effort: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: StreamOptions,
    n: Option<u32>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResponsesRequest {
    model: String,
    input: Value,
    stop: Option<OneOrManyStrings>,
    #[serde(default)]
    include: Vec<String>,
    instructions: Option<String>,
    #[serde(default)]
    tools: Vec<Value>,
    max_output_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_p")]
    top_p: f64,
    top_k: Option<u32>,
    seed: Option<i64>,
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
    #[serde(default)]
    prompt_logprobs: bool,
    routed_dp_rank: Option<Value>,
    reasoning: Option<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

const fn default_temperature() -> f64 {
    1.0
}

const fn default_top_p() -> f64 {
    1.0
}

pub async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

pub async fn health(State(state): State<AppState>) -> Response {
    if state.jobs.is_closed() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unavailable"})),
        )
            .into_response()
    } else {
        Json(json!({"status": "ok"})).into_response()
    }
}

pub async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model.as_ref(),
            "object": "model",
            "created": 0,
            "owned_by": "resin"
        }]
    }))
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<ChatRequest>,
) -> Response {
    body.routed_dp_rank = match routed_rank(&state, &headers, body.routed_dp_rank) {
        Ok(rank) => rank.map(|rank| json!(rank)),
        Err(response) => return *response,
    };
    let stream = body.stream;
    let include_usage = body.stream_options.include_usage;
    let request = match chat_request(&state, body) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let receiver = match state.enqueue(request.clone()) {
        Ok(receiver) => receiver,
        Err((status, message)) => {
            return (status, Json(json_error(message, "server_error"))).into_response();
        }
    };
    if stream {
        chat_stream(request, receiver, include_usage).into_response()
    } else {
        chat_response(request, receiver).await
    }
}

fn chat_request(state: &AppState, body: ChatRequest) -> RequestResult<ConversationRequest> {
    validate_model(state, &body.model)?;
    validate_sampling(body.temperature, body.top_p, body.top_k, body.seed)?;
    validate_logprobs(body.top_logprobs)?;
    validate_stops(body.stop.as_ref())?;
    if body.messages.is_empty() {
        return Err(invalid_request("messages must not be empty"));
    }
    if body.n.unwrap_or(1) != 1 {
        return Err(invalid_request("Resin currently supports only n=1"));
    }
    let max_output_tokens = body
        .max_completion_tokens
        .or(body.max_tokens)
        .unwrap_or(16_384);
    if max_output_tokens == 0 && !body.prompt_logprobs {
        return Err(invalid_request(
            "max_completion_tokens must be positive unless prompt_logprobs is enabled",
        ));
    }
    Ok(ConversationRequest {
        id: state.request_id("chatcmpl"),
        model: body.model,
        messages: body.messages,
        tools: function_tools(body.tools)?,
        max_output_tokens,
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        seed: body.seed,
        logprobs: body.logprobs.unwrap_or(false) || body.top_logprobs.unwrap_or(0) > 0,
        top_logprobs: body.top_logprobs.unwrap_or(0),
        prompt_logprobs: body.prompt_logprobs,
        routed_dp_rank: body
            .routed_dp_rank
            .and_then(|rank| rank.as_u64())
            .map(|rank| rank as usize),
        stop: body.stop.map_or_else(Vec::new, OneOrManyStrings::into_vec),
        reasoning_effort: body.reasoning_effort,
        extra: body.extra,
    })
}

async fn chat_response(
    request: ConversationRequest,
    receiver: mpsc::Receiver<GenerationEvent>,
) -> Response {
    let response = match ResponseAccumulator::collect(receiver).await {
        Ok(response) => response,
        Err(message) => return internal_error(message),
    };
    let reason = response.reason.expect("collector requires completion");
    let tool_calls = response
        .tool_calls
        .iter()
        .map(|tool| {
            json!({
                "id": tool.id,
                "type": "function",
                "function": {"name": tool.name, "arguments": tool.arguments},
            })
        })
        .collect::<Vec<_>>();
    let mut message = json!({"role": "assistant", "content": response.text});
    if !response.reasoning.is_empty() {
        message["reasoning_content"] = json!(response.reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }
    let logprobs = chat_logprobs(&request, &response.content_logprobs);
    Json(json!({
        "id": request.id,
        "object": "chat.completion",
        "created": unix_seconds(),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": reason.openai(!response.tool_calls.is_empty()),
            "logprobs": logprobs,
        }],
        "usage": usage(response.input_tokens, response.cached_tokens, response.output_tokens),
        "resin": raw_scores(&request, &response),
    }))
    .into_response()
}

fn chat_stream(
    request: ConversationRequest,
    mut receiver: mpsc::Receiver<GenerationEvent>,
    include_usage: bool,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let output = stream! {
        let mut terminated = false;
        let created = unix_seconds();
        let mut has_tool_calls = false;
        while let Some(event) = receiver.recv().await {
            let value = match event {
                GenerationEvent::Started { .. } => chat_chunk(&request, created, json!({"role": "assistant", "content": ""}), None),
                GenerationEvent::ReasoningDelta(delta) => chat_chunk(&request, created, json!({"reasoning_content": delta}), None),
                GenerationEvent::TextDelta { delta, logprobs } => {
                    let mut chunk = chat_chunk(&request, created, json!({"content": delta}), None);
                    chunk["choices"][0]["logprobs"] = chat_logprobs(&request, &logprobs);
                    chunk
                }
                GenerationEvent::ToolCallStart { index, id, name } => {
                    has_tool_calls = true;
                    chat_chunk(&request, created, json!({"tool_calls": [{
                        "index": index, "id": id, "type": "function",
                        "function": {"name": name, "arguments": ""}
                    }]}), None)
                }
                GenerationEvent::ToolCallArgumentsDelta { index, delta } => chat_chunk(&request, created, json!({"tool_calls": [{
                    "index": index, "function": {"arguments": delta}
                }]}), None),
                GenerationEvent::Logprobs(tokens) => {
                    let mut chunk = chat_chunk(&request, created, json!({}), None);
                    chunk["resin"] = json!({"generation_logprobs": tokens});
                    chunk
                }
                GenerationEvent::FirstToken | GenerationEvent::StopSequence(_) => continue,
                GenerationEvent::PromptLogprobs(tokens) => {
                    let mut chunk = chat_chunk(&request, created, json!({}), None);
                    chunk["resin"] = json!({"prompt_logprobs": tokens});
                    chunk
                }
                GenerationEvent::Finished { reason, input_tokens, cached_tokens, output_tokens, elapsed_ms, generation_elapsed_ms, time_to_first_token_ms } => {
                    terminated = true;
                    let mut chunk = chat_chunk(&request, created, json!({}), Some(reason.openai(has_tool_calls)));
                    let tokens_per_second = if generation_elapsed_ms > 0.0 { output_tokens as f64 * 1000.0 / generation_elapsed_ms } else { 0.0 };
                    chunk["resin"] = json!({
                        "prompt_tokens": input_tokens,
                        "cached_tokens": cached_tokens,
                        "completion_tokens": output_tokens,
                        "elapsed_ms": elapsed_ms,
                        "generation_elapsed_ms": generation_elapsed_ms,
                        "time_to_first_token_ms": time_to_first_token_ms,
                        "tokens_per_second": tokens_per_second,
                    });
                    yield Ok(Event::default().data(chunk.to_string()));
                    if include_usage {
                        yield Ok(Event::default().data(json!({
                            "id": request.id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [],
                            "usage": usage(input_tokens, cached_tokens, output_tokens),
                        }).to_string()));
                    }
                    break;
                }
                GenerationEvent::Error { message } => {
                    terminated = true;
                    yield Ok(Event::default().data(json_error(message, "server_error").to_string()));
                    break;
                }
            };
            yield Ok(Event::default().data(value.to_string()));
        }
        if !terminated {
            yield Ok(Event::default().data(json_error("the inference stream ended before completion", "server_error").to_string()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(output).keep_alive(KeepAlive::default())
}

fn chat_chunk(
    request: &ConversationRequest,
    created: u64,
    delta: Value,
    finish_reason: Option<&str>,
) -> Value {
    json!({
        "id": request.id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason, "logprobs": null}],
    })
}

pub async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<ResponsesRequest>,
) -> Response {
    body.routed_dp_rank = match routed_rank(&state, &headers, body.routed_dp_rank) {
        Ok(rank) => rank.map(|rank| json!(rank)),
        Err(response) => return *response,
    };
    let stream = body.stream;
    let request = match responses_request(&state, body) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let receiver = match state.enqueue(request.clone()) {
        Ok(receiver) => receiver,
        Err((status, message)) => {
            return (status, Json(json_error(message, "server_error"))).into_response();
        }
    };
    if stream {
        responses_stream(request, receiver).into_response()
    } else {
        responses_response(request, receiver).await
    }
}

fn responses_request(
    state: &AppState,
    mut body: ResponsesRequest,
) -> RequestResult<ConversationRequest> {
    validate_model(state, &body.model)?;
    validate_sampling(body.temperature, body.top_p, body.top_k, body.seed)?;
    validate_logprobs(body.top_logprobs)?;
    validate_stops(body.stop.as_ref())?;
    body.logprobs = Some(
        body.logprobs.unwrap_or(false)
            || body
                .include
                .iter()
                .any(|field| field == "message.output_text.logprobs"),
    );
    let mut messages = response_input_messages(body.input)?;
    if let Some(instructions) = body.instructions {
        messages.insert(0, json!({"role": "developer", "content": instructions}));
    }
    if messages.is_empty() {
        return Err(invalid_request("input must not be empty"));
    }
    let max_output_tokens = body.max_output_tokens.unwrap_or(16_384);
    if max_output_tokens == 0 && !body.prompt_logprobs {
        return Err(invalid_request(
            "max_output_tokens must be positive unless prompt_logprobs is enabled",
        ));
    }
    let reasoning_effort = body
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.get("effort"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    if let Some(reasoning) = body.reasoning.take() {
        body.extra.insert("reasoning".to_owned(), reasoning);
    }
    Ok(ConversationRequest {
        id: state.request_id("resp"),
        model: body.model,
        messages,
        tools: function_tools(body.tools)?,
        max_output_tokens,
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        seed: body.seed,
        logprobs: body.logprobs.unwrap_or(false) || body.top_logprobs.unwrap_or(0) > 0,
        top_logprobs: body.top_logprobs.unwrap_or(0),
        prompt_logprobs: body.prompt_logprobs,
        routed_dp_rank: body
            .routed_dp_rank
            .and_then(|rank| rank.as_u64())
            .map(|rank| rank as usize),
        stop: body.stop.map_or_else(Vec::new, OneOrManyStrings::into_vec),
        reasoning_effort,
        extra: body.extra,
    })
}

fn response_input_messages(input: Value) -> RequestResult<Vec<Value>> {
    match input {
        Value::String(text) => Ok(vec![json!({"role": "user", "content": text})]),
        Value::Array(items) => items
            .into_iter()
            .map(|item| {
                let item_type = item.get("type").and_then(Value::as_str);
                match item_type {
                    Some("function_call_output") => Ok(json!({
                        "role": "tool",
                        "tool_call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
                        "content": item.get("output").cloned().unwrap_or(Value::Null),
                    })),
                    Some("message") | None if item.get("role").is_some() => Ok(item),
                    _ => Err(invalid_request(
                        "Responses input currently supports message and function_call_output items",
                    )),
                }
            })
            .collect(),
        _ => Err(invalid_request(
            "input must be a string or an array of input items",
        )),
    }
}

async fn responses_response(
    request: ConversationRequest,
    receiver: mpsc::Receiver<GenerationEvent>,
) -> Response {
    let accumulator = match ResponseAccumulator::collect(receiver).await {
        Ok(response) => response,
        Err(message) => return internal_error(message),
    };
    Json(response_object(&request, &accumulator)).into_response()
}

fn response_object(request: &ConversationRequest, response: &ResponseAccumulator) -> Value {
    let mut output = Vec::new();
    if !response.reasoning.is_empty() {
        output.push(json!({
            "id": format!("rs_{}", &request.id[5..]),
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": response.reasoning}],
        }));
    }
    for (index, tool) in response.tool_calls.iter().enumerate() {
        output.push(json!({
            "id": format!("fc_{}_{index}", &request.id[5..]),
            "type": "function_call",
            "status": "completed",
            "call_id": tool.id,
            "name": tool.name,
            "arguments": tool.arguments,
        }));
    }
    if !response.text.is_empty() || response.tool_calls.is_empty() {
        output.push(json!({
            "id": format!("msg_{}", &request.id[5..]),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": response.text, "annotations": [], "logprobs": response.content_logprobs}],
        }));
    }
    json!({
        "id": request.id,
        "object": "response",
        "created_at": unix_seconds(),
        "status": "completed",
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "max_output_tokens": request.max_output_tokens,
        "model": request.model,
        "output": output,
        "parallel_tool_calls": true,
        "previous_response_id": null,
        "reasoning": {"effort": request.reasoning_effort, "summary": "auto"},
        "store": false,
        "temperature": request.temperature,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": request.top_p,
        "truncation": "disabled",
        "usage": {
            "input_tokens": response.input_tokens,
            "input_tokens_details": {"cached_tokens": response.cached_tokens},
            "output_tokens": response.output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": response.input_tokens + response.output_tokens,
        },
        "metadata": {},
        "resin": raw_scores(request, response),
    })
}

fn responses_stream(
    request: ConversationRequest,
    mut receiver: mpsc::Receiver<GenerationEvent>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let output = stream! {
        let mut terminated = false;
        let mut sequence = 0_u64;
        let mut accumulator = ResponseAccumulator::default();
        let mut text_open = false;
        let mut reasoning_open = false;
        let mut tool_open = Vec::<bool>::new();
        let skeleton = json!({"id": request.id, "object": "response", "created_at": unix_seconds(), "status": "in_progress", "model": request.model, "output": []});
        yield Ok(Event::default().data(event_json("response.created", sequence, json!({"response": skeleton})).to_string())); sequence += 1;
        while let Some(event) = receiver.recv().await {
            accumulator.apply(&event);
            match event {
                GenerationEvent::Started { .. } => {
                    yield Ok(Event::default().data(event_json("response.in_progress", sequence, json!({"response": {"id": request.id, "status": "in_progress"}})).to_string())); sequence += 1;
                }
                GenerationEvent::ReasoningDelta(delta) => {
                    if !reasoning_open {
                        reasoning_open = true;
                        yield Ok(Event::default().data(event_json("response.output_item.added", sequence, json!({"output_index": 0, "item": {"id": format!("rs_{}", &request.id[5..]), "type": "reasoning", "status": "in_progress", "summary": []}})).to_string())); sequence += 1;
                        yield Ok(Event::default().data(event_json("response.reasoning_summary_part.added", sequence, json!({"item_id": format!("rs_{}", &request.id[5..]), "output_index": 0, "summary_index": 0, "part": {"type": "summary_text", "text": ""}})).to_string())); sequence += 1;
                    }
                    yield Ok(Event::default().data(event_json("response.reasoning_summary_text.delta", sequence, json!({"item_id": format!("rs_{}", &request.id[5..]), "output_index": 0, "summary_index": 0, "delta": delta})).to_string())); sequence += 1;
                }
                GenerationEvent::TextDelta { delta, logprobs } => {
                    if !text_open {
                        text_open = true;
                        let output_index = usize::from(reasoning_open) + tool_open.len();
                        yield Ok(Event::default().data(event_json("response.output_item.added", sequence, json!({"output_index": output_index, "item": {"id": format!("msg_{}", &request.id[5..]), "type": "message", "status": "in_progress", "role": "assistant", "content": []}})).to_string())); sequence += 1;
                        yield Ok(Event::default().data(event_json("response.content_part.added", sequence, json!({"item_id": format!("msg_{}", &request.id[5..]), "output_index": output_index, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}})).to_string())); sequence += 1;
                    }
                    let output_index = usize::from(reasoning_open) + tool_open.len();
                    yield Ok(Event::default().data(event_json("response.output_text.delta", sequence, json!({"item_id": format!("msg_{}", &request.id[5..]), "output_index": output_index, "content_index": 0, "delta": delta, "logprobs": logprobs})).to_string())); sequence += 1;
                }
                GenerationEvent::ToolCallStart { index, id, name } => {
                    if tool_open.len() <= index { tool_open.resize(index + 1, false); }
                    tool_open[index] = true;
                    let output_index = usize::from(reasoning_open) + index;
                    yield Ok(Event::default().data(event_json("response.output_item.added", sequence, json!({"output_index": output_index, "item": {"id": format!("fc_{}_{index}", &request.id[5..]), "type": "function_call", "status": "in_progress", "call_id": id, "name": name, "arguments": ""}})).to_string())); sequence += 1;
                }
                GenerationEvent::ToolCallArgumentsDelta { index, delta } => {
                    let output_index = usize::from(reasoning_open) + index;
                    yield Ok(Event::default().data(event_json("response.function_call_arguments.delta", sequence, json!({"item_id": format!("fc_{}_{index}", &request.id[5..]), "output_index": output_index, "delta": delta})).to_string())); sequence += 1;
                }
                GenerationEvent::Logprobs(tokens) => {
                    yield Ok(Event::default().data(event_json("resin.generation_logprobs.delta", sequence, json!({"logprobs": tokens})).to_string())); sequence += 1;
                }
                GenerationEvent::FirstToken | GenerationEvent::StopSequence(_) => continue,
                GenerationEvent::PromptLogprobs(tokens) => {
                    yield Ok(Event::default().data(event_json("resin.prompt_logprobs", sequence, json!({"logprobs": tokens})).to_string())); sequence += 1;
                }
                GenerationEvent::Finished { .. } => {
                    terminated = true;
                    if reasoning_open {
                        let id = format!("rs_{}", &request.id[5..]);
                        yield Ok(Event::default().data(event_json("response.reasoning_summary_text.done", sequence, json!({"item_id": id, "output_index": 0, "summary_index": 0, "text": accumulator.reasoning})).to_string())); sequence += 1;
                        yield Ok(Event::default().data(event_json("response.reasoning_summary_part.done", sequence, json!({"item_id": id, "output_index": 0, "summary_index": 0, "part": {"type": "summary_text", "text": accumulator.reasoning}})).to_string())); sequence += 1;
                        yield Ok(Event::default().data(event_json("response.output_item.done", sequence, json!({"output_index": 0, "item": {"id": id, "type": "reasoning", "status": "completed", "summary": [{"type": "summary_text", "text": accumulator.reasoning}]}})).to_string())); sequence += 1;
                    }
                    for (index, tool) in accumulator.tool_calls.iter().enumerate() {
                        let output_index = usize::from(reasoning_open) + index;
                        let item_id = format!("fc_{}_{index}", &request.id[5..]);
                        yield Ok(Event::default().data(event_json("response.function_call_arguments.done", sequence, json!({"item_id": item_id, "output_index": output_index, "arguments": tool.arguments})).to_string())); sequence += 1;
                        yield Ok(Event::default().data(event_json("response.output_item.done", sequence, json!({"output_index": output_index, "item": {"id": item_id, "type": "function_call", "status": "completed", "call_id": tool.id, "name": tool.name, "arguments": tool.arguments}})).to_string())); sequence += 1;
                    }
                    if text_open {
                        let output_index = usize::from(reasoning_open) + accumulator.tool_calls.len();
                        let id = format!("msg_{}", &request.id[5..]);
                        yield Ok(Event::default().data(event_json("response.output_text.done", sequence, json!({"item_id": id, "output_index": output_index, "content_index": 0, "text": accumulator.text, "logprobs": accumulator.content_logprobs})).to_string())); sequence += 1;
                        yield Ok(Event::default().data(event_json("response.content_part.done", sequence, json!({"item_id": id, "output_index": output_index, "content_index": 0, "part": {"type": "output_text", "text": accumulator.text, "annotations": [], "logprobs": accumulator.content_logprobs}})).to_string())); sequence += 1;
                        yield Ok(Event::default().data(event_json("response.output_item.done", sequence, json!({"output_index": output_index, "item": {"id": id, "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": accumulator.text, "annotations": [], "logprobs": accumulator.content_logprobs}]}})).to_string())); sequence += 1;
                    }
                    yield Ok(Event::default().data(event_json("response.completed", sequence, json!({"response": response_object(&request, &accumulator)})).to_string()));
                    break;
                }
                GenerationEvent::Error { message } => {
                    terminated = true;
                    yield Ok(Event::default().data(event_json("error", sequence, json_error(message, "server_error")).to_string()));
                    break;
                }
            }
        }
        if !terminated {
            yield Ok(Event::default().data(event_json("error", sequence, json_error("the inference stream ended before completion", "server_error")).to_string()));
        }
    };
    Sse::new(output).keep_alive(KeepAlive::default())
}

fn event_json(event_type: &str, sequence_number: u64, fields: Value) -> Value {
    let mut value = fields;
    value["type"] = json!(event_type);
    value["sequence_number"] = json!(sequence_number);
    value
}

fn function_tools(tools: Vec<Value>) -> RequestResult<Vec<Value>> {
    tools
        .into_iter()
        .map(|tool| match tool.get("type").and_then(Value::as_str) {
            None => Ok(tool),
            Some("function") => Ok(tool.get("function").cloned().unwrap_or(tool)),
            Some(tool_type) => Err(invalid_request(format!(
                "tool type {tool_type:?} is not supported by a local Resin model"
            ))),
        })
        .collect()
}

fn validate_model(state: &AppState, model: &str) -> RequestResult<()> {
    if model == state.model.as_ref() {
        Ok(())
    } else {
        Err(Box::new(
            (
                StatusCode::NOT_FOUND,
                Json(json_error(
                    format!("model {model:?} is not served; expected {:?}", state.model),
                    "invalid_request_error",
                )),
            )
                .into_response(),
        ))
    }
}

pub(crate) fn validate_sampling(
    temperature: f64,
    top_p: f64,
    top_k: Option<u32>,
    seed: Option<i64>,
) -> RequestResult<()> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(invalid_request(
            "temperature must be finite and non-negative",
        ));
    }
    if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
        return Err(invalid_request("top_p must be finite and in (0, 1]"));
    }
    if top_k == Some(0) {
        return Err(invalid_request("top_k must be positive"));
    }
    if seed.is_some_and(|seed| seed < 0) {
        return Err(invalid_request("seed must be non-negative"));
    }
    Ok(())
}

pub(crate) fn invalid_request(message: impl Into<String>) -> Box<Response> {
    Box::new(
        (
            StatusCode::BAD_REQUEST,
            Json(json_error(message, "invalid_request_error")),
        )
            .into_response(),
    )
}

fn internal_error(message: impl Into<String>) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json_error(message, "server_error")),
    )
        .into_response()
}

fn usage(input_tokens: usize, cached_tokens: usize, output_tokens: usize) -> Value {
    json!({
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": 0},
    })
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn validate_logprobs(top_logprobs: Option<usize>) -> RequestResult<()> {
    if top_logprobs.is_some_and(|count| count > 20) {
        return Err(invalid_request("top_logprobs must be between 0 and 20"));
    }
    Ok(())
}

pub(crate) fn routed_rank(
    state: &AppState,
    headers: &HeaderMap,
    body: Option<Value>,
) -> RequestResult<Option<usize>> {
    let rank = match headers.get("resin-routed-dp-rank") {
        Some(value) => Some(
            value
                .to_str()
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or_else(|| {
                    invalid_request("resin-routed-dp-rank must be a nonnegative integer")
                })?,
        ),
        None => body
            .map(|rank| {
                rank.as_u64()
                    .and_then(|rank| usize::try_from(rank).ok())
                    .ok_or_else(|| invalid_request("routed_dp_rank must be a nonnegative integer"))
            })
            .transpose()?,
    };
    if rank.is_some_and(|rank| rank >= state.routed_dp_size) {
        return Err(invalid_request(format!(
            "routed_dp_rank must be less than {}",
            state.routed_dp_size
        )));
    }
    Ok(rank)
}

fn chat_logprobs(request: &ConversationRequest, tokens: &[crate::GeneratedTokenLogprob]) -> Value {
    if request.logprobs {
        json!({"content": tokens, "refusal": null})
    } else {
        Value::Null
    }
}

fn raw_scores(request: &ConversationRequest, response: &ResponseAccumulator) -> Value {
    let mut scores = json!({});
    if request.logprobs {
        scores["generation_logprobs"] = json!(response.logprobs);
    }
    if request.prompt_logprobs {
        scores["prompt_logprobs"] = json!(response.prompt_logprobs);
    }
    scores
}

fn validate_stops(stops: Option<&OneOrManyStrings>) -> RequestResult<()> {
    let empty = match stops {
        Some(OneOrManyStrings::One(stop)) => stop.is_empty(),
        Some(OneOrManyStrings::Many(stops)) => stops.iter().any(String::is_empty),
        None => false,
    };
    if empty {
        Err(invalid_request("stop sequences must not be empty"))
    } else {
        Ok(())
    }
}
