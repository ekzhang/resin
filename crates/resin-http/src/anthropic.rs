use std::{collections::BTreeMap, convert::Infallible};

use async_stream::stream;
use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::{
    AppState,
    openai::{RequestResult, invalid_request, routed_rank, validate_sampling},
    protocol::{ConversationRequest, GenerationEvent, ResponseAccumulator, json_error},
};

#[derive(Debug, Deserialize)]
pub(crate) struct MessagesRequest {
    model: String,
    routed_dp_rank: Option<Value>,
    messages: Vec<Value>,
    system: Option<Value>,
    #[serde(default)]
    tools: Vec<Value>,
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_p")]
    top_p: f64,
    top_k: Option<u32>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    thinking: Option<Value>,
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

pub async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<MessagesRequest>,
) -> Response {
    body.routed_dp_rank = match routed_rank(&state, &headers, body.routed_dp_rank) {
        Ok(rank) => rank.map(|rank| json!(rank)),
        Err(response) => return *response,
    };
    let stream = body.stream;
    let request = match conversation_request(&state, body) {
        Ok(request) => request,
        Err(response) => return *response,
    };
    let receiver = match state.enqueue(request.clone()) {
        Ok(receiver) => receiver,
        Err((status, message)) => {
            return (status, Json(json_error(message, "api_error"))).into_response();
        }
    };
    if stream {
        messages_stream(request, receiver).into_response()
    } else {
        messages_response(request, receiver).await
    }
}

fn conversation_request(
    state: &AppState,
    mut body: MessagesRequest,
) -> RequestResult<ConversationRequest> {
    if body.model != state.model.as_ref() {
        return Err(Box::new(
            (
                axum::http::StatusCode::NOT_FOUND,
                Json(json_error(
                    format!(
                        "model {:?} is not served; expected {:?}",
                        body.model, state.model
                    ),
                    "not_found_error",
                )),
            )
                .into_response(),
        ));
    }
    if body.messages.is_empty() {
        return Err(invalid_request("messages must not be empty"));
    }
    if body.max_tokens == 0 {
        return Err(invalid_request("max_tokens must be positive"));
    }
    validate_sampling(body.temperature, body.top_p, body.top_k, None)?;
    if body.stop_sequences.iter().any(String::is_empty) {
        return Err(invalid_request("stop sequences must not be empty"));
    }
    let mut messages = Vec::new();
    if let Some(system) = body.system {
        messages.push(json!({"role": "system", "content": content_text(&system)?}));
    }
    for message in body.messages {
        messages.extend(normalize_message(message)?);
    }
    let tools = body
        .tools
        .into_iter()
        .map(|tool| {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                return Err(invalid_request("every tool must have a name"));
            };
            Ok(json!({
                "name": name,
                "description": tool.get("description").and_then(Value::as_str).unwrap_or(""),
                "parameters": tool.get("input_schema").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(thinking) = body.thinking.take() {
        body.extra.insert("thinking".to_owned(), thinking);
    }
    Ok(ConversationRequest {
        id: state.request_id("msg"),
        model: body.model,
        messages,
        tools,
        max_output_tokens: body.max_tokens,
        temperature: body.temperature,
        top_p: body.top_p,
        top_k: body.top_k,
        seed: None,
        logprobs: false,
        top_logprobs: 0,
        prompt_logprobs: false,
        routed_dp_rank: body
            .routed_dp_rank
            .and_then(|rank| rank.as_u64())
            .map(|rank| rank as usize),
        stop: body.stop_sequences,
        reasoning_effort: None,
        extra: body.extra,
    })
}

fn normalize_message(message: Value) -> RequestResult<Vec<Value>> {
    let Some(role) = message.get("role").and_then(Value::as_str) else {
        return Err(invalid_request("every message must have a role"));
    };
    let content = message.get("content").cloned().unwrap_or(Value::Null);
    let Value::Array(blocks) = content else {
        return Ok(vec![json!({"role": role, "content": content})]);
    };
    if role == "assistant" {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    text.push_str(block.get("text").and_then(Value::as_str).unwrap_or(""))
                }
                Some("thinking") => {
                    reasoning.push_str(block.get("thinking").and_then(Value::as_str).unwrap_or(""))
                }
                Some("tool_use") => tool_calls.push(json!({
                    "id": block.get("id").cloned().unwrap_or(Value::Null),
                    "type": "function",
                    "function": {
                        "name": block.get("name").cloned().unwrap_or(Value::Null),
                        "arguments": block.get("input").cloned().unwrap_or_else(|| json!({})),
                    }
                })),
                Some(block_type) => {
                    return Err(invalid_request(format!(
                        "assistant content block {block_type:?} is not supported"
                    )));
                }
                None => return Err(invalid_request("every content block must have a type")),
            }
        }
        let mut normalized = json!({"role": "assistant", "content": text});
        if !reasoning.is_empty() {
            normalized["reasoning_content"] = json!(reasoning);
        }
        if !tool_calls.is_empty() {
            normalized["tool_calls"] = json!(tool_calls);
        }
        return Ok(vec![normalized]);
    }

    let mut normalized = Vec::new();
    let mut text = String::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(block.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("tool_result") => {
                if !text.is_empty() {
                    normalized.push(json!({"role": role, "content": std::mem::take(&mut text)}));
                }
                normalized.push(json!({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id").cloned().unwrap_or(Value::Null),
                    "content": content_text(block.get("content").unwrap_or(&Value::Null))?,
                }));
            }
            Some(block_type) => {
                return Err(invalid_request(format!(
                    "user content block {block_type:?} is not supported"
                )));
            }
            None => return Err(invalid_request("every content block must have a type")),
        }
    }
    if !text.is_empty() {
        normalized.push(json!({"role": role, "content": text}));
    }
    Ok(normalized)
}

fn content_text(content: &Value) -> RequestResult<String> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(blocks) => {
            let mut text = String::new();
            for block in blocks {
                if block.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(invalid_request(
                        "only text content blocks are supported here",
                    ));
                }
                text.push_str(block.get("text").and_then(Value::as_str).unwrap_or(""));
            }
            Ok(text)
        }
        _ => Err(invalid_request(
            "content must be text or an array of text blocks",
        )),
    }
}

async fn messages_response(
    request: ConversationRequest,
    receiver: mpsc::Receiver<GenerationEvent>,
) -> Response {
    let response = match ResponseAccumulator::collect(receiver).await {
        Ok(response) => response,
        Err(message) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(json_error(message, "api_error")),
            )
                .into_response();
        }
    };
    let reason = response.reason.expect("collector requires completion");
    let mut content = Vec::new();
    if !response.reasoning.is_empty() {
        content.push(json!({"type": "thinking", "thinking": response.reasoning, "signature": ""}));
    }
    for tool in &response.tool_calls {
        let input = serde_json::from_str::<Value>(&tool.arguments)
            .unwrap_or_else(|_| json!({"raw": tool.arguments}));
        content.push(json!({"type": "tool_use", "id": tool.id, "name": tool.name, "input": input}));
    }
    if !response.text.is_empty() || response.tool_calls.is_empty() {
        content.push(json!({"type": "text", "text": response.text}));
    }
    Json(json!({
        "id": request.id,
        "type": "message",
        "role": "assistant",
        "model": request.model,
        "content": content,
        "stop_reason": if response.stop_sequence.is_some() { "stop_sequence" } else { reason.anthropic(!response.tool_calls.is_empty()) },
        "stop_sequence": response.stop_sequence,
        "usage": {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
    }))
    .into_response()
}

fn messages_stream(
    request: ConversationRequest,
    mut receiver: mpsc::Receiver<GenerationEvent>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let output = stream! {
        let mut terminated = false;
        let mut stop_sequence: Option<String> = None;
        let mut input_tokens = 0;
        let mut block_index = 0_usize;
        let mut reasoning_open = false;
        let mut text_open = false;
        let mut tool_indices = Vec::<usize>::new();
        let mut has_tool_calls = false;
        while let Some(event) = receiver.recv().await {
            match event {
                GenerationEvent::Started { input_tokens: count } => {
                    input_tokens = count;
                    yield named_event("message_start", json!({
                        "type": "message_start",
                        "message": {"id": request.id, "type": "message", "role": "assistant", "model": request.model, "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": count, "output_tokens": 0}}
                    }));
                }
                GenerationEvent::ReasoningDelta(delta) => {
                    if !reasoning_open {
                        reasoning_open = true;
                        yield named_event("content_block_start", json!({"type": "content_block_start", "index": block_index, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}));
                    }
                    yield named_event("content_block_delta", json!({"type": "content_block_delta", "index": block_index, "delta": {"type": "thinking_delta", "thinking": delta}}));
                }
                GenerationEvent::TextDelta { delta, .. } => {
                    if reasoning_open && !text_open {
                        yield named_event("content_block_stop", json!({"type": "content_block_stop", "index": block_index}));
                        block_index += 1;
                        reasoning_open = false;
                    }
                    if !text_open {
                        text_open = true;
                        yield named_event("content_block_start", json!({"type": "content_block_start", "index": block_index, "content_block": {"type": "text", "text": ""}}));
                    }
                    yield named_event("content_block_delta", json!({"type": "content_block_delta", "index": block_index, "delta": {"type": "text_delta", "text": delta}}));
                }
                GenerationEvent::ToolCallStart { index, id, name } => {
                    has_tool_calls = true;
                    if reasoning_open {
                        yield named_event("content_block_stop", json!({"type": "content_block_stop", "index": block_index}));
                        block_index += 1;
                        reasoning_open = false;
                    }
                    if text_open {
                        yield named_event("content_block_stop", json!({"type": "content_block_stop", "index": block_index}));
                        block_index += 1;
                        text_open = false;
                    }
                    if tool_indices.len() <= index { tool_indices.resize(index + 1, usize::MAX); }
                    tool_indices[index] = block_index;
                    yield named_event("content_block_start", json!({"type": "content_block_start", "index": block_index, "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}}));
                    block_index += 1;
                }
                GenerationEvent::ToolCallArgumentsDelta { index, delta } => {
                    let wire_index = tool_indices.get(index).copied().unwrap_or(index);
                    yield named_event("content_block_delta", json!({"type": "content_block_delta", "index": wire_index, "delta": {"type": "input_json_delta", "partial_json": delta}}));
                }
                GenerationEvent::FirstToken | GenerationEvent::Logprobs(_) | GenerationEvent::PromptLogprobs(_) => {}
                GenerationEvent::StopSequence(sequence) => { stop_sequence = Some(sequence); }
                GenerationEvent::Finished { reason, output_tokens, .. } => {
                    terminated = true;
                    if reasoning_open || text_open {
                        yield named_event("content_block_stop", json!({"type": "content_block_stop", "index": block_index}));
                    }
                    for index in tool_indices.into_iter().filter(|index| *index != usize::MAX) {
                        yield named_event("content_block_stop", json!({"type": "content_block_stop", "index": index}));
                    }
                    yield named_event("message_delta", json!({"type": "message_delta", "delta": {"stop_reason": if stop_sequence.is_some() { "stop_sequence" } else { reason.anthropic(has_tool_calls) }, "stop_sequence": stop_sequence}, "usage": {"output_tokens": output_tokens, "input_tokens": input_tokens}}));
                    yield named_event("message_stop", json!({"type": "message_stop"}));
                    break;
                }
                GenerationEvent::Error { message } => {
                    terminated = true;
                    yield named_event("error", json!({"type": "error", "error": {"type": "api_error", "message": message}}));
                    break;
                }
            }
        }
        if !terminated {
            yield named_event("error", json!({"type": "error", "error": {"type": "api_error", "message": "the inference stream ended before completion"}}));
        }
    };
    Sse::new(output).keep_alive(KeepAlive::default())
}

fn named_event(name: &'static str, value: Value) -> Result<Event, Infallible> {
    Ok(Event::default().event(name).data(value.to_string()))
}
