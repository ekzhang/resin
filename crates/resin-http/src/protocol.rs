use std::{collections::BTreeMap, time::Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConversationRequest {
    pub id: String,
    pub model: String,
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub max_output_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<u32>,
    pub seed: Option<i64>,
    pub logprobs: bool,
    pub top_logprobs: usize,
    pub prompt_logprobs: bool,
    pub routed_dp_rank: Option<usize>,
    pub stop: Vec<String>,
    pub reasoning_effort: Option<String>,
    pub extra: BTreeMap<String, Value>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

impl FinishReason {
    pub const fn openai(self, has_tool_calls: bool) -> &'static str {
        if has_tool_calls {
            "tool_calls"
        } else {
            match self {
                Self::Stop => "stop",
                Self::Length => "length",
                Self::Cancelled => "stop",
            }
        }
    }

    pub const fn anthropic(self, has_tool_calls: bool) -> &'static str {
        if has_tool_calls {
            "tool_use"
        } else {
            match self {
                Self::Stop => "end_turn",
                Self::Length => "max_tokens",
                Self::Cancelled => "end_turn",
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum GenerationEvent {
    /// Internal observation of the first raw output token, before stream
    /// coalescing and parser buffering. HTTP protocol adapters omit this event.
    FirstToken,
    Started {
        input_tokens: usize,
    },
    ReasoningDelta(String),
    TextDelta {
        delta: String,
        /// Scores proven to belong to this visible text, independent of raw-token events.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        logprobs: Vec<GeneratedTokenLogprob>,
    },
    Logprobs(Vec<GeneratedTokenLogprob>),
    PromptLogprobs(Vec<Option<TokenLogprob>>),
    StopSequence(String),
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        index: usize,
        delta: String,
    },
    Finished {
        reason: FinishReason,
        input_tokens: usize,
        cached_tokens: usize,
        output_tokens: usize,
        elapsed_ms: f64,
        generation_elapsed_ms: f64,
        time_to_first_token_ms: f64,
    },
    Error {
        message: String,
    },
}

/// A vocabulary token, independent of response parsing and UTF-8 buffering.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub token: String,
    pub bytes: Option<Vec<u8>>,
    #[serde(deserialize_with = "deserialize_logprob")]
    pub logprob: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GeneratedTokenLogprob {
    #[serde(flatten)]
    pub token: TokenLogprob,
    pub token_index: usize,
    /// Only populated when the parser can prove the token's semantic channel.
    pub channel: Option<String>,
    pub top_logprobs: Vec<TokenLogprob>,
}

#[derive(Debug)]
pub struct GenerationJob {
    pub request: ConversationRequest,
    pub events: mpsc::Sender<GenerationEvent>,
    /// Frontend queue admission, before recipe rendering and engine submission.
    pub enqueued_at: Instant,
}

#[derive(Debug, Default)]
pub(crate) struct ResponseAccumulator {
    pub input_tokens: usize,
    pub cached_tokens: usize,
    pub output_tokens: usize,
    pub reasoning: String,
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub logprobs: Vec<GeneratedTokenLogprob>,
    pub content_logprobs: Vec<GeneratedTokenLogprob>,
    pub prompt_logprobs: Vec<Option<TokenLogprob>>,
    pub stop_sequence: Option<String>,
    pub reason: Option<FinishReason>,
    pub elapsed_ms: f64,
    pub generation_elapsed_ms: f64,
    pub time_to_first_token_ms: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ResponseAccumulator {
    /// Collect one non-streaming result. All wire protocols require an explicit
    /// completion event; dropping the actor's sender is never a partial success.
    pub async fn collect(mut receiver: mpsc::Receiver<GenerationEvent>) -> Result<Self, String> {
        let mut response = Self::default();
        while let Some(event) = receiver.recv().await {
            if let GenerationEvent::Error { message } = event {
                return Err(message);
            }
            response.apply(&event);
            if response.reason.is_some() {
                return Ok(response);
            }
        }
        Err("the inference engine ended the request without a result".to_owned())
    }

    pub fn apply(&mut self, event: &GenerationEvent) {
        match event {
            GenerationEvent::Started { input_tokens } => self.input_tokens = *input_tokens,
            GenerationEvent::ReasoningDelta(delta) => self.reasoning.push_str(delta),
            GenerationEvent::TextDelta { delta, logprobs } => {
                self.text.push_str(delta);
                self.content_logprobs.extend_from_slice(logprobs);
            }
            GenerationEvent::Logprobs(tokens) => self.logprobs.extend_from_slice(tokens),
            GenerationEvent::PromptLogprobs(tokens) => self.prompt_logprobs.clone_from(tokens),
            GenerationEvent::StopSequence(sequence) => self.stop_sequence = Some(sequence.clone()),
            GenerationEvent::ToolCallStart { index, id, name } => {
                if self.tool_calls.len() <= *index {
                    self.tool_calls.resize_with(*index + 1, ToolCall::default);
                }
                self.tool_calls[*index].id.clone_from(id);
                self.tool_calls[*index].name.clone_from(name);
            }
            GenerationEvent::ToolCallArgumentsDelta { index, delta } => {
                if self.tool_calls.len() <= *index {
                    self.tool_calls.resize_with(*index + 1, ToolCall::default);
                }
                self.tool_calls[*index].arguments.push_str(delta);
            }
            GenerationEvent::Finished {
                reason,
                input_tokens,
                cached_tokens,
                output_tokens,
                elapsed_ms,
                generation_elapsed_ms,
                time_to_first_token_ms,
            } => {
                self.reason = Some(*reason);
                self.input_tokens = *input_tokens;
                self.cached_tokens = *cached_tokens;
                self.output_tokens = *output_tokens;
                self.elapsed_ms = *elapsed_ms;
                self.generation_elapsed_ms = *generation_elapsed_ms;
                self.time_to_first_token_ms = *time_to_first_token_ms;
            }
            GenerationEvent::FirstToken | GenerationEvent::Error { .. } => {}
        }
    }
}

pub(crate) fn json_error(message: impl Into<String>, error_type: &str) -> Value {
    serde_json::json!({
        "error": {
            "message": message.into(),
            "type": error_type,
        }
    })
}

// JSON has no infinity literal. Null preserves a valid negative-infinite score
// across the private coordinator transport without changing finite values.
fn deserialize_logprob<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    Option::<f32>::deserialize(deserializer).map(|value| value.unwrap_or(f32::NEG_INFINITY))
}
