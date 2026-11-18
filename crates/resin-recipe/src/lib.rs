//! Model conversation formats, independent of Python and HTTP protocols.

pub mod deepseek;
pub mod gemma;
pub mod glm;
pub mod gpt_oss;
pub mod inkling2;
mod json;
pub mod lfm2;
pub mod media;
mod parser;
mod prompt;
mod provenance;
pub mod qwen;
mod recipe;
mod render;
mod tokenizer;

use std::fmt;

use serde::{Deserialize, Serialize};

pub use deepseek::{DeepSeekV41Parser, DeepSeekV41Tokens};
pub use gemma::{Gemma4Parser, Gemma4Tokens};
pub use glm::{GlmParser, GlmTokens};
pub use gpt_oss::{GptOssParser, GptOssTokens};
pub use inkling2::{Inkling2Parser, Inkling2Tokens};
pub use lfm2::{Lfm2Parser, Lfm2Tokens};
pub use prompt::{Encoder, Prompt, PromptSpan, TokenizerEncoder};
pub use provenance::{AttributedEvent, AttributedText, TokenSpan};
pub use qwen::{QwenParser, QwenTokens};
pub use recipe::{
    BlockKind, MediaProcessor, MessagePolicy, ReasoningEffort, Recipe, RecipeDefaults, ToolScope,
};
pub use tokenizer::{TokenBytes, Tokenizer, TokenizerDecoder};

/// A semantic response event, shared by streaming and complete responses.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    ReasoningDelta {
        delta: String,
    },
    TextDelta {
        delta: String,
    },
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        index: usize,
        delta: String,
    },
}

/// Why token generation ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
}

/// A model-format or tokenizer failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error(String);

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Decode tokens incrementally without revising previously returned text.
///
/// Each call returns only new stable text. `finish` flushes any buffered suffix;
/// `reset` starts a fresh decoding segment at a model control-token boundary.
pub trait Decoder: Send {
    fn push(&mut self, ids: &[u32], token_start: u32) -> Result<AttributedText, Error>;
    fn finish(&mut self) -> Result<AttributedText, Error>;
    fn reset(&mut self);
}

/// Parse generated tokens into semantic events.
pub trait ResponseParser: Send {
    /// A proven semantic lane for this complete token before it is fed.
    /// `None` means a header, mixed token, or grammar with deferred transforms.
    fn token_channel(&self, _id: u32) -> Option<&'static str> {
        None
    }

    fn feed(&mut self, ids: &[u32]) -> Result<Vec<AttributedEvent>, Error>;
    fn finish(&mut self, reason: FinishReason) -> Result<Vec<AttributedEvent>, Error>;
}
