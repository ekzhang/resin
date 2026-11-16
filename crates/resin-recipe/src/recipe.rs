//! Model-owned recipe contracts, independent of Python and serving.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{Decoder, Encoder, Error, Prompt, ResponseParser, media};

/// A model's complete conversation format and request policy.
///
/// Rendering combines constructor defaults with request options. The same
/// immutable recipe owns the defaults used when constructing a parser.
pub trait Recipe: Send + Sync {
    fn message_policy(&self) -> MessagePolicy {
        MessagePolicy::STANDARD
    }
    fn render(&self, request: &Value) -> Result<Prompt, Error>;
    fn create_parser(
        &self,
        encoder: &dyn Encoder,
        decoder: Box<dyn Decoder>,
        request: &Value,
        prompt_ids: &[u32],
    ) -> Result<Box<dyn ResponseParser>, Error>;
    fn stop_token_ids(&self, encoder: &dyn Encoder) -> Result<Vec<u32>, Error>;
    fn media_config(&self) -> Option<MediaConfig> {
        None
    }
}

/// Structural message requirements supplied by a concrete model recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MessagePolicy {
    pub roles: &'static [&'static str],
    pub assistant_order: &'static [BlockKind],
    pub reorder_wire_fields: bool,
    pub controls: &'static [&'static str],
    pub tools: ToolScope,
    pub media_roles: Option<&'static [&'static str]>,
}
impl MessagePolicy {
    pub const STANDARD: Self = Self {
        roles: &["system", "developer", "user", "assistant", "tool"],
        assistant_order: &[BlockKind::Thinking, BlockKind::Text, BlockKind::Call],
        reorder_wire_fields: false,
        controls: &[],
        tools: ToolScope::InitialInstruction,
        media_roles: None,
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolScope {
    InitialInstruction,
    AnyInstruction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Thinking,
    Text,
    Call,
    Result,
}

/// Model-owned media geometry captured when a recipe is constructed.
#[derive(Clone, Debug)]
pub enum MediaConfig {
    DeepSeek(media::deepseek::Config, Option<u32>),
    Gemma(media::gemma::Config),
    Glm(media::glm::Config),
    Inkling2(media::inkling2::Config),
    Qwen(media::qwen::Config),
}

/// Effort controls that accept either an API name or a numeric model budget.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ReasoningEffort {
    Name(String),
    Number(serde_json::Number),
}

pub(crate) fn options<T>(name: &str, params: &T, request: &Value) -> Result<T, Error>
where
    T: Serialize + DeserializeOwned,
{
    let mut options = match serde_json::to_value(params).expect("recipe params serialize to JSON") {
        Value::Object(options) => options,
        _ => unreachable!("recipe params serialize to an object"),
    };
    let extra = request
        .get("extra")
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_object()
                .ok_or_else(|| Error::new("recipe extra options must be an object"))
        })
        .transpose()?;
    let root = request
        .as_object()
        .into_iter()
        .flat_map(|root| root.iter())
        .filter(|(key, _)| !["messages", "tools", "extra"].contains(&key.as_str()));
    for (key, value) in extra
        .into_iter()
        .flat_map(|extra| extra.iter())
        .chain(root)
        .filter(|(_, value)| !value.is_null())
    {
        if !options.contains_key(key) {
            return Err(Error::new(format!(
                "{name} does not support rendering option {key}"
            )));
        }
        options.insert(key.clone(), value.clone());
    }
    options.retain(|_, value| !value.is_null());
    serde_json::from_value(Value::Object(options))
        .map_err(|error| Error::new(format!("invalid {name} options: {error}")))
}
