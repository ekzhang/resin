//! Model-owned recipe contracts, independent of Python and serving.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{Decoder, Encoder, Error, Prompt, ResponseParser};

/// Lazily supplies processor metadata or environment-dependent recipe defaults.
pub type RecipeDefaults<'a> = dyn Fn(&str) -> Result<Value, Error> + 'a;

/// A model's complete conversation format and request policy.
///
/// Rendering combines constructor defaults with request options. The same
/// immutable recipe owns the defaults used when constructing a parser.
pub trait Recipe: Send + Sync {
    fn from_config(params: Value, defaults: &RecipeDefaults<'_>) -> Result<Self, Error>
    where
        Self: Sized;

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
    fn input_modalities(&self) -> &'static [&'static str] {
        &["text"]
    }
    fn prepare_media(
        &self,
        _: Vec<crate::media::Source>,
        _: &dyn Encoder,
    ) -> Result<crate::media::Prepared, Error> {
        Err(Error::new("this recipe does not support media"))
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

/// Effort controls that accept either an API name or a numeric model budget.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ReasoningEffort {
    Name(String),
    Number(serde_json::Number),
}

pub(crate) fn params<T: DeserializeOwned>(value: Value) -> Result<T, Error> {
    serde_json::from_value(value).map_err(|error| Error::new(error.to_string()))
}

pub(crate) fn default_if_null<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

pub(crate) fn default(
    params: &mut Value,
    key: &str,
    value: impl FnOnce() -> Result<Value, Error>,
) -> Result<(), Error> {
    let params = params
        .as_object_mut()
        .ok_or_else(|| Error::new("recipe parameters must be an object"))?;
    if params.get(key).is_none_or(Value::is_null) {
        params.insert(key.into(), value()?);
    }
    Ok(())
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
