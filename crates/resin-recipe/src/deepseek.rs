//! Checkpoint-owned DeepSeek V4.1 conversation rendering and DSML parsing.

mod parser;

use serde_json::Value;

use crate::json::spaced as json_spaced;

use crate::render::{function, option, text_content};
use crate::{Error, Prompt};

pub use parser::{DeepSeekV41Parser, DeepSeekV41Tokens};

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const THINKING_START: &str = "<think>";
pub const THINKING_END: &str = "</think>";
pub const DSML: &str = "｜DSML｜";
pub const IMAGE: &str = "<｜deepseek_image｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const SYSTEM: &str = "<｜System｜>";
const REMINDER: &str = "<｜latest_reminder｜>";

pub(crate) const TAGS: (&str, &str, &str) = (" calls", " invoke", " parameter");

pub(crate) fn settings(request: &Value) -> Result<(bool, String), Error> {
    let effort = option(request, "reasoning_effort");
    let budget = match effort {
        None => 75,
        Some(Value::String(value)) => match value.as_str() {
            "low" => 50,
            "high" => 75,
            "max" => 100,
            _ => {
                return Err(Error::new(
                    "V4.1 reasoning_effort must be low, high, max, or an integer in 1..100",
                ));
            }
        },
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=100).contains(n))
            .ok_or_else(|| Error::new("V4.1 reasoning_effort must be an integer in 1..100"))?,
    };
    Ok((
        thinking_mode(request, effort.is_some())?,
        format!(
            "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        ),
    ))
}

fn starts_turn(index: usize, message: &Value) -> bool {
    (message["role"] == "user" || (message["role"] == "system" && index > 0))
        && message["content_blocks"].as_array().is_none_or(|blocks| {
            blocks.is_empty()
                || blocks
                    .iter()
                    .any(|b| matches!(b["type"].as_str(), Some("text" | "image" | "image_url")))
        })
}

fn thinking_mode(request: &Value, configured_effort: bool) -> Result<bool, Error> {
    let extra = &request["extra"];
    Ok(
        match extra.get("thinking_mode").filter(|value| !value.is_null()) {
            Some(Value::String(value)) if value == "chat" => false,
            Some(Value::String(value)) if value == "thinking" => true,
            Some(_) => return Err(Error::new("thinking_mode must be 'chat' or 'thinking'")),
            None => configured_effort || extra["thinking"]["type"] == "enabled",
        },
    )
}

fn content(message: &Value) -> Result<String, Error> {
    text_value(&message["content"])
}

fn text_value(value: &Value) -> Result<String, Error> {
    text_content(value, |_| {
        Err(Error::new("DeepSeek V4.1 text blocks must contain text"))
    })
}

fn merge_messages(messages: Vec<Value>) -> Result<Vec<Value>, Error> {
    let mut merged: Vec<Value> = Vec::new();
    for mut message in messages {
        let role = message["role"].as_str();
        if matches!(role, Some("user" | "tool")) {
            let existing = (role == Some("user"))
                .then(|| {
                    message["content_blocks"]
                        .as_array()
                        .or_else(|| message["content"].as_array())
                })
                .flatten();
            let blocks = if let Some(blocks) = existing {
                blocks.clone()
            } else if role == Some("tool") {
                vec![
                    serde_json::json!({"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": content(&message)?}),
                ]
            } else {
                vec![serde_json::json!({"type": "text", "text": content(&message)?})]
            };
            if let Some(previous) = merged.last_mut().filter(|previous| {
                previous["role"] == "user"
                    && previous["content_blocks"].is_array()
                    && (role == Some("tool") || previous["task"].is_null())
            }) {
                previous["content_blocks"]
                    .as_array_mut()
                    .unwrap()
                    .extend(blocks);
            } else {
                message["role"] = Value::String("user".into());
                message["content_blocks"] = Value::Array(blocks);
                merged.push(message);
            }
        } else {
            merged.push(message);
        }
    }
    let mut tool_order = Vec::<String>::new();
    for message in &mut merged {
        if message["role"] == "assistant" {
            if let Some(calls) = message["tool_calls"]
                .as_array()
                .filter(|calls| !calls.is_empty())
            {
                tool_order = calls
                    .iter()
                    .map(|call| {
                        call["id"]
                            .as_str()
                            .or_else(|| call["function"]["id"].as_str())
                            .unwrap_or("")
                            .to_owned()
                    })
                    .collect();
            }
        } else if let Some(blocks) = message["content_blocks"].as_array_mut() {
            let mut results: Vec<_> = blocks
                .iter()
                .filter(|block| block["type"] == "tool_result")
                .cloned()
                .collect();
            results.sort_by_key(|block| {
                tool_order
                    .iter()
                    .position(|id| Some(id.as_str()) == block["tool_use_id"].as_str())
                    .unwrap_or(0)
            });
            let mut results = results.into_iter();
            for block in blocks {
                if block["type"] == "tool_result" {
                    *block = results.next().unwrap();
                }
            }
        }
    }
    Ok(merged)
}

fn render_tools(prompt: &mut Prompt, tools: &[Value]) -> Result<(), Error> {
    prompt.template(
        include_str!("deepseek/tools.txt"),
        &[DSML, THINKING_START, THINKING_END],
    );
    for tool in tools {
        let tool = qualified_function(tool)?;
        if !tool.is_object() {
            return Err(Error::new("tool schema must be an object"));
        }
        prompt.text(json_spaced(&tool));
        prompt.text("\n");
    }
    prompt.text("\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n");
    Ok(())
}

fn tag(prompt: &mut Prompt, prefix: &str, suffix: &str) {
    prompt.text(prefix);
    prompt.special(DSML);
    prompt.text(suffix);
}

fn render_tool_calls(prompt: &mut Prompt, calls: &[Value]) -> Result<(), Error> {
    let (block, invoke, parameter) = TAGS;
    tag(prompt, "\n\n<", &format!("{block}>\n"));
    for (index, call) in calls.iter().enumerate() {
        if index > 0 {
            prompt.text("\n");
        }
        let function = qualified_function(call)?;
        let name = function["name"]
            .as_str()
            .ok_or_else(|| Error::new("tool call name must be a string"))?;
        tag(prompt, "<", &format!("{invoke} name=\""));
        prompt.text(name);
        prompt.text("\">\n");
        let original = &function["arguments"];
        let mut arguments = original.clone();
        for _ in 0..2 {
            if let Some(text) = arguments.as_str()
                && let Ok(parsed) = serde_json::from_str(text)
            {
                arguments = parsed;
            }
        }
        let arguments = arguments.as_object().cloned().unwrap_or_else(|| {
            serde_json::Map::from_iter([("arguments".to_owned(), original.clone())])
        });
        for (index, (key, value)) in arguments.iter().enumerate() {
            if index > 0 {
                prompt.text("\n");
            }
            tag(prompt, "<", &format!("{parameter} name=\""));
            prompt.text(key);
            prompt.text(if value.is_string() {
                "\" string=\"true\">"
            } else {
                "\" string=\"false\">"
            });
            prompt.text(
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| json_spaced(value)),
            );
            tag(prompt, "</", &format!("{parameter}>"));
        }
        tag(prompt, "\n</", &format!("{invoke}>"));
    }
    tag(prompt, "\n</", &format!("{block}>"));
    Ok(())
}

fn qualified_function(value: &Value) -> Result<Value, Error> {
    let mut result = function(value).clone();
    let namespace = value
        .get("namespace")
        .filter(|v| !v.is_null())
        .or_else(|| result.get("namespace").filter(|v| !v.is_null()));
    let explicit = namespace.and_then(|v| v.as_str().or_else(|| v["name"].as_str()));
    let name = result["name"]
        .as_str()
        .ok_or_else(|| Error::new("tool name must be a string"))?;
    let (prefix, name) = name
        .split_once("::")
        .map_or((explicit, name), |(prefix, name)| (Some(prefix), name));
    if prefix.is_some_and(|p| p.contains("::") || explicit.is_some_and(|e| e != p))
        || name.contains("::")
    {
        return Err(Error::new("invalid or conflicting tool namespace"));
    }
    let qualified = prefix.map_or_else(|| name.to_owned(), |p| format!("{p}::{name}"));
    let description = namespace
        .and_then(|n| n["description"].as_str())
        .map(str::to_owned);
    if let Some(description) = description {
        result["description"] = Value::String(format!(
            "{description}\n{}",
            result["description"].as_str().unwrap_or("")
        ));
    }
    result["name"] = Value::String(qualified);
    result.as_object_mut().unwrap().remove("namespace");
    Ok(result)
}

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeepSeekV41RecipeParams {
    pub add_generation_prompt: Option<bool>,
    pub reasoning_effort: Option<crate::ReasoningEffort>,
    pub thinking_mode: Option<String>,
    pub enable_thinking: Option<bool>,
    pub response_format: Option<Value>,
    #[serde(skip_serializing)]
    pub vision_config: Option<crate::media::deepseek::Config>,
    #[serde(skip_serializing)]
    pub image_token_id: Option<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct DeepSeekV41Recipe {
    params: DeepSeekV41RecipeParams,
}

impl DeepSeekV41Recipe {
    pub fn new(mut params: DeepSeekV41RecipeParams) -> Result<Self, Error> {
        if let Some(enabled) = params.enable_thinking.take() {
            let mode = if enabled { "thinking" } else { "chat" };
            if params
                .thinking_mode
                .as_deref()
                .is_some_and(|value| value != mode)
            {
                return Err(Error::new("conflicting enable_thinking and thinking_mode"));
            }
            params.thinking_mode = Some(mode.into());
        }
        Ok(Self { params })
    }
    fn request(&self, raw: &Value) -> Result<Value, crate::Error> {
        let params = crate::recipe::options("deepseek_v41", &self.params, raw)?;
        let mut request = serde_json::json!({
            "messages": raw["messages"], "tools": raw["tools"], "extra": params,
        });
        let options = request["extra"].as_object_mut().unwrap();
        options.retain(|_, value| !value.is_null());
        if option(raw, "enable_thinking").is_some() && option(raw, "thinking_mode").is_none() {
            options.remove("thinking_mode");
        }
        if let Some(enabled) = options.remove("enable_thinking") {
            let mode = if enabled == true { "thinking" } else { "chat" };
            if options
                .get("thinking_mode")
                .is_some_and(|value| value != mode)
            {
                return Err(crate::Error::new(
                    "conflicting enable_thinking and thinking_mode",
                ));
            }
            options.insert("thinking_mode".into(), Value::String(mode.into()));
        }
        Ok(request)
    }
}
impl crate::Recipe for DeepSeekV41Recipe {
    fn message_policy(&self) -> crate::MessagePolicy {
        crate::MessagePolicy {
            roles: &[
                "system",
                "developer",
                "user",
                "assistant",
                "tool",
                "latest_reminder",
            ],
            media_roles: Some(&["user"]),
            controls: &["task", "wo_eos", "response_format"],
            tools: crate::ToolScope::AnyInstruction,
            ..crate::MessagePolicy::STANDARD
        }
    }
    fn render(&self, request: &Value) -> Result<crate::Prompt, crate::Error> {
        let request = self.request(request)?;
        let request = &request;
        let (thinking, effort_prefix) = settings(request)?;
        let mut messages = request["messages"]
            .as_array()
            .ok_or_else(|| Error::new("messages must be an array"))?
            .clone();
        if messages.is_empty() {
            return Err(Error::new("conversation must contain at least one message"));
        }
        let tools = request["tools"]
            .as_array()
            .filter(|tools| !tools.is_empty());
        let response_format = request["extra"]
            .get("response_format")
            .filter(|value| !value.is_null());
        if tools.is_some() || response_format.is_some() {
            if messages[0]["role"] != "system" {
                messages.insert(0, serde_json::json!({"role": "system", "content": ""}));
            }
            if let Some(tools) = tools {
                messages[0]["tools"] = Value::Array(tools.clone());
            }
            if let Some(format) = response_format {
                messages[0]["response_format"] = format.clone();
            }
        }
        let messages = merge_messages(messages)?;
        let drop_thinking = !messages.iter().any(|message| {
            message["tools"]
                .as_array()
                .is_some_and(|tools| !tools.is_empty())
        });
        let last_user = messages
            .iter()
            .enumerate()
            .rposition(|(index, message)| starts_turn(index, message));
        let messages: Vec<_> = messages
            .into_iter()
            .enumerate()
            .filter_map(|(index, mut message)| {
                if thinking && drop_thinking && last_user.is_some_and(|last| index < last) {
                    match message["role"].as_str() {
                        Some("developer") => return None,
                        Some("assistant") => {
                            message.as_object_mut().unwrap().remove("reasoning_content");
                        }
                        _ => {}
                    }
                }
                Some(message)
            })
            .collect();
        let last_user = messages
            .iter()
            .enumerate()
            .rposition(|(index, message)| starts_turn(index, message));
        let mut prompt = Prompt::new();
        prompt.special(BOS);
        if thinking || messages[0]["role"] == "system" {
            prompt.special(SYSTEM);
        }
        if thinking {
            prompt.text(effort_prefix);
        }
        for (index, message) in messages.iter().enumerate() {
            let role = message["role"]
                .as_str()
                .ok_or_else(|| Error::new("message role must be a string"))?;
            match role {
                "system" | "developer" => {
                    if role == "developer" {
                        return Err(Error::new(
                            "DeepSeek V4.1 uses system messages for instructions",
                        ));
                    }
                    if index > 0 {
                        prompt.special(SYSTEM);
                    }
                    prompt.text(content(message)?);
                    if let Some(tools) = message["tools"]
                        .as_array()
                        .filter(|tools| !tools.is_empty())
                    {
                        prompt.text("\n\n");
                        render_tools(&mut prompt, tools)?;
                    }
                    if let Some(format) = message
                        .get("response_format")
                        .filter(|value| !value.is_null())
                    {
                        prompt.text("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                        prompt.text(json_spaced(format));
                    }
                }
                "user" => {
                    prompt.special(USER);
                    if let Some(blocks) = message["content_blocks"].as_array() {
                        for (index, block) in blocks.iter().enumerate() {
                            if index > 0 {
                                prompt.text("\n\n");
                            }
                            match block["type"].as_str() {
                                Some("text") => {
                                    prompt.text(block["text"].as_str().unwrap_or(""));
                                }
                                Some("image" | "image_url") => {
                                    prompt.special(IMAGE);
                                }
                                Some("tool_result") => {
                                    prompt.text("<tool_result>");
                                    prompt.text(text_value(&block["content"])?);
                                    prompt.text("</tool_result>");
                                }
                                _ => {
                                    return Err(Error::new(
                                        "unsupported DeepSeek user content block",
                                    ));
                                }
                            }
                        }
                    } else {
                        prompt.text(content(message)?);
                    }
                }
                "latest_reminder" => {
                    prompt.special(REMINDER);
                    prompt.text(content(message)?);
                }
                "assistant" => {
                    let after_last_user = last_user.is_none_or(|last| index > last);
                    let previous_has_task = index > 0
                        && messages[index - 1]
                            .get("task")
                            .is_some_and(|value| !value.is_null());
                    if thinking && !previous_has_task && (!drop_thinking || after_last_user) {
                        prompt.text(message["reasoning_content"].as_str().unwrap_or(""));
                        prompt.special(THINKING_END);
                    }
                    prompt.text(content(message)?);
                    if let Some(calls) = message["tool_calls"]
                        .as_array()
                        .filter(|calls| !calls.is_empty())
                    {
                        render_tool_calls(&mut prompt, calls)?;
                    }
                    if message["wo_eos"] != true {
                        prompt.special(EOS);
                    }
                }
                _ => {
                    return Err(Error::new(format!(
                        "unsupported DeepSeek message role: {role}"
                    )));
                }
            }
            let follows_assistant = messages.get(index + 1).is_none_or(|next| {
                matches!(next["role"].as_str(), Some("assistant" | "latest_reminder"))
            });
            if follows_assistant
                && (index + 1 < messages.len()
                    || option(request, "add_generation_prompt") != Some(&Value::Bool(false)))
            {
                if let Some(task) = message.get("task").filter(|value| !value.is_null()) {
                    let task = task
                        .as_str()
                        .ok_or_else(|| Error::new("DeepSeek task must be a string"))?;
                    if ![
                        "action",
                        "query",
                        "authority",
                        "domain",
                        "title",
                        "read_url",
                    ]
                    .contains(&task)
                    {
                        return Err(Error::new("unsupported DeepSeek task"));
                    }
                    if task == "action" {
                        prompt.special(ASSISTANT);
                        prompt.special(if thinking {
                            THINKING_START
                        } else {
                            THINKING_END
                        });
                    }
                    prompt.special(format!("<｜{task}｜>"));
                } else if role == "user" || (role == "system" && index > 0) {
                    prompt.special(ASSISTANT);
                    prompt.special(
                        if thinking
                            && (!drop_thinking || last_user.is_none_or(|last| index >= last))
                        {
                            THINKING_START
                        } else {
                            THINKING_END
                        },
                    );
                }
            }
        }
        Ok(prompt)
    }
    fn create_parser(
        &self,
        encoder: &dyn crate::Encoder,
        decoder: Box<dyn crate::Decoder>,
        request: &Value,
        prompt_ids: &[u32],
    ) -> Result<Box<dyn crate::ResponseParser>, crate::Error> {
        let request = self.request(request)?;
        Ok(Box::new(DeepSeekV41Parser::new(
            DeepSeekV41Tokens::resolve(encoder)?,
            decoder,
            prompt_ids,
            settings(&request)?.0,
        )?))
    }
    fn stop_token_ids(&self, encoder: &dyn crate::Encoder) -> Result<Vec<u32>, crate::Error> {
        Ok(vec![encoder.special_id(EOS)?])
    }
    fn media_config(&self) -> Option<crate::MediaConfig> {
        Some(crate::MediaConfig::DeepSeek(
            self.params.vision_config.clone().unwrap_or_default(),
            self.params.image_token_id,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_defaults_and_explicit_mode_match_v41() {
        assert!(!settings(&serde_json::json!({})).unwrap().0);
        let (thinking, effort) =
            settings(&serde_json::json!({"reasoning_effort": "high"})).unwrap();
        assert!(thinking);
        assert!(effort.starts_with("Reasoning Effort: 75 "));
        assert!(
            !settings(
                &serde_json::json!({"reasoning_effort": "max", "extra": {"thinking_mode": "chat"}})
            )
            .unwrap()
            .0
        );
        assert!(
            settings(&serde_json::json!({"extra": {"thinking": {"type": "enabled"}}}))
                .unwrap()
                .0
        );
        assert!(settings(&serde_json::json!({"reasoning_effort": "medium"})).is_err());
        let recipe = DeepSeekV41Recipe::new(
            serde_json::from_value(serde_json::json!({"enable_thinking": true})).unwrap(),
        )
        .unwrap();
        let chat = crate::Recipe::render(&recipe, &serde_json::json!({"messages": [{"role": "user", "content": "hello"}], "extra": {"thinking_mode": "chat"}})).unwrap();
        assert!(!chat.rendered_text().ends_with(THINKING_START));
        let alias_chat = crate::Recipe::render(&recipe, &serde_json::json!({"messages": [{"role": "user", "content": "hello"}], "extra": {"enable_thinking": false}})).unwrap();
        assert_eq!(chat.rendered_text(), alias_chat.rendered_text());
        assert!(
            DeepSeekV41Recipe::new(
                serde_json::from_value(
                    serde_json::json!({"enable_thinking": true, "thinking_mode": "chat"})
                )
                .unwrap()
            )
            .is_err()
        );
    }
}
