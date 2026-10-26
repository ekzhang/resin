//! Qwen3.5, Qwen3.8 and Ornith conversation formats.

mod parser;
pub use parser::{QwenParser, QwenTokens};

use crate::{
    Error, Prompt,
    render::{function, list, messages, option, whitespace},
};
use serde_json::Value;

#[derive(Default)]
struct Vision {
    images: usize,
    videos: usize,
    labels: bool,
}

impl Vision {
    fn content(&mut self, value: Option<&Value>, system: bool) -> Result<Prompt, Error> {
        let mut prompt = Prompt::new();
        match value {
            None | Some(Value::Null) => {}
            Some(Value::String(text)) => {
                prompt.text(text);
            }
            Some(Value::Array(parts)) => {
                for part in parts {
                    let kind = part.get("type").and_then(Value::as_str);
                    let image = part.get("image").is_some()
                        || part.get("image_url").is_some()
                        || kind == Some("image");
                    let video = part.get("video").is_some() || kind == Some("video");
                    if image || video {
                        if system {
                            return Err(Error::new(
                                "Qwen system message cannot contain images or videos",
                            ));
                        }
                        if image {
                            self.images += 1;
                        } else {
                            self.videos += 1;
                        }
                        if self.labels {
                            prompt.text(if image {
                                format!("Picture {}: ", self.images)
                            } else {
                                format!("Video {}: ", self.videos)
                            });
                        }
                        prompt
                            .special("<|vision_start|>")
                            .special(if image {
                                "<|image_pad|>"
                            } else {
                                "<|video_pad|>"
                            })
                            .special("<|vision_end|>");
                    } else if let Some(text) = part.get("text").and_then(Value::as_str) {
                        prompt.text(text);
                    } else {
                        return Err(Error::new("unexpected Qwen content item"));
                    }
                }
            }
            _ => return Err(Error::new("unexpected Qwen content type")),
        }
        Ok(prompt.trim())
    }
}

const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

fn start(prompt: &mut Prompt, role: &str) {
    prompt.special("<|im_start|>").text(role).text("\n");
}
fn end(prompt: &mut Prompt) {
    prompt.special("<|im_end|>").text("\n");
}

fn tool_instructions(prompt: &mut Prompt) {
    // The pinned template tokenizes its fixed examples as real tool markers.
    // Recognize these only in this trusted constant, never in caller text.
    let mut tail = TOOL_INSTRUCTIONS;
    while let Some((offset, marker)) = ["<tool_call>", "</tool_call>"]
        .iter()
        .filter_map(|marker| tail.find(marker).map(|offset| (offset, *marker)))
        .min_by_key(|(offset, _)| *offset)
    {
        prompt.text(&tail[..offset]).special(marker);
        tail = &tail[offset + marker.len()..];
    }
    prompt.text(tail);
}

/// Shared construction parameters for the Qwen-family formats.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct QwenRecipeParams {
    pub add_generation_prompt: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub enable_thinking: Option<bool>,
    pub preserve_thinking: Option<bool>,
    pub add_vision_id: Option<bool>,
    #[serde(skip_serializing)]
    pub media_config: Option<crate::media::qwen::Config>,
}

/// Checkpoint template variants rendered by [`QwenRecipe`].
pub const QWEN35: u8 = 0;
pub const QWEN38: u8 = 1;
/// Ornith's modified Qwen3.5 template: always preserves past reasoning and
/// renders non-string tool arguments as JSON.
pub const ORNITH: u8 = 2;

pub type Qwen35Recipe = QwenRecipe<QWEN35>;
pub type Qwen38Recipe = QwenRecipe<QWEN38>;
pub type OrnithRecipe = QwenRecipe<ORNITH>;

/// One family implementation with a compile-time checkpoint-format variant.
#[derive(Clone, Debug, Default)]
pub struct QwenRecipe<const FORMAT: u8> {
    params: QwenRecipeParams,
}
impl<const FORMAT: u8> QwenRecipe<FORMAT> {
    pub fn new(params: QwenRecipeParams) -> Self {
        Self { params }
    }
}
impl<const FORMAT: u8> crate::Recipe for QwenRecipe<FORMAT> {
    fn render(&self, request: &Value) -> Result<Prompt, Error> {
        let request = crate::recipe::request("Qwen", &self.params, request)?;
        let request = &request;
        let messages = messages(request, "Qwen")?;
        let variant35 = FORMAT != QWEN38;
        let thinking = option(request, "enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        // Ornith's template renders every past reasoning block unconditionally.
        let preserve = FORMAT == ORNITH
            || option(request, "preserve_thinking")
                .and_then(Value::as_bool)
                .unwrap_or(!variant35);
        let python_literals = FORMAT == QWEN35;
        let effort = option(request, "reasoning_effort")
            .and_then(Value::as_str)
            .unwrap_or("xhigh");
        let instructions = if variant35 || !thinking {
            ""
        } else {
            match effort {
                "xhigh" => {
                    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
                }
                "low" => {
                    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
                }
                "medium" => "",
                _ => {
                    return Err(Error::new(
                        "Qwen3.8 reasoning_effort must be xhigh, medium, or low",
                    ));
                }
            }
        };
        let tools = list(&request["tools"]);
        let first_system = messages[0].get("role").and_then(Value::as_str) == Some("system");
        let system = if first_system {
            Vision::default().content(messages[0].get("content"), true)?
        } else {
            Prompt::new()
        };
        let mut prompt = Prompt::new();
        if !tools.is_empty()
            || !instructions.is_empty()
            || (first_system && (variant35 || !system.rendered_text().is_empty()))
        {
            start(&mut prompt, "system");
            if !instructions.is_empty() {
                prompt.text(instructions);
                if !tools.is_empty() || !system.rendered_text().is_empty() {
                    prompt.text("\n\n");
                }
            }
            if !tools.is_empty() {
                prompt.text("# Tools\n\nYou have access to the following functions:\n\n<tools>");
                for tool in tools {
                    prompt.text("\n").text(crate::json::spaced(tool));
                }
                prompt.text("\n</tools>");
                tool_instructions(&mut prompt);
                if !system.rendered_text().is_empty() {
                    prompt.text("\n\n");
                }
            }
            prompt.append(system);
            end(&mut prompt);
        }
        let mut last_query = None;
        for (index, item) in messages.iter().enumerate().rev() {
            if item.get("role").and_then(Value::as_str) == Some("user") {
                let content = Vision::default()
                    .content(item.get("content"), false)?
                    .rendered_text();
                if !(content.starts_with("<tool_response>")
                    && content.ends_with("</tool_response>"))
                {
                    last_query = Some(index);
                    break;
                }
            }
        }
        let last_query =
            last_query.ok_or_else(|| Error::new("no user query found in Qwen messages"))?;
        let mut vision = Vision {
            labels: option(request, "add_vision_id")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            ..Default::default()
        };
        for (index, item) in messages.iter().enumerate() {
            let role = item.get("role").and_then(Value::as_str).unwrap_or("");
            let mut body = vision.content(item.get("content"), role == "system")?;
            match role {
                "system" => {
                    if index != 0 {
                        return Err(Error::new("Qwen system message must be first"));
                    }
                }
                "user" => {
                    start(&mut prompt, "user");
                    prompt.append(body);
                    end(&mut prompt);
                }
                "assistant" => {
                    let mut reasoning = Prompt::new();
                    reasoning.text(
                        item.get("reasoning_content")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .trim_matches(whitespace),
                    );
                    if variant35 && !item.get("reasoning_content").is_some_and(Value::is_string) {
                        let text = body.rendered_text();
                        if let Some((before, _)) = text.split_once("</think>") {
                            let before = before.trim_end_matches('\n');
                            let first = before
                                .rfind("<think>")
                                .map_or(0, |index| index + "<think>".len());
                            let thinking = before[first..].trim_start_matches(whitespace);
                            let begin = before.len() - thinking.len();
                            reasoning = body
                                .slice(begin..begin + thinking.trim_end_matches(whitespace).len());
                            let last = text.rfind("</think>").unwrap() + "</think>".len();
                            let suffix = text[last..].trim_start_matches('\n');
                            let begin = text.len() - suffix.len();
                            body = body.slice(begin..text.len());
                        }
                    }
                    let has_text = !body.rendered_text().is_empty();
                    start(&mut prompt, "assistant");
                    if preserve || index > last_query {
                        prompt.special("<think>").text("\n");
                        prompt.append(reasoning);
                        prompt.text("\n").special("</think>").text("\n\n");
                    }
                    prompt.append(body);
                    let calls = list(&item["tool_calls"]);
                    for (call_index, call) in calls.iter().enumerate() {
                        let function = function(call);
                        if call_index > 0 {
                            prompt.text("\n");
                        } else if has_text {
                            prompt.text("\n\n");
                        }
                        prompt
                            .special("<tool_call>")
                            .text("\n<function=")
                            .text(function.get("name").and_then(Value::as_str).unwrap_or(""))
                            .text(">\n");
                        let arguments = function.get("arguments");
                        let parsed = arguments
                            .and_then(Value::as_str)
                            .filter(|text| !text.is_empty())
                            .and_then(|text| serde_json::from_str::<Value>(text).ok());
                        if let Some(arguments) =
                            parsed.as_ref().or(arguments).and_then(Value::as_object)
                        {
                            for (key, value) in arguments {
                                let text = match value {
                                    Value::String(text) => text.clone(),
                                    Value::Bool(value) if python_literals => {
                                        if *value {
                                            "True".into()
                                        } else {
                                            "False".into()
                                        }
                                    }
                                    Value::Null if python_literals => "None".into(),
                                    _ => crate::json::spaced(value),
                                };
                                prompt
                                    .text("<parameter=")
                                    .text(key)
                                    .text(">\n")
                                    .text(text)
                                    .text("\n</parameter>\n");
                            }
                        }
                        prompt.text("</function>\n").special("</tool_call>");
                    }
                    end(&mut prompt);
                }
                "tool" => {
                    if index > 0
                        && messages[index - 1].get("role").and_then(Value::as_str) != Some("tool")
                    {
                        prompt.special("<|im_start|>").text("user");
                    }
                    prompt.text("\n").special("<tool_response>").text("\n");
                    prompt.append(body);
                    prompt.text("\n").special("</tool_response>");
                    if index + 1 == messages.len()
                        || messages[index + 1].get("role").and_then(Value::as_str) != Some("tool")
                    {
                        end(&mut prompt);
                    }
                }
                _ => return Err(Error::new("unexpected Qwen message role")),
            }
        }
        if option(request, "add_generation_prompt")
            .and_then(Value::as_bool)
            .unwrap_or(true)
        {
            start(&mut prompt, "assistant");
            prompt.special("<think>").text("\n");
            if !thinking {
                prompt.text("\n").special("</think>").text("\n\n");
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
    ) -> Result<Box<dyn crate::ResponseParser>, Error> {
        Ok(Box::new(QwenParser::new(
            QwenTokens::resolve(encoder)?,
            decoder,
            prompt_ids,
            &request["tools"],
        )?))
    }
    fn stop_token_ids(&self, encoder: &dyn crate::Encoder) -> Result<Vec<u32>, Error> {
        Ok(vec![
            encoder.special_id("<|im_end|>")?,
            encoder.special_id("<|endoftext|>")?,
        ])
    }
    fn media_config(&self) -> Option<crate::MediaConfig> {
        Some(crate::MediaConfig::Qwen(
            self.params.media_config.clone().unwrap_or_default(),
        ))
    }
}
