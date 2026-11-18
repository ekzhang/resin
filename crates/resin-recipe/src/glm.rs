//! GLM-5.3-Flash format, following the pinned NVIDIA checkpoint template.
//! Source: nvidia/GLM-5.3-Flash-NVFP4@da920bb0b9f4a06727223a349e55468e38352348.
//! The original Z.AI MIT notice is retained in licenses/glm-MIT.txt.

pub mod media;
mod parser;
pub use parser::{GlmParser, GlmTokens};

use crate::{
    Error, Prompt,
    render::{function, list, messages, object_arguments, whitespace},
};
use serde_json::Value;

const TOOL_PREFIX: &str = "\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
const TOOL_SUFFIX: &str = "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

/// The released template always opens a reasoning segment. Effort selects its
/// instruction; `clear_thinking` removes reasoning before the latest user turn.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Glm53FlashRecipe {
    pub add_generation_prompt: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub clear_thinking: Option<bool>,
    #[serde(skip_serializing)]
    pub media_config: Option<media::Processor>,
}

fn visible_content(value: &Value) -> Result<Prompt, Error> {
    let mut out = Prompt::new();
    match value {
        Value::Null => {}
        Value::String(text) => {
            out.text(text);
        }
        Value::Array(parts) => {
            for part in parts {
                if let Some(text) = part.as_str() {
                    out.text(text);
                    continue;
                }
                match part["type"].as_str() {
                    Some("text") => {
                        out.text(
                            part["text"]
                                .as_str()
                                .ok_or_else(|| Error::new("GLM text must be a string"))?,
                        );
                    }
                    Some("image" | "image_url") => {
                        out.special("<|begin_of_image|>")
                            .special("<|image|>")
                            .special("<|end_of_image|>");
                    }
                    Some("video" | "video_url") => {
                        out.special("<|begin_of_video|>")
                            .special("<|video|>")
                            .special("<|end_of_video|>");
                    }
                    Some("audio" | "audio_url" | "input_audio") => {
                        out.special("<|begin_of_audio|>")
                            .special("<|end_of_audio|>");
                    }
                    _ => return Err(Error::new("unsupported GLM content item")),
                }
            }
        }
        _ => return Err(Error::new("GLM content must be text or content blocks")),
    }
    Ok(out)
}

fn tool_json(tool: &Value) -> Value {
    let mut tool = function(tool).clone();
    if let Some(object) = tool.as_object_mut() {
        object.remove("strict");
        object.remove("defer_loading");
    }
    tool
}

fn render_call(prompt: &mut Prompt, call: &Value) -> Result<(), Error> {
    let call = function(call);
    let name = call["name"]
        .as_str()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| Error::new("GLM tool call requires a name"))?;
    prompt.special("<tool_call>").text(name);
    for (key, value) in object_arguments(&call["arguments"])?.iter() {
        prompt
            .special("<arg_key>")
            .text(key)
            .special("</arg_key>")
            .special("<arg_value>");
        if let Some(text) = value.as_str() {
            prompt.text(text);
        } else {
            prompt.text(crate::json::spaced(value));
        }
        prompt.special("</arg_value>");
    }
    prompt.special("</tool_call>");
    Ok(())
}

fn id_of(value: &Value) -> &str {
    value
        .get("tool_call_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .or_else(|| value.get("id").and_then(Value::as_str))
        .unwrap_or("")
}

fn tool_response(prompt: &mut Prompt, content: &Value, tools: &[Value]) -> Result<(), Error> {
    prompt.special("<tool_response>");
    if list(content)
        .first()
        .is_some_and(|item| item["type"] == "tool_reference")
    {
        prompt.text("<tools>\n");
        for reference in list(content) {
            for tool in tools {
                if function(tool)["name"] == reference["name"] {
                    prompt
                        .text(crate::json::spaced(&tool_json(tool)))
                        .text("\n");
                }
            }
        }
        prompt.text("</tools>");
    } else {
        prompt.append(visible_content(content)?);
    }
    prompt.special("</tool_response>");
    Ok(())
}

fn tool_block(
    prompt: &mut Prompt,
    block: &[Value],
    preceding: Option<&Value>,
    tools: &[Value],
) -> Result<(), Error> {
    let mut results = Vec::new();
    for message in block {
        let content = &message["content"];
        if list(content)
            .first()
            .is_some_and(|item| item.get("output").is_some())
        {
            results.extend(
                list(content)
                    .iter()
                    .map(|item| (id_of(item), &item["output"])),
            );
        } else {
            results.push((id_of(message), content));
        }
    }
    let calls = preceding
        .filter(|message| message["role"] == "assistant")
        .map(|message| list(&message["tool_calls"]))
        .unwrap_or(&[]);
    let call_ids: Vec<_> = calls.iter().map(id_of).collect();
    let sortable = !call_ids.is_empty()
        && call_ids
            .iter()
            .enumerate()
            .all(|(i, id)| !id.is_empty() && !call_ids[..i].contains(id))
        && results.iter().enumerate().all(|(i, (id, _))| {
            !id.is_empty()
                && call_ids.contains(id)
                && !results[..i].iter().any(|(previous, _)| previous == id)
        });
    prompt.special("<|observation|>");
    if sortable {
        for id in call_ids {
            if let Some((_, content)) = results.iter().find(|(result_id, _)| *result_id == id) {
                tool_response(prompt, content, tools)?;
            }
        }
    } else {
        for (_, content) in results {
            tool_response(prompt, content, tools)?;
        }
    }
    Ok(())
}

impl crate::Recipe for Glm53FlashRecipe {
    fn from_config(mut params: Value, defaults: &crate::RecipeDefaults<'_>) -> Result<Self, Error> {
        crate::recipe::default(&mut params, "media_config", || defaults("image_processor"))?;
        crate::recipe::params(params)
    }

    fn media_processor(&self) -> Option<Box<dyn crate::MediaProcessor>> {
        Some(Box::new(self.media_config.clone().unwrap_or_default()))
    }
    fn message_policy(&self) -> crate::MessagePolicy {
        crate::MessagePolicy {
            roles: &["system", "user", "assistant", "tool"],
            ..crate::MessagePolicy::STANDARD
        }
    }
    fn render(&self, request: &Value) -> Result<Prompt, Error> {
        let options = crate::recipe::options("GLM-5.3-Flash", self, request)?;
        let messages = messages(request, "GLM-5.3-Flash")?;
        let effort = match options.reasoning_effort.as_deref() {
            Some("low") => "Low",
            Some("high") => "High",
            _ => "Max",
        };
        let clear_thinking = options.clear_thinking.unwrap_or(false);
        let tools = list(&request["tools"]);
        let last_user = messages
            .iter()
            .rposition(|message| message["role"] == "user");
        let mut prompt = Prompt::new();
        prompt
            .special("[gMASK]")
            .special("<sop>")
            .special("<|system|>")
            .text("Reasoning Effort: ")
            .text(effort);
        if !tools.is_empty() {
            prompt.special("<|system|>").text(TOOL_PREFIX);
            for tool in tools {
                if function(tool)["defer_loading"] != true {
                    prompt
                        .text(crate::json::spaced(&tool_json(tool)))
                        .text("\n");
                }
            }
            prompt.template(
                TOOL_SUFFIX,
                &[
                    "<tool_call>",
                    "</tool_call>",
                    "<arg_key>",
                    "</arg_key>",
                    "<arg_value>",
                    "</arg_value>",
                ],
            );
        }
        let mut index = 0;
        while index < messages.len() {
            let message = &messages[index];
            match message["role"].as_str() {
                Some("system" | "user") => {
                    prompt
                        .special(if message["role"] == "system" {
                            "<|system|>"
                        } else {
                            "<|user|>"
                        })
                        .append(visible_content(&message["content"])?);
                }
                Some("assistant") => {
                    prompt.special("<|assistant|>");
                    let mut content = visible_content(&message["content"])?;
                    let rendered = content.rendered_text();
                    let reasoning = if let Some(reasoning) = message["reasoning_content"].as_str() {
                        Some(reasoning)
                    } else if let Some((first, _)) = rendered.split_once("</think>") {
                        let reasoning =
                            first.rsplit_once("<think>").map_or(first, |(_, tail)| tail);
                        let tail_start = rendered.rfind("</think>").unwrap() + "</think>".len();
                        content = content.slice(tail_start..rendered.len());
                        Some(reasoning)
                    } else {
                        None
                    };
                    prompt.special("<think>");
                    if (!clear_thinking || last_user.is_none_or(|last| index > last))
                        && let Some(reasoning) = reasoning
                    {
                        prompt.text(reasoning);
                    }
                    prompt.special("</think>").append(content.trim());
                    for call in list(&message["tool_calls"]) {
                        render_call(&mut prompt, call)?;
                    }
                }
                Some("tool") => {
                    let end = messages[index..]
                        .iter()
                        .position(|message| message["role"] != "tool")
                        .map_or(messages.len(), |length| index + length);
                    tool_block(
                        &mut prompt,
                        &messages[index..end],
                        index.checked_sub(1).map(|i| &messages[i]),
                        tools,
                    )?;
                    index = end;
                    continue;
                }
                _ => return Err(Error::new("unsupported GLM message role")),
            }
            index += 1;
        }
        if options.add_generation_prompt.unwrap_or(true) {
            prompt.special("<|assistant|>").special("<think>");
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
        Ok(Box::new(GlmParser::new(
            GlmTokens::resolve(encoder)?,
            decoder,
            prompt_ids,
            &request["tools"],
        )?))
    }
    fn stop_token_ids(&self, encoder: &dyn crate::Encoder) -> Result<Vec<u32>, Error> {
        ["<|endoftext|>", "<|user|>", "<|observation|>"]
            .iter()
            .map(|token| encoder.special_id(token))
            .collect()
    }
}
