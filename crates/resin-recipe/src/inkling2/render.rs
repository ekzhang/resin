//! Inkling2's pinned template; user text never becomes a control token.
use super::{Inkling2Parser, Inkling2Recipe, Inkling2Tokens};
use crate::render::{function, object_arguments, value_or};
use crate::{Error, Prompt};
use serde_json::{Value, json};

fn compact(value: &Value) -> String {
    let mut value = value.clone();
    value.sort_all_objects();
    crate::json::compact(&value)
}
fn block(prompt: &mut Prompt, role: &str, name: &str, kind: &str, text: &str) {
    prompt
        .special(role)
        .text(name)
        .special(kind)
        .text(text)
        .special("<|end_message|>");
}
fn effort(value: Option<&crate::ReasoningEffort>) -> Result<String, Error> {
    let num = match value {
        None => 0.9,
        Some(crate::ReasoningEffort::Name(value)) => match value.trim() {
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "max" | "xhigh" => 0.99,
            _ => return Err(Error::new("unknown Inkling2 reasoning_effort")),
        },
        Some(crate::ReasoningEffort::Number(value)) => value
            .as_f64()
            .ok_or_else(|| Error::new("reasoning_effort must be a number or effort name"))?,
    };
    if !(0.0..=0.99).contains(&num) {
        return Err(Error::new("reasoning_effort must be in [0.0, 0.99]"));
    }
    Ok(if num == 0.0 {
        "0".to_owned()
    } else {
        crate::json::compact(&json!(num))
    })
}
fn emit_effort(prompt: &mut Prompt, effort: &str) {
    block(
        prompt,
        "<|message_system|>",
        "",
        "<|content_text|>",
        &format!("Thinking effort level: {effort}"),
    );
}

impl crate::Recipe for Inkling2Recipe {
    fn from_config(mut params: Value, defaults: &crate::RecipeDefaults<'_>) -> Result<Self, Error> {
        crate::recipe::default(&mut params, "media_config", || defaults("image_processor"))?;
        crate::recipe::params(params)
    }

    fn render(&self, request: &Value) -> Result<crate::Prompt, crate::Error> {
        let options = crate::recipe::options("inkling2", self, request)?;
        let messages = request["messages"]
            .as_array()
            .ok_or_else(|| Error::new("messages must be an array"))?;
        let mut prompt = Prompt::new();
        if let Some(tools) = request["tools"]
            .as_array()
            .filter(|tools| !tools.is_empty())
        {
            let specs: Vec<Value> = tools
                .iter()
                .map(|tool| {
                    let function = function(tool);
                    json!({
                        "description": value_or(&function["description"], json!("")),
                        "name": function["name"],
                        "parameters": value_or(&function["parameters"], json!({})),
                        "type": value_or(&tool["type"], json!("function")),
                    })
                })
                .collect();
            prompt
                .special("<|message_system|>")
                .text("tool_declare")
                .special("<|content_xml|>")
                .text(compact(&Value::Array(specs)))
                .special("<|end_message|>");
        }
        let effort = effort(options.reasoning_effort.as_ref())?;
        let mut effort_emitted = false;
        for message in messages {
            let role = message["role"].as_str().unwrap_or("");
            let token = match role {
                "system" => "<|message_system|>",
                "user" => "<|message_user|>",
                "assistant" => "<|message_model|>",
                "tool" => "<|message_tool|>",
                _ => return Err(Error::new(format!("unknown Inkling2 message role: {role}"))),
            };
            if !effort_emitted && role != "system" {
                emit_effort(&mut prompt, &effort);
                effort_emitted = true;
            }
            let header_name = if role == "tool" {
                message["name"]
                    .as_str()
                    .filter(|v| !v.is_empty())
                    .or_else(|| {
                        let id = message["tool_call_id"].as_str()?;
                        messages
                            .iter()
                            .flat_map(|m| m["tool_calls"].as_array().into_iter().flatten())
                            .filter(|call| call["id"].as_str() == Some(id))
                            .filter_map(|call| call["function"]["name"].as_str())
                            .next_back()
                    })
                    .unwrap_or("")
            } else if role == "assistant" {
                message["name"].as_str().unwrap_or("")
            } else {
                ""
            };
            if role == "assistant"
                && let Some(reasoning) = message["reasoning_content"]
                    .as_str()
                    .filter(|v| !v.is_empty())
            {
                block(
                    &mut prompt,
                    token,
                    header_name,
                    "<|content_thinking|>",
                    reasoning,
                );
            }
            match &message["content"] {
                Value::String(text)
                    if !text.is_empty()
                        || role != "assistant"
                        || (message["tool_calls"].as_array().is_none_or(Vec::is_empty)
                            && message["reasoning_content"]
                                .as_str()
                                .is_none_or(str::is_empty)) =>
                {
                    block(&mut prompt, token, header_name, "<|content_text|>", text)
                }
                Value::Array(parts) => {
                    for part in parts {
                        if let Some(text) = part.as_str() {
                            block(&mut prompt, token, header_name, "<|content_text|>", text);
                            continue;
                        }
                        match part["type"].as_str().unwrap_or("text") {
                            "text" | "input_text" => block(
                                &mut prompt,
                                token,
                                header_name,
                                "<|content_text|>",
                                part["text"].as_str().unwrap_or(""),
                            ),
                            "image" | "input_image" | "image_url" => {
                                if role != "user" && role != "tool" {
                                    return Err(Error::new(
                                        "Inkling2 images require user or tool messages",
                                    ));
                                }
                                prompt
                                    .special(token)
                                    .text(header_name)
                                    .special("<|content_image|>")
                                    .special("<|unused_200054|>")
                                    .special("<|end_message|>");
                            }
                            "audio" | "input_audio" | "audio_url" => {
                                return Err(Error::new("Inkling2 does not accept audio input"));
                            }
                            other => {
                                return Err(Error::new(format!(
                                    "unsupported Inkling2 content part: {other}"
                                )));
                            }
                        }
                    }
                }
                _ => (),
            }
            if role == "assistant" {
                for call in message["tool_calls"].as_array().into_iter().flatten() {
                    let function = &call["function"];
                    let name = function["name"]
                        .as_str()
                        .ok_or_else(|| Error::new("tool call function name must be a string"))?;
                    if !header_name.is_empty() && header_name != name {
                        return Err(Error::new(
                            "Inkling2 assistant name does not match its tool call",
                        ));
                    }
                    let args = value_or(&function["arguments"], json!({}));
                    let mut args = object_arguments(&args)?.into_owned();
                    args.sort_keys();
                    prompt
                        .special(token)
                        .text(name)
                        .special("<|content_invoke_tool_json|>")
                        .text(format!(
                            "{{\"name\":{},\"args\":{}}}",
                            compact(&Value::String(name.into())),
                            crate::json::compact(&Value::Object(args))
                        ))
                        .special("<|end_message|>");
                }
                prompt.special("<|content_model_end_sampling|>");
            }
        }
        if !effort_emitted {
            emit_effort(&mut prompt, &effort);
        }
        Ok(prompt)
    }
    fn create_parser(
        &self,
        encoder: &dyn crate::Encoder,
        decoder: Box<dyn crate::Decoder>,
        _request: &Value,
        prompt_ids: &[u32],
    ) -> Result<Box<dyn crate::ResponseParser>, crate::Error> {
        Ok(Box::new(Inkling2Parser::new(
            Inkling2Tokens::resolve(encoder)?,
            decoder,
            prompt_ids,
        )?))
    }
    fn stop_token_ids(&self, encoder: &dyn crate::Encoder) -> Result<Vec<u32>, crate::Error> {
        Ok(vec![encoder.special_id("<|content_model_end_sampling|>")?])
    }
    fn input_modalities(&self) -> &'static [&'static str] {
        if self.text_only {
            &["text"]
        } else {
            &["text", "image"]
        }
    }
    fn prepare_media(
        &self,
        sources: Vec<crate::media::Source>,
        encoder: &dyn crate::Encoder,
    ) -> Result<crate::media::Prepared, Error> {
        if self.text_only {
            return Err(Error::new("this recipe does not support media"));
        }
        self.media_config.prepare(sources, encoder)
    }
}
