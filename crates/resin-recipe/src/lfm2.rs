//! The text, reasoning, and Python-style tool-call format of LFM2.5.

mod literals;

use crate::render::{function, list, messages, object_arguments, text_content};
use serde_json::Value;

use crate::{AttributedText, Decoder, Error, FinishReason, Prompt};

const BOS: &str = "<|startoftext|>";
const START: &str = "<|im_start|>";
const END: &str = "<|im_end|>";
const THINK: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL: &str = "<|tool_call_start|>";
const TOOL_END: &str = "<|tool_call_end|>";

fn content(value: &Value) -> Result<String, Error> {
    text_content(value, |block| match block["type"].as_str() {
        Some("image") => Err(Error::new(
            "the stock LFM2 runner supports text content only",
        )),
        _ => Ok(crate::json::spaced(block)),
    })
}

fn render_calls(prompt: &mut Prompt, calls: &[Value]) -> Result<(), Error> {
    prompt.special(TOOL);
    prompt.text("[");
    for (index, call) in calls.iter().enumerate() {
        if index > 0 {
            prompt.text(", ");
        }
        let function = function(call);
        let name = function["name"]
            .as_str()
            .ok_or_else(|| Error::new("tool call name must be a string"))?;
        prompt.text(name);
        prompt.text("(");
        let arguments = object_arguments(&function["arguments"])?;
        prompt.separated(arguments.iter(), ", ", |prompt, (name, value)| {
            prompt.text(name).text("=");
            prompt.text(match value {
                Value::String(text) => format!("'{text}'"),
                Value::Object(_) => crate::json::spaced(value),
                _ => literals::repr(value),
            });
        });
        prompt.text(")");
    }
    prompt.text("]");
    prompt.special(TOOL_END);
    Ok(())
}

use crate::parser::{EventSink, Grammar, Parser, emit_tool, tokens};

// Think/tool delimiters are added tokens marked `special=false` upstream.
// Their actual IDs still delimit LFM2's grammar; literal spellings do not.
tokens! { Lfm2Tokens {
    message_start => ["<|im_start|>"],
    message_end => ["<|im_end|>"],
    thinking_start => ["<think>"],
    thinking_end => ["</think>"],
    tool_call_start => ["<|tool_call_start|>"],
    tool_call_end => ["<|tool_call_end|>"],
}}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Content,
    Reasoning,
    Header,
    Tool,
    Done,
}

/// LFM2.5 reasoning and literal tool calls, buffering one unfinished tool block.
pub type Lfm2Parser = Parser<Lfm2Grammar>;
pub struct Lfm2Grammar {
    tokens: Lfm2Tokens,
    stage: Stage,
    tool: String,
    tool_index: usize,
}

impl Lfm2Parser {
    pub fn new(tokens: Lfm2Tokens, decoder: Box<dyn Decoder>) -> Result<Self, Error> {
        tokens.validate()?;
        Ok(Self::with_grammar(
            Lfm2Grammar {
                tokens,
                stage: Stage::Content,
                tool: String::new(),
                tool_index: 0,
            },
            decoder,
        ))
    }
}

impl Grammar for Lfm2Grammar {
    fn channel(&self) -> Option<&'static str> {
        match self.stage {
            Stage::Content => Some("content"),
            Stage::Reasoning => Some("reasoning"),
            Stage::Tool => Some("tool"),
            _ => None,
        }
    }

    fn is_control(&self, id: u32) -> bool {
        self.tokens.ids().contains(&id)
    }
    fn ended(&self) -> bool {
        self.stage == Stage::Done
    }
    fn text(&mut self, text: AttributedText, events: &mut EventSink) -> Result<(), Error> {
        if !text.is_empty() {
            match self.stage {
                Stage::Content => events.text(text, false),
                Stage::Reasoning => events.text(text, true),
                Stage::Tool => self.tool.push_str(&text),
                Stage::Header => {
                    if let Some(offset) = text.find('\n') {
                        self.stage = Stage::Content;
                        self.text(text.slice(offset + 1..text.len()), events)?;
                    }
                }
                Stage::Done => {}
            }
        }
        Ok(())
    }
    fn control(&mut self, id: u32, events: &mut EventSink) -> Result<(), Error> {
        if id == self.tokens.tool_call_end {
            if self.stage != Stage::Tool {
                return Err(Error::new(
                    "LFM2 emitted an unexpected tool-call end marker",
                ));
            }
            for (name, arguments) in literals::parse_calls(&self.tool)? {
                emit_tool(&mut self.tool_index, name, arguments.to_string(), events);
            }
            self.tool.clear();
            self.stage = Stage::Content;
        } else if id == self.tokens.tool_call_start {
            self.tool.clear();
            self.stage = Stage::Tool;
        } else if id == self.tokens.thinking_start {
            self.stage = Stage::Reasoning;
        } else if id == self.tokens.thinking_end {
            self.stage = Stage::Content;
        } else if id == self.tokens.message_start {
            self.stage = Stage::Header;
        } else {
            self.tool.clear();
            self.stage = Stage::Done;
        }
        Ok(())
    }
    fn finish(&mut self, _reason: FinishReason, _events: &mut EventSink) -> Result<(), Error> {
        self.tool.clear();
        Ok(())
    }
}

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Lfm2RecipeParams {
    pub add_generation_prompt: Option<bool>,
    pub preserve_thinking: Option<bool>,
}

#[derive(Clone, Debug, Default)]
pub struct Lfm2Recipe {
    params: Lfm2RecipeParams,
}

impl Lfm2Recipe {
    pub fn new(params: Lfm2RecipeParams) -> Self {
        Self { params }
    }
}
impl crate::Recipe for Lfm2Recipe {
    fn render(&self, request: &Value) -> Result<crate::Prompt, crate::Error> {
        let request = crate::recipe::request("lfm2", &self.params, request)?;
        let request = &request;
        let messages = messages(request, "LFM2")?;
        let mut prompt = Prompt::new();
        prompt.special(BOS);
        let (system, messages) = if messages[0]["role"] == "system" {
            (content(&messages[0]["content"])?, &messages[1..])
        } else {
            (String::new(), messages)
        };
        let tools = list(&request["tools"]);
        if !system.is_empty() || !tools.is_empty() {
            prompt.special(START);
            prompt.text("system\n");
            prompt.text(&system);
            if !tools.is_empty() {
                if !system.is_empty() {
                    prompt.text("\n");
                }
                prompt.text("List of tools: [");
                for (index, tool) in tools.iter().enumerate() {
                    if index > 0 {
                        prompt.text(", ");
                    }
                    prompt.text(
                        tool.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| crate::json::spaced(tool)),
                    );
                }
                prompt.text("]");
            }
            prompt.special(END);
            prompt.text("\n");
        }
        let last_user = messages
            .iter()
            .rposition(|message| message["role"] == "user");
        let preserve =
            request["extra"]["preserve_thinking"] == true || request["preserve_thinking"] == true;
        for (index, message) in messages.iter().enumerate() {
            let role = message["role"]
                .as_str()
                .ok_or_else(|| Error::new("message role must be a string"))?;
            if !["system", "developer", "user", "assistant", "tool"].contains(&role) {
                return Err(Error::new(format!("unsupported LFM2 message role: {role}")));
            }
            prompt.special(START);
            prompt.text(role);
            prompt.text("\n");
            if role == "assistant" {
                let keeps_thinking = preserve || last_user.is_none_or(|last| index > last);
                let thinking = ["thinking", "reasoning", "reasoning_content"]
                    .into_iter()
                    .filter_map(|key| message[key].as_str())
                    .find(|text| !text.is_empty());
                if let Some(thinking) = thinking.filter(|_| keeps_thinking) {
                    prompt.special(THINK);
                    prompt.text(thinking);
                    prompt.special(THINK_END);
                }
                let mut text = content(&message["content"])?;
                if !keeps_thinking && let Some((_, answer)) = text.rsplit_once(THINK_END) {
                    text = answer.trim().to_owned();
                }
                let calls = message.get("tool_calls").filter(|value| !value.is_null());
                const CONTINUE: &str = "CONTINUE_FINAL_MESSAGE_TAG ";
                let continued = calls.is_some() && text.ends_with(CONTINUE);
                prompt.text(if continued {
                    &text[..text.len() - CONTINUE.len()]
                } else {
                    &text
                });
                if let Some(calls) = calls {
                    render_calls(
                        &mut prompt,
                        calls
                            .as_array()
                            .ok_or_else(|| Error::new("tool_calls must be an array"))?,
                    )?;
                }
                if continued {
                    prompt.text(CONTINUE);
                }
            } else {
                prompt.text(content(&message["content"])?);
            }
            prompt.special(END);
            prompt.text("\n");
        }
        if crate::render::option(request, "add_generation_prompt") != Some(&Value::Bool(false)) {
            prompt.special(START);
            prompt.text("assistant\n");
        }
        Ok(prompt)
    }
    fn create_parser(
        &self,
        encoder: &dyn crate::Encoder,
        decoder: Box<dyn crate::Decoder>,
        _request: &Value,
        _prompt_ids: &[u32],
    ) -> Result<Box<dyn crate::ResponseParser>, crate::Error> {
        Ok(Box::new(Lfm2Parser::new(
            Lfm2Tokens::resolve(encoder)?,
            decoder,
        )?))
    }
    fn stop_token_ids(&self, encoder: &dyn crate::Encoder) -> Result<Vec<u32>, crate::Error> {
        Ok(vec![encoder.special_id("<|im_end|>")?])
    }
}
