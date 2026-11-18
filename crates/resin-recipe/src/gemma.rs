//! Gemma 4's turn, thought-channel, and structured tool-call format.

pub mod media;
mod values;

use serde_json::Value;

use crate::render::{function, list as array, messages, text as string, truthy};
use crate::{AttributedText, Decoder, Error, FinishReason, Prompt};

fn strip_thinking(text: &str) -> String {
    text.split("<channel|>")
        .map(|part| {
            part.split_once("<|channel>")
                .map_or(part, |(before, _)| before)
        })
        .collect::<String>()
        .trim()
        .to_owned()
}

fn content(prompt: &mut Prompt, value: Option<&Value>, assistant: bool) -> Option<&'static str> {
    let mut last_media = None;
    let text = |prompt: &mut Prompt, value: &str| {
        prompt.text(if assistant {
            strip_thinking(value)
        } else {
            value.trim().to_owned()
        });
    };
    match value {
        Some(Value::String(value)) => text(prompt, value),
        Some(Value::Array(parts)) => {
            for part in parts {
                match string(&part["type"]) {
                    "text" | "input_text" => text(prompt, string(&part["text"])),
                    "image" | "image_url" | "input_image" => {
                        prompt.special("<|image|>");
                        last_media = Some("image");
                    }
                    "video" => {
                        prompt.special("<|video|>");
                        last_media = Some("video");
                    }
                    "audio" => {
                        prompt.special("<|audio|>");
                        last_media = Some("audio");
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    last_media
}

fn tool_response(prompt: &mut Prompt, name: &str, response: &Value) {
    prompt
        .special("<|tool_response>")
        .text(format!("response:{name}{{"));
    if let Some(response) = response.as_object() {
        prompt.separated(values::sorted(response), ",", |prompt, (key, value)| {
            prompt.text(key).text(":");
            values::argument(prompt, value, false);
        });
    } else {
        prompt.text("value:");
        values::argument(prompt, response, false);
    }
    prompt.text("}").special("<tool_response|>");
}

use crate::parser::{EventSink, Grammar, Parser, emit_tool, tokens};

tokens! { Gemma4Tokens {
    turn_start => ["<|turn>"], turn_end => ["<turn|>"],
    channel_start => ["<|channel>"], channel_end => ["<channel|>"],
    tool_call_start => ["<|tool_call>"], tool_call_end => ["<tool_call|>"],
    tool_response_start => ["<|tool_response>"], tool_response_end => ["<tool_response|>"],
    escape => ["<|\"|>"], eos => ["<eos>"],
}}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Content,
    ChannelHeader,
    Reasoning,
    TurnHeader,
    Tool,
    ToolResponse,
    Ended,
}

/// Gemma's thought channels and structured tool-call arguments.
pub type Gemma4Parser = Parser<Gemma4Grammar>;
pub struct Gemma4Grammar {
    tokens: Gemma4Tokens,
    state: State,
    header: AttributedText,
    tool: Vec<values::Character>,
    tool_index: usize,
}

impl Gemma4Parser {
    pub fn new(
        tokens: Gemma4Tokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
    ) -> Result<Self, Error> {
        tokens.validate()?;
        let mut state = State::Content;
        for &id in prompt_ids {
            state = if id == tokens.channel_start {
                State::ChannelHeader
            } else if id == tokens.channel_end {
                State::Content
            } else if id == tokens.tool_call_start {
                State::Tool
            } else if id == tokens.tool_response_start {
                State::ToolResponse
            } else if [tokens.tool_response_end, tokens.turn_start, tokens.turn_end].contains(&id) {
                State::Content
            } else {
                state
            };
        }
        // The prompt has already supplied an unfinished thought header.
        if state == State::ChannelHeader {
            state = State::Reasoning;
        }
        Ok(Self::with_grammar(
            Gemma4Grammar {
                tokens,
                state,
                header: AttributedText::default(),
                tool: Vec::new(),
                tool_index: 0,
            },
            decoder,
        ))
    }
}

impl Grammar for Gemma4Grammar {
    fn channel(&self) -> Option<&'static str> {
        match self.state {
            State::Content => Some("content"),
            State::Reasoning => Some("reasoning"),
            State::Tool => Some("tool"),
            _ => None,
        }
    }

    fn is_control(&self, id: u32) -> bool {
        self.tokens.ids().contains(&id)
    }
    fn ended(&self) -> bool {
        self.state == State::Ended
    }
    fn text(&mut self, text: AttributedText, events: &mut EventSink) -> Result<(), Error> {
        if !text.is_empty() {
            match self.state {
                State::Content => events.text(text, false),
                State::Reasoning => events.text(text, true),
                State::Tool => self.tool.extend(text.chars().map(values::Character::Text)),
                State::ChannelHeader | State::TurnHeader => {
                    self.header.push(text);
                    if let Some(offset) = self.header.find('\n') {
                        let body = self.header.slice(offset + 1..self.header.len());
                        self.header.clear();
                        self.state = if self.state == State::ChannelHeader {
                            State::Reasoning
                        } else {
                            State::Content
                        };
                        self.text(body, events)?;
                    }
                }
                State::ToolResponse | State::Ended => {}
            }
        }
        Ok(())
    }
    fn control(&mut self, id: u32, events: &mut EventSink) -> Result<(), Error> {
        let t = self.tokens;
        if id == t.escape {
            if self.state == State::Tool {
                self.tool.push(values::Character::Quote);
            }
        } else if id == t.tool_call_end {
            if self.state == State::Tool {
                let (name, arguments) = values::parse_call(&self.tool)?;
                emit_tool(&mut self.tool_index, name, arguments.to_string(), events);
            }
            self.tool.clear();
            self.state = State::Content;
        } else if id == t.tool_call_start {
            self.tool.clear();
            self.state = State::Tool;
        } else if id == t.channel_start {
            self.header.clear();
            self.state = State::ChannelHeader;
        } else if id == t.channel_end {
            self.header.clear();
            self.state = State::Content;
        } else if id == t.turn_start {
            self.header.clear();
            self.state = State::TurnHeader;
        } else if id == t.tool_response_start {
            self.state = State::ToolResponse;
        } else if id == t.tool_response_end {
            self.state = State::Content;
        } else {
            self.state = State::Ended;
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
pub struct Gemma4Recipe {
    pub add_generation_prompt: Option<bool>,
    pub enable_thinking: Option<bool>,
    /// Older 31B templates close an empty thought channel in the generation
    /// prefix. The original Edge checkpoints instead end at the model header.
    pub empty_thought_prefix: Option<bool>,
    #[serde(skip_serializing)]
    pub media_config: Option<media::Processor>,
}

impl crate::Recipe for Gemma4Recipe {
    fn from_config(mut params: Value, defaults: &crate::RecipeDefaults<'_>) -> Result<Self, Error> {
        crate::recipe::default(&mut params, "media_config", || {
            Ok(serde_json::json!({
                "image_processor": defaults("image_processor")?,
                "video_processor": defaults("video_processor")?,
            }))
        })?;
        crate::recipe::params(params)
    }

    fn message_policy(&self) -> crate::MessagePolicy {
        crate::MessagePolicy {
            assistant_order: &[
                crate::BlockKind::Thinking,
                crate::BlockKind::Call,
                crate::BlockKind::Result,
                crate::BlockKind::Text,
            ],
            reorder_wire_fields: true,
            ..crate::MessagePolicy::STANDARD
        }
    }
    fn render(&self, request: &Value) -> Result<crate::Prompt, crate::Error> {
        let options = crate::recipe::options("gemma4", self, request)?;
        let messages = messages(request, "Gemma 4")?;
        let tools = array(&request["tools"]);
        let enable_thinking = options.enable_thinking.unwrap_or(false);
        let mut prompt = Prompt::new();
        prompt.special("<bos>");
        let system = matches!(string(&messages[0]["role"]), "system" | "developer");
        let mut previous_kind = None;
        if enable_thinking || !tools.is_empty() || system {
            prompt.special("<|turn>").text("system\n");
            if enable_thinking {
                prompt.special("<|think|>").text("\n");
                previous_kind = Some("think");
            }
            if system {
                match messages[0].get("content") {
                    Some(Value::String(text)) => {
                        prompt.text(text.trim());
                    }
                    Some(Value::Array(parts)) => {
                        for part in parts {
                            prompt.text(string(&part["text"]).trim()).text(" ");
                        }
                    }
                    _ => {}
                }
            }
            for tool in tools {
                prompt.special("<|tool>");
                values::declaration(&mut prompt, tool.get("function").unwrap_or(tool));
                prompt.special("<tool|>");
                previous_kind = Some("tool");
            }
            prompt.special("<turn|>").text("\n");
        }
        let messages = if system { &messages[1..] } else { messages };
        let last_user = messages
            .iter()
            .rposition(|message| string(&message["role"]) == "user");
        let mut previous_role = "";
        for (index, message) in messages.iter().enumerate() {
            let role = string(&message["role"]);
            if role == "tool" {
                continue;
            }
            if !matches!(
                role,
                "user" | "assistant" | "system" | "developer" | "model"
            ) {
                return Err(Error::new(format!("unsupported Gemma 4 role {role:?}")));
            }
            previous_kind = None;
            let assistant = role == "assistant" || role == "model";
            if !(assistant && previous_role == "assistant") {
                prompt
                    .special("<|turn>")
                    .text(if assistant { "model" } else { role })
                    .text("\n");
            }
            previous_role = role;
            let calls = array(&message["tool_calls"]);
            let reasoning = message
                .get("reasoning")
                .filter(|value| truthy(value))
                .or_else(|| message.get("reasoning_content"));
            if !calls.is_empty()
                && last_user.is_none_or(|last| index > last)
                && let Some(reasoning) = reasoning
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
            {
                prompt
                    .special("<|channel>")
                    .text("thought\n")
                    .text(reasoning)
                    .text("\n")
                    .special("<channel|>");
            }
            for call in calls {
                let function = function(call);
                prompt
                    .special("<|tool_call>")
                    .text(format!("call:{}{{", string(&function["name"])));
                match function.get("arguments") {
                    Some(Value::Object(arguments)) => {
                        prompt.separated(
                            values::sorted(arguments),
                            ",",
                            |prompt, (name, value)| {
                                prompt.text(name).text(":");
                                values::argument(prompt, value, false);
                            },
                        );
                    }
                    Some(Value::String(arguments)) => {
                        prompt.text(arguments);
                    }
                    _ => {}
                }
                prompt.text("}").special("<tool_call|>");
                previous_kind = Some("tool_call");
            }
            let mut has_response = false;
            let responses = array(&message["tool_responses"]);
            if !responses.is_empty() {
                for response in responses {
                    tool_response(
                        &mut prompt,
                        response
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown"),
                        response.get("response").unwrap_or(&Value::Null),
                    );
                    has_response = true;
                    previous_kind = Some("tool_response");
                }
            } else if !calls.is_empty() {
                for follow in messages[index + 1..]
                    .iter()
                    .take_while(|follow| string(&follow["role"]) == "tool")
                {
                    let name = calls
                        .iter()
                        .rfind(|call| call.get("id") == follow.get("tool_call_id"))
                        .map(|call| string(&function(call)["name"]))
                        .unwrap_or_else(|| {
                            follow
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown")
                        });
                    let body = follow.get("content").unwrap_or(&Value::Null);
                    if let Some(parts) = body.as_array() {
                        let text = parts
                            .iter()
                            .filter(|part| string(&part["type"]) == "text")
                            .map(|part| string(&part["text"]))
                            .collect::<String>();
                        tool_response(&mut prompt, name, &Value::String(text));
                    } else {
                        tool_response(&mut prompt, name, body);
                    }
                    has_response = true;
                    previous_kind = Some("tool_response");
                }
            }
            let mut body = Prompt::new();
            if let Some(kind) = content(&mut body, message.get("content"), assistant) {
                previous_kind = Some(kind);
            }
            let has_content = !body.rendered_text().trim().is_empty();
            prompt.append(body);
            if previous_kind == Some("tool_call") && !has_response {
                prompt.special("<|tool_response>");
            } else if !(has_response && !has_content) {
                prompt.special("<turn|>").text("\n");
            }
        }
        if options.add_generation_prompt.unwrap_or(true)
            && !matches!(previous_kind, Some("tool_response" | "tool_call"))
        {
            prompt.special("<|turn>").text("model\n");
            if !enable_thinking && options.empty_thought_prefix.unwrap_or(true) {
                prompt
                    .special("<|channel>")
                    .text("thought\n")
                    .special("<channel|>");
            }
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
        Ok(Box::new(Gemma4Parser::new(
            Gemma4Tokens::resolve(encoder)?,
            decoder,
            prompt_ids,
        )?))
    }
    fn stop_token_ids(&self, encoder: &dyn crate::Encoder) -> Result<Vec<u32>, crate::Error> {
        Ok(vec![
            encoder.special_id("<turn|>")?,
            encoder.special_id("<eos>")?,
        ])
    }
    fn media_processor(&self) -> Option<Box<dyn crate::MediaProcessor>> {
        Some(Box::new(self.media_config.clone().unwrap_or_default()))
    }
}
