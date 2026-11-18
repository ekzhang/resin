//! GPT-OSS's Harmony conversation format.

mod schema;

use serde_json::Value;

use crate::render::{function, list, messages};
use crate::{AttributedText, Decoder, Error, FinishReason, Prompt, ResponseParser};

fn content(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                Some("text") => part.get("text").and_then(Value::as_str),
                Some("thinking") => part.get("thinking").and_then(Value::as_str),
                _ => None,
            })
            .collect(),
        _ => String::new(),
    }
}

fn message(
    prompt: &mut Prompt,
    author: &str,
    channel: Option<&str>,
    recipient: Option<&str>,
    body: &str,
) {
    prompt.special("<|start|>");
    prompt.text(author);
    if let Some(recipient) = recipient {
        prompt.text(format!(" to={recipient}"));
    }
    if let Some(channel) = channel {
        prompt.special("<|channel|>");
        prompt.text(channel);
    }
    prompt.special("<|message|>");
    prompt.text(body);
    prompt.special(if author == "assistant" && recipient.is_some() {
        "<|call|>"
    } else {
        "<|end|>"
    });
}

fn tools_section(tools: &[Value]) -> String {
    let mut lines = vec!["# Tools\n\n## functions\n\nnamespace functions {\n".to_owned()];
    for tool in tools {
        let tool = function(tool);
        for line in tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .lines()
        {
            lines.push(format!("// {line}"));
        }
        let empty = serde_json::json!({});
        let parameters = tool
            .get("parameters")
            .filter(|value| !value.is_null())
            .unwrap_or(&empty);
        lines.push(format!(
            "type {} = (_: {}) => any;\n",
            tool.get("name").and_then(Value::as_str).unwrap_or(""),
            schema::schema_type(parameters, "")
        ));
    }
    lines.push("} // namespace functions".into());
    lines.join("\n")
}

use crate::parser::{EventSink, Grammar, Parser, emit_tool, tokens};

tokens! { GptOssTokens {
    start => ["<|im_start|>", "<|start|>"],
    end => ["<|im_end|>", "<|end|>"],
    turn_end => ["<|fim_suffix|>", "<|return|>"],
    tool_end => ["<|ghissue|>", "<|call|>"],
    channel => ["<|meta_sep|>", "<|channel|>"],
    message => ["<|im_sep|>", "<|message|>"],
    constrain => ["<|meta_start|>", "<|constrain|>"],
}}

enum State {
    Header,
    Text,
    Reasoning,
    Tool(String),
    Ignore,
}

/// Harmony headers and channel bodies. A tool call commits at its terminator.
pub type GptOssParser = Parser<GptOssGrammar>;
pub struct GptOssGrammar {
    tokens: GptOssTokens,
    state: State,
    header: String,
    channel_offset: Option<usize>,
    tool_text: String,
    tool_index: usize,
    ended: bool,
}

impl GptOssParser {
    pub fn new(
        tokens: GptOssTokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
    ) -> Result<Self, Error> {
        tokens.validate()?;
        let mut parser = Self::with_grammar(
            GptOssGrammar {
                tokens,
                state: State::Text,
                header: String::new(),
                channel_offset: None,
                tool_text: String::new(),
                tool_index: 0,
                ended: false,
            },
            decoder,
        );
        // Seed only the final message's header, never earlier conversation text.
        if let Some(start) = prompt_ids.iter().rposition(|&id| id == tokens.start) {
            parser.grammar.state = State::Header;
            let tail = &prompt_ids[start + 1..];
            let end = tail.iter().position(|&id| id == tokens.message);
            parser.feed(&tail[..end.map_or(tail.len(), |end| end + 1)])?;
        }
        parser.next_token_index = 0;
        Ok(parser)
    }
}

impl GptOssGrammar {
    fn open_body(&mut self) {
        let role = self.header.split_whitespace().next().unwrap_or("");
        let channel = self
            .channel_offset
            .and_then(|offset| self.header[offset..].split_whitespace().next());
        let recipient = self
            .header
            .split_whitespace()
            .find_map(|word| word.strip_prefix("to="));
        self.state = if role != "assistant" {
            State::Ignore
        } else if let Some(name) = recipient.filter(|name| *name != "all") {
            State::Tool(name.strip_prefix("functions.").unwrap_or(name).to_owned())
        } else if channel == Some("analysis") {
            State::Reasoning
        } else {
            State::Text
        };
    }
    fn close_body(&mut self, events: &mut EventSink) {
        if let State::Tool(name) = &self.state
            && !name.is_empty()
            && serde_json::from_str::<Value>(&self.tool_text).is_ok()
        {
            // Preserve JSON lexemes/whitespace; incomplete calls never commit.
            emit_tool(&mut self.tool_index, name, self.tool_text.clone(), events);
        }
        self.tool_text.clear();
        self.state = State::Ignore;
    }
}

impl Grammar for GptOssGrammar {
    fn channel(&self) -> Option<&'static str> {
        match self.state {
            State::Text => Some("content"),
            State::Reasoning => Some("reasoning"),
            State::Tool(_) => Some("tool"),
            _ => None,
        }
    }

    fn is_control(&self, id: u32) -> bool {
        self.tokens.ids().contains(&id)
    }
    fn ended(&self) -> bool {
        self.ended
    }
    fn text(&mut self, text: AttributedText, events: &mut EventSink) -> Result<(), Error> {
        if !text.is_empty() {
            match self.state {
                State::Header => self.header.push_str(&text),
                State::Text => events.text(text, false),
                State::Reasoning => events.text(text, true),
                State::Tool(_) => self.tool_text.push_str(&text),
                State::Ignore => {}
            }
        }
        Ok(())
    }
    fn control(&mut self, id: u32, events: &mut EventSink) -> Result<(), Error> {
        if id == self.tokens.start {
            self.tool_text.clear();
            self.header.clear();
            self.channel_offset = None;
            self.state = State::Header;
        } else if id == self.tokens.channel && matches!(self.state, State::Header) {
            self.header.push(' ');
            self.channel_offset = Some(self.header.len());
        } else if id == self.tokens.constrain && matches!(self.state, State::Header) {
            self.header.push(' ');
        } else if id == self.tokens.message && matches!(self.state, State::Header) {
            self.open_body();
        } else if [self.tokens.end, self.tokens.turn_end, self.tokens.tool_end].contains(&id) {
            self.close_body(events);
            self.ended = id == self.tokens.turn_end;
        }
        Ok(())
    }
    fn finish(&mut self, _reason: FinishReason, _events: &mut EventSink) -> Result<(), Error> {
        self.tool_text.clear();
        Ok(())
    }
}

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct GptOssRecipe {
    pub add_generation_prompt: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub use_system_prompt: Option<bool>,
    pub model_identity: Option<String>,
    pub knowledge_cutoff: Option<String>,
    pub conversation_start_date: Option<String>,
    pub auto_drop_analysis: Option<bool>,
}

impl crate::Recipe for GptOssRecipe {
    fn from_config(mut params: Value, defaults: &crate::RecipeDefaults<'_>) -> Result<Self, Error> {
        crate::recipe::default(&mut params, "conversation_start_date", || {
            defaults("conversation_start_date")
        })?;
        crate::recipe::params(params)
    }

    fn render(&self, request: &Value) -> Result<crate::Prompt, crate::Error> {
        let options = crate::recipe::options("gpt_oss", self, request)?;
        let messages = messages(request, "GPT-OSS")?;
        let tools = list(&request["tools"]);
        let effort = options
            .reasoning_effort
            .as_deref()
            .unwrap_or("medium")
            .to_ascii_lowercase();
        if !["low", "medium", "high"].contains(&effort.as_str()) {
            return Err(Error::new(
                "GPT-OSS reasoning_effort must be low, medium, or high",
            ));
        }
        let mut prompt = Prompt::new();
        if options.use_system_prompt.unwrap_or(true) {
            let identity = options
                .model_identity
                .as_deref()
                .unwrap_or("You are ChatGPT, a large language model trained by OpenAI.");
            let cutoff = options.knowledge_cutoff.as_deref().unwrap_or("2024-06");
            let mut system = format!("{identity}\nKnowledge cutoff: {cutoff}");
            if let Some(date) = options.conversation_start_date.as_deref() {
                system.push_str(&format!("\nCurrent date: {date}"));
            }
            system.push_str(&format!("\n\nReasoning: {effort}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message."));
            if !tools.is_empty() {
                system.push_str(
                    "\nCalls to these tools must go to the commentary channel: 'functions'.",
                );
            }
            message(&mut prompt, "system", None, None, &system);
        }
        let first_system = messages
            .iter()
            .position(|message| message.get("role").and_then(Value::as_str) == Some("system"));
        if first_system.is_some() || !tools.is_empty() {
            let mut sections = Vec::new();
            if let Some(index) = first_system {
                let instructions = content(messages[index].get("content"));
                if !instructions.is_empty() {
                    sections.push(format!("# Instructions\n\n{instructions}"));
                }
            }
            if !tools.is_empty() {
                sections.push(tools_section(tools));
            }
            message(&mut prompt, "developer", None, None, &sections.join("\n\n"));
        }
        let auto_drop_analysis = options.auto_drop_analysis.unwrap_or(false);
        let add_generation_prompt = options.add_generation_prompt.unwrap_or(true);
        for (index, item) in messages.iter().enumerate() {
            if first_system == Some(index) {
                continue;
            }
            let body = content(item.get("content"));
            match item.get("role").and_then(Value::as_str).unwrap_or("") {
                "user" => message(&mut prompt, "user", None, None, &body),
                "tool" => {
                    let name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    let author = if name.starts_with("functions.") {
                        name.to_owned()
                    } else {
                        format!("functions.{name}")
                    };
                    message(
                        &mut prompt,
                        &author,
                        Some("commentary"),
                        Some("assistant"),
                        &body,
                    );
                }
                "assistant" => {
                    let calls = list(&item["tool_calls"]);
                    let keep_analysis = !auto_drop_analysis
                        || (!calls.is_empty()
                            && !messages[index + 1..].iter().any(|later| {
                                later.get("role").and_then(Value::as_str) == Some("assistant")
                                    && later
                                        .get("tool_calls")
                                        .and_then(Value::as_array)
                                        .is_none_or(Vec::is_empty)
                                    && !content(later.get("content")).is_empty()
                            }));
                    let reasoning = item
                        .get("reasoning_content")
                        .and_then(Value::as_str)
                        .filter(|text| !text.is_empty())
                        .filter(|_| keep_analysis);
                    if let Some(reasoning) = reasoning {
                        message(&mut prompt, "assistant", Some("analysis"), None, reasoning);
                    }
                    // Assistant thinking parts live in reasoning_content, not
                    // its public text list, matching Harmony history rendering.
                    let text = match item.get("content") {
                        Some(Value::Array(parts)) => parts
                            .iter()
                            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                            .filter_map(|part| part.get("text").and_then(Value::as_str))
                            .collect(),
                        _ => body,
                    };
                    if !text.is_empty() || (reasoning.is_none() && calls.is_empty()) {
                        message(&mut prompt, "assistant", Some("final"), None, &text);
                    }
                    for call in calls {
                        let function = function(call);
                        let name = function.get("name").and_then(Value::as_str).unwrap_or("");
                        let recipient = if name.starts_with("functions.") {
                            name.to_owned()
                        } else {
                            format!("functions.{name}")
                        };
                        let arguments = function
                            .get("arguments")
                            .map(|arguments| {
                                arguments
                                    .as_str()
                                    .map(str::to_owned)
                                    .unwrap_or_else(|| crate::json::spaced(arguments))
                            })
                            .unwrap_or_else(|| "{}".into());
                        message(
                            &mut prompt,
                            "assistant",
                            Some("commentary"),
                            Some(&recipient),
                            &arguments,
                        );
                    }
                }
                _ => message(
                    &mut prompt,
                    "developer",
                    None,
                    None,
                    &format!("# Instructions\n\n{body}"),
                ),
            }
        }
        if add_generation_prompt {
            prompt.special("<|start|>");
            prompt.text("assistant");
            prompt.special("<|channel|>");
            prompt.text("analysis");
            prompt.special("<|message|>");
        } else if messages
            .last()
            .and_then(|message| message.get("role"))
            .and_then(Value::as_str)
            == Some("assistant")
            && messages
                .last()
                .and_then(|message| message.get("tool_calls"))
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
            && let Some(crate::PromptSpan::Special(last)) = prompt.spans.last_mut()
            && last == "<|end|>"
        {
            *last = "<|return|>".into();
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
        Ok(Box::new(GptOssParser::new(
            GptOssTokens::resolve(encoder)?,
            decoder,
            prompt_ids,
        )?))
    }
    fn stop_token_ids(&self, encoder: &dyn crate::Encoder) -> Result<Vec<u32>, crate::Error> {
        Ok(vec![
            crate::parser::resolve(encoder, &["<|fim_suffix|>", "<|return|>"])?,
            crate::parser::resolve(encoder, &["<|ghissue|>", "<|call|>"])?,
        ])
    }
}
