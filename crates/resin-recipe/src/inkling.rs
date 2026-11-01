mod render;

use crate::parser::{EventSink, Grammar, Parser, emit_tool, tokens};
use crate::{AttributedText, Decoder, Error, FinishReason};
use serde_json::Value;

tokens! { InklingTokens {
    message_model => ["<|message_model|>"],
    content_text => ["<|content_text|>"],
    content_thinking => ["<|content_thinking|>"],
    content_invoke_tool_json => ["<|content_invoke_tool_json|>"],
    content_invoke_tool_text => ["<|content_invoke_tool_text|>"],
    content_tool_error => ["<|content_tool_error|>"],
    content_model_end_sampling => ["<|content_model_end_sampling|>"],
    end_message => ["<|end_message|>"],
}}

#[derive(Clone, Copy)]
enum State {
    Header,
    Content,
    Reasoning,
    Tool,
    Done,
}

/// Inkling's token-delimited text, reasoning, and JSON tool-call blocks.
pub type InklingParser = Parser<InklingGrammar>;
pub struct InklingGrammar {
    tokens: InklingTokens,
    state: State,
    tool: String,
    tool_index: usize,
}

impl InklingParser {
    pub fn new(
        tokens: InklingTokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
    ) -> Result<Self, Error> {
        tokens.validate()?;
        let state = prompt_ids
            .iter()
            .rev()
            .find_map(|&id| {
                if id == tokens.content_thinking {
                    Some(State::Reasoning)
                } else if id == tokens.content_text {
                    Some(State::Content)
                } else if tokens.ids().contains(&id) {
                    Some(State::Header)
                } else {
                    None
                }
            })
            .unwrap_or(State::Header);
        Ok(Self::with_grammar(
            InklingGrammar {
                tokens,
                state,
                tool: String::new(),
                tool_index: 0,
            },
            decoder,
        ))
    }
}

impl Grammar for InklingGrammar {
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
        matches!(self.state, State::Done)
    }
    fn text(&mut self, text: AttributedText, events: &mut EventSink) -> Result<(), Error> {
        if !text.is_empty() {
            match self.state {
                State::Header | State::Done => {}
                State::Content => events.text(text, false),
                State::Reasoning => events.text(text, true),
                State::Tool => self.tool.push_str(&text),
            }
        }
        Ok(())
    }
    fn control(&mut self, id: u32, events: &mut EventSink) -> Result<(), Error> {
        if matches!(self.state, State::Tool) && !self.tool.is_empty() {
            let payload: Value = serde_json::from_str(&self.tool)
                .map_err(|_| Error::new("Inkling emitted an invalid JSON tool call"))?;
            let name = payload
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::new("Inkling tool call must contain a string name"))?;
            let empty = Value::Object(Default::default());
            let arguments = payload
                .get("args")
                .or_else(|| payload.get("arguments"))
                .unwrap_or(&empty);
            if !arguments.is_object() {
                return Err(Error::new("Inkling tool call arguments must be an object"));
            }
            emit_tool(&mut self.tool_index, name, arguments.to_string(), events);
        }
        self.tool.clear();
        self.state = if id == self.tokens.content_model_end_sampling {
            State::Done
        } else if id == self.tokens.message_model {
            State::Header
        } else if id == self.tokens.content_thinking {
            State::Reasoning
        } else if id == self.tokens.content_invoke_tool_json {
            State::Tool
        } else {
            State::Content
        };
        Ok(())
    }
    fn finish(&mut self, _reason: FinishReason, _events: &mut EventSink) -> Result<(), Error> {
        self.tool.clear();
        Ok(())
    }
}

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct InklingRecipeParams {
    pub add_generation_prompt: Option<bool>,
    pub reasoning_effort: Option<crate::ReasoningEffort>,
    #[serde(skip_serializing)]
    pub media_config: Option<crate::media::inkling::Config>,
}

#[derive(Clone, Debug, Default)]
pub struct InklingRecipe {
    params: InklingRecipeParams,
}

impl InklingRecipe {
    pub fn new(params: InklingRecipeParams) -> Self {
        Self { params }
    }
}
