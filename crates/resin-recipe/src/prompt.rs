use std::sync::Arc;

use crate::{Error, Tokenizer};

/// Ordinary text and explicitly inserted model control tokens.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptSpan {
    Text(String),
    Special(String),
}

/// A prompt before tokenization. Text coalesces across formatting operations so
/// BPE sees the same neighboring bytes as the reference encoder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Prompt {
    pub spans: Vec<PromptSpan>,
}

impl Prompt {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(&mut self, text: impl AsRef<str>) -> &mut Self {
        let text = text.as_ref();
        if !text.is_empty() {
            if let Some(PromptSpan::Text(previous)) = self.spans.last_mut() {
                previous.push_str(text);
            } else {
                self.spans.push(PromptSpan::Text(text.to_owned()));
            }
        }
        self
    }

    pub fn special(&mut self, token: impl Into<String>) -> &mut Self {
        self.spans.push(PromptSpan::Special(token.into()));
        self
    }

    pub fn append(&mut self, prompt: Self) -> &mut Self {
        for span in prompt.spans {
            match span {
                PromptSpan::Text(text) => {
                    self.text(text);
                }
                PromptSpan::Special(token) => {
                    self.special(token);
                }
            }
        }
        self
    }

    pub fn separated<T>(
        &mut self,
        values: impl IntoIterator<Item = T>,
        separator: &str,
        mut write: impl FnMut(&mut Self, T),
    ) -> &mut Self {
        for (index, value) in values.into_iter().enumerate() {
            if index > 0 {
                self.text(separator);
            }
            write(self, value);
        }
        self
    }

    /// Trim template whitespace while preserving control-token provenance.
    pub fn trim(mut self) -> Self {
        if let Some(PromptSpan::Text(first)) = self.spans.first_mut() {
            *first = first
                .trim_start_matches(crate::render::whitespace)
                .to_owned();
        }
        if let Some(PromptSpan::Text(last)) = self.spans.last_mut() {
            *last = last.trim_end_matches(crate::render::whitespace).to_owned();
        }
        self
    }

    /// Slice rendered byte positions, retaining whole control-token spans.
    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        let mut result = Self::new();
        let mut offset = 0;
        for span in &self.spans {
            let (PromptSpan::Text(spelling) | PromptSpan::Special(spelling)) = span;
            let next = offset + spelling.len();
            let first = range.start.max(offset);
            let last = range.end.min(next);
            if first < last {
                if matches!(span, PromptSpan::Special(_)) && first == offset && last == next {
                    result.special(spelling);
                } else {
                    result.text(&spelling[first - offset..last - offset]);
                }
            }
            offset = next;
        }
        result
    }

    pub fn rendered_text(&self) -> String {
        self.spans
            .iter()
            .map(|span| match span {
                PromptSpan::Text(text) | PromptSpan::Special(text) => text.as_str(),
            })
            .collect()
    }

    pub fn encode(&self, encoder: &dyn Encoder) -> Result<Vec<u32>, Error> {
        let mut ids = Vec::new();
        for span in &self.spans {
            match span {
                PromptSpan::Text(text) => ids.extend(encoder.encode_text(text)?),
                PromptSpan::Special(token) => ids.push(encoder.special_id(token)?),
            }
        }
        Ok(ids)
    }
}

/// Tokenization is injected independently of a model's serialization grammar.
pub trait Encoder: Send + Sync {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, Error>;
    fn special_id(&self, spelling: &str) -> Result<u32, Error>;
}

/// A shared HF tokenizer plus a variant that treats special spellings literally.
pub struct TokenizerEncoder {
    tokenizer: Arc<Tokenizer>,
    ordinary: Tokenizer,
}

impl TokenizerEncoder {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Result<Self, Error> {
        let mut ordinary = (*tokenizer).clone();
        // Some checkpoints mark format delimiters (e.g. LFM's <think>) as
        // ordinary added tokens. They still require the same provenance rule.
        let controls: Vec<_> = ordinary
            .get_added_tokens_decoder()
            .into_values()
            .filter(|token| token.content.starts_with('<') || token.content.starts_with('｜'))
            .map(|token| token.special(true))
            .collect();
        ordinary
            .add_special_tokens(controls)
            .map_err(|error| Error::new(error.to_string()))?;
        ordinary.set_encode_special_tokens(true);
        Ok(Self {
            tokenizer,
            ordinary,
        })
    }
}

impl Encoder for TokenizerEncoder {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.ordinary
            .encode_fast(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| Error::new(error.to_string()))
    }

    fn special_id(&self, spelling: &str) -> Result<u32, Error> {
        self.tokenizer
            .token_to_id(spelling)
            .or_else(|| {
                let alias = match spelling {
                    "<|start|>" => "<|im_start|>",
                    "<|end|>" => "<|im_end|>",
                    "<|return|>" => "<|fim_suffix|>",
                    "<|call|>" => "<|ghissue|>",
                    "<|channel|>" => "<|meta_sep|>",
                    "<|message|>" => "<|im_sep|>",
                    "<|constrain|>" => "<|meta_start|>",
                    _ => return None,
                };
                self.tokenizer.token_to_id(alias)
            })
            .ok_or_else(|| Error::new(format!("missing model control token {spelling:?}")))
    }
}
