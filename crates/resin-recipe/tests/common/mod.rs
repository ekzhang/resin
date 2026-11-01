#![allow(dead_code)]

use resin_recipe::{AttributedText, Decoder, Error, Event, FinishReason, ResponseParser};

/// Byte tokens expose split Unicode sequences without a tokenizer dependency.
#[derive(Default)]
pub struct ByteDecoder(Vec<u8>);

impl Decoder for ByteDecoder {
    fn push(&mut self, ids: &[u32], _token_start: u32) -> Result<AttributedText, Error> {
        for &id in ids {
            self.0
                .push(u8::try_from(id).map_err(|_| Error::new("invalid byte token"))?);
        }
        let length = match std::str::from_utf8(&self.0) {
            Ok(_) => self.0.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(_) => return Err(Error::new("invalid UTF-8")),
        };
        Ok(AttributedText::plain(
            String::from_utf8(self.0.drain(..length).collect()).unwrap(),
        ))
    }

    fn finish(&mut self) -> Result<AttributedText, Error> {
        Ok(AttributedText::plain(
            String::from_utf8_lossy(&std::mem::take(&mut self.0)).into_owned(),
        ))
    }

    fn reset(&mut self) {
        self.0.clear();
    }
}

pub fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

pub fn coalesce(events: Vec<Event>) -> Vec<Event> {
    let mut result = Vec::new();
    for event in events {
        match (result.last_mut(), event) {
            (Some(Event::TextDelta { delta }), Event::TextDelta { delta: next })
            | (Some(Event::ReasoningDelta { delta }), Event::ReasoningDelta { delta: next }) => {
                delta.push_str(&next)
            }
            (_, event) => result.push(event),
        }
    }
    result
}

/// Grammar fixtures compare semantic events; provenance has its own integration tests.
pub trait SemanticParser: ResponseParser {
    fn semantic_feed(&mut self, ids: &[u32]) -> Result<Vec<Event>, Error> {
        self.feed(ids)
            .map(|events| events.into_iter().map(|event| event.event).collect())
    }
    fn semantic_finish(&mut self, reason: FinishReason) -> Result<Vec<Event>, Error> {
        self.finish(reason)
            .map(|events| events.into_iter().map(|event| event.event).collect())
    }
}
impl<T: ResponseParser> SemanticParser for T {}
