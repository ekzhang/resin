use resin_recipe::ResponseParser;
mod common;
use common::{SemanticParser, coalesce};

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use resin_recipe::{
    AttributedText, Decoder, Error, Event, FinishReason, InklingParser, InklingTokens,
};

const TOKENS: InklingTokens = InklingTokens {
    message_model: 1000,
    content_text: 1001,
    content_thinking: 1002,
    content_invoke_tool_json: 1003,
    content_invoke_tool_text: 1004,
    content_tool_error: 1005,
    content_model_end_sampling: 1006,
    end_message: 1007,
};

/// Byte tokens make split Unicode sequences observable without a tokenizer dependency.
#[derive(Default)]
struct ByteDecoder {
    pending: Vec<u8>,
    pushes: Arc<AtomicUsize>,
}

impl Decoder for ByteDecoder {
    fn push(&mut self, ids: &[u32], _token_start: u32) -> Result<AttributedText, Error> {
        self.pushes.fetch_add(1, Ordering::Relaxed);
        for &id in ids {
            if id == 900 {
                // A marker spelling in ordinary text must never change parser state.
                self.pending.extend_from_slice(b"<|content_thinking|>");
            } else {
                self.pending
                    .push(u8::try_from(id).map_err(|_| Error::new("invalid byte token"))?);
            }
        }
        let length = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(_) => return Err(Error::new("invalid UTF-8")),
        };
        let text = String::from_utf8(self.pending.drain(..length).collect()).unwrap();
        Ok(AttributedText::plain(text))
    }

    fn finish(&mut self) -> Result<AttributedText, Error> {
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        Ok(AttributedText::plain(text))
    }

    fn reset(&mut self) {
        self.pending.clear();
    }
}

fn text_ids(text: &str) -> Vec<u32> {
    text.as_bytes()
        .iter()
        .map(|&byte| u32::from(byte))
        .collect()
}

fn parser(prompt: &[u32]) -> InklingParser {
    InklingParser::new(TOKENS, Box::<ByteDecoder>::default(), prompt).unwrap()
}

fn text(text: &str) -> Event {
    Event::TextDelta {
        delta: text.to_owned(),
    }
}

fn reasoning(text: &str) -> Event {
    Event::ReasoningDelta {
        delta: text.to_owned(),
    }
}

#[test]
fn chunk_boundaries_preserve_reasoning_text_and_tool_events() {
    let mut ids = vec![TOKENS.content_thinking];
    ids.extend(text_ids("thinking 💭"));
    ids.extend([
        TOKENS.end_message,
        TOKENS.message_model,
        TOKENS.content_text,
    ]);
    ids.extend(text_ids("Hello 🌍"));
    ids.push(TOKENS.content_invoke_tool_json);
    ids.extend(text_ids(r#"{"name":"weather","args":{"z":1,"a":"SF"}}"#));
    ids.extend([TOKENS.end_message, TOKENS.content_invoke_tool_json]);
    ids.extend(text_ids(r#"{"name":"clock","arguments":{}}"#));
    ids.extend([TOKENS.end_message, TOKENS.content_model_end_sampling]);

    let expected = vec![
        reasoning("thinking 💭"),
        text("Hello 🌍"),
        Event::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "weather".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 0,
            delta: r#"{"z":1,"a":"SF"}"#.into(),
        },
        Event::ToolCallStart {
            index: 1,
            id: "call_1".into(),
            name: "clock".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 1,
            delta: "{}".into(),
        },
    ];
    for width in 1..=ids.len() {
        let mut parser = parser(&[TOKENS.message_model]);
        let mut events = Vec::new();
        for chunk in ids.chunks(width) {
            events.extend(parser.semantic_feed(chunk).unwrap());
        }
        events.extend(parser.semantic_finish(FinishReason::Stop).unwrap());
        assert_eq!(coalesce(events), expected, "chunk width {width}");
    }
}

#[test]
fn prompt_tail_selects_the_open_block_and_headers_stay_hidden() {
    let mut thinking_prompt = vec![TOKENS.message_model, TOKENS.content_thinking];
    thinking_prompt.extend(text_ids("already open"));
    assert_eq!(
        parser(&thinking_prompt)
            .semantic_feed(&text_ids("continued"))
            .unwrap(),
        vec![reasoning("continued")]
    );
    assert_eq!(
        parser(&[TOKENS.content_text])
            .semantic_feed(&text_ids("visible"))
            .unwrap(),
        vec![text("visible")]
    );

    let mut parser = parser(&[TOKENS.content_thinking, TOKENS.end_message]);
    assert!(
        parser
            .semantic_feed(&text_ids("hidden header"))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        parser
            .semantic_feed(&[TOKENS.content_text, u32::from(b'x')])
            .unwrap(),
        vec![text("x")]
    );
}

#[test]
fn only_control_token_ids_change_state() {
    let mut parser = parser(&[TOKENS.content_text]);
    assert_eq!(
        parser.semantic_feed(&[900]).unwrap(),
        vec![text("<|content_thinking|>")]
    );
    assert_eq!(
        parser.semantic_feed(&text_ids(" still text")).unwrap(),
        vec![text(" still text")]
    );
    for marker in [TOKENS.content_invoke_tool_text, TOKENS.content_tool_error] {
        assert_eq!(
            parser.semantic_feed(&[marker, u32::from(b'x')]).unwrap(),
            vec![text("x")]
        );
    }
}

#[test]
fn decoder_batches_token_runs_and_flushes_at_control_boundaries() {
    let pushes = Arc::new(AtomicUsize::new(0));
    let decoder = ByteDecoder {
        pushes: Arc::clone(&pushes),
        ..Default::default()
    };
    let mut parser = InklingParser::new(TOKENS, Box::new(decoder), &[TOKENS.content_text]).unwrap();
    let mut ids = text_ids("hello");
    ids.extend([0xf0, 0x9f, TOKENS.content_thinking]);
    ids.extend(text_ids("next"));
    assert_eq!(
        parser.semantic_feed(&ids).unwrap(),
        vec![text("hello"), text("�"), reasoning("next")]
    );
    assert_eq!(pushes.load(Ordering::Relaxed), 2);
}

#[test]
fn finish_flushes_text_and_is_idempotent() {
    for reason in [FinishReason::Stop, FinishReason::Length] {
        let mut parser = parser(&[TOKENS.content_text]);
        assert!(parser.semantic_feed(&[0xf0, 0x9f]).unwrap().is_empty());
        assert_eq!(parser.semantic_finish(reason).unwrap(), vec![text("�")]);
        assert!(parser.semantic_finish(reason).unwrap().is_empty());
        assert!(
            parser
                .semantic_feed(&[])
                .unwrap_err()
                .to_string()
                .contains("after finish")
        );
    }
}

#[test]
fn unfinished_tools_are_discarded_even_when_the_json_is_complete() {
    for payload in [
        r#"{"name":"weather","args":{"city":"SF"}}"#,
        r#"{"name":"weather","args": {"#,
    ] {
        let mut parser = parser(&[TOKENS.message_model]);
        let mut ids = vec![TOKENS.content_invoke_tool_json];
        ids.extend(text_ids(payload));
        assert!(parser.semantic_feed(&ids).unwrap().is_empty());
        assert!(
            parser
                .semantic_finish(FinishReason::Length)
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn closed_tools_validate_json_names_and_argument_objects() {
    for (payload, expected) in [
        ("{", "invalid JSON"),
        ("[]", "string name"),
        (r#"{"name":4}"#, "string name"),
        (r#"{"name":"weather","args":[]}"#, "must be an object"),
        (r#"{"name":"weather","args":null}"#, "must be an object"),
    ] {
        let mut parser = parser(&[]);
        let mut ids = vec![TOKENS.content_invoke_tool_json];
        ids.extend(text_ids(payload));
        ids.push(TOKENS.end_message);
        assert!(
            parser
                .semantic_feed(&ids)
                .unwrap_err()
                .to_string()
                .contains(expected),
            "{payload}"
        );
    }
    let mut parser = parser(&[]);
    assert!(
        parser
            .semantic_feed(&[TOKENS.content_invoke_tool_json, TOKENS.end_message])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn duplicate_markers_are_rejected_and_events_match_python_dictionary_shapes() {
    let ambiguous = InklingTokens {
        end_message: TOKENS.content_text,
        ..TOKENS
    };
    assert!(ambiguous.validate().is_err());
    assert!(InklingParser::new(ambiguous, Box::<ByteDecoder>::default(), &[]).is_err());
    let event = Event::ToolCallStart {
        index: 2,
        id: "call_2".into(),
        name: "weather".into(),
    };
    let value = serde_json::to_value(&event).unwrap();
    assert_eq!(
        value,
        serde_json::json!({"type": "tool_call_start", "index": 2, "id": "call_2", "name": "weather"})
    );
    assert_eq!(serde_json::from_value::<Event>(value).unwrap(), event);
}

#[test]
fn sampling_end_is_terminal_across_batches() {
    let mut parser = parser(&[TOKENS.content_text]);
    let mut ids = text_ids("answer");
    ids.push(TOKENS.content_model_end_sampling);
    ids.extend(text_ids("ignored text"));
    ids.extend([TOKENS.message_model, TOKENS.content_thinking]);
    ids.extend(text_ids("ignored reasoning"));
    assert_eq!(parser.semantic_feed(&ids).unwrap(), vec![text("answer")]);
    assert!(
        parser
            .semantic_feed(&[TOKENS.content_text, u32::from(b'x')])
            .unwrap()
            .is_empty()
    );
    assert_eq!(parser.token_channel(u32::from(b'x')), None);
    assert!(
        parser
            .semantic_finish(FinishReason::Stop)
            .unwrap()
            .is_empty()
    );
}
