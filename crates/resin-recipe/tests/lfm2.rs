use resin_recipe::ResponseParser;
use resin_recipe::{Recipe, lfm2::Lfm2Recipe};
mod common;
use common::SemanticParser;

use resin_recipe::{AttributedText, Decoder, Error, Event, FinishReason, Lfm2Parser, Lfm2Tokens};
use serde_json::{Value, json};

const TOKENS: Lfm2Tokens = Lfm2Tokens {
    message_start: 1000,
    message_end: 1001,
    thinking_start: 1002,
    thinking_end: 1003,
    tool_call_start: 1004,
    tool_call_end: 1005,
};

#[derive(Default)]
struct Bytes(Vec<u8>);

impl Decoder for Bytes {
    fn push(&mut self, ids: &[u32], _token_start: u32) -> Result<AttributedText, Error> {
        self.0
            .extend(ids.iter().map(|&id| u8::try_from(id).unwrap()));
        let length = match std::str::from_utf8(&self.0) {
            Ok(_) => self.0.len(),
            Err(error) => error.valid_up_to(),
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

fn parser() -> Lfm2Parser {
    Lfm2Parser::new(TOKENS, Box::<Bytes>::default()).unwrap()
}
fn ids(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

fn summary(events: &[Event]) -> (String, String, Vec<(String, Value)>) {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tools = Vec::<(String, String)>::new();
    for event in events {
        match event {
            Event::TextDelta { delta } => text.push_str(delta),
            Event::ReasoningDelta { delta } => reasoning.push_str(delta),
            Event::ToolCallStart { index, name, .. } => {
                assert_eq!(*index, tools.len());
                tools.push((name.clone(), String::new()));
            }
            Event::ToolCallArgumentsDelta { index, delta } => tools[*index].1.push_str(delta),
        }
    }
    (
        text,
        reasoning,
        tools
            .into_iter()
            .map(|(name, args)| (name, serde_json::from_str(&args).unwrap()))
            .collect(),
    )
}

#[test]
fn rendering_matches_pinned_huggingface_template() {
    // LiquidAI/LFM2.5-230M@13a53837c4906b4f7405932532ba85d182bb013b.
    for line in include_str!("fixtures/lfm2_prompts.jsonl").lines() {
        let fixture: Value = serde_json::from_str(line).unwrap();
        let prompt = Lfm2Recipe::default().render(&fixture["request"]).unwrap();
        assert_eq!(
            prompt.rendered_text(),
            fixture["expected"].as_str().unwrap()
        );
    }
}

#[test]
fn parses_thinking_text_and_literal_tool_calls_for_every_chunk_size() {
    let mut response = vec![TOKENS.thinking_start];
    response.extend(ids("thinking 🦀"));
    response.push(TOKENS.thinking_end);
    response.extend(ids("answer "));
    response.push(TOKENS.tool_call_start);
    response.extend(ids("[weather(q='SF\\n東京', n=True, xs=[1, None, {'ok': False}], opts={\"enabled\": true, \"value\": null}), clock() ]"));
    response.push(TOKENS.tool_call_end);
    response.push(TOKENS.message_end);
    for size in [1, 2, 7, response.len()] {
        let mut parser = parser();
        let mut events = Vec::new();
        for chunk in response.chunks(size) {
            events.extend(parser.semantic_feed(chunk).unwrap());
        }
        events.extend(parser.semantic_finish(FinishReason::Stop).unwrap());
        assert_eq!(
            summary(&events),
            (
                "answer ".into(),
                "thinking 🦀".into(),
                vec![
                    (
                        "weather".into(),
                        json!({"q":"SF\n東京", "n":true, "xs":[1,null,{"ok":false}], "opts":{"enabled":true,"value":null}})
                    ),
                    ("clock".into(), json!({}))
                ]
            )
        );
    }
}

#[test]
fn truncated_tools_stay_uncommitted_and_literal_markers_are_text() {
    let mut parser = parser();
    let mut events = parser
        .semantic_feed(&ids("<think>literal</think>"))
        .unwrap();
    events.extend(parser.semantic_feed(&[TOKENS.tool_call_start]).unwrap());
    events.extend(parser.semantic_feed(&ids("[f(q='unfinished")).unwrap());
    events.extend(parser.semantic_finish(FinishReason::Length).unwrap());
    assert_eq!(
        summary(&events),
        ("<think>literal</think>".into(), String::new(), vec![])
    );
    assert!(
        parser
            .semantic_finish(FinishReason::Length)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn malformed_tool_calls_are_errors_instead_of_executable_expressions() {
    for tool in [
        "[f(q=__import__('os'))]",
        "[f(q=1,q=2)]",
        "[f(q=1)]; x()",
        "[f(q='bad\\z')]",
        "[f(q=object.attr)]",
    ] {
        let mut parser = parser();
        parser.semantic_feed(&[TOKENS.tool_call_start]).unwrap();
        parser.semantic_feed(&ids(tool)).unwrap();
        assert!(
            parser.semantic_feed(&[TOKENS.tool_call_end]).is_err(),
            "accepted {tool}"
        );
    }
    let mut parser = parser();
    parser.semantic_feed(&[TOKENS.tool_call_start]).unwrap();
    parser
        .semantic_feed(&ids(&format!(
            "[f(q={}0{})]",
            "[".repeat(65),
            "]".repeat(65)
        )))
        .unwrap();
    assert!(parser.semantic_feed(&[TOKENS.tool_call_end]).is_err());
}

#[test]
fn escaped_unicode_surrogate_pairs_in_json_arguments_decode_to_unicode() {
    let mut parser = parser();
    parser.semantic_feed(&[TOKENS.tool_call_start]).unwrap();
    parser
        .semantic_feed(&ids(r#"[f(q={"emoji": "\ud83d\ude00"})]"#))
        .unwrap();
    let events = parser.semantic_feed(&[TOKENS.tool_call_end]).unwrap();
    assert_eq!(
        summary(&events).2,
        vec![("f".into(), json!({"q": {"emoji": "😀"}}))]
    );
}

#[test]
fn json_string_history_arguments_are_accepted_at_the_api_boundary() {
    let prompt = Lfm2Recipe::default().render(&json!({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{\"n\": true}"}}]}]})).unwrap();
    assert!(prompt.rendered_text().contains("[f(n=True)]"));
}

#[test]
fn python_repr_uses_pinned_unicode_15_1_printability() {
    let prompt = Lfm2Recipe::default().render(&json!({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"xs":["\u{13460}\u{2ebf0}\0"]}}}]}]})).unwrap();
    assert!(
        prompt
            .rendered_text()
            .contains("[f(xs=['\\U00013460\u{2ebf0}\\x00'])]")
    );
}

#[test]
fn message_end_is_terminal_across_batches() {
    let mut parser = parser();
    let mut response = ids("answer");
    response.extend([TOKENS.message_end, TOKENS.thinking_start]);
    response.extend(ids("ignored reasoning"));
    response.extend([TOKENS.thinking_end, TOKENS.message_start]);
    response.extend(ids("assistant\nignored text"));
    assert_eq!(
        parser.semantic_feed(&response).unwrap(),
        vec![Event::TextDelta {
            delta: "answer".into()
        }]
    );
    assert!(
        parser
            .semantic_feed(&[TOKENS.thinking_start, u32::from(b'x'), TOKENS.message_start])
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
