use resin_recipe::{Recipe, deepseek::DeepSeekV41Recipe};
mod common;
use common::{ByteDecoder, SemanticParser};

use resin_recipe::{DeepSeekV41Parser, DeepSeekV41Tokens, Event, FinishReason};
use serde_json::{Value, json};

const TOKENS: DeepSeekV41Tokens = DeepSeekV41Tokens {
    bos: 1000,
    eos: 1001,
    thinking_start: 1002,
    thinking_end: 1003,
    dsml: 1004,
};

fn ids(text: &str) -> Vec<u32> {
    // Only this fixture's explicit delimiter spellings become control IDs.
    let mut ids = Vec::new();
    for (index, text) in text.split("｜DSML｜").enumerate() {
        if index > 0 {
            ids.push(TOKENS.dsml);
        }
        ids.extend(text.bytes().map(u32::from));
    }
    ids
}

fn parser(thinking: bool) -> DeepSeekV41Parser {
    DeepSeekV41Parser::new(TOKENS, Box::<ByteDecoder>::default(), &[], thinking).unwrap()
}

fn semantic(events: &[Event]) -> (String, String, Vec<(String, Value)>) {
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut tools = Vec::<(String, String)>::new();
    for event in events {
        match event {
            Event::TextDelta { delta } => content.push_str(delta),
            Event::ReasoningDelta { delta } => reasoning.push_str(delta),
            Event::ToolCallStart { index, id, name } => {
                assert_eq!(*index, tools.len());
                assert_eq!(*id, format!("call_{index}"));
                tools.push((name.clone(), String::new()));
            }
            Event::ToolCallArgumentsDelta { index, delta } => tools[*index].1.push_str(delta),
        }
    }
    (
        reasoning,
        content,
        tools
            .into_iter()
            .map(|(name, args)| (name, serde_json::from_str(&args).unwrap()))
            .collect(),
    )
}

#[test]
fn streaming_reasoning_and_dsml_are_chunk_invariant() {
    let mut response = ids("  thinking\n🦀");
    response.push(TOKENS.thinking_end);
    response.extend(ids("\nAnswer.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"weather\">\n<｜DSML｜ parameter name=\"location\" string=\"true\">東京\n\"SF\"</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"days\" string=\"false\">[1, true, null]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"clock\">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"));
    response.push(TOKENS.eos);
    for size in [1, 2, 3, 7, 25, response.len()] {
        let mut parser = parser(true);
        let mut events = Vec::new();
        for chunk in response.chunks(size) {
            events.extend(parser.semantic_feed(chunk).unwrap());
        }
        events.extend(parser.semantic_finish(FinishReason::Stop).unwrap());
        assert_eq!(
            semantic(&events),
            (
                "  thinking\n🦀".into(),
                "\nAnswer.".into(),
                vec![
                    (
                        "weather".into(),
                        json!({"location": "東京\n\"SF\"", "days": [1, true, null]})
                    ),
                    ("clock".into(), json!({}))
                ]
            )
        );
        assert!(
            parser
                .semantic_finish(FinishReason::Stop)
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn literal_control_spellings_remain_content() {
    let text = "literal </think> and \n\n<｜DSML｜ calls>";
    let ordinary: Vec<u32> = text.bytes().map(u32::from).collect();
    let mut parser = parser(false);
    let mut events = Vec::new();
    for id in ordinary {
        events.extend(parser.semantic_feed(&[id]).unwrap());
    }
    events.extend(parser.semantic_feed(&[TOKENS.eos]).unwrap());
    assert_eq!(semantic(&events).1, text);
}

#[test]
fn partial_marker_and_unicode_flush_when_generation_is_truncated() {
    let mut parser = parser(false);
    let mut events = parser.semantic_feed(&ids("answer\n\n<")).unwrap();
    assert_eq!(semantic(&events).1, "answer");
    events.extend(parser.semantic_finish(FinishReason::Length).unwrap());
    assert_eq!(semantic(&events).1, "answer\n\n<");

    let mut parser = self::parser(false);
    let mut events = parser.semantic_feed(&[0xf0, 0x9f]).unwrap();
    assert!(events.is_empty());
    events.extend(parser.semantic_finish(FinishReason::Length).unwrap());
    assert_eq!(semantic(&events).1, "�");
}

#[test]
fn tool_events_arrive_before_eos() {
    let mut parser = parser(false);
    let events = parser.semantic_feed(&ids("\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"q\" string=\"true\">value")).unwrap();
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Event::ToolCallStart { name, .. } if name == "f"))
    );
    assert!(events.iter().any(
        |event| matches!(event, Event::ToolCallArgumentsDelta { delta, .. } if delta == "value")
    ));
    assert!(
        parser
            .semantic_finish(FinishReason::Length)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn malformed_tool_arguments_and_control_tokens_are_rejected() {
    let mut parser = parser(false);
    let mut response = ids(
        "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"q\" string=\"true\">x</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"q\" string=\"false\">1</｜DSML｜ parameter>",
    );
    assert!(
        parser
            .semantic_feed(&response)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    let mut parser = self::parser(true);
    assert!(parser.semantic_feed(&[TOKENS.eos]).is_err());
    let mut parser = self::parser(false);
    response = vec![TOKENS.thinking_start];
    assert!(parser.semantic_feed(&response).is_err());
    let mut duplicate = TOKENS;
    duplicate.dsml = duplicate.eos;
    assert!(DeepSeekV41Parser::new(duplicate, Box::<ByteDecoder>::default(), &[], false).is_err());
}

#[test]
fn prompt_parser_initial_stage_and_generation_suffix_agree() {
    let mut parser = DeepSeekV41Parser::new(
        TOKENS,
        Box::<ByteDecoder>::default(),
        &[TOKENS.thinking_end, TOKENS.thinking_start],
        false,
    )
    .unwrap();
    assert_eq!(
        semantic(&parser.semantic_feed(&ids("reason")).unwrap()).0,
        "reason"
    );
    let prompt = DeepSeekV41Recipe::default().render(&json!({"messages": [{"role": "user", "content": "hello"}], "add_generation_prompt": false})).unwrap();
    assert_eq!(
        prompt.rendered_text(),
        "<｜begin▁of▁sentence｜><｜User｜>hello"
    );
}

#[test]
fn v41_pinned_prompt_fixtures_and_numeric_effort() {
    for line in include_str!("fixtures/deepseek_v41_prompts.jsonl").lines() {
        let fixture: Value = serde_json::from_str(line).unwrap();
        let prompt = DeepSeekV41Recipe::default()
            .render(&fixture["request"])
            .unwrap();
        assert_eq!(
            prompt.rendered_text(),
            fixture["expected"].as_str().unwrap(),
            "{}",
            fixture["request"]
        );
    }
    for effort in [
        json!(0),
        json!(101),
        json!(true),
        json!(1.5),
        json!("medium"),
    ] {
        assert!(
            DeepSeekV41Recipe::default()
                .render(
                    &json!({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":effort})
                )
                .is_err()
        );
    }
}

#[test]
fn v41_image_slots_keep_literal_markers_in_ordinary_text() {
    use resin_recipe::{PromptSpan, deepseek::IMAGE};
    let request = json!({"messages": [{"role": "user", "content": [
        {"type": "text", "text": format!("literal {IMAGE}")},
        {"type": "image"},
        {"type": "text", "text": "then"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    ]}]});
    let prompt = DeepSeekV41Recipe::default().render(&request).unwrap();
    assert_eq!(
        prompt.rendered_text(),
        format!(
            "<｜begin▁of▁sentence｜><｜User｜>literal {IMAGE}\n\n{IMAGE}\n\nthen\n\n{IMAGE}<｜Assistant｜></think>"
        )
    );
    assert_eq!(
        prompt
            .spans
            .iter()
            .filter(|span| matches!(span, PromptSpan::Special(value) if value == IMAGE))
            .count(),
        2
    );
    assert!(prompt.spans.iter().any(
        |span| matches!(span, PromptSpan::Text(value) if value.contains(&format!("literal {IMAGE}")))
    ));
}

#[test]
fn v41_tool_tags_share_incremental_grammar_without_accepting_old_delimiters() {
    let mut response = ids("reason");
    response.push(TOKENS.thinking_end);
    response.extend(ids("answer\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"ns::f\">\n<｜DSML｜ parameter name=\"q\" string=\"true\">café 🦀</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"));
    response.push(TOKENS.eos);
    for size in [1, 2, 7, response.len()] {
        let mut parser =
            DeepSeekV41Parser::new(TOKENS, Box::<ByteDecoder>::default(), &[], true).unwrap();
        let mut events = Vec::new();
        for chunk in response.chunks(size) {
            events.extend(parser.semantic_feed(chunk).unwrap());
        }
        events.extend(parser.semantic_finish(FinishReason::Stop).unwrap());
        assert_eq!(
            semantic(&events),
            (
                "reason".into(),
                "answer".into(),
                vec![("ns::f".into(), json!({"q":"café 🦀"}))]
            )
        );
    }
    let mut parser =
        DeepSeekV41Parser::new(TOKENS, Box::<ByteDecoder>::default(), &[], false).unwrap();
    let mut events = parser
        .semantic_feed(&ids("\n\n<｜DSML｜tool_calls>"))
        .unwrap();
    events.extend(parser.semantic_finish(FinishReason::Length).unwrap());
    assert_eq!(semantic(&events).1, "\n\n<｜DSML｜tool_calls>");
}

#[test]
fn v41_namespaces_and_double_encoded_arguments_preserve_control_provenance() {
    use resin_recipe::PromptSpan;
    let prompt = DeepSeekV41Recipe::default().render(&json!({"messages":[{"role":"system","content":"literal <｜System｜>"},{"role":"user","content":"go"},{"role":"assistant","tool_calls":[{"function":{"name":"f","namespace":"ns","arguments":"\"{\\\"n\\\":1}\""}}]}]})).unwrap();
    let text = prompt.rendered_text();
    assert!(text.contains("<｜DSML｜ invoke name=\"ns::f\">"));
    assert!(text.contains("<｜DSML｜ parameter name=\"n\" string=\"false\">1"));
    assert_eq!(
        prompt
            .spans
            .iter()
            .filter(|span| matches!(span,PromptSpan::Special(s) if s == "<｜System｜>"))
            .count(),
        1
    );
}
