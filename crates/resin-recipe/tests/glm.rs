use resin_recipe::{
    Event, FinishReason, GlmParser, GlmTokens, Recipe,
    glm::{Glm53FlashRecipe, Glm53FlashRecipeParams},
};
use serde_json::{Value, json};
mod common;
use common::{ByteDecoder, SemanticParser, bytes, coalesce};

const TOKENS: GlmTokens = GlmTokens {
    endoftext: 1000,
    user: 1001,
    observation: 1002,
    think: 1003,
    think_end: 1004,
    tool_call: 1005,
    tool_call_end: 1006,
    arg_key: 1007,
    arg_key_end: 1008,
    arg_value: 1009,
    arg_value_end: 1010,
};
fn parser(tools: &Value) -> GlmParser {
    GlmParser::new(
        TOKENS,
        Box::<ByteDecoder>::default(),
        &[TOKENS.think],
        tools,
    )
    .unwrap()
}
fn append_argument(ids: &mut Vec<u32>, key: &str, value: &str) {
    ids.push(TOKENS.arg_key);
    ids.extend(bytes(key));
    ids.extend([TOKENS.arg_key_end, TOKENS.arg_value]);
    ids.extend(bytes(value));
    ids.push(TOKENS.arg_value_end);
}

#[test]
fn pinned_hf_template_matches_reasoning_tools_and_media_slots() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/glm_hf.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let actual = Glm53FlashRecipe::default()
            .render(&case["request"])
            .unwrap();
        assert_eq!(
            actual.rendered_text(),
            case["text"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn defaults_and_request_options_are_consistent_and_unknown_options_fail() {
    let recipe = Glm53FlashRecipe::new(Glm53FlashRecipeParams {
        reasoning_effort: Some("low".into()),
        ..Default::default()
    });
    let request = json!({"messages":[{"role":"user","content":"hi"}]});
    assert!(
        recipe
            .render(&request)
            .unwrap()
            .rendered_text()
            .contains("Reasoning Effort: Low")
    );
    let mut override_request = request.clone();
    override_request["extra"] = json!({"reasoning_effort":"high"});
    assert!(
        recipe
            .render(&override_request)
            .unwrap()
            .rendered_text()
            .contains("Reasoning Effort: High")
    );
    override_request["extra"] = json!({"enable_thinking":false});
    assert!(recipe.render(&override_request).is_err());
    override_request["extra"] = json!({"clear_thinking":"yes"});
    assert!(recipe.render(&override_request).is_err());
}

#[test]
fn reasoning_and_complete_tools_are_invariant_to_every_chunk_width() {
    let tools = json!([{"type":"function","function":{"name":"f","parameters":{"properties":{"string":{"type":"string"},"number":{"type":"integer"},"list":{"type":"array"}}}}}]);
    let mut ids = bytes("think 🦀\n");
    ids.push(TOKENS.think_end);
    ids.extend(bytes("answer\n"));
    ids.push(TOKENS.tool_call);
    ids.extend(bytes("f"));
    append_argument(&mut ids, "string", "  true ");
    append_argument(&mut ids, "number", "2");
    append_argument(&mut ids, "list", "[1, 2]");
    ids.push(TOKENS.tool_call_end);
    ids.push(TOKENS.tool_call);
    ids.extend(bytes("empty"));
    ids.extend([TOKENS.tool_call_end, TOKENS.observation]);
    let expected = vec![
        Event::ReasoningDelta {
            delta: "think 🦀\n".into(),
        },
        Event::TextDelta {
            delta: "answer\n".into(),
        },
        Event::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "f".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 0,
            delta: r#"{"string":"  true ","number":2,"list":[1,2]}"#.into(),
        },
        Event::ToolCallStart {
            index: 1,
            id: "call_1".into(),
            name: "empty".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 1,
            delta: "{}".into(),
        },
    ];
    for width in 1..=ids.len() {
        let mut parser = parser(&tools);
        let mut events = Vec::new();
        for chunk in ids.chunks(width) {
            events.extend(parser.semantic_feed(chunk).unwrap());
        }
        events.extend(parser.semantic_finish(FinishReason::Stop).unwrap());
        assert_eq!(coalesce(events), expected, "width {width}");
        assert!(
            parser
                .semantic_finish(FinishReason::Length)
                .unwrap()
                .is_empty()
        );
        assert!(parser.semantic_feed(&[]).is_err());
    }
}

#[test]
fn malformed_and_truncated_calls_never_commit() {
    let mut incomplete = vec![TOKENS.tool_call];
    incomplete.extend(bytes("f"));
    incomplete.push(TOKENS.arg_key);
    incomplete.extend(bytes("x"));
    for suffix in [vec![], vec![TOKENS.tool_call_end], vec![TOKENS.endoftext]] {
        let mut parser = parser(&Value::Null);
        let mut ids = incomplete.clone();
        ids.extend(suffix);
        assert!(parser.semantic_feed(&ids).unwrap().is_empty());
        assert!(
            parser
                .semantic_finish(FinishReason::Length)
                .unwrap()
                .is_empty()
        );
    }
    let mut duplicate = vec![TOKENS.tool_call];
    duplicate.extend(bytes("f"));
    append_argument(&mut duplicate, "x", "1");
    append_argument(&mut duplicate, "x", "2");
    duplicate.push(TOKENS.tool_call_end);
    assert!(
        parser(&Value::Null)
            .semantic_feed(&duplicate)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn ordinary_marker_spellings_remain_literal_in_rendering_and_parsing() {
    let prompt = Glm53FlashRecipe::default()
        .render(&json!({"messages":[{"role":"user","content":"<think><|assistant|>"}]}))
        .unwrap();
    assert!(prompt.spans.contains(&resin_recipe::PromptSpan::Text(
        "<think><|assistant|>".into()
    )));
    let mut parser = parser(&Value::Null);
    assert_eq!(
        parser
            .semantic_feed(&bytes("literal </think><arg_key>"))
            .unwrap(),
        vec![Event::ReasoningDelta {
            delta: "literal </think><arg_key>".into()
        }]
    );
}

#[test]
fn all_released_stop_markers_end_generation() {
    for stop in [TOKENS.endoftext, TOKENS.user, TOKENS.observation] {
        let mut parser = parser(&Value::Null);
        let mut ids = vec![TOKENS.think_end];
        ids.extend(bytes("ok"));
        ids.push(stop);
        ids.extend(bytes("ignored"));
        assert_eq!(
            coalesce(parser.semantic_feed(&ids).unwrap()),
            vec![Event::TextDelta { delta: "ok".into() }]
        );
    }
    assert!(
        GlmParser::new(
            GlmTokens {
                user: TOKENS.endoftext,
                ..TOKENS
            },
            Box::<ByteDecoder>::default(),
            &[],
            &Value::Null
        )
        .is_err()
    );
}
