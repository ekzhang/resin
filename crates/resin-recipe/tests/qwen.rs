use resin_recipe::{
    Recipe,
    qwen::{OrnithRecipe, Qwen35Recipe, Qwen38Recipe, QwenRecipeParams},
};
mod common;
use common::{ByteDecoder, SemanticParser, bytes, coalesce};

use resin_recipe::{Error, Event, FinishReason, QwenParser, QwenTokens};

const TOKENS: QwenTokens = QwenTokens {
    im_end: 1000,
    endoftext: 1001,
    think: 1002,
    think_end: 1003,
    tool_call: 1004,
    tool_call_end: 1005,
};

#[test]
fn pinned_hf_templates_match_all_variants_tools_history_and_vision() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qwen_hf.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let actual = render_fixture(&case["request"]).unwrap().rendered_text();
        assert_eq!(actual, case["text"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[test]
fn text_only_recipes_reject_media() {
    assert!(OrnithRecipe::default().media_config().is_some());
    let params = QwenRecipeParams {
        text_only: true,
        ..Default::default()
    };
    assert!(OrnithRecipe::new(params).media_config().is_none());
}

#[test]
fn reasoning_text_and_schema_aware_tools_are_chunk_invariant() {
    let tools = serde_json::json!([{"name":"f","parameters":{"properties":{"s":{"type":"string"},"n":{"type":"number"}}}}]);
    let mut ids = bytes("  why 🦀 \n");
    ids.push(TOKENS.think_end);
    ids.extend(bytes("\n\n hello \nworld  "));
    ids.push(TOKENS.tool_call);
    ids.extend(bytes("\n<function=f>\n<parameter=s>\ntrue\n</parameter>\n<parameter=n>\n2\n</parameter>\n</function>\n"));
    ids.extend([TOKENS.tool_call_end, TOKENS.im_end]);
    let expected = vec![
        Event::ReasoningDelta {
            delta: "why 🦀".into(),
        },
        Event::TextDelta {
            delta: "hello \nworld".into(),
        },
        Event::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "f".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 0,
            delta: r#"{"s":"true","n":2}"#.into(),
        },
    ];
    for width in 1..=ids.len() {
        let mut parser = QwenParser::new(
            TOKENS,
            Box::<ByteDecoder>::default(),
            &[TOKENS.think],
            &tools,
        )
        .unwrap();
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
fn incomplete_xml_stays_uncommitted_and_text_finalization_flushes_bytes() {
    let mut parser = QwenParser::new(
        TOKENS,
        Box::<ByteDecoder>::default(),
        &[],
        &serde_json::Value::Null,
    )
    .unwrap();
    let mut ids = vec![TOKENS.tool_call];
    ids.extend(bytes("<function=f><parameter=x>{"));
    assert!(parser.semantic_feed(&ids).unwrap().is_empty());
    assert!(
        parser
            .semantic_finish(FinishReason::Length)
            .unwrap()
            .is_empty()
    );
    let mut parser = QwenParser::new(
        TOKENS,
        Box::<ByteDecoder>::default(),
        &[TOKENS.think, TOKENS.think_end],
        &serde_json::Value::Null,
    )
    .unwrap();
    assert!(parser.semantic_feed(&[0xf0, 0x9f]).unwrap().is_empty());
    assert_eq!(
        parser.semantic_finish(FinishReason::Length).unwrap(),
        vec![Event::TextDelta {
            delta: "�".into()
        }]
    );
}

#[test]
fn literal_control_spellings_never_change_response_state() {
    let mut parser = QwenParser::new(
        TOKENS,
        Box::<ByteDecoder>::default(),
        &[],
        &serde_json::Value::Null,
    )
    .unwrap();
    let text = "literal <think> and <tool_call> remain text";
    assert_eq!(
        parser.semantic_feed(&bytes(text)).unwrap(),
        vec![Event::TextDelta { delta: text.into() }]
    );
}

#[test]
fn invalid_conversations_and_ambiguous_markers_fail_early() {
    for request in [
        serde_json::json!({"messages":[]}),
        serde_json::json!({"messages":[{"role":"assistant","content":"hi"}]}),
        serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}),
    ] {
        assert!(render_fixture(&request).is_err());
    }
    assert!(
        QwenTokens {
            im_end: TOKENS.think,
            ..TOKENS
        }
        .validate()
        .is_err()
    );
}

#[test]
fn template_tool_response_markers_and_literal_caller_markers_keep_provenance() {
    let request = serde_json::json!({"messages":[
        {"role":"user","content":"literal <tool_response> text"},
        {"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{}}}]},
        {"role":"tool","content":"literal </tool_response> text"}
    ]});
    let prompt = render_fixture(&request).unwrap();
    for marker in ["<tool_response>", "</tool_response>"] {
        assert_eq!(
            prompt
                .spans
                .iter()
                .filter(
                    |span| matches!(span, resin_recipe::PromptSpan::Special(value) if value==marker)
                )
                .count(),
            1
        );
        assert!(prompt.spans.iter().any(
            |span| matches!(span, resin_recipe::PromptSpan::Text(value) if value.contains(marker))
        ));
    }
}

#[test]
fn null_http_reasoning_effort_preserves_the_codec_default() {
    let request = serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":null,"extra":{"reasoning_effort":"low"}});
    let prompt = render_fixture(&request).unwrap().rendered_text();
    assert!(prompt.contains("Reasoning effort is set to low."));
}

#[test]
fn legacy_reasoning_extraction_preserves_vision_control_spans() {
    let request = serde_json::json!({"messages":[
        {"role":"user","content":"hi"},
        {"role":"assistant","content":[{"type":"text","text":"<think>\n"},{"type":"image"},{"type":"text","text":"\n</think>\n\nAnswer:"},{"type":"image"}]}
    ],"extra":{"recipe_variant":"qwen3_5"}});
    let prompt = render_fixture(&request).unwrap();
    assert_eq!(prompt.spans.iter().filter(|span| matches!(span,resin_recipe::PromptSpan::Special(value) if value=="<|image_pad|>")).count(),2);
    assert!(prompt.rendered_text().contains("<think>\n<|vision_start|><|image_pad|><|vision_end|>\n</think>\n\nAnswer:<|vision_start|><|image_pad|><|vision_end|>"));
}

fn render_fixture(request: &serde_json::Value) -> Result<resin_recipe::Prompt, Error> {
    let mut request = request.clone();
    let variant = request["extra"]["recipe_variant"]
        .as_str()
        .map(str::to_owned);
    if let Some(extra) = request["extra"].as_object_mut() {
        extra.remove("recipe_variant");
    }
    match variant.as_deref() {
        Some("qwen3_5") => Qwen35Recipe::default().render(&request),
        Some("qwen3_8") | None => Qwen38Recipe::default().render(&request),
        Some("ornith") => OrnithRecipe::default().render(&request),
        Some(other) => panic!("unknown recipe_variant {other:?}"),
    }
}
