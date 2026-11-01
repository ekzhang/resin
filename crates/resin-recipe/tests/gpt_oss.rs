use resin_recipe::{Recipe, gpt_oss::GptOssRecipe};
mod common;
use common::{ByteDecoder, SemanticParser, bytes, coalesce};

use resin_recipe::{Event, FinishReason, GptOssParser, GptOssTokens};

const TOKENS: GptOssTokens = GptOssTokens {
    start: 1000,
    end: 1001,
    turn_end: 1002,
    tool_end: 1003,
    channel: 1004,
    message: 1005,
    constrain: 1006,
};

fn header(channel: &str, recipient: Option<&str>) -> Vec<u32> {
    let mut ids = vec![TOKENS.start];
    ids.extend(bytes("assistant"));
    ids.push(TOKENS.channel);
    ids.extend(bytes(channel));
    if let Some(recipient) = recipient {
        ids.extend(bytes(&format!(" to={recipient}")));
    }
    ids.push(TOKENS.message);
    ids
}

#[test]
fn channel_headers_tools_and_unicode_are_independent_of_chunk_boundaries() {
    let prompt = header("analysis", None);
    let mut ids = bytes("Plan 🦀.");
    ids.push(TOKENS.end);
    ids.extend(header("commentary", None));
    ids.extend(bytes("Searching."));
    ids.push(TOKENS.end);
    ids.extend(header("commentary", Some("functions.weather")));
    ids.extend(bytes(r#"{"city":"東京","n":2}"#));
    ids.push(TOKENS.tool_end);
    ids.extend(header("final", None));
    ids.extend(bytes("Done. <|start|> stays literal."));
    ids.push(TOKENS.turn_end);
    ids.extend(bytes("ignored"));
    let expected = vec![
        Event::ReasoningDelta {
            delta: "Plan 🦀.".into(),
        },
        Event::TextDelta {
            delta: "Searching.".into(),
        },
        Event::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "weather".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 0,
            delta: r#"{"city":"東京","n":2}"#.into(),
        },
        Event::TextDelta {
            delta: "Done. <|start|> stays literal.".into(),
        },
    ];
    for width in 1..=ids.len() {
        let mut parser = GptOssParser::new(TOKENS, Box::<ByteDecoder>::default(), &prompt).unwrap();
        let mut events = Vec::new();
        for chunk in ids.chunks(width) {
            events.extend(parser.semantic_feed(chunk).unwrap());
        }
        events.extend(parser.semantic_finish(FinishReason::Stop).unwrap());
        assert_eq!(coalesce(events), expected, "chunk width {width}");
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
fn prompt_initialization_does_not_reemit_a_prompt_body() {
    let mut prompt = header("analysis", None);
    prompt.extend(bytes("Already generated."));
    let mut parser = GptOssParser::new(TOKENS, Box::<ByteDecoder>::default(), &prompt).unwrap();
    assert_eq!(
        parser.semantic_feed(&bytes("New.")).unwrap(),
        vec![Event::ReasoningDelta {
            delta: "New.".into()
        }]
    );
}

#[test]
fn incomplete_and_invalid_tool_calls_stay_uncommitted() {
    for payload in ["{", r#"{"x":1}"#] {
        let mut parser = GptOssParser::new(
            TOKENS,
            Box::<ByteDecoder>::default(),
            &header("commentary", Some("functions.f")),
        )
        .unwrap();
        assert!(parser.semantic_feed(&bytes(payload)).unwrap().is_empty());
        assert!(
            parser
                .semantic_finish(FinishReason::Length)
                .unwrap()
                .is_empty()
        );
    }
    let mut parser = GptOssParser::new(
        TOKENS,
        Box::<ByteDecoder>::default(),
        &header("commentary", Some("functions.f")),
    )
    .unwrap();
    let mut ids = bytes("broken");
    ids.push(TOKENS.tool_end);
    assert!(parser.semantic_feed(&ids).unwrap().is_empty());
}

#[test]
fn tool_arguments_preserve_numeric_lexemes_and_whitespace() {
    let payload = "{\"x\": 3.777859190353473e-35, \"y\": -0.0}";
    let mut parser = GptOssParser::new(
        TOKENS,
        Box::<ByteDecoder>::default(),
        &header("commentary", Some("functions.f")),
    )
    .unwrap();
    let mut ids = bytes(payload);
    ids.push(TOKENS.tool_end);
    let events = parser.semantic_feed(&ids).unwrap();
    assert_eq!(
        events[1],
        Event::ToolCallArgumentsDelta {
            index: 0,
            delta: payload.into()
        }
    );
}

#[test]
fn partial_byte_sequence_flushes_at_a_model_control_boundary() {
    let mut parser = GptOssParser::new(
        TOKENS,
        Box::<ByteDecoder>::default(),
        &header("analysis", None),
    )
    .unwrap();
    assert_eq!(
        parser.semantic_feed(&[0xf0, 0x9f, TOKENS.end]).unwrap(),
        vec![Event::ReasoningDelta {
            delta: "�".into()
        }]
    );
}

#[test]
fn recipient_before_channel_and_constrained_headers_are_hidden() {
    let mut ids = vec![TOKENS.start];
    ids.extend(bytes("assistant to=functions.f"));
    ids.push(TOKENS.channel);
    ids.extend(bytes("commentary"));
    ids.push(TOKENS.constrain);
    ids.extend(bytes("json"));
    ids.push(TOKENS.message);
    ids.extend(bytes("{}"));
    ids.push(TOKENS.tool_end);
    let mut parser = GptOssParser::new(TOKENS, Box::<ByteDecoder>::default(), &[]).unwrap();
    assert_eq!(
        parser.semantic_feed(&ids).unwrap(),
        vec![
            Event::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "f".into()
            },
            Event::ToolCallArgumentsDelta {
                index: 0,
                delta: "{}".into()
            },
        ]
    );
}

#[test]
fn ambiguous_control_ids_are_rejected() {
    assert!(
        GptOssTokens {
            end: TOKENS.start,
            ..TOKENS
        }
        .validate()
        .is_err()
    );
}

#[test]
fn canonical_system_developer_tools_and_generation_prefix() {
    let prompt = GptOssRecipe::default().render(&serde_json::json!({
        "messages": [{"role":"system","content":"Be brief."},{"role":"user","content":"hi"}],
        "tools": [{"name":"weather","description":"Weather.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
        "extra": {"conversation_start_date":"2026-10-21","reasoning_effort":"high"},
    })).unwrap();
    assert_eq!(
        prompt.rendered_text(),
        concat!(
            "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n",
            "Knowledge cutoff: 2024-06\nCurrent date: 2026-10-21\n\nReasoning: high\n\n",
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.\n",
            "Calls to these tools must go to the commentary channel: 'functions'.<|end|>",
            "<|start|>developer<|message|># Instructions\n\nBe brief.\n\n",
            "# Tools\n\n## functions\n\nnamespace functions {\n\n// Weather.\n",
            "type weather = (_: {\ncity: string,\n}) => any;\n\n} // namespace functions<|end|>",
            "<|start|>user<|message|>hi<|end|>",
            "<|start|>assistant<|channel|>analysis<|message|>",
        )
    );
}

#[test]
fn history_tools_reasoning_and_terminal_turns_render_without_python() {
    let request = serde_json::json!({
        "messages": [
            {"role":"user","content":"hi"},
            {"role":"assistant","reasoning_content":"Think.","tool_calls":[{"function":{"name":"f","arguments":{"x":[1,2]}}}]},
            {"role":"tool","name":"f","content":"ok"},
            {"role":"assistant","content":"Done."},
        ],
        "use_system_prompt": false,
        "add_generation_prompt": false,
    });
    let text = GptOssRecipe::default()
        .render(&request)
        .unwrap()
        .rendered_text();
    assert_eq!(
        text,
        concat!(
            "<|start|>user<|message|>hi<|end|>",
            "<|start|>assistant<|channel|>analysis<|message|>Think.<|end|>",
            "<|start|>assistant to=functions.f<|channel|>commentary<|message|>{\"x\": [1, 2]}<|call|>",
            "<|start|>functions.f to=assistant<|channel|>commentary<|message|>ok<|end|>",
            "<|start|>assistant<|channel|>final<|message|>Done.<|return|>",
        )
    );
    let mut dropped = request;
    dropped["auto_drop_analysis"] = serde_json::json!(true);
    assert!(
        !GptOssRecipe::default()
            .render(&dropped)
            .unwrap()
            .rendered_text()
            .contains("Think.")
    );
}

#[test]
fn official_harmony_fixtures_cover_tool_schemas_and_history_options() {
    // Captured from the official Harmony encoder through its OpenAI-message
    // adapter. The live differential harness additionally fuzzes the grammar.
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/gpt_oss_harmony.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let actual = GptOssRecipe::default()
            .render(&case["request"])
            .unwrap()
            .rendered_text();
        assert_eq!(actual, case["text"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[test]
fn null_http_reasoning_effort_preserves_the_codec_default() {
    let request = serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":null,"extra":{"reasoning_effort":"high"}});
    let prompt = GptOssRecipe::default()
        .render(&request)
        .unwrap()
        .rendered_text();
    assert!(prompt.contains("Reasoning: high"));
    let recipe = GptOssRecipe::new(
        serde_json::from_value(serde_json::json!({"reasoning_effort": "low"})).unwrap(),
    );
    let inherited = recipe.render(&serde_json::json!({"messages": request["messages"], "extra": {"reasoning_effort": null}})).unwrap();
    assert!(inherited.rendered_text().contains("Reasoning: low"));
}
