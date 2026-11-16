use resin_recipe::deepseek::DeepSeekV41Recipe;
use resin_recipe::gemma::Gemma4Recipe;
use resin_recipe::gpt_oss::GptOssRecipe;
use resin_recipe::inkling2::Inkling2Recipe;
use resin_recipe::lfm2::Lfm2Recipe;
use resin_recipe::qwen::Qwen38Recipe;
use resin_recipe::{
    AttributedEvent, Decoder, Event, FinishReason, Recipe, ResponseParser, Tokenizer,
    TokenizerDecoder, TokenizerEncoder,
};
use serde_json::{Map, json};
use std::{collections::BTreeMap, sync::Arc};

const CONTROLS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    "<|fim_suffix|>",
    "<|ghissue|>",
    "<|meta_sep|>",
    "<|im_sep|>",
    "<|meta_start|>",
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|content_tool_error|>",
    "<|content_model_end_sampling|>",
    "<|end_message|>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<think>",
    "</think>",
    "｜DSML｜",
    "<|endoftext|>",
    "<tool_call>",
    "</tool_call>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|\"|>",
    "<eos>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
];
fn byte_character(byte: u8) -> char {
    if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte) {
        return char::from(byte);
    }
    let offset = (0..byte)
        .filter(|b| !(33..=126).contains(b) && !(161..=172).contains(b) && !(174..=255).contains(b))
        .count();
    char::from_u32(256 + offset as u32).unwrap()
}
fn tokenizer(fallback: bool, extra: &[&str]) -> Arc<Tokenizer> {
    let mut vocab = Map::new();
    for byte in 0..=255_u8 {
        vocab.insert(
            if fallback {
                format!("<0x{byte:02X}>")
            } else {
                byte_character(byte).to_string()
            },
            json!(byte),
        );
    }
    for (index, text) in CONTROLS.iter().chain(extra).enumerate() {
        vocab.insert((*text).to_owned(), json!(256 + index));
    }
    let decoder = if fallback {
        json!({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]})
    } else {
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false})
    };
    Arc::new(Tokenizer::from_bytes(serde_json::to_vec(&json!({"version":"1.0","added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":decoder,"model":{"type":"BPE","vocab":vocab,"merges":[]}})).unwrap()).unwrap())
}
fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}
fn id(tokenizer: &Tokenizer, text: &str) -> u32 {
    tokenizer.token_to_id(text).unwrap()
}
fn token_parts(tokenizer: &Tokenizer, parts: &[&str]) -> Vec<u32> {
    parts
        .iter()
        .flat_map(|part| {
            tokenizer
                .token_to_id(part)
                .map_or_else(|| bytes(part), |id| vec![id])
        })
        .collect()
}

fn recipe<R: Recipe + Default + 'static>() -> Box<dyn Recipe> {
    Box::new(R::default())
}

fn parser(
    model: &dyn Recipe,
    tokenizer: Arc<Tokenizer>,
    prompt: &[u32],
) -> Box<dyn ResponseParser> {
    model
        .create_parser(
            &TokenizerEncoder::new(tokenizer.clone()).unwrap(),
            Box::new(TokenizerDecoder::new(tokenizer).unwrap()),
            &json!({}),
            prompt,
        )
        .unwrap()
}

type Ownership = BTreeMap<usize, Vec<(usize, usize, usize, usize, usize)>>;
fn normalize(events: Vec<AttributedEvent>) -> (String, String, Ownership) {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut owners = BTreeMap::<_, Vec<_>>::new();
    for event in events {
        let target = match event.event {
            Event::TextDelta { delta } => {
                content.push_str(&delta);
                (&content, delta.len(), 0)
            }
            Event::ReasoningDelta { delta } => {
                reasoning.push_str(&delta);
                (&reasoning, delta.len(), 1)
            }
            _ => {
                assert!(event.spans.is_empty());
                continue;
            }
        };
        let base = target.0.len() - target.1;
        for span in event.spans {
            assert!(span.text.end as usize <= target.1);
            assert!(span.source.start <= span.source.end);
            owners.entry(span.token as usize).or_default().push((
                span.source.start as usize,
                span.source.end as usize,
                base + span.text.start as usize,
                base + span.text.end as usize,
                target.2,
            ));
        }
    }
    (content, reasoning, owners)
}

#[test]
fn all_model_grammars_preserve_unicode_and_source_indices_across_batching() {
    for (model, prompt, before, transition) in [
        (
            recipe::<GptOssRecipe>(),
            &[
                "<|im_start|>",
                "assistant",
                "<|meta_sep|>",
                "analysis",
                "<|im_sep|>",
            ][..],
            &[][..],
            &["<|im_end|>", "<|im_start|>", "assistant", "<|im_sep|>"][..],
        ),
        (
            recipe::<Inkling2Recipe>(),
            &["<|content_thinking|>"],
            &[],
            &["<|content_text|>"],
        ),
        (
            recipe::<DeepSeekV41Recipe>(),
            &["<think>"],
            &[],
            &["</think>"],
        ),
        (recipe::<Qwen38Recipe>(), &["<think>"], &[], &["</think>"]),
        (
            recipe::<Gemma4Recipe>(),
            &["<|channel>"],
            &[],
            &["<channel|>"],
        ),
        (recipe::<Lfm2Recipe>(), &[], &["<think>"], &["</think>"]),
    ] {
        for fallback in [false, true] {
            let tokenizer = tokenizer(fallback, &[]);
            let prompt = token_parts(&tokenizer, prompt);
            let mut generated = token_parts(&tokenizer, before);
            generated.extend(bytes("hi 🦀!"));
            generated.extend(token_parts(&tokenizer, transition));
            generated.extend(bytes("ok 🧠!"));
            let mut expected = None;
            for chunk_size in [1, 2, 3, 7, generated.len()] {
                let mut decoder = parser(model.as_ref(), tokenizer.clone(), &prompt);
                let mut events = Vec::new();
                for chunk in generated.chunks(chunk_size) {
                    events.extend(decoder.feed(chunk).unwrap());
                }
                events.extend(decoder.finish(FinishReason::Length).unwrap());
                let actual = normalize(events);
                assert_eq!(
                    (&actual.0, &actual.1),
                    (&"ok 🧠!".to_owned(), &"hi 🦀!".to_owned())
                );
                assert_eq!(actual.2.len(), "hi 🦀!ok 🧠!".len());
                if let Some(expected) = &expected {
                    assert_eq!(&actual, expected);
                } else {
                    expected = Some(actual);
                }
            }
        }
    }
}

#[test]
fn qwen_trim_retains_delayed_internal_whitespace_and_marks_partial_tokens() {
    let tokenizer = tokenizer(false, &["ĠhelloĠ", "worldĠ"]);
    let mut parser = parser(recipe::<Qwen38Recipe>().as_ref(), tokenizer.clone(), &[]);
    let first = id(&tokenizer, "ĠhelloĠ");
    let second = id(&tokenizer, "worldĠ");
    let mut events = parser.feed(&[first]).unwrap();
    events.extend(parser.feed(&[second]).unwrap());
    events.extend(parser.finish(FinishReason::Length).unwrap());
    let (text, _, owners) = normalize(events);
    assert_eq!(text, "hello world");
    assert_eq!(owners[&0], vec![(1, 6, 0, 5, 0), (6, 7, 5, 6, 0)]);
    assert_eq!(owners[&1], vec![(0, 5, 6, 11, 0)]);
}

#[test]
fn gemma_header_body_inside_one_token_and_deepseek_marker_prefix_are_attributed() {
    let tokenizer = tokenizer(false, &["analysisĊbody", "aĊĊ<", "x"]);
    let mut gemma = parser(recipe::<Gemma4Recipe>().as_ref(), tokenizer.clone(), &[]);
    let events = gemma
        .feed(&[
            id(&tokenizer, "<|channel>"),
            id(&tokenizer, "analysisĊbody"),
        ])
        .unwrap();
    let (_, reasoning, owners) = normalize(events);
    assert_eq!(reasoning, "body");
    assert_eq!(owners[&1], vec![(9, 13, 0, 4, 1)]);
    let mut deepseek = parser(
        recipe::<DeepSeekV41Recipe>().as_ref(),
        tokenizer.clone(),
        &[],
    );
    let mut events = deepseek.feed(&[id(&tokenizer, "aĊĊ<")]).unwrap();
    events.extend(deepseek.feed(&[id(&tokenizer, "x")]).unwrap());
    events.extend(deepseek.finish(FinishReason::Length).unwrap());
    let (text, _, owners) = normalize(events);
    assert_eq!(text, "a\n\n<x");
    assert_eq!(owners[&0], vec![(0, 1, 0, 1, 0), (1, 4, 1, 4, 0)]);
}

#[test]
fn incomplete_utf8_flush_owns_replacements() {
    for fallback in [false, true] {
        let tokenizer = tokenizer(fallback, &[]);
        let mut decoder = TokenizerDecoder::new(tokenizer).unwrap();
        assert!(decoder.push(&[0xF0, 0x9F], 7).unwrap().is_empty());
        let tail = decoder.finish().unwrap();
        assert_eq!(tail.text, if fallback { "��" } else { "�" });
        assert_eq!(
            tail.spans.iter().map(|s| s.token).collect::<Vec<_>>(),
            vec![7, 8]
        );
        assert!(tail.spans.iter().all(|s| s.source == (0..1)));
    }
}

#[test]
fn transformed_tools_have_no_fabricated_provenance_and_serde_keeps_flat_events() {
    let tokenizer = tokenizer(false, &[]);
    for (model, open, body, close) in [
        (
            recipe::<Inkling2Recipe>(),
            "<|content_invoke_tool_json|>",
            r#"{"name":"f","args":{"a":1}}"#,
            "<|content_text|>",
        ),
        (
            recipe::<Qwen38Recipe>(),
            "<tool_call>",
            "<function=f><parameter=a>1</parameter></function>",
            "</tool_call>",
        ),
        (
            recipe::<Gemma4Recipe>(),
            "<|tool_call>",
            "call:f{a:1}",
            "<tool_call|>",
        ),
        (
            recipe::<Lfm2Recipe>(),
            "<|tool_call_start|>",
            "[f(a=1)]",
            "<|tool_call_end|>",
        ),
    ] {
        let mut parser = parser(model.as_ref(), tokenizer.clone(), &[]);
        let mut ids = vec![id(&tokenizer, open)];
        ids.extend(bytes(body));
        ids.push(id(&tokenizer, close));
        let events = parser.feed(&ids).unwrap();
        assert!(
            events
                .iter()
                .any(|event| matches!(event.event, Event::ToolCallStart { .. }))
        );
        assert!(events.iter().all(|event| event.spans.is_empty()));
    }
    let mut parser = parser(recipe::<Qwen38Recipe>().as_ref(), tokenizer, &[]);
    let event = parser.feed(&bytes("x")).unwrap().pop().unwrap();
    let value = serde_json::to_value(&event).unwrap();
    assert_eq!(value["type"], "text_delta");
    assert_eq!(value["delta"], "x");
    assert_eq!(value["token_spans"][0], json!([0, 0, 1, 0, 1]));
    assert_eq!(
        serde_json::from_value::<AttributedEvent>(value).unwrap(),
        event
    );
    let plain: AttributedEvent = Event::TextDelta { delta: "x".into() }.into();
    assert_eq!(
        serde_json::to_value(plain).unwrap(),
        json!({"type":"text_delta","delta":"x"})
    );
}

#[test]
fn source_bytes_share_decoder_conventions_and_unsupported_chains_fail_upfront() {
    use resin_recipe::TokenBytes;
    let tokenizer = tokenizer(true, &["▁hello", "<0x0000>", "<0xGG>"]);
    let bytes = TokenBytes::new(&tokenizer);
    for (spelling, expected) in [
        ("▁hello", b" hello".as_slice()),
        ("<0x0000>", b"<0x0000>"),
        ("<0xGG>", b"<0xGG>"),
    ] {
        assert_eq!(
            bytes.get(id(&tokenizer, spelling), spelling).unwrap(),
            expected
        );
    }
    let tokenizer = Tokenizer::from_bytes(br#"{"version":"1.0","added_tokens":[],"decoder":{"type":"Fuse"},"model":{"type":"BPE","vocab":{"hello":0},"merges":[]}}"#).unwrap();
    assert!(TokenizerDecoder::new(Arc::new(tokenizer)).is_err());
}

#[test]
fn qwen_and_deepseek_tool_channel_hints_exclude_deferred_content() {
    let tokenizer = tokenizer(false, &[]);
    let mut qwen = parser(recipe::<Qwen38Recipe>().as_ref(), tokenizer.clone(), &[]);
    assert_eq!(qwen.token_channel(u32::from(b'x')), None);
    let open = id(&tokenizer, "<tool_call>");
    assert_eq!(qwen.token_channel(open), Some("control"));
    qwen.feed(&[open]).unwrap();
    assert_eq!(qwen.token_channel(u32::from(b'x')), Some("tool"));
    qwen.feed(&bytes("<function=f></function>")).unwrap();
    qwen.feed(&[id(&tokenizer, "</tool_call>")]).unwrap();
    assert_eq!(qwen.token_channel(u32::from(b' ')), Some("tool"));
    qwen.feed(&[id(&tokenizer, "<think>")]).unwrap();
    assert_eq!(qwen.token_channel(u32::from(b'x')), None);

    let mut deepseek = parser(
        recipe::<DeepSeekV41Recipe>().as_ref(),
        tokenizer.clone(),
        &[],
    );
    deepseek.feed(&bytes("\n\n<")).unwrap();
    assert_eq!(deepseek.token_channel(u32::from(b'x')), None);
    deepseek.feed(&[id(&tokenizer, "｜DSML｜")]).unwrap();
    assert_eq!(deepseek.token_channel(u32::from(b'x')), None);
    deepseek.feed(&bytes(" calls>")).unwrap();
    assert_eq!(deepseek.token_channel(u32::from(b'<')), Some("tool"));
}

#[test]
fn compact_spans_keep_split_scalars_and_move_full_buffers() {
    use resin_recipe::{AttributedText, TokenSpan};
    assert_eq!(std::mem::size_of::<TokenSpan>(), 20);
    let first = format!("a{}", byte_character(0xC3));
    let second = format!("{}{}", byte_character(0xA9), byte_character(b' '));
    let tokenizer = tokenizer(false, &[&first, &second]);
    let mut decoder = TokenizerDecoder::new(tokenizer.clone()).unwrap();
    assert!(
        decoder
            .push(&[id(&tokenizer, &first)], 0)
            .unwrap()
            .is_empty()
    );
    let mut text = decoder.push(&[id(&tokenizer, &second)], 1).unwrap();
    assert_eq!(text.text, "aé ");
    assert_eq!(
        text.spans,
        vec![
            TokenSpan {
                token: 0,
                source: 0..1,
                text: 0..1
            },
            TokenSpan {
                token: 0,
                source: 1..2,
                text: 1..3
            },
            TokenSpan {
                token: 1,
                source: 0..1,
                text: 1..3
            },
            TokenSpan {
                token: 1,
                source: 1..2,
                text: 3..4
            },
        ]
    );
    let (text_ptr, spans_ptr) = (text.text.as_ptr(), text.spans.as_ptr());
    let whole = text.take(text.len());
    assert_eq!(
        (whole.text.as_ptr(), whole.spans.as_ptr()),
        (text_ptr, spans_ptr)
    );
    let mut empty = AttributedText::default();
    empty.push(whole);
    assert_eq!(
        (empty.text.as_ptr(), empty.spans.as_ptr()),
        (text_ptr, spans_ptr)
    );
    let head = empty.take(1);
    assert_eq!(head.text, "a");
    assert_eq!(empty.text, "é ");
    assert_eq!(
        empty.spans[0],
        TokenSpan {
            token: 0,
            source: 1..2,
            text: 0..2
        }
    );
}

#[test]
fn invalid_byte_groups_preserve_the_exact_native_replacement_policy() {
    for fallback in [false, true] {
        let mut decoder = TokenizerDecoder::new(tokenizer(fallback, &[])).unwrap();
        let mut text = decoder.push(&[b'a' as u32, 0xF0, 0x9F], 0).unwrap();
        text.push(decoder.finish().unwrap());
        assert_eq!(text.text, if fallback { "���" } else { "a�" });
        assert_eq!(
            text.spans.iter().map(|s| s.token).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(text.spans.iter().all(|s| s.source == (0..1)));
    }
}
