use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use resin_http::{
    FinishReason, GeneratedTokenLogprob, GenerationEvent, GenerationJob, RouteConfig, TokenLogprob,
    router,
};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt;

fn test_router() -> axum::Router {
    let (jobs, mut receiver) = mpsc::channel::<GenerationJob>(32);
    tokio::spawn(async move {
        while let Some(job) = receiver.recv().await {
            let _ = job
                .events
                .try_send(GenerationEvent::Started { input_tokens: 3 });
            let _ = job.events.try_send(GenerationEvent::FirstToken);
            let _ = job
                .events
                .try_send(GenerationEvent::ReasoningDelta("brief thought".to_owned()));
            let _ = job.events.try_send(GenerationEvent::TextDelta {
                delta: "hello".to_owned(),
                logprobs: vec![],
            });
            let _ = job.events.try_send(GenerationEvent::Finished {
                reason: FinishReason::Stop,
                input_tokens: 3,
                cached_tokens: 2,
                output_tokens: 2,
                elapsed_ms: 20.0,
                generation_elapsed_ms: 10.0,
                time_to_first_token_ms: 10.0,
            });
        }
    });
    router("test-model", jobs, RouteConfig::default(), None)
}

fn tool_router() -> axum::Router {
    let (jobs, mut receiver) = mpsc::channel::<GenerationJob>(32);
    tokio::spawn(async move {
        while let Some(job) = receiver.recv().await {
            let _ = job
                .events
                .try_send(GenerationEvent::Started { input_tokens: 7 });
            let _ = job.events.try_send(GenerationEvent::FirstToken);
            let _ = job
                .events
                .try_send(GenerationEvent::ReasoningDelta("check weather".to_owned()));
            let _ = job.events.try_send(GenerationEvent::ToolCallStart {
                index: 0,
                id: "call_weather".to_owned(),
                name: "weather".to_owned(),
            });
            let _ = job
                .events
                .try_send(GenerationEvent::ToolCallArgumentsDelta {
                    index: 0,
                    delta: r#"{"city":"Paris"}"#.to_owned(),
                });
            let _ = job.events.try_send(GenerationEvent::Finished {
                reason: FinishReason::Stop,
                input_tokens: 7,
                cached_tokens: 4,
                output_tokens: 4,
                elapsed_ms: 30.0,
                generation_elapsed_ms: 20.0,
                time_to_first_token_ms: 10.0,
            });
        }
    });
    router("test-model", jobs, RouteConfig::default(), None)
}

async fn json_response(app: axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn scored_token(index: usize, text: &str, channel: Option<&str>) -> GeneratedTokenLogprob {
    GeneratedTokenLogprob {
        token: TokenLogprob {
            token_id: index as u32,
            token: text.to_owned(),
            bytes: Some(text.as_bytes().to_owned()),
            logprob: -0.25,
        },
        token_index: index,
        channel: channel.map(str::to_owned),
        top_logprobs: vec![],
    }
}

fn scored_router() -> axum::Router {
    // Raw events can precede/follow visible text; only the text event identifies
    // which scores belong to that delta. Raw channel hints are not sufficient.
    let first = scored_token(1, "α", None);
    let last = scored_token(3, "β", Some("content"));
    let events = vec![
        GenerationEvent::Started { input_tokens: 3 },
        GenerationEvent::Logprobs(vec![scored_token(0, "hidden", Some("content"))]),
        GenerationEvent::TextDelta {
            delta: "α".to_owned(),
            logprobs: vec![first.clone()],
        },
        GenerationEvent::Logprobs(vec![first, scored_token(2, "!", None)]),
        GenerationEvent::TextDelta {
            delta: "!".to_owned(),
            logprobs: vec![],
        },
        GenerationEvent::TextDelta {
            delta: "β".to_owned(),
            logprobs: vec![last.clone()],
        },
        GenerationEvent::Logprobs(vec![last]),
        GenerationEvent::Finished {
            reason: FinishReason::Stop,
            input_tokens: 3,
            cached_tokens: 0,
            output_tokens: 4,
            elapsed_ms: 20.0,
            generation_elapsed_ms: 10.0,
            time_to_first_token_ms: 10.0,
        },
    ];
    let (jobs, mut receiver) = mpsc::channel::<GenerationJob>(32);
    tokio::spawn(async move {
        while let Some(job) = receiver.recv().await {
            for event in &events {
                if job.events.send(event.clone()).await.is_err() {
                    break;
                }
            }
        }
    });
    router("test-model", jobs, RouteConfig::default(), None)
}

fn scored_request(path: &str, stream: bool) -> Request<Body> {
    let mut body = json!({"model": "test-model", "stream": stream, "logprobs": true});
    if path == "/v1/responses" {
        body["input"] = json!("hi");
    } else {
        body["messages"] = json!([{"role": "user", "content": "hi"}]);
        body["max_tokens"] = json!(16);
    }
    Request::post(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn stream_events(app: axum::Router, request: Request<Body>) -> Vec<Value> {
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    std::str::from_utf8(&body)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[test]
fn text_score_transport_defaults_empty_and_preserves_attached_scores() {
    let value = json!({"type": "text_delta", "data": {"delta": "plain"}});
    let event: GenerationEvent = serde_json::from_value(value.clone()).unwrap();
    assert!(matches!(&event, GenerationEvent::TextDelta { logprobs, .. } if logprobs.is_empty()));
    assert_eq!(serde_json::to_value(event).unwrap(), value);

    let mut token = scored_token(1, "α", None);
    token.token.logprob = f32::NEG_INFINITY;
    let event = GenerationEvent::TextDelta {
        delta: "α".to_owned(),
        logprobs: vec![token],
    };
    let restored: GenerationEvent =
        serde_json::from_str(&serde_json::to_string(&event).unwrap()).unwrap();
    let GenerationEvent::TextDelta { delta, logprobs } = restored else {
        panic!("wrong event")
    };
    assert_eq!(delta, "α");
    assert_eq!(logprobs[0].token.logprob, f32::NEG_INFINITY);
    assert_eq!(logprobs[0].token.bytes, Some("α".as_bytes().to_vec()));
}

#[tokio::test]
async fn chat_stream_attaches_canonical_scores_to_the_matching_text_chunk() {
    let events = stream_events(
        scored_router(),
        scored_request("/v1/chat/completions", true),
    )
    .await;
    let mut visible = Vec::new();
    let mut raw = Vec::new();
    for event in &events {
        let choice = &event["choices"][0];
        if let Some(delta) = choice["delta"]["content"]
            .as_str()
            .filter(|s| !s.is_empty())
        {
            let scores = choice["logprobs"]["content"].as_array().unwrap();
            let expected: &[u64] = match delta {
                "α" => &[1],
                "!" => &[],
                "β" => &[3],
                _ => panic!("unexpected text delta"),
            };
            assert_eq!(
                scores
                    .iter()
                    .map(|s| s["token_index"].as_u64().unwrap())
                    .collect::<Vec<_>>(),
                expected
            );
            visible.extend_from_slice(scores);
        }
        if let Some(scores) = event["resin"]["generation_logprobs"].as_array() {
            assert_eq!(choice["delta"], json!({}));
            assert!(choice["logprobs"].is_null());
            raw.extend_from_slice(scores);
        }
    }
    assert_eq!(
        raw.iter()
            .map(|s| s["token_index"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    let (_, full) = json_response(
        scored_router(),
        scored_request("/v1/chat/completions", false),
    )
    .await;
    assert_eq!(full["choices"][0]["message"]["content"], "α!β");
    assert_eq!(full["choices"][0]["logprobs"]["content"], json!(visible));
    assert_eq!(full["resin"]["generation_logprobs"], json!(raw));
}

#[tokio::test]
async fn responses_live_and_final_content_scores_have_the_same_provenance() {
    let events = stream_events(scored_router(), scored_request("/v1/responses", true)).await;
    let deltas = events
        .iter()
        .filter(|event| event["type"] == "response.output_text.delta")
        .collect::<Vec<_>>();
    assert_eq!(deltas.len(), 3);
    assert_eq!(deltas[0]["delta"], "α");
    assert_eq!(deltas[0]["logprobs"][0]["token_index"], 1);
    assert_eq!(deltas[1]["delta"], "!");
    assert_eq!(deltas[1]["logprobs"], json!([]));
    assert_eq!(deltas[2]["delta"], "β");
    assert_eq!(deltas[2]["logprobs"][0]["token_index"], 3);
    let visible = deltas
        .iter()
        .flat_map(|event| event["logprobs"].as_array().unwrap().iter().cloned())
        .collect::<Vec<_>>();
    let raw = events
        .iter()
        .filter(|event| event["type"] == "resin.generation_logprobs.delta")
        .flat_map(|event| event["logprobs"].as_array().unwrap().iter().cloned())
        .collect::<Vec<_>>();
    assert_eq!(raw.len(), 4);
    for event in &events {
        match event["type"].as_str().unwrap() {
            "response.output_text.done" => assert_eq!(event["logprobs"], json!(visible)),
            "response.content_part.done" => assert_eq!(event["part"]["logprobs"], json!(visible)),
            "response.output_item.done" => {
                assert_eq!(event["item"]["content"][0]["logprobs"], json!(visible))
            }
            "response.completed" => {
                assert_eq!(
                    event["response"]["output"][0]["content"][0]["logprobs"],
                    json!(visible)
                );
                assert_eq!(
                    event["response"]["resin"]["generation_logprobs"],
                    json!(raw)
                );
            }
            _ => {}
        }
    }
    let (_, full) = json_response(scored_router(), scored_request("/v1/responses", false)).await;
    assert_eq!(full["output"][0]["content"][0]["text"], "α!β");
    assert_eq!(full["output"][0]["content"][0]["logprobs"], json!(visible));
    assert_eq!(full["resin"]["generation_logprobs"], json!(raw));
}

#[tokio::test]
async fn anthropic_preserves_text_and_ignores_attached_and_raw_scores() {
    let events = stream_events(scored_router(), scored_request("/v1/messages", true)).await;
    let text = events
        .iter()
        .filter_map(|event| event["delta"]["text"].as_str())
        .collect::<String>();
    assert_eq!(text, "α!β");
    assert!(
        events
            .iter()
            .all(|event| !event.to_string().contains("logprobs"))
    );
    let (_, full) = json_response(scored_router(), scored_request("/v1/messages", false)).await;
    assert_eq!(full["content"], json!([{"type": "text", "text": "α!β"}]));
}

#[test]
fn coordinator_score_transport_preserves_valid_negative_infinity() {
    let event = GenerationEvent::Logprobs(vec![resin_http::GeneratedTokenLogprob {
        token: resin_http::TokenLogprob {
            token_id: 42,
            token: "x".to_owned(),
            bytes: Some(vec![b'x']),
            logprob: f32::NEG_INFINITY,
        },
        token_index: 0,
        channel: Some("content".to_owned()),
        top_logprobs: vec![],
    }]);
    let wire = serde_json::to_string(&vec![("global-id", event)]).unwrap();
    let restored: Vec<(String, GenerationEvent)> = serde_json::from_str(&wire).unwrap();
    assert_eq!(restored[0].0, "global-id");
    let GenerationEvent::Logprobs(tokens) = &restored[0].1 else {
        panic!("wrong event")
    };
    assert_eq!(tokens[0].token.logprob, f32::NEG_INFINITY);
    assert_eq!(tokens[0].token.bytes, Some(vec![b'x']));
}

fn recording_router(ranks: usize) -> axum::Router {
    let (jobs, mut receiver) = mpsc::channel::<GenerationJob>(32);
    tokio::spawn(async move {
        while let Some(job) = receiver.recv().await {
            let request = serde_json::to_string(&job.request).unwrap();
            let _ = job
                .events
                .send(GenerationEvent::TextDelta {
                    delta: request,
                    logprobs: vec![],
                })
                .await;
            let _ = job
                .events
                .send(GenerationEvent::Finished {
                    reason: FinishReason::Stop,
                    input_tokens: 1,
                    cached_tokens: 0,
                    output_tokens: 1,
                    elapsed_ms: 1.0,
                    generation_elapsed_ms: 1.0,
                    time_to_first_token_ms: 0.0,
                })
                .await;
        }
    });
    router(
        "test-model",
        jobs,
        RouteConfig {
            routed_dp_size: ranks,
            ..RouteConfig::default()
        },
        None,
    )
}

#[tokio::test]
async fn routing_metadata_uses_only_explicit_body_or_resin_header() {
    for (path, body) in [
        (
            "/v1/chat/completions",
            json!({"messages": [{"role":"user","content":"hi"}]}),
        ),
        ("/v1/responses", json!({"input":"hi"})),
        (
            "/v1/messages",
            json!({"messages": [{"role":"user","content":"hi"}], "max_tokens":1}),
        ),
    ] {
        for (body_rank, header_rank, expected) in [
            (Value::Null, None, Value::Null),
            (json!(2), None, json!(2)),
            (json!(2), Some("1"), json!(1)),
            (
                json!("invalid body overridden by header"),
                Some("0"),
                json!(0),
            ),
        ] {
            let mut body = body.clone();
            body["model"] = json!("test-model");
            body["routed_dp_rank"] = body_rank;
            let mut request = Request::post(path)
                .header(header::CONTENT_TYPE, "application/json")
                .header("X-Data-Parallel-Rank", "3")
                .header("routed-dp-rank", "3")
                .header("data-parallel-rank", "3");
            if let Some(rank) = header_rank {
                request = request.header("resin-routed-dp-rank", rank);
            }
            let (status, response) = json_response(
                recording_router(4),
                request.body(Body::from(body.to_string())).unwrap(),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{path}: {response}");
            let content = match path {
                "/v1/chat/completions" => &response["choices"][0]["message"]["content"],
                "/v1/responses" => &response["output"][0]["content"][0]["text"],
                _ => &response["content"][0]["text"],
            };
            let normalized: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
            assert_eq!(normalized["routed_dp_rank"], expected);
        }
        for rank in [json!(-1), json!(4), json!(1.5), json!(true), json!("1")] {
            let mut body = body.clone();
            body["model"] = json!("test-model");
            body["routed_dp_rank"] = rank;
            let (status, _) = json_response(
                recording_router(4),
                Request::post(path)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
        }
        for rank in ["-1", "4", "1.5", "banana", ""] {
            let mut body = body.clone();
            body["model"] = json!("test-model");
            let (status, _) = json_response(
                recording_router(4),
                Request::post(path)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("resin-routed-dp-rank", rank)
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {rank}");
        }
    }
}

#[tokio::test]
async fn score_options_are_explicit_normalized_request_fields() {
    for (path, body) in [
        (
            "/v1/chat/completions",
            json!({"messages": [{"role":"user","content":"hi"}], "logprobs":true}),
        ),
        (
            "/v1/responses",
            json!({"input":"hi", "include":["message.output_text.logprobs"]}),
        ),
    ] {
        let mut body = body;
        body["model"] = json!("test-model");
        body["top_logprobs"] = json!(3);
        body["prompt_logprobs"] = json!(true);
        let (status, response) = json_response(
            recording_router(1),
            Request::post(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let content = if path.ends_with("completions") {
            &response["choices"][0]["message"]["content"]
        } else {
            &response["output"][0]["content"][0]["text"]
        };
        let normalized: Value = serde_json::from_str(content.as_str().unwrap()).unwrap();
        assert_eq!(normalized["logprobs"], true);
        assert_eq!(normalized["top_logprobs"], 3);
        assert_eq!(normalized["prompt_logprobs"], true);
        body["top_logprobs"] = json!(21);
        let (status, _) = json_response(
            recording_router(1),
            Request::post(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn chat_completions_returns_reasoning_text_and_usage() {
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            })
            .to_string(),
        ))
        .unwrap();

    let (status, body) = json_response(test_router(), request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "hello");
    assert!(body["choices"][0]["logprobs"].is_null());
    assert!(body["resin"].get("generation_logprobs").is_none());
    assert_eq!(
        body["choices"][0]["message"]["reasoning_content"],
        "brief thought"
    );
    assert_eq!(body["usage"]["total_tokens"], 5);
    assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 2);
}

#[tokio::test]
async fn responses_api_returns_reasoning_and_message_items() {
    let request = Request::post("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "test-model", "input": "hi"}).to_string(),
        ))
        .unwrap();

    let (status, body) = json_response(test_router(), request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "response");
    assert_eq!(body["output"][0]["type"], "reasoning");
    assert_eq!(body["output"][1]["type"], "message");
    assert_eq!(body["output"][1]["content"][0]["text"], "hello");
    assert_eq!(body["output"][1]["content"][0]["logprobs"], json!([]));
    assert!(body["resin"].get("generation_logprobs").is_none());
    assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 2);
}

#[tokio::test]
async fn anthropic_messages_returns_thinking_and_text_blocks() {
    let request = Request::post("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })
            .to_string(),
        ))
        .unwrap();

    let (status, body) = json_response(test_router(), request).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["type"], "thinking");
    assert_eq!(body["content"][1], json!({"type": "text", "text": "hello"}));
    assert_eq!(body["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn route_flags_remove_disabled_apis_and_demo() {
    let (jobs, _receiver) = mpsc::channel(32);
    let app = router(
        "test-model",
        jobs,
        RouteConfig {
            enable_openai: false,
            enable_anthropic: false,
            enable_demo: false,
            ..RouteConfig::default()
        },
        None,
    );
    for path in ["/", "/v1/models", "/v1/chat/completions", "/v1/messages"] {
        let response = app
            .clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
    }
    let response = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_completions_streams_tool_calls_and_resin_metrics() {
    let request = Request::post("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "weather?"}],
                "stream": true,
                "stream_options": {"include_usage": true},
            })
            .to_string(),
        ))
        .unwrap();

    let response = tool_router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let events = std::str::from_utf8(&body).unwrap();
    assert!(events.contains(r#""reasoning_content":"check weather""#));
    assert!(events.contains(r#""name":"weather""#));
    assert!(events.contains(r#""finish_reason":"tool_calls""#));
    assert!(events.contains(r#""generation_elapsed_ms":20.0"#));
    assert!(events.contains(r#""cached_tokens":4"#));
    assert!(events.contains(r#""tokens_per_second":200.0"#));
    assert!(events.contains("data: [DONE]"));
}

#[tokio::test]
async fn tool_calls_map_to_responses_and_anthropic_shapes() {
    let responses_request = Request::post("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "test-model", "input": "weather?"}).to_string(),
        ))
        .unwrap();
    let (status, response) = json_response(tool_router(), responses_request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["output"][1]["type"], "function_call");
    assert_eq!(response["output"][1]["name"], "weather");
    assert_eq!(response["output"][1]["arguments"], r#"{"city":"Paris"}"#);

    let messages_request = Request::post("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "weather?"}],
            })
            .to_string(),
        ))
        .unwrap();
    let (status, message) = json_response(tool_router(), messages_request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(message["content"][1]["type"], "tool_use");
    assert_eq!(message["content"][1]["input"], json!({"city": "Paris"}));
    assert_eq!(message["stop_reason"], "tool_use");
}

#[tokio::test]
async fn responses_and_anthropic_streams_emit_protocol_events() {
    let responses_request = Request::post("/v1/responses")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": "test-model", "input": "hi", "stream": true}).to_string(),
        ))
        .unwrap();
    let response = test_router().oneshot(responses_request).await.unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let events = std::str::from_utf8(&body).unwrap();
    assert!(events.contains(r#""type":"response.reasoning_summary_text.delta""#));
    assert!(events.contains(r#""type":"response.output_text.delta""#));
    assert!(events.contains(r#""type":"response.completed""#));

    let messages_request = Request::post("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            })
            .to_string(),
        ))
        .unwrap();
    let response = test_router().oneshot(messages_request).await.unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let events = std::str::from_utf8(&body).unwrap();
    assert!(events.contains("event: message_start"));
    assert!(events.contains(r#""type":"thinking_delta""#));
    assert!(events.contains(r#""type":"text_delta""#));
    assert!(events.contains("event: message_stop"));
}

fn generation_request(path: &str, stream: bool) -> Request<Body> {
    let body = if path == "/v1/responses" {
        json!({"model": "test-model", "input": "hi", "stream": stream})
    } else {
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": stream,
        })
    };
    Request::post(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn full_admission_queue_rejects_each_api_without_waiting() {
    for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages"] {
        let (jobs, receiver) = mpsc::channel(1);
        let app = router("test-model", jobs, RouteConfig::default(), None);
        let first = app
            .clone()
            .oneshot(generation_request(path, true))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let (status, body) = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            json_response(app, generation_request(path, false)),
        )
        .await
        .expect("overload must be rejected immediately");
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("queue is full")
        );
        assert_eq!(receiver.len(), 1);
    }
}

#[tokio::test]
async fn stopped_actor_fails_health_and_generation_admission() {
    let (jobs, receiver) = mpsc::channel(1);
    let app = router("test-model", jobs, RouteConfig::default(), None);
    drop(receiver);
    let (status, _) = json_response(
        app.clone(),
        Request::get("/health").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let (status, _) = json_response(app, generation_request("/v1/responses", false)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn slow_stream_has_a_bounded_queue_and_does_not_block_another_response() {
    let (jobs, mut receiver) = mpsc::channel(2);
    let app = router("test-model", jobs, RouteConfig::default(), None);
    let slow = app
        .clone()
        .oneshot(generation_request("/v1/chat/completions", true))
        .await
        .unwrap();
    let slow_job = receiver.recv().await.unwrap();
    for _ in 0..resin_http::EVENT_QUEUE_CAPACITY {
        slow_job
            .events
            .try_send(GenerationEvent::TextDelta {
                delta: "x".to_owned(),
                logprobs: vec![],
            })
            .unwrap();
    }
    assert!(matches!(
        slow_job.events.try_send(GenerationEvent::TextDelta {
            delta: "overflow".to_owned(),
            logprobs: vec![]
        }),
        Err(mpsc::error::TrySendError::Full(_)),
    ));
    drop(slow_job);

    let fast = tokio::spawn(json_response(
        app,
        generation_request("/v1/responses", false),
    ));
    let fast_job = receiver.recv().await.unwrap();
    fast_job
        .events
        .try_send(GenerationEvent::TextDelta {
            delta: "finished".to_owned(),
            logprobs: vec![],
        })
        .unwrap();
    fast_job
        .events
        .try_send(GenerationEvent::Finished {
            reason: FinishReason::Stop,
            input_tokens: 1,
            cached_tokens: 0,
            output_tokens: 1,
            elapsed_ms: 1.0,
            generation_elapsed_ms: 1.0,
            time_to_first_token_ms: 0.0,
        })
        .unwrap();
    let (status, _) = tokio::time::timeout(std::time::Duration::from_secs(1), fast)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status, StatusCode::OK);

    let bytes = slow.into_body().collect().await.unwrap().to_bytes();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("ended before completion"));
    assert!(!text.contains("overflow"));
}

#[tokio::test]
async fn failed_requests_never_return_partial_success() {
    for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages"] {
        for streaming in [false, true] {
            for explicit_error in [false, true] {
                let (jobs, mut receiver) = mpsc::channel(1);
                let app = router("test-model", jobs, RouteConfig::default(), None);
                let pending = tokio::spawn(app.oneshot(generation_request(path, streaming)));
                let job = receiver.recv().await.unwrap();
                job.events
                    .try_send(GenerationEvent::TextDelta {
                        delta: "partial".to_owned(),
                        logprobs: vec![],
                    })
                    .unwrap();
                if explicit_error {
                    job.events
                        .try_send(GenerationEvent::Error {
                            message: "model execution failed".to_owned(),
                        })
                        .unwrap();
                }
                drop(job);
                let response = pending.await.unwrap().unwrap();
                assert_eq!(
                    response.status(),
                    if streaming {
                        StatusCode::OK
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                );
                let bytes = response.into_body().collect().await.unwrap().to_bytes();
                let text = std::str::from_utf8(&bytes).unwrap();
                assert!(
                    text.contains(if explicit_error {
                        "model execution failed"
                    } else if streaming {
                        "ended before completion"
                    } else {
                        "ended the request without a result"
                    }),
                    "{path}"
                );
                if !streaming {
                    let body: Value = serde_json::from_slice(&bytes).unwrap();
                    assert_eq!(
                        body["error"]["type"],
                        if path == "/v1/messages" {
                            "api_error"
                        } else {
                            "server_error"
                        }
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn programmatic_shutdown_closes_the_listener() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (jobs, _receiver) = mpsc::channel(1);
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(resin_http::serve_with_shutdown(
        listener,
        "test-model",
        jobs,
        RouteConfig::default(),
        None,
        async {
            let _ = stopped.await;
        },
    ));
    stop.send(()).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(tokio::net::TcpStream::connect(address).await.is_err());
}
