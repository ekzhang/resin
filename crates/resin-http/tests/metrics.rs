use axum::{
    body::Body,
    http::{
        Request, StatusCode,
        header::{ACCEPT, CONTENT_TYPE, VARY},
    },
};
use http_body_util::BodyExt;
use prometheus_client::encoding::prometheus_protobuf::prometheus_data_model::{
    Counter, LabelPair, Metric, MetricFamily, MetricType,
};
use resin_http::{MetricsEndpoint, RouteConfig, decode_metrics, encode_metrics, router};
use tokio::sync::mpsc;
use tower::ServiceExt;

const PROTO_CONTENT_TYPE: &str =
    "application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=delimited";

fn requests(count: f64) -> Vec<u8> {
    encode_metrics(&[MetricFamily {
        name: "resin:num_requests_total".into(),
        help: "Completed requests.".into(),
        r#type: MetricType::Counter as i32,
        metric: vec![Metric {
            label: vec![LabelPair {
                name: "rank".into(),
                value: "0".into(),
            }],
            counter: Some(Counter {
                value: count,
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }])
}

#[tokio::test]
async fn scrape_uses_published_snapshot_without_a_running_engine() {
    let (jobs, receiver) = mpsc::channel(1);
    drop(receiver);
    let endpoint = MetricsEndpoint::default();
    let app = router("test", jobs, RouteConfig::default(), Some(endpoint.clone()));
    endpoint.publish(vec![requests(7.0)]).unwrap();
    let response = app
        .clone()
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[CONTENT_TYPE],
        "text/plain; version=0.0.4; charset=utf-8"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        body,
        "# HELP resin:num_requests_total Completed requests.\n# TYPE resin:num_requests_total counter\nresin:num_requests_total{rank=\"0\"} 7.0\n"
    );

    let publication = requests(9.0);
    endpoint.publish(vec![publication.clone()]).unwrap();
    let response = app
        .oneshot(
            Request::get("/metrics")
                .header(ACCEPT, PROTO_CONTENT_TYPE)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.headers()[CONTENT_TYPE], PROTO_CONTENT_TYPE);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body, publication);
}

#[tokio::test]
async fn scrape_negotiates_native_histograms_across_accept_headers() {
    let (jobs, receiver) = mpsc::channel(1);
    drop(receiver);
    let endpoint = MetricsEndpoint::new("test");
    endpoint.observe_completed(0.1);
    endpoint.observe_completed(3.0);
    endpoint.publish(vec![]).unwrap();
    let app = router("test", jobs, RouteConfig::default(), Some(endpoint));
    let response = app
        .clone()
        .oneshot(
            Request::get("/metrics")
                .header(ACCEPT, "text/plain;q=0")
                .header(ACCEPT, format!("{PROTO_CONTENT_TYPE};q=0.9,*/*;q=0.1"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[CONTENT_TYPE], PROTO_CONTENT_TYPE);
    assert_eq!(response.headers()[VARY], "Accept");
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let families = decode_metrics(&body).unwrap();
    let histogram = families
        .iter()
        .find(|family| family.name == "resin:http_request_latency_seconds")
        .unwrap()
        .metric[0]
        .histogram
        .as_ref()
        .unwrap();
    assert_eq!(histogram.sample_count, 2);
    assert_eq!(histogram.sample_sum, 3.1);
    assert!(!histogram.positive_span.is_empty());
    assert!(histogram.bucket.is_empty());

    let response = app
        .oneshot(
            Request::get("/metrics")
                .header(ACCEPT, "text/plain;q=0")
                .header(ACCEPT, "application/vnd.google.protobuf;q=0,*/*;q=1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_ACCEPTABLE);
}

#[tokio::test]
async fn metrics_route_is_optional_and_independent_of_inference_routes() {
    for enabled in [false, true] {
        let (jobs, _receiver) = mpsc::channel(1);
        let app = router(
            "test",
            jobs,
            RouteConfig {
                enable_openai: false,
                enable_anthropic: false,
                enable_demo: false,
                ..RouteConfig::default()
            },
            enabled.then(MetricsEndpoint::default),
        );
        let response = app
            .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            if enabled {
                StatusCode::OK
            } else {
                StatusCode::NOT_FOUND
            }
        );
        if enabled {
            assert!(
                response
                    .into_body()
                    .collect()
                    .await
                    .unwrap()
                    .to_bytes()
                    .is_empty()
            );
        }
    }
}
