use axum::{
    body::Body,
    http::{
        Request, StatusCode,
        header::{ACCEPT, CONTENT_TYPE, VARY},
    },
};
use http_body_util::BodyExt;
use resin_http::{MetricsEndpoint, RouteConfig, decode_metrics, router};
use tokio::sync::mpsc;
use tower::ServiceExt;

const PROTO_CONTENT_TYPE: &str =
    "application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=delimited";

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
