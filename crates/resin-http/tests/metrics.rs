use axum::{
    body::Body,
    http::{Request, StatusCode, header::CONTENT_TYPE},
};
use http_body_util::BodyExt;
use resin_http::{MetricsEndpoint, RouteConfig, router};
use tokio::sync::mpsc;
use tower::ServiceExt;

#[tokio::test]
async fn scrape_uses_published_snapshot_without_a_running_engine() {
    let (jobs, receiver) = mpsc::channel(1);
    drop(receiver);
    let endpoint = MetricsEndpoint::default();
    let app = router("test", jobs, RouteConfig::default(), Some(endpoint.clone()));
    endpoint
        .publish(vec![
            "# TYPE resin:num_requests_total counter\nresin:num_requests_total{rank=\"0\"} 7\n"
                .into(),
        ])
        .unwrap();
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
        "# TYPE resin:num_requests_total counter\nresin:num_requests_total{rank=\"0\"} 7\n"
    );

    endpoint
        .publish(vec!["resin:num_requests_total{rank=\"0\"} 9\n".into()])
        .unwrap();
    let response = app
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(body, "resin:num_requests_total{rank=\"0\"} 9\n");
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
