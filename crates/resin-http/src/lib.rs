mod anthropic;
mod metrics;
mod metrics_encoder;
mod openai;
mod protocol;
mod ui;

use std::{
    future::{Future, IntoFuture},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::{HeaderMap, StatusCode},
    routing::{get, post},
};
use resin_core::logprobs::DEFAULT_MAX_TOP_LOGPROBS;
use tokio::sync::{mpsc, oneshot};

pub use metrics::{
    MetricsEndpoint, MetricsPublishError, decode_metrics, encode_metrics, metrics_text,
};
pub use metrics_encoder::{MetricsEncodeError, MetricsEncoder};
pub use protocol::{
    ConversationRequest, FinishReason, GeneratedTokenLogprob, GenerationEvent, GenerationJob,
    TokenLogprob,
};

pub const EVENT_QUEUE_CAPACITY: usize = 64;

/// Schedule counting without blocking the HTTP handler, then reply through the sender.
pub type TokenCounter =
    Arc<dyn Fn(ConversationRequest, oneshot::Sender<Result<usize, String>>) + Send + Sync>;

/// Static configuration for HTTP route handlers.
#[derive(Clone, Copy, Debug)]
pub struct RouteConfig {
    /// Enable OpenAI API endpoints (/v1/responses, /v1/chat/completions).
    pub enable_openai: bool,

    /// Enable Anthropic API endpoints (/v1/messages).
    pub enable_anthropic: bool,

    /// Enable the demo web UI at the root path (/).
    pub enable_demo: bool,

    /// Allow passing "priority" in the request to control scheduling order.
    pub enable_priority_scheduling: bool,

    /// Number of attention data-parallel replicas.
    pub routed_dp_size: usize,

    /// The hosting engine's shared limit on token-score alternatives.
    pub max_top_logprobs: u32,
}

impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            enable_openai: true,
            enable_anthropic: true,
            enable_demo: true,
            enable_priority_scheduling: false,
            routed_dp_size: 1,
            max_top_logprobs: DEFAULT_MAX_TOP_LOGPROBS,
        }
    }
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<GenerationJob>,
    token_counter: Option<TokenCounter>,
    model: Arc<str>,
    next_id: Arc<AtomicU64>,
    routed_dp_size: usize,
    enable_priority_scheduling: bool,
    max_top_logprobs: u32,
}

impl AppState {
    fn request_id(&self, prefix: &str) -> String {
        let sequence = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}_{sequence:016x}")
    }

    fn enqueue(
        &self,
        request: ConversationRequest,
    ) -> Result<mpsc::Receiver<GenerationEvent>, (StatusCode, &'static str)> {
        let (events, receiver) = mpsc::channel(EVENT_QUEUE_CAPACITY);
        match self.jobs.try_send(GenerationJob {
            request,
            events,
            enqueued_at: Instant::now(),
        }) {
            Ok(()) => Ok(receiver),
            Err(mpsc::error::TrySendError::Full(_)) => Err((
                StatusCode::TOO_MANY_REQUESTS,
                "the inference request queue is full; retry later",
            )),
            Err(mpsc::error::TrySendError::Closed(_)) => Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "the inference engine is not running",
            )),
        }
    }
}

pub fn router(
    model: impl Into<Arc<str>>,
    jobs: mpsc::Sender<GenerationJob>,
    token_counter: Option<TokenCounter>,
    routes: RouteConfig,
    metrics: Option<MetricsEndpoint>,
) -> Router {
    let state = AppState {
        jobs,
        token_counter,
        model: model.into(),
        next_id: Arc::new(AtomicU64::new(1)),
        routed_dp_size: routes.routed_dp_size,
        enable_priority_scheduling: routes.enable_priority_scheduling,
        max_top_logprobs: routes.max_top_logprobs,
    };
    let mut router = Router::new().route("/health", get(openai::health));
    if let Some(metrics) = metrics {
        router = router.route(
            "/metrics",
            get(move |headers: HeaderMap| {
                let metrics = metrics.clone();
                async move { metrics.response(&headers) }
            }),
        );
    }
    if routes.enable_demo {
        router = router.route("/", get(openai::index));
    }
    if routes.enable_openai {
        router = router
            .route("/v1/models", get(openai::models))
            .route("/v1/chat/completions", post(openai::chat_completions))
            .route("/v1/responses", post(openai::responses))
            .route("/v1/responses/input_tokens", post(openai::input_tokens));
    }
    if routes.enable_anthropic {
        router = router
            .route("/v1/messages", post(anthropic::messages))
            .route("/v1/messages/count_tokens", post(anthropic::count_tokens));
    }
    router
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}

/// Stop accepting connections when requested and bound the time spent draining
/// existing HTTP connections. The hosting runtime may then be dropped even if
/// a client has stopped reading its response.
pub async fn serve_router_with_shutdown(
    listener: tokio::net::TcpListener,
    app: Router,
    shutdown: impl Future<Output = ()>,
) -> std::io::Result<()> {
    let (stop, stopped) = oneshot::channel::<()>();
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = stopped.await;
        })
        .into_future();
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result,
        () = shutdown => {
            let _ = stop.send(());
            tokio::time::timeout(Duration::from_secs(5), &mut server)
                .await
                .unwrap_or(Ok(()))
        }
    }
}
