use std::{
    net::{IpAddr, SocketAddr},
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use resin_http::{GenerationJob, MetricsEndpoint, RouteConfig};
use tokio::sync::{mpsc, oneshot, watch};

use crate::Engine;
use crate::http_session::Session;

pub(crate) const MAX_ACTIVE_REQUESTS: usize = 128;
pub(crate) const ADMISSIONS_PER_STEP: usize = 4;

#[pyclass(name = "_HttpServer", module = "resin._resin")]
pub(crate) struct NativeHttpServer {
    engine: Py<Engine>,
    distributed: Option<Py<PyAny>>,
    recipe: Py<PyAny>,
    model: String,
    stream_interval: u32,
    request_queue_capacity: u32,
    routes: RouteConfig,
    running: AtomicBool,
    shutdown: Mutex<Option<watch::Sender<bool>>>,
}

#[pymethods]
impl NativeHttpServer {
    #[new]
    #[pyo3(signature = (
        engine,
        recipe,
        *,
        model,
        stream_interval = 1,
        request_queue_capacity = 1024,
        enable_openai = true,
        enable_anthropic = true,
        enable_demo = true,
        distributed = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        engine: Py<Engine>,
        recipe: Py<PyAny>,
        model: String,
        stream_interval: u32,
        request_queue_capacity: u32,
        enable_openai: bool,
        enable_anthropic: bool,
        enable_demo: bool,
        distributed: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if stream_interval == 0 {
            return Err(PyValueError::new_err("stream_interval must be positive"));
        }
        if request_queue_capacity == 0 {
            return Err(PyValueError::new_err(
                "request_queue_capacity must be positive",
            ));
        }
        let routed_dp_size = distributed
            .as_ref()
            .map(|coordinator| {
                Python::attach(|py| coordinator.bind(py).getattr("size")?.extract::<usize>())
            })
            .transpose()?
            .unwrap_or(1);
        if routed_dp_size == 0 {
            return Err(PyValueError::new_err("routed_dp_size must be positive"));
        }
        if model.trim().is_empty() {
            return Err(PyValueError::new_err("model must be nonempty"));
        }
        if enable_demo && !enable_openai {
            return Err(PyValueError::new_err(
                "enable_demo=True requires enable_openai=True",
            ));
        }
        Ok(Self {
            engine,
            distributed,
            recipe,
            model,
            stream_interval,
            request_queue_capacity,
            routes: RouteConfig {
                enable_openai,
                enable_anthropic,
                enable_demo,
                routed_dp_size,
            },
            running: AtomicBool::new(false),
            shutdown: Mutex::new(None),
        })
    }

    #[pyo3(signature = (*, host = "127.0.0.1", port = 8000))]
    fn serve(&self, py: Python<'_>, host: &str, port: u16) -> PyResult<()> {
        let host = host.parse::<IpAddr>().map_err(|_| {
            PyValueError::new_err("host must be an IPv4 or IPv6 address such as '0.0.0.0'")
        })?;
        let (shutdown, _) = watch::channel(false);
        {
            let mut slot = self
                .shutdown
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if self.running.swap(true, Ordering::AcqRel) {
                return Err(PyRuntimeError::new_err("HttpServer is already running"));
            }
            *slot = Some(shutdown.clone());
        }
        let engine = self.engine.clone_ref(py);
        let distributed = self.distributed.as_ref().map(|value| value.clone_ref(py));
        let recipe = self.recipe.clone_ref(py);
        let model = self.model.clone();
        let routes = self.routes;
        let stream_interval = self.stream_interval;
        let request_queue_capacity = self.request_queue_capacity;
        let result = py.detach(move || {
            run_server(
                engine,
                distributed,
                recipe,
                model,
                routes,
                stream_interval,
                request_queue_capacity,
                SocketAddr::new(host, port),
                shutdown,
            )
        });
        {
            let mut slot = self
                .shutdown
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            *slot = None;
            self.running.store(false, Ordering::Release);
        }
        result.map_err(PyRuntimeError::new_err)
    }

    /// Request shutdown from another thread; serve returns after the current
    /// inference step has finished and its requests have been cancelled.
    fn shutdown(&self) {
        let slot = self
            .shutdown
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(shutdown) = slot.as_ref() {
            shutdown.send_replace(true);
        }
    }

    #[getter]
    fn model(&self) -> &str {
        &self.model
    }

    #[getter]
    fn running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn __repr__(&self) -> String {
        format!(
            "_HttpServer(model={:?}, stream_interval={}, request_queue_capacity={}, enable_openai={}, enable_anthropic={}, enable_demo={}, running={})",
            self.model,
            self.stream_interval,
            self.request_queue_capacity,
            self.routes.enable_openai,
            self.routes.enable_anthropic,
            self.routes.enable_demo,
            self.running.load(Ordering::Acquire),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_server(
    engine: Py<Engine>,
    distributed: Option<Py<PyAny>>,
    recipe: Py<PyAny>,
    model: String,
    routes: RouteConfig,
    stream_interval: u32,
    request_queue_capacity: u32,
    address: SocketAddr,
    shutdown: watch::Sender<bool>,
) -> Result<(), String> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("failed to create the HTTP runtime: {error}"))?;
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind(address))
        .map_err(|error| format!("failed to bind {address}: {error}"))?;
    let (jobs, receiver) = mpsc::channel(request_queue_capacity as usize);
    let metrics = Python::attach(|py| engine.bind(py).borrow().metrics_enabled())
        .then(|| MetricsEndpoint::new(&model));
    let actor_metrics = metrics.clone();
    let actor_model = model.clone();
    let actor_shutdown = shutdown.subscribe();
    let (actor_finished, actor_done) = oneshot::channel();
    let actor = thread::Builder::new()
        .name("resin-http-engine".to_owned())
        .spawn(move || {
            let result = if let Some(coordinator) = distributed {
                crate::http_distributed::run(
                    coordinator,
                    receiver,
                    actor_shutdown,
                    routes.routed_dp_size,
                    actor_metrics,
                )
            } else {
                engine_actor(
                    engine,
                    recipe,
                    receiver,
                    actor_shutdown,
                    stream_interval,
                    actor_metrics,
                    &actor_model,
                )
            };
            if let Err(error) = &result {
                eprintln!("[resin-http] inference actor failed: {error}");
            }
            let _ = actor_finished.send(());
            result
        })
        .map_err(|error| format!("failed to start the inference thread: {error}"))?;
    let mut stopping = shutdown.subscribe();
    let stop_actor = shutdown.clone();
    let server_result = runtime.block_on(resin_http::serve_with_shutdown(
        listener,
        model,
        jobs,
        routes,
        metrics,
        async move {
            if !*stopping.borrow() {
                tokio::select! {
                    _ = actor_done => {},
                    _ = stopping.changed() => {},
                    _ = tokio::signal::ctrl_c() => {},
                }
            }
            stop_actor.send_replace(true);
        },
    ));
    shutdown.send_replace(true);
    let actor_result = actor
        .join()
        .map_err(|_| "the inference thread panicked".to_owned())?;
    server_result
        .map_err(|error| format!("HTTP server failed: {error}"))
        .and(actor_result)
}

fn engine_actor(
    engine: Py<Engine>,
    recipe: Py<PyAny>,
    mut jobs: mpsc::Receiver<GenerationJob>,
    shutdown: watch::Receiver<bool>,
    stream_interval: u32,
    metrics: Option<MetricsEndpoint>,
    model: &str,
) -> Result<(), String> {
    let mut session =
        Python::attach(|py| Session::new(py, engine.clone_ref(py), recipe, stream_interval))
            .map_err(|error| format_python_error("HTTP session preparation failed", error))?;
    session.metrics = metrics.clone();
    let mut next_metrics = Instant::now();
    let mut disconnected = false;
    loop {
        if let Some(endpoint) = &metrics {
            endpoint.set_queue_size(jobs.len());
            if Instant::now() >= next_metrics {
                let encoded = Python::attach(|py| {
                    engine
                        .bind(py)
                        .borrow()
                        .collect_metrics(py, true)?
                        .expect("enabled metrics")
                        .encode_protobuf(model, 0, 0)
                })
                .map_err(|error| format_python_error("metrics collection failed", error))?;
                endpoint
                    .publish(vec![encoded])
                    .map_err(|error| error.to_string())?;
                next_metrics = Instant::now() + Duration::from_secs(1);
            }
        }
        for _ in 0..ADMISSIONS_PER_STEP {
            if disconnected || *shutdown.borrow() || session.active_count() >= MAX_ACTIVE_REQUESTS {
                break;
            }
            match jobs.try_recv() {
                Ok(job) => session.admit(job),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected || *shutdown.borrow() {
            session
                .close()
                .map_err(|error| format_python_error("failed to cancel HTTP requests", error))?;
            Python::attach(|py| engine.bind(py).borrow_mut().close(py)).map_err(|error| {
                format_python_error("failed to stop the inference engine", error)
            })?;
            return Ok(());
        }
        session.cancel_disconnected()?;
        if session.active_count() == 0 && !engine_has_work(&engine)? {
            thread::sleep(Duration::from_millis(1));
            continue;
        }
        let token_events = Python::attach(|py| {
            let mut engine = engine.bind(py).borrow_mut();
            engine.step(py, None, true).map(|events| {
                events
                    .into_iter()
                    .map(|event| {
                        let event = event.bind(py).borrow();
                        event.clone()
                    })
                    .collect::<Vec<_>>()
            })
        })
        .map_err(|error| format_python_error("engine step failed", error))?;

        if token_events.is_empty() {
            thread::sleep(Duration::from_micros(50));
        }
        session.process(token_events)?;
    }
}

fn engine_has_work(engine: &Py<Engine>) -> Result<bool, String> {
    Ok(Python::attach(|py| engine.bind(py).borrow().has_work()))
}

fn format_python_error(context: &str, error: PyErr) -> String {
    format!("{context}: {error}")
}
