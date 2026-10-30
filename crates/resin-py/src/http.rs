use std::{
    net::{IpAddr, SocketAddr},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc as std_mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use resin_http::{GenerationJob, JobKind, MetricsEndpoint, RouteConfig};
use tokio::sync::{mpsc, oneshot, watch};

use crate::Engine;
use crate::http_session::{RenderedJob, Session, finish_count, render_job};

#[derive(Clone, Copy)]
pub(crate) struct AdmissionConfig {
    /// Maximum admitted requests, including prompts being rendered and requests
    /// waiting in the engine.
    pub max_active_requests: usize,
    /// Threads that render prompts and prepare media off the engine thread.
    pub render_workers: usize,
    /// Maximum requests admitted between inference steps, scaled by replica
    /// count when distributed.
    pub admissions_per_step: usize,
}

#[pyclass(name = "_HttpServer", module = "resin._resin")]
pub(crate) struct NativeHttpServer {
    engine: Py<Engine>,
    distributed: Option<Py<PyAny>>,
    recipe: Py<PyAny>,
    model: String,
    stream_interval: u32,
    request_queue_capacity: u32,
    admission: AdmissionConfig,
    routes: RouteConfig,
    running: AtomicBool,
    shutdown: Mutex<Option<watch::Sender<bool>>>,
}

#[pymethods]
impl NativeHttpServer {
    #[new]
    #[pyo3(signature = (
        engine,
        recipe: "Recipe",
        *,
        model,
        stream_interval = 1,
        request_queue_capacity = 1024,
        max_active_requests = None,
        render_workers = None,
        admissions_per_step = 32,
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
        max_active_requests: Option<u32>,
        render_workers: Option<u32>,
        admissions_per_step: u32,
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
        if max_active_requests == Some(0) {
            return Err(PyValueError::new_err(
                "max_active_requests must be positive",
            ));
        }
        if render_workers == Some(0) {
            return Err(PyValueError::new_err("render_workers must be positive"));
        }
        if admissions_per_step == 0 {
            return Err(PyValueError::new_err(
                "admissions_per_step must be positive",
            ));
        }
        let render_workers = render_workers.map_or_else(
            || {
                thread::available_parallelism()
                    .map_or(1, usize::from)
                    .min(16)
            },
            |value| value as usize,
        );
        let max_active_requests = max_active_requests.map_or_else(
            || {
                Python::attach(|py| {
                    engine
                        .bind(py)
                        .borrow()
                        .max_running_requests()
                        .saturating_add(render_workers)
                })
            },
            |value| value as usize,
        );
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
            admission: AdmissionConfig {
                max_active_requests,
                render_workers,
                admissions_per_step: admissions_per_step as usize,
            },
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
        let admission = self.admission;
        let result = py.detach(move || {
            run_server(
                engine,
                distributed,
                recipe,
                model,
                routes,
                stream_interval,
                request_queue_capacity,
                admission,
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
            "_HttpServer(model={:?}, stream_interval={}, request_queue_capacity={}, max_active_requests={}, render_workers={}, admissions_per_step={}, enable_openai={}, enable_anthropic={}, enable_demo={}, running={})",
            self.model,
            self.stream_interval,
            self.request_queue_capacity,
            self.admission.max_active_requests,
            self.admission.render_workers,
            self.admission.admissions_per_step,
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
    admission: AdmissionConfig,
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
                    recipe,
                    receiver,
                    actor_shutdown,
                    routes.routed_dp_size,
                    admission,
                    actor_metrics,
                )
            } else {
                engine_actor(
                    engine,
                    recipe,
                    receiver,
                    actor_shutdown,
                    stream_interval,
                    admission,
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

/// Shared fixed-size render workers. Callers bound queued work with their
/// admission budget; token counts finish here without reaching inference.
pub(crate) struct RenderWorkers {
    pub jobs: std_mpsc::Sender<GenerationJob>,
    pub completed: std_mpsc::Receiver<Option<(GenerationJob, RenderedJob)>>,
}

impl RenderWorkers {
    pub fn new(recipe: &Py<PyAny>, workers: usize) -> Result<Self, String> {
        let (render_jobs, render_queue) = std_mpsc::channel::<GenerationJob>();
        let render_queue = Arc::new(Mutex::new(render_queue));
        let (rendered_jobs, rendered_queue) =
            std_mpsc::channel::<Option<(GenerationJob, RenderedJob)>>();
        for worker in 0..workers {
            let queue = render_queue.clone();
            let results = rendered_jobs.clone();
            let recipe = Python::attach(|py| recipe.clone_ref(py));
            thread::Builder::new()
                .name(format!("resin-render-{worker}"))
                .spawn(move || {
                    loop {
                        let job = match queue.lock().unwrap_or_else(|p| p.into_inner()).recv() {
                            Ok(job) => job,
                            Err(_) => return,
                        };
                        let rendered = render_job(&recipe, &job);
                        let completed = if job.kind == JobKind::CountTokens {
                            finish_count(job, rendered);
                            None
                        } else {
                            Some((job, rendered))
                        };
                        if results.send(completed).is_err() {
                            return;
                        }
                    }
                })
                .map_err(|error| format!("failed to start a render worker: {error}"))?;
        }
        drop(rendered_jobs);
        Ok(Self {
            jobs: render_jobs,
            completed: rendered_queue,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn engine_actor(
    engine: Py<Engine>,
    recipe: Py<PyAny>,
    mut jobs: mpsc::Receiver<GenerationJob>,
    shutdown: watch::Receiver<bool>,
    stream_interval: u32,
    admission: AdmissionConfig,
    metrics: Option<MetricsEndpoint>,
    model: &str,
) -> Result<(), String> {
    let mut session =
        Python::attach(|py| Session::new(py, engine.clone_ref(py), recipe, stream_interval))
            .map_err(|error| format_python_error("HTTP session preparation failed", error))?;
    session.metrics = metrics.clone();
    let renders = RenderWorkers::new(session.recipe(), admission.render_workers)?;
    let mut rendering = 0usize;
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
        loop {
            if disconnected
                || *shutdown.borrow()
                || session.active_count() + rendering >= admission.max_active_requests
            {
                break;
            }
            match jobs.try_recv() {
                Ok(job) => {
                    rendering += 1;
                    if let Err(std_mpsc::SendError(job)) = renders.jobs.send(job) {
                        rendering -= 1;
                        if job.kind == JobKind::CountTokens {
                            let _ = job.events.try_send(resin_http::GenerationEvent::Error {
                                message: "prompt rendering is unavailable".to_owned(),
                            });
                        } else {
                            session.admit(job);
                        }
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        for _ in 0..admission.admissions_per_step {
            match renders.completed.try_recv() {
                Ok(completed) => {
                    rendering -= 1;
                    if let Some((job, rendered)) = completed {
                        session.admit_rendered(job, rendered);
                    }
                }
                Err(_) => break,
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
            // Rendering requests arrive through the worker queue.
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
