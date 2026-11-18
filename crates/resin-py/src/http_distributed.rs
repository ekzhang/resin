//! The shared HTTP frontend keeps client channels; replica sessions own inference.

use std::{
    collections::HashMap,
    thread,
    time::{Duration, Instant},
};

use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use resin_http::{ConversationRequest, GenerationEvent, GenerationJob, MetricsEndpoint};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

use crate::http::AdmissionConfig;

/// Native ingress stays typed across the Python coordinator. Only transport
/// between ranks encodes it; JSON never becomes a Python object graph.
#[pyclass(name = "_HttpIngress", skip_from_py_object, module = "resin._resin")]
#[derive(Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HttpIngress {
    pub(crate) requests: Vec<ConversationRequest>,
    pub(crate) canceled: Vec<String>,
    #[pyo3(get)]
    pub(crate) shutdown: bool,
    #[pyo3(get)]
    pub(crate) error: Option<String>,
    #[pyo3(get)]
    pub(crate) collect_metrics: bool,
}

#[pymethods]
impl HttpIngress {
    #[new]
    #[pyo3(signature = (*, shutdown = false))]
    fn new(shutdown: bool) -> Self {
        Self {
            shutdown,
            ..Self::default()
        }
    }

    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        if data.is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_slice(data).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = if self.requests.is_empty()
            && self.canceled.is_empty()
            && !self.shutdown
            && self.error.is_none()
            && !self.collect_metrics
        {
            Vec::new()
        } else {
            serde_json::to_vec(self).map_err(|error| PyValueError::new_err(error.to_string()))?
        };
        Ok(PyBytes::new(py, &bytes))
    }
}

#[pyclass(
    name = "_HttpEvents",
    frozen,
    skip_from_py_object,
    module = "resin._resin"
)]
#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct HttpEvents {
    pub(crate) events: Vec<(String, GenerationEvent)>,
    pub(crate) metrics: Vec<Vec<u8>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) released: Vec<String>,
}

#[pymethods]
impl HttpEvents {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        if data.is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_slice(data).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = if self.events.is_empty() && self.metrics.is_empty() && self.released.is_empty()
        {
            Vec::new()
        } else {
            serde_json::to_vec(self).map_err(|error| PyValueError::new_err(error.to_string()))?
        };
        Ok(PyBytes::new(py, &bytes))
    }

    fn with_metrics(&self, snapshots: Vec<Vec<u8>>) -> Self {
        Self {
            events: self.events.clone(),
            metrics: snapshots,
            released: self.released.clone(),
        }
    }

    #[staticmethod]
    fn combine(packets: Vec<PyRef<'_, Self>>) -> Self {
        Self {
            metrics: packets
                .iter()
                .flat_map(|packet| packet.metrics.iter().cloned())
                .collect(),
            events: packets
                .iter()
                .flat_map(|packet| packet.events.iter().cloned())
                .collect(),
            released: packets
                .iter()
                .flat_map(|packet| packet.released.iter().cloned())
                .collect(),
        }
    }
}

struct Client {
    // After disconnect, hold the admission slot until the replica releases it.
    events: Option<mpsc::Sender<GenerationEvent>>,
    enqueued_at: Instant,
}

pub(crate) fn run(
    coordinator: Py<PyAny>,
    mut jobs: mpsc::Receiver<GenerationJob>,
    shutdown: watch::Receiver<bool>,
    replicas: usize,
    admission: AdmissionConfig,
    metrics: Option<MetricsEndpoint>,
) -> Result<(), String> {
    let mut clients = HashMap::<String, Client>::new();
    let mut disconnected = false;
    let mut canceled_next = Vec::new();
    let mut next_rank = 0;
    let mut next_metrics = Instant::now();
    loop {
        if let Some(endpoint) = &metrics {
            endpoint.set_queue_size(jobs.len());
        }
        let collect_metrics = metrics.is_some() && Instant::now() >= next_metrics;
        if collect_metrics {
            next_metrics = Instant::now() + Duration::from_secs(1);
        }
        let mut admissions = Vec::new();
        let mut canceled = std::mem::take(&mut canceled_next);
        canceled.retain(|id| clients.contains_key(id));
        for (id, client) in &mut clients {
            if client.events.as_ref().is_some_and(mpsc::Sender::is_closed) {
                client.events = None;
                canceled.push(id.clone());
            }
        }
        let mut ingress_error = None;
        for _ in 0..admission.admissions_per_step.saturating_mul(replicas) {
            if *shutdown.borrow()
                || clients.len() >= admission.max_active_requests.saturating_mul(replicas)
            {
                break;
            }
            match jobs.try_recv() {
                Ok(mut job) if !job.events.is_closed() => {
                    if clients.contains_key(&job.request.id) {
                        ingress_error = Some("HTTP request IDs must be unique strings".to_owned());
                        break;
                    }
                    let rank = job.request.routed_dp_rank.unwrap_or_else(|| {
                        let rank = next_rank;
                        next_rank = (next_rank + 1) % replicas;
                        rank
                    });
                    if rank >= replicas {
                        ingress_error =
                            Some("routed_dp_rank is outside the attention-DP mesh".to_owned());
                        break;
                    }
                    job.request.routed_dp_rank = Some(rank);
                    if let Some(endpoint) = &metrics {
                        endpoint.observe_queue_time(job.enqueued_at.elapsed().as_secs_f64());
                    }
                    clients.insert(
                        job.request.id.clone(),
                        Client {
                            events: Some(job.events),
                            enqueued_at: job.enqueued_at,
                        },
                    );
                    admissions.push(job.request);
                }
                Ok(_) => {}
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        let ingress = HttpIngress {
            requests: admissions,
            canceled,
            shutdown: disconnected || *shutdown.borrow() || ingress_error.is_some(),
            error: ingress_error,
            collect_metrics,
        };
        let result = Python::attach(|py| {
            let object = coordinator.bind(py);
            let output = object.call_method1("round", (ingress,))?;
            let packet = output.cast::<HttpEvents>()?.borrow().clone();
            let closed: bool = object.getattr("closed")?.extract()?;
            Ok::<_, PyErr>((packet, closed))
        });
        let (output, closed) = match result {
            Ok(output) => output,
            Err(error) => {
                let message = format!("distributed HTTP execution failed: {error}");
                for client in clients.values().filter_map(|client| client.events.as_ref()) {
                    let _ = client.try_send(GenerationEvent::Error {
                        message: message.clone(),
                    });
                }
                return Err(message);
            }
        };
        if let Some(endpoint) = &metrics
            && !output.metrics.is_empty()
        {
            endpoint
                .publish(output.metrics)
                .map_err(|error| error.to_string())?;
        }
        for (id, event) in output.events {
            let terminal = matches!(
                event,
                GenerationEvent::Finished { .. } | GenerationEvent::Error { .. }
            );
            if let Some(client) = clients.get_mut(&id) {
                if let Some(endpoint) = &metrics {
                    let since = client.enqueued_at;
                    match &event {
                        GenerationEvent::FirstToken => {
                            endpoint.observe_first_token(since.elapsed().as_secs_f64())
                        }
                        GenerationEvent::Finished { .. } => {
                            endpoint.observe_completed(since.elapsed().as_secs_f64())
                        }
                        _ => {}
                    }
                }
                // Backpressure remains local to one client. Cancel its owner
                // on the next round instead of delaying every active replica.
                if !matches!(event, GenerationEvent::FirstToken)
                    && let Some(sender) = &client.events
                    && sender.try_send(event).is_err()
                    && !terminal
                {
                    client.events = None;
                    canceled_next.push(id.clone());
                }
            }
            if terminal {
                clients.remove(&id);
            }
        }
        for id in output.released {
            clients.remove(&id);
        }
        if closed {
            return Ok(());
        }
        if clients.values().all(|client| client.events.is_none()) {
            thread::sleep(Duration::from_millis(1));
        }
    }
}
