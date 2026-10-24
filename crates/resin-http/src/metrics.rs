//! A scrape endpoint backed entirely by previously published native state.

use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, RwLock},
};

use axum::{
    body::Bytes,
    http::header::CONTENT_TYPE,
    response::{IntoResponse, Response},
};
use prometheus::{Histogram, HistogramOpts, IntGauge, Opts, Registry, TextEncoder};

/// Prometheus exposition shared by the inference publisher and HTTP runtime.
///
/// Publishing formats complete per-rank snapshots before swapping the buffer.
/// Scrapes clone that buffer without invoking inference, Python, or collectives.
#[derive(Clone, Debug, Default)]
pub struct MetricsEndpoint {
    published: Arc<RwLock<Bytes>>,
    http: Option<Arc<HttpMetrics>>,
}

impl MetricsEndpoint {
    /// Enable frontend metrics alongside engine snapshots. Frontend counters
    /// are singleton observations, independent of the distributed rank mesh.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            http: Some(Arc::new(HttpMetrics::new(model_name.into()))),
            ..Self::default()
        }
    }

    /// Observe HTTP admission through the first raw committed output token.
    /// Called before parser buffering or stream-interval coalescing.
    pub fn observe_first_token(&self, seconds: f64) {
        if let Some(http) = &self.http {
            http.first_token.observe(seconds);
        }
    }

    /// Observe a successful HTTP generation request's completion time, measured
    /// from initial queue admission. Aborted requests do not enter this histogram.
    pub fn observe_completed(&self, seconds: f64) {
        if let Some(http) = &self.http {
            http.latency.observe(seconds);
        }
    }

    /// Observe time in the frontend queue before recipe rendering starts.
    pub fn observe_queue_time(&self, seconds: f64) {
        if let Some(http) = &self.http {
            http.queue_time.observe(seconds);
        }
    }

    /// Publish the current frontend admission queue length.
    pub fn set_queue_size(&self, requests: usize) {
        if let Some(http) = &self.http {
            http.queue_size.set(requests.min(i64::MAX as usize) as i64);
        }
    }

    /// Replace the complete snapshot, preserving rank labels and grouping all
    /// samples of a metric family behind one consistent HELP/TYPE descriptor.
    /// Conflicting descriptors reject the publication, preserving the prior one.
    pub fn publish(&self, mut snapshots: Vec<String>) -> Result<(), MetricsPublishError> {
        if let Some(http) = &self.http {
            snapshots.push(
                TextEncoder::new()
                    .encode_to_string(&http.registry.gather())
                    .expect("registered HTTP metrics have valid descriptors"),
            );
        }
        let mut families = BTreeMap::<&str, Family<'_>>::new();
        for snapshot in &snapshots {
            for line in snapshot.lines() {
                let (kind, description) = if let Some(description) = line.strip_prefix("# HELP ") {
                    ("HELP", description)
                } else if let Some(description) = line.strip_prefix("# TYPE ") {
                    ("TYPE", description)
                } else {
                    continue;
                };
                let name = description.split_ascii_whitespace().next().unwrap_or("");
                let family = families.entry(name).or_default();
                let slot = if kind == "HELP" {
                    &mut family.help
                } else {
                    &mut family.kind
                };
                if slot.is_some_and(|previous| previous != line) {
                    return Err(MetricsPublishError {
                        metric: name.to_owned(),
                        descriptor: kind,
                    });
                }
                *slot = Some(line);
            }
        }
        for snapshot in &snapshots {
            for line in snapshot
                .lines()
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
            {
                let name = line.split(['{', ' ', '\t']).next().unwrap_or("");
                let family = if families.contains_key(name) {
                    name
                } else {
                    ["_bucket", "_sum", "_count", "_created"]
                        .iter()
                        .find_map(|suffix| {
                            name.strip_suffix(suffix)
                                .filter(|base| families.contains_key(base))
                        })
                        .unwrap_or(name)
                };
                families.entry(family).or_default().samples.push(line);
            }
        }
        let mut output = String::with_capacity(snapshots.iter().map(String::len).sum());
        for family in families.into_values() {
            for line in family
                .help
                .into_iter()
                .chain(family.kind)
                .chain(family.samples)
            {
                output.push_str(line);
                output.push('\n');
            }
        }
        *self
            .published
            .write()
            .unwrap_or_else(|error| error.into_inner()) = Bytes::from(output);
        Ok(())
    }

    pub(crate) fn response(&self) -> Response {
        let snapshot = self
            .published
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        (
            [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
            snapshot,
        )
            .into_response()
    }
}

#[derive(Debug)]
struct HttpMetrics {
    registry: Registry,
    first_token: Histogram,
    latency: Histogram,
    queue_time: Histogram,
    queue_size: IntGauge,
}

impl HttpMetrics {
    fn new(model_name: String) -> Self {
        let registry = Registry::new_custom(
            None,
            Some(
                [
                    ("model_name".into(), model_name),
                    ("engine_type".into(), "resin".into()),
                ]
                .into(),
            ),
        )
        .expect("HTTP metric labels are valid");
        let histogram = |name, help| {
            let histogram = Histogram::with_opts(HistogramOpts::new(name, help).buckets(vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0,
                80.0, 160.0, 320.0, 640.0,
            ]))
            .expect("HTTP histogram descriptor is valid");
            registry
                .register(Box::new(histogram.clone()))
                .expect("HTTP histogram names are distinct");
            histogram
        };
        let first_token = histogram(
            "resin:http_time_to_first_token_seconds",
            "HTTP queue admission to first committed output token, before stream buffering.",
        );
        let latency = histogram(
            "resin:http_request_latency_seconds",
            "HTTP queue admission to successful completion, including rendering.",
        );
        let queue_time = histogram(
            "resin:http_queue_time_seconds",
            "HTTP admission queue time before recipe rendering starts.",
        );
        let queue_size = IntGauge::with_opts(Opts::new(
            "resin:http_num_queue_reqs",
            "Requests waiting in the frontend admission queue.",
        ))
        .expect("HTTP queue descriptor is valid");
        registry
            .register(Box::new(queue_size.clone()))
            .expect("HTTP queue metric name is distinct");
        Self {
            registry,
            first_token,
            latency,
            queue_time,
            queue_size,
        }
    }
}

#[derive(Default)]
struct Family<'a> {
    help: Option<&'a str>,
    kind: Option<&'a str>,
    samples: Vec<&'a str>,
}

/// Two publishers disagreed on the descriptor for one metric family.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetricsPublishError {
    metric: String,
    descriptor: &'static str,
}

impl fmt::Display for MetricsPublishError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "conflicting {} descriptors for metric {}",
            self.descriptor, self.metric
        )
    }
}

impl std::error::Error for MetricsPublishError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(endpoint: &MetricsEndpoint) -> String {
        String::from_utf8(endpoint.published.read().unwrap().to_vec()).unwrap()
    }

    #[test]
    fn merges_rank_samples_and_histograms_under_single_descriptors() {
        let endpoint = MetricsEndpoint::default();
        endpoint.publish((0..2).map(|rank| format!(
            "# HELP resin:requests_total Completed requests.\n# TYPE resin:requests_total counter\nresin:requests_total{{rank=\"{rank}\"}} 2\n# HELP resin:latency_seconds Latency.\n# TYPE resin:latency_seconds histogram\nresin:latency_seconds_bucket{{rank=\"{rank}\",le=\"+Inf\"}} 2\nresin:latency_seconds_sum{{rank=\"{rank}\"}} 4\nresin:latency_seconds_count{{rank=\"{rank}\"}} 2\n"
        )).collect()).unwrap();
        let merged = text(&endpoint);
        assert_eq!(merged.matches("# HELP resin:requests_total").count(), 1);
        assert_eq!(merged.matches("# TYPE resin:latency_seconds").count(), 1);
        assert_eq!(merged.matches("resin:latency_seconds_bucket{").count(), 2);
        assert!(
            merged
                .find("resin:latency_seconds_count{rank=\"1\"}")
                .unwrap()
                < merged.find("# HELP resin:requests_total").unwrap()
        );
        for rank in 0..2 {
            assert!(merged.contains(&format!("resin:requests_total{{rank=\"{rank}\"}} 2\n")));
        }
    }

    #[test]
    fn conflicting_metadata_preserves_previous_publication() {
        let endpoint = MetricsEndpoint::default();
        endpoint
            .publish(vec![
                "# TYPE resin:requests_total counter\nresin:requests_total 2\n".into(),
            ])
            .unwrap();
        let previous = text(&endpoint);
        let error = endpoint
            .publish(vec![
                "# TYPE resin:requests_total counter\n".into(),
                "# TYPE resin:requests_total gauge\n".into(),
            ])
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "conflicting TYPE descriptors for metric resin:requests_total"
        );
        assert_eq!(text(&endpoint), previous);
        endpoint.publish(vec![]).unwrap();
        assert_eq!(text(&endpoint), "");
    }

    #[test]
    fn concurrent_readers_only_observe_complete_snapshots() {
        let endpoint = MetricsEndpoint::default();
        endpoint
            .publish(vec!["resin:gauge 0\n".repeat(100)])
            .unwrap();
        std::thread::scope(|scope| {
            let publisher = endpoint.clone();
            scope.spawn(move || {
                for value in 1..100 {
                    publisher
                        .publish(vec![format!("resin:gauge {value}\n").repeat(100)])
                        .unwrap();
                }
            });
            for _ in 0..100 {
                let snapshot = text(&endpoint);
                let mut lines = snapshot.lines();
                let first = lines.next().unwrap();
                assert_eq!(lines.clone().count(), 99);
                assert!(lines.all(|line| line == first));
            }
        });
    }

    #[test]
    fn frontend_observations_publish_with_stable_labels_and_separate_clocks() {
        let endpoint = MetricsEndpoint::new("test-model");
        endpoint.set_queue_size(3);
        endpoint.observe_queue_time(0.25);
        endpoint.observe_first_token(0.75);
        endpoint.observe_completed(1.5);
        // Zero-output successful requests have no first-token observation.
        endpoint.observe_completed(0.5);
        assert!(text(&endpoint).is_empty());
        endpoint.publish(vec![]).unwrap();
        let first = text(&endpoint);
        let sample = |snapshot: &str, name: &str| {
            let line = snapshot
                .lines()
                .find(|line| line.starts_with(&format!("resin:{name}{{")))
                .unwrap();
            assert!(line.contains("engine_type=\"resin\""));
            assert!(line.contains("model_name=\"test-model\""));
            line.rsplit_once(' ').unwrap().1.parse::<f64>().unwrap()
        };
        assert_eq!(sample(&first, "http_num_queue_reqs"), 3.0);
        assert_eq!(sample(&first, "http_request_latency_seconds_count"), 2.0);
        assert_eq!(sample(&first, "http_request_latency_seconds_sum"), 2.0);
        assert_eq!(
            sample(&first, "http_time_to_first_token_seconds_count"),
            1.0
        );
        assert_eq!(sample(&first, "http_time_to_first_token_seconds_sum"), 0.75);
        assert_eq!(sample(&first, "http_queue_time_seconds_sum"), 0.25);
        endpoint.set_queue_size(0);
        assert_eq!(text(&endpoint), first);
        endpoint.publish(vec![]).unwrap();
        assert_eq!(sample(&text(&endpoint), "http_num_queue_reqs"), 0.0);
    }
}
