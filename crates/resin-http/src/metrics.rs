//! A scrape endpoint backed entirely by previously published native state.

use std::{
    collections::BTreeMap,
    fmt::{self, Write},
    sync::{Arc, RwLock},
};

use axum::{
    body::Bytes,
    http::{
        HeaderMap, StatusCode,
        header::{ACCEPT, CONTENT_TYPE, VARY},
    },
    response::{IntoResponse, Response},
};
use prometheus_client::{
    collector::Collector,
    encoding::{
        DescriptorEncoder, EncodeLabelValue, LabelValueEncoder, NoLabelSet,
        prometheus_protobuf::{
            self,
            prometheus_data_model::{MetricFamily, MetricType as ProtoMetricType},
        },
        text,
    },
    metrics::{
        MetricType,
        gauge::Gauge,
        histogram::{Histogram, NativeHistogramConfig},
    },
    registry::Registry,
};
use prost::Message;

const TEXT_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";
const PROTO_CONTENT_TYPE: &str =
    "application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=delimited";

/// Encode length-delimited Prometheus MetricFamily messages for publication.
pub fn encode_metrics(families: &[MetricFamily]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for family in families {
        family
            .encode_length_delimited(&mut bytes)
            .expect("Vec has sufficient capacity");
    }
    bytes
}

/// Decode an entire publication, rejecting malformed or truncated framing.
pub fn decode_metrics(mut bytes: &[u8]) -> Result<Vec<MetricFamily>, prost::DecodeError> {
    let mut families = Vec::new();
    while !bytes.is_empty() {
        families.push(MetricFamily::decode_length_delimited(&mut bytes)?);
    }
    Ok(families)
}

/// Human-readable classic exposition. Native distributions require protobuf;
/// text retains histogram counts, sums, and the unbounded bucket.
pub fn metrics_text(families: &[MetricFamily]) -> Result<String, fmt::Error> {
    let mut registry = Registry::default();
    registry.register_collector(Box::new(TextSnapshot(families.to_vec())));
    let mut output = String::new();
    text::encode_registry(&mut output, &registry)?;
    Ok(output)
}

/// Adapt published protobuf values to the crate's text encoder, which owns
/// label syntax and numeric formatting. Counter names are already suffixed.
#[derive(Debug)]
struct TextSnapshot(Vec<MetricFamily>);

impl Collector for TextSnapshot {
    fn encode(&self, mut encoder: DescriptorEncoder) -> Result<(), fmt::Error> {
        for family in &self.0 {
            let kind = match ProtoMetricType::try_from(family.r#type).map_err(|_| fmt::Error)? {
                ProtoMetricType::Counter => MetricType::Counter,
                ProtoMetricType::Gauge => MetricType::Gauge,
                ProtoMetricType::Histogram => MetricType::Histogram,
                _ => return Err(fmt::Error),
            };
            let mut metric_encoder =
                encoder.encode_descriptor(&family.name, &family.help, None, kind)?;
            for metric in &family.metric {
                let labels: Vec<_> = metric
                    .label
                    .iter()
                    .map(|label| (label.name.as_str(), TextLabel(&label.value)))
                    .collect();
                let mut sample = metric_encoder.encode_family(&labels)?;
                match kind {
                    MetricType::Counter => {
                        sample.encode_gauge(&metric.counter.as_ref().ok_or(fmt::Error)?.value)?
                    }
                    MetricType::Gauge => {
                        sample.encode_gauge(&metric.gauge.as_ref().ok_or(fmt::Error)?.value)?
                    }
                    MetricType::Histogram => {
                        let histogram = metric.histogram.as_ref().ok_or(fmt::Error)?;
                        sample.encode_histogram::<NoLabelSet>(
                            histogram.sample_sum,
                            histogram.sample_count,
                            &[(f64::MAX, histogram.sample_count)],
                            None,
                        )?;
                    }
                    _ => unreachable!(),
                }
            }
        }
        Ok(())
    }
}

// prometheus-client writes string label values verbatim. This adapter is used
// only by the text collector; protobuf retains the original, unescaped values.
struct TextLabel<'a>(&'a str);

impl EncodeLabelValue for TextLabel<'_> {
    fn encode(&self, encoder: &mut LabelValueEncoder) -> Result<(), fmt::Error> {
        for character in self.0.chars() {
            match character {
                '\\' => encoder.write_str("\\\\")?,
                '"' => encoder.write_str("\\\"")?,
                '\n' => encoder.write_str("\\n")?,
                character => encoder.write_char(character)?,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct Published {
    text: Bytes,
    protobuf: Bytes,
}

/// Prometheus exposition shared by the inference publisher and HTTP runtime.
///
/// Publishing encodes complete per-rank snapshots before swapping both formats.
/// Scrapes clone cached bytes without invoking inference, Python, or collectives.
#[derive(Clone, Debug, Default)]
pub struct MetricsEndpoint {
    published: Arc<RwLock<Published>>,
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
    /// samples under consistent descriptors. Invalid publications leave both
    /// cached representations unchanged.
    pub fn publish(&self, snapshots: Vec<Vec<u8>>) -> Result<(), MetricsPublishError> {
        let mut families = BTreeMap::<String, MetricFamily>::new();
        let mut merge = |family: MetricFamily| -> Result<(), MetricsPublishError> {
            if let Some(previous) = families.get_mut(&family.name) {
                for (conflict, descriptor) in [
                    (previous.help != family.help, "HELP"),
                    (previous.r#type != family.r#type, "TYPE"),
                    (previous.unit != family.unit, "UNIT"),
                ] {
                    if conflict {
                        return Err(MetricsPublishError::Conflict {
                            metric: family.name,
                            descriptor,
                        });
                    }
                }
                previous.metric.extend(family.metric);
            } else {
                families.insert(family.name.clone(), family);
            }
            Ok(())
        };
        for snapshot in snapshots {
            for family in decode_metrics(&snapshot).map_err(MetricsPublishError::Decode)? {
                merge(family)?;
            }
        }
        if let Some(http) = &self.http {
            for family in
                prometheus_protobuf::encode(&http.registry).map_err(MetricsPublishError::Encode)?
            {
                merge(family)?;
            }
        }
        let families: Vec<_> = families.into_values().collect();
        let published = Published {
            text: Bytes::from(metrics_text(&families).map_err(MetricsPublishError::Encode)?),
            protobuf: Bytes::from(encode_metrics(&families)),
        };
        *self
            .published
            .write()
            .unwrap_or_else(|error| error.into_inner()) = published;
        Ok(())
    }

    pub(crate) fn response(&self, headers: &HeaderMap) -> Response {
        let Some(format) = Exposition::negotiate(headers) else {
            return (
                StatusCode::NOT_ACCEPTABLE,
                [(VARY, "Accept")],
                "supported metrics formats: Prometheus protobuf and text/plain; version=0.0.4",
            )
                .into_response();
        };
        let snapshot = self
            .published
            .read()
            .unwrap_or_else(|error| error.into_inner());
        let (content_type, bytes) = match format {
            Exposition::Text => (TEXT_CONTENT_TYPE, snapshot.text.clone()),
            Exposition::Protobuf => (PROTO_CONTENT_TYPE, snapshot.protobuf.clone()),
        };
        ([(CONTENT_TYPE, content_type), (VARY, "Accept")], bytes).into_response()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Exposition {
    Text,
    Protobuf,
}

impl Exposition {
    fn negotiate(headers: &HeaderMap) -> Option<Self> {
        if !headers.contains_key(ACCEPT) {
            return Some(Self::Text);
        }
        // The most specific matching media range determines each format's q;
        // in particular an explicit q=0 takes precedence over a wildcard.
        let preference = |format: Self| {
            let mut best = None::<((usize, usize), f64)>;
            for header in headers.get_all(ACCEPT) {
                let Ok(header) = header.to_str() else {
                    continue;
                };
                for range in header.split(',') {
                    let mut parts = range.split(';').map(str::trim);
                    let media = parts.next().unwrap_or("");
                    let (target, wildcard) = match format {
                        Self::Text => ("text/plain", "text/*"),
                        Self::Protobuf => ("application/vnd.google.protobuf", "application/*"),
                    };
                    let media_specificity = if media.eq_ignore_ascii_case(target) {
                        2
                    } else if media.eq_ignore_ascii_case(wildcard) {
                        1
                    } else if media == "*/*" {
                        0
                    } else {
                        continue;
                    };
                    let mut specificity = (media_specificity, 0);
                    let mut quality = 1.0;
                    let mut valid = true;
                    let mut extensions = false;
                    for parameter in parts {
                        let Some((name, value)) = parameter.split_once('=') else {
                            valid = false;
                            break;
                        };
                        let value = value.trim().trim_matches('"');
                        if name.trim().eq_ignore_ascii_case("q") {
                            quality = value
                                .parse::<f64>()
                                .ok()
                                .filter(|q| (0.0..=1.0).contains(q))
                                .unwrap_or(0.0);
                            extensions = true;
                        } else if !extensions {
                            let matches = match (format, name.trim().to_ascii_lowercase().as_str())
                            {
                                (_, "charset") => value.eq_ignore_ascii_case("utf-8"),
                                (Self::Text, "version") => value == "0.0.4",
                                (Self::Protobuf, "proto") => {
                                    value == "io.prometheus.client.MetricFamily"
                                }
                                (Self::Protobuf, "encoding") => value == "delimited",
                                // Both formats use legacy-valid metric names.
                                (_, "escaping") => value == "underscores" || value == "allow-utf-8",
                                _ => false,
                            };
                            valid &= matches;
                            specificity.1 += 1;
                        }
                    }
                    if valid
                        && best.is_none_or(|(prior, q)| {
                            specificity > prior || (specificity == prior && quality > q)
                        })
                    {
                        best = Some((specificity, quality));
                    }
                }
            }
            best.filter(|(_, quality)| *quality > 0.0)
        };
        match (preference(Self::Text), preference(Self::Protobuf)) {
            (None, None) => None,
            (Some(_), None) => Some(Self::Text),
            (None, Some(_)) => Some(Self::Protobuf),
            (Some((text_specificity, text_q)), Some((proto_specificity, proto_q))) => Some(
                if proto_q > text_q || (proto_q == text_q && proto_specificity > text_specificity) {
                    Self::Protobuf
                } else {
                    Self::Text
                },
            ),
        }
    }
}

#[derive(Debug)]
struct HttpMetrics {
    registry: Registry,
    first_token: Histogram,
    latency: Histogram,
    queue_time: Histogram,
    queue_size: Gauge,
}

impl HttpMetrics {
    fn new(model_name: String) -> Self {
        let mut registry = Registry::with_labels(
            [
                ("model_name".into(), model_name.into()),
                ("engine_type".into(), "resin".into()),
            ]
            .into_iter(),
        );
        let mut histogram = |name, help| {
            let histogram =
                Histogram::new_native(NativeHistogramConfig::with_schema(4).max_buckets(160));
            registry.register(name, help, histogram.clone());
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
        let queue_size = Gauge::default();
        registry.register(
            "resin:http_num_queue_reqs",
            "Requests waiting in the frontend admission queue.",
            queue_size.clone(),
        );
        Self {
            registry,
            first_token,
            latency,
            queue_time,
            queue_size,
        }
    }
}

/// Invalid metric publications preserve the last successfully published state.
#[derive(Debug)]
pub enum MetricsPublishError {
    Conflict {
        metric: String,
        descriptor: &'static str,
    },
    Decode(prost::DecodeError),
    Encode(fmt::Error),
}

impl fmt::Display for MetricsPublishError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conflict { metric, descriptor } => write!(
                formatter,
                "conflicting {descriptor} descriptors for metric {metric}"
            ),
            Self::Decode(error) => write!(formatter, "invalid metrics publication: {error}"),
            Self::Encode(error) => write!(formatter, "cannot encode metrics publication: {error}"),
        }
    }
}

impl std::error::Error for MetricsPublishError {}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus_client::encoding::prometheus_protobuf::prometheus_data_model::{
        Counter, LabelPair, Metric,
    };

    fn text(endpoint: &MetricsEndpoint) -> String {
        String::from_utf8(endpoint.published.read().unwrap().text.to_vec()).unwrap()
    }

    fn requests(rank: u32, count: f64) -> MetricFamily {
        MetricFamily {
            name: "resin:requests_total".into(),
            help: "Completed requests.".into(),
            r#type: ProtoMetricType::Counter as i32,
            metric: vec![Metric {
                label: vec![LabelPair {
                    name: "rank".into(),
                    value: rank.to_string(),
                }],
                counter: Some(Counter {
                    value: count,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn merges_rank_samples_under_single_descriptors() {
        let endpoint = MetricsEndpoint::default();
        endpoint
            .publish(
                (0..2)
                    .map(|rank| encode_metrics(&[requests(rank, 2.0)]))
                    .collect(),
            )
            .unwrap();
        let merged = text(&endpoint);
        assert_eq!(merged.matches("# HELP resin:requests_total").count(), 1);
        for rank in 0..2 {
            assert!(merged.contains(&format!("resin:requests_total{{rank=\"{rank}\"}} 2.0\n")));
        }
        let families = decode_metrics(&endpoint.published.read().unwrap().protobuf).unwrap();
        assert_eq!(families.len(), 1);
        assert_eq!(families[0].metric.len(), 2);
    }

    #[test]
    fn text_escapes_labels_without_changing_protobuf_values() {
        let endpoint = MetricsEndpoint::default();
        let mut family = requests(0, 2.0);
        let model_name = "model\"\\line\nnext λ";
        family.metric[0].label.push(LabelPair {
            name: "model_name".into(),
            value: model_name.into(),
        });
        endpoint
            .publish(vec![encode_metrics(&[family.clone()])])
            .unwrap();
        assert!(text(&endpoint).contains(r#"model_name="model\"\\line\nnext λ""#));
        let decoded = decode_metrics(&endpoint.published.read().unwrap().protobuf).unwrap();
        assert_eq!(decoded, [family]);
    }

    #[test]
    fn failed_publications_preserve_both_previous_formats() {
        let endpoint = MetricsEndpoint::default();
        endpoint
            .publish(vec![encode_metrics(&[requests(0, 2.0)])])
            .unwrap();
        let previous = endpoint.published.read().unwrap().clone();
        for descriptor in ["HELP", "TYPE", "UNIT"] {
            let mut conflict = requests(1, 2.0);
            match descriptor {
                "HELP" => conflict.help = "Changed help".into(),
                "TYPE" => conflict.r#type = ProtoMetricType::Gauge as i32,
                "UNIT" => conflict.unit = "seconds".into(),
                _ => unreachable!(),
            }
            let error = endpoint
                .publish(vec![encode_metrics(&[requests(0, 2.0), conflict])])
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                format!("conflicting {descriptor} descriptors for metric resin:requests_total")
            );
        }
        assert!(endpoint.publish(vec![vec![0xff]]).is_err());
        let current = endpoint.published.read().unwrap().clone();
        assert_eq!(current.text, previous.text);
        assert_eq!(current.protobuf, previous.protobuf);
        endpoint.publish(vec![]).unwrap();
        assert_eq!(text(&endpoint), "");
    }
}
