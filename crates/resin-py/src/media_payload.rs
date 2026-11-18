//! Immutable media spans shared by recipes, inference, and worker transport.

use numpy::{
    IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods, ndarray,
};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyTuple},
};
use resin_recipe::media::{Input, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sha3::{
    Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};

pub(crate) fn error(value: impl ToString) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(value.to_string())
}

fn tensor(value: &Bound<'_, PyAny>) -> PyResult<Tensor<f32>> {
    let value = value.extract::<PyReadonlyArrayDyn<'_, f32>>()?;
    Tensor::new(
        value.shape().to_vec(),
        value.as_array().iter().copied().collect(),
    )
    .map_err(error)
}

fn array(py: Python<'_>, value: Tensor<f32>) -> PyResult<Py<PyArrayDyn<f32>>> {
    let value = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&value.shape), value.data)
        .map_err(error)?
        .into_pyarray(py);
    value.readwrite().make_nonwriteable();
    Ok(value.unbind())
}

fn spans(
    py: Python<'_>,
    values: impl IntoIterator<Item = (usize, usize)>,
) -> PyResult<Py<PyTuple>> {
    let class = py.import("resin.protocols")?.getattr("MediaTokenSpan")?;
    let values = values
        .into_iter()
        .map(|span| class.call1(span))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, values)?.unbind())
}

fn end((position, length): (usize, usize)) -> PyResult<usize> {
    position
        .checked_add(length)
        .filter(|&end| length > 0 && end <= u32::MAX as usize)
        .ok_or_else(|| {
            error("media spans must have positive length and fit in uint32 token positions")
        })
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Content {
    Image,
    Video {
        frame_spans: Vec<(usize, usize)>,
        fps: f64,
        frame_indices: Vec<usize>,
    },
    Audio {
        sample_rate: u32,
    },
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    span: (usize, usize),
    atomic: bool,
    content: Content,
}

/// One image, video, or audio source and its token range in a prompt.
#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct Media {
    descriptor: Descriptor,
    buffers: Vec<Py<PyArrayDyn<f32>>>,
    identity: [u8; 32],
}

impl Media {
    fn build(py: Python<'_>, descriptor: Descriptor, values: Vec<Tensor<f32>>) -> PyResult<Self> {
        let limit = end(descriptor.span)?;
        let mut hash = Sha256::new();
        match &descriptor.content {
            Content::Image => {
                Digest::update(&mut hash, b"image");
                if values.len() != 1 {
                    return Err(error("images need exactly one pixel buffer"));
                }
            }
            Content::Video {
                frame_spans,
                fps,
                frame_indices,
            } => {
                if values.is_empty()
                    || values.len() != frame_spans.len()
                    || values.len() != frame_indices.len()
                    || !fps.is_finite()
                    || *fps <= 0.0
                    || frame_indices.windows(2).any(|p| p[0] >= p[1])
                {
                    return Err(error(
                        "video frames need matching spans, increasing frame indices, and a positive finite fps",
                    ));
                }
                let mut previous = descriptor.span.0;
                for &span in frame_spans {
                    let stop = end(span)?;
                    if span.0 < previous || stop > limit {
                        return Err(error(
                            "video frame spans must be ordered inside the video span",
                        ));
                    }
                    previous = stop;
                }
                Digest::update(&mut hash, b"video");
                Digest::update(&mut hash, fps.to_le_bytes());
                Digest::update(&mut hash, (frame_indices.len() as u64).to_le_bytes());
                for (&index, &(position, length)) in frame_indices.iter().zip(frame_spans) {
                    Digest::update(&mut hash, (index as u64).to_le_bytes());
                    Digest::update(
                        &mut hash,
                        ((position - descriptor.span.0) as u64).to_le_bytes(),
                    );
                    Digest::update(&mut hash, (length as u64).to_le_bytes());
                }
            }
            Content::Audio { sample_rate } => {
                if *sample_rate == 0 || values.len() != 1 {
                    return Err(error(
                        "audio needs one PCM buffer and a positive sample rate",
                    ));
                }
                Digest::update(&mut hash, b"audio");
                Digest::update(&mut hash, sample_rate.to_le_bytes());
            }
        }
        for value in &values {
            let valid_shape = match descriptor.content {
                Content::Audio { .. } => matches!(value.shape.len(), 1 | 2),
                _ => value.shape.len() == 3 && value.shape[2] == 3,
            };
            if !valid_shape || value.shape.contains(&0) || value.data.iter().any(|v| !v.is_finite())
            {
                return Err(error(
                    "media buffers must be finite nonempty float32 arrays: HWC RGB pixels or mono/multichannel PCM",
                ));
            }
            Digest::update(&mut hash, (value.shape.len() as u64).to_le_bytes());
            for &dimension in &value.shape {
                Digest::update(&mut hash, (dimension as u64).to_le_bytes());
            }
            Digest::update(&mut hash, bytemuck::cast_slice(&value.data));
        }
        Ok(Self {
            descriptor,
            identity: hash.finalize().into(),
            buffers: values
                .into_iter()
                .map(|value| array(py, value))
                .collect::<PyResult<_>>()?,
        })
    }
}

#[pymethods]
impl Media {
    #[new]
    #[pyo3(signature = (span, *, image: "npt.NDArray[np.float32] | None" = None, video: "tuple[npt.NDArray[np.float32], ...] | None" = None, audio: "npt.NDArray[np.float32] | None" = None, audio_sample_rate=None, video_fps=None, video_frame_indices=None, video_frame_spans=None, atomic=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        span: (usize, usize),
        image: Option<&Bound<'_, PyAny>>,
        video: Option<&Bound<'_, PyTuple>>,
        audio: Option<&Bound<'_, PyAny>>,
        audio_sample_rate: Option<u32>,
        video_fps: Option<f64>,
        video_frame_indices: Option<Vec<usize>>,
        video_frame_spans: Option<Vec<(usize, usize)>>,
        atomic: bool,
    ) -> PyResult<Self> {
        let (content, values) = match (image, video, audio) {
            (Some(image), None, None)
                if audio_sample_rate.is_none()
                    && video_fps.is_none()
                    && video_frame_indices.is_none()
                    && video_frame_spans.is_none() =>
            {
                (Content::Image, vec![tensor(image)?])
            }
            (None, Some(video), None) if audio_sample_rate.is_none() => {
                let indices = video_frame_indices.unwrap_or_else(|| (0..video.len()).collect());
                (
                    Content::Video {
                        frame_spans: video_frame_spans
                            .ok_or_else(|| error("video requires video_frame_spans"))?,
                        fps: video_fps.ok_or_else(|| error("video requires video_fps"))?,
                        frame_indices: indices,
                    },
                    video.iter().map(|v| tensor(&v)).collect::<PyResult<_>>()?,
                )
            }
            (None, None, Some(audio))
                if video_fps.is_none()
                    && video_frame_indices.is_none()
                    && video_frame_spans.is_none() =>
            {
                (
                    Content::Audio {
                        sample_rate: audio_sample_rate
                            .ok_or_else(|| error("audio requires audio_sample_rate"))?,
                    },
                    vec![tensor(audio)?],
                )
            }
            _ => {
                return Err(error(
                    "Media requires exactly one of image, video, or audio, with matching metadata",
                ));
            }
        };
        Self::build(
            py,
            Descriptor {
                span,
                atomic,
                content,
            },
            values,
        )
    }

    #[getter]
    fn span(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py
            .import("resin.protocols")?
            .getattr("MediaTokenSpan")?
            .call1(self.descriptor.span)?
            .unbind())
    }
    #[getter]
    fn atomic(&self) -> bool {
        self.descriptor.atomic
    }
    #[getter]
    fn digest(&self) -> String {
        self.identity.iter().map(|v| format!("{v:02x}")).collect()
    }
    #[getter]
    fn image(&self, py: Python<'_>) -> Option<Py<PyArrayDyn<f32>>> {
        matches!(self.descriptor.content, Content::Image).then(|| self.buffers[0].clone_ref(py))
    }
    #[getter]
    fn video(&self, py: Python<'_>) -> PyResult<Option<Py<PyTuple>>> {
        matches!(self.descriptor.content, Content::Video { .. })
            .then(|| {
                PyTuple::new(py, self.buffers.iter().map(|v| v.clone_ref(py))).map(Bound::unbind)
            })
            .transpose()
    }
    #[getter]
    fn audio(&self, py: Python<'_>) -> Option<Py<PyArrayDyn<f32>>> {
        matches!(self.descriptor.content, Content::Audio { .. })
            .then(|| self.buffers[0].clone_ref(py))
    }
    #[getter]
    fn audio_sample_rate(&self) -> Option<u32> {
        match self.descriptor.content {
            Content::Audio { sample_rate } => Some(sample_rate),
            _ => None,
        }
    }
    #[getter]
    fn video_fps(&self) -> Option<f64> {
        match self.descriptor.content {
            Content::Video { fps, .. } => Some(fps),
            _ => None,
        }
    }
    #[getter]
    fn video_frame_indices(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let values = match &self.descriptor.content {
            Content::Video { frame_indices, .. } => frame_indices.as_slice(),
            _ => &[],
        };
        Ok(PyTuple::new(py, values)?.unbind())
    }
    #[getter]
    fn video_frame_spans(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let values = match &self.descriptor.content {
            Content::Video { frame_spans, .. } => frame_spans.as_slice(),
            _ => &[],
        };
        spans(py, values.iter().copied())
    }
}

pub(crate) fn bind(
    py: Python<'_>,
    items: Vec<Input>,
    placements: &[(usize, usize)],
) -> PyResult<Py<PyTuple>> {
    if items.len() != placements.len() {
        return Err(error("media inputs and token placements differ"));
    }
    let items = items
        .into_iter()
        .zip(placements)
        .map(|(input, &placement)| {
            let (span, atomic, content, values) = match input {
                Input::Image {
                    pixels,
                    span,
                    atomic,
                } => {
                    if end(span)? > placement.1 {
                        return Err(error("image span exceeds its token placement"));
                    }
                    (
                        (
                            placement
                                .0
                                .checked_add(span.0)
                                .ok_or_else(|| error("media span overflows"))?,
                            span.1,
                        ),
                        atomic,
                        Content::Image,
                        vec![pixels],
                    )
                }
                Input::Video {
                    frames,
                    frame_spans,
                    fps,
                    frame_indices,
                    atomic,
                } => {
                    let frame_spans = frame_spans
                        .into_iter()
                        .map(|(p, n)| {
                            Ok((
                                placement
                                    .0
                                    .checked_add(p)
                                    .ok_or_else(|| error("media span overflows"))?,
                                n,
                            ))
                        })
                        .collect::<PyResult<_>>()?;
                    (
                        placement,
                        atomic,
                        Content::Video {
                            frame_spans,
                            fps,
                            frame_indices,
                        },
                        frames,
                    )
                }
            };
            Py::new(
                py,
                Media::build(
                    py,
                    Descriptor {
                        span,
                        atomic,
                        content,
                    },
                    values,
                )?,
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, items)?.unbind())
}

type Admission = (Vec<u32>, Vec<(u32, u32)>);

/// Salt media tokens without losing text identity; return atomic [start, end) ranges.
pub(crate) fn admission(items: &Bound<'_, PyTuple>, tokens: &[u32]) -> PyResult<Admission> {
    let mut keys = tokens.to_vec();
    let mut atomic = Vec::new();
    let mut previous = 0;
    for item in items {
        let item = item.cast::<Media>()?.borrow();
        let (start, _) = item.descriptor.span;
        let stop = end(item.descriptor.span)?;
        if start < previous || stop > tokens.len() {
            return Err(error(
                "media spans must be ordered, non-overlapping, and inside the prompt",
            ));
        }
        previous = stop;
        let mut salt = Shake256::default();
        Update::update(&mut salt, &item.identity);
        Update::update(&mut salt, bytemuck::cast_slice(&tokens[start..stop]));
        let mut stream = salt.finalize_xof();
        for key in &mut keys[start..stop] {
            let mut bytes = [0; 4];
            stream.read(&mut bytes);
            *key = u32::from_le_bytes(bytes);
        }
        if item.descriptor.atomic {
            match &item.descriptor.content {
                Content::Video { frame_spans, .. } => {
                    atomic.extend(frame_spans.iter().map(|&(p, n)| (p as u32, (p + n) as u32)))
                }
                _ => atomic.push((start as u32, stop as u32)),
            }
        }
    }
    Ok((keys, atomic))
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Transfer {
    version: u8,
    items: Vec<Descriptor>,
}

#[pyfunction]
#[pyo3(signature = (media: "tuple[Media, ...]") -> "tuple[bytes, tuple[npt.NDArray[Any], ...]]")]
pub(crate) fn _media_to_buffers(
    py: Python<'_>,
    media: &Bound<'_, PyTuple>,
) -> PyResult<(Py<PyBytes>, Py<PyTuple>)> {
    let mut descriptors = Vec::with_capacity(media.len());
    let mut buffers = Vec::new();
    let mut previous = 0;
    for item in media {
        let item = item.cast::<Media>()?.borrow();
        if item.descriptor.span.0 < previous {
            return Err(error("media spans must be ordered and non-overlapping"));
        }
        previous = end(item.descriptor.span)?;
        descriptors.push(item.descriptor.clone());
        buffers.extend(item.buffers.iter().map(|v| v.clone_ref(py)));
    }
    let metadata = serde_json::to_vec(&Transfer {
        version: 2,
        items: descriptors,
    })
    .map_err(error)?;
    Ok((
        PyBytes::new(py, &metadata).unbind(),
        PyTuple::new(py, buffers)?.unbind(),
    ))
}

#[pyfunction]
#[pyo3(signature = (metadata, buffers: "Sequence[npt.NDArray[Any]]") -> "tuple[Media, ...]")]
pub(crate) fn _media_from_buffers(
    py: Python<'_>,
    metadata: &[u8],
    buffers: Vec<Py<PyAny>>,
) -> PyResult<Py<PyTuple>> {
    let transfer: Transfer = serde_json::from_slice(metadata).map_err(error)?;
    if transfer.version != 2 {
        return Err(error("unsupported media transfer version"));
    }
    let mut buffers = buffers.into_iter();
    let mut items = Vec::with_capacity(transfer.items.len());
    let mut previous = 0;
    for descriptor in transfer.items {
        if descriptor.span.0 < previous {
            return Err(error("media spans must be ordered and non-overlapping"));
        }
        previous = end(descriptor.span)?;
        let count = match &descriptor.content {
            Content::Video { frame_spans, .. } => frame_spans.len(),
            _ => 1,
        };
        let values = (0..count)
            .map(|_| {
                let value = buffers
                    .next()
                    .ok_or_else(|| error("media transfer is missing a buffer"))?;
                tensor(value.bind(py))
            })
            .collect::<PyResult<_>>()?;
        items.push(Py::new(py, Media::build(py, descriptor, values)?)?);
    }
    if buffers.next().is_some() {
        return Err(error("media transfer has unexpected buffers"));
    }
    Ok(PyTuple::new(py, items)?.unbind())
}
