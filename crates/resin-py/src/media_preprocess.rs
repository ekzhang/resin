//! Native recipe media preparation. Python is only the input/output buffer boundary.

use std::{collections::VecDeque, mem::size_of, sync::Arc};

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict, PyString, PyTuple},
};
use pythonize::depythonize;
use resin_recipe::{
    Encoder, Error, MediaConfig,
    media::{self, RgbImage, Source, Video},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    media_payload::{
        DeepSeekV41Image, DeepSeekV41Media, Gemma4Media, Glm53FlashMedia, InklingMedia,
        Qwen35Media, error,
    },
    recipe_media::MediaTokens,
    recipe_types::MediaSources,
};

#[derive(Clone)]
enum Payload {
    DeepSeek(media::deepseek::Prepared, u32),
    Gemma(media::gemma::Prepared, u32, u32),
    Inkling(media::inkling::Prepared, u32),
    Qwen(media::qwen::Prepared, usize, u32),
    Glm(media::qwen::Prepared, usize, u32),
}
#[derive(Clone)]
pub(crate) struct Prepared {
    pub(crate) expansions: Vec<MediaTokens>,
    payload: Payload,
}

pub(crate) type CacheKey = [u8; 32];

struct CacheEntry {
    key: CacheKey,
    prepared: Arc<Prepared>,
    bytes: usize,
}

/// Owned by one immutable native recipe, so processor/tokenizer configuration is
/// implicit in the key. No Python object, source string, or conversation survives
/// in this cache. Cache the ordered image set because its prepared tensors may
/// share padded batch dimensions; bind positions against each new prompt.
pub(crate) struct PreparationCache {
    entries: VecDeque<CacheEntry>,
    retained_bytes: usize,
    max_entries: usize,
    max_bytes: usize,
    hits: usize,
    misses: usize,
}

impl Default for PreparationCache {
    fn default() -> Self {
        Self::new(8, 128 * 1024 * 1024)
    }
}

impl PreparationCache {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            retained_bytes: 0,
            max_entries,
            max_bytes,
            hits: 0,
            misses: 0,
        }
    }

    pub(crate) fn get(&mut self, key: CacheKey) -> Option<Arc<Prepared>> {
        let Some(index) = self.entries.iter().position(|entry| entry.key == key) else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        self.hits = self.hits.saturating_add(1);
        let entry = self.entries.remove(index).unwrap();
        let prepared = entry.prepared.clone();
        self.entries.push_back(entry);
        Some(prepared)
    }

    pub(crate) fn insert(&mut self, key: CacheKey, prepared: Arc<Prepared>) {
        // Include tensor capacities, metadata, Arc counters, and entry storage.
        let bytes = prepared.retained_bytes() + 2 * size_of::<usize>() + size_of::<CacheEntry>();
        if self.max_entries == 0 || bytes > self.max_bytes {
            return;
        }
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            self.retained_bytes -= self.entries.remove(index).unwrap().bytes;
        }
        while self.entries.len() >= self.max_entries || bytes > self.max_bytes - self.retained_bytes
        {
            self.retained_bytes -= self.entries.pop_front().unwrap().bytes;
        }
        self.retained_bytes += bytes;
        self.entries.push_back(CacheEntry {
            key,
            prepared,
            bytes,
        });
    }

    pub(crate) fn info(&self) -> [(&'static str, usize); 4] {
        [
            ("entries", self.entries.len()),
            ("retained_bytes", self.retained_bytes),
            ("hits", self.hits),
            ("misses", self.misses),
        ]
    }
}

fn hash_field(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

/// Hash encoded immutable sources before decoding. Mutable arrays/bytearrays and
/// videos follow the uncached path; object identity is never a content key.
pub(crate) fn cache_key(py: Python<'_>, sources: &MediaSources) -> PyResult<Option<CacheKey>> {
    if sources.is_empty() || sources.iter().any(|(kind, _)| *kind != "image") {
        return Ok(None);
    }
    let mut hash = Sha256::new();
    hash.update(b"resin.native-image-preparation.v1");
    hash.update((sources.len() as u64).to_le_bytes());
    for (_, source) in sources {
        let source = source.bind(py);
        if let Ok(value) = source.cast::<PyString>() {
            hash.update([0]);
            hash_field(&mut hash, value.to_str()?.as_bytes());
        } else if let Ok(pair) = source.cast::<PyTuple>()
            && pair.len() == 2
        {
            let data = pair.get_item(0)?;
            let mime = pair.get_item(1)?;
            let (Ok(data), Ok(mime)) = (data.cast::<PyBytes>(), mime.cast::<PyString>()) else {
                return Ok(None);
            };
            hash.update([1]);
            hash_field(&mut hash, mime.to_str()?.as_bytes());
            hash_field(&mut hash, data.as_bytes());
        } else {
            return Ok(None);
        }
    }
    Ok(Some(hash.finalize().into()))
}

fn vector_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity() * size_of::<T>()
}

fn tensor_bytes<T>(tensor: &media::Tensor<T>) -> usize {
    vector_bytes(&tensor.shape) + vector_bytes(&tensor.data)
}

pub(crate) fn config_of(owner: &Bound<'_, PyAny>, name: &str) -> PyResult<Value> {
    let Some(processor) = crate::recipe::optional_attr(owner, name)? else {
        return Ok(json!({}));
    };
    if processor.is_none() || !processor.hasattr("to_dict")? {
        return Ok(json!({}));
    }
    Ok(depythonize(&processor.call_method0("to_dict")?)?)
}
pub(crate) fn process(
    config: &MediaConfig,
    sources: Vec<Source>,
    encoder: &dyn Encoder,
) -> Result<Prepared, Error> {
    let mut expansions = Vec::new();
    if let MediaConfig::Gemma(config) = config {
        let prepared = config.process(&sources)?;
        let image_id = encoder.special_id("<|image|>")?;
        let video_id = encoder.special_id("<|video|>")?;
        let boi = encoder.special_id("<|image>")?;
        let eoi = encoder.special_id("<image|>")?;
        for expansion in &prepared.expansions {
            let mut ids = Vec::new();
            let mut prefix = String::new();
            for (index, frame) in expansion.frames.iter().enumerate() {
                if let Some(timestamp) = &frame.timestamp {
                    let text = format!("{timestamp} ");
                    if index == 0 {
                        prefix = text;
                    } else {
                        ids.extend(encoder.encode_text(&format!(" {text}"))?);
                    }
                }
                ids.push(boi);
                ids.extend(std::iter::repeat_n(
                    if expansion.is_video {
                        video_id
                    } else {
                        image_id
                    },
                    frame.token_count,
                ));
                ids.push(eoi);
            }
            expansions.push(MediaTokens::from_parts(
                if expansion.is_video {
                    "<|video|>"
                } else {
                    "<|image|>"
                },
                ids,
                prefix,
                String::new(),
            ));
        }
        return Ok(Prepared {
            expansions,
            payload: Payload::Gemma(prepared, image_id, video_id),
        });
    }
    let images = sources
        .into_iter()
        .map(|source| match source {
            Source::Image(image) => Ok(image),
            Source::Video(_) => Err(Error::new("this recipe does not support video input")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (counts, marker, payload) = match config {
        MediaConfig::DeepSeek(config, expected) => {
            let id = encoder.special_id("<｜deepseek_image｜>")?;
            if expected.is_some_and(|v| v != id) {
                return Err(Error::new(
                    "tokenizer image token does not match checkpoint image_token_id",
                ));
            }
            let prepared = config.process(&images)?;
            (
                prepared.token_counts(),
                "<｜deepseek_image｜>",
                Payload::DeepSeek(prepared, id),
            )
        }
        MediaConfig::Inkling(config) => {
            let id = encoder.special_id("<|unused_200054|>")?;
            let prepared = config.process(&images)?;
            (
                prepared.token_counts.clone(),
                "<|unused_200054|>",
                Payload::Inkling(prepared, id),
            )
        }
        MediaConfig::Qwen(config) => {
            let id = encoder.special_id("<|image_pad|>")?;
            let prepared = config.process(&images)?;
            (
                prepared.token_counts.clone(),
                "<|image_pad|>",
                Payload::Qwen(prepared, config.merge_size, id),
            )
        }
        MediaConfig::Glm(config) => {
            let id = encoder.special_id("<|image|>")?;
            let prepared = config.process(&images)?;
            (
                prepared.token_counts.clone(),
                "<|image|>",
                Payload::Glm(prepared, config.merge_size, id),
            )
        }
        MediaConfig::Gemma(_) => unreachable!(),
    };
    let id = encoder.special_id(marker)?;
    for count in counts {
        expansions.push(MediaTokens::from_parts(
            marker,
            vec![id; count],
            String::new(),
            String::new(),
        ));
    }
    Ok(Prepared {
        expansions,
        payload,
    })
}
impl Prepared {
    fn retained_bytes(&self) -> usize {
        let payload = match &self.payload {
            Payload::DeepSeek(data, _) => {
                vector_bytes(&data.images)
                    + data
                        .images
                        .iter()
                        .map(|image| tensor_bytes(&image.patches) + vector_bytes(&image.types))
                        .sum::<usize>()
            }
            Payload::Gemma(data, _, _) => {
                data.pixel_values.as_ref().map_or(0, tensor_bytes)
                    + data.image_position_ids.as_ref().map_or(0, tensor_bytes)
                    + data.pixel_values_videos.as_ref().map_or(0, tensor_bytes)
                    + data.video_position_ids.as_ref().map_or(0, tensor_bytes)
                    + vector_bytes(&data.expansions)
                    + data
                        .expansions
                        .iter()
                        .map(|expansion| {
                            vector_bytes(&expansion.frames)
                                + expansion
                                    .frames
                                    .iter()
                                    .map(|frame| {
                                        frame.timestamp.as_ref().map_or(0, String::capacity)
                                    })
                                    .sum::<usize>()
                        })
                        .sum::<usize>()
            }
            Payload::Inkling(data, _) => {
                tensor_bytes(&data.pixel_values) + vector_bytes(&data.token_counts)
            }
            Payload::Qwen(data, _, _) | Payload::Glm(data, _, _) => {
                tensor_bytes(&data.pixel_values)
                    + tensor_bytes(&data.image_grid_thw)
                    + vector_bytes(&data.token_counts)
            }
        };
        size_of::<Self>()
            + vector_bytes(&self.expansions)
            + self
                .expansions
                .iter()
                .map(MediaTokens::heap_bytes)
                .sum::<usize>()
            + payload
    }

    pub(crate) fn bind(
        self,
        py: Python<'_>,
        token_ids: &[u32],
        spans: &[(usize, usize)],
    ) -> PyResult<Py<PyAny>> {
        let tokens = token_ids.to_vec();
        Ok(match self.payload {
            Payload::Glm(data, merge, marker) => {
                let positions = spans.iter().flat_map(|&(p, n)| p..p + n).collect();
                Py::new(
                    py,
                    Glm53FlashMedia::build(
                        py,
                        tokens,
                        positions,
                        data.pixel_values,
                        data.image_grid_thw,
                        merge,
                        marker,
                    )?,
                )?
                .into_any()
            }
            Payload::Inkling(data, marker) => Py::new(
                py,
                InklingMedia::build(py, tokens, marker, Some(data.pixel_values))?,
            )?
            .into_any(),
            Payload::Qwen(data, merge, marker) => {
                let positions = spans.iter().flat_map(|&(p, n)| p..p + n).collect();
                Py::new(
                    py,
                    Qwen35Media::build(
                        py,
                        tokens,
                        positions,
                        data.pixel_values,
                        data.image_grid_thw,
                        merge,
                        marker,
                    )?,
                )?
                .into_any()
            }
            Payload::Gemma(data, image_id, video_id) => {
                let mut types = vec![0u8; tokens.len()];
                for &(position, length) in spans {
                    for i in position..position + length {
                        types[i] = if tokens[i] == image_id {
                            1
                        } else if tokens[i] == video_id {
                            2
                        } else {
                            0
                        };
                    }
                }
                Py::new(
                    py,
                    Gemma4Media::build(
                        py,
                        tokens,
                        types,
                        data.pixel_values,
                        data.image_position_ids,
                        data.pixel_values_videos,
                        data.video_position_ids,
                    )?,
                )?
                .into_any()
            }
            Payload::DeepSeek(data, marker) => {
                let images = data
                    .images
                    .into_iter()
                    .zip(spans)
                    .map(|(image, &(start, _))| {
                        Py::new(
                            py,
                            DeepSeekV41Image::build(
                                py,
                                start,
                                image.patches,
                                image.n_vit_h,
                                image.n_vit_w,
                                image.types,
                            )?,
                        )
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Py::new(py, DeepSeekV41Media::build(py, tokens, images, marker)?)?.into_any()
            }
        })
    }
}

pub(crate) fn image_source(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<RgbImage> {
    if let Ok(value) = source.extract::<String>() {
        return py.detach(|| RgbImage::data_url(&value)).map_err(error);
    }
    if let Ok(pair) = source.cast::<PyTuple>()
        && pair.len() == 2
    {
        let data: Vec<u8> = pair.get_item(0)?.extract()?;
        let mime: String = pair.get_item(1)?.extract()?;
        return py.detach(|| RgbImage::decode(&data, &mime)).map_err(error);
    }
    let array = source
        .extract::<PyReadonlyArrayDyn<'_, f32>>()
        .map_err(|_| error("images must be numpy arrays with dtype float32"))?;
    let shape = array.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(error("images must have RGB/HWC shape [height, width, 3]"));
    }
    RgbImage::new(
        shape[1],
        shape[0],
        array.as_array().iter().copied().collect(),
    )
    .map_err(error)
}
fn video_source(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Video> {
    let (frames, fps, indices) = if let Ok(dict) = source.cast::<PyDict>() {
        (
            dict.get_item("frames")?
                .ok_or_else(|| error("video requires decoded frames"))?,
            dict.get_item("fps")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(24.0),
            dict.get_item("frame_indices")?
                .map(|v| v.extract::<Vec<usize>>())
                .transpose()?,
        )
    } else {
        (source.clone(), 24.0, None)
    };
    if frames.is_instance_of::<pyo3::types::PyString>()
        || frames.is_instance_of::<pyo3::types::PyBytes>()
    {
        return Err(error("video content must contain decoded frames"));
    }
    let mut images = Vec::new();
    for frame in frames.try_iter()? {
        let frame = frame?;
        if let Ok(array) = frame.extract::<PyReadonlyArrayDyn<'_, u8>>() {
            let shape = array.shape();
            if shape.len() != 3 || shape[2] != 3 {
                return Err(error("video frames must be RGB/HWC"));
            }
            images.push(
                RgbImage::new(
                    shape[1],
                    shape[0],
                    array
                        .as_array()
                        .iter()
                        .map(|&v| f32::from(v) / 255.)
                        .collect(),
                )
                .map_err(error)?,
            );
        } else {
            images.push(image_source(py, &frame)?);
        }
    }
    let frame_indices = indices.unwrap_or_else(|| (0..images.len()).collect());
    if images.is_empty() || !fps.is_finite() || fps <= 0.0 || frame_indices.len() != images.len() {
        return Err(error("video frames, fps and frame indices are invalid"));
    }
    Ok(Video {
        frames: images,
        fps,
        frame_indices,
    })
}
pub(crate) fn sources(py: Python<'_>, sources: MediaSources) -> PyResult<Vec<Source>> {
    sources
        .into_iter()
        .map(|(kind, source)| match kind {
            "image" => image_source(py, source.bind(py)).map(Source::Image),
            "video" => video_source(py, source.bind(py)).map(Source::Video),
            _ => Err(error("this recipe does not support audio input")),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared(capacity: usize) -> Arc<Prepared> {
        let mut data = Vec::with_capacity(capacity);
        data.push(1.0);
        Arc::new(Prepared {
            expansions: vec![MediaTokens::from_parts(
                "image",
                vec![1],
                String::new(),
                String::new(),
            )],
            payload: Payload::Inkling(
                media::inkling::Prepared {
                    pixel_values: media::Tensor::new(vec![1], data).unwrap(),
                    token_counts: vec![1],
                },
                1,
            ),
        })
    }

    #[test]
    fn preparation_cache_evicts_least_recently_used_entry() {
        let mut cache = PreparationCache::new(2, usize::MAX);
        let a = prepared(1);
        let b = prepared(1);
        let weak_b = Arc::downgrade(&b);
        cache.insert([1; 32], a.clone());
        cache.insert([2; 32], b);
        assert!(Arc::ptr_eq(&cache.get([1; 32]).unwrap(), &a));
        cache.insert([3; 32], prepared(1));
        assert!(weak_b.upgrade().is_none());
        assert!(cache.get([2; 32]).is_none());
        assert!(cache.get([1; 32]).is_some());
        assert!(cache.get([3; 32]).is_some());
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn preparation_cache_bounds_bytes_and_replacement_does_not_accumulate() {
        let value = prepared(8);
        let bytes = value.retained_bytes() + 2 * size_of::<usize>() + size_of::<CacheEntry>();
        let mut cache = PreparationCache::new(8, bytes);
        cache.insert([1; 32], value.clone());
        cache.insert([1; 32], value.clone());
        assert_eq!(cache.retained_bytes, bytes);
        assert_eq!(cache.entries.len(), 1);
        cache.insert([2; 32], value.clone());
        assert!(cache.get([1; 32]).is_none());
        assert!(cache.get([2; 32]).is_some());
        cache.insert([3; 32], prepared(1024));
        assert!(cache.get([3; 32]).is_none());
        assert_eq!(cache.retained_bytes, bytes);
        let mut disabled = PreparationCache::new(0, usize::MAX);
        disabled.insert([1; 32], value);
        assert_eq!(disabled.retained_bytes, 0);
    }

    #[test]
    fn preparation_cache_accounts_for_tensor_spare_capacity_and_metadata() {
        let small = prepared(1);
        let large = prepared(1024);
        assert!(large.retained_bytes() >= 1024 * size_of::<f32>() + size_of::<Prepared>());
        assert_eq!(
            large.retained_bytes() - small.retained_bytes(),
            1023 * size_of::<f32>()
        );
    }
}
