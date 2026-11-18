//! Native recipe media preparation. Python is only the input/output buffer boundary.

use std::{
    collections::VecDeque,
    mem::size_of,
    sync::{Arc, Mutex, MutexGuard},
};

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict, PyString, PyTuple},
};
use resin_recipe::{
    Encoder, Prompt, Recipe,
    media::{self, ImageSource, Prepared, RgbImage, Source, Video},
};
use sha2::{Digest, Sha256};

use crate::{media_payload::error, recipe_types::MediaSources};

type CacheInfo = [(&'static str, usize); 4];
#[derive(Default)]
pub(crate) struct MediaCache(Mutex<PreparationCache>);

impl MediaCache {
    fn cache(&self) -> MutexGuard<'_, PreparationCache> {
        self.0.lock().unwrap_or_else(|poison| poison.into_inner())
    }
    pub(crate) fn info(&self) -> CacheInfo {
        self.cache().info()
    }
    pub(crate) fn render(
        &self,
        py: Python<'_>,
        recipe: &dyn Recipe,
        prompt: Prompt,
        sources: MediaSources,
        encoder: &dyn Encoder,
        cacheable: bool,
    ) -> PyResult<(Vec<u32>, Py<PyTuple>)> {
        let key = if cacheable {
            cache_key(py, &sources)?
        } else {
            None
        };
        let cached = key.and_then(|key| self.cache().get(key));
        // Cache lookup precedes decoding and resizing.
        let sources = if cached.is_some() {
            Vec::new()
        } else {
            self::sources(py, sources)?
        };
        let (ids, spans, input) = py.detach(|| -> PyResult<_> {
            let prepared = if let Some(prepared) = cached {
                prepared
            } else {
                let prepared = Arc::new(recipe.prepare_media(sources, encoder).map_err(error)?);
                if let Some(key) = key {
                    self.cache().insert(key, prepared.clone());
                }
                prepared
            };
            let (ids, spans) =
                media::encode(prompt, &prepared.expansions, encoder).map_err(error)?;
            // Cached tensors copy outside the GIL; uncached tensors move. Expansions never copy.
            let input = Arc::try_unwrap(prepared).map_or_else(|p| p.items.clone(), |p| p.items);
            Ok((ids, spans, input))
        })?;
        let media = crate::media_payload::bind(py, input, &spans)?;
        Ok((ids, media))
    }
}

type CacheKey = [u8; 32];

struct CacheEntry {
    key: CacheKey,
    prepared: Arc<Prepared>,
    bytes: usize,
}

/// Owned by one immutable native recipe, so processor/tokenizer configuration is
/// implicit in the key. No Python object, source string, or conversation survives
/// in this cache. Image order is part of the key. Each prompt binds its own spans.
struct PreparationCache {
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

    fn get(&mut self, key: CacheKey) -> Option<Arc<Prepared>> {
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

    fn insert(&mut self, key: CacheKey, prepared: Arc<Prepared>) {
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

    fn info(&self) -> CacheInfo {
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
fn cache_key(py: Python<'_>, sources: &MediaSources) -> PyResult<Option<CacheKey>> {
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

fn image_source(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<ImageSource> {
    if let Ok(value) = source.extract::<String>() {
        return py.detach(|| ImageSource::data_url(&value)).map_err(error);
    }
    if let Ok(pair) = source.cast::<PyTuple>()
        && pair.len() == 2
    {
        let data: Vec<u8> = pair.get_item(0)?.extract()?;
        let mime: String = pair.get_item(1)?.extract()?;
        return py
            .detach(|| ImageSource::decode(&data, &mime))
            .map_err(error);
    }
    let array = source
        .extract::<PyReadonlyArrayDyn<'_, f32>>()
        .map_err(|_| error("images must be numpy arrays with dtype float32"))?;
    rgb_array(array, std::convert::identity).map(ImageSource::Rgb)
}

fn rgb_array<T: numpy::Element + Copy>(
    array: PyReadonlyArrayDyn<'_, T>,
    pixel: impl Fn(T) -> f32,
) -> PyResult<RgbImage> {
    let shape = array.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(error("images must have RGB/HWC shape [height, width, 3]"));
    }
    RgbImage::new(
        shape[1],
        shape[0],
        array.as_array().iter().copied().map(pixel).collect(),
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
            images.push(rgb_array(array, |v| f32::from(v) / 255.)?);
        } else {
            let image = image_source(py, &frame)?;
            images.push(py.detach(|| image.into_rgb()).map_err(error)?);
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
fn sources(py: Python<'_>, sources: MediaSources) -> PyResult<Vec<Source>> {
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
        let mut expansions = Vec::with_capacity(capacity);
        expansions.push(media::MediaTokens::from_parts(
            "image",
            vec![1],
            String::new(),
            String::new(),
        ));
        Arc::new(Prepared {
            expansions,
            items: vec![],
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
    fn preparation_cache_accounts_for_spare_capacity_and_metadata() {
        let small = prepared(1);
        let large = prepared(1024);
        assert!(
            large.retained_bytes()
                >= 1024 * size_of::<media::MediaTokens>() + size_of::<Prepared>()
        );
        assert_eq!(
            large.retained_bytes() - small.retained_bytes(),
            1023 * size_of::<media::MediaTokens>()
        );
    }
}
