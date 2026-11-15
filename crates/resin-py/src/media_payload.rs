//! Model inputs owned by Rust, exposed as NumPy buffers without copying into Torch.

use numpy::{
    IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods, ndarray,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyTuple},
};
use resin_recipe::media::{Tensor, qwen};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sha3::{
    Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};

pub(crate) fn error(value: impl ToString) -> PyErr {
    PyValueError::new_err(value.to_string())
}

pub(crate) fn array<T: numpy::Element>(
    py: Python<'_>,
    tensor: Tensor<T>,
) -> PyResult<Py<PyArrayDyn<T>>> {
    let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&tensor.shape), tensor.data)
        .map_err(error)?
        .into_pyarray(py);
    // Content digests and prefix/vision caches describe this snapshot. Exposing
    // writable pixels or geometry would let them diverge from that identity.
    array.readwrite().make_nonwriteable();
    Ok(array.unbind())
}

fn tensor<T: numpy::Element + Copy>(value: &Bound<'_, PyAny>) -> PyResult<Tensor<T>> {
    let array = value.extract::<PyReadonlyArrayDyn<'_, T>>()?;
    Tensor::new(
        array.shape().to_vec(),
        array.as_array().iter().copied().collect(),
    )
    .map_err(error)
}

fn tuple<'py, T: IntoPyObject<'py>, I: IntoIterator<Item = T>>(
    py: Python<'py>,
    values: I,
) -> PyResult<Py<PyTuple>>
where
    I::IntoIter: ExactSizeIterator,
{
    Ok(PyTuple::new(py, values)?.unbind())
}
fn spans(
    py: Python<'_>,
    values: impl IntoIterator<Item = (usize, usize)>,
) -> PyResult<Py<PyTuple>> {
    let class = py.import("resin.protocols")?.getattr("MediaTokenSpan")?;
    let items = values
        .into_iter()
        .map(|span| class.call1(span))
        .collect::<PyResult<Vec<_>>>()?;
    tuple(py, items)
}

struct Identity {
    tokens: Vec<u32>,
    digest: String,
    keys: Vec<u32>,
}
impl Identity {
    fn new(tokens: Vec<u32>, positions: &[usize], hash: Sha256) -> PyResult<Self> {
        if tokens.is_empty() || positions.iter().any(|&p| p >= tokens.len()) {
            return Err(error("media positions must lie inside a nonempty prompt"));
        }
        let digest = hash.finalize();
        let mut salt = Shake256::default();
        Update::update(&mut salt, &digest);
        let mut stream = salt.finalize_xof();
        let mut keys = tokens.clone();
        for &position in positions {
            let mut bytes = [0; 4];
            stream.read(&mut bytes);
            keys[position] = u32::from_le_bytes(bytes);
        }
        Ok(Self {
            tokens,
            digest: digest.iter().map(|b| format!("{b:02x}")).collect(),
            keys,
        })
    }
}
fn hash_tensor<T: bytemuck::Pod>(hash: &mut Sha256, name: &str, tensor: &Tensor<T>) {
    Digest::update(hash, name.as_bytes());
    Digest::update(hash, (tensor.shape.len() as u64).to_le_bytes());
    for &d in &tensor.shape {
        Digest::update(hash, (d as u64).to_le_bytes());
    }
    Digest::update(hash, bytemuck::cast_slice(&tensor.data));
}

// Every payload exposes the same text/cache identity; model-specific methods
// remain alongside each constructor below.
macro_rules! media_methods {
    (impl $class:ident { $($methods:tt)* }) => {
        #[pymethods]
        impl $class {
            #[getter]
            fn token_ids(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
                tuple(py, self.identity.tokens.iter().copied())
            }
            #[getter]
            fn digest(&self) -> &str {
                &self.identity.digest
            }
            #[getter]
            fn cache_key(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
                tuple(py, self.identity.keys.iter().copied())
            }
            $($methods)*
        }
    };
}

#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct InklingMedia {
    identity: Identity,
    #[pyo3(get, name = "image_token_id")]
    marker: u32,
    positions: Vec<usize>,
    #[pyo3(get, name = "pixel_values")]
    pixels: Option<Py<PyArrayDyn<f32>>>,
}
media_methods! {
    impl InklingMedia {
        #[new]
        #[pyo3(signature=(*, token_ids, image_token_id, pixel_values=None))]
        fn new(
            py: Python<'_>,
            token_ids: Vec<u32>,
            image_token_id: u32,
            pixel_values: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            Self::build(
                py,
                token_ids,
                image_token_id,
                pixel_values.map(tensor).transpose()?,
            )
        }
        #[getter]
        fn image_positions(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            tuple(py, self.positions.iter().copied())
        }
        #[getter]
        fn token_spans(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            spans(py, [])
        }
    }
}
impl InklingMedia {
    pub(crate) fn build(
        py: Python<'_>,
        tokens: Vec<u32>,
        marker: u32,
        pixels: Option<Tensor<f32>>,
    ) -> PyResult<Self> {
        let positions = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == marker).then_some(i))
            .collect::<Vec<_>>();
        let mut hash = Sha256::new();
        Digest::update(&mut hash, b"resin.inkling.media.v2");
        let pixels = if let Some(pixels) = pixels {
            if pixels.shape.len() != 5 || pixels.shape[0] != positions.len() {
                return Err(error(
                    "Inkling image placeholder count does not match pixel patches",
                ));
            }
            hash_tensor(&mut hash, "pixels", &pixels);
            Some(array(py, pixels)?)
        } else {
            if !positions.is_empty() {
                return Err(error("Inkling image placeholders require pixel_values"));
            }
            None
        };
        Ok(Self {
            identity: Identity::new(tokens, &positions, hash)?,
            marker,
            positions,
            pixels,
        })
    }
}

#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct Qwen35Media {
    identity: Identity,
    positions: Vec<usize>,
    #[pyo3(get, name = "image_token_id")]
    marker: u32,
    #[pyo3(get, name = "spatial_merge_size")]
    merge: usize,
    #[pyo3(get, name = "pixel_values")]
    pixels: Py<PyArrayDyn<f32>>,
    #[pyo3(get, name = "image_grid_thw")]
    grids: Py<PyArrayDyn<i64>>,
    #[pyo3(get, name = "position_ids")]
    rope: Py<PyArrayDyn<i64>>,
    #[pyo3(get, name = "rope_delta")]
    delta: i64,
}
media_methods! {
    impl Qwen35Media {
        #[new]
        #[pyo3(signature=(*, token_ids, image_positions, pixel_values, image_grid_thw, spatial_merge_size, image_token_id))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            py: Python<'_>,
            token_ids: Vec<u32>,
            image_positions: Vec<usize>,
            pixel_values: &Bound<'_, PyAny>,
            image_grid_thw: &Bound<'_, PyAny>,
            spatial_merge_size: usize,
            image_token_id: u32,
        ) -> PyResult<Self> {
            Self::build(
                py,
                token_ids,
                image_positions,
                tensor(pixel_values)?,
                tensor(image_grid_thw)?,
                spatial_merge_size,
                image_token_id,
            )
        }
        #[getter]
        fn image_positions(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            tuple(py, self.positions.iter().copied())
        }
        #[getter]
        fn token_spans(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            spans(py, [])
        }
    }
}

#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct Glm53FlashMedia {
    identity: Identity,
    positions: Vec<usize>,
    #[pyo3(get, name = "image_token_id")]
    marker: u32,
    #[pyo3(get, name = "spatial_merge_size")]
    merge: usize,
    #[pyo3(get, name = "pixel_values")]
    pixels: Py<PyArrayDyn<f32>>,
    #[pyo3(get, name = "image_grid_thw")]
    grids: Py<PyArrayDyn<i64>>,
}
media_methods! {
    impl Glm53FlashMedia {
        #[new]
        #[pyo3(signature=(*, token_ids, image_positions, pixel_values, image_grid_thw, spatial_merge_size, image_token_id))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            py: Python<'_>,
            token_ids: Vec<u32>,
            image_positions: Vec<usize>,
            pixel_values: &Bound<'_, PyAny>,
            image_grid_thw: &Bound<'_, PyAny>,
            spatial_merge_size: usize,
            image_token_id: u32,
        ) -> PyResult<Self> {
            Self::build(
                py,
                token_ids,
                image_positions,
                tensor(pixel_values)?,
                tensor(image_grid_thw)?,
                spatial_merge_size,
                image_token_id,
            )
        }
        #[getter]
        fn image_positions(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            tuple(py, self.positions.iter().copied())
        }
        #[getter]
        fn token_spans(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            spans(py, [])
        }
    }
}

#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct Gemma4Media {
    identity: Identity,
    /// Image/video tensor identity, independent of conversational text positions.
    #[pyo3(get)]
    vision_digest: String,
    types: Vec<u8>,
    #[pyo3(get, name = "pixel_values")]
    pixels: Option<Py<PyArrayDyn<f32>>>,
    #[pyo3(get, name = "image_position_ids")]
    positions: Option<Py<PyArrayDyn<i64>>>,
    #[pyo3(get, name = "pixel_values_videos")]
    video_pixels: Option<Py<PyArrayDyn<f32>>>,
    #[pyo3(get, name = "video_position_ids")]
    video_positions: Option<Py<PyArrayDyn<i64>>>,
}
media_methods! {
    impl Gemma4Media {
        #[new]
        #[pyo3(signature=(*, token_ids, mm_token_type_ids, pixel_values=None, image_position_ids=None, pixel_values_videos=None, video_position_ids=None))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            py: Python<'_>,
            token_ids: Vec<u32>,
            mm_token_type_ids: Vec<u8>,
            pixel_values: Option<&Bound<'_, PyAny>>,
            image_position_ids: Option<&Bound<'_, PyAny>>,
            pixel_values_videos: Option<&Bound<'_, PyAny>>,
            video_position_ids: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Self> {
            Self::build(
                py,
                token_ids,
                mm_token_type_ids,
                pixel_values.map(tensor).transpose()?,
                image_position_ids.map(tensor).transpose()?,
                pixel_values_videos.map(tensor).transpose()?,
                video_position_ids.map(tensor).transpose()?,
            )
        }
        #[getter]
        fn mm_token_type_ids(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            tuple(py, self.types.iter().copied())
        }
        #[getter]
        fn image_positions(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            tuple(py, self.positions_of(1))
        }
        #[getter]
        fn video_positions(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            tuple(py, self.positions_of(2))
        }
        #[getter]
        fn block_ids(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            let mut block = -1i64;
            let mut previous = false;
            tuple(
                py,
                self.types.iter().map(|&kind| {
                    let vision = kind != 0;
                    if vision && !previous {
                        block += 1;
                    }
                    previous = vision;
                    if vision { block } else { -1 }
                }),
            )
        }
        #[getter]
        fn token_spans(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            spans(
                py,
                contiguous_ranges(
                    &self
                        .types
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &t)| (t != 0).then_some(i))
                        .collect::<Vec<_>>(),
                ),
            )
        }
        #[pyo3(signature=(*,image_token_id,video_token_id))]
        fn validate_placeholder_tokens(
            &self,
            image_token_id: u32,
            video_token_id: u32,
        ) -> PyResult<()> {
            for (&token, &kind) in self.identity.tokens.iter().zip(&self.types) {
                if kind != 0
                    && token
                        != if kind == 1 {
                            image_token_id
                        } else {
                            video_token_id
                        }
                {
                    return Err(error(
                        "media token type does not match its placeholder token",
                    ));
                }
            }
            Ok(())
        }
    }
}
impl Gemma4Media {
    fn positions_of(&self, kind: u8) -> Vec<usize> {
        self.types
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == kind).then_some(i))
            .collect()
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        py: Python<'_>,
        tokens: Vec<u32>,
        types: Vec<u8>,
        pixels: Option<Tensor<f32>>,
        positions: Option<Tensor<i64>>,
        video_pixels: Option<Tensor<f32>>,
        video_positions: Option<Tensor<i64>>,
    ) -> PyResult<Self> {
        if tokens.len() != types.len() || types.iter().any(|&t| t > 2) {
            return Err(error(
                "media token types must match the prompt and be 0, 1 or 2",
            ));
        }
        if pixels.is_some() != positions.is_some()
            || video_pixels.is_some() != video_positions.is_some()
        {
            return Err(error(
                "pixel values and position IDs must be supplied together",
            ));
        }
        if types.contains(&1) && pixels.is_none() || types.contains(&2) && video_pixels.is_none() {
            return Err(error("media placeholders require pixel_values"));
        }
        let mut hash = Sha256::new();
        Digest::update(&mut hash, b"resin.gemma.media.v2");
        Digest::update(&mut hash, &types);
        let mut vision_hash = Sha256::new();
        Digest::update(&mut vision_hash, b"resin.gemma.vision.v1");
        for (name, tensor, rank) in [
            ("pixel_values", &pixels, 3),
            ("pixel_values_videos", &video_pixels, 4),
        ] {
            if let Some(t) = tensor {
                if t.shape.len() != rank {
                    return Err(error(format!("{name} must have rank {rank}")));
                }
                hash_tensor(&mut hash, name, t);
                hash_tensor(&mut vision_hash, name, t);
            }
        }
        for (name, tensor, rank) in [
            ("image_position_ids", &positions, 3),
            ("video_position_ids", &video_positions, 4),
        ] {
            if let Some(t) = tensor {
                if t.shape.len() != rank {
                    return Err(error(format!("{name} must have rank {rank}")));
                }
                hash_tensor(&mut hash, name, t);
                hash_tensor(&mut vision_hash, name, t);
            }
        }
        let visual = types
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t != 0).then_some(i))
            .collect::<Vec<_>>();
        Ok(Self {
            identity: Identity::new(tokens, &visual, hash)?,
            vision_digest: vision_hash
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
            types,
            pixels: pixels.map(|p| array(py, p)).transpose()?,
            positions: positions.map(|p| array(py, p)).transpose()?,
            video_pixels: video_pixels.map(|p| array(py, p)).transpose()?,
            video_positions: video_positions.map(|p| array(py, p)).transpose()?,
        })
    }
}

#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct DeepSeekV41Image {
    #[pyo3(get, name = "start")]
    start: usize,
    #[pyo3(get, name = "patches")]
    patches: Py<PyArrayDyn<f32>>,
    #[pyo3(get, name = "n_vit_h")]
    height: usize,
    #[pyo3(get, name = "n_vit_w")]
    width: usize,
    types: Vec<u32>,
}
#[pymethods]
impl DeepSeekV41Image {
    #[new]
    #[pyo3(signature=(*, start, patches, n_vit_h, n_vit_w, types))]
    fn new(
        py: Python<'_>,
        start: usize,
        patches: &Bound<'_, PyAny>,
        n_vit_h: usize,
        n_vit_w: usize,
        types: Vec<u32>,
    ) -> PyResult<Self> {
        Self::build(py, start, tensor(patches)?, n_vit_h, n_vit_w, types)
    }
    #[getter]
    fn types(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        tuple(py, self.types.iter().copied())
    }
}
impl DeepSeekV41Image {
    pub(crate) fn build(
        py: Python<'_>,
        start: usize,
        patches: Tensor<f32>,
        height: usize,
        width: usize,
        types: Vec<u32>,
    ) -> PyResult<Self> {
        if height == 0
            || width == 0
            || height.checked_mul(width).is_none()
            || patches.shape != [height * width, 3, 14, 14]
        {
            return Err(error("image patches must be [grid_h * grid_w, 3, 14, 14]"));
        }
        if types
            != resin_recipe::media::deepseek::image_token_types(
                height.div_ceil(3),
                width.div_ceil(3),
            )
        {
            return Err(error("image token types do not match the 3x3 aligner grid"));
        }
        Ok(Self {
            start,
            patches: array(py, patches)?,
            height,
            width,
            types,
        })
    }
}
#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct DeepSeekV41Media {
    identity: Identity,
    images: Vec<Py<DeepSeekV41Image>>,
    #[pyo3(get, name = "image_token_id")]
    marker: u32,
}
media_methods! {
    impl DeepSeekV41Media {
        #[new]
        #[pyo3(signature=(*, token_ids, images, image_token_id=129264))]
        fn new(
            py: Python<'_>,
            token_ids: Vec<u32>,
            images: Vec<Py<DeepSeekV41Image>>,
            image_token_id: u32,
        ) -> PyResult<Self> {
            Self::build(py, token_ids, images, image_token_id)
        }
        #[getter]
        fn images(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            tuple(py, self.images.iter().map(|p| p.clone_ref(py)))
        }
        #[getter]
        fn token_spans(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
            spans(py, [])
        }
    }
}
impl DeepSeekV41Media {
    pub(crate) fn build(
        py: Python<'_>,
        tokens: Vec<u32>,
        images: Vec<Py<DeepSeekV41Image>>,
        marker: u32,
    ) -> PyResult<Self> {
        if images.is_empty() {
            return Err(error("image media requires nonempty images"));
        }
        let mut positions = Vec::new();
        let mut end = 0;
        let mut hash = Sha256::new();
        Digest::update(&mut hash, b"resin.deepseek.media.v2");
        for image in &images {
            let image = image.borrow(py);
            let next = image
                .start
                .checked_add(image.types.len())
                .ok_or_else(|| error("image span overflow"))?;
            if image.start < end || next > tokens.len() {
                return Err(error(
                    "image spans must be ordered, disjoint and inside the prompt",
                ));
            }
            end = next;
            if tokens[image.start..end].iter().any(|&t| t != marker) {
                return Err(error(
                    "image span contains a different placeholder token ID",
                ));
            }
            positions.extend(image.start..end);
            Digest::update(&mut hash, (image.start as u64).to_le_bytes());
            Digest::update(&mut hash, (image.height as u64).to_le_bytes());
            Digest::update(&mut hash, (image.width as u64).to_le_bytes());
            let pixels = image.patches.bind(py).readonly();
            Digest::update(&mut hash, bytemuck::cast_slice(pixels.as_slice()?));
        }
        if tokens.iter().filter(|&&t| t == marker).count() != positions.len() {
            return Err(error(
                "image placeholder token appears outside an image span",
            ));
        }
        Ok(Self {
            identity: Identity::new(tokens, &positions, hash)?,
            images,
            marker,
        })
    }
}
fn validate_packed_images(
    tokens: &[u32],
    positions: &[usize],
    pixels: &Tensor<f32>,
    grids: &Tensor<i64>,
    merge: usize,
    marker: u32,
    family: &str,
) -> PyResult<Vec<(usize, usize)>> {
    if positions.is_empty()
        || positions.iter().any(|&p| tokens.get(p) != Some(&marker))
        || positions.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(error(format!(
            "{family} image placeholders do not match image token IDs"
        )));
    }
    if pixels.shape.len() != 2 {
        return Err(error("pixel_values must have rank 2"));
    }
    if merge == 0 || grids.shape.len() != 2 || grids.shape[1] != 3 {
        return Err(error(format!(
            "{family} image grids must have shape [images, 3] and a positive merge size"
        )));
    }
    // Grids retain image boundaries even when two expansions are adjacent.
    let mut ranges = Vec::new();
    let mut remaining = positions;
    let mut patch_count = 0usize;
    for grid in grids.data.as_chunks::<3>().0 {
        let dims = grid.map(|d| usize::try_from(d).ok().filter(|&d| d > 0));
        let [Some(t), Some(h), Some(w)] = dims else {
            return Err(error(format!("{family} image grids must be positive")));
        };
        if !h.is_multiple_of(merge) || !w.is_multiple_of(merge) {
            return Err(error(format!(
                "{family} image grid must be divisible by spatial merge size"
            )));
        }
        patch_count = t
            .checked_mul(h)
            .and_then(|n| n.checked_mul(w))
            .and_then(|n| patch_count.checked_add(n))
            .ok_or_else(|| error(format!("{family} image grid overflows")))?;
        let count = t
            .checked_mul(h / merge)
            .and_then(|n| n.checked_mul(w / merge))
            .filter(|&n| n <= remaining.len())
            .ok_or_else(|| {
                error(format!(
                    "{family} image grid does not match its placeholder range"
                ))
            })?;
        let (image, tail) = remaining.split_at(count);
        if image.windows(2).any(|p| p[1] != p[0] + 1) {
            return Err(error(format!(
                "{family} image placeholders must be contiguous within each image"
            )));
        }
        ranges.push((image[0], count));
        remaining = tail;
    }
    if !remaining.is_empty() || patch_count != pixels.shape[0] {
        return Err(error(format!(
            "{family} image grids do not match pixels and placeholders"
        )));
    }
    Ok(ranges)
}
impl Qwen35Media {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        py: Python<'_>,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        pixels: Tensor<f32>,
        grids: Tensor<i64>,
        merge: usize,
        marker: u32,
    ) -> PyResult<Self> {
        let ranges =
            validate_packed_images(&tokens, &positions, &pixels, &grids, merge, marker, "Qwen")?;
        let (rope, delta) =
            qwen::mrope_positions(tokens.len(), &ranges, &grids, merge).map_err(error)?;
        let mut hash = Sha256::new();
        Digest::update(&mut hash, b"resin.qwen.media.v2");
        hash_tensor(&mut hash, "pixels", &pixels);
        hash_tensor(&mut hash, "grids", &grids);
        Ok(Self {
            identity: Identity::new(tokens, &positions, hash)?,
            positions,
            marker,
            merge,
            pixels: array(py, pixels)?,
            grids: array(py, grids)?,
            rope: array(py, rope)?,
            delta,
        })
    }
}
impl Glm53FlashMedia {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        py: Python<'_>,
        tokens: Vec<u32>,
        positions: Vec<usize>,
        pixels: Tensor<f32>,
        grids: Tensor<i64>,
        merge: usize,
        marker: u32,
    ) -> PyResult<Self> {
        validate_packed_images(&tokens, &positions, &pixels, &grids, merge, marker, "GLM")?;
        if grids
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .any(|grid| grid[0] != 1)
        {
            return Err(error("GLM image payload supports still images only"));
        }
        let mut hash = Sha256::new();
        Digest::update(&mut hash, b"resin.glm53flash.media.v1");
        hash_tensor(&mut hash, "pixels", &pixels);
        hash_tensor(&mut hash, "grids", &grids);
        Ok(Self {
            identity: Identity::new(tokens, &positions, hash)?,
            positions,
            marker,
            merge,
            pixels: array(py, pixels)?,
            grids: array(py, grids)?,
        })
    }
}
pub(crate) fn contiguous_ranges(positions: &[usize]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for &position in positions {
        if let Some((start, length)) = result.last_mut()
            && *start + *length == position
        {
            *length += 1;
            continue;
        }
        result.push((position, 1));
    }
    result
}

/// Request registration metadata. Tensor storage travels separately, without
/// serialization into Python objects or bytes. Tags identify fixed native types.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MediaTransfer {
    version: u8,
    media: MediaDescriptor,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum MediaDescriptor {
    Inkling {
        marker: u32,
        pixels: bool,
    },
    Qwen {
        positions: Vec<usize>,
        merge: usize,
        marker: u32,
    },
    Glm {
        positions: Vec<usize>,
        merge: usize,
        marker: u32,
    },
    Gemma {
        images: bool,
        videos: bool,
    },
    DeepSeek {
        marker: u32,
        images: Vec<ImageDescriptor>,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageDescriptor {
    start: usize,
    height: usize,
    width: usize,
    types: Vec<u32>,
}

fn push_buffer<T: numpy::Element>(
    py: Python<'_>,
    buffers: &mut Vec<Py<PyAny>>,
    value: &Py<PyArrayDyn<T>>,
) {
    buffers.push(value.clone_ref(py).into_any());
}

fn push_values<T: numpy::Element>(
    owner: &Bound<'_, PyAny>,
    buffers: &mut Vec<Py<PyAny>>,
    values: &[T],
) {
    buffers.push(
        crate::arrays::readonly_view(owner.clone(), values)
            .into_any()
            .unbind(),
    );
}

#[pyfunction]
#[pyo3(signature = (media: "MediaInput") -> "tuple[bytes, tuple[npt.NDArray[Any], ...]]")]
pub(crate) fn _media_to_buffers(
    py: Python<'_>,
    media: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyBytes>, Py<PyTuple>)> {
    let mut buffers = Vec::new();
    let descriptor = if let Ok(value) = media.cast::<InklingMedia>() {
        let value = value.borrow();
        push_values(media, &mut buffers, &value.identity.tokens);
        if let Some(pixels) = &value.pixels {
            push_buffer(py, &mut buffers, pixels);
        }
        MediaDescriptor::Inkling {
            marker: value.marker,
            pixels: value.pixels.is_some(),
        }
    } else if let Ok(value) = media.cast::<Qwen35Media>() {
        let value = value.borrow();
        push_values(media, &mut buffers, &value.identity.tokens);
        push_buffer(py, &mut buffers, &value.pixels);
        push_buffer(py, &mut buffers, &value.grids);
        MediaDescriptor::Qwen {
            positions: value.positions.clone(),
            merge: value.merge,
            marker: value.marker,
        }
    } else if let Ok(value) = media.cast::<Glm53FlashMedia>() {
        let value = value.borrow();
        push_values(media, &mut buffers, &value.identity.tokens);
        push_buffer(py, &mut buffers, &value.pixels);
        push_buffer(py, &mut buffers, &value.grids);
        MediaDescriptor::Glm {
            positions: value.positions.clone(),
            merge: value.merge,
            marker: value.marker,
        }
    } else if let Ok(value) = media.cast::<Gemma4Media>() {
        let value = value.borrow();
        push_values(media, &mut buffers, &value.identity.tokens);
        push_values(media, &mut buffers, &value.types);
        if let (Some(pixels), Some(positions)) = (&value.pixels, &value.positions) {
            push_buffer(py, &mut buffers, pixels);
            push_buffer(py, &mut buffers, positions);
        }
        if let (Some(pixels), Some(positions)) = (&value.video_pixels, &value.video_positions) {
            push_buffer(py, &mut buffers, pixels);
            push_buffer(py, &mut buffers, positions);
        }
        MediaDescriptor::Gemma {
            images: value.pixels.is_some(),
            videos: value.video_pixels.is_some(),
        }
    } else if let Ok(value) = media.cast::<DeepSeekV41Media>() {
        let value = value.borrow();
        push_values(media, &mut buffers, &value.identity.tokens);
        let images = value
            .images
            .iter()
            .map(|image| {
                let image = image.borrow(py);
                push_buffer(py, &mut buffers, &image.patches);
                ImageDescriptor {
                    start: image.start,
                    height: image.height,
                    width: image.width,
                    types: image.types.clone(),
                }
            })
            .collect();
        MediaDescriptor::DeepSeek {
            marker: value.marker,
            images,
        }
    } else {
        return Err(PyTypeError::new_err(
            "distributed media must be a native Resin media payload; custom media is supported only for local execution",
        ));
    };
    let metadata = serde_json::to_vec(&MediaTransfer {
        version: 1,
        media: descriptor,
    })
    .map_err(error)?;
    Ok((PyBytes::new(py, &metadata).unbind(), tuple(py, buffers)?))
}

fn take_buffer<T: numpy::Element + Copy>(
    py: Python<'_>,
    buffers: &mut impl Iterator<Item = Py<PyAny>>,
) -> PyResult<Tensor<T>> {
    let value = buffers
        .next()
        .ok_or_else(|| error("media transfer is missing a tensor buffer"))?;
    tensor(value.bind(py))
}

#[pyfunction]
#[pyo3(signature = (metadata, buffers: "Sequence[npt.NDArray[Any]]") -> "MediaInput")]
pub(crate) fn _media_from_buffers(
    py: Python<'_>,
    metadata: &[u8],
    buffers: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let descriptor: MediaTransfer = serde_json::from_slice(metadata).map_err(error)?;
    if descriptor.version != 1 {
        return Err(error("unsupported media transfer version"));
    }
    let mut buffers = buffers.into_iter();
    let tokens = take_buffer::<u32>(py, &mut buffers)?;
    if tokens.shape.len() != 1 {
        return Err(error("media transfer token IDs must have rank 1"));
    }
    let tokens = tokens.data;
    let media = match descriptor.media {
        MediaDescriptor::Inkling { marker, pixels } => {
            let pixels = pixels.then(|| take_buffer(py, &mut buffers)).transpose()?;
            Py::new(py, InklingMedia::build(py, tokens, marker, pixels)?)?.into_any()
        }
        MediaDescriptor::Qwen {
            positions,
            merge,
            marker,
        } => {
            let pixels = take_buffer(py, &mut buffers)?;
            let grids = take_buffer(py, &mut buffers)?;
            Py::new(
                py,
                Qwen35Media::build(py, tokens, positions, pixels, grids, merge, marker)?,
            )?
            .into_any()
        }
        MediaDescriptor::Glm {
            positions,
            merge,
            marker,
        } => {
            let pixels = take_buffer(py, &mut buffers)?;
            let grids = take_buffer(py, &mut buffers)?;
            Py::new(
                py,
                Glm53FlashMedia::build(py, tokens, positions, pixels, grids, merge, marker)?,
            )?
            .into_any()
        }
        MediaDescriptor::Gemma { images, videos } => {
            let types = take_buffer::<u8>(py, &mut buffers)?;
            if types.shape.len() != 1 {
                return Err(error("media transfer token types must have rank 1"));
            }
            let pixels = images.then(|| take_buffer(py, &mut buffers)).transpose()?;
            let positions = images.then(|| take_buffer(py, &mut buffers)).transpose()?;
            let video_pixels = videos.then(|| take_buffer(py, &mut buffers)).transpose()?;
            let video_positions = videos.then(|| take_buffer(py, &mut buffers)).transpose()?;
            Py::new(
                py,
                Gemma4Media::build(
                    py,
                    tokens,
                    types.data,
                    pixels,
                    positions,
                    video_pixels,
                    video_positions,
                )?,
            )?
            .into_any()
        }
        MediaDescriptor::DeepSeek { marker, images } => {
            let images = images
                .into_iter()
                .map(|image| {
                    let patches = take_buffer(py, &mut buffers)?;
                    Py::new(
                        py,
                        DeepSeekV41Image::build(
                            py,
                            image.start,
                            patches,
                            image.height,
                            image.width,
                            image.types,
                        )?,
                    )
                })
                .collect::<PyResult<Vec<_>>>()?;
            Py::new(py, DeepSeekV41Media::build(py, tokens, images, marker)?)?.into_any()
        }
    };
    if buffers.next().is_some() {
        return Err(error("media transfer has unexpected tensor buffers"));
    }
    Ok(media)
}
