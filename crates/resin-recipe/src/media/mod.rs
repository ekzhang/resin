//! CPU media preprocessing shared by native recipes. Pixels are RGB/HWC in [0, 1].

use base64::Engine;
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb, imageops::FilterType};

use crate::Error;

mod prompt;
pub use prompt::{MediaTokens, encode};

/// Resized RGB bitmaps and their placeholder ranges within one source expansion.
#[derive(Clone)]
pub enum Input {
    Image {
        pixels: Tensor<f32>,
        span: (usize, usize),
        atomic: bool,
    },
    Video {
        frames: Vec<Tensor<f32>>,
        frame_spans: Vec<(usize, usize)>,
        fps: f64,
        frame_indices: Vec<usize>,
        atomic: bool,
    },
}
impl Input {
    fn heap_bytes(&self) -> usize {
        match self {
            Self::Image { pixels, .. } => pixels.heap_bytes(),
            Self::Video {
                frames,
                frame_spans,
                frame_indices,
                ..
            } => {
                vector_bytes(frames)
                    + vector_bytes(frame_spans)
                    + vector_bytes(frame_indices)
                    + frames.iter().map(Tensor::heap_bytes).sum::<usize>()
            }
        }
    }
}

/// Model-independent media and token expansions. Tensor packing belongs to runners.
#[derive(Clone)]
pub struct Prepared {
    pub expansions: Vec<MediaTokens>,
    pub items: Vec<Input>,
}
impl Prepared {
    pub fn new(expansions: Vec<MediaTokens>, items: Vec<Input>) -> Self {
        Self { expansions, items }
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + vector_bytes(&self.expansions)
            + vector_bytes(&self.items)
            + self
                .expansions
                .iter()
                .map(MediaTokens::heap_bytes)
                .sum::<usize>()
            + self.items.iter().map(Input::heap_bytes).sum::<usize>()
    }
}

pub(crate) fn bitmap(
    mut image: RgbImage,
    normalize: bool,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Tensor<f32>, Error> {
    if normalize {
        for (index, pixel) in image.data.iter_mut().enumerate() {
            *pixel = (*pixel - mean[index % 3]) / std[index % 3];
        }
    }
    Tensor::new(vec![image.height, image.width, 3], image.data)
}

pub(crate) fn vector_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity() * std::mem::size_of::<T>()
}

pub(crate) fn images(sources: Vec<Source>) -> Result<Vec<ImageSource>, Error> {
    sources
        .into_iter()
        .map(|source| match source {
            Source::Image(image) => Ok(image),
            Source::Video(_) => Err(Error::new("this recipe does not support video input")),
        })
        .collect()
}

pub(crate) fn image_tokens(marker: &str, id: u32, count: usize) -> MediaTokens {
    MediaTokens::from_parts(marker, vec![id; count], String::new(), String::new())
}

#[derive(Clone, Debug)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn heap_bytes(&self) -> usize {
        vector_bytes(&self.shape) + vector_bytes(&self.data)
    }

    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Result<Self, Error> {
        if shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(data.len()) {
            return Err(Error::new("media tensor shape does not match its storage"));
        }
        Ok(Self { shape, data })
    }
}

#[derive(Clone, Debug)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Result<Self, Error> {
        if width == 0
            || height == 0
            || width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(data.len())
        {
            return Err(Error::new(
                "images must have nonempty RGB/HWC shape [height, width, 3]",
            ));
        }
        if data
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(Error::new("images must contain finite values in [0, 1]"));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn resize(&self, width: usize, height: usize, filter: FilterType) -> Result<Self, Error> {
        let dims = [self.width, self.height, width, height].map(u32::try_from);
        let [Ok(sw), Ok(sh), Ok(dw), Ok(dh)] = dims else {
            return Err(Error::new("image dimensions exceed u32"));
        };
        if dw == 0 || dh == 0 {
            return Err(Error::new("resized image dimensions must be positive"));
        }
        if (sw, sh) == (dw, dh) {
            return Ok(self.clone());
        }
        let source = ImageBuffer::<Rgb<f32>, _>::from_raw(sw, sh, self.data.as_slice())
            .ok_or_else(|| Error::new("invalid RGB image storage"))?;
        let resized = image::imageops::resize(&source, dw, dh, filter);
        Ok(Self {
            width,
            height,
            data: resized.into_raw(),
        })
    }
}

/// A still image before the model chooses how to convert its color and alpha.
#[derive(Clone, Debug)]
pub enum ImageSource {
    Rgb(RgbImage),
    Decoded(DynamicImage),
}

impl ImageSource {
    pub fn decode(data: &[u8], media_type: &str) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::new("image bytes must not be empty"));
        }
        let format = match media_type.to_ascii_lowercase().as_str() {
            "image/png" => ImageFormat::Png,
            "image/jpeg" => ImageFormat::Jpeg,
            "image/webp" => ImageFormat::WebP,
            "image/gif" => ImageFormat::Gif,
            "image/bmp" => ImageFormat::Bmp,
            "image/tiff" => ImageFormat::Tiff,
            _ => {
                return Err(Error::new(format!(
                    "unsupported image media type: {media_type:?}"
                )));
            }
        };
        let decoded = image::load_from_memory_with_format(data, format)
            .map_err(|error| Error::new(format!("invalid image: {error}")))?;
        Ok(Self::Decoded(decoded))
    }

    pub fn data_url(source: &str) -> Result<Self, Error> {
        let (header, encoded) = source.split_once(",").ok_or_else(|| {
            Error::new("image URLs must be base64 data URLs; remote fetching is disabled")
        })?;
        let media_type = header
            .strip_prefix("data:")
            .and_then(|s| s.strip_suffix(";base64"))
            .ok_or_else(|| {
                Error::new("image URLs must be base64 data URLs; remote fetching is disabled")
            })?;
        let data = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| Error::new("image data URL contains invalid base64"))?;
        Self::decode(&data, media_type)
    }

    pub fn into_rgb(self) -> Result<RgbImage, Error> {
        match self {
            Self::Rgb(image) => Ok(image),
            Self::Decoded(image) => {
                let rgb = image.into_rgb32f();
                RgbImage::new(rgb.width() as usize, rgb.height() as usize, rgb.into_raw())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Video {
    pub frames: Vec<RgbImage>,
    pub fps: f64,
    pub frame_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum Source {
    Image(ImageSource),
    Video(Video),
}

pub fn resize_filter(value: u32) -> Result<FilterType, Error> {
    match value {
        0 => Ok(FilterType::Nearest),
        1 => Ok(FilterType::Lanczos3),
        2 => Ok(FilterType::Triangle),
        3 => Ok(FilterType::CatmullRom),
        _ => Err(Error::new("unsupported image resampling filter")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    pub fn pixels(prepared: &super::Prepared, index: usize) -> &super::Tensor<f32> {
        match &prepared.items[index] {
            super::Input::Image { pixels, .. } => pixels,
            _ => panic!("expected image"),
        }
    }
    pub struct Encoder;
    impl crate::Encoder for Encoder {
        fn encode_text(&self, text: &str) -> Result<Vec<u32>, crate::Error> {
            Ok(text.bytes().map(u32::from).collect())
        }
        fn special_id(&self, token: &str) -> Result<u32, crate::Error> {
            Ok(match token {
                "<|image|>" => 1001,
                "<|video|>" => 1002,
                "<|image>" => 1003,
                "<image|>" => 1004,
                _ => 1000,
            })
        }
    }
}
