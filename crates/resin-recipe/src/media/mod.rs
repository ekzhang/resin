//! CPU media preprocessing shared by native recipes. Pixels are RGB/HWC in [0, 1].

use base64::Engine;
use image::{ImageBuffer, ImageFormat, Rgb, imageops::FilterType};

use crate::Error;

pub mod deepseek;
pub mod gemma;
pub mod glm;
pub mod inkling2;
pub mod qwen;

#[derive(Clone, Debug)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> Tensor<T> {
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

    pub fn decode(data: &[u8], media_type: &str) -> Result<Self, Error> {
        Self::decode_with_background(data, media_type, false)
    }

    pub fn decode_with_background(
        data: &[u8],
        media_type: &str,
        checkerboard: bool,
    ) -> Result<Self, Error> {
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
        if checkerboard && decoded.color().has_alpha() {
            let rgba = decoded.to_rgba8();
            let mut pixels = Vec::with_capacity(rgba.width() as usize * rgba.height() as usize * 3);
            for (x, y, pixel) in rgba.enumerate_pixels() {
                let background = if (x / 8 + y / 8) % 2 == 0 { 204 } else { 153 };
                let alpha = u32::from(pixel[3]);
                for channel in 0..3 {
                    let value =
                        (u32::from(pixel[channel]) * alpha + background * (255 - alpha) + 127)
                            / 255;
                    pixels.push(value as f32 / 255.0);
                }
            }
            return Self::new(rgba.width() as usize, rgba.height() as usize, pixels);
        }
        let decoded = decoded.to_rgb32f();
        Self::new(
            decoded.width() as usize,
            decoded.height() as usize,
            decoded.into_raw(),
        )
    }

    pub fn data_url(source: &str) -> Result<Self, Error> {
        Self::data_url_with_background(source, false)
    }

    pub fn data_url_with_background(source: &str, checkerboard: bool) -> Result<Self, Error> {
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
        Self::decode_with_background(&data, media_type, checkerboard)
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

#[derive(Clone, Debug)]
pub struct Video {
    pub frames: Vec<RgbImage>,
    pub fps: f64,
    pub frame_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum Source {
    Image(RgbImage),
    Video(Video),
}

/// Append merge-block-major patches, each laid out as channel/time/height/width.
fn append_patches(
    output: &mut Vec<f32>,
    [grid_h, grid_w]: [usize; 2],
    patch: usize,
    merge: usize,
    temporal: usize,
    pixel: impl Fn(usize, usize, usize) -> f32,
) {
    for block_y in 0..grid_h / merge {
        for block_x in 0..grid_w / merge {
            for merge_y in 0..merge {
                for merge_x in 0..merge {
                    for channel in 0..3 {
                        for _ in 0..temporal {
                            for y in 0..patch {
                                for x in 0..patch {
                                    output.push(pixel(
                                        (block_y * merge + merge_y) * patch + y,
                                        (block_x * merge + merge_x) * patch + x,
                                        channel,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
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
