//! DeepSeek V4.1 image geometry and 3×3 patch-to-language alignment.
// Geometry and patch packing follow the released DeepSeek-V4.1-Flash-1017
// inference/image_processor.py (MIT); resizing uses the shared Rust backend.

use half::bf16;
use image::imageops::FilterType;
use serde::{Deserialize, Serialize};

use super::{RgbImage, Tensor};
use crate::Error;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    pub patch_size: usize,
    pub downsample_ratio: usize,
    pub max_image_tokens: usize,
    pub min_pixels: usize,
    pub max_wh_ratio: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            patch_size: 14,
            downsample_ratio: 3,
            max_image_tokens: 1024,
            min_pixels: 544 * 544,
            max_wh_ratio: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Image {
    pub patches: Tensor<f32>,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub types: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Prepared {
    pub images: Vec<Image>,
}

impl Prepared {
    pub fn token_counts(&self) -> Vec<usize> {
        self.images.iter().map(|image| image.types.len()).collect()
    }
}

impl Config {
    fn validate(&self) -> Result<(), Error> {
        if self.patch_size != 14 || self.downsample_ratio != 3 {
            return Err(Error::new(
                "DeepSeek V4.1 requires patch_size=14 and downsample_ratio=3",
            ));
        }
        if self.max_image_tokens < 4 {
            return Err(Error::new("max_image_tokens must be at least 4"));
        }
        if self
            .max_wh_ratio
            .is_some_and(|ratio| !ratio.is_finite() || ratio < 1.0)
        {
            return Err(Error::new("max_wh_ratio must be finite and at least 1"));
        }
        Ok(())
    }

    /// Return the ViT patch grid (height, width), before 3×3 language alignment.
    pub fn grid(&self, width: usize, height: usize) -> Result<(usize, usize), Error> {
        self.validate()?;
        if width == 0 || height == 0 {
            return Err(Error::new("image dimensions must be positive"));
        }
        if u32::try_from(width).is_err() || u32::try_from(height).is_err() {
            return Err(Error::new("image dimensions exceed u32"));
        }
        let (mut width, mut height) = (width as f64, height as f64);
        if let Some(ratio) = self.max_wh_ratio {
            width = width.min(height * ratio);
        }
        if width * height < self.min_pixels as f64 {
            let scale = (self.min_pixels as f64 / (width * height)).sqrt();
            width = (width * scale).floor();
            height = (height * scale).floor();
        }
        let patch = self.patch_size as f64;
        let cell = patch * self.downsample_ratio as f64;
        let mut best_h = (height / patch).ceil() * patch;
        let mut best_w = (width / patch).ceil() * patch;
        let count = (best_h / cell).ceil() * ((best_w / cell).ceil() + 1.0) + 2.0;
        if count > self.max_image_tokens as f64 {
            let ratio = height / width;
            let max_w = (((self.max_image_tokens - 2) as f64 / ratio) + 0.25).sqrt() - 0.5;
            let max_h = max_w * ratio;
            if max_w < 1.0 {
                best_h = ((self.max_image_tokens - 2) / 2) as f64 * cell;
                best_w = cell;
            } else if max_h < 1.0 {
                best_h = cell;
                best_w = (self.max_image_tokens - 3) as f64 * cell;
            } else {
                let scale = (max_w.floor() * cell / width).min(max_h.floor() * cell / height);
                best_h = (height * scale / patch).floor() * patch;
                best_w = (width * scale / patch).floor() * patch;
            }
        }
        if best_h > u32::MAX as f64 || best_w > u32::MAX as f64 {
            return Err(Error::new("resized image dimensions exceed u32"));
        }
        Ok(((best_h / patch) as usize, (best_w / patch) as usize))
    }

    pub fn process(&self, images: &[RgbImage]) -> Result<Prepared, Error> {
        self.validate()?;
        Ok(Prepared {
            images: images
                .iter()
                .map(|image| self.process_image(image))
                .collect::<Result<_, _>>()?,
        })
    }

    fn process_image(&self, image: &RgbImage) -> Result<Image, Error> {
        let (n_vit_h, n_vit_w) = self.grid(image.width, image.height)?;
        let (height, width) = (n_vit_h * self.patch_size, n_vit_w * self.patch_size);
        // The checkpoint processor decodes to RGB8 before interpolation.
        let source = RgbImage::new(
            image.width,
            image.height,
            image
                .data
                .iter()
                .map(|value| (value * 255.0).round_ties_even() / 255.0)
                .collect(),
        )?;
        let direct = self
            .max_wh_ratio
            .is_some_and(|ratio| image.width as f64 >= ratio * image.height as f64);
        let resized = if direct {
            source.resize(width, height, FilterType::CatmullRom)?
        } else {
            pad(&source, width, height)?
        };
        let mut data = Vec::with_capacity(resized.data.len());
        for patch_y in 0..n_vit_h {
            for patch_x in 0..n_vit_w {
                for channel in 0..3 {
                    for y in 0..self.patch_size {
                        for x in 0..self.patch_size {
                            let index = ((patch_y * self.patch_size + y) * width
                                + patch_x * self.patch_size
                                + x)
                                * 3
                                + channel;
                            let rgb8 = (resized.data[index] * 255.0)
                                .round_ties_even()
                                .clamp(0.0, 255.0);
                            data.push(bf16::from_f32((rgb8 / 255.0 - 0.5) / 0.5).to_f32());
                        }
                    }
                }
            }
        }
        Ok(Image {
            patches: Tensor::new(vec![n_vit_h * n_vit_w, 3, 14, 14], data)?,
            n_vit_h,
            n_vit_w,
            types: image_token_types(n_vit_h.div_ceil(3), n_vit_w.div_ceil(3)),
        })
    }
}

/// Fit with gray padding, using Pillow's round-to-even centering geometry.
fn pad(image: &RgbImage, width: usize, height: usize) -> Result<RgbImage, Error> {
    let ratio = image.width as f64 / image.height as f64;
    let (inner_w, inner_h) = if ratio > width as f64 / height as f64 {
        (width, (width as f64 / ratio).round_ties_even() as usize)
    } else {
        ((height as f64 * ratio).round_ties_even() as usize, height)
    };
    if inner_w == 0 || inner_h == 0 {
        return Err(Error::new(
            "image aspect ratio is too extreme for its patch grid",
        ));
    }
    let resized = image.resize(inner_w, inner_h, FilterType::CatmullRom)?;
    if (inner_w, inner_h) == (width, height) {
        return Ok(resized);
    }
    let x = ((width - inner_w) as f64 / 2.0).round_ties_even() as usize;
    let y = ((height - inner_h) as f64 / 2.0).round_ties_even() as usize;
    let len = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| Error::new("padded image size overflows"))?;
    let mut data = vec![127.0 / 255.0; len];
    for row in 0..inner_h {
        let target = ((row + y) * width + x) * 3;
        data[target..target + inner_w * 3]
            .copy_from_slice(&resized.data[row * inner_w * 3..(row + 1) * inner_w * 3]);
    }
    Ok(RgbImage {
        width,
        height,
        data,
    })
}

pub fn image_token_types(height: usize, width: usize) -> Vec<u32> {
    let mut types = Vec::with_capacity(height * (width + 1) + 2);
    types.push(0);
    for _ in 0..height {
        types.extend(std::iter::repeat_n(1, width));
        types.push(2);
    }
    types.push(3);
    types
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_released_checkpoint_fixtures() {
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/deepseek_v41_images.json"
        ))
        .unwrap();
        for case in fixtures["cases"].as_array().unwrap() {
            let config = Config {
                max_wh_ratio: case["max_wh_ratio"].as_f64(),
                ..Config::default()
            };
            let (w, h) = (
                case["width"].as_u64().unwrap() as usize,
                case["height"].as_u64().unwrap() as usize,
            );
            let grid = config.grid(w, h).unwrap();
            assert_eq!(
                (grid.0 as u64, grid.1 as u64),
                (
                    case["grid"][0].as_u64().unwrap(),
                    case["grid"][1].as_u64().unwrap()
                )
            );
            assert!(
                image_token_types(grid.0.div_ceil(3), grid.1.div_ceil(3)).len()
                    <= config.max_image_tokens
            );
        }
    }

    #[test]
    fn geometry_matches_official_recipe_for_varied_aspect_ratios() {
        let config = Config::default();
        // Token counts from deepseek-recipe 0.2.0's image_token_adjustment + 1.
        // Check large and extreme dimensions without allocating image pixels.
        for (width, height, expected) in [
            (1, 65536, 1024),
            (65536, 1, 1024),
            (65536, 65536, 994),
            (4000, 3000, 1001),
            (256, 4096, 786),
            (4096, 256, 695),
            (1920, 1080, 968),
            (13, 43, 218),
            (43, 13, 202),
        ] {
            let (h, w) = config.grid(width, height).unwrap();
            assert_eq!(
                image_token_types(h.div_ceil(3), w.div_ceil(3)).len(),
                expected
            );
        }
    }

    #[test]
    fn patches_are_patch_major_chw_bfloat16() {
        let config = Config {
            min_pixels: 0,
            ..Config::default()
        };
        let (width, height) = (28, 14);
        let data = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    (0..3)
                        .map(move |channel| ((y * 7 + x * 11 + channel * 73) % 256) as f32 / 255.0)
                })
            })
            .collect();
        let source = RgbImage::new(width, height, data).unwrap();
        let prepared = config.process(std::slice::from_ref(&source)).unwrap();
        let image = &prepared.images[0];
        assert_eq!(image.patches.shape, [2, 3, 14, 14]);
        assert_eq!(image.types, [0, 1, 2, 3]);
        for patch in 0..2 {
            for channel in 0..3 {
                for y in 0..14 {
                    for x in 0..14 {
                        let pixel = source.data[(y * width + patch * 14 + x) * 3 + channel];
                        let expected = bf16::from_f32((pixel - 0.5) / 0.5).to_f32();
                        assert_eq!(
                            image.patches.data[((patch * 3 + channel) * 14 + y) * 14 + x],
                            expected
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn padding_uses_gray_and_source_is_not_mutated() {
        let image = RgbImage::new(2, 1, vec![1.0; 6]).unwrap();
        let padded = pad(&image, 4, 4).unwrap();
        assert_eq!(&padded.data[..12], &[127.0 / 255.0; 12]);
        assert!(
            padded.data[12..36]
                .iter()
                .all(|value| (value - 1.0).abs() < 1e-6)
        );
        assert_eq!(&padded.data[36..], &[127.0 / 255.0; 12]);
        assert_eq!(image.data, vec![1.0; 6]);
    }

    #[test]
    fn invalid_geometry_options_fail_before_processing() {
        assert!(Config::default().grid(0, 32).is_err());
        assert!(Config::default().grid(usize::MAX, 32).is_err());
        assert!(
            Config {
                patch_size: 16,
                ..Config::default()
            }
            .process(&[])
            .is_err()
        );
        assert!(
            Config {
                max_wh_ratio: Some(f64::NAN),
                ..Config::default()
            }
            .grid(2, 2)
            .is_err()
        );
        assert!(
            Config {
                max_image_tokens: 3,
                ..Config::default()
            }
            .grid(2, 2)
            .is_err()
        );
    }
}
