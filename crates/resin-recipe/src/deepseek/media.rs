//! DeepSeek V4.1 image geometry and 3×3 patch-to-language alignment.
// Geometry and patch packing follow the released DeepSeek-V4.1-Flash-1017
// inference/image_processor.py (MIT); resizing uses the shared Rust backend.

use half::bf16;
use image::imageops::FilterType;
use serde::{Deserialize, Serialize};

use crate::media::{Input, Prepared, RgbImage, Source, Tensor, image_tokens, images};
use crate::{Encoder, Error};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Processor {
    pub image_token_id: Option<u32>,
    pub patch_size: usize,
    pub downsample_ratio: usize,
    pub max_image_tokens: usize,
    pub min_pixels: usize,
    pub max_wh_ratio: Option<f64>,
}

impl crate::MediaProcessor for Processor {
    fn prepare(&self, sources: Vec<Source>, encoder: &dyn Encoder) -> Result<Prepared, Error> {
        let marker = "<｜deepseek_image｜>";
        let image_token_id = encoder.special_id(marker)?;
        if self
            .image_token_id
            .is_some_and(|expected| expected != image_token_id)
        {
            return Err(Error::new(
                "tokenizer image token does not match checkpoint image_token_id",
            ));
        }
        self.validate()?;
        let mut expansions = Vec::new();
        let mut items = Vec::new();
        for image in images(sources)? {
            let pixels = self.process_image(&image)?;
            let (rows, columns) = (
                pixels.shape[0] / self.patch_size,
                pixels.shape[1] / self.patch_size,
            );
            let count = rows.div_ceil(3) * (columns.div_ceil(3) + 1) + 2;
            expansions.push(image_tokens(marker, image_token_id, count));
            items.push(Input::Image {
                pixels,
                span: (0, count),
                atomic: false,
            });
        }
        Ok(Prepared::new(expansions, items))
    }
}

impl Default for Processor {
    fn default() -> Self {
        Self {
            image_token_id: None,
            patch_size: 14,
            downsample_ratio: 3,
            max_image_tokens: 1024,
            min_pixels: 544 * 544,
            max_wh_ratio: None,
        }
    }
}

impl Processor {
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

    fn process_image(&self, image: &RgbImage) -> Result<Tensor<f32>, Error> {
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
        let data = resized
            .data
            .into_iter()
            .map(|value| {
                let rgb8 = (value * 255.0).round_ties_even().clamp(0.0, 255.0);
                bf16::from_f32((rgb8 / 255.0 - 0.5) / 0.5).to_f32()
            })
            .collect();
        Tensor::new(vec![height, width, 3], data)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MediaProcessor;
    use crate::media::tests::Encoder;

    #[test]
    fn geometry_matches_released_checkpoint_fixtures() {
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/deepseek_v41_images.json"
        ))
        .unwrap();
        for case in fixtures["cases"].as_array().unwrap() {
            let config = Processor {
                max_wh_ratio: case["max_wh_ratio"].as_f64(),
                ..Processor::default()
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
            assert!(grid.0.div_ceil(3) * (grid.1.div_ceil(3) + 1) + 2 <= config.max_image_tokens);
        }
    }

    #[test]
    fn geometry_matches_official_recipe_for_varied_aspect_ratios() {
        let config = Processor::default();
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
            assert_eq!(h.div_ceil(3) * (w.div_ceil(3) + 1) + 2, expected);
        }
    }

    #[test]
    fn bitmap_preserves_rgb_order_and_bfloat16_normalization() {
        let config = Processor {
            min_pixels: 0,
            ..Processor::default()
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
        let prepared = config
            .prepare(vec![Source::Image(source.clone())], &Encoder)
            .unwrap();
        let pixels = crate::media::tests::pixels(&prepared, 0);
        assert_eq!(pixels.shape, [height, width, 3]);
        assert_eq!(prepared.expansions[0].token_ids.len(), 4);
        for (actual, pixel) in pixels.data.iter().zip(&source.data) {
            let expected = bf16::from_f32((pixel - 0.5) / 0.5).to_f32();
            assert_eq!(*actual, expected);
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
        assert!(Processor::default().grid(0, 32).is_err());
        assert!(Processor::default().grid(usize::MAX, 32).is_err());
        assert!(
            Processor {
                patch_size: 16,
                ..Processor::default()
            }
            .prepare(vec![], &Encoder)
            .is_err()
        );
        assert!(
            Processor {
                max_wh_ratio: Some(f64::NAN),
                ..Processor::default()
            }
            .grid(2, 2)
            .is_err()
        );
        assert!(
            Processor {
                max_image_tokens: 3,
                ..Processor::default()
            }
            .grid(2, 2)
            .is_err()
        );
    }
}
