//! Qwen bitmap sizing, normalization and placeholder geometry.
// Geometry and packing follow Transformers' Qwen2VLImageProcessor (Apache-2.0).

use serde::{Deserialize, Serialize};

use crate::media::{Input, Prepared, Source, bitmap, image_tokens, images, resize_filter};
use crate::{Encoder, Error};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Size {
    pub shortest_edge: usize,
    pub longest_edge: usize,
}

impl Default for Size {
    fn default() -> Self {
        Self {
            shortest_edge: 65_536,
            longest_edge: 16_777_216,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    pub size: Size,
    pub min_pixels: Option<usize>,
    pub max_pixels: Option<usize>,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub do_resize: bool,
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub resample: u32,
}

impl Config {
    pub(super) fn prepare(
        &self,
        sources: Vec<Source>,
        encoder: &dyn Encoder,
    ) -> Result<Prepared, Error> {
        let marker = "<|image_pad|>";
        let image_token_id = encoder.special_id(marker)?;
        self.validate()?;
        let filter = resize_filter(self.resample)?;
        let mut expansions = Vec::new();
        let mut items = Vec::new();
        for image in images(sources)? {
            let (height, width) = self.resize_shape(image.width, image.height)?;
            let image = if (width, height) == (image.width, image.height) {
                image
            } else {
                image.resize(width, height, filter)?
            };
            let count = (height / self.patch_size / self.merge_size)
                * (width / self.patch_size / self.merge_size);
            expansions.push(image_tokens(marker, image_token_id, count));
            items.push(Input::Image {
                pixels: bitmap(image, self.do_normalize, self.image_mean, self.image_std)?,
                span: (0, count),
                atomic: false,
            });
        }
        Ok(Prepared::new(expansions, items))
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            size: Size::default(),
            min_pixels: None,
            max_pixels: None,
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            do_resize: true,
            do_normalize: true,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            resample: 3,
        }
    }
}

impl Config {
    fn validate(&self) -> Result<(usize, usize, usize), Error> {
        let min = self.min_pixels.unwrap_or(self.size.shortest_edge);
        let max = self.max_pixels.unwrap_or(self.size.longest_edge);
        if min == 0 || min > max {
            return Err(Error::new(
                "Qwen image pixel bounds must be positive and ordered",
            ));
        }
        let factor = self
            .patch_size
            .checked_mul(self.merge_size)
            .filter(|&n| n != 0 && u32::try_from(n).is_ok())
            .ok_or_else(|| Error::new("Qwen patch and merge sizes must be positive"))?;
        if self.temporal_patch_size == 0
            || self
                .patch_size
                .checked_mul(self.patch_size)
                .and_then(|n| n.checked_mul(3))
                .and_then(|n| n.checked_mul(self.temporal_patch_size))
                .is_none()
        {
            return Err(Error::new("Qwen temporal patch size must be positive"));
        }
        if self.image_mean.iter().any(|x| !x.is_finite())
            || self.image_std.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(Error::new(
                "Qwen normalization must have finite means and positive finite standard deviations",
            ));
        }
        Ok((factor, min, max))
    }

    /// Return resized (height, width), matching HF's ties-to-even grid rounding.
    pub fn resize_shape(&self, width: usize, height: usize) -> Result<(usize, usize), Error> {
        let (factor, min, max) = self.validate()?;
        if width == 0 || height == 0 {
            return Err(Error::new("image dimensions must be positive"));
        }
        if u32::try_from(width).is_err() || u32::try_from(height).is_err() {
            return Err(Error::new("image dimensions exceed u32"));
        }
        if !self.do_resize {
            if !width.is_multiple_of(factor) || !height.is_multiple_of(factor) {
                return Err(Error::new(
                    "unresized Qwen images must be divisible by patch_size * merge_size",
                ));
            }
            return Ok((height, width));
        }
        let (h, w, f) = (height as f64, width as f64, factor as f64);
        if h.max(w) / h.min(w) > 200.0 {
            return Err(Error::new("Qwen image aspect ratio must not exceed 200"));
        }
        let mut h_bar = (h / f).round_ties_even() * f;
        let mut w_bar = (w / f).round_ties_even() * f;
        if h_bar * w_bar > max as f64 {
            let beta = (h * w / max as f64).sqrt();
            h_bar = ((h / beta / f).floor() * f).max(f);
            w_bar = ((w / beta / f).floor() * f).max(f);
        } else if h_bar * w_bar < min as f64 {
            let beta = (min as f64 / (h * w)).sqrt();
            h_bar = (h * beta / f).ceil() * f;
            w_bar = (w * beta / f).ceil() * f;
        }
        if h_bar > u32::MAX as f64 || w_bar > u32::MAX as f64 {
            return Err(Error::new("resized image dimensions exceed u32"));
        }
        Ok((h_bar as usize, w_bar as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::{RgbImage, tests::Encoder};

    #[test]
    fn checkpoint_config_and_smart_resize() {
        use crate::{Recipe, qwen::OrnithRecipe};

        for params in [
            serde_json::json!({}),
            serde_json::json!({"media_config": null}),
            serde_json::json!({"media_config": {}}),
        ] {
            let recipe = OrnithRecipe::from_config(params, &|_| Ok(serde_json::json!({}))).unwrap();
            assert!(recipe.prepare_media(vec![], &Encoder).is_ok());
            let text_only = OrnithRecipe::new(crate::qwen::QwenRecipeParams {
                text_only: true,
                ..Default::default()
            });
            assert!(text_only.prepare_media(vec![], &Encoder).is_err());
        }
        let config: Config = serde_json::from_value(serde_json::json!({
            "size": {"shortest_edge":65536,"longest_edge":16777216},
            "patch_size":16,"temporal_patch_size":2,"merge_size":2,
            "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"do_rescale":true
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(Config::default()).unwrap(),
            serde_json::to_value(&config).unwrap()
        );
        assert_eq!(config.resize_shape(64, 64).unwrap(), (256, 256));
        assert_eq!(config.resize_shape(512, 256).unwrap(), (256, 512));
        assert_eq!(config.resize_shape(80, 80).unwrap(), (256, 256));
        let small = Config {
            min_pixels: Some(1),
            ..config
        };
        // Python's round(2.5) is 2, not 3.
        assert_eq!(small.resize_shape(80, 80).unwrap(), (64, 64));
        assert!(small.resize_shape(201, 1).is_err());
    }

    #[test]
    fn bitmap_normalization_and_placeholder_geometry_match_hf() {
        let config = Config {
            patch_size: 2,
            temporal_patch_size: 2,
            merge_size: 2,
            do_resize: false,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            ..Config::default()
        };
        let (width, height) = (8, 4);
        let data = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    (0..3).map(move |channel| (y * width * 3 + x * 3 + channel) as f32 / 100.0)
                })
            })
            .collect();
        let image = RgbImage::new(width, height, data).unwrap();
        let prepared = config
            .prepare(vec![Source::Image(image.clone())], &Encoder)
            .unwrap();
        let pixels = crate::media::tests::pixels(&prepared, 0);
        assert_eq!(pixels.shape, [height, width, 3]);
        assert_eq!(prepared.expansions[0].token_ids.len(), 2);
        for (actual, source) in pixels.data.iter().zip(&image.data) {
            assert_eq!(*actual, (source - 0.5) / 0.5);
        }
    }

    #[test]
    fn malformed_sizes_and_normalization_fail_without_allocating_pixels() {
        for config in [
            Config {
                patch_size: 0,
                ..Config::default()
            },
            Config {
                patch_size: usize::MAX,
                ..Config::default()
            },
            Config {
                temporal_patch_size: usize::MAX,
                ..Config::default()
            },
            Config {
                min_pixels: Some(0),
                ..Config::default()
            },
            Config {
                max_pixels: Some(1),
                ..Config::default()
            },
            Config {
                image_std: [0.0; 3],
                ..Config::default()
            },
            Config {
                image_mean: [f32::NAN; 3],
                ..Config::default()
            },
        ] {
            assert!(config.prepare(vec![], &Encoder).is_err());
        }
        assert!(Config::default().resize_shape(0, 32).is_err());
        assert!(Config::default().resize_shape(usize::MAX, 32).is_err());
        assert!(
            Config {
                do_resize: false,
                ..Config::default()
            }
            .resize_shape(31, 32)
            .is_err()
        );
    }
}
