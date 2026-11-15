//! Qwen image patches and prompt-relative multimodal rotary positions.
// Geometry and packing follow Transformers' Qwen2VLImageProcessor (Apache-2.0).

use serde::{Deserialize, Serialize};

use super::{RgbImage, Tensor, append_patches, resize_filter};
use crate::Error;

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

#[derive(Clone, Debug)]
pub struct Prepared {
    pub pixel_values: Tensor<f32>,
    pub image_grid_thw: Tensor<i64>,
    pub token_counts: Vec<usize>,
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
        if self.temporal_patch_size == 0 {
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

    pub fn process(&self, images: &[RgbImage]) -> Result<Prepared, Error> {
        self.validate()?;
        let filter = resize_filter(self.resample)?;
        let patch = self.patch_size;
        let merge = self.merge_size;
        let temporal = self.temporal_patch_size;
        let patch_width = 3usize
            .checked_mul(temporal)
            .and_then(|n| n.checked_mul(patch))
            .and_then(|n| n.checked_mul(patch))
            .ok_or_else(|| Error::new("Qwen patch size overflows"))?;
        let mut pixels = Vec::new();
        let mut grids = Vec::with_capacity(images.len() * 3);
        let mut token_counts = Vec::with_capacity(images.len());
        let mut total_patches = 0usize;
        for image in images {
            let (height, width) = self.resize_shape(image.width, image.height)?;
            let resized;
            let image = if (width, height) == (image.width, image.height) {
                image
            } else {
                resized = image.resize(width, height, filter)?;
                &resized
            };
            let (grid_h, grid_w) = (height / patch, width / patch);
            let patch_count = grid_h
                .checked_mul(grid_w)
                .ok_or_else(|| Error::new("Qwen patch grid overflows"))?;
            total_patches = total_patches
                .checked_add(patch_count)
                .ok_or_else(|| Error::new("Qwen patch count overflows"))?;
            let count = patch_count
                .checked_mul(patch_width)
                .ok_or_else(|| Error::new("Qwen pixel tensor size overflows"))?;
            pixels
                .try_reserve(count)
                .map_err(|_| Error::new("Qwen pixel tensor is too large"))?;
            append_patches(
                &mut pixels,
                [grid_h, grid_w],
                patch,
                merge,
                temporal,
                |y, x, c| {
                    let value = image.data[(y * width + x) * 3 + c];
                    if self.do_normalize {
                        (value - self.image_mean[c]) / self.image_std[c]
                    } else {
                        value
                    }
                },
            );
            grids.extend([1, grid_h as i64, grid_w as i64]);
            token_counts.push((grid_h / merge) * (grid_w / merge));
        }
        Ok(Prepared {
            pixel_values: Tensor::new(vec![total_patches, patch_width], pixels)?,
            image_grid_thw: Tensor::new(vec![images.len(), 3], grids)?,
            token_counts,
        })
    }
}

/// Bind ordered media ranges to full-prompt T/H/W positions, without scanning
/// token IDs or conflating adjacent image spans.
pub fn mrope_positions(
    token_count: usize,
    spans: &[(usize, usize)],
    grids: &Tensor<i64>,
    merge_size: usize,
) -> Result<(Tensor<i64>, i64), Error> {
    if merge_size == 0 || grids.shape != [spans.len(), 3] || grids.data.len() != spans.len() * 3 {
        return Err(Error::new("Qwen media spans and image grids do not match"));
    }
    let length = token_count
        .checked_mul(3)
        .filter(|_| token_count <= i64::MAX as usize)
        .ok_or_else(|| Error::new("Qwen prompt length overflows"))?;
    let mut positions = vec![0; length];
    let (mut cursor, mut current) = (0usize, 0i64);
    for (&(start, length), grid) in spans.iter().zip(grids.data.as_chunks::<3>().0) {
        let end = start
            .checked_add(length)
            .filter(|&end| end <= token_count)
            .ok_or_else(|| Error::new("Qwen image span is outside the prompt"))?;
        if start < cursor || length == 0 {
            return Err(Error::new(
                "Qwen image spans must be nonempty, ordered and disjoint",
            ));
        }
        for position in cursor..start {
            for axis in 0..3 {
                positions[axis * token_count + position] = current + (position - cursor) as i64;
            }
        }
        current += (start - cursor) as i64;
        let dims = grid
            .iter()
            .map(|&value| {
                usize::try_from(value)
                    .ok()
                    .filter(|&value| value > 0)
                    .ok_or_else(|| Error::new("Qwen image grids must be positive"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !dims[1].is_multiple_of(merge_size) || !dims[2].is_multiple_of(merge_size) {
            return Err(Error::new(
                "Qwen image grid must be divisible by spatial merge size",
            ));
        }
        let (t, h, w) = (dims[0], dims[1] / merge_size, dims[2] / merge_size);
        if t.checked_mul(h).and_then(|n| n.checked_mul(w)) != Some(length) {
            return Err(Error::new(
                "Qwen image grid does not match its placeholder range",
            ));
        }
        let mut position = start;
        for time in 0..t {
            for row in 0..h {
                for col in 0..w {
                    positions[position] = current + time as i64;
                    positions[token_count + position] = current + row as i64;
                    positions[2 * token_count + position] = current + col as i64;
                    position += 1;
                }
            }
        }
        current += h.max(w) as i64;
        cursor = end;
    }
    for position in cursor..token_count {
        for axis in 0..3 {
            positions[axis * token_count + position] = current + (position - cursor) as i64;
        }
    }
    let delta = positions
        .iter()
        .max()
        .map_or(0, |maximum| maximum + 1 - token_count as i64);
    Ok((Tensor::new(vec![3, token_count], positions)?, delta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_config_and_smart_resize() {
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
    fn patches_match_hf_merge_temporal_and_channel_order() {
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
        let prepared = config.process(std::slice::from_ref(&image)).unwrap();
        assert_eq!(prepared.pixel_values.shape, [8, 24]);
        assert_eq!(prepared.image_grid_thw.data, [1, 2, 4]);
        assert_eq!(prepared.token_counts, [2]);
        let patch_origins = [
            (0, 0),
            (2, 0),
            (0, 2),
            (2, 2),
            (4, 0),
            (6, 0),
            (4, 2),
            (6, 2),
        ];
        for (patch, &(x, y)) in patch_origins.iter().enumerate() {
            for channel in 0..3 {
                for time in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let expected =
                                (image.data[((y + dy) * width + x + dx) * 3 + channel] - 0.5) / 0.5;
                            assert_eq!(
                                prepared.pixel_values.data
                                    [patch * 24 + channel * 8 + time * 4 + dy * 2 + dx],
                                expected
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mrope_matches_reference_prompt_and_keeps_adjacent_images_distinct() {
        let grids = Tensor::new(vec![1, 3], vec![1, 4, 4]).unwrap();
        let (positions, delta) = mrope_positions(6, &[(1, 4)], &grids, 2).unwrap();
        assert_eq!(
            positions.data,
            [0, 1, 1, 1, 1, 3, 0, 1, 1, 2, 2, 3, 0, 1, 2, 1, 2, 3]
        );
        assert_eq!(delta, -2);
        let grids = Tensor::new(vec![2, 3], vec![1, 4, 4, 1, 2, 4]).unwrap();
        let (positions, delta) = mrope_positions(8, &[(1, 4), (5, 2)], &grids, 2).unwrap();
        assert_eq!(
            positions.data,
            [
                0, 1, 1, 1, 1, 3, 3, 5, 0, 1, 1, 2, 2, 3, 3, 5, 0, 1, 2, 1, 2, 3, 4, 5
            ]
        );
        assert_eq!(delta, -2);
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
            assert!(config.process(&[]).is_err());
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

    #[test]
    fn mrope_validates_geometry_and_ranges() {
        let grids = Tensor::new(vec![1, 3], vec![1, 4, 4]).unwrap();
        assert!(mrope_positions(4, &[(1, 4)], &grids, 2).is_err());
        assert!(mrope_positions(6, &[(1, 3)], &grids, 2).is_err());
        assert!(mrope_positions(6, &[(1, 4)], &grids, 0).is_err());
        assert!(mrope_positions(6, &[], &grids, 2).is_err());
        let grids = Tensor::new(vec![2, 3], vec![1, 4, 4, 1, 4, 4]).unwrap();
        assert!(mrope_positions(10, &[(1, 4), (3, 4)], &grids, 2).is_err());
    }
}
