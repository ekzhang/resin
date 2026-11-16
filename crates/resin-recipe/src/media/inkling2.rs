//! Native Inkling2 40-pixel image patches, matching the pinned Hugging Face processor.
//! Inputs are RGB/HWC in [0,1]; padding corresponds to -1 on the original byte scale.

use super::{RgbImage, Tensor};
use crate::Error;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Size {
    pub height: usize,
    pub width: usize,
}
impl Default for Size {
    fn default() -> Self {
        Self {
            height: 40,
            width: 40,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub size: Size,
    pub do_normalize: bool,
    pub do_rescale: bool,
    pub rescale_factor: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}
impl Default for Config {
    fn default() -> Self {
        Self {
            size: Size::default(),
            do_normalize: true,
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        }
    }
}
#[derive(Clone)]
pub struct Prepared {
    pub pixel_values: Tensor<f32>,
    pub image_grid_thw: Tensor<i64>,
    pub token_counts: Vec<usize>,
}
impl Config {
    pub fn process(&self, images: &[RgbImage]) -> Result<Prepared, Error> {
        if self.size.height == 0 || self.size.height != self.size.width {
            return Err(Error::new("Inkling2 requires a positive square patch size"));
        }
        if !self.rescale_factor.is_finite()
            || self.rescale_factor <= 0.0
            || self.image_mean.iter().any(|x| !x.is_finite())
            || self.image_std.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(Error::new(
                "Inkling2 normalization parameters must be finite and scales positive",
            ));
        }
        let patch = self.size.height;
        let mut data = Vec::new();
        let mut grids = Vec::new();
        let mut token_counts = Vec::new();
        for image in images {
            let rows = image.height.div_ceil(patch);
            let columns = image.width.div_ceil(patch);
            let count = rows
                .checked_mul(columns)
                .ok_or_else(|| Error::new("Inkling2 image grid is too large"))?;
            let elements = count
                .checked_mul(patch)
                .and_then(|v| v.checked_mul(patch))
                .and_then(|v| v.checked_mul(3))
                .ok_or_else(|| Error::new("Inkling2 patch tensor is too large"))?;
            data.try_reserve(elements).map_err(|error| {
                Error::new(format!("cannot allocate Inkling2 patches: {error}"))
            })?;
            grids.extend([1, rows as i64, columns as i64]);
            token_counts.push(count);
            for row in 0..rows {
                for column in 0..columns {
                    for y in 0..patch {
                        for x in 0..patch {
                            let (y, x) = (row * patch + y, column * patch + x);
                            for channel in 0..3 {
                                let mut value = if y < image.height && x < image.width {
                                    image.data[(y * image.width + x) * 3 + channel] * 255.0
                                } else {
                                    -1.0
                                };
                                if self.do_rescale {
                                    value *= self.rescale_factor;
                                }
                                if self.do_normalize {
                                    value = (value - self.image_mean[channel])
                                        / self.image_std[channel];
                                }
                                data.push(value);
                            }
                        }
                    }
                }
            }
        }
        Ok(Prepared {
            pixel_values: Tensor::new(vec![token_counts.iter().sum(), patch, patch, 3], data)?,
            image_grid_thw: Tensor::new(vec![images.len(), 3], grids)?,
            token_counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image(width: usize, height: usize) -> RgbImage {
        RgbImage::new(width, height, vec![0.5; width * height * 3]).unwrap()
    }
    #[test]
    fn exact_patch_multiples_do_not_add_a_column() {
        let result = Config::default()
            .process(&[image(40, 40), image(80, 40), image(39, 41), image(41, 39)])
            .unwrap();
        assert_eq!(result.token_counts, [1, 2, 2, 2]);
        assert_eq!(result.pixel_values.shape, [7, 40, 40, 3]);
        assert_eq!(
            result.image_grid_thw.data,
            [1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2]
        );
    }
    #[test]
    fn padding_is_negative_one_on_the_byte_scale() {
        let config = Config {
            size: Size {
                height: 2,
                width: 2,
            },
            ..Config::default()
        };
        let image = RgbImage::new(2, 1, vec![0.0, 0.5, 1.0, 0.25, 0.75, 0.5]).unwrap();
        let result = config.process(&[image]).unwrap();
        assert_eq!(result.pixel_values.shape, [1, 2, 2, 3]);
        for (actual, expected) in result.pixel_values.data[..6]
            .iter()
            .zip([-1.0, 0.0, 1.0, -0.5, 0.5, 0.0])
        {
            assert!((actual - expected).abs() < 1e-6);
        }
        for value in &result.pixel_values.data[6..] {
            assert!((*value - (-1.0 - 2.0 / 255.0)).abs() < 1e-6);
        }
    }
}
