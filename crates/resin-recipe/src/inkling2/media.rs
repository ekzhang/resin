//! Inkling2 image padding and normalization, matching the pinned Hugging Face processor.
//! Inputs are RGB/HWC in [0,1]; padding corresponds to -1 on the original byte scale.

use crate::media::{ImageSource, Input, Prepared, RgbImage, Source, Tensor, image_tokens, images};
use crate::{Encoder, Error};
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
impl Config {
    pub(super) fn prepare(
        &self,
        sources: Vec<Source>,
        encoder: &dyn Encoder,
    ) -> Result<Prepared, Error> {
        let marker = "<|unused_200054|>";
        let image_token_id = encoder.special_id(marker)?;
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
        let mut expansions = Vec::new();
        let mut items = Vec::new();
        for image in images(sources)? {
            let image = into_rgb(image)?;
            let rows = image.height.div_ceil(patch);
            let columns = image.width.div_ceil(patch);
            let height = rows
                .checked_mul(patch)
                .ok_or_else(|| Error::new("Inkling2 image grid is too large"))?;
            let width = columns
                .checked_mul(patch)
                .ok_or_else(|| Error::new("Inkling2 image grid is too large"))?;
            let count = rows
                .checked_mul(columns)
                .ok_or_else(|| Error::new("Inkling2 image grid is too large"))?;
            let elements = height
                .checked_mul(width)
                .and_then(|v| v.checked_mul(3))
                .ok_or_else(|| Error::new("Inkling2 image bitmap is too large"))?;
            let mut data = Vec::new();
            data.try_reserve(elements)
                .map_err(|error| Error::new(format!("cannot allocate Inkling2 bitmap: {error}")))?;
            for y in 0..height {
                for x in 0..width {
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
                            value = (value - self.image_mean[channel]) / self.image_std[channel];
                        }
                        data.push(value);
                    }
                }
            }
            expansions.push(image_tokens(marker, image_token_id, count));
            items.push(Input::Image {
                pixels: Tensor::new(vec![height, width, 3], data)?,
                span: (0, count),
                atomic: false,
            });
        }
        Ok(Prepared::new(expansions, items))
    }
}

fn into_rgb(source: ImageSource) -> Result<RgbImage, Error> {
    let rgba = match source {
        ImageSource::Decoded(image) if image.color().has_alpha() => image.into_rgba8(),
        source => return source.into_rgb(),
    };
    // Inkling composites transparent pixels over an 8-pixel gray checkerboard.
    let mut pixels = Vec::with_capacity(rgba.width() as usize * rgba.height() as usize * 3);
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let background = if (x / 8 + y / 8) % 2 == 0 { 204 } else { 153 };
        let alpha = u32::from(pixel[3]);
        for channel in 0..3 {
            let value =
                (u32::from(pixel[channel]) * alpha + background * (255 - alpha) + 127) / 255;
            pixels.push(value as f32 / 255.0);
        }
    }
    RgbImage::new(rgba.width() as usize, rgba.height() as usize, pixels)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::{RgbImage, tests::Encoder};
    fn image(width: usize, height: usize) -> RgbImage {
        RgbImage::new(width, height, vec![0.5; width * height * 3]).unwrap()
    }
    #[test]
    fn exact_patch_multiples_do_not_add_a_column() {
        let result = Config::default()
            .prepare(
                [image(40, 40), image(80, 40), image(39, 41), image(41, 39)]
                    .into_iter()
                    .map(|image| Source::Image(ImageSource::Rgb(image)))
                    .collect(),
                &Encoder,
            )
            .unwrap();
        assert_eq!(
            result
                .expansions
                .iter()
                .map(|tokens| tokens.token_ids.len())
                .collect::<Vec<_>>(),
            [1, 2, 2, 2]
        );
        assert_eq!(
            (0..result.items.len())
                .map(|index| crate::media::tests::pixels(&result, index).shape.as_slice())
                .collect::<Vec<_>>(),
            [[40, 40, 3], [40, 80, 3], [80, 40, 3], [40, 80, 3]]
        );
    }
    #[test]
    fn text_only_recipe_refuses_media() {
        use crate::{Recipe, inkling2::Inkling2Recipe};

        assert!(
            Inkling2Recipe::default()
                .prepare_media(vec![], &Encoder)
                .is_ok()
        );
        let text_only = Inkling2Recipe {
            text_only: true,
            ..Default::default()
        };
        assert!(text_only.prepare_media(vec![], &Encoder).is_err());
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
        let result = config
            .prepare(vec![Source::Image(ImageSource::Rgb(image))], &Encoder)
            .unwrap();
        let pixels = crate::media::tests::pixels(&result, 0);
        assert_eq!(pixels.shape, [2, 2, 3]);
        for (actual, expected) in pixels.data[..6]
            .iter()
            .zip([-1.0, 0.0, 1.0, -0.5, 0.5, 0.0])
        {
            assert!((actual - expected).abs() < 1e-6);
        }
        for value in &pixels.data[6..] {
            assert!((*value - (-1.0 - 2.0 / 255.0)).abs() < 1e-6);
        }
    }
}
