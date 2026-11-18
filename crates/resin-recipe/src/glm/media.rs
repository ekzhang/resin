//! GLM-5.3-Flash image sizing and normalization. GLM preserves aspect
//! ratio inside a ceil-aligned, black-padded canvas. Input pixels are RGB [0,1].
use serde::{Deserialize, Serialize};

use crate::media::{Input, Prepared, RgbImage, Source, bitmap, image_tokens, images};
use crate::{Encoder, Error};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub patch_expand_factor: usize,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub do_resize: bool,
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub resample: u8,
}
impl Config {
    pub(super) fn prepare(
        &self,
        sources: Vec<Source>,
        encoder: &dyn Encoder,
    ) -> Result<Prepared, Error> {
        let marker = "<|image|>";
        let image_token_id = encoder.special_id(marker)?;
        self.factor()?;
        let mut expansions = Vec::new();
        let mut items = Vec::new();
        for image in images(sources)? {
            let image = image.into_rgb()?;
            let [height, width, content_height, content_width] =
                self.resize_shape(image.height, image.width)?;
            let resized = if (content_height, content_width) == (image.height, image.width) {
                image.data
            } else {
                resize_bicubic(&image, content_height, content_width)
            };
            let data = if (height, width) == (content_height, content_width) {
                resized
            } else {
                let len = height
                    .checked_mul(width)
                    .and_then(|v| v.checked_mul(3))
                    .ok_or_else(|| Error::new("GLM image canvas overflows"))?;
                let mut canvas = vec![0.0; len];
                for y in 0..content_height {
                    canvas[y * width * 3..y * width * 3 + content_width * 3].copy_from_slice(
                        &resized[y * content_width * 3..(y + 1) * content_width * 3],
                    );
                }
                canvas
            };
            let count = height / self.patch_size / self.merge_size
                * (width / self.patch_size / self.merge_size);
            expansions.push(image_tokens(marker, image_token_id, count));
            items.push(Input::Image {
                pixels: bitmap(
                    RgbImage {
                        width,
                        height,
                        data,
                    },
                    self.do_normalize,
                    self.image_mean,
                    self.image_std,
                )?,
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
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
            patch_expand_factor: 1,
            min_image_tokens: 16,
            max_image_tokens: 8000,
            do_resize: true,
            do_normalize: true,
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.261_302_6, 0.275_777_1],
            resample: 3,
        }
    }
}
impl Config {
    fn factor(&self) -> Result<usize, Error> {
        let factor = self
            .patch_size
            .checked_mul(self.merge_size)
            .and_then(|v| v.checked_mul(self.patch_expand_factor));
        if factor.is_none_or(|v| v == 0)
            || self.temporal_patch_size == 0
            || self.min_image_tokens == 0
            || self.max_image_tokens < self.min_image_tokens
            || factor
                .and_then(|v| v.checked_mul(v))
                .and_then(|v| self.max_image_tokens.checked_mul(v))
                .is_none()
            || self
                .patch_size
                .checked_mul(self.patch_size)
                .and_then(|v| v.checked_mul(3))
                .and_then(|v| v.checked_mul(self.temporal_patch_size))
                .is_none()
            || self.resample != 3
            || self.image_mean.iter().any(|v| !v.is_finite())
            || self.image_std.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::new(
                "invalid GLM image geometry, normalization, or bicubic resample configuration",
            ));
        }
        Ok(factor.unwrap())
    }

    /// (canvas height, canvas width, content height, content width), matching
    /// upstream smart_resize(num_frames=temporal_patch_size) and resize padding.
    pub fn resize_shape(&self, height: usize, width: usize) -> Result<[usize; 4], Error> {
        let factor = self.factor()?;
        if height == 0 || width == 0 {
            return Err(Error::new("GLM image must be nonempty"));
        }
        if !self.do_resize {
            if !height.is_multiple_of(factor) || !width.is_multiple_of(factor) {
                return Err(Error::new(
                    "GLM image dimensions must be aligned when resizing is disabled",
                ));
            }
            return Ok([height, width, height, width]);
        }
        let align = |v: usize| v.div_ceil(factor) * factor;
        let min_pixels = self.min_image_tokens * factor * factor;
        let max_pixels = self.max_image_tokens * factor * factor;
        let (mut h, mut w) = (align(height), align(width));
        if h.saturating_mul(w) < min_pixels {
            let scale = (min_pixels as f64 / (height as f64 * width as f64)).sqrt();
            h = align((height as f64 * scale).ceil().max(1.0) as usize);
            w = align((width as f64 * scale).ceil().max(1.0) as usize);
        }
        if h.saturating_mul(w) > max_pixels {
            let (mut low, mut high) = (1, height);
            (h, w) = (factor, factor);
            while low <= high {
                let content_h = low + (high - low) / 2;
                let content_w = (width as f64 * content_h as f64 / height as f64)
                    .floor()
                    .max(1.0) as usize;
                let (a, b) = (align(content_h), align(content_w));
                if a.saturating_mul(b) <= max_pixels {
                    (h, w) = (a, b);
                    low = content_h + 1;
                } else {
                    high = content_h - 1;
                }
            }
        }
        let mut scale = (h as f64 / height as f64).min(w as f64 / width as f64);
        if height.saturating_mul(width) >= min_pixels {
            scale = scale.min(1.0);
        }
        let ch = (height as f64 * scale).floor().max(1.0).min(h as f64) as usize;
        let cw = (width as f64 * scale).floor().max(1.0).min(w as f64) as usize;
        Ok([h, w, ch, cw])
    }
}

// Separable, half-pixel bicubic with antialiasing: the float image path used by
// torchvision's resize (Pillow's cubic coefficient -0.5 and truncated edges).
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}
fn weights(input: usize, output: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let low = (center - 2.0 * filter_scale + 0.5).floor().max(0.0) as usize;
            let high = (center + 2.0 * filter_scale + 0.5)
                .floor()
                .min(input as f64) as usize;
            let mut row: Vec<_> = (low..high)
                .map(|j| (j, cubic((j as f64 + 0.5 - center) / filter_scale) as f32))
                .collect();
            let sum: f32 = row.iter().map(|(_, v)| v).sum();
            for (_, v) in &mut row {
                *v /= sum;
            }
            row
        })
        .collect()
}
fn resize_bicubic(image: &RgbImage, height: usize, width: usize) -> Vec<f32> {
    if (height, width) == (image.height, image.width) {
        return image.data.clone();
    }
    let wx = weights(image.width, width);
    let wy = weights(image.height, height);
    let mut horizontal = vec![0.0; image.height * width * 3];
    for y in 0..image.height {
        for (x, row) in wx.iter().enumerate() {
            for &(src, weight) in row {
                for c in 0..3 {
                    horizontal[(y * width + x) * 3 + c] +=
                        image.data[(y * image.width + src) * 3 + c] * weight;
                }
            }
        }
    }
    let mut out = vec![0.0; height * width * 3];
    for (y, row) in wy.iter().enumerate() {
        for &(src, weight) in row {
            for x in 0..width {
                for c in 0..3 {
                    out[(y * width + x) * 3 + c] += horizontal[(src * width + x) * 3 + c] * weight;
                }
            }
        }
    }
    out
}
