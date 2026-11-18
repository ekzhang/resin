//! Gemma 4 image/video sizing and normalization. Geometry follows the Apache-2.0
//! Transformers Gemma4 image/video processors; resizing uses our shared Rust filter.

use serde::{Deserialize, Deserializer};

use crate::media::{
    Input, MediaTokens, Prepared, RgbImage, Source, Tensor, Video, bitmap, resize_filter,
};
use crate::{Encoder, Error};

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct PatchConfig {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub max_soft_tokens: Option<usize>,
    pub do_resize: bool,
    pub do_normalize: Option<bool>,
    #[serde(deserialize_with = "channels")]
    pub image_mean: [f32; 3],
    #[serde(deserialize_with = "channels")]
    pub image_std: [f32; 3],
    pub resample: u32,
}
impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens: None,
            do_resize: true,
            do_normalize: None,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            resample: 3,
        }
    }
}
fn channels<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[f32; 3], D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Channels {
        Scalar(f32),
        Rgb([f32; 3]),
    }
    Ok(match Channels::deserialize(deserializer)? {
        Channels::Scalar(value) => [value; 3],
        Channels::Rgb(values) => values,
    })
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct VideoConfig {
    #[serde(flatten)]
    pub patches: PatchConfig,
    pub do_sample_frames: bool,
    pub num_frames: Option<usize>,
    pub fps: Option<f64>,
}
impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            patches: PatchConfig::default(),
            do_sample_frames: true,
            num_frames: Some(32),
            fps: None,
        }
    }
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub image_processor: PatchConfig,
    pub video_processor: VideoConfig,
}

impl Config {
    pub(super) fn prepare(
        &self,
        sources: Vec<Source>,
        encoder: &dyn Encoder,
    ) -> Result<Prepared, Error> {
        let image_id = encoder.special_id("<|image|>")?;
        let video_id = encoder.special_id("<|video|>")?;
        let boi = encoder.special_id("<|image>")?;
        let eoi = encoder.special_id("<image|>")?;
        let mut expansions = Vec::new();
        let mut items = Vec::new();
        for source in sources {
            match source {
                Source::Image(image) => {
                    let (pixels, count) = self.image_processor.process(image.into_rgb()?, false)?;
                    let ids = std::iter::once(boi)
                        .chain(std::iter::repeat_n(image_id, count))
                        .chain([eoi])
                        .collect();
                    expansions.push(MediaTokens::from_parts(
                        "<|image|>",
                        ids,
                        String::new(),
                        String::new(),
                    ));
                    items.push(Input::Image {
                        pixels,
                        span: (1, count),
                        atomic: true,
                    });
                }
                Source::Video(video) => {
                    let selected = self.video_processor.select(&video)?;
                    let mut frames = Vec::new();
                    let mut frame_spans = Vec::new();
                    let mut frame_indices = Vec::new();
                    let mut ids = Vec::new();
                    let mut prefix = String::new();
                    for (index, frame) in video
                        .frames
                        .into_iter()
                        .enumerate()
                        .filter(|(index, _)| selected.binary_search(index).is_ok())
                    {
                        let (pixels, count) = self.video_processor.patches.process(frame, true)?;
                        let text =
                            format!("{} ", timestamp(video.frame_indices[index], video.fps)?);
                        if frames.is_empty() {
                            prefix = text;
                        } else {
                            ids.extend(encoder.encode_text(&format!(" {text}"))?);
                        }
                        ids.push(boi);
                        frame_spans.push((ids.len(), count));
                        ids.extend(std::iter::repeat_n(video_id, count));
                        ids.push(eoi);
                        frames.push(pixels);
                        frame_indices.push(video.frame_indices[index]);
                    }
                    expansions.push(MediaTokens::from_parts(
                        "<|video|>",
                        ids,
                        prefix,
                        String::new(),
                    ));
                    items.push(Input::Video {
                        frames,
                        frame_spans,
                        fps: video.fps,
                        frame_indices,
                        atomic: true,
                    });
                }
            }
        }
        Ok(Prepared::new(expansions, items))
    }
}

fn product(values: &[usize]) -> Result<usize, Error> {
    values
        .iter()
        .try_fold(1usize, |n, value| n.checked_mul(*value))
        .ok_or_else(|| Error::new("Gemma patch geometry overflows addressable storage"))
}

impl PatchConfig {
    fn max_patches(&self, is_video: bool) -> Result<usize, Error> {
        let tokens = self
            .max_soft_tokens
            .unwrap_or(if is_video { 70 } else { 280 });
        if ![70, 140, 280, 560, 1120].contains(&tokens) {
            return Err(Error::new(
                "Gemma max_soft_tokens must be 70, 140, 280, 560, or 1120",
            ));
        }
        if self.patch_size == 0 || self.pooling_kernel_size == 0 {
            return Err(Error::new("Gemma patch and pooling sizes must be positive"));
        }
        product(&[tokens, self.pooling_kernel_size, self.pooling_kernel_size])
    }
    fn target_size(
        &self,
        height: usize,
        width: usize,
        max_patches: usize,
    ) -> Result<(usize, usize), Error> {
        let target_pixels = product(&[max_patches, self.patch_size, self.patch_size])?;
        let side = product(&[self.patch_size, self.pooling_kernel_size])?;
        let factor = (target_pixels as f64 / (height as f64 * width as f64)).sqrt();
        let mut target_height = ((factor * height as f64 / side as f64).floor() as usize) * side;
        let mut target_width = ((factor * width as f64 / side as f64).floor() as usize) * side;
        let max_side =
            max_patches / product(&[self.pooling_kernel_size, self.pooling_kernel_size])? * side;
        if target_height == 0 && target_width == 0 {
            return Err(Error::new("Gemma resized image would have no patches"));
        }
        if target_height == 0 {
            target_height = side;
            target_width = product(&[width / height, side])?.min(max_side);
        } else if target_width == 0 {
            target_width = side;
            target_height = product(&[height / width, side])?.min(max_side);
        }
        if product(&[target_height, target_width])? > target_pixels {
            return Err(Error::new("Gemma resized image exceeds the patch budget"));
        }
        Ok((target_height, target_width))
    }
    fn process(&self, image: RgbImage, is_video: bool) -> Result<(Tensor<f32>, usize), Error> {
        let max_patches = self.max_patches(is_video)?;
        let image = if self.do_resize {
            let (height, width) = self.target_size(image.height, image.width, max_patches)?;
            if (height, width) == (image.height, image.width) {
                image
            } else {
                image.resize(width, height, resize_filter(self.resample)?)?
            }
        } else {
            image
        };
        if !image.height.is_multiple_of(self.patch_size)
            || !image.width.is_multiple_of(self.patch_size)
        {
            return Err(Error::new(
                "Gemma image dimensions must be divisible by patch_size",
            ));
        }
        let normalize = self.do_normalize.unwrap_or(is_video);
        if normalize
            && (self.image_mean.iter().any(|v| !v.is_finite())
                || self.image_std.iter().any(|v| !v.is_finite() || *v <= 0.0))
        {
            return Err(Error::new(
                "Gemma normalization needs finite means and positive standard deviations",
            ));
        }
        let count = product(&[
            image.height / self.patch_size,
            image.width / self.patch_size,
        ])? / product(&[self.pooling_kernel_size, self.pooling_kernel_size])?;
        Ok((
            bitmap(image, normalize, self.image_mean, self.image_std)?,
            count,
        ))
    }
}

impl VideoConfig {
    fn select(&self, video: &Video) -> Result<Vec<usize>, Error> {
        let total = video.frames.len();
        if total == 0 || video.frame_indices.len() != total {
            return Err(Error::new(
                "Gemma video needs nonempty frames and one original index per frame",
            ));
        }
        if !video.fps.is_finite() || video.fps <= 0.0 {
            return Err(Error::new("Gemma video fps must be finite and positive"));
        }
        let first = &video.frames[0];
        if video
            .frames
            .iter()
            .any(|f| (f.width, f.height) != (first.width, first.height))
        {
            return Err(Error::new(
                "Gemma video frames must have matching image dimensions",
            ));
        }
        if !self.do_sample_frames {
            return Ok((0..total).collect());
        }
        let count = match (self.num_frames, self.fps) {
            (Some(_), Some(_)) => {
                return Err(Error::new(
                    "Gemma num_frames and fps sampling are mutually exclusive",
                ));
            }
            (Some(n), None) => n,
            (None, Some(fps)) if fps.is_finite() && fps > 0.0 => {
                (total as f64 / video.fps * fps).floor() as usize
            }
            (None, Some(_)) => {
                return Err(Error::new("Gemma sampling fps must be finite and positive"));
            }
            (None, None) => total,
        };
        if count == 0 || count > total {
            return Err(Error::new(
                "Gemma sampled frame count must be positive and no larger than the video",
            ));
        }
        Ok((0..count)
            .map(|i| ((i as u128 * total as u128) / count as u128) as usize)
            .collect())
    }
}
fn timestamp(index: usize, fps: f64) -> Result<String, Error> {
    let seconds = (index as f64 / fps).floor();
    if !seconds.is_finite() || seconds >= u64::MAX as f64 {
        return Err(Error::new(
            "Gemma video timestamp is outside the supported range",
        ));
    }
    let seconds = seconds as u64;
    Ok(format!("{:02}:{:02}", seconds / 60, seconds % 60))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::ImageSource;
    use crate::media::tests::Encoder;
    use serde_json::json;

    fn config() -> Config {
        serde_json::from_value(json!({
            "image_processor": {"patch_size":2,"pooling_kernel_size":1,"max_soft_tokens":70,"do_resize":false},
            "video_processor": {"patch_size":2,"pooling_kernel_size":1,"max_soft_tokens":70,"do_resize":false,"num_frames":2}
        })).unwrap()
    }
    fn image(width: usize, height: usize) -> RgbImage {
        let n = width * height * 3;
        RgbImage::new(width, height, (0..n).map(|i| i as f32 / n as f32).collect()).unwrap()
    }
    #[test]
    fn bitmap_preserves_rgb_order_normalization_and_atomic_placeholder_span() {
        let mut cfg = config();
        cfg.image_processor.do_normalize = Some(true);
        cfg.image_processor.image_mean = [0.1, 0.2, 0.3];
        cfg.image_processor.image_std = [0.5, 0.25, 2.0];
        let rgb = image(4, 2);
        let prepared = cfg
            .prepare(vec![Source::Image(ImageSource::Rgb(rgb.clone()))], &Encoder)
            .unwrap();
        let pixels = crate::media::tests::pixels(&prepared, 0);
        assert_eq!(pixels.shape, [2, 4, 3]);
        for (index, (actual, source)) in pixels.data.iter().zip(&rgb.data).enumerate() {
            let channel = index % 3;
            assert_eq!(
                *actual,
                (source - cfg.image_processor.image_mean[channel])
                    / cfg.image_processor.image_std[channel]
            );
        }
        assert!(matches!(
            &prepared.items[0],
            Input::Image {
                span: (1, 2),
                atomic: true,
                ..
            }
        ));
        assert_eq!(prepared.expansions[0].token_ids, [1003, 1001, 1001, 1004]);
    }
    #[test]
    fn mixed_sources_keep_expansion_order_video_sampling_and_original_timestamps() {
        let frames = (0..5).map(|_| image(2, 2)).collect();
        let sources = vec![
            Source::Image(ImageSource::Rgb(image(2, 2))),
            Source::Video(Video {
                frames,
                fps: 2.0,
                frame_indices: vec![120, 121, 124, 125, 130],
            }),
            Source::Image(ImageSource::Rgb(image(2, 2))),
        ];
        let result = config().prepare(sources, &Encoder).unwrap();
        assert_eq!(crate::media::tests::pixels(&result, 0).shape, [2, 2, 3]);
        assert_eq!(crate::media::tests::pixels(&result, 2).shape, [2, 2, 3]);
        let Input::Video {
            frames,
            frame_spans,
            fps,
            frame_indices,
            atomic,
        } = &result.items[1]
        else {
            panic!("expected video")
        };
        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(|frame| frame.shape == [2, 2, 3]));
        assert_eq!(*fps, 2.0);
        assert_eq!(frame_indices, &[120, 124]);
        assert_eq!(frame_spans, &[(1, 1), (11, 1)]);
        assert!(*atomic);
        assert_eq!(
            result
                .expansions
                .iter()
                .map(|e| e.placeholder.as_str())
                .collect::<Vec<_>>(),
            ["<|image|>", "<|video|>", "<|image|>"]
        );
        assert_eq!(result.expansions[1].prefix, "01:00 ");
        let mut expected = vec![1003, 1002, 1004];
        expected.extend(" 01:02 ".bytes().map(u32::from));
        expected.extend([1003, 1002, 1004]);
        assert_eq!(result.expansions[1].token_ids, expected);
    }
    #[test]
    fn aspect_geometry_matches_reference_including_extreme_ratios() {
        let cfg = PatchConfig::default();
        let max = cfg.max_patches(false).unwrap();
        for (height, width, target) in [
            (100, 100, (768, 768)),
            (640, 960, (624, 960)),
            (1, 10000, (48, 13440)),
            (10000, 1, (13440, 48)),
        ] {
            assert_eq!(cfg.target_size(height, width, max).unwrap(), target);
        }
        assert_eq!(cfg.max_patches(true).unwrap(), 630);
    }
    #[test]
    fn partial_configs_keep_distinct_image_and_video_defaults() {
        let cfg: Config = serde_json::from_value(json!({
            "image_processor": {"patch_size":1,"pooling_kernel_size":1,"do_resize":false,"image_mean":0.5},
            "video_processor": {"patch_size":1,"pooling_kernel_size":1,"do_resize":false,"image_mean":0.5,"do_sample_frames":false}
        })).unwrap();
        let image = RgbImage::new(1, 1, vec![0.25; 3]).unwrap();
        let result = cfg
            .prepare(
                vec![
                    Source::Image(ImageSource::Rgb(image.clone())),
                    Source::Video(Video {
                        frames: vec![image],
                        fps: 24.0,
                        frame_indices: vec![0],
                    }),
                ],
                &Encoder,
            )
            .unwrap();
        let image = crate::media::tests::pixels(&result, 0);
        let Input::Video { frames, .. } = &result.items[1] else {
            panic!("expected video")
        };
        assert_eq!(image.shape, [1, 1, 3]);
        assert_eq!(frames[0].shape, [1, 1, 3]);
        assert_eq!(image.data, [0.25; 3]);
        assert_eq!(frames[0].data, [-0.25; 3]);
    }
    #[test]
    fn malformed_geometry_video_metadata_and_sampling_fail_before_work() {
        let cfg = config();
        assert!(
            cfg.prepare(
                vec![Source::Video(Video {
                    frames: vec![],
                    fps: 24.0,
                    frame_indices: vec![]
                })],
                &Encoder
            )
            .is_err()
        );
        assert!(
            cfg.prepare(
                vec![Source::Video(Video {
                    frames: vec![image(2, 2)],
                    fps: 24.0,
                    frame_indices: vec![0]
                })],
                &Encoder
            )
            .is_err()
        );
        let mut bad = cfg.clone();
        bad.image_processor.patch_size = 0;
        assert!(
            bad.prepare(vec![Source::Image(ImageSource::Rgb(image(2, 2)))], &Encoder)
                .is_err()
        );
    }
}
