//! Typed media expansions and token placement, independent of model preprocessing.

use std::collections::{HashMap, VecDeque};

use resin_recipe::{Encoder, Error, Prompt, PromptSpan};

/// Explicit media IDs replacing one trusted template placeholder.
///
/// Prefix and suffix are ordinary text and share BPE boundaries with nearby text.
#[derive(Clone)]
pub(crate) struct MediaTokens {
    placeholder: String,
    token_ids: Vec<u32>,
    prefix: String,
    suffix: String,
}
impl MediaTokens {
    pub(crate) fn heap_bytes(&self) -> usize {
        self.placeholder.capacity()
            + self.token_ids.capacity() * std::mem::size_of::<u32>()
            + self.prefix.capacity()
            + self.suffix.capacity()
    }

    pub(crate) fn from_parts(
        placeholder: &str,
        token_ids: Vec<u32>,
        prefix: String,
        suffix: String,
    ) -> Self {
        Self {
            placeholder: placeholder.into(),
            token_ids,
            prefix,
            suffix,
        }
    }
}

type EncodedMedia = (Vec<u32>, Vec<(usize, usize)>);

/// Encode trusted media slots and return their absolute (position, length) in input order.
/// Literal marker spellings in ordinary message text never become media slots.
pub(crate) fn encode(
    prompt: Prompt,
    media: &[MediaTokens],
    encoder: &dyn Encoder,
) -> Result<EncodedMedia, Error> {
    if media.is_empty() {
        return Ok((prompt.encode(encoder)?, Vec::new()));
    }
    let mut replacements: HashMap<&str, VecDeque<usize>> = HashMap::new();
    for (index, span) in media.iter().enumerate() {
        replacements
            .entry(&span.placeholder)
            .or_default()
            .push_back(index);
    }
    let mut ids = Vec::new();
    let mut placements = vec![(0, 0); media.len()];
    let mut text = String::new();
    let flush = |text: &mut String, ids: &mut Vec<u32>| -> Result<(), Error> {
        if !text.is_empty() {
            ids.extend(encoder.encode_text(text)?);
            text.clear();
        }
        Ok(())
    };
    for span in prompt.spans {
        match span {
            PromptSpan::Text(value) => text.push_str(&value),
            PromptSpan::Special(token) => {
                if let Some(indices) = replacements.get_mut(token.as_str()) {
                    let index = indices.pop_front().ok_or_else(|| {
                        Error::new(format!("missing media expansion for {token}"))
                    })?;
                    let replacement = &media[index];
                    text.push_str(&replacement.prefix);
                    flush(&mut text, &mut ids)?;
                    placements[index] = (ids.len(), replacement.token_ids.len());
                    ids.extend_from_slice(&replacement.token_ids);
                    text.push_str(&replacement.suffix);
                } else {
                    flush(&mut text, &mut ids)?;
                    ids.push(encoder.special_id(&token)?);
                }
            }
        }
    }
    for (marker, indices) in replacements {
        if !indices.is_empty() {
            return Err(Error::new(format!(
                "media expansion count does not match {marker}"
            )));
        }
    }
    flush(&mut text, &mut ids)?;
    Ok((ids, placements))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TextEncoder;
    impl Encoder for TextEncoder {
        fn encode_text(&self, text: &str) -> Result<Vec<u32>, Error> {
            Ok(match text {
                "left 00:00 " => vec![7],
                "tail right<image>" => vec![8],
                _ => return Err(Error::new(format!("unexpected text boundary: {text:?}"))),
            })
        }
        fn special_id(&self, _: &str) -> Result<u32, Error> {
            Ok(9)
        }
    }
    #[test]
    fn media_tokens_preserve_text_boundaries_and_literal_markers() {
        let mut prompt = Prompt::new();
        prompt
            .text("left ")
            .special("<image>")
            .text(" right<image>");
        let media = [MediaTokens::from_parts(
            "<image>",
            vec![100, 101],
            "00:00 ".into(),
            "tail".into(),
        )];
        let (ids, positions) = encode(prompt, &media, &TextEncoder).unwrap();
        assert_eq!(ids, [7, 100, 101, 8]);
        assert_eq!(positions, [(1, 2)]);
    }
    #[test]
    fn media_positions_follow_input_order_and_counts_must_match() {
        let mut prompt = Prompt::new();
        prompt
            .special("<video>")
            .special("<image>")
            .special("<image>");
        let media = [
            MediaTokens::from_parts("<image>", vec![10], String::new(), String::new()),
            MediaTokens::from_parts("<video>", vec![20, 21], String::new(), String::new()),
            MediaTokens::from_parts("<image>", vec![30], String::new(), String::new()),
        ];
        let (ids, positions) = encode(prompt.clone(), &media, &TextEncoder).unwrap();
        assert_eq!(ids, [20, 21, 10, 30]);
        assert_eq!(positions, [(2, 1), (0, 2), (3, 1)]);
        assert!(encode(prompt, &media[..2], &TextEncoder).is_err());
        assert!(encode(Prompt::new(), &media, &TextEncoder).is_err());
    }
}
