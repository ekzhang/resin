//! Borrowed columns from one synchronous forward pass.

use crate::engine::{Error, Result};
use crate::output::TokenRows;

/// Default limit on top alternatives per generated or prompt token.
pub const DEFAULT_MAX_TOP_LOGPROBS: u32 = 32;

#[derive(Clone, Debug)]
pub struct TokenLogprobs {
    pub logprob: f32,
    pub top_logprobs: Vec<(u32, f32)>,
    pub target_logprobs: Vec<(u32, f32)>,
}

#[derive(Clone, Copy)]
pub struct TopLogprobView<'a> {
    pub indptr: &'a [u32],
    pub token_ids: &'a [u32],
    pub values: &'a [f32],
}

/// CSR score rows may be empty to opt out, otherwise they cover the raw token row.
/// Storage belongs to the runner and remains borrowed until events are constructed.
#[derive(Clone, Copy)]
pub struct LogprobView<'a> {
    pub indptr: &'a [u32],
    pub values: &'a [f32],
    pub top: Option<TopLogprobView<'a>>,
    pub targets: Option<TopLogprobView<'a>>,
}

pub fn validate_indptr(indptr: &[u32], rows: usize, values: usize) -> Result<()> {
    if indptr.len() != rows + 1
        || indptr.first() != Some(&0)
        || indptr.last().map(|&end| end as usize) != Some(values)
        || indptr.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(Error::execution("invalid output CSR offsets"));
    }
    Ok(())
}

impl LogprobView<'_> {
    pub fn validate_required_lengths(&self, lengths: &[usize]) -> Result<()> {
        validate_indptr(self.indptr, lengths.len(), self.values.len())?;
        if self
            .indptr
            .windows(2)
            .zip(lengths)
            .any(|(pair, &count)| (pair[1] - pair[0]) as usize != count)
            || self
                .values
                .iter()
                .any(|value| value.is_nan() || *value > 0.0)
        {
            return Err(Error::execution(
                "prompt logprobs must cover every requested predecessor with non-positive scores",
            ));
        }
        self.validate_columns()
    }

    /// Validate both distribution alternatives and exact per-position targets.
    pub fn validate_prompt(
        &self,
        lengths: &[usize],
        top_counts: &[u32],
        target_indptr: &[u32],
        target_ids: &[u32],
    ) -> Result<()> {
        self.validate_required_lengths(lengths)?;
        if !self.values.is_empty() || !target_indptr.is_empty() || !target_ids.is_empty() {
            validate_indptr(target_indptr, self.values.len(), target_ids.len())?;
        }
        if top_counts.len() != self.values.len() {
            return Err(Error::execution(
                "prompt top counts must cover every predecessor",
            ));
        }
        for (index, &requested) in top_counts.iter().enumerate() {
            let count = self
                .top
                .map_or(0, |top| top.indptr[index + 1] - top.indptr[index]);
            if (requested == 0 && count != 0)
                || (requested > 0 && (count == 0 || count > requested))
            {
                return Err(Error::execution(
                    "runner did not return requested prompt top logprobs",
                ));
            }
            let expected =
                &target_ids[target_indptr[index] as usize..target_indptr[index + 1] as usize];
            let actual = self.targets.map_or(&[][..], |targets| {
                &targets.token_ids
                    [targets.indptr[index] as usize..targets.indptr[index + 1] as usize]
            });
            if actual != expected {
                return Err(Error::execution(
                    "prompt target logprob IDs must match requested positions",
                ));
            }
        }
        Ok(())
    }

    fn validate_columns(&self) -> Result<()> {
        for columns in [self.top, self.targets].into_iter().flatten() {
            validate_indptr(columns.indptr, self.values.len(), columns.values.len())?;
            if columns.token_ids.len() != columns.values.len() {
                return Err(Error::execution(
                    "alternative logprob IDs and scores must align",
                ));
            }
        }
        let valid = |value: &f32| !value.is_nan() && *value <= 0.0;
        if !self.values.iter().all(valid)
            || [self.top, self.targets]
                .into_iter()
                .flatten()
                .any(|columns| !columns.values.iter().all(valid))
        {
            return Err(Error::execution(
                "logprobs must be non-positive and not NaN",
            ));
        }
        Ok(())
    }
    pub fn validate(&self, token_rows: TokenRows<'_>) -> Result<()> {
        validate_indptr(self.indptr, token_rows.len(), self.values.len())?;
        for (row, pair) in token_rows.iter().zip(self.indptr.windows(2)) {
            let count = (pair[1] - pair[0]) as usize;
            if count != 0 && count != row.len() {
                return Err(Error::execution(
                    "logprobs must cover the complete raw token row",
                ));
            }
        }
        self.validate_columns()
    }

    /// Called after validation; only committed events acquire owned score storage.
    pub fn token(&self, row: usize, column: usize) -> Option<TokenLogprobs> {
        let start = self.indptr[row] as usize;
        let end = self.indptr[row + 1] as usize;
        if start == end {
            return None;
        }
        let index = start + column;
        let alternatives = |columns: Option<TopLogprobView<'_>>| {
            columns.map_or_else(Vec::new, |top| {
                let range = top.indptr[index] as usize..top.indptr[index + 1] as usize;
                top.token_ids[range.clone()]
                    .iter()
                    .copied()
                    .zip(top.values[range].iter().copied())
                    .collect()
            })
        };
        Some(TokenLogprobs {
            logprob: self.values[index],
            top_logprobs: alternatives(self.top),
            target_logprobs: alternatives(self.targets),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_targets_preserve_order_and_duplicates_and_validate_every_column() {
        let scores = LogprobView {
            indptr: &[0, 2],
            values: &[-0.5, -1.0],
            top: Some(TopLogprobView {
                indptr: &[0, 1, 1],
                token_ids: &[3],
                values: &[-0.1],
            }),
            targets: Some(TopLogprobView {
                indptr: &[0, 2, 3],
                token_ids: &[9, 9, 4],
                values: &[-2.0, -2.0, f32::NEG_INFINITY],
            }),
        };
        scores
            .validate_prompt(&[2], &[1, 0], &[0, 2, 3], &[9, 9, 4])
            .unwrap();
        assert_eq!(
            scores.token(0, 0).unwrap().target_logprobs,
            [(9, -2.0), (9, -2.0)]
        );
        assert!(
            scores
                .validate_prompt(&[2], &[1, 0], &[0, 2, 3], &[9, 4, 9])
                .is_err()
        );
        assert!(
            scores
                .validate_prompt(&[2], &[0, 0], &[0, 2, 3], &[9, 9, 4])
                .is_err()
        );
        assert!(
            scores
                .validate_prompt(&[2], &[1, 1], &[0, 2, 3], &[9, 9, 4])
                .is_err()
        );
        for invalid in [f32::NAN, f32::INFINITY, 0.1] {
            let invalid_values = [invalid, -2.0, -2.0];
            let invalid_scores = LogprobView {
                targets: Some(TopLogprobView {
                    values: &invalid_values,
                    ..scores.targets.unwrap()
                }),
                ..scores
            };
            assert!(
                invalid_scores
                    .validate_prompt(&[2], &[1, 0], &[0, 2, 3], &[9, 9, 4])
                    .is_err()
            );
        }
        let malformed = LogprobView {
            targets: Some(TopLogprobView {
                indptr: &[0, 3, 2],
                ..scores.targets.unwrap()
            }),
            ..scores
        };
        assert!(
            malformed
                .validate_prompt(&[2], &[1, 0], &[0, 2, 3], &[9, 9, 4])
                .is_err()
        );
    }
}
