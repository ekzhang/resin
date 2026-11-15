use std::fmt;
use std::num::NonZeroU32;

use crate::{Phase, SequenceStep, TokenRange};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanningConfig {
    pub max_batch_tokens: Option<NonZeroU32>,
    pub prefill_chunk_size: NonZeroU32,
    pub checkpoint_alignment: Option<NonZeroU32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestProgress {
    pub phase: Phase,
    pub prefill_length: u32,
    pub logical_context: u32,
    pub sample_at_prefill_end: bool,
    /// Half-open token ranges which must be processed in one prefill row.
    pub atomic_prefill_spans: Vec<(u32, u32)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanningError {
    MixedPhases,
    InvalidPrefillProgress,
    AtomicPrefillSpanExceedsBudget,
    TokenRangeOverflow,
}

impl fmt::Display for PlanningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MixedPhases => {
                formatter.write_str("scheduler selected requests from different phases")
            }
            Self::InvalidPrefillProgress => {
                formatter.write_str("prefill cursor must be before prompt length")
            }
            Self::AtomicPrefillSpanExceedsBudget => {
                formatter.write_str("an atomic prefill span exceeds the current row token budget")
            }
            Self::TokenRangeOverflow => formatter.write_str("scheduled token range overflows u32"),
        }
    }
}

impl std::error::Error for PlanningError {}

/// Choose the token range for each request in the current dispatch.
///
pub fn schedule_ready_prefix(
    requests: &[RequestProgress],
    config: PlanningConfig,
) -> Result<Vec<SequenceStep>, PlanningError> {
    let Some(first) = requests.first() else {
        return Ok(Vec::new());
    };
    if requests.iter().any(|request| request.phase != first.phase) {
        return Err(PlanningError::MixedPhases);
    }

    let mut remaining_batch_tokens = config.max_batch_tokens.map(NonZeroU32::get);
    let mut steps = Vec::with_capacity(requests.len());
    for request in requests {
        if remaining_batch_tokens == Some(0) {
            break;
        }
        let desired = match request.phase {
            Phase::Prefill => request
                .prefill_length
                .checked_sub(request.logical_context)
                .filter(|remaining| *remaining != 0)
                .ok_or(PlanningError::InvalidPrefillProgress)?
                .min(config.prefill_chunk_size.get()),
            Phase::Decode => 1,
        };
        let count = remaining_batch_tokens.map_or(desired, |budget| desired.min(budget));
        let hard_end = request
            .logical_context
            .checked_add(count)
            .ok_or(PlanningError::TokenRangeOverflow)?;
        let mut end = hard_end;
        if request.phase == Phase::Prefill
            && let Some(alignment) = config.checkpoint_alignment
        {
            // A cached request must replay at least one token to recover its
            // logits. At prompt end, therefore stop at the final checkpoint
            // strictly before the input boundary. Intermediate chunks merely
            // round down to a checkpoint without forcing tiny per-block calls.
            let checkpoint_limit = if hard_end == request.prefill_length {
                hard_end.saturating_sub(1)
            } else {
                hard_end
            };
            let checkpoint_end = checkpoint_limit / alignment * alignment.get();
            if checkpoint_end > request.logical_context && checkpoint_end < hard_end {
                end = checkpoint_end;
            }
        }
        if request.phase == Phase::Prefill {
            for &(span_start, span_end) in &request.atomic_prefill_spans {
                if span_start < end && end < span_end {
                    if span_end <= hard_end {
                        end = span_end;
                    } else if span_start > request.logical_context {
                        end = span_start;
                    } else if steps.is_empty() {
                        return Err(PlanningError::AtomicPrefillSpanExceedsBudget);
                    } else {
                        return Ok(steps);
                    }
                    break;
                }
            }
        }
        let count = NonZeroU32::new(end - request.logical_context)
            .expect("atomic spans preserve positive request work");
        steps.push(SequenceStep {
            phase: request.phase,
            tokens: TokenRange {
                start: request.logical_context,
                count,
            },
            sample: match request.phase {
                Phase::Prefill => request.sample_at_prefill_end && end == request.prefill_length,
                Phase::Decode => true,
            },
        });
        if let Some(remaining) = &mut remaining_batch_tokens {
            *remaining -= count.get();
        }
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefill(prefill_length: u32, logical_context: u32) -> RequestProgress {
        RequestProgress {
            phase: Phase::Prefill,
            prefill_length,
            logical_context,
            sample_at_prefill_end: true,
            atomic_prefill_spans: Vec::new(),
        }
    }

    #[test]
    fn token_budget_and_chunk_limit_compose() {
        let steps = schedule_ready_prefix(
            &[prefill(4, 0), prefill(4, 0)],
            PlanningConfig {
                max_batch_tokens: NonZeroU32::new(3),
                prefill_chunk_size: NonZeroU32::new(2).unwrap(),
                checkpoint_alignment: None,
            },
        )
        .unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].tokens.count.get(), 2);
        assert_eq!(steps[1].tokens.count.get(), 1);
        assert!(!steps[0].sample);
        assert!(!steps[1].sample);
    }

    #[test]
    fn token_budget_is_consumed_in_request_order() {
        let steps = schedule_ready_prefix(
            &[prefill(4, 0), prefill(4, 0)],
            PlanningConfig {
                max_batch_tokens: NonZeroU32::new(4),
                prefill_chunk_size: NonZeroU32::new(8).unwrap(),
                checkpoint_alignment: None,
            },
        )
        .unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].tokens.count.get(), 4);
        assert!(steps[0].sample);
    }

    #[test]
    fn prefills_are_not_limited_by_cache_page_boundaries() {
        let steps = schedule_ready_prefix(
            &[prefill(100, 30)],
            PlanningConfig {
                max_batch_tokens: None,
                prefill_chunk_size: NonZeroU32::new(50).unwrap(),
                checkpoint_alignment: None,
            },
        )
        .unwrap();
        assert_eq!(steps[0].tokens.count.get(), 50);
    }

    #[test]
    fn recurrent_prefill_stops_once_at_the_last_checkpoint_before_prompt_end() {
        let first = schedule_ready_prefix(
            &[prefill(10, 0)],
            PlanningConfig {
                max_batch_tokens: None,
                prefill_chunk_size: NonZeroU32::new(32).unwrap(),
                checkpoint_alignment: NonZeroU32::new(4),
            },
        )
        .unwrap();
        assert_eq!(first[0].token_end(), Some(8));
        assert!(!first[0].sample);

        let suffix = schedule_ready_prefix(
            &[prefill(10, 8)],
            PlanningConfig {
                max_batch_tokens: None,
                prefill_chunk_size: NonZeroU32::new(32).unwrap(),
                checkpoint_alignment: NonZeroU32::new(4),
            },
        )
        .unwrap();
        assert_eq!(suffix[0].token_end(), Some(10));
        assert!(suffix[0].sample);

        let exact = schedule_ready_prefix(
            &[prefill(8, 0)],
            PlanningConfig {
                max_batch_tokens: None,
                prefill_chunk_size: NonZeroU32::new(32).unwrap(),
                checkpoint_alignment: NonZeroU32::new(4),
            },
        )
        .unwrap();
        assert_eq!(exact[0].token_end(), Some(4));
        assert!(!exact[0].sample);
    }

    #[test]
    fn checkpoint_rounding_never_splits_an_atomic_prefill_span() {
        let mut request = prefill(284, 0);
        request.atomic_prefill_spans = vec![(14, 280)];
        let first = schedule_ready_prefix(
            &[request],
            PlanningConfig {
                max_batch_tokens: NonZeroU32::new(4096),
                prefill_chunk_size: NonZeroU32::new(4096).unwrap(),
                checkpoint_alignment: NonZeroU32::new(64),
            },
        )
        .unwrap();

        assert_eq!(first[0].tokens.start, 0);
        assert_eq!(first[0].token_end(), Some(280));
        assert!(!first[0].sample);
    }

    #[test]
    fn replay_prefill_does_not_sample_an_already_generated_token() {
        let steps = schedule_ready_prefix(
            &[RequestProgress {
                phase: Phase::Prefill,
                prefill_length: 6,
                logical_context: 0,
                sample_at_prefill_end: false,
                atomic_prefill_spans: Vec::new(),
            }],
            PlanningConfig {
                max_batch_tokens: None,
                prefill_chunk_size: NonZeroU32::new(8).unwrap(),
                checkpoint_alignment: None,
            },
        )
        .unwrap();
        assert_eq!(steps[0].token_end(), Some(6));
        assert!(!steps[0].sample);
    }

    #[test]
    fn phases_must_be_homogeneous() {
        let decode = RequestProgress {
            phase: Phase::Decode,
            prefill_length: 4,
            logical_context: 4,
            sample_at_prefill_end: false,
            atomic_prefill_spans: Vec::new(),
        };
        assert_eq!(
            schedule_ready_prefix(
                &[prefill(4, 0), decode],
                PlanningConfig {
                    max_batch_tokens: None,
                    prefill_chunk_size: NonZeroU32::new(4).unwrap(),
                    checkpoint_alignment: None,
                },
            ),
            Err(PlanningError::MixedPhases)
        );
    }
}
