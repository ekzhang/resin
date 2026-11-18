use std::{num::NonZeroU32, sync::Arc};

use crate::{
    Phase,
    engine::{Error, Result},
    resources::kv::{KvGroupPlan, KvLayout},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenRange {
    pub start: u32,
    pub count: NonZeroU32,
}

impl TokenRange {
    pub fn end(self) -> Option<u32> {
        self.start.checked_add(self.count.get())
    }
}

impl SequenceStep {
    pub const fn token_start(self) -> u32 {
        self.tokens.start
    }

    pub const fn token_count(self) -> NonZeroU32 {
        self.tokens.count
    }

    pub fn token_end(self) -> Option<u32> {
        self.tokens.end()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceStep {
    pub phase: Phase,
    pub tokens: TokenRange,
    pub sample: bool,
}

/// Numeric execution plan shared by native runners and replica transport.
#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BatchPlan {
    pub cohort: Arc<BatchCohort>,
    pub batch_id: u64,
    pub context_lengths: Vec<u32>,
    pub token_ids: Vec<u32>,
    pub cache_groups: Vec<Arc<KvGroupPlan>>,
}

/// Numerical columns that are fixed for the life of a decode cohort.
#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BatchCohort {
    /// Flattened predecessor rows and their next prompt tokens, partitioned by sequence.
    pub prompt_score_indices: Vec<u32>,
    pub prompt_score_token_ids: Vec<u32>,
    pub prompt_score_indptr: Vec<u32>,
    pub prompt_score_top_counts: Vec<u32>,
    pub prompt_target_indptr: Vec<u32>,
    pub prompt_target_token_ids: Vec<u32>,
    pub batch_key: Option<u64>,
    pub phase: Phase,
    pub sequence_ids: Vec<u64>,
    /// Final prefill-history endpoints, including generated inputs on retraction.
    /// Only meaningful for prefill; constant across that history's chunks.
    pub prefill_lengths: Vec<u32>,
    pub token_indptr: Vec<u32>,
    pub sampling_mask: Vec<bool>,
}

impl std::ops::Deref for BatchPlan {
    type Target = BatchCohort;

    fn deref(&self) -> &BatchCohort {
        &self.cohort
    }
}

const PLAN_WIRE_MAGIC: &[u8; 8] = b"RSNPLAN\0";
// Private same-build protocol. Bump when the archived schema or format changes.
const PLAN_WIRE_VERSION: u32 = 3;
const PLAN_WIRE_HEADER: usize = 20;

impl BatchPlan {
    pub fn encode_wire(&self) -> Result<Vec<u8>> {
        let mut bytes =
            rkyv::api::high::to_bytes_in::<_, rkyv::rancor::Error>(self, vec![0; PLAN_WIRE_HEADER])
                .map_err(|error| Error::invalid(format!("cannot encode plan: {error}")))?;
        let length = (bytes.len() - PLAN_WIRE_HEADER) as u64;
        bytes[..8].copy_from_slice(PLAN_WIRE_MAGIC);
        bytes[8..12].copy_from_slice(&PLAN_WIRE_VERSION.to_le_bytes());
        bytes[12..PLAN_WIRE_HEADER].copy_from_slice(&length.to_le_bytes());
        Ok(bytes)
    }

    pub fn decode_wire(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < PLAN_WIRE_HEADER || &bytes[..8] != PLAN_WIRE_MAGIC {
            return Err(Error::invalid("invalid plan wire format"));
        }
        if u32::from_le_bytes(bytes[8..12].try_into().unwrap()) != PLAN_WIRE_VERSION {
            return Err(Error::invalid("unsupported plan wire version"));
        }
        let length = u64::from_le_bytes(bytes[12..PLAN_WIRE_HEADER].try_into().unwrap());
        if length != (bytes.len() - PLAN_WIRE_HEADER) as u64 {
            return Err(Error::invalid("plan wire length mismatch"));
        }
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(&bytes[PLAN_WIRE_HEADER..])
            .map_err(|error| Error::invalid(format!("invalid plan archive: {error}")))
    }
}

impl BatchPlan {
    pub fn validate(&self) -> Result<()> {
        let Self {
            token_ids,
            context_lengths,
            ..
        } = self;
        let BatchCohort {
            sequence_ids,
            token_indptr,
            prefill_lengths,
            sampling_mask,
            ..
        } = &*self.cohort;
        if token_ids.is_empty() {
            return Err(Error::invalid("a batch must contain at least one token"));
        }
        if token_ids.len() > u32::MAX as usize {
            return Err(Error::invalid(
                "batch token count exceeds the u32 metadata range",
            ));
        }
        if sequence_ids.is_empty() {
            return Err(Error::invalid("a batch must contain at least one sequence"));
        }
        if token_indptr.len() != sequence_ids.len() + 1 {
            return Err(Error::invalid(
                "token_indptr must contain one more item than sequence_ids",
            ));
        }
        for (name, count) in [
            ("context_lengths", context_lengths.len()),
            ("prefill_lengths", prefill_lengths.len()),
            ("sampling_mask", sampling_mask.len()),
        ] {
            if count != sequence_ids.len() {
                return Err(Error::invalid(format!(
                    "{name} must contain one item per sequence"
                )));
            }
        }
        if token_indptr.first() != Some(&0) {
            return Err(Error::invalid("token_indptr must start at zero"));
        }
        if token_indptr.last().copied() != Some(token_ids.len() as u32) {
            return Err(Error::invalid("token_indptr must end at the token count"));
        }
        if token_indptr.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::invalid(
                "each sequence must contain at least one token",
            ));
        }
        if context_lengths
            .iter()
            .zip(token_indptr.windows(2))
            .any(|(context, range)| context.checked_add(range[1] - range[0]).is_none())
        {
            return Err(Error::invalid(
                "scheduled positions exceed the u32 metadata range",
            ));
        }
        if self.phase == Phase::Prefill
            && context_lengths
                .iter()
                .zip(token_indptr.windows(2))
                .zip(prefill_lengths)
                .any(|((context, range), length)| *context + (range[1] - range[0]) > *length)
        {
            return Err(Error::invalid(
                "prefill_lengths must cover every scheduled prefill row",
            ));
        }
        let mut sorted_ids = sequence_ids.clone();
        sorted_ids.sort_unstable();
        if sorted_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::invalid("sequence ids must be unique within a batch"));
        }

        self.validate_cache()?;
        crate::logprobs::validate_indptr(
            &self.prompt_score_indptr,
            sequence_ids.len(),
            self.prompt_score_indices.len(),
        )?;
        if !self.prompt_score_indices.is_empty()
            || !self.prompt_target_indptr.is_empty()
            || !self.prompt_target_token_ids.is_empty()
        {
            crate::logprobs::validate_indptr(
                &self.prompt_target_indptr,
                self.prompt_score_indices.len(),
                self.prompt_target_token_ids.len(),
            )?;
        }
        if self.prompt_score_token_ids.len() != self.prompt_score_indices.len()
            || self.prompt_score_top_counts.len() != self.prompt_score_indices.len()
            || self
                .prompt_score_indptr
                .windows(2)
                .enumerate()
                .any(|(row, range)| {
                    self.prompt_score_indices[range[0] as usize..range[1] as usize]
                        .iter()
                        .any(|&index| index < token_indptr[row] || index >= token_indptr[row + 1])
                })
            || self.phase != Phase::Prefill && !self.prompt_score_indices.is_empty()
        {
            return Err(Error::invalid(
                "prompt score indices and targets must align with their prefill sequence",
            ));
        }

        Ok(())
    }

    fn validate_cache(&self) -> Result<()> {
        if self.cache_groups.is_empty() {
            return Ok(());
        }

        for (group_index, group) in self.cache_groups.iter().enumerate() {
            group
                .validate()
                .map_err(|error| Error::invalid(error.to_string()))?;
            if group.group_index != group_index as u32 {
                return Err(Error::invalid(
                    "cache_groups must be ordered by contiguous group_index",
                ));
            }
            if group.block_table_start_indices.len() != self.sequence_ids.len() {
                return Err(Error::invalid(
                    "each cache group must contain one block-table segment per sequence",
                ));
            }
            if group.append_slab_ids.len() != self.token_ids.len()
                || group.append_slotnums.len() != self.token_ids.len()
            {
                return Err(Error::invalid(
                    "each cache group must contain one append target per batch token",
                ));
            }
            for sequence_index in 0..self.sequence_ids.len() {
                let block_start = group.block_indptr[sequence_index] as usize;
                let block_end = group.block_indptr[sequence_index + 1] as usize;
                let row_slabs = &group.block_slab_ids[block_start..block_end];
                let row_slotnums = &group.block_slotnums[block_start..block_end];
                let token_start = self.token_indptr[sequence_index] as usize;
                let token_end = self.token_indptr[sequence_index + 1] as usize;
                if group.append_slab_ids[token_start..token_end]
                    .iter()
                    .zip(&group.append_slotnums[token_start..token_end])
                    .any(|(append_slab, append_slotnum)| {
                        !row_slabs
                            .iter()
                            .zip(row_slotnums)
                            .any(|(slab, slotnum)| slab == append_slab && slotnum == append_slotnum)
                    })
                {
                    return Err(Error::invalid(
                        "cache append targets must belong to their sequence's block table",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Check a structurally validated batch against the runner's physical layout.
    pub fn validate_cache_layout(&self, layout: &KvLayout, decode_lookahead: u64) -> Result<()> {
        if self.cache_groups.len() != layout.groups().len() {
            return Err(Error::invalid(
                "BatchPlan cache groups do not match CacheLayout",
            ));
        }
        let decode = self.phase == Phase::Decode;
        let lookahead = u128::from(if decode { decode_lookahead } else { 0 });
        let parent_capacity = layout.parent_capacity().get();
        for (index, (group, geometry)) in self.cache_groups.iter().zip(layout.groups()).enumerate()
        {
            let invalid = |message: &str| Error::invalid(format!("cache group {index} {message}"));
            if group.blocks_per_slab != geometry.blocks_per_slab().get() {
                return Err(invalid("has the wrong slab packing"));
            }
            let block_tokens = geometry.block_tokens().get();
            if group
                .block_slab_ids
                .iter()
                .chain(&group.append_slab_ids)
                .any(|&slab| slab > parent_capacity)
                || group
                    .append_block_offsets
                    .iter()
                    .any(|&offset| offset >= block_tokens)
            {
                return Err(invalid("has a physical address outside its layout"));
            }
            match (geometry.is_recurrent(), group.state.is_some()) {
                (true, true) => continue, // Structural validation already checks live STATE slots.
                (true, false) => return Err(invalid("requires private live state")),
                (false, true) => return Err(invalid("is not a STATE group")),
                (false, false) => {}
            }
            let window_blocks = geometry.window_resident_blocks();
            let read_window = geometry.read_window_tokens();
            let write_limit = geometry.max_write_tokens();
            for (row, (&context, tokens)) in self
                .context_lengths
                .iter()
                .zip(self.token_indptr.windows(2))
                .enumerate()
            {
                let query_tokens = tokens[1] - tokens[0];
                let writes = u128::from(query_tokens) + lookahead;
                if let Some(maximum) = write_limit
                    && writes > u128::from(maximum.get())
                {
                    return Err(invalid(&format!(
                        "row {row} write span {writes} exceeds {maximum}-token Window read/write capacity"
                    )));
                }
                let block_base = group.block_indptr[row];
                let block_count = group.block_indptr[row + 1] - block_base;
                let first = group.block_table_start_indices[row];
                let end = u128::from(first) + u128::from(block_count);
                let required = (u128::from(context) + writes).div_ceil(u128::from(block_tokens));
                let expected_first =
                    window_blocks.map_or(0, |window| end.saturating_sub(u128::from(window.get())));
                // Decode may reserve extra future writes, but must still cover today's read window.
                let invalid_extent = if decode {
                    read_window.is_some_and(|window| {
                        first > context.saturating_sub(window.get() - 1) / block_tokens
                    })
                } else {
                    end != required
                };
                if u128::from(first) != expected_first || end < required || invalid_extent {
                    return Err(invalid(&format!("row {row} has an invalid active range")));
                }
                for (token, position) in
                    (tokens[0] as usize..tokens[1] as usize).zip(context..context + query_tokens)
                {
                    let Some(block) = (position / block_tokens).checked_sub(first) else {
                        continue;
                    };
                    let address = (block_base + block) as usize;
                    if group.append_slab_ids[token] != group.block_slab_ids[address]
                        || group.append_slotnums[token] != group.block_slotnums[address]
                        || group.append_block_offsets[token] != position % block_tokens
                    {
                        return Err(invalid(&format!(
                            "row {row} append target is not row-owned"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}
