//! Native plan conversion and validation against the runner's cache layout.
use crate::{BatchPhase, BatchPlan, RequestPayload, cache::CacheGroupPlan};
use pyo3::{PyResult, exceptions::PyValueError};
use resin_core::{Phase, engine::ExecutionPlan, resources::kv::KvLayout};
use std::sync::OnceLock;

pub(super) fn from_core(plan: ExecutionPlan<RequestPayload>) -> BatchPlan {
    let mut sampling_params = Vec::with_capacity(plan.payloads.len());
    let mut media = Vec::with_capacity(plan.payloads.len());
    for payload in plan.payloads {
        sampling_params.push(payload.sampling_params);
        media.push(payload.media);
    }
    // The core constructs this plan from validated request and allocator state.
    // Public Python constructors still validate untrusted caller-provided arrays.
    BatchPlan {
        prompt_score_indices: plan.prompt_score_indices,
        prompt_score_token_ids: plan.prompt_score_token_ids,
        prompt_score_indptr: plan.prompt_score_indptr,
        prompt_score_top_counts: plan.prompt_score_top_counts,
        prompt_target_indptr: plan.prompt_target_indptr,
        prompt_target_token_ids: plan.prompt_target_token_ids,
        sequence_ids: plan.sequence_ids,
        context_lengths: plan.context_lengths,
        prefill_lengths: plan.prefill_lengths,
        token_ids: plan.token_ids,
        token_indptr: plan.token_indptr,
        phase: match plan.phase {
            Phase::Prefill => BatchPhase::Prefill,
            Phase::Decode => BatchPhase::Decode,
        },
        batch_id: plan.batch_id,
        batch_key: plan.batch_key,
        cache_groups: plan
            .cache_groups
            .into_iter()
            .map(CacheGroupPlan::from_inner)
            .collect(),
        sampling_mask: plan.sampling_mask,
        sampling_params,
        media,
        positions: OnceLock::new(),
    }
}

/// Check a structurally validated batch against the runner's physical layout.
pub(super) fn validate_cache_layout(
    plan: &BatchPlan,
    layout: &KvLayout,
    decode_lookahead: u64,
) -> PyResult<()> {
    if plan.cache_groups.len() != layout.groups().len() {
        return Err(PyValueError::new_err(
            "BatchPlan cache groups do not match CacheLayout",
        ));
    }
    let decode = plan.phase == BatchPhase::Decode;
    let lookahead = u128::from(if decode { decode_lookahead } else { 0 });
    let parent_capacity = layout.parent_capacity().get();
    for (index, (group, geometry)) in plan.cache_groups.iter().zip(layout.groups()).enumerate() {
        let group = group.inner_native();
        let invalid =
            |message: &str| PyValueError::new_err(format!("cache group {index} {message}"));
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
        for (row, (&context, tokens)) in plan
            .context_lengths
            .iter()
            .zip(plan.token_indptr.windows(2))
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
