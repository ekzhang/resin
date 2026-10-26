use std::num::NonZeroU32;

use resin_core::engine::{
    EngineCore, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig, SchedulerConfig,
    StorageImportStart, Submission,
};
use resin_core::resources::kv::{KvGroupLayout, KvLayout, KvRetention};

#[derive(Default)]
struct Model {
    plans: Vec<ExecutionPlan<()>>,
}

impl Runner<()> for Model {
    fn forward(&mut self, plan: ExecutionPlan<()>) -> resin_core::engine::Result<OwnedTokenRows> {
        let ids = plan
            .token_indptr
            .windows(2)
            .map(|range| plan.token_ids[range[1] as usize - 1] + 1)
            .collect::<Vec<_>>();
        let offsets = (0..=ids.len() as u32).collect();
        self.plans.push(plan);
        Ok(OwnedTokenRows::new(ids, offsets).unwrap())
    }
}

fn engine(capacity: u32, running: usize, pending: usize) -> EngineCore<()> {
    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    capacity,
                    vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            continuation_prefill: true,
            ..Default::default()
        },
        running,
        4096,
        SchedulerConfig {
            prefill_chunk_size: NonZeroU32::new(2).unwrap(),
            ..Default::default()
        },
        true,
    )
    .unwrap();
    engine.enable_metrics().unwrap();
    engine.enable_cache_storage(pending).unwrap();
    engine
}

fn input(tokens: Vec<u32>, key: Option<u64>) -> Submission<()> {
    let mut input = Submission::new(tokens, ());
    input.max_output_tokens = Some(1);
    input.batch_key = key;
    input
}

#[test]
fn asynchronous_restore_does_not_block_other_keys_and_reuses_promoted_prefix() {
    let mut engine = engine(32, 2, 4);
    let mut model = Model::default();
    let first = engine.submit(input((1..=7).collect(), Some(1))).unwrap();
    let candidates = engine.storage_candidates().unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].max_prefix_tokens, 6);
    assert_eq!(engine.cache_occupied_block_count(), Some(0));
    let StorageImportStart::Transfer(job) = engine.begin_storage_import(first.id(), 6).unwrap()
    else {
        panic!("expected import");
    };
    let unrelated = engine.submit(input(vec![90], Some(2))).unwrap();
    assert_eq!(
        engine.step(&mut model).unwrap()[0].request_id,
        unrelated.id()
    );
    assert_eq!(unrelated.take_result().unwrap(), [91]);
    assert_eq!(first.cached_tokens(), 0);
    let second = engine.submit(input((1..=7).collect(), Some(1))).unwrap();
    assert_eq!(
        engine.storage_candidates().unwrap()[0].request_id,
        second.id()
    );
    assert!(matches!(
        engine.begin_storage_import(second.id(), 6).unwrap(),
        StorageImportStart::Pending
    ));
    assert_eq!(engine.cache_occupied_block_count(), Some(3));
    assert_eq!(engine.finish_storage_import(job.id, true).unwrap(), 6);
    assert!(matches!(
        engine.begin_storage_import(second.id(), 6).unwrap(),
        StorageImportStart::Resident(6)
    ));
    assert_eq!(engine.cache_occupied_block_count(), Some(3));
    assert_eq!(engine.step(&mut model).unwrap().len(), 2);
    assert_eq!(first.take_result().unwrap(), [8]);
    assert_eq!(second.take_result().unwrap(), [8]);
    assert_eq!((first.cached_tokens(), second.cached_tokens()), (6, 6));
    let metrics = engine.metrics_snapshot().unwrap();
    assert_eq!(
        (metrics.prefill_cache_tokens, metrics.cached_tokens),
        (12, 12)
    );
    assert_eq!(metrics.prefill_compute_tokens, 3);
    assert_eq!(metrics.storage_cached_tokens, 6);
    assert!(!engine.has_work());
    engine.close(&mut model).unwrap();
}

#[test]
fn concurrent_restore_reuses_new_prefix_even_with_shorter_resident_history() {
    let mut engine = engine(32, 2, 4);
    let mut model = Model::default();
    let seed = engine.submit(input(vec![1, 2], None)).unwrap();
    engine.step(&mut model).unwrap();
    assert!(seed.done());
    for export in engine.take_storage_exports() {
        engine.finish_storage_export(export.id).unwrap();
    }
    let first = engine.submit(input((1..=7).collect(), None)).unwrap();
    let second = engine.submit(input((1..=7).collect(), None)).unwrap();
    assert_eq!(engine.storage_candidates().unwrap().len(), 2);
    assert_eq!((first.cached_tokens(), second.cached_tokens()), (2, 2));
    assert!(engine.begin_storage_import(first.id(), 2).is_err());
    let StorageImportStart::Transfer(job) = engine.begin_storage_import(first.id(), 6).unwrap()
    else {
        panic!("expected import");
    };
    assert!(matches!(
        engine.begin_storage_import(second.id(), 6).unwrap(),
        StorageImportStart::Pending
    ));
    engine.finish_storage_import(job.id, true).unwrap();
    assert!(matches!(
        engine.begin_storage_import(second.id(), 6).unwrap(),
        StorageImportStart::Resident(6)
    ));
    // One old checkpoint page and one promoted three-page snapshot; no second
    // private copy of that snapshot is needed for the concurrent request.
    assert_eq!(engine.cache_occupied_block_count(), Some(4));
    assert_eq!(engine.step(&mut model).unwrap().len(), 2);
    assert_eq!(engine.metrics_snapshot().unwrap().prefill_cache_tokens, 12);
    assert_eq!(engine.metrics_snapshot().unwrap().storage_cached_tokens, 4);
    engine.close(&mut model).unwrap();
}

#[test]
fn metadata_lookup_batch_is_independent_of_transfer_budget_and_cancel_holds_import() {
    let mut engine = engine(32, 32, 2);
    let mut model = Model::default();
    let requests = (0..32)
        .map(|branch| engine.submit(input(vec![branch; 5], None)).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(engine.storage_candidates().unwrap().len(), 32);
    assert_eq!(engine.cache_occupied_block_count(), Some(0));
    let StorageImportStart::Transfer(first) =
        engine.begin_storage_import(requests[0].id(), 4).unwrap()
    else {
        panic!("expected import");
    };
    let StorageImportStart::Transfer(second) =
        engine.begin_storage_import(requests[1].id(), 4).unwrap()
    else {
        panic!("expected import");
    };
    assert!(matches!(
        engine.begin_storage_import(requests[2].id(), 4).unwrap(),
        StorageImportStart::Pending
    ));
    assert_eq!(engine.metrics_snapshot().unwrap().running_requests, 2);
    assert!(engine.cancel(&mut model, &requests[0]).unwrap());
    assert_eq!(engine.cache_occupied_block_count(), Some(4));
    assert_eq!(engine.finish_storage_import(first.id, true).unwrap(), 0);
    assert_eq!(engine.cache_occupied_block_count(), Some(2));
    assert_eq!(engine.finish_storage_import(second.id, false).unwrap(), 0);
    assert_eq!(requests[1].cached_tokens(), 0);
    engine
        .storage_lookup_miss(
            &requests[2..]
                .iter()
                .map(|request| request.id())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert!(engine.storage_candidates().unwrap().is_empty());
    assert!(!engine.storage_has_work());
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(32));
}

#[test]
fn exact_capacity_prefill_exports_and_cold_restore_rebuilds_only_suffix() {
    let mut engine = engine(3, 1, 4);
    let mut model = Model::default();
    let warm = engine.submit(input((1..=6).collect(), None)).unwrap();
    engine.storage_candidates().unwrap();
    engine.storage_lookup_miss(&[warm.id()]).unwrap();
    let mut held = None;
    while !warm.done() {
        engine.step(&mut model).unwrap();
        for job in engine.take_storage_exports() {
            if job.transfer.resident_tokens == 6 {
                held = Some(job);
            } else {
                engine.finish_storage_export(job.id).unwrap();
            }
        }
    }
    let held = held.expect("a full-capacity checkpoint must be exported");
    assert_eq!(held.tokens, (1..=6).collect::<Vec<_>>());
    assert_eq!(engine.cache_free_parent_count(), Some(0));
    assert!(engine.has_work());
    assert!(engine.close(&mut model).is_err());
    assert!(!engine.poisoned());
    engine.clear_prefix_cache().unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(0));
    engine.finish_storage_export(held.id).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(3));

    let cold = engine.submit(input((1..=6).collect(), None)).unwrap();
    let candidates = engine.storage_candidates().unwrap();
    assert_eq!(candidates[0].max_prefix_tokens, 4);
    let StorageImportStart::Transfer(job) = engine.begin_storage_import(cold.id(), 4).unwrap()
    else {
        panic!("expected import");
    };
    assert_eq!(engine.finish_storage_import(job.id, true).unwrap(), 4);
    engine.step(&mut model).unwrap();
    assert_eq!(model.plans.last().unwrap().token_ids, [5, 6]);
    assert_eq!(cold.take_result().unwrap(), [7]);
    assert_eq!(cold.cached_tokens(), 4);
    for job in engine.take_storage_exports() {
        engine.finish_storage_export(job.id).unwrap();
    }
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(3));
}

#[test]
fn randomized_storage_scheduling_preserves_capacity_and_completes_cancellations() {
    let mut engine = engine(24, 3, 4);
    let mut model = Model::default();
    let mut requests = Vec::new();
    let mut lookups = Vec::new();
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut seed = 0x810c_befd_9327_0401u64;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..1000 {
        if random() % 5 == 0 && engine.active_request_count() < 10 {
            let branch = (random() % 16) as u32;
            requests.push(engine.submit(input(vec![branch; 5], None)).unwrap());
        }
        lookups.extend(engine.storage_candidates().unwrap());
        match random() % 6 {
            0 if !lookups.is_empty() => {
                let candidate = lookups.swap_remove(random() as usize % lookups.len());
                engine.storage_lookup_miss(&[candidate.request_id]).unwrap();
            }
            1 if !lookups.is_empty() => {
                let candidate = lookups.swap_remove(random() as usize % lookups.len());
                match engine
                    .begin_storage_import(candidate.request_id, candidate.max_prefix_tokens)
                    .unwrap()
                {
                    StorageImportStart::Transfer(job) => imports.push(job.id),
                    StorageImportStart::Pending => lookups.push(candidate),
                    _ => {}
                }
            }
            2 if !imports.is_empty() => {
                let id = imports.swap_remove(random() as usize % imports.len());
                engine.finish_storage_import(id, random() % 3 != 0).unwrap();
            }
            3 if !exports.is_empty() => {
                let id = exports.swap_remove(random() as usize % exports.len());
                engine.finish_storage_export(id).unwrap();
            }
            4 if !requests.is_empty() => {
                engine
                    .cancel(&mut model, &requests[random() as usize % requests.len()])
                    .unwrap();
            }
            _ => {
                engine.step(&mut model).unwrap();
            }
        }
        exports.extend(engine.take_storage_exports().into_iter().map(|job| job.id));
        assert!(engine.metrics_snapshot().unwrap().running_requests <= 3);
        assert!(engine.cache_metrics().unwrap().occupied_slabs <= 24);
    }
    for id in imports {
        engine.finish_storage_import(id, false).unwrap();
    }
    engine
        .storage_lookup_miss(
            &lookups
                .iter()
                .map(|candidate| candidate.request_id)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for id in exports {
        engine.finish_storage_export(id).unwrap();
    }
    engine.close(&mut model).unwrap();
    assert!(requests.iter().all(|request| request.done()));
    assert_eq!(engine.cache_free_parent_count(), Some(24));
}
