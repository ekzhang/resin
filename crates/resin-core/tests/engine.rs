use resin_core::engine::{
    EngineCore, Error, ErrorKind, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig,
    SchedulerConfig, Submission, TokenRows,
};
use resin_core::resources::kv::{KvGroupFamily, KvGroupLayout, KvLayout, KvRetention};
use std::collections::VecDeque;
use std::num::NonZeroU32;

#[derive(Default)]
struct Model {
    plans: Vec<ExecutionPlan<&'static str>>,
    outputs: VecDeque<Result<Vec<Vec<u32>>, Error>>,
    released: Vec<u64>,
    release_fails: bool,
    report_checkpoints: bool,
    checkpoints: Vec<u64>,
}
impl Runner<&'static str> for Model {
    fn forward(&mut self, plan: ExecutionPlan<&'static str>) -> Result<OwnedTokenRows, Error> {
        let fallback = plan
            .token_indptr
            .windows(2)
            .map(|range| vec![plan.token_ids[range[1] as usize - 1] + 1])
            .collect();
        let output = self.outputs.pop_front().unwrap_or(Ok(fallback));
        self.checkpoints.clear();
        if self.report_checkpoints
            && let Ok(rows) = &output
        {
            for group in &plan.cache_groups {
                if let Some(state) = &group.state {
                    for slot in &state.checkpoint_slots {
                        let row = slot.row as usize;
                        let input_count = plan.token_indptr[row + 1] - plan.token_indptr[row];
                        let valid = plan.context_lengths[row]
                            + input_count
                            + if plan.phase == resin_core::Phase::Decode {
                                rows[row].len() as u32 - 1
                            } else {
                                0
                            };
                        if valid == slot.prefix_length
                            && !self.checkpoints.contains(&slot.reservation_id)
                        {
                            self.checkpoints.push(slot.reservation_id);
                        }
                    }
                }
            }
        }
        self.plans.push(plan);
        output.map(owned_rows)
    }
    fn state_checkpoints(&self) -> &[u64] {
        &self.checkpoints
    }
    fn release(&mut self, ids: &[u64]) -> Result<(), Error> {
        if self.release_fails {
            return Err(Error::execution("release failed"));
        }
        self.released.extend(ids);
        Ok(())
    }
}
fn owned_rows(rows: Vec<Vec<u32>>) -> OwnedTokenRows {
    let mut ids = Vec::new();
    let mut offsets = vec![0];
    for row in rows {
        ids.extend(row);
        offsets.push(u32::try_from(ids.len()).unwrap());
    }
    OwnedTokenRows::new(ids, offsets).unwrap()
}
fn engine(lookahead: u32, chunk: u32) -> EngineCore<&'static str> {
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    16,
                    vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            continuation_prefill: true,
            max_accepted_per_step: lookahead + 1,
            ..Default::default()
        },
        4,
        128,
        SchedulerConfig {
            prefill_chunk_size: NonZeroU32::new(chunk).unwrap(),
            ..Default::default()
        },
        false,
    )
    .unwrap()
}
fn input(tokens: Vec<u32>, limit: usize) -> Submission<&'static str> {
    let mut input = Submission::new(tokens, "opaque model metadata");
    input.max_output_tokens = Some(limit);
    input
}

#[test]
fn admission_credits_restored_physical_prefix_pages_only() {
    for cache_hit in [true, false] {
        let mut engine = EngineCore::new(
            RunnerConfig {
                cache_layout: Some(
                    KvLayout::new(
                        8,
                        vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
                    )
                    .unwrap(),
                ),
                continuation_prefill: true,
                ..Default::default()
            },
            4,
            32,
            SchedulerConfig {
                max_batch_tokens: NonZeroU32::new(32),
                prefill_chunk_size: NonZeroU32::new(32).unwrap(),
                ..Default::default()
            },
            true,
        )
        .unwrap();
        engine.enable_metrics().unwrap();
        let mut model = Model::default();
        let prefix: Vec<u32> = (0..8).collect();
        let mut submission = input(prefix.clone(), 1);
        submission.cache_salt = Some("a".into());
        let seed = engine.submit(submission).unwrap();
        while engine.has_work() {
            engine.step(&mut model).unwrap();
        }
        seed.take_result().unwrap();
        let requests: Vec<_> = (0..4)
            .map(|index| {
                let mut prompt = prefix.clone();
                prompt.extend([100 + index, 200]);
                let mut submission = input(prompt, 1);
                submission.cache_salt = Some(if cache_hit { "a" } else { "b" }.into());
                engine.submit(submission).unwrap()
            })
            .collect();
        let events = engine.step(&mut model).unwrap();
        let plan = model.plans.last().unwrap();
        assert_eq!(plan.sequence_ids.len(), if cache_hit { 4 } else { 1 });
        if cache_hit {
            assert_eq!(events.len(), 4);
            assert_eq!(plan.context_lengths, [8; 4]);
            assert_eq!(plan.token_ids.len(), 8);
            let tables: Vec<_> = plan.cache_groups[0]
                .block_slab_ids
                .as_chunks::<5>()
                .0
                .iter()
                .collect();
            assert_eq!(tables.len(), 4);
            for table in &tables[1..] {
                assert_eq!(&table[..4], &tables[0][..4]);
            }
            assert_eq!(
                tables
                    .iter()
                    .map(|table| table[4])
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                4
            );
            for request in requests {
                request.take_result().unwrap();
            }
            let metrics = engine.metrics_snapshot().unwrap();
            assert_eq!(metrics.cached_tokens, 32);
            assert_eq!(metrics.prefill_cache_tokens, 32);
            assert_eq!(metrics.prefill_compute_tokens, 16);
        }
        engine.close(&mut model).unwrap();
    }
}

#[test]
fn uncacheable_requests_skip_prefix_lookup_and_publish() {
    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    8,
                    vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            continuation_prefill: true,
            ..Default::default()
        },
        4,
        32,
        SchedulerConfig {
            max_batch_tokens: NonZeroU32::new(32),
            prefill_chunk_size: NonZeroU32::new(32).unwrap(),
            ..Default::default()
        },
        true,
    )
    .unwrap();
    let mut model = Model::default();
    let mut private = input(vec![1, 2, 3, 4], 1);
    private.cacheable = false;
    let first = engine.submit(private).unwrap();
    assert_eq!(first.cached_tokens(), 0);
    while engine.has_work() {
        engine.step(&mut model).unwrap();
    }
    first.take_result().unwrap();
    assert_eq!(engine.prefix_cache_stats(), Some((0, 0, 0, 0)));

    let second = engine.submit(input(vec![1, 2, 3, 4, 5], 1)).unwrap();
    assert_eq!(second.cached_tokens(), 0);
    while engine.has_work() {
        engine.step(&mut model).unwrap();
    }
    second.take_result().unwrap();
    let published = engine.prefix_cache_stats().unwrap();
    assert!(published.0 > 0, "cacheable request should publish prefixes");
    assert_eq!(published.1, 0);

    let mut skipped = input(vec![1, 2, 3, 4, 5, 6], 1);
    skipped.cacheable = false;
    let third = engine.submit(skipped).unwrap();
    assert_eq!(third.cached_tokens(), 0);
    assert_eq!(engine.prefix_cache_stats().unwrap().1, 0);
    while engine.has_work() {
        engine.step(&mut model).unwrap();
    }
    third.take_result().unwrap();

    let fourth = engine.submit(input(vec![1, 2, 3, 4, 5], 1)).unwrap();
    assert!(fourth.cached_tokens() > 0);
    assert_eq!(engine.prefix_cache_stats().unwrap().1, 1);
    engine.close(&mut model).unwrap();
}

fn check_stopped_speculative_prefix(stop_token: bool) {
    for family in [KvGroupFamily::State, KvGroupFamily::History] {
        for truncated in [true, false] {
            let mut groups = vec![
                KvGroupLayout::new_with_family(
                    if family == KvGroupFamily::State { 2 } else { 4 },
                    1,
                    KvRetention::FullContext,
                    family,
                )
                .unwrap(),
            ];
            if family == KvGroupFamily::State {
                // Hybrid attention: the STATE group need not be the first.
                groups.insert(
                    0,
                    KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap(),
                );
            }
            let mut engine = EngineCore::new(
                RunnerConfig {
                    cache_layout: Some(KvLayout::new(16, groups).unwrap()),
                    continuation_prefill: true,
                    max_accepted_per_step: 4,
                    ..Default::default()
                },
                1,
                32,
                SchedulerConfig::default(),
                true,
            )
            .unwrap();
            let mut model = Model {
                report_checkpoints: true,
                ..Default::default()
            };
            let mut submission = input(vec![10, 11, 12], if stop_token { 8 } else { 2 });
            submission.cache_salt = Some("a".into());
            if stop_token {
                submission.stop_token_ids = vec![14];
            }
            let request = engine.submit(submission).unwrap();
            while engine.next_phase().unwrap() == Some(resin_core::Phase::Prefill) {
                engine.step(&mut model).unwrap();
            } // Prompt state at3, sampled anchor13; STATE also cached prefix2.
            model.outputs.push_back(Ok(vec![if truncated {
                vec![14, 15, 16]
            } else {
                vec![14]
            }]));
            engine.step(&mut model).unwrap();
            if family == KvGroupFamily::State && truncated {
                // The GPU really materialized the reserved raw end6, even
                // though EOS/output limits retain only the prefix through4.
                assert_eq!(model.checkpoints.len(), 1);
                let state = model
                    .plans
                    .last()
                    .unwrap()
                    .cache_groups
                    .iter()
                    .find_map(|group| group.state.as_ref())
                    .unwrap();
                let slot = state
                    .checkpoint_slots
                    .iter()
                    .find(|slot| slot.reservation_id == model.checkpoints[0])
                    .unwrap();
                assert_eq!(slot.prefix_length, 6);
            }
            assert_eq!(request.take_result().unwrap(), [13, 14]);
            assert!(!engine.has_work());

            // The logical history ends at resident token 4, but a truncated
            // physical block committed state at 6. A token-addressed cache can
            // reuse its first four entries; a final-only recurrent checkpoint
            // cannot supply state at that intermediate boundary.
            let mut continuation = input(vec![10, 11, 12, 13, 99], 1);
            continuation.cache_salt = Some("a".into());
            let resumed = engine.submit(continuation).unwrap();
            let expected = if family == KvGroupFamily::State && truncated {
                2 // The earlier authentic prefill checkpoint remains usable.
            } else {
                4
            };
            assert_eq!(
                resumed.cached_tokens(),
                expected,
                "family={family:?}, truncated={truncated}, stop_token={stop_token}"
            );
            engine.step(&mut model).unwrap();
            assert_eq!(model.plans.last().unwrap().context_lengths, [expected]);
            while engine.has_work() {
                engine.step(&mut model).unwrap();
            }
            resumed.take_result().unwrap();
            let mut cold = input(vec![10, 11, 12, 13, 99], 1);
            cold.cache_salt = Some("b".into());
            assert_eq!(engine.submit(cold).unwrap().cached_tokens(), 0);
            engine.close(&mut model).unwrap();
        }
    }
}

#[test]
fn eos_inside_speculative_block_does_not_publish_intermediate_state() {
    check_stopped_speculative_prefix(true);
}

#[test]
fn output_limit_inside_speculative_block_does_not_publish_intermediate_state() {
    check_stopped_speculative_prefix(false);
}

#[test]
fn a_native_model_uses_the_complete_lifecycle_without_python() {
    let mut engine = engine(0, 2);
    let mut model = Model::default();
    let request = engine.submit(input(vec![1, 2, 3, 4, 5], 3)).unwrap();
    assert!(engine.step(&mut model).unwrap().is_empty());
    assert!(engine.step(&mut model).unwrap().is_empty());
    let mut events = Vec::new();
    while engine.has_work() {
        events.extend(engine.step(&mut model).unwrap());
    }
    assert_eq!(
        events
            .iter()
            .map(|event| event.token_id)
            .collect::<Vec<_>>(),
        [6, 7, 8]
    );
    assert_eq!(
        model
            .plans
            .iter()
            .map(|plan| plan.context_lengths[0])
            .collect::<Vec<_>>(),
        [0, 2, 4, 5, 6]
    );
    assert!(
        model
            .plans
            .iter()
            .all(|plan| plan.payloads == ["opaque model metadata"])
    );
    assert_eq!(request.take_result().unwrap(), [6, 7, 8]);
    assert!(request.take_result().is_err());
    assert_eq!(model.released, [request.id()]);
    assert_eq!(engine.cache_free_parent_count(), Some(16));
    engine.close(&mut model).unwrap();
    assert!(engine.closed());
}

#[test]
fn rejected_execution_restores_requests_and_releases_batch_access() {
    let mut engine = engine(0, 8);
    let mut model = Model::default();
    model
        .outputs
        .push_back(Err(Error::rejected("no state changed")));
    let first = engine.submit(input(vec![1, 2], 1)).unwrap();
    let second = engine.submit(input(vec![3, 4], 1)).unwrap();
    assert_eq!(
        engine.step(&mut model).unwrap_err().kind,
        ErrorKind::Rejected
    );
    assert!(!engine.poisoned());
    assert_eq!(engine.ready_request_count(), 2);
    engine.step(&mut model).unwrap();
    assert_eq!(model.plans[0].sequence_ids, model.plans[1].sequence_ids);
    assert!(model.plans[1].batch_id > model.plans[0].batch_id);
    assert_eq!(model.plans[0].cache_groups, model.plans[1].cache_groups);
    assert_eq!(first.take_result().unwrap(), [3]);
    assert_eq!(second.take_result().unwrap(), [5]);
}

#[test]
fn stop_tokens_are_request_local_and_empty_stops_generate_to_the_limit() {
    let mut engine = engine(0, 8);
    let mut model = Model::default();
    let unbounded = engine.submit(input(vec![1], 3)).unwrap();
    let mut stopping_input = input(vec![1], 3);
    stopping_input.stop_token_ids = vec![99, 2, 99];
    let stopping = engine.submit(stopping_input).unwrap();
    let first = engine.step(&mut model).unwrap();
    let stopped = first
        .iter()
        .find(|event| event.request_id == stopping.id())
        .unwrap();
    assert_eq!(
        stopped.finish_reason,
        Some(resin_core::engine::CompletionReason::Stop)
    );
    assert!(!unbounded.done());
    assert_eq!(stopping.take_result().unwrap(), [2]);

    engine.step(&mut model).unwrap();
    let last = engine.step(&mut model).unwrap();
    assert_eq!(
        last[0].finish_reason,
        Some(resin_core::engine::CompletionReason::Length)
    );
    assert_eq!(unbounded.take_result().unwrap(), [2, 3, 4]);
    assert!(!engine.has_work());
}

#[test]
fn terminal_limits_do_not_hide_unleased_runner_output() {
    for stops in [vec![], vec![3]] {
        let mut engine = engine(0, 8);
        let mut model = Model::default();
        let mut request_input = input(vec![1], 2);
        request_input.stop_token_ids = stops;
        let request = engine.submit(request_input).unwrap();
        engine.step(&mut model).unwrap();
        model.outputs.push_back(Ok(vec![vec![3, 4, 5]]));
        assert!(
            engine
                .step(&mut model)
                .unwrap_err()
                .to_string()
                .contains("declared output maximum")
        );
        assert!(engine.poisoned());
        assert!(!request.done());
        assert!(model.released.is_empty());
        assert!(engine.close(&mut model).is_err());
    }
}

#[test]
fn frontend_event_failure_quarantines_before_native_commit() {
    let mut engine = engine(0, 8);
    engine.enable_metrics().unwrap();
    let mut model = Model::default();
    let request = engine.submit(input(vec![1, 2], 1)).unwrap();
    let error = engine
        .step_with::<()>(&mut model, |_| {
            Err(Error::memory("event allocation failed"))
        })
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Memory);
    assert!(engine.poisoned());
    assert!(!request.done());
    assert!(model.released.is_empty());
    assert_eq!(engine.cache_occupied_block_count(), Some(1));
    let metrics = engine.metrics_snapshot().unwrap();
    assert_eq!(metrics.running_requests, 1);
    assert_eq!(metrics.completed_requests, 0);
    assert_eq!(metrics.prefill_compute_tokens, 0);
    assert_eq!(metrics.time_to_first_token_seconds.count, 0);
}

#[test]
fn failed_release_keeps_request_ownership_quarantined() {
    let mut engine = engine(0, 8);
    let mut model = Model {
        release_fails: true,
        ..Default::default()
    };
    let request = engine.submit(input(vec![1], 1)).unwrap();
    assert!(engine.step(&mut model).is_err());
    assert!(engine.poisoned());
    assert!(!request.done());
    assert_eq!(engine.cache_occupied_block_count(), Some(1));
}

#[test]
fn handle_allocation_failure_does_not_submit_a_request() {
    let mut engine = engine(0, 8);
    assert!(
        engine
            .submit_with::<()>(input(vec![1], 1), |_| Err(Error::memory(
                "handle allocation failed"
            )))
            .is_err()
    );
    assert!(!engine.has_work());
    assert_eq!(engine.next_request_id(), 1);
    assert_eq!(engine.cache_free_parent_count(), Some(16));
}

#[test]
fn cancellation_checks_engine_identity_and_returns_native_cache() {
    let mut first = engine(0, 8);
    let mut second = engine(0, 8);
    let mut model = Model::default();
    let request = first.submit(input(vec![1], 3)).unwrap();
    assert_eq!(
        second.cancel(&mut model, &request).unwrap_err().kind,
        ErrorKind::InvalidInput
    );
    first.step(&mut model).unwrap();
    assert!(first.cancel(&mut model, &request).unwrap());
    assert!(request.cancelled());
    assert_eq!(request.take_result().unwrap(), [2]);
    assert_eq!(first.cache_free_parent_count(), Some(16));
}

#[test]
fn a_panicking_native_runner_cannot_leave_a_healthy_engine() {
    struct PanickingRunner {
        score_access: bool,
    }
    impl Runner<&'static str> for PanickingRunner {
        fn forward(&mut self, _plan: ExecutionPlan<&'static str>) -> Result<OwnedTokenRows, Error> {
            assert!(self.score_access, "model state may already have changed");
            OwnedTokenRows::new(vec![2], vec![0, 1])
        }
        fn logprobs(&self) -> Option<resin_core::logprobs::LogprobView<'_>> {
            panic!("score storage access failed after model execution");
        }
    }
    for score_access in [false, true] {
        let mut engine = engine(0, 8);
        let request = engine.submit(input(vec![1], 1)).unwrap();
        let error = engine
            .step(&mut PanickingRunner { score_access })
            .unwrap_err();
        assert_eq!(error.kind, ErrorKind::Execution);
        assert!(engine.poisoned());
        assert!(!request.done());
        assert_eq!(engine.cache_occupied_block_count(), Some(1));
    }
}

#[test]
fn a_panicking_payload_clone_restores_the_cohort_and_releases_its_cache_access() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Payload {
        token: u32,
        fail_clone: Arc<AtomicBool>,
    }
    impl Clone for Payload {
        fn clone(&self) -> Self {
            assert!(
                !self.fail_clone.swap(false, Ordering::SeqCst),
                "payload clone failed"
            );
            Self {
                token: self.token,
                fail_clone: Arc::clone(&self.fail_clone),
            }
        }
    }
    #[derive(Default)]
    struct NativeModel {
        calls: usize,
        ids: Vec<u64>,
    }
    impl Runner<Payload> for NativeModel {
        fn forward(&mut self, plan: ExecutionPlan<Payload>) -> Result<OwnedTokenRows, Error> {
            self.calls += 1;
            self.ids = plan.sequence_ids;
            let indptr = (0..=plan.payloads.len() as u32).collect();
            OwnedTokenRows::new(
                plan.payloads
                    .into_iter()
                    .map(|payload| payload.token)
                    .collect(),
                indptr,
            )
        }
    }

    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    8,
                    vec![KvGroupLayout::new(2, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            ..Default::default()
        },
        4,
        128,
        SchedulerConfig::default(),
        false,
    )
    .unwrap();
    let mut handles = Vec::new();
    for (token, fail_clone) in [(11, false), (21, true)] {
        let mut submission = Submission::new(
            vec![1, 2],
            Payload {
                token,
                fail_clone: Arc::new(AtomicBool::new(fail_clone)),
            },
        );
        submission.max_output_tokens = Some(1);
        handles.push(engine.submit(submission).unwrap());
    }
    let mut model = NativeModel::default();
    let error = engine.step(&mut model).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Rejected);
    assert!(error.to_string().contains("payload cloning"));
    assert!(!engine.poisoned());
    assert_eq!(engine.ready_request_count(), 2);
    assert_eq!(model.calls, 0);
    assert!(handles.iter().all(|handle| !handle.done()));

    engine.step(&mut model).unwrap();
    assert_eq!(model.calls, 1);
    assert_eq!(
        model.ids,
        handles.iter().map(|handle| handle.id()).collect::<Vec<_>>()
    );
    assert_eq!(handles[0].take_result().unwrap(), [11]);
    assert_eq!(handles[1].take_result().unwrap(), [21]);
    assert_eq!(engine.cache_free_parent_count(), Some(8));
}

#[test]
fn enqueued_transaction_retains_requests_and_pages_until_harvest() {
    let mut engine = engine(0, 128);
    let mut model = Model::default();
    let request = engine.submit(input(vec![10], 2)).unwrap();
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert!(model.plans.is_empty());
    assert!(engine.has_in_flight_step());
    assert!(engine.has_work());
    assert_eq!(engine.ready_request_count(), 0);
    assert_eq!(engine.active_request_count(), 1);
    assert_eq!(engine.cache_free_parent_count(), Some(15));
    assert_eq!(
        engine.next_phase().unwrap(),
        Some(resin_core::Phase::Prefill)
    );
    assert!(!request.done());
    assert!(engine.prepare_step(&mut model, None).is_err());
    assert!(engine.cancel(&mut model, &request).is_err());
    assert!(engine.close(&mut model).is_err());
    assert!(!engine.poisoned());

    let output = model.forward(plan);
    let output = output
        .as_ref()
        .map(OwnedTokenRows::view)
        .map_err(Clone::clone);
    let events = engine.complete_step_with(&mut model, output, Ok).unwrap();
    assert_eq!(events[0].token_id, 11);
    assert!(!engine.has_in_flight_step());
    assert_eq!(engine.ready_request_count(), 1);
    assert!(engine.cancel(&mut model, &request).unwrap());
    assert_eq!(engine.cache_free_parent_count(), Some(16));
}

#[test]
fn borrowed_output_storage_is_reusable_after_atomic_commit() {
    let mut engine = engine(0, 128);
    let mut model = Model::default();
    let request = engine.submit(input(vec![10], 1)).unwrap();
    engine.prepare_step(&mut model, None).unwrap().unwrap();
    let mut ids = [u32::MAX];
    let mut offsets = [0, 1];
    let events = engine
        .complete_step_with(&mut model, TokenRows::new(&ids, &offsets), Ok)
        .unwrap();
    ids.fill(0);
    offsets.fill(0);
    assert_eq!(events[0].token_id, u32::MAX);
    assert_eq!(request.take_result().unwrap(), [u32::MAX]);
    assert_eq!(engine.cache_free_parent_count(), Some(16));
    assert!(!engine.poisoned());
}

#[test]
fn malformed_csr_output_quarantines_the_accepted_transaction() {
    let mut engine = engine(0, 128);
    let mut model = Model::default();
    let request = engine.submit(input(vec![10], 1)).unwrap();
    engine.prepare_step(&mut model, None).unwrap().unwrap();
    let error = engine
        .complete_step_with(&mut model, TokenRows::new(&[11], &[0, 2]), Ok)
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Execution);
    assert!(engine.poisoned());
    assert_eq!(engine.cache_free_parent_count(), Some(15));
    assert!(model.released.is_empty());
    assert!(!request.done());
}

#[test]
fn rejected_enqueued_transaction_restores_exact_cohort_but_failure_quarantines() {
    let mut engine = engine(0, 128);
    let mut model = Model::default();
    let request = engine.submit(input(vec![10], 2)).unwrap();
    engine.prepare_step(&mut model, None).unwrap().unwrap();
    let rejected = engine.complete_step_with(&mut model, Err(Error::rejected("not launched")), Ok);
    assert_eq!(rejected.unwrap_err().kind, ErrorKind::Rejected);
    assert!(!engine.poisoned());
    assert!(!engine.has_in_flight_step());
    assert_eq!(engine.ready_request_count(), 1);
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(plan.sequence_ids, [request.id()]);
    let failed = engine.complete_step_with(&mut model, Err(Error::execution("device failed")), Ok);
    assert_eq!(failed.unwrap_err().kind, ErrorKind::Execution);
    assert!(engine.poisoned());
    assert_eq!(engine.cache_free_parent_count(), Some(15));
    assert!(model.released.is_empty());
    assert!(!request.done());
}

#[test]
fn decode_continuation_transfers_writer_lease_and_commits_exact_feedback_depth() {
    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    4,
                    vec![KvGroupLayout::new(128, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            max_accepted_per_step: 4,
            ..Default::default()
        },
        4,
        512,
        SchedulerConfig::default(),
        true,
    )
    .unwrap();
    let mut model = Model::default();
    let request = engine.submit(input(vec![1], 40)).unwrap();
    engine.step(&mut model).unwrap();
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(plan.context_lengths, [1]);
    assert!(engine.retain_continuation(&plan));
    let next = engine.continuation_plan().unwrap().unwrap();
    assert_eq!(next.context_lengths, [5]); // GPU corrects conservative host length.
    assert_eq!(next.cache_groups[0].append_block_offsets, [5]);
    let leased = engine.cache_free_parent_count();
    let events = engine
        .complete_continuing_step_with(&mut model, TokenRows::new(&[3, 4], &[0, 2]), Ok)
        .unwrap();
    assert_eq!(
        events
            .iter()
            .map(|event| event.token_id)
            .collect::<Vec<_>>(),
        [3, 4]
    );
    assert!(engine.has_in_flight_step());
    assert_eq!(engine.cache_free_parent_count(), leased);
    let following = engine.continuation_plan().unwrap().unwrap();
    assert_eq!(following.context_lengths, [7]); // Actual 1 + 2, then maximum 4.
    assert_eq!(following.token_ids, [4]);
    engine
        .complete_step_with(&mut model, TokenRows::new(&[5], &[0, 1]), Ok)
        .unwrap();
    assert!(!engine.has_in_flight_step());
    let next = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(next.context_lengths, [4]);
    assert_eq!(next.token_ids, [5]);
    engine
        .complete_step_with(&mut model, Err(Error::rejected("not launched")), Ok)
        .unwrap_err();
    engine.close(&mut model).unwrap();
    assert_eq!(request.take_result().unwrap(), [2, 3, 4, 5]);
    assert_eq!(engine.cache_free_parent_count(), Some(4));
}

#[test]
fn unsafe_decode_boundaries_fall_back_before_enqueuing_a_continuation() {
    for (prompt, limit) in [(vec![1; 124], 40), (vec![1], 4)] {
        let mut engine = EngineCore::new(
            RunnerConfig {
                cache_layout: Some(
                    KvLayout::new(
                        8,
                        vec![KvGroupLayout::new(128, 1, KvRetention::FullContext).unwrap()],
                    )
                    .unwrap(),
                ),
                max_accepted_per_step: 4,
                ..Default::default()
            },
            4,
            512,
            SchedulerConfig::default(),
            false,
        )
        .unwrap();
        let mut model = Model::default();
        engine.submit(input(prompt, limit)).unwrap();
        engine.step(&mut model).unwrap();
        let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
        assert!(!engine.retain_continuation(&plan));
        assert!(engine.continuation_plan().unwrap().is_none());
        let output = model.forward(plan);
        let output = output
            .as_ref()
            .map(OwnedTokenRows::view)
            .map_err(Clone::clone);
        engine.complete_step_with(&mut model, output, Ok).unwrap();
        engine.close(&mut model).unwrap();
    }
}

fn cross_page_engine(prefix_cache: bool) -> EngineCore<&'static str> {
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    64,
                    vec![
                        KvGroupLayout::new(4, 2, KvRetention::FullContext).unwrap(),
                        KvGroupLayout::new(
                            4,
                            1,
                            KvRetention::SlidingWindow {
                                tokens: NonZeroU32::new(15).unwrap(),
                            },
                        )
                        .unwrap()
                        .with_read_window_tokens(4)
                        .unwrap(),
                    ],
                )
                .unwrap(),
            ),
            max_accepted_per_step: 4,
            continuation_prefill: true,
            ..Default::default()
        },
        4,
        256,
        SchedulerConfig::default(),
        prefix_cache,
    )
    .unwrap()
}

#[test]
fn cross_page_continuation_reserves_pages_and_updates_physical_append_addresses() {
    let mut engine = cross_page_engine(false);
    let mut model = Model::default();
    let requests: Vec<_> = [31, 32]
        .into_iter()
        .map(|length| engine.submit(input(vec![1; length], 40)).unwrap())
        .collect();
    while engine.next_phase().unwrap() == Some(resin_core::Phase::Prefill) {
        engine.step(&mut model).unwrap();
    }
    let plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert_eq!(plan.context_lengths, [31, 32]);
    assert!(engine.retain_continuation(&plan));
    let next = engine.continuation_plan().unwrap().unwrap();
    assert_eq!(next.context_lengths, [35, 36]);
    for (old, new) in plan.cache_groups.iter().zip(&next.cache_groups) {
        assert_eq!(old.block_indptr, new.block_indptr);
        assert_eq!(old.block_slab_ids, new.block_slab_ids);
        assert_eq!(old.block_slotnums, new.block_slotnums);
        assert_eq!(old.block_table_start_indices, new.block_table_start_indices);
        for (row, &position) in next.context_lengths.iter().enumerate() {
            let logical = position / 4;
            let index =
                (old.block_indptr[row] + logical - old.block_table_start_indices[row]) as usize;
            assert_eq!(new.append_slab_ids[row], old.block_slab_ids[index]);
            assert_eq!(new.append_slotnums[row], old.block_slotnums[index]);
            assert_eq!(new.append_block_offsets[row], position % 4);
            assert_ne!(
                (new.append_slab_ids[row], new.append_slotnums[row]),
                (old.append_slab_ids[row], old.append_slotnums[row])
            );
            // The earliest query's complete four-token local window and all
            // eight possible writes are in this immutable physical lease.
            assert!(old.block_table_start_indices[row] <= (plan.context_lengths[row] - 3) / 4);
            assert!(
                old.block_table_start_indices[row] + old.block_indptr[row + 1]
                    - old.block_indptr[row]
                    >= (plan.context_lengths[row] + 8).div_ceil(4)
            );
        }
    }
    let leased = engine.cache_free_parent_count();
    // Different accepted lengths keep the same dense row/page ownership. The
    // engine-sized lease still permits continuation; a runtime may also drain
    // early and reacquire after this already-enqueued successor completes.
    engine
        .complete_continuing_step_with(&mut model, TokenRows::new(&[3, 4, 5, 6, 7], &[0, 1, 5]), Ok)
        .unwrap();
    assert_eq!(engine.cache_free_parent_count(), leased);
    assert!(engine.has_continuation());
    engine
        .complete_step_with(&mut model, TokenRows::new(&[8, 9], &[0, 1, 2]), Ok)
        .unwrap();
    let refreshed = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert_eq!(refreshed.context_lengths, [33, 37]);
    assert!(engine.retain_continuation(&refreshed));
    engine
        .complete_step_with(&mut model, Err(Error::rejected("not launched")), Ok)
        .unwrap_err();
    engine.close(&mut model).unwrap();
    assert_eq!(model.released.len(), requests.len());
    assert_eq!(engine.cache_free_parent_count(), Some(64));
}

#[test]
fn wider_decode_reservation_does_not_increase_the_committed_output_limit() {
    let mut engine = cross_page_engine(false);
    let mut model = Model::default();
    engine.submit(input(vec![1; 31], 40)).unwrap();
    while engine.next_phase().unwrap() == Some(resin_core::Phase::Prefill) {
        engine.step(&mut model).unwrap();
    }
    let plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert!(engine.retain_continuation(&plan));
    let error = engine
        .complete_step_with(&mut model, TokenRows::new(&[3, 4, 5, 6, 7], &[0, 5]), Ok)
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Internal);
    assert!(error.to_string().contains("declared output maximum is 4"));
}

fn shifted_decode_engine() -> Result<EngineCore<&'static str>, Error> {
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    32,
                    vec![KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap()
                .with_decode_write_extra_tokens(1),
            ),
            max_accepted_per_step: 2,
            ..Default::default()
        },
        1,
        128,
        SchedulerConfig::default(),
        false,
    )
}

#[test]
fn accepted_progress_is_independent_of_the_resolved_cache_write_footprint() {
    let mut engine = shifted_decode_engine().unwrap();
    let mut model = Model::default();
    engine.submit(input(vec![1; 2], 40)).unwrap();
    engine.step(&mut model).unwrap();
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    // M2 plus one shifted persistent write leases three positions, crossing
    // into a second page. It does not authorize a third accepted output.
    assert_eq!(plan.cache_groups[0].block_indptr, [0, 2]);
    let error = engine
        .complete_step_with(&mut model, TokenRows::new(&[3, 4, 5], &[0, 3]), Ok)
        .unwrap_err();
    assert!(error.to_string().contains("declared output maximum is 2"));

    let mut engine = shifted_decode_engine().unwrap(); // M2+W3, not2*M2.
    let mut model = Model::default();
    let request = engine.submit(input(vec![1; 2], 40)).unwrap();
    engine.step(&mut model).unwrap();
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert!(engine.retain_continuation(&plan));
    let next = engine.continuation_plan().unwrap().unwrap();
    assert_eq!(next.context_lengths, [4]); // Advance M2, not write span3.
    engine
        .complete_continuing_step_with(&mut model, TokenRows::new(&[3], &[0, 1]), Ok)
        .unwrap();
    let next = engine.continuation_plan().unwrap().unwrap();
    assert_eq!(next.context_lengths, [5]); // Actual2+1, then maximum advance2.
    engine
        .complete_step_with(&mut model, TokenRows::new(&[4, 5], &[0, 2]), Ok)
        .unwrap();
    engine.close(&mut model).unwrap();
    assert_eq!(request.take_result().unwrap(), [2, 3, 4, 5]);
    assert_eq!(engine.cache_free_parent_count(), Some(32));
}

#[test]
fn accepted_limit_defaults_to_one_and_rejects_zero_or_overflowing_write_footprints() {
    assert_eq!(RunnerConfig::default().max_accepted_per_step, 1);
    for (accepted, extra) in [(0, 0), (u32::MAX, 1)] {
        let result = EngineCore::<&str>::new(
            RunnerConfig {
                max_accepted_per_step: accepted,
                cache_layout: Some(
                    KvLayout::new(
                        4,
                        vec![KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap()],
                    )
                    .unwrap()
                    .with_decode_write_extra_tokens(extra),
                ),
                ..Default::default()
            },
            1,
            128,
            SchedulerConfig::default(),
            false,
        );
        assert_eq!(result.err().unwrap().kind, ErrorKind::InvalidInput);
    }
}

#[test]
fn shifted_write_guard_survives_a_near_context_limit_without_changing_outputs() {
    for overlap in [false, true] {
        let mut engine = EngineCore::new(
            RunnerConfig {
                cache_layout: Some(
                    KvLayout::new(
                        33,
                        vec![KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap()],
                    )
                    .unwrap()
                    .with_decode_write_extra_tokens(1),
                ),
                max_accepted_per_step: 2,
                ..Default::default()
            },
            1,
            128,
            SchedulerConfig::default(),
            false,
        )
        .unwrap();
        assert!(engine.submit(input(vec![1; 126], 4)).is_err());
        let request = engine.submit(input(vec![1; 126], 3)).unwrap();
        let mut model = Model::default();
        engine.step(&mut model).unwrap();
        let plan = if overlap {
            engine.prepare_overlap_step(&mut model, None)
        } else {
            engine.prepare_step(&mut model, None)
        }
        .unwrap()
        .unwrap();
        assert_eq!(plan.context_lengths, [126]);
        // Target inputs126/127 fit the logical context; shifted persistent
        // row128 still needs its own physical guard page. Near the limit both
        // admission paths use the mandatory three-row write footprint.
        assert_eq!(plan.cache_groups[0].block_indptr, [0, 33]);
        assert!(!engine.retain_continuation(&plan));
        engine
            .complete_step_with(&mut model, TokenRows::new(&[3, 4], &[0, 2]), Ok)
            .unwrap();
        assert_eq!(request.take_result().unwrap(), [2, 3, 4]);
        assert_eq!(engine.cache_free_parent_count(), Some(33));
    }
}

#[test]
fn stopped_cross_page_successor_keeps_its_full_lease_until_physical_completion() {
    for stop in [3, 4, 5, 6] {
        let mut engine = cross_page_engine(true);
        let mut model = Model::default();
        let mut submission = input(vec![1; 31], 40);
        submission.stop_token_ids = vec![stop];
        let request = engine.submit(submission).unwrap();
        while engine.next_phase().unwrap() == Some(resin_core::Phase::Prefill) {
            engine.step(&mut model).unwrap();
        }
        let plan = engine
            .prepare_overlap_step(&mut model, None)
            .unwrap()
            .unwrap();
        assert!(engine.retain_continuation(&plan));
        engine.continuation_plan().unwrap().unwrap();
        let leased = engine.cache_free_parent_count();
        engine
            .complete_continuing_step_with(&mut model, TokenRows::new(&[3, 4, 5, 6], &[0, 4]), Ok)
            .unwrap();
        assert!(request.done());
        assert_eq!(
            request.take_result().unwrap(),
            (2..=stop).collect::<Vec<_>>()
        );
        assert!(model.released.is_empty());
        assert_eq!(engine.cache_free_parent_count(), leased);
        assert!(engine.cancel(&mut model, &request).is_err());
        engine
            .complete_step_with(&mut model, TokenRows::new(&[90, 91, 92, 93], &[0, 4]), Ok)
            .unwrap();
        assert_eq!(model.released, [request.id()]);
        // A cached prefix may retain pages after retirement; closing evicts it.
        engine.close(&mut model).unwrap();
        assert_eq!(engine.cache_free_parent_count(), Some(64));
    }
}

#[test]
fn window_guard_bounds_automatic_reservation_and_rejects_an_oversized_write() {
    for read_window in [4, 9] {
        let layout = KvLayout::new(
            8,
            vec![
                KvGroupLayout::new(
                    4,
                    1,
                    KvRetention::SlidingWindow {
                        tokens: NonZeroU32::new(11).unwrap(),
                    },
                )
                .unwrap()
                .with_read_window_tokens(read_window)
                .unwrap(),
            ],
        )
        .unwrap();
        let result = EngineCore::<&str>::new(
            RunnerConfig {
                cache_layout: Some(layout),
                max_accepted_per_step: 4,
                continuation_prefill: true,
                ..Default::default()
            },
            4,
            256,
            SchedulerConfig::default(),
            false,
        );
        if read_window == 9 {
            assert_eq!(result.err().unwrap().kind, ErrorKind::InvalidInput);
        } else {
            let mut engine = result.unwrap();
            let mut model = Model::default();
            engine.submit(input(vec![1; 32], 40)).unwrap();
            while engine.next_phase().unwrap() == Some(resin_core::Phase::Prefill) {
                engine.step(&mut model).unwrap();
            }
            let plan = engine
                .prepare_overlap_step(&mut model, None)
                .unwrap()
                .unwrap();
            let group = &plan.cache_groups[0];
            assert_eq!(group.block_table_start_indices, [6]);
            assert_eq!(group.block_indptr, [0, 4]);
            assert!(engine.retain_continuation(&plan));
            engine
                .complete_step_with(&mut model, TokenRows::new(&[3], &[0, 1]), Ok)
                .unwrap();
            engine.close(&mut model).unwrap();
        }
    }
    let state = KvLayout::new(
        8,
        vec![
            KvGroupLayout::new_with_family(4, 1, KvRetention::FullContext, KvGroupFamily::State)
                .unwrap(),
        ],
    )
    .unwrap();
    assert!(
        EngineCore::<&str>::new(
            RunnerConfig {
                cache_layout: Some(state),
                max_accepted_per_step: 4,
                ..Default::default()
            },
            4,
            256,
            SchedulerConfig::default(),
            false,
        )
        .is_ok()
    );
}

fn hybrid_reservation_engine(prefix_cache: bool) -> EngineCore<&'static str> {
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    128,
                    vec![
                        KvGroupLayout::new(128, 1, KvRetention::FullContext).unwrap(),
                        KvGroupLayout::new_with_family(
                            4096,
                            1,
                            KvRetention::FullContext,
                            KvGroupFamily::State,
                        )
                        .unwrap(),
                        KvGroupLayout::new(
                            128,
                            1,
                            KvRetention::SlidingWindow {
                                tokens: NonZeroU32::new(1024).unwrap(),
                            },
                        )
                        .unwrap()
                        .with_read_window_tokens(512)
                        .unwrap(),
                    ],
                )
                .unwrap()
                .with_replay_suffix_tokens(1),
            ),
            continuation_prefill: true,
            max_accepted_per_step: 3,
            ..Default::default()
        },
        4,
        12288,
        SchedulerConfig {
            prefill_chunk_size: NonZeroU32::new(8192).unwrap(),
            ..Default::default()
        },
        prefix_cache,
    )
    .unwrap()
}

#[test]
fn hybrid_reservation_crosses_history_and_window_pages_without_rotating_state() {
    let mut engine = hybrid_reservation_engine(false);
    let mut model = Model::default();
    let request = engine.submit(input(vec![1; 127], 40)).unwrap();
    engine.step(&mut model).unwrap();
    let plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert_eq!(plan.context_lengths, [127]);
    assert_eq!(plan.cache_groups[0].block_indptr, [0, 2]);
    assert_eq!(plan.cache_groups[1].block_indptr, [0, 1]);
    assert!(engine.retain_continuation(&plan));
    let next = engine.continuation_plan().unwrap().unwrap();
    assert_eq!(next.context_lengths, [130]);
    for (index, (old, new)) in plan.cache_groups.iter().zip(&next.cache_groups).enumerate() {
        assert_eq!(old.block_indptr, new.block_indptr);
        assert_eq!(old.block_slab_ids, new.block_slab_ids);
        assert_eq!(old.block_slotnums, new.block_slotnums);
        assert_eq!(old.block_table_start_indices, new.block_table_start_indices);
        let old_append = (old.append_slab_ids[0], old.append_slotnums[0]);
        let new_append = (new.append_slab_ids[0], new.append_slotnums[0]);
        if index == 1 {
            assert_eq!(old_append, new_append);
            assert_eq!(new.append_block_offsets, [0]);
        } else {
            assert_ne!(old_append, new_append);
            assert_eq!(new.append_block_offsets, [2]);
        }
    }
    let leased = engine.cache_free_parent_count();
    engine
        .complete_continuing_step_with(&mut model, TokenRows::new(&[3, 4], &[0, 2]), Ok)
        .unwrap();
    assert_eq!(engine.cache_free_parent_count(), leased);
    let following = engine.continuation_plan().unwrap().unwrap();
    assert_eq!(following.context_lengths, [132]); // Actual127+2, then maximum3.
    assert_eq!(
        following.cache_groups[1].block_slab_ids,
        plan.cache_groups[1].block_slab_ids
    );
    engine
        .complete_step_with(&mut model, TokenRows::new(&[5], &[0, 1]), Ok)
        .unwrap();
    engine.close(&mut model).unwrap();
    assert_eq!(request.take_result().unwrap(), [2, 3, 4, 5]);
    assert_eq!(engine.cache_free_parent_count(), Some(128));
}

#[test]
fn hybrid_live_state_crosses_checkpoint_boundaries_without_reallocating() {
    for start in [4095_u32, 8191] {
        let mut engine = hybrid_reservation_engine(false);
        let mut model = Model::default();
        let request = engine.submit(input(vec![1; start as usize], 40)).unwrap();
        while engine.next_phase().unwrap() == Some(resin_core::Phase::Prefill) {
            engine.step(&mut model).unwrap();
        }
        let plan = engine
            .prepare_overlap_step(&mut model, None)
            .unwrap()
            .unwrap();
        assert_eq!(plan.context_lengths, [start]);
        assert_eq!(
            plan.cache_groups[0].block_indptr,
            [0, (start + 41).div_ceil(128)]
        );
        assert_eq!(plan.cache_groups[1].block_indptr, [0, 1]);
        assert!(engine.retain_continuation(&plan));
        let next = engine.continuation_plan().unwrap().unwrap();
        assert_eq!(next.context_lengths, [start + 3]);
        assert_eq!(
            next.cache_groups[1].block_slab_ids,
            plan.cache_groups[1].block_slab_ids
        );
        assert_eq!(
            next.cache_groups[1].append_slab_ids,
            plan.cache_groups[1].append_slab_ids
        );
        assert_eq!(next.cache_groups[1].append_block_offsets, [0]);
        assert!(
            next.cache_groups[1]
                .state
                .as_ref()
                .unwrap()
                .restore_copies
                .is_empty()
        );
        let leased = engine.cache_free_parent_count();
        engine
            .complete_continuing_step_with(&mut model, TokenRows::new(&[3, 4], &[0, 2]), Ok)
            .unwrap();
        assert_eq!(engine.cache_free_parent_count(), leased);
        engine
            .complete_step_with(&mut model, TokenRows::new(&[5], &[0, 1]), Ok)
            .unwrap();
        engine.close(&mut model).unwrap();
        assert_eq!(request.take_result().unwrap(), [2, 3, 4, 5]);
        assert_eq!(engine.cache_free_parent_count(), Some(128));
    }
}

#[test]
fn state_boundaries_do_not_downshift_other_rows_token_reservations() {
    let mut engine = hybrid_reservation_engine(false);
    let mut model = Model::default();
    let requests: Vec<_> = [3960, 4095]
        .into_iter()
        .map(|start| engine.submit(input(vec![1; start], 40)).unwrap())
        .collect();
    while engine.next_phase().unwrap() == Some(resin_core::Phase::Prefill) {
        engine.step(&mut model).unwrap();
    }
    let plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert_eq!(plan.context_lengths, [3960, 4095]);
    // Both rows retain their token reservations independently of STATE cadence.
    assert_eq!(plan.cache_groups[0].block_indptr, [0, 32, 65]);
    assert!(engine.retain_continuation(&plan));
    engine
        .complete_step_with(&mut model, TokenRows::new(&[3, 4], &[0, 1, 2]), Ok)
        .unwrap();
    engine.close(&mut model).unwrap();
    assert!(requests.iter().all(|request| request.done()));
    assert_eq!(engine.cache_free_parent_count(), Some(128));
}

#[test]
fn hybrid_reservation_supports_distinct_private_state_checkpoint_cadences() {
    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    32,
                    vec![
                        KvGroupLayout::new(128, 1, KvRetention::FullContext).unwrap(),
                        KvGroupLayout::new_with_family(
                            4096,
                            1,
                            KvRetention::FullContext,
                            KvGroupFamily::State,
                        )
                        .unwrap(),
                        KvGroupLayout::new_with_family(
                            512,
                            1,
                            KvRetention::FullContext,
                            KvGroupFamily::State,
                        )
                        .unwrap(),
                    ],
                )
                .unwrap(),
            ),
            continuation_prefill: true,
            max_accepted_per_step: 3,
            ..Default::default()
        },
        1,
        2048,
        SchedulerConfig::default(),
        false,
    )
    .unwrap();
    let mut model = Model::default();
    engine.submit(input(vec![1; 511], 40)).unwrap();
    engine.step(&mut model).unwrap();
    let plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert_eq!(plan.cache_groups[0].block_indptr, [0, 5]);
    assert_eq!(plan.cache_groups[1].block_indptr, [0, 1]);
    assert_eq!(plan.cache_groups[2].block_indptr, [0, 1]);
    assert!(engine.retain_continuation(&plan));
    engine
        .complete_step_with(&mut model, TokenRows::new(&[3], &[0, 1]), Ok)
        .unwrap();
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(32));
}

#[test]
fn hybrid_reserved_successor_keeps_prefix_input_and_private_state_until_drain() {
    let mut engine = hybrid_reservation_engine(true);
    let mut model = Model {
        report_checkpoints: true,
        ..Default::default()
    };
    let seed = engine.submit(input(vec![1; 4096], 1)).unwrap();
    while engine.has_work() {
        engine.step(&mut model).unwrap();
    }
    assert_eq!(seed.take_result().unwrap(), [2]);
    let requests: Vec<_> = (0..2)
        .map(|row| {
            let mut submission = input(vec![1; 4097], 40);
            if row == 0 {
                submission.stop_token_ids = vec![99];
            }
            engine.submit(submission).unwrap()
        })
        .collect();
    assert!(
        requests
            .iter()
            .all(|request| request.cached_tokens() == 4096)
    );
    engine.step(&mut model).unwrap();
    let plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert_eq!(plan.context_lengths, [4097, 4097]);
    let state = &plan.cache_groups[1];
    assert_eq!(state.block_indptr, [0, 1, 2]);
    assert_ne!(
        (state.block_slab_ids[0], state.block_slotnums[0]),
        (state.block_slab_ids[1], state.block_slotnums[1])
    );
    let restored = model.plans.last().unwrap().cache_groups[1]
        .state
        .as_ref()
        .unwrap();
    assert_eq!(restored.restore_copies.len(), 2);
    let shared_input = (
        restored.restore_copies[0].source_slab_id,
        restored.restore_copies[0].source_slotnum,
    );
    assert_eq!(
        shared_input,
        (
            restored.restore_copies[1].source_slab_id,
            restored.restore_copies[1].source_slotnum
        )
    );
    assert_ne!(
        (state.append_slab_ids[0], state.append_slotnums[0]),
        (state.append_slab_ids[1], state.append_slotnums[1])
    );
    assert!(engine.retain_continuation(&plan));
    engine.continuation_plan().unwrap().unwrap();
    let leased = engine.cache_free_parent_count();
    let released = model.released.clone();
    assert!(engine.cancel(&mut model, &requests[0]).is_err());
    engine
        .complete_continuing_step_with(&mut model, TokenRows::new(&[99, 500, 3], &[0, 2, 3]), Ok)
        .unwrap();
    assert_eq!(requests[0].take_result().unwrap(), [2, 99]);
    assert_eq!(engine.cache_free_parent_count(), leased);
    assert_eq!(model.released, released);
    assert!(!engine.has_continuation());
    engine
        .complete_step_with(
            &mut model,
            TokenRows::new(&[900, 901, 4, 5], &[0, 2, 4]),
            Ok,
        )
        .unwrap();
    assert!(engine.cancel(&mut model, &requests[1]).unwrap());
    assert_eq!(requests[1].take_result().unwrap(), [2, 3, 4, 5]);
    let resumed = engine.submit(input(vec![1; 4097], 1)).unwrap();
    assert_eq!(resumed.cached_tokens(), 4096);
    engine.step(&mut model).unwrap();
    let state = &model.plans.last().unwrap().cache_groups[1];
    let copy = &state.state.as_ref().unwrap().restore_copies[0];
    assert_eq!(shared_input, (copy.source_slab_id, copy.source_slotnum));
    assert_ne!(
        shared_input,
        (state.append_slab_ids[0], state.append_slotnums[0])
    );
    assert_eq!(resumed.take_result().unwrap(), [2]);
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(128));
}

#[test]
fn rounded_reservation_tail_cannot_exceed_a_windows_declared_write_guard() {
    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    32,
                    vec![
                        KvGroupLayout::new(
                            4,
                            1,
                            KvRetention::SlidingWindow {
                                tokens: NonZeroU32::new(11).unwrap(),
                            },
                        )
                        .unwrap()
                        .with_read_window_tokens(4)
                        .unwrap(),
                    ],
                )
                .unwrap(),
            ),
            continuation_prefill: true,
            max_accepted_per_step: 4,
            ..Default::default()
        },
        4,
        256,
        SchedulerConfig::default(),
        false,
    )
    .unwrap();
    let mut model = Model::default();
    engine.submit(input(vec![1; 31], 40)).unwrap();
    while engine.next_phase().unwrap() == Some(resin_core::Phase::Prefill) {
        engine.step(&mut model).unwrap();
    }
    let plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    // Eight writes from P31 would end at39 (exclusive), but the page ends
    // at40. The engine falls back to ordinary writes rather than advertising
    // a rounded ninth write beyond the declared guard8.
    assert!(!engine.retain_continuation(&plan));
    engine
        .complete_step_with(&mut model, TokenRows::new(&[3, 4, 5, 6], &[0, 4]), Ok)
        .unwrap();
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(32));
}

fn bounded_reservation_engine(capacity: u32, context: u32) -> EngineCore<&'static str> {
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    capacity,
                    vec![KvGroupLayout::new(4, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            max_accepted_per_step: 4,
            ..Default::default()
        },
        4,
        context,
        SchedulerConfig {
            schedule_conservativeness: 0.0,
            ..Default::default()
        },
        false,
    )
    .unwrap()
}

#[test]
fn only_overlap_admission_reserves_an_engine_sized_bounded_envelope() {
    for (overlap, prompt, output, context, pages) in [
        (false, 3, 40, 256, 2),
        (true, 3, 40, 256, 4),
        (true, 1, 11, 11, 3),
    ] {
        let mut engine = bounded_reservation_engine(64, context);
        let mut model = Model::default();
        engine.submit(input(vec![1; prompt], output)).unwrap();
        engine.step(&mut model).unwrap();
        let plan = if overlap {
            engine.prepare_overlap_step(&mut model, None)
        } else {
            engine.prepare_step(&mut model, None)
        }
        .unwrap()
        .unwrap();
        assert_eq!(plan.cache_groups[0].block_indptr, [0, pages]);
        assert_eq!(engine.retain_continuation(&plan), overlap);
        engine
            .complete_step_with(&mut model, TokenRows::new(&[3], &[0, 1]), Ok)
            .unwrap();
        engine.close(&mut model).unwrap();
        assert_eq!(engine.cache_free_parent_count(), Some(64));
    }
}

#[test]
fn optional_reservation_preserves_near_context_final_short_output_capacity() {
    for overlap in [false, true] {
        let mut engine = bounded_reservation_engine(16, 64);
        let mut model = Model::default();
        let request = engine.submit(input(vec![1; 60], 4)).unwrap();
        engine.step(&mut model).unwrap();
        let plan = if overlap {
            engine.prepare_overlap_step(&mut model, None)
        } else {
            engine.prepare_step(&mut model, None)
        }
        .unwrap()
        .unwrap();
        assert_eq!(plan.context_lengths, [60]);
        assert_eq!(plan.cache_groups[0].block_indptr, [0, 16]);
        assert!(!engine.retain_continuation(&plan));
        engine
            .complete_step_with(&mut model, TokenRows::new(&[3, 4, 5], &[0, 3]), Ok)
            .unwrap();
        assert_eq!(request.take_result().unwrap(), [2, 3, 4, 5]);
        assert_eq!(engine.cache_free_parent_count(), Some(16));
    }
}

#[test]
fn one_short_row_downshifts_the_whole_cohort_without_overallocating_long_rows() {
    let mut engine = bounded_reservation_engine(32, 256);
    let mut model = Model::default();
    let short = engine.submit(input(vec![1; 60], 4)).unwrap();
    let long = engine.submit(input(vec![10; 60], 40)).unwrap();
    engine.step(&mut model).unwrap();
    let plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert_eq!(plan.sequence_ids, [short.id(), long.id()]);
    assert_eq!(plan.cache_groups[0].block_indptr, [0, 16, 32]);
    assert!(!engine.retain_continuation(&plan));
    engine
        .complete_step_with(&mut model, TokenRows::new(&[3, 4, 5, 12], &[0, 3, 4]), Ok)
        .unwrap();
    assert_eq!(short.take_result().unwrap(), [2, 3, 4, 5]);
    assert!(!long.done());
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(32));
}

#[test]
fn optional_wide_cohort_capacity_is_checked_before_any_partial_growth() {
    let mut engine = bounded_reservation_engine(32, 256);
    let mut model = Model::default();
    engine.submit(input(vec![1; 60], 40)).unwrap();
    engine.submit(input(vec![10; 60], 40)).unwrap();
    engine.step(&mut model).unwrap();
    let plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    // Both ordinary four-token spans fit exactly. Growing the first row's
    // optional page-plus-envelope span before discovering the second doesn't fit would strand
    // capacity and force a needless retraction/failure.
    assert_eq!(plan.cache_groups[0].block_indptr, [0, 16, 32]);
    assert!(!engine.retain_continuation(&plan));
    engine
        .complete_step_with(&mut model, TokenRows::new(&[3, 12], &[0, 1, 2]), Ok)
        .unwrap();
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(32));
}

#[test]
fn engine_sized_reservation_chains_without_allocating_or_inflating_outputs() {
    let mut engine = bounded_reservation_engine(64, 256);
    let mut model = Model::default();
    let request = engine.submit(input(vec![1; 3], 100)).unwrap();
    engine.step(&mut model).unwrap();
    let mut plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert_eq!(plan.cache_groups[0].block_indptr, [0, 4]);
    let leased = engine.cache_free_parent_count();
    let mut token = 3;
    let mut continuations = 0;
    while engine.retain_continuation(&plan) {
        let next = engine.continuation_plan().unwrap().unwrap();
        engine
            .complete_continuing_step_with(
                &mut model,
                TokenRows::new(&[token, token + 1, token + 2, token + 3], &[0, 4]),
                Ok,
            )
            .unwrap();
        assert_eq!(engine.cache_free_parent_count(), leased);
        continuations += 1;
        token += 4;
        // core updates the anchor from actual committed output. In this test
        // all four were accepted, so the conservative successor is exact.
        plan = next;
    }
    assert!(continuations >= 2);
    engine
        .complete_step_with(&mut model, TokenRows::new(&[token], &[0, 1]), Ok)
        .unwrap();
    engine.close(&mut model).unwrap();
    assert_eq!(request.take_result().unwrap().len(), 2 + 4 * continuations);
    assert_eq!(engine.cache_free_parent_count(), Some(64));
}

#[test]
fn falling_back_after_a_wide_lease_preserves_unused_owned_pages() {
    let mut engine = bounded_reservation_engine(40, 256);
    let mut model = Model::default();
    let request = engine.submit(input(vec![1; 58], 11)).unwrap();
    engine.step(&mut model).unwrap();
    let plan = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert_eq!(plan.cache_groups[0].block_indptr, [0, 18]);
    assert!(engine.retain_continuation(&plan));
    engine.continuation_plan().unwrap().unwrap();
    engine
        .complete_continuing_step_with(&mut model, TokenRows::new(&[3], &[0, 1]), Ok)
        .unwrap();
    // Drain early as a runtime could request. Two minimal accepted blocks move
    // the next anchor to60. A normal span now ends64, before the old wide end70.
    engine
        .complete_step_with(&mut model, TokenRows::new(&[4], &[0, 1]), Ok)
        .unwrap();
    let next = engine
        .prepare_overlap_step(&mut model, None)
        .unwrap()
        .unwrap();
    assert_eq!(next.context_lengths, [60]);
    assert_eq!(next.cache_groups[0].block_indptr, [0, 16]);
    assert!(!engine.retain_continuation(&next));
    engine
        .complete_step_with(&mut model, TokenRows::new(&[5], &[0, 1]), Ok)
        .unwrap();
    engine.close(&mut model).unwrap();
    assert_eq!(request.take_result().unwrap(), [2, 3, 4, 5]);
    assert_eq!(engine.cache_free_parent_count(), Some(40));
}

fn stop_overlap_engine() -> EngineCore<&'static str> {
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    8,
                    vec![KvGroupLayout::new(128, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            max_accepted_per_step: 4,
            ..Default::default()
        },
        4,
        512,
        SchedulerConfig::default(),
        true,
    )
    .unwrap()
}

#[test]
fn stopped_continuation_publishes_result_but_keeps_pages_until_successor_finishes() {
    for stop in [3, 4, 5] {
        let mut engine = stop_overlap_engine();
        engine.enable_metrics().unwrap();
        let mut model = Model::default();
        let mut submission = input(vec![1], 40);
        submission.stop_token_ids = vec![stop];
        let request = engine.submit(submission).unwrap();
        engine.step(&mut model).unwrap();
        let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
        assert!(engine.retain_continuation(&plan));
        engine.continuation_plan().unwrap().unwrap(); // Successor accepted on device.
        let leased = engine.cache_free_parent_count();
        let idle_before = engine.metrics_snapshot().unwrap().scheduler_idle_seconds;

        let events = engine
            .complete_continuing_step_with(&mut model, TokenRows::new(&[3, 4, 5], &[0, 3]), Ok)
            .unwrap();
        assert_eq!(
            events
                .iter()
                .map(|event| event.token_id)
                .collect::<Vec<_>>(),
            (3..=stop).collect::<Vec<_>>()
        );
        assert!(events.last().unwrap().finished);
        assert_eq!(
            events.last().unwrap().finish_reason,
            Some(resin_core::engine::CompletionReason::Stop)
        );
        assert!(request.done());
        assert_eq!(
            request.take_result().unwrap(),
            (2..=stop).collect::<Vec<_>>()
        );
        assert_eq!(engine.active_request_count(), 0);
        let metrics = engine.metrics_snapshot().unwrap();
        assert_eq!(metrics.scheduler_idle_seconds, idle_before);
        assert_eq!(metrics.completed_requests, 1);
        assert_eq!(metrics.generation_tokens, u64::from(stop - 1));
        assert_eq!(
            metrics.inter_token_latency_seconds.count,
            u64::from(stop - 2)
        );
        assert_eq!(engine.ready_request_count(), 0);
        assert!(engine.has_work() && engine.has_in_flight_step());
        assert!(engine.continuation_plan().unwrap().is_none());
        assert_eq!(engine.cache_free_parent_count(), leased);
        assert!(model.released.is_empty());
        assert!(engine.cancel(&mut model, &request).is_err());

        // The device used the untruncated accepted block for feedback. Neither
        // its extra positions nor its sampled output belong to this request.
        let discarded = engine
            .complete_step_with(
                &mut model,
                TokenRows::new(&[100, 101, 102, 103], &[0, 4]),
                Ok,
            )
            .unwrap();
        assert!(discarded.is_empty());
        assert_eq!(model.released, [request.id()]);
        assert_eq!(engine.cache_free_parent_count(), Some(8));
        assert!(!engine.has_work());
        assert!(!engine.cancel(&mut model, &request).unwrap());
        assert!(request.take_result().is_err()); // Never refill a consumed result.
        engine.close(&mut model).unwrap();
        assert_eq!(model.released, [request.id()]);
        let metrics = engine.metrics_snapshot().unwrap();
        assert_eq!(metrics.aborted_requests, 0);
        assert_eq!(metrics.decode_tokens, u64::from(stop - 1));
    }
}

#[test]
fn metrics_distinguish_waiting_running_rejection_cancellation_and_completion() {
    let mut engine = engine(0, 2);
    engine.enable_metrics().unwrap();
    let request = engine.submit(input(vec![1, 2, 3, 4], 2)).unwrap();
    let cancelled = engine.submit(input(vec![5, 6], 2)).unwrap();
    let mut model = Model::default();
    assert_eq!(engine.metrics_snapshot().unwrap().waiting_requests, 2);
    engine.cancel(&mut model, &cancelled).unwrap();
    model.outputs.push_back(Err(Error::rejected("retry")));
    assert!(engine.step(&mut model).is_err());
    let metrics = engine.metrics_snapshot().unwrap();
    assert_eq!(metrics.waiting_requests, 1);
    assert_eq!(metrics.running_requests, 0);
    assert_eq!(metrics.prefill_compute_tokens, 0);
    assert_eq!(metrics.queue_time_seconds.count, 0);

    engine.step(&mut model).unwrap();
    let metrics = engine.metrics_snapshot().unwrap();
    assert_eq!(metrics.running_requests, 1);
    assert_eq!(metrics.prefill_compute_tokens, 2);
    assert_eq!(metrics.decode_tokens, 0);
    assert_eq!(metrics.time_to_first_token_seconds.count, 0);
    while engine.has_work() {
        engine.step(&mut model).unwrap();
    }
    assert!(request.done());
    let metrics = engine.metrics_snapshot().unwrap();
    assert_eq!(metrics.accepted_requests, 2);
    assert_eq!(metrics.aborted_requests, 1);
    assert_eq!(metrics.completed_requests, 1);
    assert_eq!(metrics.waiting_requests + metrics.running_requests, 0);
    assert_eq!(metrics.prompt_tokens, 4);
    assert_eq!(metrics.prefill_compute_tokens, 4);
    assert_eq!(metrics.generation_tokens, 2);
    assert_eq!(metrics.queue_time_seconds.count, 1);
    assert_eq!(metrics.time_to_first_token_seconds.count, 1);
}

#[test]
fn retraction_counts_compute_replay_without_duplicating_logical_tokens() {
    let mut engine = EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    3,
                    vec![KvGroupLayout::new(1, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            continuation_prefill: true,
            ..Default::default()
        },
        2,
        3,
        SchedulerConfig {
            schedule_conservativeness: 0.0,
            ..Default::default()
        },
        false,
    )
    .unwrap();
    engine.enable_metrics().unwrap();
    engine.submit(input(vec![1], 3)).unwrap();
    engine.submit(input(vec![10], 3)).unwrap();
    let mut model = Model::default();
    while engine.has_work() {
        engine.step(&mut model).unwrap();
    }
    let metrics = engine.metrics_snapshot().unwrap();
    assert_eq!(metrics.retracted_requests, 1);
    assert_eq!(metrics.prompt_tokens, 2);
    assert_eq!(metrics.prefill_compute_tokens, 3);
    assert_eq!(metrics.generation_tokens, 6);
    assert_eq!(metrics.queue_time_seconds.count, 2);
}

#[test]
fn stopped_middle_row_retains_dense_successor_slots_and_variable_acceptance() {
    let mut engine = stop_overlap_engine();
    let mut model = Model::default();
    let requests: Vec<_> = [1, 10, 20]
        .into_iter()
        .map(|token| {
            let mut submission = input(vec![token], 40);
            submission.stop_token_ids = vec![99];
            engine.submit(submission).unwrap()
        })
        .collect();
    engine.step(&mut model).unwrap();
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert!(engine.retain_continuation(&plan));
    engine.continuation_plan().unwrap().unwrap();
    let events = engine
        .complete_continuing_step_with(
            &mut model,
            TokenRows::new(&[3, 4, 99, 100, 101, 22], &[0, 2, 5, 6]),
            Ok,
        )
        .unwrap();
    assert_eq!(events.len(), 4);
    assert_eq!(requests[1].take_result().unwrap(), [11, 99]);
    assert_eq!(engine.active_request_count(), 2);
    assert_eq!(engine.cache_free_parent_count(), Some(5));
    assert!(model.released.is_empty());
    assert!(!engine.has_continuation());

    let events = engine
        .complete_step_with(
            &mut model,
            TokenRows::new(&[5, 900, 901, 23, 24, 25, 26], &[0, 1, 3, 7]),
            Ok,
        )
        .unwrap();
    assert_eq!(
        events
            .iter()
            .map(|event| (event.request_id, event.token_id))
            .collect::<Vec<_>>(),
        [
            (requests[0].id(), 5),
            (requests[2].id(), 23),
            (requests[2].id(), 24),
            (requests[2].id(), 25),
            (requests[2].id(), 26),
        ]
    );
    assert_eq!(model.released, [requests[1].id()]);
    assert_eq!(engine.cache_free_parent_count(), Some(6));
    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(plan.sequence_ids, [requests[0].id(), requests[2].id()]);
    assert_eq!(plan.context_lengths, [4, 6]);
    assert_eq!(plan.token_ids, [5, 26]);
    engine
        .complete_step_with(&mut model, Err(Error::rejected("not launched")), Ok)
        .unwrap_err();
    assert!(engine.cancel(&mut model, &requests[0]).unwrap());
    assert!(requests[0].cancelled());
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(8));
    assert_eq!(requests[0].take_result().unwrap(), [2, 3, 4, 5]);
    assert_eq!(requests[2].take_result().unwrap(), [21, 22, 23, 24, 25, 26]);
}

#[test]
fn failed_successor_quarantines_terminal_rows_without_resurrection() {
    for failure in 0..3 {
        let mut engine = stop_overlap_engine();
        let mut model = Model::default();
        let mut submission = input(vec![1], 40);
        submission.stop_token_ids = vec![3];
        let request = engine.submit(submission).unwrap();
        engine.step(&mut model).unwrap();
        let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
        assert!(engine.retain_continuation(&plan));
        engine.continuation_plan().unwrap().unwrap();
        engine
            .complete_continuing_step_with(&mut model, TokenRows::new(&[3, 4], &[0, 2]), Ok)
            .unwrap();
        assert_eq!(request.take_result().unwrap(), [2, 3]);
        let output = match failure {
            0 => Err(Error::rejected("accepted successor unavailable")),
            1 => TokenRows::new(&[], &[0, 0]),
            _ => {
                model.release_fails = true;
                TokenRows::new(&[5], &[0, 1])
            }
        };
        engine
            .complete_step_with(&mut model, output, Ok)
            .unwrap_err();
        assert!(engine.poisoned());
        assert!(request.done() && !request.cancelled());
        assert!(request.take_result().is_err());
        assert_eq!(engine.ready_request_count(), 0);
        assert_eq!(engine.cache_free_parent_count(), Some(7));
        assert!(model.released.is_empty());
        assert!(engine.close(&mut model).is_err());
    }
}

#[test]
fn failure_after_next_enqueue_keeps_transferred_lease_quarantined() {
    for output in [
        Err(Error::rejected("prior result unavailable")),
        TokenRows::new(&[], &[0, 0]),
    ] {
        let mut engine = EngineCore::new(
            RunnerConfig {
                cache_layout: Some(
                    KvLayout::new(
                        4,
                        vec![KvGroupLayout::new(128, 1, KvRetention::FullContext).unwrap()],
                    )
                    .unwrap(),
                ),
                max_accepted_per_step: 4,
                ..Default::default()
            },
            4,
            512,
            SchedulerConfig::default(),
            false,
        )
        .unwrap();
        let mut model = Model::default();
        let request = engine.submit(input(vec![1], 40)).unwrap();
        engine.step(&mut model).unwrap();
        let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
        assert!(engine.retain_continuation(&plan));
        engine.continuation_plan().unwrap().unwrap(); // Next GPU work is accepted.
        engine
            .complete_continuing_step_with(&mut model, output, Ok)
            .unwrap_err();
        assert!(engine.poisoned());
        assert_eq!(engine.ready_request_count(), 0);
        assert_eq!(engine.cache_free_parent_count(), Some(3));
        assert!(model.released.is_empty());
        assert!(!request.done());
        assert!(engine.close(&mut model).is_err());
    }
}
