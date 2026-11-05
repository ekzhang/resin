//! Stateful contracts for aborting preparation before a runner accepts work.
use resin_core::Phase;
use resin_core::engine::{
    EngineCore, Error, ErrorKind, ExecutionPlan, OwnedTokenRows, RequestHandle, Runner,
    RunnerConfig, SchedulerConfig, Submission,
};
use resin_core::resources::kv::{KvGroupLayout, KvGroupPlan, KvLayout, KvRetention};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
struct Payload(Arc<AtomicBool>);
impl Clone for Payload {
    fn clone(&self) -> Self {
        assert!(
            !self.0.swap(false, Ordering::SeqCst),
            "late payload clone failed"
        );
        Self(Arc::clone(&self.0))
    }
}

/// Logical work must be identical on a rejected retry. Batch IDs and optional
/// cache checkpoint reservations may advance without changing that work.
#[derive(Debug, PartialEq, Eq)]
struct Work {
    ids: Vec<u64>,
    tokens: Vec<u32>,
    offsets: Vec<u32>,
    contexts: Vec<u32>,
    prefill_lengths: Vec<u32>,
    sampling: Vec<bool>,
    phase: Phase,
    key: Option<u64>,
}
impl From<&ExecutionPlan<Payload>> for Work {
    fn from(plan: &ExecutionPlan<Payload>) -> Self {
        Self {
            ids: plan.sequence_ids.clone(),
            tokens: plan.token_ids.clone(),
            offsets: plan.token_indptr.clone(),
            contexts: plan.context_lengths.clone(),
            prefill_lengths: plan.prefill_lengths.clone(),
            sampling: plan.sampling_mask.clone(),
            phase: plan.phase,
            key: plan.batch_key,
        }
    }
}
#[derive(Default)]
struct Model {
    work: Vec<Work>,
    tables: Vec<Vec<KvGroupPlan>>,
    reject: bool,
}
impl Runner<Payload> for Model {
    fn forward(&mut self, plan: ExecutionPlan<Payload>) -> Result<OwnedTokenRows, Error> {
        self.work.push(Work::from(&plan));
        self.tables.push(plan.cache_groups.clone());
        if std::mem::take(&mut self.reject) {
            return Err(Error::rejected("not accepted"));
        }
        let tokens = plan
            .token_indptr
            .windows(2)
            .map(|range| plan.token_ids[range[1] as usize - 1] + 1)
            .collect();
        OwnedTokenRows::new(tokens, (0..=plan.sequence_ids.len() as u32).collect())
    }
}
fn engine(prefix: bool) -> EngineCore<Payload> {
    EngineCore::new(
        RunnerConfig {
            continuation_prefill: true,
            cache_layout: Some(
                KvLayout::new(
                    32,
                    vec![
                        KvGroupLayout::new(2, 2, KvRetention::FullContext).unwrap(),
                        KvGroupLayout::new(
                            2,
                            1,
                            KvRetention::SlidingWindow {
                                tokens: NonZeroU32::new(7).unwrap(),
                            },
                        )
                        .unwrap()
                        .with_read_window_tokens(3)
                        .unwrap(),
                    ],
                )
                .unwrap(),
            ),
            ..Default::default()
        },
        2,
        64,
        SchedulerConfig {
            prefill_chunk_size: NonZeroU32::new(2).unwrap(),
            ..Default::default()
        },
        prefix,
    )
    .unwrap()
}
fn submit(
    engine: &mut EngineCore<Payload>,
    prompt: Vec<u32>,
    output: usize,
    failure: &Arc<AtomicBool>,
    priority: i32,
    key: Option<u64>,
) -> RequestHandle {
    let mut input = Submission::new(prompt, Payload(Arc::clone(failure)));
    input.max_output_tokens = Some(output);
    input.priority = priority;
    input.batch_key = key;
    engine.submit(input).unwrap()
}
fn drain(engine: &mut EngineCore<Payload>, model: &mut Model) {
    for _ in 0..64 {
        if !engine.has_work() {
            return;
        }
        engine.step(model).unwrap();
    }
    panic!("finite requests did not drain");
}

#[test]
fn late_clone_failure_preserves_a_prior_prefix_and_releases_the_whole_lease() {
    let mut engine = engine(true);
    let mut model = Model::default();
    let healthy = Arc::new(AtomicBool::new(false));
    let warm = submit(
        &mut engine,
        vec![10, 11, 12, 13, 14, 15],
        1,
        &healthy,
        0,
        None,
    );
    drain(&mut engine, &mut model);
    assert_eq!(warm.take_result().unwrap(), [16]);

    let hit = submit(
        &mut engine,
        vec![10, 11, 12, 13, 14, 15],
        2,
        &healthy,
        0,
        None,
    );
    assert_eq!(hit.cached_tokens(), 4);
    let failing = Arc::new(AtomicBool::new(true));
    let cold = submit(
        &mut engine,
        vec![20, 21, 22, 23, 24, 25],
        2,
        &failing,
        0,
        None,
    );
    let executed = model.work.len();
    assert_eq!(
        engine.prepare_step(&mut model, None).unwrap_err().kind,
        ErrorKind::Rejected
    );
    assert_eq!(model.work.len(), executed);
    assert!(!engine.has_in_flight_step());
    assert!(!engine.poisoned());
    assert_eq!(engine.ready_request_count(), 2);
    assert!(!hit.done() && !cold.done());

    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(plan.sequence_ids, [hit.id(), cold.id()]);
    assert_eq!(plan.context_lengths, [4, 0]);
    assert_eq!(plan.prefill_lengths, [6, 6]);
    assert_eq!(plan.token_ids, [14, 15, 20, 21]);
    assert_eq!(plan.token_indptr, [0, 2, 4]);
    assert_eq!(plan.sampling_mask, [true, false]);
    assert_eq!(plan.cache_groups.len(), 2);
    assert_eq!(
        engine
            .complete_step_with(&mut model, Err(Error::rejected("not launched")), Ok)
            .unwrap_err()
            .kind,
        ErrorKind::Rejected
    );
    assert!(engine.cancel(&mut model, &hit).unwrap());
    assert!(engine.cancel(&mut model, &cold).unwrap());
    engine.clear_prefix_cache().unwrap();
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(32));
}

#[test]
fn decode_preparation_abort_preserves_committed_generation_and_replay_endpoints() {
    let mut engine = engine(false);
    let mut model = Model::default();
    let first = Arc::new(AtomicBool::new(false));
    let second = Arc::new(AtomicBool::new(false));
    let a = submit(&mut engine, vec![10, 11], 4, &first, 0, None);
    let b = submit(&mut engine, vec![20, 21], 4, &second, 0, None);
    engine.step(&mut model).unwrap();
    engine.step(&mut model).unwrap();
    second.store(true, Ordering::SeqCst);
    let executed = model.work.len();
    assert_eq!(
        engine.prepare_step(&mut model, None).unwrap_err().kind,
        ErrorKind::Rejected
    );
    assert_eq!(model.work.len(), executed);
    assert!(!engine.poisoned() && !engine.has_in_flight_step());

    let plan = engine.prepare_step(&mut model, None).unwrap().unwrap();
    assert_eq!(plan.phase, Phase::Decode);
    assert_eq!(plan.sequence_ids, [a.id(), b.id()]);
    assert_eq!(plan.context_lengths, [3, 3]);
    assert_eq!(plan.prefill_lengths, [3, 3]);
    assert_eq!(plan.token_ids, [13, 23]);
    let output = model.forward(plan).unwrap();
    let events = engine
        .complete_step_with(&mut model, Ok(output.view()), Ok)
        .unwrap();
    assert_eq!(
        events
            .iter()
            .map(|event| (event.index, event.token_id))
            .collect::<Vec<_>>(),
        [(2, 14), (2, 24)]
    );
    drain(&mut engine, &mut model);
    assert_eq!(a.take_result().unwrap(), [12, 13, 14, 15]);
    assert_eq!(b.take_result().unwrap(), [22, 23, 24, 25]);
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(32));
}

#[test]
fn repeated_rejections_preserve_priority_key_order_and_logical_work() {
    let mut engine = engine(false);
    let mut model = Model::default();
    let healthy = Arc::new(AtomicBool::new(false));
    let a = submit(&mut engine, vec![10, 11], 2, &healthy, 1, Some(7));
    let b = submit(&mut engine, vec![20, 21], 2, &healthy, 1, Some(7));
    let c = submit(&mut engine, vec![30, 31], 2, &healthy, 0, Some(9));
    for _ in 0..2 {
        model.reject = true;
        assert_eq!(
            engine.step(&mut model).unwrap_err().kind,
            ErrorKind::Rejected
        );
        assert!(!engine.poisoned() && !engine.has_in_flight_step());
        assert_eq!(engine.ready_request_count(), 3);
        assert!(!a.done() && !b.done() && !c.done());
    }
    assert_eq!(model.work[0], model.work[1]);
    assert_eq!(model.tables[0], model.tables[1]);
    assert_eq!(model.work[0].ids, [a.id(), b.id()]);
    assert_eq!(model.work[0].key, Some(7));
    engine.step(&mut model).unwrap();
    assert_eq!(model.work[0], model.work[2]);
    assert_eq!(model.tables[0], model.tables[2]);
    drain(&mut engine, &mut model);
    assert_eq!(a.take_result().unwrap(), [12, 13]);
    assert_eq!(b.take_result().unwrap(), [22, 23]);
    assert_eq!(c.take_result().unwrap(), [32, 33]);
    engine.close(&mut model).unwrap();
    assert_eq!(engine.cache_free_parent_count(), Some(32));
}
