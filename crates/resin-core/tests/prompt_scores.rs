use resin_core::engine::{
    EngineCore, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig, SchedulerConfig, Submission,
};
use resin_core::logprobs::{LogprobView, TopLogprobView};
use std::num::NonZeroU32;

fn engine(chunk: u32) -> EngineCore<()> {
    EngineCore::new(
        RunnerConfig {
            continuation_prefill: true,
            ..Default::default()
        },
        4,
        64,
        SchedulerConfig {
            prefill_chunk_size: NonZeroU32::new(chunk).unwrap(),
            ..Default::default()
        },
        false,
    )
    .unwrap()
}

#[derive(Default)]
struct Model {
    plans: Vec<ExecutionPlan<()>>,
    indptr: Vec<u32>,
    values: Vec<f32>,
    top_indptr: Vec<u32>,
    top_ids: Vec<u32>,
    top_values: Vec<f32>,
    target_indptr: Vec<u32>,
    target_ids: Vec<u32>,
    target_values: Vec<f32>,
    corrupt_target: bool,
}
impl Runner<()> for Model {
    fn forward(&mut self, plan: ExecutionPlan<()>) -> resin_core::engine::Result<OwnedTokenRows> {
        self.indptr = plan.prompt_score_indptr.clone();
        self.values = vec![-0.5; plan.prompt_score_indices.len()];
        self.top_indptr = vec![0];
        self.top_ids.clear();
        for &count in &plan.prompt_score_top_counts {
            self.top_ids.extend(0..count);
            self.top_indptr.push(self.top_ids.len() as u32);
        }
        self.top_values = vec![-0.25; self.top_ids.len()];
        self.target_indptr = plan.prompt_target_indptr.clone();
        self.target_ids = plan.prompt_target_token_ids.clone();
        if self.corrupt_target && !self.target_ids.is_empty() {
            self.target_ids[0] += 1;
        }
        self.target_values = vec![-1.0; self.target_ids.len()];
        let output = OwnedTokenRows::new(
            vec![99; plan.sequence_ids.len()],
            (0..=plan.sequence_ids.len() as u32).collect(),
        );
        self.plans.push(plan);
        output
    }
    fn prompt_logprobs(&self) -> Option<LogprobView<'_>> {
        if self.values.is_empty() {
            return None;
        }
        Some(LogprobView {
            indptr: &self.indptr,
            values: &self.values,
            top: Some(TopLogprobView {
                indptr: &self.top_indptr,
                token_ids: &self.top_ids,
                values: &self.top_values,
            }),
            targets: Some(TopLogprobView {
                indptr: &self.target_indptr,
                token_ids: &self.target_ids,
                values: &self.target_values,
            }),
        })
    }
}

#[test]
fn unscored_plans_do_not_allocate_prompt_target_metadata() {
    let mut engine = engine(4);
    let mut input = Submission::new(vec![1, 2, 3], ());
    input.max_output_tokens = Some(1);
    engine.submit(input).unwrap();
    let mut model = Model::default();
    engine.step(&mut model).unwrap();
    assert_eq!(model.plans[0].prompt_target_indptr.capacity(), 0);
    assert_eq!(model.plans[0].prompt_target_token_ids.capacity(), 0);
}

#[test]
fn prompt_top_and_position_targets_work_alone_or_together_across_chunks() {
    for chunk in [1, 2, 4] {
        for (top, targets) in [
            (2, None),
            (0, Some(vec![vec![], vec![9, 9], vec![], vec![4]])),
            (2, Some(vec![vec![], vec![9, 9], vec![], vec![4]])),
        ] {
            let mut engine = engine(chunk);
            engine.enable_metrics().unwrap();
            let mut input = Submission::new(vec![1, 2, 3, 4], ());
            input.max_output_tokens = Some(0);
            input.prompt_top_logprobs = top;
            input.prompt_target_logprobs = targets.clone();
            let request = engine.submit(input).unwrap();
            let mut model = Model::default();
            while engine.has_work() {
                assert!(engine.step(&mut model).unwrap().is_empty());
            }
            assert!(request.done());
            let metrics = engine.metrics_snapshot().unwrap();
            assert_eq!(metrics.generation_tokens_histogram.count, 1);
            assert_eq!(metrics.generation_tokens_histogram.sum, 0.0);
            assert_eq!(metrics.time_to_first_token_seconds.count, 0);
            assert!(request.take_result().unwrap().is_empty());
            let scores = request.prompt_logprobs().unwrap();
            assert!(scores[0].is_none());
            for (position, score) in scores.iter().enumerate().skip(1) {
                let score = score.as_ref().unwrap();
                assert_eq!(score.top_logprobs.len(), top as usize);
                assert_eq!(
                    score
                        .target_logprobs
                        .iter()
                        .map(|&(id, _)| id)
                        .collect::<Vec<_>>(),
                    targets
                        .as_ref()
                        .map_or(Vec::new(), |rows| rows[position].clone())
                );
            }
            assert_eq!(
                model
                    .plans
                    .iter()
                    .flat_map(|plan| plan.prompt_score_token_ids.iter().copied())
                    .collect::<Vec<_>>(),
                [2, 3, 4]
            );
        }
    }
}

#[test]
fn bad_target_matrix_is_rejected_before_admission() {
    for targets in [vec![], vec![vec![], vec![]], vec![vec![1], vec![], vec![]]] {
        let mut engine = engine(2);
        let mut input = Submission::new(vec![1, 2, 3], ());
        input.prompt_target_logprobs = Some(targets);
        assert!(engine.submit(input).is_err());
        assert!(!engine.has_work());
        assert!(!engine.poisoned());
    }
}

#[test]
fn wrong_target_id_quarantines_before_publishing_any_prompt_scores() {
    let mut engine = engine(2);
    let mut input = Submission::new(vec![1, 2, 3], ());
    input.prompt_target_logprobs = Some(vec![vec![], vec![9], vec![4]]);
    let request = engine.submit(input).unwrap();
    let mut model = Model {
        corrupt_target: true,
        ..Default::default()
    };
    assert!(engine.step(&mut model).is_err());
    assert!(engine.poisoned());
    assert!(!request.done());
    assert!(
        request
            .prompt_logprobs()
            .unwrap()
            .iter()
            .all(Option::is_none)
    );
}
