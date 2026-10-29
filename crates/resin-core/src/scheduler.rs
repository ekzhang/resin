use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Prefill,
    Decode,
}

/// Native metadata required to order and co-batch a ready item.
///
/// Turn ordinals never cross the Python boundary. A `u128` gives the scheduler
/// a practically inexhaustible monotonically increasing age domain: even one
/// billion new turns per second would take vastly longer than the lifetime of
/// any process to exhaust it.
pub trait Schedulable {
    fn priority(&self) -> i32;
    fn batch_key(&self) -> Option<u64>;
    fn phase(&self) -> Phase;
    fn is_running(&self) -> bool;
    fn set_running(&mut self, running: bool);
    fn turn_ordinal(&self) -> u128;
    fn set_turn_ordinal(&mut self, turn: u128);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SelectedIdentity {
    index: usize,
    turn: u128,
}

/// The immutable identity of a ready-cohort selection.
///
/// Planning inspects selected requests before the engine pins and removes
/// them. The epoch and identities make the later removal fail loudly if an
/// accidental queue mutation or reordering is inserted between those steps.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Selection {
    epoch: u64,
    selected: Vec<SelectedIdentity>,
}

impl Selection {
    /// Retain the oldest prefix of a previously selected homogeneous cohort.
    ///
    /// Token-budget planning happens after the engine has inspected the ready
    /// requests. It may admit fewer rows than the sequence limit, but it must
    /// never skip an older matching row to backfill a later one.
    pub fn truncate(&mut self, len: usize) {
        self.selected.truncate(len);
    }
}

/// Prefill-first homogeneous scheduling with sticky request admission.
///
/// Admitted prefill continues before decode, and waiting prefill may use a
/// vacant running slot. Priority and age order compatible requests within the
/// chosen phase. Survivors remain admitted until completion or retraction.
#[derive(Debug)]
pub struct Scheduler<T> {
    ready: Vec<T>,
    max_running_requests: NonZeroUsize,
    next_turn: u128,
    epoch: u64,
}

impl<T: Schedulable> Scheduler<T> {
    pub fn new(max_running_requests: NonZeroUsize) -> Self {
        Self {
            ready: Vec::new(),
            max_running_requests,
            next_turn: 1,
            epoch: 0,
        }
    }

    pub fn max_running_requests(&self) -> usize {
        self.max_running_requests.get()
    }

    pub fn len(&self) -> usize {
        self.ready.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ready.is_empty()
    }

    pub fn enqueue(&mut self, mut item: T) {
        let turn = self.take_fresh_turn();
        item.set_turn_ordinal(turn);
        self.ready.push(item);
        self.bump_epoch();
    }

    pub fn selected(&self) -> (Vec<&T>, Selection) {
        let selected = self.selected_identities();
        self.selection(selected)
    }

    /// Select already-admitted work without considering waiting requests.
    pub fn selected_running(&self) -> (Vec<&T>, Selection) {
        let selected = self.running_identities();
        self.selection(selected)
    }

    fn selection(&self, selected: Vec<SelectedIdentity>) -> (Vec<&T>, Selection) {
        let items = selected
            .iter()
            .map(|identity| &self.ready[identity.index])
            .collect();
        (
            items,
            Selection {
                epoch: self.epoch,
                selected,
            },
        )
    }

    pub fn take_selected(&mut self, selection: Selection) -> Vec<T> {
        assert_eq!(
            selection.epoch, self.epoch,
            "ready queue changed after its dispatch cohort was selected"
        );
        let current = self.selected_identities();
        let running = self.running_identities();
        let matches = |candidate: &[SelectedIdentity]| {
            selection.selected.len() <= candidate.len()
                && selection.selected == candidate[..selection.selected.len()]
        };
        assert!(
            matches(&current) || matches(&running),
            "selected dispatch cohort no longer matches the ready queue"
        );

        let mut selected = std::iter::repeat_with(|| None)
            .take(selection.selected.len())
            .collect::<Vec<Option<T>>>();
        // Dispatch order may put an admitted row before an older waiting row.
        // Walk the queue in index order, retaining each row's dispatch position.
        let mut locations = selection
            .selected
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();
        locations.sort_unstable_by_key(|(_, identity)| identity.index);
        let mut locations = locations.into_iter().peekable();
        let mut remaining = Vec::with_capacity(self.ready.len() - selected.len());
        for (index, item) in self.ready.drain(..).enumerate() {
            if locations
                .peek()
                .is_some_and(|(_, identity)| identity.index == index)
            {
                let (position, identity) = locations.next().expect("selected location was peeked");
                assert_eq!(item.turn_ordinal(), identity.turn);
                selected[position] = Some(item);
            } else {
                remaining.push(item);
            }
        }
        assert!(
            selected.iter().all(Option::is_some),
            "selected index is out of range"
        );
        self.ready = remaining;
        self.bump_epoch();
        selected
            .into_iter()
            .map(|item| item.expect("selected item was found"))
            .collect()
    }

    pub fn restore_rejected(&mut self, selected: Vec<T>) {
        if selected.is_empty() {
            return;
        }
        self.merge_by_turn(selected);
        self.bump_epoch();
    }

    pub fn requeue_survivors(&mut self, mut survivors: Vec<T>) {
        if survivors.is_empty() {
            return;
        }
        for survivor in &mut survivors {
            survivor.set_running(true);
        }
        self.merge_by_turn(survivors);
        self.bump_epoch();
    }

    /// Return a memory-retracted request to the waiting tail.
    pub fn requeue_retracted(&mut self, mut item: T) {
        item.set_running(false);
        let turn = self.take_fresh_turn();
        item.set_turn_ordinal(turn);
        self.ready.push(item);
        self.bump_epoch();
    }

    pub fn position(&self, mut predicate: impl FnMut(&T) -> bool) -> Option<usize> {
        self.ready.iter().position(&mut predicate)
    }

    pub fn remove(&mut self, index: usize) -> T {
        let item = self.ready.remove(index);
        self.bump_epoch();
        item
    }

    pub fn restore(&mut self, index: usize, item: T) {
        self.ready.insert(index, item);
        self.bump_epoch();
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.ready.iter()
    }

    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.bump_epoch();
        self.ready.drain(..)
    }

    fn selected_identities(&self) -> Vec<SelectedIdentity> {
        let limit = self.max_running_requests.get();
        let running = self.running_identities();
        let running_count = self.ready.iter().filter(|item| item.is_running()).count();
        if running
            .first()
            .is_some_and(|identity| self.ready[identity.index].phase() == Phase::Prefill)
        {
            let anchor = &self.ready[running[0].index];
            let free = limit - running_count;
            if free == 0 {
                return running;
            }
            let waiting = self.ready.iter().filter(|item| {
                !item.is_running()
                    && item.batch_key() == anchor.batch_key()
                    && item.phase() == anchor.phase()
            });
            let Some(max_priority) = waiting.clone().map(Schedulable::priority).max() else {
                return running;
            };
            let mut selected = running;
            selected.extend(
                self.ready
                    .iter()
                    .enumerate()
                    .filter(|(_, item)| {
                        !item.is_running()
                            && item.priority() == max_priority
                            && item.batch_key() == anchor.batch_key()
                            && item.phase() == anchor.phase()
                    })
                    .take(free)
                    .map(|(index, item)| SelectedIdentity {
                        index,
                        turn: item.turn_ordinal(),
                    }),
            );
            return selected;
        }

        if running_count < limit {
            let running_key = self
                .ready
                .iter()
                .find(|item| item.is_running())
                .and_then(Schedulable::batch_key);
            let waiting = self.ready.iter().filter(|item| {
                !item.is_running()
                    && item.phase() == Phase::Prefill
                    && (running_count == 0 || item.batch_key() == running_key)
            });
            if let Some(max_priority) = waiting.clone().map(Schedulable::priority).max() {
                let anchor = waiting
                    .clone()
                    .find(|item| item.priority() == max_priority)
                    .expect("maximum waiting priority has an anchor");
                return self
                    .ready
                    .iter()
                    .enumerate()
                    .filter(|(_, item)| {
                        !item.is_running()
                            && item.priority() == max_priority
                            && item.batch_key() == anchor.batch_key()
                            && item.phase() == Phase::Prefill
                    })
                    .take(limit - running_count)
                    .map(|(index, item)| SelectedIdentity {
                        index,
                        turn: item.turn_ordinal(),
                    })
                    .collect();
            }
        }
        if !running.is_empty() {
            return running;
        }

        let waiting = self.ready.iter();
        let Some(max_priority) = waiting.clone().map(Schedulable::priority).max() else {
            return Vec::new();
        };
        let anchor = waiting
            .clone()
            .find(|item| item.priority() == max_priority)
            .expect("maximum waiting priority has an anchor");
        self.ready
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                item.priority() == max_priority
                    && item.batch_key() == anchor.batch_key()
                    && item.phase() == anchor.phase()
            })
            .take(limit)
            .map(|(index, item)| SelectedIdentity {
                index,
                turn: item.turn_ordinal(),
            })
            .collect()
    }

    fn running_identities(&self) -> Vec<SelectedIdentity> {
        let limit = self.max_running_requests.get();
        let Some(anchor) = self
            .ready
            .iter()
            .filter(|item| item.is_running())
            .find(|item| item.phase() == Phase::Prefill)
            .or_else(|| self.ready.iter().find(|item| item.is_running()))
        else {
            return Vec::new();
        };
        let anchor_key = anchor.batch_key();
        let anchor_phase = anchor.phase();
        self.ready
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                item.is_running() && item.batch_key() == anchor_key && item.phase() == anchor_phase
            })
            .take(limit)
            .map(|(index, item)| SelectedIdentity {
                index,
                turn: item.turn_ordinal(),
            })
            .collect()
    }

    fn take_fresh_turn(&mut self) -> u128 {
        let turn = self.next_turn;
        self.next_turn = self
            .next_turn
            .checked_add(1)
            .expect("practically inexhaustible u128 scheduler turn domain exhausted");
        turn
    }

    /// Queue mutations preserve age order. Only the returning batch may arrive
    /// out of order, so sort it and merge instead of sorting every waiting row.
    fn merge_by_turn(&mut self, mut returning: Vec<T>) {
        returning.sort_by_key(Schedulable::turn_ordinal);
        let mut merged = Vec::with_capacity(self.ready.len() + returning.len());
        let mut ready = std::mem::take(&mut self.ready).into_iter().peekable();
        let mut returning = returning.into_iter().peekable();
        while let (Some(queued), Some(returned)) = (ready.peek(), returning.peek()) {
            if queued.turn_ordinal() <= returned.turn_ordinal() {
                merged.push(ready.next().expect("queued item was peeked"));
            } else {
                merged.push(returning.next().expect("returning item was peeked"));
            }
        }
        merged.extend(ready);
        merged.extend(returning);
        self.ready = merged;
    }

    fn bump_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{Phase, Schedulable, Scheduler};

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct Item {
        id: u32,
        priority: i32,
        key: Option<u64>,
        phase: Phase,
        running: bool,
        turn: u128,
    }

    impl Schedulable for Item {
        fn priority(&self) -> i32 {
            self.priority
        }

        fn batch_key(&self) -> Option<u64> {
            self.key
        }

        fn phase(&self) -> Phase {
            self.phase
        }

        fn is_running(&self) -> bool {
            self.running
        }

        fn set_running(&mut self, running: bool) {
            self.running = running;
        }

        fn turn_ordinal(&self) -> u128 {
            self.turn
        }

        fn set_turn_ordinal(&mut self, turn: u128) {
            self.turn = turn;
        }
    }

    fn item(id: u32, priority: i32, key: Option<u64>, phase: Phase) -> Item {
        Item {
            id,
            priority,
            key,
            phase,
            running: false,
            turn: 0,
        }
    }

    fn scheduler(limit: usize, items: Vec<Item>) -> Scheduler<Item> {
        let mut scheduler = Scheduler::new(NonZeroUsize::new(limit).unwrap());
        for item in items {
            scheduler.enqueue(item);
        }
        scheduler
    }

    fn default_scheduler(limit: usize, count: u32) -> Scheduler<Item> {
        scheduler(
            limit,
            (0..count)
                .map(|id| item(id, 0, None, Phase::Prefill))
                .collect(),
        )
    }

    fn ready_ids(scheduler: &Scheduler<Item>) -> Vec<u32> {
        scheduler.iter().map(|item| item.id).collect()
    }

    fn ready_id_turns(scheduler: &Scheduler<Item>) -> Vec<(u32, u128)> {
        scheduler.iter().map(|item| (item.id, item.turn)).collect()
    }

    fn selected_ids(scheduler: &Scheduler<Item>) -> Vec<u32> {
        scheduler
            .selected()
            .0
            .into_iter()
            .map(|item| item.id)
            .collect()
    }

    #[test]
    fn survivors_remain_ahead_of_waiting_requests() {
        for queue_len in 0_u32..=8 {
            for limit in 1_usize..=8 {
                let scheduler = default_scheduler(limit, queue_len);
                assert_eq!(
                    selected_ids(&scheduler),
                    (0..queue_len.min(limit as u32)).collect::<Vec<_>>()
                );
                if queue_len == 0 {
                    continue;
                }
                let selected_len = (queue_len as usize).min(limit);
                for retired_mask in 0_u32..(1_u32 << selected_len) {
                    let mut scheduler = default_scheduler(limit, queue_len);
                    let (_, selection) = scheduler.selected();
                    let selected = scheduler.take_selected(selection);
                    let survivors = selected
                        .into_iter()
                        .enumerate()
                        .filter_map(|(row, request)| {
                            ((retired_mask & (1 << row)) == 0).then_some(request)
                        })
                        .collect::<Vec<_>>();
                    let survivor_ids = survivors.iter().map(|item| item.id).collect::<Vec<_>>();
                    scheduler.requeue_survivors(survivors);

                    let mut expected = survivor_ids;
                    expected.extend(selected_len as u32..queue_len);
                    assert_eq!(
                        ready_ids(&scheduler),
                        expected,
                        "len={queue_len}, limit={limit}, retired={retired_mask:#b}"
                    );
                }
            }
        }
    }

    #[test]
    fn admitted_survivors_fill_vacancies_without_yielding_the_running_set() {
        let mut scheduler = default_scheduler(2, 3);
        let (_, selection) = scheduler.selected();
        let mut admitted = scheduler.take_selected(selection);
        admitted.pop();
        scheduler.requeue_survivors(admitted);
        assert_eq!(selected_ids(&scheduler), vec![0, 2]);

        let (_, selection) = scheduler.selected();
        let newcomer = scheduler.take_selected(selection);
        scheduler.requeue_survivors(newcomer);
        assert_eq!(selected_ids(&scheduler), vec![0, 2]);
        assert_eq!(
            scheduler
                .selected_running()
                .0
                .into_iter()
                .map(|item| item.id)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn retracted_request_returns_to_the_waiting_tail() {
        let mut scheduler = default_scheduler(2, 3);
        let (_, selection) = scheduler.selected();
        let mut admitted = scheduler.take_selected(selection);
        let retracted = admitted.remove(0);
        scheduler.requeue_survivors(admitted);
        scheduler.requeue_retracted(retracted);
        assert_eq!(ready_ids(&scheduler), vec![1, 2, 0]);
        assert_eq!(selected_ids(&scheduler), vec![1, 2]);
    }

    #[test]
    fn exhaustive_small_queues_match_anchor_and_homogeneity_reference() {
        let classes = [
            (-1, None, Phase::Prefill),
            (-1, Some(7), Phase::Decode),
            (0, None, Phase::Decode),
            (0, Some(7), Phase::Prefill),
            (1, None, Phase::Prefill),
            (1, None, Phase::Decode),
            (1, Some(7), Phase::Prefill),
            (1, Some(7), Phase::Decode),
        ];
        for queue_len in 1_u32..=4 {
            let case_count = classes.len().pow(queue_len);
            for mut case in 0..case_count {
                let mut items = Vec::new();
                for id in 0..queue_len {
                    let (priority, key, phase) = classes[case % classes.len()];
                    case /= classes.len();
                    items.push(item(id, priority, key, phase));
                }
                for limit in 1..=queue_len as usize {
                    let scheduler = scheduler(limit, items.clone());
                    let preferred_phase = if items.iter().any(|item| item.phase == Phase::Prefill) {
                        Phase::Prefill
                    } else {
                        Phase::Decode
                    };
                    let max_priority = items
                        .iter()
                        .filter(|item| item.phase == preferred_phase)
                        .map(|item| item.priority)
                        .max()
                        .unwrap();
                    let anchor = items
                        .iter()
                        .find(|item| item.priority == max_priority && item.phase == preferred_phase)
                        .unwrap();
                    let expected = items
                        .iter()
                        .filter(|candidate| {
                            candidate.priority == max_priority
                                && candidate.key == anchor.key
                                && candidate.phase == anchor.phase
                        })
                        .take(limit)
                        .map(|item| item.id)
                        .collect::<Vec<_>>();
                    assert_eq!(selected_ids(&scheduler), expected);
                }
            }
        }
    }

    #[test]
    fn priority_is_strict_and_never_backfills_an_incompatible_cohort() {
        let scheduler = scheduler(
            4,
            vec![
                item(0, 0, None, Phase::Prefill),
                item(1, 2, Some(9), Phase::Prefill),
                item(2, 1, Some(9), Phase::Prefill),
                item(3, 2, None, Phase::Prefill),
            ],
        );
        assert_eq!(selected_ids(&scheduler), vec![1]);
    }

    #[test]
    fn none_is_a_real_key_and_phase_is_always_homogeneous() {
        let items = vec![
            item(0, 0, None, Phase::Prefill),
            item(1, 0, Some(0), Phase::Prefill),
            item(2, 0, None, Phase::Decode),
        ];
        let phase_homogeneous = scheduler(3, items);
        assert_eq!(selected_ids(&phase_homogeneous), vec![0]);
    }

    #[test]
    fn survivors_retain_age_ahead_of_tail_and_inflight_submissions() {
        let mut scheduler = scheduler(
            2,
            (0..3).map(|id| item(id, 0, None, Phase::Prefill)).collect(),
        );
        let original_turns = ready_id_turns(&scheduler);
        let (_, selection) = scheduler.selected();
        let selected = scheduler.take_selected(selection);
        scheduler.enqueue(item(3, 0, None, Phase::Prefill));
        scheduler.requeue_survivors(vec![selected[0].clone(), selected[1].clone()]);
        assert_eq!(ready_ids(&scheduler), vec![0, 1, 2, 3]);
        assert_eq!(&ready_id_turns(&scheduler)[..2], &original_turns[..2]);
    }

    #[test]
    fn rejection_restores_exact_noncontiguous_items_and_ordinals() {
        let mut scheduler = scheduler(
            3,
            vec![
                item(0, 1, None, Phase::Prefill),
                item(1, 0, None, Phase::Prefill),
                item(2, 1, None, Phase::Prefill),
                item(3, 1, Some(4), Phase::Prefill),
                item(4, 1, None, Phase::Prefill),
            ],
        );
        let original = ready_id_turns(&scheduler);
        assert_eq!(selected_ids(&scheduler), vec![0, 2, 4]);
        let (_, selection) = scheduler.selected();
        let rejected = scheduler.take_selected(selection);
        scheduler.restore_rejected(rejected);
        assert_eq!(ready_id_turns(&scheduler), original);
    }

    #[test]
    fn truncated_noncontiguous_prefix_takes_and_restores_exact_identities() {
        let mut scheduler = scheduler(
            4,
            vec![
                item(0, 1, Some(7), Phase::Prefill),
                item(1, 1, None, Phase::Prefill),
                item(2, 1, Some(7), Phase::Prefill),
                item(3, 0, Some(7), Phase::Prefill),
                item(4, 1, Some(7), Phase::Prefill),
            ],
        );
        let original = ready_id_turns(&scheduler);
        let (selected, mut selection) = scheduler.selected();
        assert_eq!(
            selected.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![0, 2, 4]
        );

        selection.truncate(2);
        let taken = scheduler.take_selected(selection);
        assert_eq!(
            taken.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(ready_ids(&scheduler), vec![1, 3, 4]);

        scheduler.restore_rejected(taken);
        assert_eq!(ready_id_turns(&scheduler), original);
    }

    #[test]
    fn cancellation_rollback_restores_exact_index_and_ordinal() {
        for index in 0..5 {
            let mut scheduler = default_scheduler(3, 5);
            let original = ready_id_turns(&scheduler);
            let request = scheduler.remove(index);
            scheduler.restore(index, request);
            assert_eq!(ready_id_turns(&scheduler), original);
        }
    }

    #[test]
    #[should_panic(expected = "ready queue changed")]
    fn selection_cannot_be_taken_after_an_intervening_mutation() {
        let mut scheduler = default_scheduler(2, 3);
        let (_, selection) = scheduler.selected();
        scheduler.enqueue(item(3, 0, None, Phase::Prefill));
        let _ = scheduler.take_selected(selection);
    }

    #[test]
    fn drain_preserves_global_turn_order_and_empties_scheduler() {
        let mut scheduler = default_scheduler(2, 5);
        assert_eq!(
            scheduler.drain().map(|item| item.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert!(scheduler.is_empty());
    }

    #[test]
    fn dispatch_order_can_differ_from_queue_age_without_changing_requeue_order() {
        let waiting = item(1, 0, Some(7), Phase::Prefill);
        let mut running = item(2, 5, Some(7), Phase::Prefill);
        running.running = true;
        let mut scheduler = scheduler(2, vec![waiting, running]);

        let (_, selection) = scheduler.selected();
        let selected = scheduler.take_selected(selection);
        assert_eq!(
            selected.iter().map(|item| item.id).collect::<Vec<_>>(),
            [2, 1]
        );
        scheduler.enqueue(item(3, 0, None, Phase::Prefill));
        scheduler.requeue_survivors(selected);
        assert_eq!(ready_ids(&scheduler), [1, 2, 3]);

        let (_, selection) = scheduler.selected();
        let rejected = scheduler.take_selected(selection);
        scheduler.enqueue(item(4, 0, None, Phase::Prefill));
        scheduler.restore_rejected(rejected);
        assert_eq!(ready_ids(&scheduler), [1, 2, 3, 4]);
    }
}
