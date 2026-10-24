//! CPU-only control-plane workloads, with no model execution or dependencies.
//!
//! Run with `cargo run -p resin-core --release --example control_plane_bench`.
//! An optional argument selects `validation`, `queue`, `snapshots`, or `lru`.
//! Build once, then run the binary several times to compare medians. To measure
//! snapshot memory, run only `snapshots` under the platform's process-RSS tool;
//! for example, `/usr/bin/time -l` on macOS. Timings exclude workload setup.

use std::hint::black_box;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::time::Instant;

use resin_core::resources::kv::{KvGroupLayout, KvLayout, KvRetention};
use resin_core::resources::paged::{CachePool, GroupDemand, GroupId, SequenceId};
use resin_core::resources::paged_kv::PagedKvResources;
use resin_core::{Phase, RequestId, Schedulable, Scheduler};

fn report(name: &str, started: Instant) {
    println!("{name}_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
}

fn validation() {
    let mut pool = CachePool::new(4096, &[NonZeroU32::new(1).unwrap()]).unwrap();
    let pending = pool
        .reserve_request(
            SequenceId::new(1),
            &[GroupDemand::new(GroupId::new(0), 4096)],
        )
        .unwrap();
    let request = pool.promote_request(pending);
    let started = Instant::now();
    for _ in 0..100 {
        black_box(pool.validate_request(black_box(&request))).unwrap();
    }
    report("validation_4096_pages_100x", started);
    pool.retire_request(request).unwrap();
}

struct Row {
    turn: u128,
    running: bool,
}

impl Schedulable for Row {
    fn priority(&self) -> i32 {
        0
    }
    fn batch_key(&self) -> Option<u64> {
        None
    }
    fn phase(&self) -> Phase {
        Phase::Decode
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

fn queue() {
    let mut scheduler = Scheduler::new(NonZeroUsize::new(512).unwrap());
    for _ in 0..100_000 {
        scheduler.enqueue(Row {
            turn: 0,
            running: false,
        });
    }
    let started = Instant::now();
    for _ in 0..20 {
        let (_, selection) = scheduler.selected();
        let rows = scheduler.take_selected(selection);
        scheduler.requeue_survivors(rows);
    }
    report("queue_100000_ready_512_batch_20x", started);
    assert_eq!(scheduler.len(), 100_000);
}

fn snapshots() {
    let mut pool = CachePool::new(2048, &[NonZeroU32::new(1).unwrap()]).unwrap();
    let pending = pool
        .reserve_request(SequenceId::new(1), &[GroupDemand::new(GroupId::new(0), 0)])
        .unwrap();
    let mut request = pool.promote_request(pending);
    let mut snapshots = Vec::new();
    let started = Instant::now();
    for count in 1..=2048 {
        let demands = [GroupDemand::new(GroupId::new(0), count)];
        pool.grow_request(&mut request, &demands).unwrap();
        snapshots.push(pool.share_request_prefix(&mut request, &demands).unwrap());
    }
    report("snapshots_2048_growing_publish", started);
    let started = Instant::now();
    for snapshot in snapshots {
        pool.retire_cached_prefix(snapshot).unwrap();
    }
    report("snapshots_2048_clear", started);
    pool.retire_request(request).unwrap();
    assert_eq!(pool.free_parent_count(), 2048);
}

fn lru() {
    let count = 16_384;
    let layout = KvLayout::new(
        count,
        vec![KvGroupLayout::new(1, 1, KvRetention::FullContext).unwrap()],
    )
    .unwrap();
    let mut cache = PagedKvResources::with_prefix_cache(layout).unwrap();
    for token in 0..count {
        let id = RequestId::new(NonZeroU64::new(u64::from(token) + 1).unwrap());
        let mut request = cache.attach(id, 1).unwrap();
        cache.publish_prefix(&mut request, &[token], 1).unwrap();
        cache.retire(request).unwrap();
    }
    assert_eq!(cache.prefix_metrics().unwrap().entries, u64::from(count));
    let started = Instant::now();
    cache.clear_prefix_cache().unwrap();
    report("lru_16384_entries_clear", started);
    assert_eq!(cache.pool().free_parent_count(), count as usize);
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        None | Some("all") => {
            validation();
            queue();
            snapshots();
            lru();
        }
        Some("validation") => validation(),
        Some("queue") => queue(),
        Some("snapshots") => snapshots(),
        Some("lru") => lru(),
        Some(_) => {
            eprintln!("usage: control_plane_bench [all|validation|queue|snapshots|lru]");
            std::process::exit(2);
        }
    }
}
