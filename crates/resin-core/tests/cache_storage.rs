use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64};

use resin_core::resources::RequestResourceRef;
use resin_core::resources::kv::{KvGroupFamily, KvGroupLayout, KvLayout, KvRetention};
use resin_core::resources::paged::{CachePoolError, RequestPages};
use resin_core::resources::paged_kv::{
    PagedKvResources, PrefixExport, PrefixImport, PrefixTransfer,
};
use resin_core::{Phase, RequestId, SequenceStep, TokenRange};

type Address = (u32, u32, u32);
type Arena = HashMap<Address, Vec<u8>>;
type Objects = Vec<Vec<Vec<u8>>>;

fn cache_resources(capacity: u32) -> PagedKvResources {
    let groups = vec![
        KvGroupLayout::new(2, 2, KvRetention::FullContext).unwrap(),
        KvGroupLayout::new(
            2,
            3,
            KvRetention::SlidingWindow {
                tokens: NonZeroU32::new(4).unwrap(),
            },
        )
        .unwrap(),
        KvGroupLayout::new_with_family(4, 2, KvRetention::FullContext, KvGroupFamily::State)
            .unwrap(),
    ];
    PagedKvResources::with_prefix_cache(KvLayout::new(capacity, groups).unwrap()).unwrap()
}

fn request(id: u64) -> RequestId {
    RequestId::new(NonZeroU64::new(id).unwrap())
}

fn tokens(branch: u32, length: u32) -> Vec<u32> {
    (0..length)
        .map(|position| branch * 1000 + position)
        .collect()
}

fn objects(branch: u32, view: &PrefixTransfer) -> Objects {
    view.groups
        .iter()
        .map(|group| {
            (0..group.slab_ids.len())
                .map(|index| {
                    [branch, group.group_index, group.first_block + index as u32]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn write(arena: &mut Arena, view: &PrefixTransfer, objects: &Objects) {
    assert_eq!(view.groups.len(), objects.len());
    for (group, objects) in view.groups.iter().zip(objects) {
        assert_eq!(group.slab_ids.len(), objects.len());
        for ((&slab, &slot), bytes) in group.slab_ids.iter().zip(&group.slotnums).zip(objects) {
            arena.insert((group.group_index, slab, slot), bytes.clone());
        }
    }
}

fn read(arena: &Arena, view: &PrefixTransfer) -> Objects {
    view.groups
        .iter()
        .map(|group| {
            group
                .slab_ids
                .iter()
                .zip(&group.slotnums)
                .map(|(&slab, &slot)| arena[&(group.group_index, slab, slot)].clone())
                .collect()
        })
        .collect()
}

#[test]
fn export_pins_survive_eviction_while_live_requests_advance() {
    let mut resources = cache_resources(20);
    let mut arena = Arena::new();
    let import = resources.begin_prefix_import(request(1), 20).unwrap();
    let expected = objects(1, import.view());
    write(&mut arena, import.view(), &expected);
    let mut pages = resources
        .commit_prefix_import(import, &tokens(1, 20))
        .unwrap();
    let counters = resources.prefix_metrics().unwrap();
    let export = resources
        .export_prefix(&tokens(1, 20), 20)
        .unwrap()
        .unwrap();
    assert_eq!(resources.prefix_metrics().unwrap(), counters);
    assert_eq!(
        export
            .view()
            .groups
            .iter()
            .map(|group| group.first_block)
            .collect::<Vec<_>>(),
        [0, 7, 3]
    );
    assert_eq!(read(&arena, export.view()), expected);

    // A pending export pins bytes, not the live request table. Growth and COW
    // must continue while the provider is still reading the old checkpoint.
    resources.ensure_context(&mut pages, 20, 21).unwrap();
    let access = resources
        .acquire(
            &[RequestResourceRef {
                request_id: request(1),
                state: &pages,
            }],
            &[SequenceStep {
                phase: Phase::Decode,
                tokens: TokenRange {
                    start: 20,
                    count: NonZeroU32::new(1).unwrap(),
                },
                sample: true,
            }],
            0,
        )
        .unwrap();
    for group in &access.view.groups {
        let index = group.group_index;
        for (&slab, &slot) in group.append_slab_ids.iter().zip(&group.append_slotnums) {
            arena.insert((index, slab, slot), vec![99]);
        }
    }
    resources.release(access.lease);
    resources.retire(pages).unwrap();
    resources.clear_prefix_cache().unwrap();
    assert_eq!(resources.prefix_metrics().unwrap().entries, 0);
    assert_eq!(resources.pool().occupied_block_count(), 15);
    assert_eq!(
        resources
            .metrics()
            .groups
            .iter()
            .map(|group| group.pinned_blocks)
            .sum::<u64>(),
        15
    );
    assert_eq!(resources.metrics().reclaimable_slabs, 0);
    resources.validate_prefix_export(&export).unwrap();
    assert_eq!(read(&arena, export.view()), expected);

    let other = resources.begin_prefix_import(request(2), 20).unwrap();
    write(&mut arena, other.view(), &objects(2, other.view()));
    assert_eq!(read(&arena, export.view()), expected);
    resources.abort_prefix_import(other).unwrap();
    resources.release_prefix_export(export);
    assert_eq!(resources.pool().free_parent_count(), 20);
}

#[test]
fn complete_hybrid_checkpoint_roundtrips_and_restores_shared_pages() {
    let mut resources = cache_resources(20);
    let mut arena = Arena::new();
    let import = resources.begin_prefix_import(request(1), 20).unwrap();
    let expected = objects(7, import.view());
    write(&mut arena, import.view(), &expected);
    let pages = resources
        .commit_prefix_import(import, &tokens(7, 20))
        .unwrap();
    resources.retire(pages).unwrap();
    let export = resources
        .export_prefix(&tokens(7, 20), 20)
        .unwrap()
        .unwrap();
    let source = export.view().clone();
    let saved = read(&arena, &source);
    resources.clear_prefix_cache().unwrap();
    resources.release_prefix_export(export);
    let import = resources.begin_prefix_import(request(2), 20).unwrap();
    assert_ne!(*import.view(), source);
    write(&mut arena, import.view(), &saved);
    let pages = resources
        .commit_prefix_import(import, &tokens(7, 20))
        .unwrap();
    let mut reused = resources.attach(request(3), 0).unwrap();
    let mut prompt = tokens(7, 20);
    prompt.push(99);
    assert_eq!(resources.restore_prefix(&mut reused, &prompt).unwrap(), 20);
    for (original, attached) in pages.groups().iter().zip(reused.groups()) {
        assert_eq!(
            original
                .pages()
                .iter()
                .map(|page| page.handle())
                .collect::<Vec<_>>(),
            attached
                .pages()
                .iter()
                .map(|page| page.handle())
                .collect::<Vec<_>>()
        );
        assert!(attached.pages().iter().all(|page| !page.is_writable()));
    }
    let restored = resources
        .export_prefix(&tokens(7, 20), 20)
        .unwrap()
        .unwrap();
    assert_eq!(read(&arena, restored.view()), saved);
    resources.retire(pages).unwrap();
    resources.retire(reused).unwrap();
    resources.clear_prefix_cache().unwrap();
    resources.release_prefix_export(restored);
    assert_eq!(resources.pool().free_parent_count(), 20);
}

#[test]
fn import_rejection_returns_owners_without_partial_publication() {
    let mut resources = cache_resources(20);
    let mut foreign = cache_resources(20);
    for length in [0, 3] {
        assert!(matches!(
            resources.begin_prefix_import(request(1), length),
            Err(CachePoolError::InvalidPrefixLength { .. })
        ));
    }
    let import = resources.begin_prefix_import(request(1), 20).unwrap();
    let failure = foreign
        .commit_prefix_import(import, &tokens(1, 20))
        .unwrap_err();
    assert_eq!(failure.error, CachePoolError::ForeignPool);
    let failure = resources
        .commit_prefix_import(failure.import, &[1])
        .unwrap_err();
    assert_eq!(failure.error, CachePoolError::StaleHandle);
    assert!(
        resources
            .export_prefix(&tokens(1, 20), 20)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        resources
            .metrics()
            .groups
            .iter()
            .map(|group| group.pinned_blocks)
            .sum::<u64>(),
        15
    );
    resources.abort_prefix_import(failure.import).unwrap();
    assert_eq!(resources.pool().free_parent_count(), 20);
    assert_eq!(foreign.pool().free_parent_count(), 20);

    let mut small = cache_resources(1);
    assert!(matches!(
        small.begin_prefix_import(request(2), 20),
        Err(CachePoolError::InsufficientCapacity { .. })
    ));
    assert_eq!(small.pool().occupied_block_count(), 0);
    assert_eq!(small.pool().free_parent_count(), 1);
}

#[test]
fn randomized_delayed_transfers_preserve_bytes_and_reclaim_all_owners() {
    let mut resources = cache_resources(24);
    let mut arena = Arena::new();
    let mut pending: Vec<(PrefixImport, Vec<u32>, Objects)> = Vec::new();
    let mut held: Vec<(PrefixExport, Objects)> = Vec::new();
    let mut live: Vec<RequestPages> = Vec::new();
    let mut checkpoints: Vec<(Vec<u32>, u32)> = Vec::new();
    let mut seed = 0x7bd9_3254_ee61_0ca5u64;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut commits = 0;
    let mut aborts = 0;
    let mut exports = 0;
    let mut pressure = 0;
    for step in 0..3000u64 {
        match random() % 8 {
            0 | 1 if pending.len() < 5 => {
                let length = (1 + random() % 5) as u32 * 4;
                let branch = (random() % 12) as u32;
                match resources.begin_prefix_import(request(step + 1), length) {
                    Ok(import) => {
                        let bytes = objects(branch, import.view());
                        write(&mut arena, import.view(), &bytes);
                        pending.push((import, tokens(branch, length), bytes));
                    }
                    Err(CachePoolError::InsufficientCapacity { .. }) => pressure += 1,
                    Err(error) => panic!("unexpected allocation failure: {error}"),
                }
            }
            2 if !pending.is_empty() => {
                let (import, tokens, bytes) =
                    pending.swap_remove(random() as usize % pending.len());
                assert_eq!(read(&arena, import.view()), bytes);
                let length = import.view().resident_tokens;
                live.push(resources.commit_prefix_import(import, &tokens).unwrap());
                checkpoints.push((tokens, length));
                commits += 1;
            }
            3 if !pending.is_empty() => {
                let (import, _, _) = pending.swap_remove(random() as usize % pending.len());
                resources.abort_prefix_import(import).unwrap();
                aborts += 1;
            }
            4 if !live.is_empty() => {
                resources
                    .retire(live.swap_remove(random() as usize % live.len()))
                    .unwrap();
            }
            5 if !checkpoints.is_empty() && held.len() < 8 => {
                let index = checkpoints.len() - 1 - random() as usize % checkpoints.len().min(8);
                let (tokens, length) = &checkpoints[index];
                if let Some(export) = resources.export_prefix(tokens, *length).unwrap() {
                    let bytes = read(&arena, export.view());
                    held.push((export, bytes));
                    exports += 1;
                }
            }
            6 if !held.is_empty() => {
                let (export, _) = held.swap_remove(random() as usize % held.len());
                resources.release_prefix_export(export);
            }
            _ => resources.clear_prefix_cache().unwrap(),
        }
        for (export, bytes) in &held {
            resources.validate_prefix_export(export).unwrap();
            assert_eq!(&read(&arena, export.view()), bytes);
        }
        for (import, _, bytes) in &pending {
            assert_eq!(&read(&arena, import.view()), bytes);
        }
        let metrics = resources.metrics();
        assert_eq!(metrics.free_slabs + metrics.occupied_slabs, 24);
        assert_eq!(
            metrics
                .groups
                .iter()
                .map(|group| group.occupied_blocks)
                .sum::<u64>(),
            resources.pool().occupied_block_count() as u64
        );
    }
    assert!(
        commits > 20 && aborts > 20 && exports > 5 && pressure > 0,
        "commits={commits} aborts={aborts} exports={exports} pressure={pressure}"
    );
    for (import, _, _) in pending {
        resources.abort_prefix_import(import).unwrap();
    }
    for pages in live {
        resources.retire(pages).unwrap();
    }
    resources.clear_prefix_cache().unwrap();
    for (export, _) in held {
        resources.release_prefix_export(export);
    }
    assert_eq!(resources.pool().occupied_block_count(), 0);
    assert_eq!(resources.pool().free_parent_count(), 24);
}
