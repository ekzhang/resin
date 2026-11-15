//! Rank membership shared by the scheduler and device runtime.
//!
//! Attention uses `[ATTN_DP, CP, REPLICATION, ATTN_TP]`, while experts use
//! `[EP, EXPERT_TP]` over the same ranks. Unassigned attention ranks replicate
//! the same shard within a request replica; they do not create new schedulers.

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Parallelism {
    world_size: usize,
    attn_dp: usize,
    attn_tp: usize,
    cp: usize,
    ep: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ParallelismError {
    #[error("parallel sizes must be positive integers")]
    InvalidSizes,
    #[error("world_size must be divisible by ep")]
    ExpertNotDivisible,
    #[error("world_size must be divisible by attn_dp * cp * attn_tp")]
    AttentionNotDivisible,
    #[error("attn_dp * cp * attn_tp exceeds usize")]
    AttentionSizeOverflow,
    #[error("rank is outside the topology")]
    InvalidRank,
    #[error("coordinates are outside the topology")]
    InvalidCoordinates,
}

impl Default for Parallelism {
    fn default() -> Self {
        Self {
            world_size: 1,
            attn_dp: 1,
            attn_tp: 1,
            cp: 1,
            ep: 1,
        }
    }
}

impl Parallelism {
    pub fn new(
        world_size: usize,
        attn_dp: usize,
        attn_tp: usize,
        cp: usize,
        ep: usize,
    ) -> Result<Self, ParallelismError> {
        if [world_size, attn_dp, attn_tp, cp, ep].contains(&0) {
            return Err(ParallelismError::InvalidSizes);
        }
        if !world_size.is_multiple_of(ep) {
            return Err(ParallelismError::ExpertNotDivisible);
        }
        let attention_size = attn_dp
            .checked_mul(cp)
            .and_then(|size| size.checked_mul(attn_tp))
            .ok_or(ParallelismError::AttentionSizeOverflow)?;
        if !world_size.is_multiple_of(attention_size) {
            return Err(ParallelismError::AttentionNotDivisible);
        }
        Ok(Self {
            world_size,
            attn_dp,
            attn_tp,
            cp,
            ep,
        })
    }

    pub const fn world_size(&self) -> usize {
        self.world_size
    }

    pub const fn attn_dp(&self) -> usize {
        self.attn_dp
    }

    pub const fn attn_tp(&self) -> usize {
        self.attn_tp
    }

    pub const fn cp(&self) -> usize {
        self.cp
    }

    pub const fn ep(&self) -> usize {
        self.ep
    }

    /// Experts are never replicated: EP and expert TP span the entire world.
    pub fn expert_tp(&self) -> usize {
        self.world_size / self.ep
    }

    /// Copies of each attention shard within the same request replica.
    pub fn attn_replication(&self) -> usize {
        self.world_size / (self.attn_dp * self.cp * self.attn_tp)
    }

    pub fn coordinates(&self, rank: usize) -> Result<[usize; 4], ParallelismError> {
        if rank >= self.world_size {
            return Err(ParallelismError::InvalidRank);
        }
        let attn_tp = rank % self.attn_tp;
        let slot = rank / self.attn_tp;
        let replication = slot % self.attn_replication();
        let slot = slot / self.attn_replication();
        Ok([slot / self.cp, slot % self.cp, replication, attn_tp])
    }

    pub fn rank(
        &self,
        attn_dp: usize,
        cp: usize,
        replication: usize,
        attn_tp: usize,
    ) -> Result<usize, ParallelismError> {
        if attn_dp >= self.attn_dp
            || cp >= self.cp
            || replication >= self.attn_replication()
            || attn_tp >= self.attn_tp
        {
            return Err(ParallelismError::InvalidCoordinates);
        }
        Ok(
            ((attn_dp * self.cp + cp) * self.attn_replication() + replication) * self.attn_tp
                + attn_tp,
        )
    }

    /// Deterministic memberships; device runtimes own the resulting handles.
    pub fn partitions(&self) -> [(&'static str, Vec<Vec<usize>>); 9] {
        let attention = [self.attn_dp, self.cp, self.attn_replication(), self.attn_tp];
        let experts = [self.ep, self.expert_tp()];
        let replica_size = self.world_size / self.attn_dp;
        [
            ("attn_tp", axis_groups(&attention, 3)),
            ("cp", axis_groups(&attention, 1)),
            ("attn_replication", axis_groups(&attention, 2)),
            ("ep", axis_groups(&experts, 0)),
            ("expert_tp", axis_groups(&experts, 1)),
            ("attn_dp", axis_groups(&attention, 0)),
            ("replica", axis_groups(&[self.attn_dp, replica_size], 1)),
            ("domain", vec![(0..self.world_size).collect()]),
            (
                "drivers",
                vec![(0..self.world_size).step_by(replica_size).collect()],
            ),
        ]
    }
}

/// Vary one row-major mesh axis, keeping all other coordinates fixed.
fn axis_groups(shape: &[usize], axis: usize) -> Vec<Vec<usize>> {
    let size = shape[axis];
    let stride: usize = shape[axis + 1..].iter().product();
    let world_size = shape.iter().product();
    (0..world_size)
        .filter(|rank| (rank / stride).is_multiple_of(size))
        .map(|base| (0..size).map(|offset| base + offset * stride).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_sizes_divisibility_and_overflow() {
        for index in 0..5 {
            let mut sizes = [1; 5];
            sizes[index] = 0;
            assert_eq!(
                Parallelism::new(sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]),
                Err(ParallelismError::InvalidSizes)
            );
        }
        assert_eq!(
            Parallelism::new(8, 1, 1, 1, 3),
            Err(ParallelismError::ExpertNotDivisible)
        );
        assert_eq!(
            Parallelism::new(12, 4, 1, 2, 1),
            Err(ParallelismError::AttentionNotDivisible)
        );
        assert_eq!(
            Parallelism::new(usize::MAX, usize::MAX, 2, 1, 1),
            Err(ParallelismError::AttentionSizeOverflow)
        );
        assert_eq!(
            Parallelism::new(usize::MAX, usize::MAX, 1, 2, 1),
            Err(ParallelismError::AttentionSizeOverflow)
        );
        // Valid large sizes do not allocate or overflow during construction.
        assert_eq!(
            Parallelism::new(usize::MAX, 1, usize::MAX, 1, 1)
                .unwrap()
                .attn_replication(),
            1
        );
    }

    #[test]
    fn attention_replication_preserves_request_replicas() {
        let layout = Parallelism::new(16, 2, 2, 2, 4).unwrap();
        assert_eq!(layout.attn_replication(), 2);
        assert_eq!(layout.expert_tp(), 4);
        let groups = layout
            .partitions()
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(groups["attn_tp"][0], [0, 1]);
        assert_eq!(groups["cp"][0], [0, 4]);
        assert_eq!(groups["attn_replication"][0], [0, 2]);
        assert_eq!(groups["attn_dp"][0], [0, 8]);
        assert_eq!(groups["ep"][0], [0, 4, 8, 12]);
        assert_eq!(groups["expert_tp"][0], [0, 1, 2, 3]);
        assert_eq!(
            groups["replica"],
            [(0..8).collect::<Vec<_>>(), (8..16).collect()]
        );
        assert_eq!(groups["drivers"], [vec![0, 8]]);
        assert_eq!(groups["domain"], [(0..16).collect::<Vec<_>>()]);
    }

    #[test]
    fn coordinates_roundtrip_and_partitions_cover_every_rank() {
        for world_size in [1, 2, 4, 8, 12] {
            for attn_dp in 1..=world_size {
                for attn_tp in 1..=world_size {
                    for cp in 1..=world_size {
                        for ep in 1..=world_size {
                            let Ok(layout) = Parallelism::new(world_size, attn_dp, attn_tp, cp, ep)
                            else {
                                continue;
                            };
                            for rank in 0..world_size {
                                let [dp, cp, replication, tp] = layout.coordinates(rank).unwrap();
                                assert_eq!(layout.rank(dp, cp, replication, tp), Ok(rank));
                            }
                            assert_eq!(
                                layout.coordinates(world_size),
                                Err(ParallelismError::InvalidRank)
                            );
                            for coordinates in [
                                [attn_dp, 0, 0, 0],
                                [0, cp, 0, 0],
                                [0, 0, layout.attn_replication(), 0],
                                [0, 0, 0, attn_tp],
                            ] {
                                let [dp, cp, replication, tp] = coordinates;
                                assert_eq!(
                                    layout.rank(dp, cp, replication, tp),
                                    Err(ParallelismError::InvalidCoordinates)
                                );
                            }
                            for (name, groups) in layout.partitions() {
                                if name == "drivers" {
                                    continue;
                                }
                                let mut members: Vec<_> = groups.into_iter().flatten().collect();
                                members.sort_unstable();
                                assert_eq!(
                                    members,
                                    (0..world_size).collect::<Vec<_>>(),
                                    "{name}: {layout:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn singleton_is_the_default() {
        let layout = Parallelism::default();
        assert_eq!(layout, Parallelism::new(1, 1, 1, 1, 1).unwrap());
        assert_eq!(layout.coordinates(0), Ok([0; 4]));
        for (_, groups) in layout.partitions() {
            assert_eq!(groups, [vec![0]]);
        }
    }
}
