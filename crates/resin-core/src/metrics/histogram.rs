const INITIAL_SCHEMA: i8 = 4;
const MAX_BUCKETS: usize = 160;

// Standard Prometheus schema-4 boundaries in [0.5, 1), matching
// prometheus-client's native histograms. Lower schemas use subsets of these.
#[allow(clippy::approx_constant)] // Match the standard boundary's exact rounding.
const FRACTION_BOUNDS: [f64; 16] = [
    0.5,
    0.5221368912137069,
    0.5452538663326288,
    0.5693943173783458,
    0.5946035575013605,
    0.620928906036742,
    0.6484197773255048,
    0.6771277734684463,
    0.7071067811865475,
    0.7384130729697496,
    0.7711054127039704,
    0.805245165974627,
    0.8408964152537144,
    0.8781260801866495,
    0.9170040432046711,
    0.9576032806985735,
];

/// Bounded base-2 exponential histogram of nonnegative finite observations.
///
/// Positive bucket `i` has upper bound `2^(i * 2^-schema)`. Counts are
/// noncumulative and contiguous from `offset`; exact zeros are separate.
/// Resolution decreases as needed to keep at most 160 buckets, preserving all
/// observations. Cloning produces an independent snapshot, unlike a collector
/// handle. The representation maps directly to Prometheus native histograms.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Histogram {
    pub schema: i8,
    pub zero_count: u64,
    pub offset: i32,
    pub counts: Vec<u64>,
    pub count: u64,
    pub sum: f64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            schema: INITIAL_SCHEMA,
            zero_count: 0,
            offset: 0,
            counts: Vec::new(),
            count: 0,
            sum: 0.0,
        }
    }
}

impl Histogram {
    /// Validate externally supplied snapshots before merging or exporting them.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(-4..=INITIAL_SCHEMA).contains(&self.schema) || !self.sum.is_finite() || self.sum < 0.0
        {
            return Err("histogram requires schema -4..4 and a finite nonnegative sum");
        }
        if self.counts.len() > MAX_BUCKETS {
            return Err("histogram exceeds 160 buckets");
        }
        if self
            .counts
            .iter()
            .try_fold(self.zero_count, |total, value| total.checked_add(*value))
            != Some(self.count)
        {
            return Err("histogram count must equal zero_count plus bucket counts");
        }
        let scale = 2.0_f64.powi(i32::from(self.schema));
        let minimum = (-1074.0 * scale).ceil() as i32;
        let maximum = (1024.0 * scale).ceil() as i32;
        let valid_indices = if self.counts.is_empty() {
            self.offset == 0
        } else {
            self.offset >= minimum
                && self
                    .offset
                    .checked_add(self.counts.len() as i32 - 1)
                    .is_some_and(|last| last <= maximum)
        };
        if !valid_indices {
            return Err("histogram bucket indices are outside the finite positive range");
        }
        Ok(())
    }

    pub fn observe(&mut self, value: f64) {
        self.observe_weighted(value, 1);
    }

    /// Observe repeated equal values without allocating or looping over them.
    pub fn observe_weighted(&mut self, value: f64, weight: u64) {
        assert!(value.is_finite() && value >= 0.0);
        if weight == 0 {
            return;
        }
        if value == 0.0 {
            self.zero_count += weight;
        } else {
            let mut index = bucket_index(value, self.schema);
            while !self.fits(index, index) {
                self.downscale();
                index = downscaled_index(index, 1);
            }
            self.extend(index, index);
            self.counts[(index - self.offset) as usize] += weight;
        }
        self.count += weight;
        self.sum += value * weight as f64;
    }

    /// Merge independent observations, aligning their resolutions if needed.
    pub fn merge(&mut self, other: &Self) {
        while self.schema > other.schema {
            self.downscale();
        }
        if !other.counts.is_empty() {
            loop {
                let shift = (other.schema - self.schema) as u32;
                let first = downscaled_index(other.offset, shift);
                let last = downscaled_index(other.last_index(), shift);
                if self.fits(first, last) {
                    self.extend(first, last);
                    for (position, count) in other.counts.iter().enumerate() {
                        let index = downscaled_index(other.offset + position as i32, shift);
                        self.counts[(index - self.offset) as usize] += count;
                    }
                    break;
                }
                self.downscale();
            }
        }
        self.zero_count += other.zero_count;
        self.count += other.count;
        self.sum += other.sum;
    }

    fn last_index(&self) -> i32 {
        self.offset + self.counts.len() as i32 - 1
    }

    fn fits(&self, first: i32, last: i32) -> bool {
        let (first, last) = if self.counts.is_empty() {
            (first, last)
        } else {
            (first.min(self.offset), last.max(self.last_index()))
        };
        last - first < MAX_BUCKETS as i32
    }

    fn extend(&mut self, first: i32, last: i32) {
        if self.counts.is_empty() {
            self.offset = first;
        } else if first < self.offset {
            let prefix = (self.offset - first) as usize;
            self.counts.resize(self.counts.len() + prefix, 0);
            self.counts.rotate_right(prefix);
            self.offset = first;
        }
        self.counts
            .resize(self.counts.len().max((last - self.offset + 1) as usize), 0);
    }

    fn downscale(&mut self) {
        // The entire positive finite f64 range fits in 132 buckets at -4.
        debug_assert!(self.schema > -4);
        self.schema -= 1;
        if self.counts.is_empty() {
            return;
        }
        let first = downscaled_index(self.offset, 1);
        let last = downscaled_index(self.last_index(), 1);
        let mut counts = vec![0; (last - first + 1) as usize];
        for (position, count) in self.counts.iter().enumerate() {
            let index = downscaled_index(self.offset + position as i32, 1);
            counts[(index - first) as usize] += count;
        }
        self.offset = first;
        self.counts = counts;
    }
}

fn downscaled_index(index: i32, shift: u32) -> i32 {
    // ceil(index / 2^shift), including negative bucket indexes.
    (index + (1 << shift) - 1) >> shift
}

fn bucket_index(value: f64, schema: i8) -> i32 {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & ((1_u64 << 52) - 1);
    let (fraction, exponent) = if exponent == 0 {
        let highest_bit = 63 - mantissa.leading_zeros();
        (
            mantissa as f64 / (1_u64 << (highest_bit + 1)) as f64,
            highest_bit as i32 - 1073,
        )
    } else {
        (f64::from_bits(mantissa | (1022_u64 << 52)), exponent - 1022)
    };
    // Decomposition avoids log2 rounding near powers of two and preserves
    // subnormal values that a floating-point scale/log expression would lose.
    let bucket = FRACTION_BOUNDS.partition_point(|bound| *bound < fraction) as i32;
    downscaled_index(
        bucket + (exponent - 1) * 16,
        (INITIAL_SCHEMA - schema) as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus_client::{
        encoding::prometheus_protobuf,
        metrics::histogram::{Histogram as NativeHistogram, NativeHistogramConfig},
        registry::Registry,
    };
    use std::collections::BTreeMap;

    #[test]
    fn merging_different_resolutions_matches_observing_all_values() {
        let groups = [
            vec![(0.0, 3), (2.0_f64.powi(-80), 2), (0.25, 7)],
            vec![(1.0, 2), (1.01, 9), (1.25, 4), (2.0_f64.powi(80), 3)],
        ];
        let mut combined = Histogram::default();
        let mut merged = Histogram::default();
        for values in groups {
            let mut shard = Histogram::default();
            for (value, weight) in values {
                shard.observe_weighted(value, weight);
                combined.observe_weighted(value, weight);
            }
            merged.merge(&shard);
        }
        assert!(merged.schema < INITIAL_SCHEMA);
        assert_eq!(merged, combined);
        assert_eq!(merged.validate(), Ok(()));
        assert_eq!(merged.count, 30);
        assert_eq!(merged.zero_count + merged.counts.iter().sum::<u64>(), 30);
    }

    #[test]
    fn indexes_match_native_histogram_for_boundaries_and_adjacent_values() {
        let mut values = vec![0.0, f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX];
        for exponent in [-100, -1, 0, 1, 100] {
            for fraction in FRACTION_BOUNDS {
                let value = fraction * 2.0_f64.powi(exponent);
                values.extend([
                    f64::from_bits(value.to_bits() - 1),
                    value,
                    f64::from_bits(value.to_bits() + 1),
                ]);
            }
        }
        let mut bounded = Histogram::default();
        for &value in &values {
            bounded.observe(value);
            assert!(bounded.counts.len() <= MAX_BUCKETS);
        }
        assert_eq!(
            bounded.zero_count + bounded.counts.iter().sum::<u64>(),
            values.len() as u64
        );
        for schema in -4..=INITIAL_SCHEMA {
            let native = NativeHistogram::new_native(
                NativeHistogramConfig::with_schema(schema).zero_threshold(-1.0),
            );
            let mut expected = BTreeMap::<i32, u64>::new();
            for &value in &values {
                native.observe(value);
                if value > 0.0 {
                    *expected.entry(bucket_index(value, schema)).or_default() += 1;
                }
            }
            let mut registry = Registry::default();
            registry.register("reference", "Native histogram reference", native);
            let families = prometheus_protobuf::encode(&registry).unwrap();
            let native = families[0].metric[0].histogram.as_ref().unwrap();
            assert_eq!(native.zero_count, 1);
            let mut actual = BTreeMap::new();
            let mut deltas = native.positive_delta.iter();
            let mut index = 0;
            let mut count = 0_i64;
            for span in &native.positive_span {
                index += span.offset;
                for _ in 0..span.length {
                    count += deltas.next().unwrap();
                    if count > 0 {
                        actual.insert(index, count as u64);
                    }
                    index += 1;
                }
            }
            assert_eq!(expected, actual, "schema {schema}");
        }
    }
}
