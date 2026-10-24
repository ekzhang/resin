//! Dependency-free request lifecycle and resource ownership for Resin.
//!
//! [`engine::EngineCore`] owns synchronous execution transactions and request
//! progress. Runners receive frozen plans carrying opaque frontend payloads.
//! Its constituent pieces remain available independently:
//! - [`Scheduler`] chooses a phase-homogeneous, key-homogeneous cohort.
//! - [`schedule_ready_prefix`] freezes each row's current token range.
//! - [`resources`] owns allocator state and just-in-time batch leases.
//!
//! The KV coordinator is deliberately single-dispatch, so it cannot plan or
//! lock a future batch before the current lease is released.

#![forbid(unsafe_code)]

mod batch;
pub mod engine;
mod ids;
pub mod logprobs;
pub mod metrics;
pub mod output;
pub mod parallelism;
mod planning;
pub mod resources;
mod scheduler;

pub use batch::{SequenceStep, TokenRange};
pub use ids::RequestId;
pub use parallelism::{Parallelism, ParallelismError};
pub use planning::{PlanningConfig, PlanningError, RequestProgress, schedule_ready_prefix};
pub use scheduler::{Phase, Schedulable, Scheduler, Selection};
