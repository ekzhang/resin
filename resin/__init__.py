"""Resin's Rust control plane.

Importing this module intentionally does not import an accelerator framework.
Choose a separately installed model runner only when constructing an engine.
"""

import resin.numa as numa
import resin.process as process
import resin.recipe as recipe
import resin.server as server
from resin._resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupFamily,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
    CacheRetention,
    CacheStatePlan,
    Engine,
    EngineMetrics,
    ForwardOutput,
    Parallelism,
    RequestLease,
    ResourceBinding,
    RunnerConfig,
    RunnerMetrics,
    SamplingParams,
    SchedulerConfig,
    TokenEvent,
    TokenLogprobs,
    __version__,
)
from resin.cache_storage import CacheStorage, MemoryCacheStorage
from resin.protocols import (
    ManagedRunner,
    MediaInput,
    MediaTokenSpan,
    Runner,
)

__all__ = [
    "BatchPhase",
    "BatchPlan",
    "CacheGroupFamily",
    "CacheGroupLayout",
    "CacheGroupPlan",
    "CacheLayout",
    "CacheRetention",
    "CacheStatePlan",
    "CacheStorage",
    "MemoryCacheStorage",
    "Engine",
    "EngineMetrics",
    "ForwardOutput",
    "ManagedRunner",
    "Parallelism",
    "MediaInput",
    "MediaTokenSpan",
    "ResourceBinding",
    "RequestLease",
    "RunnerConfig",
    "RunnerMetrics",
    "SamplingParams",
    "SchedulerConfig",
    "TokenEvent",
    "TokenLogprobs",
    "Runner",
    "__version__",
    "numa",
    "process",
    "recipe",
    "server",
]
