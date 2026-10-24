from collections.abc import Sequence
from typing import Any, Final, final
from collections.abc import Mapping
from typing import Literal
from resin.recipe import Recipe
from resin.recipe import ContentBlock
import numpy as np
import numpy.typing as npt
from resin.protocols import MediaInput, MediaTokenSpan, Runner, TokenSpan

__version__: Final[str]

@final
class Audio:
    def __new__(cls, /, source: Any) -> Audio: ...
    @property
    def source(self, /) -> Any: ...

@final
class BatchPhase:
    DECODE: Final[BatchPhase]
    PREFILL: Final[BatchPhase]
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class BatchPlan:
    """
    Immutable execution plan for one prefill or decode step.

    This runs one forward pass on a batch of sequences. `token_ids` contains
    the newly scheduled input tokens, packed by sequence into one flat array;
    `token_indptr` is the CSR row pointer, so sequence `i` owns
    `token_ids[token_indptr[i]:token_indptr[i + 1]]`.

    `context_lengths[i]` is the number of tokens processed before this step,
    so that sequence's new tokens occupy consecutive absolute positions
    beginning there. `cache_groups` contains one placement plan per declared
    `CacheLayout` group, and every group plan covers every sequence.
    `prefill_lengths` records the final endpoint of each prefill history,
    including generated inputs being reconstructed after retraction. It stays
    fixed across those prefill chunks and is only meaningful during prefill.
    Public constructors that omit it infer each endpoint from its scheduled end.
    """
    def __new__(
        cls,
        /,
        sequence_ids: Sequence[int] | npt.NDArray[np.uint64],
        context_lengths: Sequence[int] | npt.NDArray[np.uint32],
        token_ids: Sequence[int] | npt.NDArray[np.uint32],
        token_indptr: Sequence[int] | npt.NDArray[np.uint32],
        *,
        prefill_lengths: Sequence[int] | npt.NDArray[np.uint32] | None = None,
        phase: BatchPhase = ...,
        batch_id: int = 0,
        batch_key: int | None = None,
        cache_groups: Sequence[CacheGroupPlan] | None = None,
        sampling_mask: Sequence[bool] | npt.NDArray[np.bool_] | None = None,
        sampling_params: Sequence[SamplingParams] | None = None,
        media: Sequence[MediaInput | None] | None = None,
        prompt_score_indices: Sequence[int] | npt.NDArray[np.uint32] | None = None,
        prompt_score_token_ids: Sequence[int] | npt.NDArray[np.uint32] | None = None,
        prompt_score_indptr: Sequence[int] | npt.NDArray[np.uint32] | None = None,
        prompt_score_top_counts: Sequence[int] | npt.NDArray[np.uint32] | None = None,
        prompt_target_indptr: Sequence[int] | npt.NDArray[np.uint32] | None = None,
        prompt_target_token_ids: Sequence[int] | npt.NDArray[np.uint32] | None = None,
    ) -> BatchPlan: ...
    def __repr__(self, /) -> str: ...
    @staticmethod
    def _from_wire(
        data: npt.NDArray[np.uint8], requests: dict[int, Any], pending: dict[int, Any]
    ) -> BatchPlan: ...
    def _to_wire(self, /) -> npt.NDArray[np.uint8]: ...
    @property
    def batch_id(self, /) -> int:
        """
        Monotonic identifier for tracing and correlating runner steps.
        """
    @property
    def batch_key(self, /) -> int | None:
        """
        Optional key that prevents incompatible requests from sharing a batch.
        """
    @property
    def cache_groups(self, /) -> tuple[CacheGroupPlan, ...]:
        """
        Ordered physical placement plans corresponding to CacheLayout groups.
        There is one plan per layout group, not one plan per sequence; each plan
        contains one block-table row for every sequence in this batch.
        """
    @property
    def context_lengths(self, /) -> npt.NDArray[np.uint32]:
        """
        Number of tokens already processed for each sequence.
        """
    @property
    def media(self, /) -> tuple[MediaInput | None, ...]:
        """
        Opaque runner inputs, one per sequence. The scheduler never inspects them.
        """
    @property
    def needs_logprobs(self, /) -> bool:
        """
        Check natively so token-only graph replay does not box per-row policies.
        """
    @property
    def needs_prompt_logprobs(self, /) -> bool: ...
    @property
    def num_sequences(self, /) -> int:
        """
        Number of sequences (batch size).
        """
    @property
    def num_tokens(self, /) -> int:
        """
        Number of flattened scheduled tokens.
        """
    @property
    def phase(self, /) -> BatchPhase:
        """
        Whether this is a prefill or decode batch. Mixed-chunk is not supported.
        """
    @property
    def positions(self, /) -> npt.NDArray[np.uint32]:
        """
        Absolute position of every flattened scheduled token.
        """
    @property
    def prefill_lengths(self, /) -> npt.NDArray[np.uint32]:
        """
        Final history endpoint for each prefill, including retracted generated inputs.
        Only meaningful during prefill; constant across that history's chunks.
        """
    @property
    def prompt_score_indices(self, /) -> npt.NDArray[np.uint32]: ...
    @property
    def prompt_score_indptr(self, /) -> npt.NDArray[np.uint32]: ...
    @property
    def prompt_score_token_ids(self, /) -> npt.NDArray[np.uint32]: ...
    @property
    def prompt_score_top_counts(self, /) -> npt.NDArray[np.uint32]: ...
    @property
    def prompt_target_indptr(self, /) -> npt.NDArray[np.uint32]: ...
    @property
    def prompt_target_token_ids(self, /) -> npt.NDArray[np.uint32]: ...
    @property
    def sampling_mask(self, /) -> npt.NDArray[np.bool_]:
        """
        Which sequences should generate a sampled token. Used for chunked
        prefill, when some sequences only advance the cache.
        """
    @property
    def sampling_params(self, /) -> tuple[SamplingParams, ...]:
        """
        Effective sampling parameters, one per sequence.
        """
    @property
    def sequence_ids(self, /) -> npt.NDArray[np.uint64]:
        """
        Stable request identifier for each sequence.
        """
    @property
    def token_ids(self, /) -> npt.NDArray[np.uint32]:
        """
        Flattened scheduled token IDs, grouped by sequence.
        """
    @property
    def token_indptr(self, /) -> npt.NDArray[np.uint32]:
        """
        Offsets into `token_ids` for each sequence.
        """
    def token_range(self, /, sequence_index: int) -> tuple[int, int]:
        """
        Return the half-open range in `token_ids` for one sequence.
        """

@final
class CacheGroupFamily:
    """
    Whether a group stores token history or recurrent checkpoints.
    """

    HISTORY: Final[CacheGroupFamily]
    STATE: Final[CacheGroupFamily]
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class CacheGroupLayout:
    """
    Geometry and retention policy for one type of cache.

    Cache groups share a pool of equal-sized slabs. `blocks_per_slab` is how
    many of this group's blocks fit in one slab.
    """
    def __new__(
        cls,
        /,
        block_tokens: int,
        *,
        blocks_per_slab: int = 1,
        retention: CacheRetention = ...,
        window_tokens: int | None = None,
        family: CacheGroupFamily = ...,
        read_window_tokens: int | None = None,
    ) -> CacheGroupLayout: ...
    def __repr__(self, /) -> str: ...
    @property
    def block_tokens(self, /) -> int:
        """
        Number of token positions represented by one logical block.
        """
    @property
    def blocks_per_slab(self, /) -> int:
        """
        Number of this group's blocks packed into one slab.
        """
    @property
    def family(self, /) -> CacheGroupFamily:
        """
        Whether this group stores token history or recurrent checkpoints.
        """
    @property
    def read_window_tokens(self, /) -> int | None:
        """
        Logical read window including the current token. Defaults to the full
        retained window; reserve extra `window_tokens` for simultaneous writes.
        """
    @property
    def retention(self, /) -> CacheRetention:
        """
        Lifetime policy for logical blocks in this group.
        """
    @property
    def window_resident_block_count(self, /) -> int | None:
        """
        Maximum resident logical blocks per sequence for a window group.
        """
    @property
    def window_tokens(self, /) -> int | None:
        """
        Retained suffix length for `WINDOW` groups, otherwise `None`.
        """

@final
class CacheGroupPlan:
    """
    Plan for one type of cache in a BatchPlan.

    A model can have one or more types of cache, such as full attention,
    sliding-window attention, or recurrent state, each described by one
    `CacheGroupLayout`. Cache memory is organized into fungible slabs. Each
    slab is assigned to one cache group and holds `blocks_per_slab` blocks.

    A block is indexed by `(slab_id, slotnum)`: `block_slab_ids` contains
    one-based slab IDs and `block_slotnums` contains zero-based slot numbers. A
    cached token location is `(block, block_offset)`, equivalently
    `(slab_id, slotnum, block_offset)`.

    The plan covers every sequence in the batch. Adjacent `block_indptr`
    values select each sequence's chronological block-table row.
    `block_table_start_indices[i]` is sequence `i`'s block-table start index, so
    row entry `j` represents block `block_table_start_indices[i] + j`; block
    `b` begins at token position `b * CacheGroupLayout.block_tokens`. Full-
    attention tables start at zero; sliding-window and recurrent tables advance
    as old blocks leave their resident rings.

    For flattened batch token `k`, the `append_*` arrays supply exactly that
    cached-token location: `(append_slab_ids[k], append_slotnums[k],
    append_block_offsets[k])`.
    """
    def __new__(
        cls,
        /,
        group_index: int,
        block_indptr: Sequence[int] | npt.NDArray[np.uint32],
        block_slab_ids: Sequence[int] | npt.NDArray[np.uint32],
        block_slotnums: Sequence[int] | npt.NDArray[np.uint32],
        block_table_start_indices: Sequence[int] | npt.NDArray[np.uint32],
        append_slab_ids: Sequence[int] | npt.NDArray[np.uint32],
        append_slotnums: Sequence[int] | npt.NDArray[np.uint32],
        append_block_offsets: Sequence[int] | npt.NDArray[np.uint32],
        *,
        blocks_per_slab: int = 1,
    ) -> CacheGroupPlan: ...
    def __repr__(self, /) -> str: ...
    @property
    def append_block_offsets(self, /) -> npt.NDArray[np.uint32]:
        """
        Token offset within each token's destination block. Entry `k`
        corresponds to `BatchPlan.token_ids[k]` and is less than the matching
        `CacheGroupLayout.block_tokens`.
        """
    @property
    def append_slab_ids(self, /) -> npt.NDArray[np.uint32]:
        """
        One-based slab ID of each token's write target. Entry `k`
        corresponds to `BatchPlan.token_ids[k]`.
        """
    @property
    def append_slotnums(self, /) -> npt.NDArray[np.uint32]:
        """
        Zero-based slot number of each token's write target. Entry `k`
        corresponds to `BatchPlan.token_ids[k]`.
        """
    @property
    def block_indptr(self, /) -> npt.NDArray[np.uint32]:
        """
        CSR row pointer into the block-address arrays. Its length is one more
        than the batch sequence count.
        """
    @property
    def block_slab_ids(self, /) -> npt.NDArray[np.uint32]:
        """
        One-based slab ID for each flattened block-table entry. Pair with
        `block_slotnums`; slab zero is reserved as null.
        """
    @property
    def block_slotnums(self, /) -> npt.NDArray[np.uint32]:
        """
        Zero-based slot number for each flattened block-table entry. Pair with
        `block_slab_ids`.
        """
    @property
    def block_table_start_indices(self, /) -> npt.NDArray[np.uint32]:
        """
        Absolute logical-block index at which each sequence's table row starts.
        This is zero for full-attention history and advances for sliding-window
        and recurrent rings. It has one entry per sequence and is unrelated to
        the physical slab ID.
        """
    @property
    def blocks_per_slab(self, /) -> int:
        """
        Number of this group's blocks packed into one slab. Valid slot numbers
        are `0 <= slotnum < blocks_per_slab`.
        """
    @property
    def group_index(self, /) -> int:
        """
        Zero-based index of this plan's matching CacheGroupLayout. BatchPlan
        orders group plans by this index and rejects gaps or reordering.
        """

@final
class CacheLayout:
    """
    Immutable grouped-cache geometry owned by one Engine.

    `parent_capacity` is the number of fungible physical slabs shared by every
    group. Group order is stable and identifies each group in BatchPlan.
    `replay_suffix_tokens` caps prefix hits so every request rebuilds that many
    final prompt tokens. Models using this policy must not consume omitted
    consumer prefix state: cached opaque bytes are not valid until rebuilt.
    """
    def __new__(
        cls,
        /,
        parent_capacity: int,
        groups: Sequence[CacheGroupLayout],
        *,
        replay_suffix_tokens: int = 0,
    ) -> CacheLayout: ...
    def __repr__(self, /) -> str: ...
    def estimated_parent_count(
        self, /, context_lengths: Sequence[int], *, shared_prefix_tokens: int = 0
    ) -> int:
        """
        Physical slabs needed for contexts in an empty pool.
        Applies retention and packing across requests without allocating pages;
        an aligned shared prefix may be declared, and demand may exceed capacity.
        """
    @property
    def group_count(self, /) -> int:
        """
        Number of logical cache groups.
        """
    def group_kernel_page_capacity(self, /, group_index: int) -> int: ...
    def group_kernel_page_count(self, /, group_index: int) -> int: ...
    @property
    def groups(self, /) -> tuple[CacheGroupLayout, ...]:
        """
        Ordered immutable cache-group descriptors.
        """
    @property
    def parent_capacity(self, /) -> int:
        """
        Number of fungible physical slabs in the pool.
        """
    @property
    def replay_suffix_tokens(self, /) -> int:
        """
        Minimum final prompt suffix rebuilt after every prefix-cache hit.
        """

@final
class CacheRetention:
    """
    Cache-block lifetime policy.

    For history groups, `HISTORY` retains every logical block and `WINDOW`
    retains the suffix selected by `CacheGroupLayout.window_tokens`. Recurrent
    state groups use `HISTORY` and retain their two-page checkpoint ring.
    """

    HISTORY: Final[CacheRetention]
    WINDOW: Final[CacheRetention]
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class DeepSeekV41Image:
    def __new__(
        cls,
        /,
        start: int,
        patches: Any,
        n_vit_h: int,
        n_vit_w: int,
        types: Sequence[int],
    ) -> DeepSeekV41Image: ...
    @property
    def n_vit_h(self, /) -> int: ...
    @property
    def n_vit_w(self, /) -> int: ...
    @property
    def patches(self, /) -> npt.NDArray[np.float32]: ...
    @property
    def start(self, /) -> int: ...
    @property
    def types(self, /) -> tuple[int, ...]: ...

@final
class DeepSeekV41Media:
    def __new__(
        cls,
        /,
        token_ids: Sequence[int],
        images: Sequence[DeepSeekV41Image],
        image_token_id: int = 129264,
    ) -> DeepSeekV41Media: ...
    @property
    def cache_key(self, /) -> tuple[int, ...]: ...
    @property
    def digest(self, /) -> str: ...
    @property
    def image_token_id(self, /) -> int: ...
    @property
    def images(self, /) -> tuple[DeepSeekV41Image, ...]: ...
    @property
    def token_ids(self, /) -> tuple[int, ...]: ...
    @property
    def token_spans(self, /) -> tuple[MediaTokenSpan, ...]: ...

@final
class DeepSeekV41Recipe(_NativeRecipe):
    def __new__(
        cls,
        /,
        tokenizer: Any,
        *,
        options: RenderOptions | None = None,
        vision_config: Any | None = None,
        image_token_id: int | None = None,
    ) -> DeepSeekV41Recipe: ...

@final
class DeepSeekV4Recipe(_NativeRecipe):
    def __new__(
        cls,
        /,
        tokenizer: Any,
        *,
        options: RenderOptions | None = None,
        media_config: Any | None = None,
    ) -> DeepSeekV4Recipe: ...

@final
class Engine:
    def __enter__(self, /) -> Engine: ...
    def __exit__(self, /, _exc_type: Any, _exc_value: Any, _traceback: Any) -> bool: ...
    def __new__(
        cls,
        /,
        runner: Runner,
        *,
        max_running_requests: int = 32,
        context_length: int = 32768,
        scheduler_config: SchedulerConfig | None = None,
        prefix_cache: bool = False,
        enable_overlap: bool = True,
        enable_metrics: bool = True,
    ) -> Engine: ...
    def __repr__(self, /) -> str: ...
    def _metrics_prometheus_protobuf(
        self,
        /,
        *,
        model_name: str,
        rank: int = 0,
        attn_dp_rank: int = 0,
        include_runner: bool = True,
    ) -> bytes | None:
        """
        Native serving export; avoids constructing Python inspector dictionaries.
        """
    @property
    def active_request_count(self, /) -> int: ...
    @property
    def cache_free_parent_count(self, /) -> int | None: ...
    @property
    def cache_layout(self, /) -> CacheLayout | None: ...
    @property
    def cache_occupied_block_count(self, /) -> int | None: ...
    @property
    def cache_parent_capacity(self, /) -> int | None: ...
    @property
    def cache_pool_id(self, /) -> int | None: ...
    def cancel(self, /, request: RequestLease) -> bool: ...
    def clear_prefix_cache(self, /) -> None:
        """
        Drop all reusable cache prefixes.
        """
    def close(self, /) -> None: ...
    @property
    def closed(self, /) -> bool: ...
    @property
    def context_length(self, /) -> int: ...
    def has_work(self, /) -> bool: ...
    @property
    def max_running_requests(self, /) -> int: ...
    @property
    def metrics_enabled(self, /) -> bool: ...
    def metrics_snapshot(self, /) -> EngineMetrics | None:
        """
        Collect a read-only native snapshot of engine and runner observations.
        Call periodically: cache inventory requires a scan of allocated blocks.
        """
    @property
    def new_token_ratio(self, /) -> float: ...
    def next_phase(self, /) -> BatchPhase | None:
        """
        Read-only admission-aware proposal; reserves no requests or resources.
        """
    @property
    def overlap(self, /) -> bool: ...
    @property
    def overlap_pending(self, /) -> bool: ...
    @property
    def overlap_ready(self, /) -> bool: ...
    @property
    def overlap_stats(self, /) -> tuple[int, int]: ...
    def poison(self, /, reason: str) -> None:
        """
        Contain a failure reported by a peer without releasing local resources.
        """
    @property
    def poison_error(self, /) -> str | None: ...
    @property
    def poisoned(self, /) -> bool: ...
    @property
    def prefix_cache_stats(self, /) -> tuple[int, int, int, int] | None:
        """
        `(entries, hits, misses, evictions)` when prefix caching is enabled.
        """
    @property
    def ready_request_count(self, /) -> int: ...
    @property
    def scheduler_config(self, /) -> SchedulerConfig: ...
    def step(
        self, /, *, phase: BatchPhase | None = None, allow_overlap: bool = True
    ) -> list[TokenEvent]: ...
    def submit(
        self,
        /,
        token_ids: Sequence[int] | None = None,
        *,
        priority: int = 0,
        batch_key: int | None = None,
        sampling_params: SamplingParams | None = None,
        media: MediaInput | None = None,
        prompt_target_logprobs: Sequence[Sequence[int]] | None = None,
    ) -> RequestLease: ...

@final
class EngineMetrics:
    """
    An immutable engine snapshot with request, cache, and runner observations.
    Request and cache properties return independent dictionary views.
    """
    @property
    def cache(self, /) -> dict[str, Any] | None: ...
    @property
    def requests(self, /) -> dict[str, Any]: ...
    @property
    def runner(self, /) -> RunnerMetrics | None: ...

@final
class ForwardOutput:
    """
    Borrowed NumPy SoA output, valid until the synchronous step returns.

    All arrays must have their exact declared dtype, one dimension, and contiguous
    storage. Construction retains the arrays without copying. Do not mutate or
    recycle their storage until step() returns. Token rows use token_indptr;
    score rows use logprob_indptr (defaults to token_indptr). An empty score row
    opts out; every other score row must cover the entire raw output token row.
    top_indptr partitions top_token_ids/top_logprobs by scored token. All three
    top_* arrays are optional together.
    """
    def __new__(
        cls,
        /,
        *,
        token_ids: npt.NDArray[np.uint32],
        token_indptr: npt.NDArray[np.uint32],
        logprobs: npt.NDArray[np.float32] | None = None,
        logprob_indptr: npt.NDArray[np.uint32] | None = None,
        top_indptr: npt.NDArray[np.uint32] | None = None,
        top_token_ids: npt.NDArray[np.uint32] | None = None,
        top_logprobs: npt.NDArray[np.float32] | None = None,
        prompt_logprobs: npt.NDArray[np.float32] | None = None,
        prompt_logprob_indptr: npt.NDArray[np.uint32] | None = None,
        prompt_top_indptr: npt.NDArray[np.uint32] | None = None,
        prompt_top_token_ids: npt.NDArray[np.uint32] | None = None,
        prompt_top_logprobs: npt.NDArray[np.float32] | None = None,
        prompt_target_indptr: npt.NDArray[np.uint32] | None = None,
        prompt_target_token_ids: npt.NDArray[np.uint32] | None = None,
        prompt_target_logprobs: npt.NDArray[np.float32] | None = None,
    ) -> ForwardOutput: ...
    @property
    def logprob_indptr(self, /) -> npt.NDArray[np.uint32] | None: ...
    @property
    def logprobs(self, /) -> npt.NDArray[np.float32] | None: ...
    @property
    def prompt_logprob_indptr(self, /) -> npt.NDArray[np.uint32] | None: ...
    @property
    def prompt_logprobs(self, /) -> npt.NDArray[np.float32] | None: ...
    @property
    def prompt_target_indptr(self, /) -> npt.NDArray[np.uint32] | None: ...
    @property
    def prompt_target_logprobs(self, /) -> npt.NDArray[np.float32] | None: ...
    @property
    def prompt_target_token_ids(self, /) -> npt.NDArray[np.uint32] | None: ...
    @property
    def prompt_top_indptr(self, /) -> npt.NDArray[np.uint32] | None: ...
    @property
    def prompt_top_logprobs(self, /) -> npt.NDArray[np.float32] | None: ...
    @property
    def prompt_top_token_ids(self, /) -> npt.NDArray[np.uint32] | None: ...
    @property
    def token_ids(self, /) -> npt.NDArray[np.uint32]: ...
    @property
    def token_indptr(self, /) -> npt.NDArray[np.uint32]: ...
    @property
    def top_indptr(self, /) -> npt.NDArray[np.uint32] | None: ...
    @property
    def top_logprobs(self, /) -> npt.NDArray[np.float32] | None: ...
    @property
    def top_token_ids(self, /) -> npt.NDArray[np.uint32] | None: ...

@final
class Gemma4Media:
    def __new__(
        cls,
        /,
        token_ids: Sequence[int],
        mm_token_type_ids: Sequence[int],
        pixel_values: Any | None = None,
        image_position_ids: Any | None = None,
        pixel_values_videos: Any | None = None,
        video_position_ids: Any | None = None,
    ) -> Gemma4Media: ...
    @property
    def block_ids(self, /) -> tuple[int, ...]: ...
    @property
    def cache_key(self, /) -> tuple[int, ...]: ...
    @property
    def digest(self, /) -> str: ...
    @property
    def image_position_ids(self, /) -> npt.NDArray[np.int64] | None: ...
    @property
    def image_positions(self, /) -> tuple[int, ...]: ...
    @property
    def mm_token_type_ids(self, /) -> tuple[int, ...]: ...
    @property
    def pixel_values(self, /) -> npt.NDArray[np.float32] | None: ...
    @property
    def pixel_values_videos(self, /) -> npt.NDArray[np.float32] | None: ...
    @property
    def token_ids(self, /) -> tuple[int, ...]: ...
    @property
    def token_spans(self, /) -> tuple[MediaTokenSpan, ...]: ...
    def validate_placeholder_tokens(
        self, /, *, image_token_id: int, video_token_id: int
    ) -> None: ...
    @property
    def video_position_ids(self, /) -> npt.NDArray[np.int64] | None: ...
    @property
    def video_positions(self, /) -> tuple[int, ...]: ...

@final
class Gemma4Recipe(_NativeRecipe):
    def __new__(
        cls,
        /,
        tokenizer: Any,
        *,
        options: RenderOptions | None = None,
        eos_token_ids: Sequence[int] = ...,
        media_config: Any | None = None,
    ) -> Gemma4Recipe: ...

@final
class GptOssRecipe(_NativeRecipe):
    def __new__(
        cls,
        /,
        tokenizer: Any,
        *,
        options: RenderOptions | None = None,
        media_config: Any | None = None,
    ) -> GptOssRecipe: ...

@final
class Image:
    def __new__(cls, /, source: Any) -> Image: ...
    @staticmethod
    def from_bytes(data: bytes, *, media_type: str) -> Image: ...
    @staticmethod
    def from_rgb(array: npt.NDArray[np.float32]) -> Image: ...
    @property
    def source(self, /) -> Any: ...

@final
class InklingMedia:
    def __new__(
        cls,
        /,
        token_ids: Sequence[int],
        image_token_id: int,
        pixel_values: Any | None = None,
    ) -> InklingMedia: ...
    @property
    def cache_key(self, /) -> tuple[int, ...]: ...
    @property
    def digest(self, /) -> str: ...
    @property
    def image_positions(self, /) -> tuple[int, ...]: ...
    @property
    def image_token_id(self, /) -> int: ...
    @property
    def pixel_values(self, /) -> npt.NDArray[np.float32] | None: ...
    @property
    def token_ids(self, /) -> tuple[int, ...]: ...
    @property
    def token_spans(self, /) -> tuple[MediaTokenSpan, ...]: ...

@final
class InklingRecipe(_NativeRecipe):
    def __new__(
        cls,
        /,
        tokenizer: Any,
        *,
        options: RenderOptions | None = None,
        media_config: Any | None = None,
    ) -> InklingRecipe: ...

@final
class InklingResponseParser:
    """
    Parse Inkling response blocks into the events accepted by HttpServer.
    """
    def __new__(
        cls,
        /,
        tokenizer: RecipeTokenizer,
        token_ids: Mapping[str, int],
        *,
        prompt_ids: Sequence[int] = ...,
    ) -> InklingResponseParser: ...
    def feed(self, /, token_ids: Sequence[int]) -> list[ParserEvent]: ...
    def finish(self, /, reason: Literal["stop", "length"]) -> list[ParserEvent]: ...

@final
class Lfm2Recipe(_NativeRecipe):
    def __new__(
        cls,
        /,
        tokenizer: Any,
        *,
        options: RenderOptions | None = None,
        media_config: Any | None = None,
    ) -> Lfm2Recipe: ...

@final
class Message:
    def __new__(
        cls,
        /,
        role: Role,
        content: Sequence[ContentBlock],
        *,
        tools: Sequence[Tool] = ...,
        task: str | None = None,
        continuation: bool = False,
        response_format: Any | None = None,
    ) -> Message: ...
    @property
    def content(self, /) -> tuple[ContentBlock, ...]: ...
    @property
    def continuation(self, /) -> bool: ...
    @property
    def response_format(self, /) -> Any: ...
    @property
    def role(self, /) -> Role: ...
    @property
    def task(self, /) -> str | None: ...
    @property
    def tools(self, /) -> list[Tool]: ...

@final
class Parallelism:
    """
    Attention and expert meshes over the same GPU ranks.

    Attention uses [ATTN_DP, CP, REPLICATION, ATTN_TP]; unassigned attention
    ranks replicate the same request. Experts use [EP, EXPERT_TP], where
    EXPERT_TP is world_size // ep. CP shards prefill and replicates decode.
    """
    def __eq__(self, value: object, /) -> bool: ...
    def __getnewargs_ex__(self, /) -> "tuple[tuple[()], dict[str, int]]": ...
    def __hash__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __new__(
        cls,
        /,
        *,
        world_size: "int" = 1,
        attn_dp: "int" = 1,
        attn_tp: "int" = 1,
        cp: "int" = 1,
        ep: "int" = 1,
    ) -> Parallelism: ...
    def __repr__(self, /) -> str: ...
    @property
    def attn_dp(self, /) -> int: ...
    @property
    def attn_replication(self, /) -> int: ...
    @property
    def attn_tp(self, /) -> int: ...
    def coordinates(self, /, rank: "int") -> tuple[int, int, int, int]: ...
    @property
    def cp(self, /) -> int: ...
    @property
    def ep(self, /) -> int: ...
    @property
    def expert_tp(self, /) -> int: ...
    def partitions(self, /) -> "dict[str, tuple[tuple[int, ...], ...]]": ...
    def rank(
        self, /, attn_dp: "int", cp: "int", replication: "int", attn_tp: "int"
    ) -> int: ...
    @property
    def world_size(self, /) -> int: ...

class ParserEvent:
    """
    A semantic parser event with token-to-text byte provenance.

    Empty token_spans identify synthetic text or unknown provenance.
    """
    def __eq__(self, value: object, /) -> bool: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...
    @property
    def token_spans(self, /) -> tuple[TokenSpan, ...]: ...
    @property
    def type(self, /) -> str: ...

@final
class Qwen35Media:
    def __new__(
        cls,
        /,
        token_ids: Sequence[int],
        image_positions: Sequence[int],
        pixel_values: Any,
        image_grid_thw: Any,
        spatial_merge_size: int,
        image_token_id: int,
    ) -> Qwen35Media: ...
    @property
    def cache_key(self, /) -> tuple[int, ...]: ...
    @property
    def digest(self, /) -> str: ...
    @property
    def image_grid_thw(self, /) -> npt.NDArray[np.int64]: ...
    @property
    def image_positions(self, /) -> tuple[int, ...]: ...
    @property
    def image_token_id(self, /) -> int: ...
    @property
    def pixel_values(self, /) -> npt.NDArray[np.float32]: ...
    @property
    def position_ids(self, /) -> npt.NDArray[np.int64]: ...
    @property
    def rope_delta(self, /) -> int: ...
    @property
    def spatial_merge_size(self, /) -> int: ...
    @property
    def token_ids(self, /) -> tuple[int, ...]: ...
    @property
    def token_spans(self, /) -> tuple[MediaTokenSpan, ...]: ...

@final
class Qwen35Recipe(_NativeRecipe):
    def __new__(
        cls,
        /,
        tokenizer: Any,
        *,
        options: RenderOptions | None = None,
        media_config: Any | None = None,
    ) -> Qwen35Recipe: ...

@final
class Qwen38Recipe(_NativeRecipe):
    def __new__(
        cls,
        /,
        tokenizer: Any,
        *,
        options: RenderOptions | None = None,
        media_config: Any | None = None,
    ) -> Qwen38Recipe: ...

@final
class ReasoningDelta(ParserEvent):
    def __new__(
        cls,
        /,
        delta: str,
        *,
        token_spans: Sequence[tuple[int, int, int, int, int]] = ...,
    ) -> ReasoningDelta: ...
    @property
    def delta(self, /) -> str: ...

@final
class RecipeResponseParser:
    """
    Common native handle recognized directly by the HTTP actor.
    """
    def feed(self, /, token_ids: Sequence[int]) -> list[ParserEvent]: ...
    def finish(self, /, reason: Literal["stop", "length"]) -> list[ParserEvent]: ...

@final
class RecipeTokenizer:
    """
    A tokenizer prepared once for a model's response parsers.

    Fast Hugging Face tokenizers are snapshotted into Rust. Custom Python
    tokenizers remain supported through their decode method.
    """
    def __new__(cls, /, tokenizer: Any) -> RecipeTokenizer: ...
    @property
    def is_native(self, /) -> bool: ...

@final
class RenderOptions:
    """
    Explicit rendering controls; absent fields inherit recipe defaults.
    """
    def __new__(
        cls,
        /,
        *,
        reasoning_effort: str | int | float | None = None,
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
        add_generation_prompt: bool | None = None,
        add_vision_id: bool | None = None,
        use_system_prompt: bool | None = None,
        model_identity: str | None = None,
        knowledge_cutoff: str | None = None,
        conversation_start_date: str | None = None,
        auto_drop_analysis: bool | None = None,
        thinking_mode: str | None = None,
        response_format: Any | None = None,
    ) -> RenderOptions: ...
    def __repr__(self, /) -> str: ...
    @property
    def add_generation_prompt(self, /) -> bool | None: ...
    @property
    def add_vision_id(self, /) -> bool | None: ...
    @property
    def auto_drop_analysis(self, /) -> bool | None: ...
    @property
    def conversation_start_date(self, /) -> str | None: ...
    @property
    def enable_thinking(self, /) -> bool | None: ...
    @property
    def knowledge_cutoff(self, /) -> str | None: ...
    @property
    def model_identity(self, /) -> str | None: ...
    @property
    def preserve_thinking(self, /) -> bool | None: ...
    @property
    def reasoning_effort(self, /) -> str | int | float | None: ...
    @property
    def response_format(self, /) -> Any: ...
    @property
    def thinking_mode(self, /) -> str | None: ...
    @property
    def use_system_prompt(self, /) -> bool | None: ...

@final
class RenderedPrompt:
    """
    Rendered IDs and optional prepared media. Native parser state contains no pixels.
    """
    def __new__(
        cls,
        /,
        token_ids: Sequence[int],
        *,
        stop_token_ids: Sequence[int] = ...,
        media: MediaInput | None = None,
    ) -> RenderedPrompt: ...
    @property
    def media(self, /) -> MediaInput | None: ...
    @property
    def stop_token_ids(self, /) -> list[int]: ...
    @property
    def token_ids(self, /) -> list[int]: ...

@final
class RequestLease:
    """
    A caller-owned lifetime handle for one submitted request.

    The Engine keeps the shared state only while the request is active. Once
    the request retires, its generated tokens are moved into this lease. If
    the caller drops the lease without taking the result, the token buffer is
    released when the last internal reference disappears.
    """
    def __eq__(self, other: object, /) -> bool: ...
    def __index__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __repr__(self, /) -> str: ...
    @property
    def cached_tokens(self, /) -> int:
        """
        Number of prompt tokens restored from the prefix cache at admission.
        """
    @property
    def cancelled(self, /) -> bool: ...
    @property
    def done(self, /) -> bool: ...
    @property
    def id(self, /) -> int: ...
    @property
    def prompt_logprobs(self, /) -> list[TokenLogprobs | None] | None:
        """
        Raw input-token logprobs; the first token has no predecessor and is None.
        """
    @property
    def request_id(self, /) -> int: ...
    def take_result(self, /) -> list[int]:
        """
        Consume the completed token buffer exactly once.
        """

@final
class ResourceBinding:
    """
    Concrete Engine-owned resource identities passed to `bind_resources`.
    """
    def __repr__(self, /) -> str: ...
    @property
    def cache_layout(self, /) -> CacheLayout | None: ...
    @property
    def cache_pool_id(self, /) -> int | None: ...

@final
class Role:
    """
    Conversation role. This enum is extensible: new roles may be added.
    Code matching roles should include a fallback for unsupported roles.
    Some roles are model-specific; recipes validate supported roles.
    """

    ASSISTANT: Final[Role]
    DEVELOPER: Final[Role]
    LATEST_REMINDER: Final[Role]
    SYSTEM: Final[Role]
    TOOL: Final[Role]
    USER: Final[Role]
    def __eq__(self, value: object, /) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class RunnerConfig:
    """
    Frozen scheduling limits and persistent-resource declarations for a runner.

    Runners without special requirements may omit `runner_config` entirely.
    """
    def __new__(
        cls,
        /,
        *,
        max_batch_tokens: int | None = None,
        prefill_chunk_size: int | None = None,
        continuation_prefill: bool = False,
        cache_write_lookahead_tokens: int = 0,
        cache_layout: CacheLayout | None = None,
    ) -> RunnerConfig: ...
    def __repr__(self, /) -> str: ...
    @property
    def cache_layout(self, /) -> CacheLayout | None: ...
    @property
    def cache_write_lookahead_tokens(self, /) -> int: ...
    @property
    def continuation_prefill(self, /) -> bool: ...
    @property
    def max_batch_tokens(self, /) -> int | None: ...
    @property
    def prefill_chunk_size(self, /) -> int | None: ...

@final
class RunnerMetrics:
    """
    A nonblocking snapshot of cumulative observations for one physical rank.

    Forward categories are exclusive: prefill, decode, target_verify, draft,
    and idle. Graph counts describe executions, not regions; DP wait uses a
    bounded exponential histogram. The core snapshot is stored directly.
    """
    def __eq__(self, value: object, /) -> bool: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __new__(
        cls,
        /,
        *,
        forward_execution_seconds: dict[str, float] | None = None,
        cuda_graph_passes: dict[str, int] | None = None,
        spec_draft_tokens: int = 0,
        spec_accepted_tokens: int = 0,
        spec_verify_rows: int = 0,
        dp_attention_sync_wait_seconds: dict[str, Any] | None = None,
    ) -> RunnerMetrics: ...
    def as_dict(self, /) -> dict[str, Any]:
        """
        Plain fields for inspection and explicit distributed transport.
        """
    @property
    def cuda_graph_passes(self, /) -> dict[str, int]: ...
    @property
    def dp_attention_sync_wait_seconds(self, /) -> dict[str, Any] | None: ...
    @property
    def forward_execution_seconds(self, /) -> dict[str, float]: ...
    @property
    def spec_accepted_tokens(self, /) -> int: ...
    @property
    def spec_draft_tokens(self, /) -> int: ...
    @property
    def spec_verify_rows(self, /) -> int: ...

@final
class SamplingParams:
    """
    Per-request generation policy, including sampling, length, and stop tokens.

    A zero temperature selects the maximum-logit token. Positive temperatures
    enable stochastic sampling after optional top-k and top-p filtering.
    An omitted `max_output_tokens` uses the remaining context budget. Stop tokens
    are explicit: pass a rendered prompt's `stop_token_ids` for recipe stopping.
    """
    def __new__(
        cls,
        /,
        *,
        max_output_tokens: int | None = None,
        stop_token_ids: Sequence[int] = ...,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        prompt_logprobs: bool = False,
        prompt_top_logprobs: int = 0,
        extra: Any | None = None,
    ) -> SamplingParams: ...
    def __repr__(self, /) -> str: ...
    @property
    def extra(self, /) -> Any | None: ...
    @property
    def logprobs(self, /) -> bool: ...
    @property
    def max_output_tokens(self, /) -> int | None: ...
    @property
    def prompt_logprobs(self, /) -> bool: ...
    @property
    def prompt_top_logprobs(self, /) -> int: ...
    @property
    def seed(self, /) -> int | None: ...
    @property
    def stop_token_ids(self, /) -> tuple[int, ...]: ...
    @property
    def temperature(self, /) -> float: ...
    @property
    def top_k(self, /) -> int | None: ...
    @property
    def top_logprobs(self, /) -> int: ...
    @property
    def top_p(self, /) -> float: ...

@final
class SchedulerConfig:
    """
    Native admission and token-budget policy for an Engine.
    """
    def __new__(
        cls,
        /,
        *,
        max_batch_tokens: int | None = None,
        prefill_chunk_size: int = 8192,
        schedule_conservativeness: float = 1.0,
    ) -> SchedulerConfig: ...
    def __repr__(self, /) -> str: ...
    @property
    def max_batch_tokens(self, /) -> int | None: ...
    @property
    def prefill_chunk_size(self, /) -> int: ...
    @property
    def schedule_conservativeness(self, /) -> float: ...

@final
class Text:
    def __new__(cls, /, text: str) -> Text: ...
    @property
    def text(self, /) -> str: ...

@final
class TextDelta(ParserEvent):
    def __new__(
        cls,
        /,
        delta: str,
        *,
        token_spans: Sequence[tuple[int, int, int, int, int]] = ...,
    ) -> TextDelta: ...
    @property
    def delta(self, /) -> str: ...

@final
class Thinking:
    def __new__(cls, /, text: str) -> Thinking: ...
    @property
    def text(self, /) -> str: ...

@final
class TokenEvent:
    def __repr__(self, /) -> str: ...
    @property
    def finish_reason(self, /) -> str | None: ...
    @property
    def finished(self, /) -> bool: ...
    @property
    def index(self, /) -> int: ...
    @property
    def logprobs(self, /) -> TokenLogprobs | None: ...
    @property
    def request_id(self, /) -> int: ...
    @property
    def token_id(self, /) -> int: ...

@final
class TokenLogprobs:
    def __new__(
        cls,
        /,
        logprob: float,
        top_logprobs: Sequence[tuple[int, float]] = ...,
        target_logprobs: Sequence[tuple[int, float]] = ...,
    ) -> TokenLogprobs: ...
    @property
    def logprob(self, /) -> float: ...
    @property
    def target_logprobs(self, /) -> list[tuple[int, float]]: ...
    @property
    def top_logprobs(self, /) -> list[tuple[int, float]]: ...

@final
class Tool:
    def __new__(
        cls,
        /,
        name: str,
        *,
        description: str | None = None,
        parameters: Any | None = None,
        response: Any | None = None,
        namespace: Any | None = None,
    ) -> Tool: ...
    @property
    def description(self, /) -> str | None: ...
    @property
    def name(self, /) -> str: ...
    @property
    def namespace(self, /) -> Any: ...
    @property
    def parameters(self, /) -> Any: ...
    @property
    def response(self, /) -> Any: ...

@final
class ToolCall:
    def __new__(
        cls,
        /,
        name: str,
        arguments: Any,
        *,
        id: str | None = None,
        namespace: Any | None = None,
    ) -> ToolCall: ...
    @property
    def arguments(self, /) -> Any: ...
    @property
    def id(self, /) -> str | None: ...
    @property
    def name(self, /) -> str: ...
    @property
    def namespace(self, /) -> Any: ...

@final
class ToolCallArgumentsDelta(ParserEvent):
    def __new__(
        cls,
        /,
        index: int,
        delta: str,
        *,
        token_spans: Sequence[tuple[int, int, int, int, int]] = ...,
    ) -> ToolCallArgumentsDelta: ...
    @property
    def delta(self, /) -> str: ...
    @property
    def index(self, /) -> int: ...

@final
class ToolCallStart(ParserEvent):
    def __new__(
        cls,
        /,
        index: int,
        id: str,
        name: str,
        *,
        token_spans: Sequence[tuple[int, int, int, int, int]] = ...,
    ) -> ToolCallStart: ...
    @property
    def id(self, /) -> str: ...
    @property
    def index(self, /) -> int: ...
    @property
    def name(self, /) -> str: ...

@final
class ToolResult:
    def __new__(
        cls, /, content: Any, *, call_id: str | None = None, name: str | None = None
    ) -> ToolResult: ...
    @property
    def call_id(self, /) -> str | None: ...
    @property
    def content(self, /) -> Any: ...
    @property
    def name(self, /) -> str | None: ...

@final
class Video:
    def __new__(cls, /, source: Any) -> Video: ...
    @property
    def source(self, /) -> Any: ...

@final
class _HistogramAccumulator:
    """
    Native accumulator; snapshots are plain Python dictionaries.
    """
    def __new__(cls, /) -> _HistogramAccumulator: ...
    def attach(self, /, snapshot: RunnerMetrics) -> RunnerMetrics:
        """
        Return a snapshot with this coordinator's wait observations included.
        """
    def observe(self, /, value: float, *, weight: int = 1) -> None: ...
    def snapshot(self, /) -> dict[str, Any]: ...

@final
class _HttpEvents:
    def __new__(cls, /) -> _HttpEvents: ...
    @staticmethod
    def combine(packets: Sequence[_HttpEvents]) -> _HttpEvents: ...
    @staticmethod
    def from_bytes(data: bytes) -> _HttpEvents: ...
    def to_bytes(self, /) -> bytes: ...
    def with_metrics(self, /, snapshots: Sequence[Sequence[int]]) -> _HttpEvents: ...

@final
class _HttpIngress:
    """
    Native ingress stays typed across the Python coordinator. Only transport
    between ranks encodes it; JSON never becomes a Python object graph.
    """
    def __new__(cls, /, *, shutdown: bool = False) -> _HttpIngress: ...
    @property
    def collect_metrics(self, /) -> bool: ...
    @property
    def error(self, /) -> str | None: ...
    @staticmethod
    def from_bytes(data: bytes) -> _HttpIngress: ...
    @property
    def shutdown(self, /) -> bool: ...
    def to_bytes(self, /) -> bytes: ...

@final
class _HttpServer:
    def __new__(
        cls,
        /,
        engine: Engine,
        recipe: Recipe,
        *,
        model: str,
        stream_interval: int = 1,
        request_queue_capacity: int = 1024,
        enable_openai: bool = True,
        enable_anthropic: bool = True,
        enable_demo: bool = True,
        distributed: Any | None = None,
    ) -> _HttpServer: ...
    def __repr__(self, /) -> str: ...
    @property
    def model(self, /) -> str: ...
    @property
    def running(self, /) -> bool: ...
    def serve(self, /, *, host: str = "127.0.0.1", port: int = 8000) -> None: ...
    def shutdown(self, /) -> None:
        """
        Request shutdown from another thread; serve returns after the current
        inference step has finished and its requests have been cancelled.
        """

@final
class _HttpSession:
    """
    The same native lifecycle used by the ordinary actor, attached to one DP driver.
    """
    def __new__(
        cls,
        /,
        engine: Engine,
        recipe: Recipe,
        *,
        stream_interval: int = 1,
        emit_first_token: bool = False,
    ) -> _HttpSession: ...
    @property
    def active_count(self, /) -> int: ...
    def admit(self, /, ingress: _HttpIngress, *, rank: int = 0) -> None: ...
    def cancel(self, /, ids: Sequence[str]) -> None: ...
    def close(self, /) -> None: ...
    def process(self, /, events: Sequence[TokenEvent]) -> _HttpEvents: ...

class _NativeRecipe:
    """
    Shared native implementation inherited by the concrete model recipes.
    """
    def create_parser(self, /, prompt: RenderedPrompt) -> RecipeResponseParser: ...
    def get_stop_token_ids(self, /) -> list[int]: ...
    def parse_response(
        self, /, token_ids: Sequence[int], *, prompt: RenderedPrompt
    ) -> Any: ...
    def render(
        self,
        /,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] = ...,
        options: RenderOptions | None = None,
    ) -> RenderedPrompt: ...
    def render_ids(
        self,
        /,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] = ...,
        options: RenderOptions | None = None,
    ) -> list[int]: ...
    def render_text(
        self,
        /,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] = ...,
        options: RenderOptions | None = None,
    ) -> str: ...
    @property
    def tokenizer(self, /) -> Any: ...

def _media_from_buffers(
    metadata: bytes, buffers: Sequence[npt.NDArray[Any]]
) -> MediaInput: ...
def _media_to_buffers(
    media: MediaInput,
) -> tuple[bytes, tuple[npt.NDArray[Any], ...]]: ...
def _runner_metrics_prometheus_protobuf(
    snapshot: RunnerMetrics, *, model_name: str, rank: int, attn_dp_rank: int
) -> bytes:
    """
    Encode physical-rank observations separately from logical request ownership.
    """

def image_token_types(height: int, width: int) -> tuple[int, ...]: ...
