"""BF16 Gemma 4 E2B/E4B on Ampere with Engine-owned paged shared KV."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from resin import BatchPhase
from resin.backend.cuda.completion import (
    CudaContinuation,
    DeviceAcceptance,
    DeviceCompletion,
)
from resin.backend.cuda.failures import GraphInputError
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.runner import pad_sampling_inputs
from resin.backend.cuda.tensor_graphs import CudaGraphTensorRegion, TensorSpec
from resin.backend.sampling import SAMPLING_OFFSET_STRIDE
from resin.models.gemma4.config import SLIDING_ATTENTION
from resin.models.gemma4.plan import (
    PAGE_SIZE,
    expand_gemma4_decode_batch,
    gemma4_assistant_context,
)
from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner
from resin_models_cuda.gemma4.edge_checkpoint import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    load_edge_model,
)
from resin_models_cuda.gemma4.edge_kernels import norm_residual
from resin_models_cuda.scoring import score_completion


class _EdgeHiddenLease:
    """A batched buffer retained until producer and all consumers finish."""

    __slots__ = ("pool", "tensor", "event")

    def __init__(self, pool, tensor):
        self.pool, self.tensor, self.event = pool, tensor, None

    def record_use(self):
        self.event = None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.tensor.device))
        self.event = event

    def __del__(self):
        # Finalization does not call CUDA. None quarantines uncertain work.
        self.pool.pending.append((self.tensor, self.event))


class _EdgeHiddenPool:
    def __init__(self, device, width):
        self.device, self.width = device, width
        self.pending = []
        self.allocations = 0

    def acquire(self, rows):
        for index, (tensor, event) in enumerate(self.pending):
            if tensor.shape[0] == rows and event is not None and event.query():
                self.pending.pop(index)
                return _EdgeHiddenLease(self, tensor)
        with torch.inference_mode(False):
            tensor = torch.empty(
                rows, self.width, dtype=torch.bfloat16, device=self.device
            )
        self.allocations += 1
        return _EdgeHiddenLease(self, tensor)


class _EdgeDeviceFeedback:
    """Owned accepted inputs plus target-hidden state for one continuation."""

    __slots__ = ("_base", "hidden_lease", "continuation")

    def __init__(self, base, hidden_lease, continuation):
        self._base = base
        self.hidden_lease = hidden_lease
        self.continuation = continuation

    @property
    def sequence_ids(self):
        return self.continuation.sequence_ids

    @property
    def shape(self):
        return self.continuation.shape

    @property
    def target_hidden(self):
        return None if self.hidden_lease is None else self.hidden_lease.tensor

    @property
    def token_ids(self):
        return self._base.token_ids

    @property
    def positions(self):
        return self._base.positions

    def restrict(self, rows):
        self._base.restrict(rows)
        if self.target_hidden is not None:
            self.target_hidden[rows:].zero_()
        return self

    def inputs(self):
        # Gemma captures flat token IDs. Exact stochastic offsets and all
        # fractional cache metadata are derived from positions inside the graph.
        inputs = {"token_ids": self.token_ids, "positions": self.positions}
        if self.target_hidden is not None:
            inputs["target_hidden"] = self.target_hidden
        return inputs


class Gemma4EdgeCudaRunner(Gemma4CudaRunner):
    """Text-only original BF16 weights, BF16 KV, and full decode graphs."""

    max_num_seqs = 256
    assistant_geometry_fields = (
        "num_key_value_heads",
        "num_global_key_value_heads",
        "head_dim",
        "global_head_dim",
    )

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        compact_prefill: bool = True,
        cross_page_overlap: bool = True,
        **kwargs,
    ) -> None:
        self._continuation = CudaContinuation()
        self._edge_graph_inputs = {}
        self.cross_page_overlap = cross_page_overlap
        self.compact_prefill = compact_prefill
        self._compact_prefill_state = None
        groups = kwargs.get("parallel_groups")
        if groups is not None and groups.topology.world_size != 1:
            raise ValueError("Gemma 4 E2B currently supports one GPU")
        kwargs.setdefault("kv_cache_dtype", "bfloat16")
        if str(model) == DEFAULT_MODEL:
            kwargs.setdefault("revision", DEFAULT_REVISION)
        if (
            kwargs.get("speculative_draft_tokens", 0)
            and kwargs.get("assistant_model") is None
        ):
            if str(model) != DEFAULT_MODEL:
                raise ValueError(
                    "a local Gemma edge target requires an explicit assistant_model"
                )
            from resin_models_cuda.gemma4.edge_assistant import (
                DEFAULT_ASSISTANT,
                DEFAULT_ASSISTANT_REVISION,
            )

            kwargs["assistant_model"] = DEFAULT_ASSISTANT
            kwargs.setdefault("assistant_revision", DEFAULT_ASSISTANT_REVISION)
        super().__init__(model, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def _prepare_weight_backend(self, backend: str) -> str:
        capability = torch.cuda.get_device_capability(self.device)
        if capability < (8, 0):
            raise RuntimeError("Gemma 4 E2B BF16 requires Ampere or newer hardware")
        return "bf16"

    def _load_target(self, *args, **kwargs):
        return load_edge_model(*args, **kwargs)

    def _load_assistant(self, *args, **kwargs):
        from resin_models_cuda.gemma4.edge_assistant import load_edge_assistant

        return load_edge_assistant(*args, **kwargs)

    def _build_attention_wrappers(self):
        super()._build_attention_wrappers()
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        self._compact_prefill_wrappers = tuple(
            BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
            for workspace in self._workspaces
        )
        if self._assistant is not None:
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper

            self._assistant_decode_wrappers = tuple(
                BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
                for workspace in self._workspaces
            )

    def _plan_assistant_attention(self, batch, local, global_, custom_mask=None):
        assert self._assistant is not None
        for is_local, wrapper, plan in zip(
            (True, False),
            self._assistant_decode_wrappers,
            (local, global_),
            strict=True,
        ):
            wrapper.plan(
                plan.kv_indptr,
                plan.page_indices,
                plan.last_page_lengths,
                self._assistant.config.num_attention_heads,
                self.config.num_key_value_heads
                if is_local
                else self.config.num_global_key_value_heads,
                self.config.head_dim if is_local else self.config.global_head_dim,
                self.page_size,
                q_data_type=torch.bfloat16,
                kv_data_type=self._cache.dtype,
                o_data_type=torch.bfloat16,
                sm_scale=1.0,
                window_left=self.sliding_window - 1 if is_local else -1,
            )
        return self._assistant_decode_wrappers

    def _configure_feed_forward_graphs(self):
        hidden = TensorSpec((self.config.hidden_size,))
        per_layer = TensorSpec((self.config.hidden_size_per_layer_input,))
        regions = tuple(
            CudaGraphTensorRegion(
                str(index),
                layer.feed_forward,
                (hidden, per_layer),
                hidden,
            )
            for index, layer in enumerate(self._model.language_model.layers)
        )
        return self._configure_prefill_tensor_graphs(
            regions,
            max_tokens=self.prefill_chunk_size,
            mode="prefill",
            coordinated=True,
        )

    def _eager_transformer(self, native, batch):
        if (
            self.compact_prefill
            and native.phase == BatchPhase.PREFILL
            and not native.needs_prompt_logprobs
            and len(batch.token_ids) > len(batch.last_token_indices)
        ):
            self._compact_prefill_state = torch.as_tensor(
                batch.last_token_indices if any(native.sampling_mask) else [],
                dtype=torch.int64,
                device=self.device,
            )
        try:
            return super()._eager_transformer(native, batch)
        finally:
            self._compact_prefill_state = None

    def _plan_attention(self, batch, local, global_, custom_mask):
        wrappers = super()._plan_attention(batch, local, global_, custom_mask)
        if self._compact_prefill_state is not None and len(self._compact_prefill_state):
            qo_indptr = torch.arange(
                len(batch.last_token_indices) + 1, dtype=torch.int32, device=self.device
            )
            for is_local, wrapper, plan in zip(
                (True, False),
                self._compact_prefill_wrappers,
                (local, global_),
                strict=True,
            ):
                wrapper.plan(
                    qo_indptr,
                    plan.kv_indptr,
                    plan.page_indices,
                    plan.last_page_lengths,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads
                    if is_local
                    else self.config.num_global_key_value_heads,
                    self.config.head_dim if is_local else self.config.global_head_dim,
                    self.page_size,
                    causal=True,
                    q_data_type=torch.bfloat16,
                    kv_data_type=self._cache.dtype,
                    o_data_type=torch.bfloat16,
                    sm_scale=1.0,
                    window_left=self.sliding_window - 1 if is_local else -1,
                )
        return wrappers

    def _merge_media_embeddings(self, native, batch, hidden):
        if any(media is not None for media in getattr(native, "media", ())):
            raise ValueError("Gemma 4 E2B runner currently accepts text inputs only")
        return {}

    def _run_transformer(
        self,
        hidden,
        positions,
        local,
        global_,
        local_wrapper,
        global_wrapper,
        token_ids: Tensor | None = None,
    ) -> Tensor:
        if token_ids is None:
            raise ValueError("E2B per-layer embeddings require token IDs")
        per_layer_inputs = self._model.prepare_per_layer_inputs(hidden, token_ids)
        rotary = self._model.language_model.rotary_emb
        embeddings = {
            kind: rotary(positions, kind, hidden.dtype)
            for kind in set(self.config.layer_types)
        }
        original_tokens = len(hidden)
        final_rows = self._compact_prefill_state
        for index, layer in enumerate(self._model.language_model.layers):
            if final_rows is not None and index == self.config.num_kv_producer_layers:
                # Consumer layers only read producer KV. Their non-final queries
                # cannot influence any final query or future decode state.
                if not len(final_rows):
                    # Intermediate prompt chunks need only persistent producer
                    # KV; no consumer query contributes to an emitted token.
                    return hidden.new_zeros((original_tokens, self.config.hidden_size))
                hidden = hidden.index_select(0, final_rows)
                per_layer_inputs = per_layer_inputs.index_select(0, final_rows)
                embeddings = {
                    kind: (
                        cos.index_select(0, final_rows),
                        sin.index_select(0, final_rows),
                    )
                    for kind, (cos, sin) in embeddings.items()
                }
                local_wrapper, global_wrapper = self._compact_prefill_wrappers
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            cos, sin = embeddings[layer.self_attn.layer_type]
            query, key, value = layer.self_attn.project_qkv(normalized, cos, sin)
            cache_index = self.config.cache_layer_index(index)
            if layer.self_attn.layer_type == SLIDING_ATTENTION:
                keys = self._cache.local_keys[cache_index]
                values = self._cache.local_values[cache_index]
                pages, offsets, wrapper = (
                    local.append_pages,
                    local.append_offsets,
                    local_wrapper,
                )
            else:
                keys = self._cache.global_keys[cache_index]
                values = self._cache.global_values[cache_index]
                pages, offsets, wrapper = (
                    global_.append_pages,
                    global_.append_offsets,
                    global_wrapper,
                )
            if key is not None:
                keys[pages, offsets] = key.to(self._cache.dtype)
                values[pages, offsets] = value.to(self._cache.dtype)
            attended = wrapper.run(query, (keys, values))
            hidden = layer.self_attn.o_proj(attended.reshape(len(hidden), -1))
            hidden = norm_residual(
                hidden,
                layer.post_attention_layernorm.weight,
                residual,
                eps=layer.post_attention_layernorm.eps,
            )
            hidden = self._prefill_feed_forward[index](
                hidden, per_layer_inputs[:, index]
            )
        hidden = self._model.language_model.norm(hidden)
        if final_rows is not None:
            output = hidden.new_zeros((original_tokens, self.config.hidden_size))
            output.index_copy_(0, final_rows, hidden)
            return output
        return hidden

    @property
    def attention_plan_cache_stats(self):
        return {
            "hits": getattr(self, "_attention_plan_cache_hits", 0),
            "misses": getattr(self, "_attention_plan_cache_misses", 0),
        }

    @property
    def benchmark_counters(self):
        return {
            **{
                f"attention_plan_cache.{key}": value
                for key, value in self.attention_plan_cache_stats.items()
            },
            "attention_metadata.device_rebuilds": getattr(
                self, "_edge_device_metadata_rebuilds", 0
            ),
            "device_completion.enqueues": getattr(self, "_edge_async_enqueues", 0),
            "device_hidden.allocations": getattr(
                getattr(self, "_edge_hidden_pool", None), "allocations", 0
            ),
            "device_feedback.continuations": getattr(
                self, "_edge_async_continuations", 0
            ),
        }

    def _wide_graph(self, shape):
        return (
            getattr(self, "cross_page_overlap", False)
            and bool(self.speculative_draft_tokens)
            and shape[0] >= 32
        )

    def _new_assistant_graph_attention_wrapper(self, *args, **kwargs):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        rows = len(kwargs["paged_kv_last_page_len_buffer"])
        if self._wide_graph((rows, 0)):
            kwargs.update(use_tensor_cores=True, backend="fa2")
        return BatchDecodeWithPagedKVCacheWrapper(*args, **kwargs)

    def _prepare_graph_attention(self, shape, inputs):
        if self._wide_graph(shape):
            # Fixed Q1/Q4 no-split schedules are independent of KV CSR. Each
            # graph launch repacks the actual readable view from device data.
            return
        """Reuse only the last identical host schedule of each graph wrapper.

        FlashInfer 0.7.1's native decode scheduler depends on CSR page counts,
        not physical page IDs or the fractional last page. Verification also
        has fixed query indptr. All mutable data is still staged by the normal
        CudaGraphStep before replay. The plan-object identity invalidates this
        cache if another owner explicitly re-plans the same wrapper.
        """
        if self.speculative_draft_tokens:
            wrappers = self._graph_spec_attention[shape]
            stages = (
                ("assistant_", wrappers[:2], None),
                ("verify_", wrappers[2:], inputs["verify_qo_indptr"]),
            )
        else:
            stages = (("", self._graph_attention[shape], None),)

        def host_tuple(values):
            if torch.is_tensor(values) and values.is_cuda:
                raise ValueError("plan signatures require host CSR metadata")
            return tuple(int(value) for value in values)

        staging = None
        cuda_device = self.device.type == "cuda"

        def stage(name, values, *, device=False):
            nonlocal staging
            if not cuda_device:
                return torch.as_tensor(values, dtype=torch.int32, device="cpu")
            if staging is None:
                staging = self._acquire_attention_plan_staging()
            method = staging.device if device else staging.host
            return method(name, values, torch.int32)

        try:
            for prefix, wrappers, queries in stages:
                for local, wrapper in zip((True, False), wrappers, strict=True):
                    name = prefix + ("local" if local else "global")
                    indptr = inputs[f"{name}_kv_indptr"]
                    signature = (
                        host_tuple(indptr),
                        None if queries is None else host_tuple(queries),
                        bool(local),
                        prefix == "assistant_",
                    )
                    previous = getattr(wrapper, "_resin_edge_last_host_plan", None)
                    current_plan = getattr(wrapper, "_plan_info", None)
                    if (
                        previous is not None
                        and previous[0] == signature
                        and current_plan is not None
                        and previous[1] is current_plan
                    ):
                        self._attention_plan_cache_hits = (
                            getattr(self, "_attention_plan_cache_hits", 0) + 1
                        )
                        continue
                    self._attention_plan_cache_misses = (
                        getattr(self, "_attention_plan_cache_misses", 0) + 1
                    )
                    page_count = int(indptr[-1])
                    # A failing planner may already have mutated workspace while
                    # preserving _plan_info identity. Never retain an older hit.
                    wrapper._resin_edge_last_host_plan = None
                    # FlashInfer also reuses its own pinned integer workspace.
                    # Wait only for its previous planning copies, whose event
                    # precedes the previous model replay on this stream.
                    copies_done = getattr(wrapper, "_resin_edge_plan_copies_done", None)
                    if copies_done is not None and not copies_done.query():
                        copies_done.synchronize()
                    try:
                        self._plan_graph_attention(
                            wrapper,
                            stage(name + "_indptr", indptr),
                            stage(
                                name + "_indices",
                                inputs[f"{name}_page_indices"][:page_count],
                                device=True,
                            ),
                            stage(name + "_last", inputs[f"{name}_last_page_lengths"]),
                            local=local,
                            qo_indptr=None
                            if queries is None
                            else stage(name + "_queries", queries),
                            assistant=prefix == "assistant_",
                            non_blocking=cuda_device,
                        )
                    finally:
                        if cuda_device:
                            copies_done = torch.cuda.Event()
                            copies_done.record(torch.cuda.current_stream(self.device))
                            wrapper._resin_edge_plan_copies_done = copies_done
                    # Publish only after successful planning. A failure must not
                    # turn a later retry into a hit on uncertain workspace state.
                    wrapper._resin_edge_last_host_plan = (
                        signature,
                        getattr(wrapper, "_plan_info", None),
                    )

        finally:
            if staging is not None:
                # Keep this host generation alive until every asynchronous
                # planner copy has consumed it, including on a failed plan.
                event = None
                try:
                    recorded_event = torch.cuda.Event()
                    recorded_event.record(torch.cuda.current_stream(self.device))
                    event = recorded_event
                finally:
                    # If event creation/recording fails, retain this generation
                    # permanently rather than treating an unrecorded event as done.
                    self._attention_plan_staging_pending.append((event, staging))

    def _new_verify_attention_wrapper(self, *args, **kwargs):
        from resin_models_cuda.gemma4.edge_kernels import (
            UniformCausalPagedVerifyWrapper,
        )

        return UniformCausalPagedVerifyWrapper(
            *args,
            query_width=self.speculative_draft_tokens + 1,
            split_kv=not self._wide_graph(
                (len(kwargs["paged_kv_last_page_len_buf"]), 0)
            ),
            **kwargs,
        )

    def _graph_capture_inputs(self, shape, vocab_size=262208):
        inputs = super()._graph_capture_inputs(shape, vocab_size)
        if self._wide_graph(shape):
            for kind, page_width in (
                ("local", self.local_graph_page_capacity),
                ("global", shape[1]),
            ):
                inputs[f"envelope_{kind}_page_table"] = np.zeros(
                    (shape[0], page_width), dtype=np.int32
                )
                inputs[f"envelope_{kind}_first_pages"] = np.zeros(
                    shape[0], dtype=np.int64
                )
        return inputs

    def _new_speculative_graph_step(self, shape):
        step = super()._new_speculative_graph_step(shape)
        if self._wide_graph(shape):
            from resin_models_cuda.gemma4.edge_kernels import (
                update_reserved_paged_metadata,
            )

            inputs = cast(dict[str, Tensor], step.inputs)
            for kind, page_width in (
                ("local", self.local_graph_page_capacity),
                ("global", shape[1]),
            ):
                inputs[f"envelope_{kind}_page_table"] = torch.empty(
                    (shape[0], page_width),
                    dtype=torch.int32,
                    device=self.device,
                )
                inputs[f"envelope_{kind}_first_pages"] = torch.empty(
                    shape[0],
                    dtype=torch.int64,
                    device=self.device,
                )

            def launch_reserved():
                update_reserved_paged_metadata(
                    step.inputs,
                    query_width=self.speculative_draft_tokens + 1,
                    sliding_window=self.sliding_window,
                )
                step.launch()

            return replace(step, launch=launch_reserved)
        columns = torch.arange(
            self.speculative_draft_tokens + 1,
            dtype=torch.int64,
            device=self.device,
        )[None, :]

        def launch():
            # Engine continuation plans reserve the maximum previous output;
            # actual accepted positions remain device data. Page identities and
            # CSR counts are invariant under Engine's two-step block guard.
            positions = step.inputs["positions"]
            assistant_last = ((positions - 1) % PAGE_SIZE + 1).to(torch.int32)
            verify_last = (
                (positions + self.speculative_draft_tokens) % PAGE_SIZE + 1
            ).to(torch.int32)
            offsets = ((positions[:, None] + columns) % PAGE_SIZE).reshape(-1)
            for kind in ("local", "global"):
                step.inputs[f"assistant_{kind}_last_page_lengths"].copy_(assistant_last)
                step.inputs[f"verify_{kind}_last_page_lengths"].copy_(verify_last)
                step.inputs[f"verify_{kind}_append_offsets"].copy_(offsets)
            step.inputs["sampling_offsets"].copy_(
                (positions + 1) * SAMPLING_OFFSET_STRIDE
            )
            step.launch()

        return replace(step, launch=launch)

    def _new_decode_graph_step(self, shape):
        step = super()._new_decode_graph_step(shape)
        self._edge_graph_inputs[shape] = step.inputs
        return step

    def release(self, request_ids):
        self._continuation.drain()
        super().release(request_ids)

    def forward(self, plan):
        self._continuation.drain()
        return super().forward(plan)

    def _speculative_graph_forward(self, native, batch):
        if self.cross_page_overlap:
            completion = self._enqueue_graph(native, batch, None)
            if completion is not None:
                return completion.result()
        return super()._speculative_graph_forward(native, batch)

    def _enqueue_graph(self, native, batch, feedback):
        if not self._cuda_graph_mode_enabled("decode"):
            return None
        sequence_ids = tuple(map(int, native.sequence_ids))
        if (
            self.speculative_draft_tokens
            and feedback is None
            and any(sequence not in self._target_hidden for sequence in sequence_ids)
        ):
            self._cuda_graph_fallback("decode", "target_hidden")
            return None
        rows = native.num_sequences
        history_envelope = batch.global_envelope or batch.global_
        local_envelope = batch.local_envelope or batch.local
        history_width = int(np.diff(history_envelope.kv_indptr).max())
        local_width = int(np.diff(local_envelope.kv_indptr).max())
        if local_width > self.local_graph_page_capacity:
            raise GraphInputError("local page table exceeds CUDA graph shape")
        shape = self._select_cuda_graph(
            "decode",
            match=lambda value: value[0] >= rows and value[1] >= history_width,
        )
        if shape is None:
            return None
        graph_rows, graph_history_width = shape
        padding = graph_rows - rows
        width = self.speculative_draft_tokens + 1
        if feedback is not None:
            if not isinstance(feedback, _EdgeDeviceFeedback):
                raise GraphInputError("Gemma continuation requires owned Edge feedback")
            self._continuation.resume(sequence_ids, feedback.continuation, shape=shape)
            host = np.asarray(batch.positions)
            if not self._wide_graph(shape) and np.any(
                (host - width) // PAGE_SIZE != (host + width - 1) // PAGE_SIZE
            ):
                raise GraphInputError(
                    "Gemma continuation exceeds its guarded cache block"
                )
        inputs = {
            "token_ids": np.pad(batch.token_ids, (0, padding)),
            "positions": np.pad(batch.positions, (0, padding), constant_values=1),
        }
        if self.speculative_draft_tokens:
            context = gemma4_assistant_context(
                batch, local_window_tokens=self.sliding_window
            )
            fixed = self._speculation.validated_benchmark_accepted_drafts
            verify = expand_gemma4_decode_batch(
                batch,
                np.zeros((rows, self.speculative_draft_tokens), dtype=np.int64),
            )
            if feedback is None:
                hidden = self._gather_target_hidden(sequence_ids, graph_rows)
            else:
                hidden = feedback.target_hidden
            inputs.update(
                target_hidden=hidden,
                fixed_acceptance=-1 if fixed is None else fixed,
                verify_qo_indptr=np.arange(
                    0, (graph_rows + 1) * width, width, dtype=np.int32
                ),
            )
            for kind, assistant_plan, verify_plan, page_width in (
                ("local", context.local, verify.local, self.local_graph_page_capacity),
                ("global", context.global_, verify.global_, graph_history_width),
            ):
                inputs.update(
                    self._spec_graph_page_inputs(
                        kind,
                        assistant_plan,
                        verify_plan,
                        active_rows=rows,
                        graph_rows=graph_rows,
                        page_width=page_width,
                        query_width=width,
                    )
                )
        else:
            for kind, plan, page_width in (
                ("local", batch.local, self.local_graph_page_capacity),
                ("global", batch.global_, graph_history_width),
            ):
                inputs.update(
                    self._graph_page_inputs(
                        kind,
                        plan,
                        active_rows=rows,
                        graph_rows=graph_rows,
                        width=page_width,
                    )
                )
        if self._wide_graph(shape):
            for kind, envelope, page_width in (
                ("local", local_envelope, self.local_graph_page_capacity),
                ("global", history_envelope, graph_history_width),
            ):
                table = np.zeros((graph_rows, page_width), dtype=np.int32)
                first = np.zeros(graph_rows, dtype=np.int64)
                for row in range(rows):
                    start, end = map(int, envelope.kv_indptr[row : row + 2])
                    if end - start > page_width:
                        raise GraphInputError(
                            "reserved envelope exceeds graph page capacity"
                        )
                    table[row, : end - start] = envelope.page_indices[start:end]
                    first[row] = envelope.first_token_positions[row] // PAGE_SIZE
                inputs[f"envelope_{kind}_page_table"] = table
                inputs[f"envelope_{kind}_first_pages"] = first
        inputs.update(
            pad_sampling_inputs(self._sampler, native, self.config.vocab_size, padding)
        )
        # Planning is required only for non-wide wrappers. Wide graph schedules
        # were initialized once and consume the exact GPU-generated CSR.
        self._prepare_graph_attention(shape, inputs)
        return self._finish_enqueue(native, shape, inputs, feedback, batch.positions)

    def _enqueue_reserved_continuation(self, native, feedback):
        shape = feedback.shape
        sequence_ids = tuple(map(int, native.sequence_ids))
        # Engine retains the same immutable lease and row order until this
        # completion drains. Unchanged graph inputs alias their own buffers, so
        # staging copies only the newly accepted token, position and hidden data.
        inputs = self._continuation.resume(sequence_ids, feedback.continuation)
        return self._finish_enqueue(native, shape, inputs, feedback, None)

    def _finish_enqueue(self, native, shape, inputs, feedback, host_positions):
        sequence_ids = tuple(map(int, native.sequence_ids))
        rows, graph_rows = native.num_sequences, shape[0]
        width = self.speculative_draft_tokens + 1
        continuation = (
            self._continuation.begin(
                sequence_ids, shape, self._edge_graph_inputs[shape]
            )
            if feedback is None
            else feedback.continuation
        )
        if feedback is not None:
            inputs.update(feedback.inputs())
            positions = feedback.positions
        else:
            positions = torch.as_tensor(
                inputs["positions"], dtype=torch.int64, device=self.device
            )
        if feedback is not None and feedback.hidden_lease is not None:
            # Input staging can enqueue a read before raising. Invalidate the
            # old producer event before accepting any new consumer work.
            feedback.hidden_lease.event = None
        output = self._run_cuda_graph("decode", shape, inputs)
        if feedback is not None and feedback.hidden_lease is not None:
            # Retain the prior hidden source through its queued consumer graph.
            feedback.hidden_lease.record_use()
        accepted = (
            self._graph_acceptance[shape]
            if self.speculative_draft_tokens
            else DeviceAcceptance.from_samples(output[:, :0], output, positions)
        )
        next_inputs = accepted.feedback()
        hidden_lease = None
        if self.speculative_draft_tokens:
            if not hasattr(self, "_edge_hidden_pool"):
                self._edge_hidden_pool = _EdgeHiddenPool(
                    self.device, self.config.hidden_size
                )
            hidden_lease = self._edge_hidden_pool.acquire(graph_rows)
            selected = hidden_lease.tensor
            indices = accepted.accepted_drafts[:, None, None].expand(
                -1, 1, self.config.hidden_size
            )
            torch.gather(
                self._graph_verify_hidden[shape],
                1,
                indices,
                out=selected[:, None, :],
            )
            self._set_target_hidden(sequence_ids, selected[:rows])
        next_feedback = _EdgeDeviceFeedback(next_inputs, hidden_lease, continuation)
        completion = DeviceCompletion.from_acceptance(
            accepted,
            feedback=next_feedback,
            rows=rows,
            stats=self._speculation if self.speculative_draft_tokens else None,
            device=self.device,
            plan=native,
            scores=score_completion(self._graph_logits[shape], accepted, native),
        )
        if hidden_lease is not None:
            hidden_lease.record_use()
        if self.cross_page_overlap and not self._wide_graph(shape):
            # Preserve the faster split/native small-batch kernels. Explicit
            # None tells Engine to harvest before obtaining a fresh lease.
            host = host_positions
            lower = host - width if feedback is not None else host
            if np.any(lower // PAGE_SIZE != (host + 2 * width - 1) // PAGE_SIZE):
                completion.feedback = None
                self._continuation.drain(continuation)
        self._edge_device_metadata_rebuilds = getattr(
            self, "_edge_device_metadata_rebuilds", 0
        ) + int(self._wide_graph(shape))
        self._edge_async_enqueues = getattr(self, "_edge_async_enqueues", 0) + 1
        self._edge_async_continuations = getattr(
            self, "_edge_async_continuations", 0
        ) + int(feedback is not None)
        return completion

    @torch.inference_mode()
    @observe_execution
    def enqueue(self, plan, *, feedback=None):
        """Queue genuine graph work; Engine owns completion and writer leases."""
        if feedback is None:
            self._continuation.drain()
        if plan.phase != BatchPhase.DECODE:
            if feedback is not None:
                raise GraphInputError("a continuation requires captured decode")
            return self.forward(plan)
        self._check_cuda_graphs(plan.phase)
        if isinstance(feedback, _EdgeDeviceFeedback) and self._wide_graph(
            feedback.shape
        ):
            with self._cuda_failure:
                return self._enqueue_reserved_continuation(plan, feedback)
        batch = self._build_batch_plan(plan)
        with self._cuda_failure:
            result = self._enqueue_graph(plan, batch, feedback)
        if result is None and feedback is not None:
            raise GraphInputError(
                "a device continuation requires its captured Gemma decode graph"
            )
        return self.forward(plan) if result is None else result

    def _acquire_attention_plan_staging(self):
        from resin.backend.cuda.staging import TensorStager

        pending = getattr(self, "_attention_plan_staging_pending", [])
        available = getattr(self, "_attention_plan_staging_available", [])
        outstanding = []
        for event, staging in pending:
            if event is not None and event.query():
                available.append(staging)
            else:
                outstanding.append((event, staging))
        self._attention_plan_staging_pending = outstanding
        self._attention_plan_staging_available = available
        return available.pop() if available else TensorStager(self.device)
