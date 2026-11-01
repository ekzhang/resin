"""Cross-mode graph scratch reuse must preserve state and delayed outputs."""

from __future__ import annotations

import pytest
import torch

from resin.backend.cuda.graphs import (
    CudaGraphExecutor,
    CudaGraphMode,
    CudaGraphProgram,
    CudaGraphRegion,
    CudaGraphStep,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


@pytest.mark.parametrize("mode_order", [("prefill", "decode"), ("decode", "prefill")])
def test_shared_pool_alternating_modes_buckets_state_and_delayed_outputs(mode_order):
    # Persistent state is external to every capture, like an Engine cache slab.
    state = torch.zeros(16, device="cuda")
    saved_buffers = {}
    temporary_addresses = {}
    width = 65536

    def mode(name):
        offset = 5 if name == "prefill" else 11

        def build(shape):
            rows = shape[0]
            value = torch.zeros(rows, device="cuda")
            active = torch.ones(rows, dtype=torch.bool, device="cuda")
            intermediate = torch.zeros_like(value)
            output = torch.zeros_like(value)
            saved_buffers[name, rows] = output

            def first():
                # Substantial scratch, initialized on every launch and released
                # before the next region. Other modes may reuse these addresses.
                temporary = value[:, None].expand(rows, width).contiguous()
                squared = temporary.square()
                if torch.cuda.is_current_stream_capturing():
                    temporary_addresses[name, rows] = (
                        temporary.data_ptr(),
                        squared.data_ptr(),
                    )
                intermediate.copy_(squared[:, 0] + offset + state[:rows])

            def last():
                output.copy_(torch.where(active, intermediate * 2, -1))
                state[:rows].copy_(torch.where(active, output, state[:rows]))

            def graph_break():
                intermediate.add_(3)

            def prepare():
                state.zero_()

            return CudaGraphProgram(
                inputs={"value": value, "active": active},
                output=output,
                regions=(
                    CudaGraphRegion("first", first),
                    CudaGraphRegion("break", graph_break, capture=False),
                    CudaGraphRegion("last", last),
                ),
                prepare=prepare,
            )

        return CudaGraphMode(
            ((1,), (4,), (16,)),
            build,
            lambda shape: {
                "value": torch.ones(shape[0], device="cuda"),
                "active": torch.ones(shape[0], dtype=torch.bool, device="cuda"),
            },
        )

    graphs = CudaGraphExecutor(
        "cuda",
        {name: mode(name) for name in mode_order},
        synchronize_failure=lambda: torch.cuda.synchronize(),
    )
    try:
        graphs.capture_all()
        pools = {
            graph.pool()
            for name in mode_order
            for entry in graphs.mode(name)._entries.values()
            for graph in entry.graphs
            if graph is not None
        }
        assert len(pools) == 1
        # The test exercises actual reused storage, not just equal pool handles.
        assert set(temporary_addresses["prefill", 16]) & set(
            temporary_addresses["decode", 16]
        )
        state.zero_()
        expected_state = torch.zeros(16)
        pending, latest = [], {}
        schedule = [
            ("decode", 1),
            ("prefill", 16),
            ("decode", 4),
            ("prefill", 4),
            ("decode", 16),
            ("prefill", 1),
            ("decode", 4),
            ("prefill", 16),
            ("decode", 1),
        ]
        for step, (name, rows) in enumerate(schedule):
            value = torch.arange(rows, dtype=torch.float32) + step
            active = (torch.arange(rows) + step) % 3 != 0
            # Include a completely idle bucket between active work.
            if step == 3:
                active.zero_()
            offset = 5 if name == "prefill" else 11
            expected = torch.where(
                active, (value.square() + offset + expected_state[:rows] + 3) * 2, -1
            )
            expected_state[:rows] = torch.where(active, expected, expected_state[:rows])
            output = graphs.mode(name).run(
                (rows,), {"value": value.to("cuda"), "active": active.to("cuda")}
            )
            assert output is saved_buffers[name, rows]
            latest[name, rows] = expected.clone()
            owned = output.clone()
            host = torch.empty_like(expected, pin_memory=True)
            host.copy_(output, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
            pending.append((done, host, owned, expected.clone()))
        # Harvest only after other modes and buckets have reused the pool.
        for done, host, owned, expected in reversed(pending):
            done.synchronize()
            torch.testing.assert_close(host, expected, rtol=0, atol=0)
            torch.testing.assert_close(owned.cpu(), expected, rtol=0, atol=0)
        for key, expected in latest.items():
            torch.testing.assert_close(
                saved_buffers[key].cpu(), expected, rtol=0, atol=0
            )
        torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)

        # Free one mode's graphs while the other mode still uses the same pool.
        graphs.mode("prefill").close()
        output = graphs.mode("decode").run(
            (1,),
            {
                "value": torch.ones(1, device="cuda"),
                "active": torch.ones(1, dtype=torch.bool, device="cuda"),
            },
        )
        torch.testing.assert_close(
            output.cpu(), (expected_state[:1] + 15) * 2, rtol=0, atol=0
        )
    finally:
        graphs.close()


@pytest.mark.parametrize("mode_order", [("decode", "verify"), ("verify", "decode")])
@torch.inference_mode()
def test_fused_sampling_shared_pool_replays_policies_and_preserves_owned_outputs(
    mode_order,
):
    from resin.backend.cuda.sampling import FusedSampler, SamplingTensors

    sampler = FusedSampler()
    vocab, buckets = 2051, (1, 4, 8)
    widths = {"decode": 1, "verify": 4}
    saved_buffers = {}
    fields = (
        "temperatures",
        "top_ps",
        "top_ks",
        "seeds",
        "offsets",
        "stochastic_mask",
    )

    def payload(name, rows, turn):
        width = widths[name]
        generator = torch.Generator().manual_seed(193 + turn * 17 + rows + width)
        logits = torch.randn(rows, width, vocab, generator=generator).to(
            "cuda", dtype=torch.bfloat16
        )
        # Reuse graph shapes while changing every supported runtime policy.
        k, p, stochastic = (
            (vocab, 1.0, True),
            (40, 0.9, True),
            (vocab, 0.7, True),
            (7, 1.0, True),
            (vocab, 1.0, False),
        )[turn % 5]
        tensors = SamplingTensors(
            temperatures=torch.linspace(0.3, 1.7, rows, device="cuda"),
            top_ps=torch.full((rows,), p, device="cuda"),
            top_ks=torch.full((rows,), k, dtype=torch.int32, device="cuda"),
            seeds=torch.arange(rows, dtype=torch.int64, device="cuda") + 2**48 + turn,
            offsets=torch.full(
                (rows,), 2**40 + turn * 32, dtype=torch.int64, device="cuda"
            ),
            stochastic_mask=torch.full(
                (rows,), stochastic, dtype=torch.bool, device="cuda"
            ),
        )
        if rows > 1:
            tensors.stochastic_mask[-1] = False
        return {"logits": logits, **sampler.graph_inputs(tensors)}

    def sampling(inputs):
        return SamplingTensors(
            **{field: inputs["sampling_" + field] for field in fields}
        )

    def draw(name, inputs, *, out=None):
        if name == "verify":
            return sampler.sample_block(
                inputs["logits"], sampling(inputs), out=out, start_step=1
            )
        result = sampler.sample_tensors(
            inputs["logits"][:, 0],
            sampling(inputs),
            out=None if out is None else out[:, 0],
        )
        return result[:, None] if out is None else out

    def mode(name):
        def build(shape):
            rows = shape[0]
            inputs = payload(name, rows, 0)
            output = torch.empty(rows, widths[name], dtype=torch.int64, device="cuda")
            saved_buffers[name, rows] = output
            return CudaGraphStep(inputs, output, lambda: draw(name, inputs, out=output))

        return CudaGraphMode(
            tuple((rows,) for rows in buckets),
            build,
            lambda shape: payload(name, shape[0], 0),
        )

    graphs = CudaGraphExecutor(
        "cuda",
        {name: mode(name) for name in mode_order},
        synchronize_failure=lambda: torch.cuda.synchronize(),
    )
    try:
        graphs.capture_all()
        pools = {
            graph.pool()
            for name in mode_order
            for entry in graphs.mode(name)._entries.values()
            for graph in entry.graphs
            if graph is not None
        }
        assert len(pools) == 1
        assert all(graphs.mode(name).stats["replay_count"] == 0 for name in mode_order)
        pending, latest = [], {}
        schedule = (
            ("decode", 8),
            ("verify", 4),
            ("decode", 1),
            ("verify", 8),
            ("decode", 4),
            ("verify", 1),
            ("decode", 8),
            ("verify", 4),
            ("decode", 1),
            ("verify", 8),
        )
        for turn, (name, rows) in enumerate(schedule):
            inputs = payload(name, rows, turn)
            seeds, offsets = (
                inputs["sampling_seeds"].clone(),
                inputs["sampling_offsets"].clone(),
            )
            # Eager comparison uses separate storage and the same request-local
            # random streams. Independent distribution oracles live in sampling tests.
            expected = draw(name, inputs)
            output = graphs.mode(name).run((rows,), inputs)
            assert output is saved_buffers[name, rows]
            latest[name, rows] = expected
            owned = output.clone()
            host = torch.empty(output.shape, dtype=torch.int64, pin_memory=True)
            host.copy_(output, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
            pending.append((done, host, owned, expected, inputs, seeds, offsets))

        # Keep completion copies alive through later sampling calls using the
        # same pool, including another replay of the same mode and bucket.
        for done, host, owned, expected, inputs, seeds, offsets in reversed(pending):
            done.synchronize()
            torch.testing.assert_close(host, expected.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(owned, expected, rtol=0, atol=0)
            torch.testing.assert_close(inputs["sampling_seeds"], seeds, rtol=0, atol=0)
            torch.testing.assert_close(
                inputs["sampling_offsets"], offsets, rtol=0, atol=0
            )
        for key, expected in latest.items():
            torch.testing.assert_close(saved_buffers[key], expected, rtol=0, atol=0)
        for name in mode_order:
            stats = graphs.mode(name).stats
            assert stats["capture_count"] == len(buckets)
            assert stats["replay_count"] == sum(name == item[0] for item in schedule)
            assert stats["eager_fallback_count"] == 0
    finally:
        graphs.close()


@torch.inference_mode()
def test_pending_feedback_drives_successor_before_either_shared_pool_result_is_harvested():
    from resin.backend.cuda.completion import DeviceAcceptance, DeviceCompletion

    def build(shape):
        rows = shape[0]
        tokens = torch.zeros(rows, dtype=torch.int64, device="cuda")
        positions = torch.ones(rows, dtype=torch.int64, device="cuda")
        samples = torch.empty((rows, 3), dtype=torch.int64, device="cuda")

        def launch():
            samples[:, 0].copy_(tokens + 1)
            samples[:, 1].copy_(tokens + 1)
            samples[:, 2].copy_(tokens + 2)

        return CudaGraphStep(
            {"tokens": tokens, "positions": positions}, (samples, positions), launch
        )

    def mode(shapes):
        return CudaGraphMode(
            shapes,
            build,
            lambda shape: {
                "tokens": torch.zeros(shape[0], dtype=torch.int64, device="cuda"),
                "positions": torch.ones(shape[0], dtype=torch.int64, device="cuda"),
            },
        )

    graphs = CudaGraphExecutor(
        "cuda",
        {"decode": mode(((1,),)), "other": mode(((4,),))},
        synchronize_failure=lambda: torch.cuda.synchronize(),
    )
    try:
        graphs.capture_all()

        def enqueue(tokens, positions):
            samples, anchors = graphs.mode("decode").run(
                (1,), {"tokens": tokens, "positions": positions}
            )
            accepted = DeviceAcceptance.from_samples(
                samples[:, :1], samples[:, 1:], anchors
            )
            return DeviceCompletion.from_acceptance(accepted, rows=1)

        first = enqueue(
            torch.tensor([3], device="cuda"), torch.tensor([5], device="cuda")
        )
        second = enqueue(first.feedback.token_ids, first.feedback.positions)
        graphs.mode("other").run(
            (4,),
            {
                "tokens": torch.full((4,), 100, dtype=torch.int64, device="cuda"),
                "positions": torch.ones(4, dtype=torch.int64, device="cuda"),
            },
        )
        assert second.result().token_ids.tolist() == [6, 7]
        assert first.result().token_ids.tolist() == [4, 5]
        assert first.feedback.positions.tolist() == [7]
        assert second.feedback.positions.tolist() == [9]
    finally:
        graphs.close()
