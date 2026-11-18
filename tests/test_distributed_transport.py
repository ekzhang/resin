"""Real Gloo transport keeps large immutable request inputs off the step path."""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from output_helpers import forward_output

import resin._resin as native
from resin import (
    BatchPhase,
    CacheGroupLayout,
    CacheLayout,
    Parallelism,
    RunnerConfig,
    SamplingParams,
)
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.backend.topology import ParallelGroups
from resin.recipe import Media

TOKENS = (10, 11, 77, 12, 13, 14)
PIXEL_SHAPE = (384, 384, 3)
PIXEL_BYTES = int(np.prod(PIXEL_SHAPE)) * 4


def _media() -> tuple[Media, ...]:
    return (Media((2, 1), image=np.full(PIXEL_SHAPE, 0.25, dtype=np.float32)),)


class _MediaRunner:
    runner_config = RunnerConfig(
        cache_layout=CacheLayout(parent_capacity=32, groups=[CacheGroupLayout(2)]),
        max_batch_tokens=2,
        prefill_chunk_size=2,
        continuation_prefill=True,
    )

    def __init__(self):
        self.bindings = []
        self.active = {}
        self.released = []
        self.calls = []

    def bind_resources(self, binding):
        assert binding.cache_pool_id is not None
        assert binding.cache_layout is self.runner_config.cache_layout
        self.bindings.append(binding.cache_pool_id)

    def forward(self, plan):
        for raw_id, media, sampling in zip(
            plan.sequence_ids, plan.media, plan.sampling_params, strict=True
        ):
            sequence_id = int(raw_id)
            assert isinstance(media, tuple) and isinstance(media[0], Media)
            if sequence_id in self.active:
                assert self.active[sequence_id] is media
            else:
                assert media[0].span == (2, 1)
                assert media[0].image is not None
                assert media[0].image.shape == PIXEL_SHAPE
                assert np.all(media[0].image == 0.25)
                self.active[sequence_id] = media
            assert sampling.temperature == 0.75
            assert sampling.top_p == 0.875
            assert sampling.top_k == 13
            assert sampling.seed == 123
            assert sampling.extra == {"nested": {"scale": 0.125}}
        self.calls.append(
            {
                "phase": "prefill" if plan.phase == BatchPhase.PREFILL else "decode",
                "ids": plan.sequence_ids.tolist(),
                "tokens": plan.token_ids.tolist(),
                "context": plan.context_lengths.tolist(),
                "cache": [
                    {
                        "slabs": group.block_slab_ids.tolist(),
                        "slots": group.block_slotnums.tolist(),
                        "append": group.append_block_offsets.tolist(),
                    }
                    for group in plan.cache_groups
                ],
            }
        )
        return forward_output([7] * plan.num_sequences)

    def release(self, sequence_ids):
        for sequence_id in map(int, sequence_ids):
            self.active.pop(sequence_id, None)
            self.released.append(sequence_id)


def _transport_process(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    groups = ParallelGroups.create(
        Parallelism(world_size=2, attn_tp=2), device_backend="gloo"
    )
    runner = _MediaRunner()
    worker = DistributedWorker(runner, groups)
    encodings = []
    large_broadcasts = []
    original_encode = native._media_to_buffers
    original_broadcast = dist.broadcast

    def encode(media):
        encodings.append(media)
        return original_encode(media)

    def broadcast(tensor, *args, **kwargs):
        assert not torch.is_inference(tensor)
        size = tensor.numel() * tensor.element_size()
        if size >= PIXEL_BYTES:
            large_broadcasts.append(size)
        return original_broadcast(tensor, *args, **kwargs)

    def object_collective(*_args, **_kwargs):
        raise AssertionError("object collectives are forbidden after worker setup")

    try:
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            patch.setattr(native, "_media_to_buffers", encode)
            patch.setattr(dist, "broadcast", broadcast)
            patch.setattr(dist, "broadcast_object_list", object_collective)
            patch.setattr(dist, "all_gather_object", object_collective)
            if groups.is_driver:
                engine = DistributedEngine(worker)
                media = _media()
                sampling = SamplingParams(
                    max_output_tokens=3,
                    temperature=0.75,
                    top_p=0.875,
                    top_k=13,
                    seed=123,
                    extra={"nested": {"scale": 0.125}},
                )

                def submit():
                    return engine.submit(TOKENS, media=media, sampling_params=sampling)

                completed = submit()
                never_started = submit()
                assert engine.cancel(never_started)
                assert never_started.take_result() == []
                assert not worker._plans._requests
                while engine.has_work():
                    engine.step()
                assert completed.take_result() == [7, 7, 7]
                assert len(encodings) == 1
                assert not worker._plans._requests

                cancelled = submit()
                engine.step()
                assert len(worker._plans._requests) == 1
                assert engine.cancel(cancelled)
                assert len(encodings) == 2
                assert not worker._plans._requests

                closed = submit()
                engine.step()
                assert len(worker._plans._requests) == 1
                engine.close()
                assert closed.done
                assert len(encodings) == 3
                assert never_started.id not in {
                    sequence for call in runner.calls for sequence in call["ids"]
                }
            else:
                worker.serve()
            assert worker.closed
            assert not worker._plans._requests
            assert not runner.active
        Path(directory, f"{rank}.json").write_text(
            json.dumps(
                {
                    "bindings": runner.bindings,
                    "calls": runner.calls,
                    "released": runner.released,
                    "encodings": len(encodings),
                    "large_broadcasts": large_broadcasts,
                }
            )
        )
    finally:
        groups.close()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required"
)
def test_gloo_registers_media_once_and_reuses_it_until_release(tmp_path):
    processes = mp.start_processes(
        _transport_process,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=2,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + 60
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                raise AssertionError(
                    "two-rank typed transport failed to finish within 60s"
                )
    finally:
        for child in processes.processes:
            if child.is_alive():
                child.terminate()
        for child in processes.processes:
            child.join(timeout=5)
    driver, follower = [
        json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)
    ]
    assert driver["bindings"] == follower["bindings"]
    assert len(driver["bindings"]) == 1
    assert driver["calls"] == follower["calls"]
    assert driver["released"] == follower["released"]
    assert [call["phase"] for call in follower["calls"]] == [
        "prefill",
        "prefill",
        "prefill",
        "decode",
        "decode",
        "prefill",
        "prefill",
    ]
    assert (driver["encodings"], follower["encodings"]) == (3, 0)
    # Repeated prefill/decode vectors are small. Pixels cross ranks once for
    # each admitted request, including after a prior request used the same media.
    assert (
        driver["large_broadcasts"] == follower["large_broadcasts"] == [PIXEL_BYTES] * 3
    )
