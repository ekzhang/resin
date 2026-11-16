"""Real-checkpoint smoke test for TP/DP, DSpark graphs, overlap, and prefix reuse.

Run with RESIN_TEST_CHECKPOINT=/models/Inkling-2-Small and
RESIN_TEST_ATTN_TP=1 or 2 under a two-rank torchrun; four ranks with TP=2
exercise TP2/DP2/EP4. Generated text is not compared across execution shapes.
"""

from __future__ import annotations

import gc
import json
import os
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not os.environ.get("RESIN_TEST_CHECKPOINT")
    or "LOCAL_RANK" not in os.environ,
    reason="requires a local Inkling 2 checkpoint and a two/four-rank Blackwell torchrun",
)


@pytest.fixture(scope="module")
def real_runtime():
    from resin import Parallelism
    from resin.backend.topology import ParallelGroups
    from resin_models_cuda.inkling2.cuda import Inkling2CudaRunner

    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    tp = int(os.environ.get("RESIN_TEST_ATTN_TP", "1"))
    assert (world, tp) in ((2, 1), (2, 2), (4, 2))
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability(rank)[0] == 10
    torch.set_num_threads(4)
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", rank), timeout=timedelta(minutes=15)
    )
    groups = ParallelGroups.create(
        Parallelism(
            world_size=world,
            attn_tp=tp,
            attn_dp=world // tp,
            ep=int(os.environ.get("RESIN_TEST_EP", str(world))),
        )
    )
    runner = Inkling2CudaRunner(
        os.environ["RESIN_TEST_CHECKPOINT"],
        device=f"cuda:{rank}",
        max_seq_len=1280,
        max_batch_size=2,
        prefill_chunk_size=256,
        parallel_groups=groups,
        cache_parent_capacity=64,
        speculative_draft_tokens=7,
        cuda_graph_prefill="breakable",
        cuda_graph_decode="full",
        cuda_graph_decode_shapes=[(2, 1280)],
        **json.loads(os.environ.get("RESIN_TEST_RUNNER_OPTIONS", "{}")),
    )
    runner.capture_cuda_graphs()
    try:
        yield runner, groups
    finally:
        runner.close_cuda_graphs()
        dist.barrier()
        groups.close()
        dist.destroy_process_group()
        del runner
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def test_real_checkpoint_prefix_reuse_idle_dp_and_overlap(real_runtime):
    from transformers import AutoTokenizer

    from resin import SamplingParams, SchedulerConfig
    from resin.backend.distributed import DistributedEngine, DistributedWorker

    runner, groups = real_runtime
    try:
        worker = DistributedWorker(runner, groups)
        if not groups.is_driver:
            worker.serve()
            return
        tokenizer = AutoTokenizer.from_pretrained(
            os.environ["RESIN_TEST_CHECKPOINT"], local_files_only=True
        )
        assert tokenizer is not None
        seed = tokenizer.encode(
            "We measured the latency, throughput, and numerical correctness of a research experiment. ",
            add_special_tokens=True,
        )
        assert seed
        engine = DistributedEngine(
            worker,
            context_length=runner.max_seq_len,
            max_running_requests=2,
            prefix_cache=True,
            enable_overlap=True,
            scheduler_config=SchedulerConfig(
                max_batch_tokens=256, prefill_chunk_size=256
            ),
        )
        output_count = 4 * runner.runner_config.max_accepted_per_step

        def submit(length, *, salt="cold", count=output_count):
            return engine.submit(
                (seed * ((length + len(seed) - 1) // len(seed)))[:length],
                sampling_params=SamplingParams(
                    max_output_tokens=count, temperature=0.0, stop_token_ids=[], seed=17
                ),
                cache_salt=salt,
            )

        def drain(leases, count=output_count):
            while engine.has_work():
                engine.step(allow_overlap=True)
            for lease in leases:
                tokens = lease.take_result()
                assert len(tokens) == count
                assert all(0 <= token < runner.vocab_size for token in tokens)

        # Unequal local batches exercise graph padding and idle prefill regions.
        for pair in ((127, 129), (1023, 1025)):
            lengths = pair if groups.rank == 0 else pair[-1:]
            drain([submit(length) for length in lengths])
        cached, fresh = submit(1025), submit(1025, salt="fresh")
        drain([cached, fresh])
        assert cached.cached_tokens >= 1024 and fresh.cached_tokens == 0

        # Other DP drivers remain idle while rank zero crosses a page boundary.
        drain([submit(127, salt="idle")] if groups.rank == 0 else [])
        drain([submit(1025, salt="retire", count=1)], count=1)
        drain([submit(1025, salt="reuse-slot", count=5)], count=5)
        assert runner.cuda_graph_stats["decode"]["replay_count"] > 0
        assert engine.local.overlap_stats[1] > 0
        engine.close()
    except BaseException:
        # A failed rank must exit so torchrun can stop peers waiting in collectives.
        traceback.print_exc()
        os._exit(1)
