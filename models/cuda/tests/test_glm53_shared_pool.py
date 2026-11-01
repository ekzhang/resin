"""GLM decode pool policy and owned state/results across arbitrary graph buckets."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

CATALOG = (1, 2, 4, 8, 16, 32, 64, 128)
ORDER = (128, 1, 32, 8, 64, 64, 2, 16, 4, 1, 128)
PAGES, FEATURES = 257, 32


@pytest.mark.parametrize("drafts", (0, 3, 5))
@pytest.mark.parametrize("maximum", (8, 128))
def test_glm_decode_uses_one_pool_without_changing_graph_catalog(
    monkeypatch, drafts, maximum
):
    import resin_models_cuda.glm5_3_flash.cuda as module
    from resin.models.glm5_3_flash.config import Glm53Config

    monkeypatch.setattr(
        module.CudaRunner,
        "__init__",
        lambda self, **kwargs: setattr(self, "device", torch.device("cpu")),
    )
    monkeypatch.setattr(module, "resolve_model_path", lambda value: value)
    monkeypatch.setattr(
        module,
        "load_model",
        lambda *args, **kwargs: (
            SimpleNamespace(layers=(), mtp=None),
            Glm53Config(),
            {},
        ),
    )
    monkeypatch.setattr(
        module.CudaRunner, "_configure_parallel_graphs", lambda *a: None
    )
    monkeypatch.setattr(module.CudaRunner, "_configure_cache_binding", lambda *a: None)
    monkeypatch.setattr(
        module.CudaRunner, "_configure_prefill_graphs", lambda *a, **kw: ()
    )
    monkeypatch.setattr(
        module.CudaRunner,
        "_configure_cuda_graphs",
        lambda self, modes: setattr(self, "observed_modes", modes),
    )
    monkeypatch.setattr(
        module, "CacheArena", lambda *a, **kw: SimpleNamespace(binding=None)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (128 << 30, 128 << 30))
    catalog = tuple((b, 131236) for b in CATALOG if b <= maximum)
    runner = module.Glm53FlashCudaRunner(
        "unused",
        device="cpu",
        max_seq_len=131236,
        max_batch_size=maximum,
        speculative_draft_tokens=drafts,
        cuda_graph_decode_shapes=catalog,
    )
    mode = getattr(runner, "observed_modes")["decode"]
    assert mode.shapes == catalog


def inputs(batch, turn, *, active=None):
    active = batch if active is None else active
    pages = (np.arange(batch, dtype=np.int64) * 3 + turn * 5) % (PAGES - 1) + 1
    pages[active:] = 0
    return dict(
        token_ids=(np.arange(batch, dtype=np.int64) + turn * 17).reshape(batch, 1),
        positions=np.arange(batch, dtype=np.int64) + turn + 1,
        pages=pages,
        valid=np.arange(batch) < active,
        storage=(
            np.arange(PAGES * FEATURES, dtype=np.float32).reshape(PAGES, FEATURES) % 97
            + turn
        ),
    )


def oracle(data, draft):
    base = data["storage"][data["pages"]]
    result = base + data["token_ids"] + data["positions"][:, None]
    result = np.where(data["valid"][:, None], result, base)
    token = result.sum(axis=1).astype(np.int64) % 100000
    if not draft:
        output = token
    else:
        targets = token[:, None] + np.arange(draft + 1)
        proposed = targets[:, :draft].copy()
        accepts = data["positions"] % (draft + 1)
        for row, count in enumerate(accepts):
            if count < draft:
                proposed[row, count] += 11
        output = np.concatenate((proposed, targets), axis=1)
    stored = data["storage"].copy()
    stored[data["pages"]] = result
    return output, stored


def expected_tokens(output, draft, rows):
    if not draft:
        return [[int(t)] for t in output[:rows]]
    result = []
    for values in output[:rows]:
        count = 0
        while count < draft and values[count] == values[draft + count]:
            count += 1
        result.append([int(v) for v in values[:count]] + [int(values[draft + count])])
    return result


@dataclass
class Harness:
    cache: object
    other: object
    executor: object
    buffers: dict
    storage: object
    draft: int


def build_harness(draft):
    from resin.backend.cuda.graphs import (
        CudaGraphExecutor,
        CudaGraphMode,
        CudaGraphStep,
    )

    buffers = {}
    # Persistent request state belongs to the cache arena, outside graph pools.
    storage = torch.as_tensor(inputs(1, 0)["storage"], device="cuda")

    def build(shape, workspace_scale=1024):
        batch = shape[0]
        example = inputs(batch, 0)
        staged = {
            k: torch.empty_like(torch.as_tensor(v), device="cuda")
            for k, v in example.items()
            if k != "storage"
        }
        output = torch.empty(
            (batch, 2 * draft + 1) if draft else (batch,),
            dtype=torch.long,
            device="cuda",
        )
        # These reusable externally visible buffers are allocated outside capture.
        buffers[shape] = (staged, output)
        columns = torch.arange(draft + 1, dtype=torch.long, device="cuda")

        def launch():
            selected = storage.index_select(0, staged["pages"])
            # Graph-private scratch deliberately has a bucket-dependent lifetime.
            scratch = (
                selected[:, None, :]
                .expand(batch, workspace_scale, FEATURES)
                .contiguous()
            )
            scratch.add_(staged["token_ids"][:, :, None])
            changed = scratch[:, 0, :] + staged["positions"][:, None]
            final = torch.where(staged["valid"][:, None], changed, selected)
            storage.index_copy_(0, staged["pages"], final)
            token = final.sum(dim=1).long().remainder(100000)
            if draft:
                targets = token[:, None] + columns
                accepts = staged["positions"].remainder(draft + 1)
                proposed = (
                    targets[:, :draft]
                    + (columns[:draft][None, :] == accepts[:, None]) * 11
                )
                output.copy_(torch.cat((proposed, targets), dim=1))
            else:
                output.copy_(token)

        # Each warmup/capture parity comparison starts from identical state.
        initial = torch.as_tensor(example["storage"], device="cuda")

        def prepare():
            storage.copy_(initial)

        return CudaGraphStep(
            staged,
            (output, storage),
            launch,
            prepare=prepare,
        )

    def capture_inputs(shape):
        return {k: v for k, v in inputs(shape[0], 0).items() if k != "storage"}

    executor = CudaGraphExecutor(
        "cuda:0",
        {
            "decode": CudaGraphMode(
                tuple((b, 16384) for b in CATALOG), build, capture_inputs
            ),
            "other": CudaGraphMode(
                ((3, 32768), (17, 32768)),
                lambda shape: build(shape, workspace_scale=8192),
                capture_inputs,
            ),
        },
        synchronize_failure=lambda: torch.cuda.synchronize(),
    )
    executor.capture_all()
    cache, other = executor.mode("decode"), executor.mode("other")
    assert cache._workspace is other._workspace
    return Harness(cache, other, executor, buffers, storage, draft)


def rows_from_completion(completion):
    result = completion.result()
    offsets = result.token_indptr.tolist()
    tokens = result.token_ids.tolist()
    return [tokens[a:b] for a, b in zip(offsets, offsets[1:])]


def enqueue(harness, data, rows):
    from resin.backend.cuda.completion import DeviceAcceptance, DeviceCompletion

    shape = (len(data["positions"]), 16384)
    result = harness.cache.run(shape, {k: v for k, v in data.items() if k != "storage"})
    positions = torch.as_tensor(data["positions"], device="cuda")
    output, storage = result
    samples = output if harness.draft else output.reshape(-1, 1)
    acceptance = DeviceAcceptance.from_samples(
        samples[:, : harness.draft], samples[:, harness.draft :], positions
    )
    pending = DeviceCompletion.from_acceptance(acceptance, rows=rows, device="cuda:0")
    # Match externally consumed state: copy before another graph may reuse scratch.
    snapshot = storage.clone()
    return pending, snapshot


def test_protocol_includes_repeated_unsorted_buckets_and_all_capture_shapes():
    assert ORDER[:5] == (128, 1, 32, 8, 64)
    assert set(ORDER) == set(CATALOG)
    assert any(a == b for a, b in zip(ORDER, ORDER[1:]))
    for draft in (0, 3, 5):
        for batch in CATALOG:
            data = inputs(batch, 3, active=max(0, batch - 2))
            output, stored = oracle(data, draft)
            assert output.shape == ((batch,) if not draft else (batch, 2 * draft + 1))
            assert np.array_equal(stored[0], data["storage"][0])
            assert len(expected_tokens(output, draft, max(0, batch - 2))) == max(
                0, batch - 2
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA graph-pool ownership requires a CUDA device",
)
@pytest.mark.parametrize("draft", (0, 3, 5))
def test_arbitrary_order_owned_outputs_feedback_and_idle_across_modes(draft):
    torch.cuda.reset_peak_memory_stats()
    harness = build_harness(draft)
    snapshots = []
    try:
        for turn, batch in enumerate(ORDER, start=1):
            active = 0 if turn % 5 == 0 else max(1, batch - turn % 3)
            first = inputs(batch, turn, active=active)
            expected1, state1 = oracle(first, draft)
            harness.storage.copy_(torch.as_tensor(first["storage"], device="cuda"))
            pending1, snapshot1 = enqueue(harness, first, active)
            # Device feedback is used before harvesting either completion. No CPU read.
            second = inputs(batch, turn + 1, active=active)
            second["token_ids"] = pending1.feedback.token_ids[:, None]
            second["positions"] = pending1.feedback.positions
            # Expected continuation is independently derived on CPU.
            rows1 = expected_tokens(expected1, draft, batch)
            logical2 = inputs(batch, turn + 1, active=active)
            logical2["storage"] = state1
            logical2["token_ids"] = np.array([r[-1] for r in rows1], dtype=np.int64)[
                :, None
            ]
            logical2["positions"] = first["positions"] + np.array(
                [len(r) for r in rows1]
            )
            logical2["token_ids"][active:] = 0
            logical2["positions"][active:] = 1
            expected2, state2 = oracle(logical2, draft)
            pending2, snapshot2 = enqueue(harness, second, active)
            # Replay a different bucket before either host harvest; snapshots and D2H
            # staging must remain independently owned even when pool blocks are reused.
            other = ORDER[turn % len(ORDER)]
            third = inputs(other, turn + 2)
            third["storage"] = state2
            expected3, state3 = oracle(third, draft)
            result3 = harness.cache.run(
                (other, 16384), {k: v for k, v in third.items() if k != "storage"}
            )
            assert np.array_equal(result3[0].cpu().numpy(), expected3)
            assert np.array_equal(harness.storage.cpu().numpy(), state3)
            # A different semantic mode shares the allocator pool too. It may
            # overwrite temporary blocks, but never retained results or pages.
            extra_batch = 3 if turn % 2 else 17
            fourth = inputs(extra_batch, turn + 3)
            fourth["storage"] = state3
            expected4, state4 = oracle(fourth, draft)
            result4 = harness.other.run(
                (extra_batch, 32768),
                {k: v for k, v in fourth.items() if k != "storage"},
            )
            assert np.array_equal(result4[0].cpu().numpy(), expected4)
            assert np.array_equal(harness.storage.cpu().numpy(), state4)
            assert rows_from_completion(pending2) == expected_tokens(
                expected2, draft, active
            )
            assert rows_from_completion(pending1) == expected_tokens(
                expected1, draft, active
            )
            assert rows_from_completion(pending1) == expected_tokens(
                expected1, draft, active
            )
            assert np.array_equal(snapshot1.cpu().numpy(), state1)
            assert np.array_equal(snapshot2.cpu().numpy(), state2)
            assert (
                pending1.feedback.token_ids.untyped_storage().data_ptr()
                != harness.buffers[(batch, 16384)][1].untyped_storage().data_ptr()
            )
            snapshots.append((snapshot1, state1))
            gc.collect()
            pressure = torch.empty((64, 64, FEATURES), device="cuda").fill_(turn)
            del pressure
        torch.cuda.synchronize()
        for retained, expected in snapshots:
            assert np.array_equal(retained.cpu().numpy(), expected)
        assert torch.cuda.max_memory_allocated() <= 2 * 1024**3
        assert not harness.cache.stats["poisoned"]
        assert harness.cache.stats["eager_fallback_count"] == 0
        assert harness.other.stats["eager_fallback_count"] == 0
        assert harness.other.stats["replay_count"] == len(ORDER)
        assert harness.cache.stats["replay_count"] == 3 * len(ORDER)
        assert harness.cache.stats["captured_shapes"] == tuple(
            (b, 16384) for b in CATALOG
        )
    finally:
        harness.executor.close()
