"""Reject invalid benchmark topology and undersized resident cache budgets."""

import json
from types import SimpleNamespace

import pytest

from resin import Parallelism
from scripts import bench


@pytest.mark.parametrize("fails", [False, True])
def test_persistent_progress_survives_worker_failure(tmp_path, monkeypatch, fails):
    output = tmp_path / "results" / "completed.json"
    record = {"phase": "decode", "committed_tokens": 8}
    monkeypatch.setattr(bench.torch.cuda, "device_count", lambda: 4)

    def spawn(function, *, args, devices, numa):
        # The real worker uses this callback only after a complete timed case.
        save = bench._result_callback(args[1])
        save(record)
        if fails:
            raise RuntimeError("next case failed")
        return SimpleNamespace(placements=[None] * 4)

    monkeypatch.setattr(bench.resin.process, "spawn", spawn)

    def run():
        return bench.run_benchmark(
            "glm-5.3-flash",
            "unused",
            [{"phase": "decode", "context_length": 128, "batch_size": 1}],
            parallelism=Parallelism(world_size=4, attn_dp=4, ep=4),
            prefill_chunk_size=512,
            progress_path=output,
        )

    if fails:
        with pytest.raises(RuntimeError, match="next case failed"):
            run()
        assert json.loads(output.read_text()) == [record]
    else:
        records = run()
        assert json.loads(output.read_text()) == records
        assert records[0]["numa"] == {"mode": "auto", "bindings": [None] * 4}
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        run()
    assert output.read_bytes() == saved
    assert not output.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize(
    "model,topology,message",
    [
        ("deepseek-v4", Parallelism(), "unknown model"),
        ("gpt-oss-20b", Parallelism(), "currently requires V4.1"),
        ("deepseek-v41", Parallelism(), "at least two ranks"),
        (
            "deepseek-v41",
            Parallelism(world_size=8, attn_dp=8, ep=4),
            "ADP=EP=world_size",
        ),
        (
            "deepseek-v41",
            Parallelism(world_size=8, cp=2, attn_dp=4, ep=8),
            "ADP=EP=world_size",
        ),
    ],
)
def test_deepgemm_rejects_wrong_models_and_meshes_before_cuda(
    monkeypatch, model, topology, message
):
    def cannot_inspect_cuda():
        raise AssertionError("invalid options must fail before CUDA inspection")

    monkeypatch.setattr(bench.torch.cuda, "device_count", cannot_inspect_cuda)
    with pytest.raises(ValueError, match=message):
        bench.run_benchmark(
            model, "unused", [], parallelism=topology, expert_backend="deepgemm"
        )


def test_cache_preflight_rejects_decode_growth_before_prefill_or_capture(capsys):
    from resin import CacheGroupLayout, CacheLayout
    from resin.backend.capacity import plan_cache_capacity
    from resin.bench import BenchCase

    layout = CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(4)])
    raw = SimpleNamespace(
        cache_layout=layout,
        cache_capacity_plan=plan_cache_capacity(100, 100, 0.5, 10),
    )
    options = dict(warmup_steps=0, measure_steps=1, draft_tokens=0)
    bench._cache_preflight(
        raw, [BenchCase(phase="prefill", context_length=4, batch_size=4)], options
    )
    with pytest.raises(
        ValueError, match="decode ctx=4 batch=4 needs 8 cache parents, capacity=4"
    ):
        bench._cache_preflight(
            raw, [BenchCase(phase="decode", context_length=4, batch_size=4)], options
        )
    rows = [
        json.loads(line.split(" ", 1)[1])
        for line in capsys.readouterr().out.splitlines()
    ]
    assert rows[0]["cases"][0]["required_parents"] == 4
    assert rows[1]["cases"][0]["peak_context"] == 6
    assert rows[1]["capacity"]["bytes_per_parent"] == 10
    assert rows[1]["capacity"]["reserved_parent_count"] == 1


def test_cache_preflight_propagates_a_peers_capacity_failure():
    from resin import CacheGroupLayout, CacheLayout
    from resin.backend.capacity import plan_cache_capacity
    from resin.bench import BenchCase

    class Distributed:
        @staticmethod
        def all_gather_object(values, local, *, group):
            assert not local  # This rank fits; a peer does not.
            values[:] = [[], ["rank 1 requires more cache parents"]]

    groups = SimpleNamespace(
        rank=0,
        domain=SimpleNamespace(size=2, handle=object()),
        distributed=Distributed(),
    )
    raw = SimpleNamespace(
        cache_layout=CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(4)]),
        cache_capacity_plan=plan_cache_capacity(100, 100, 0.5, 10),
    )
    with pytest.raises(ValueError, match="rank 1 requires more cache parents"):
        bench._cache_preflight(
            raw,
            [BenchCase(phase="prefill", context_length=4, batch_size=4)],
            dict(warmup_steps=0, measure_steps=1, draft_tokens=0),
            groups,
        )


def test_cache_preflight_credits_only_the_configured_shared_history():
    from resin import CacheGroupLayout, CacheLayout
    from resin.backend.capacity import plan_cache_capacity
    from resin.bench import BenchCase

    raw = SimpleNamespace(
        cache_layout=CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(4)]),
        cache_capacity_plan=plan_cache_capacity(100, 100, 0.5, 10),
    )
    options = dict(warmup_steps=0, measure_steps=1, draft_tokens=0)
    bench._cache_preflight(
        raw,
        [
            BenchCase(
                phase="prefill", context_length=12, batch_size=2, shared_prefix_tokens=8
            )
        ],
        options,
    )
    with pytest.raises(ValueError, match="needs 6 cache parents, capacity=4"):
        bench._cache_preflight(
            raw, [BenchCase(phase="prefill", context_length=12, batch_size=2)], options
        )


@pytest.mark.parametrize("drafts,lookahead,output_budget", [(0, 0, 22), (3, 4, 82)])
def test_trace_cache_preflight_covers_actual_continuation_reservation(
    capsys, drafts, lookahead, output_budget
):
    from resin import CacheGroupLayout, CacheLayout, RunnerConfig
    from resin.backend.capacity import plan_cache_capacity
    from resin.bench import BenchCase

    options = dict(
        warmup_steps=3,
        measure_steps=12,
        draft_tokens=drafts,
        overlap=True,
        trace_dir="traces",
    )
    context = 60
    peak_context = context + output_budget + lookahead
    required = (peak_context + 15) // 16
    layout = CacheLayout(
        parent_capacity=required - 1,
        groups=[CacheGroupLayout(16)],
        decode_write_extra_tokens=lookahead - drafts,
    )
    raw = SimpleNamespace(
        cache_layout=layout,
        cache_capacity_plan=plan_cache_capacity(100, 100, 0.5, 10),
        runner_config=RunnerConfig(
            max_accepted_per_step=drafts + 1, cache_layout=layout
        ),
        enqueue=lambda plan: None,
    )
    case = BenchCase(phase="decode", context_length=context, batch_size=1)
    # Explicit independent budgets include the strict two-output-step guard:
    # K0: 19 generated +3; K3: 73 generated +9 (maximum output4).
    assert bench._decode_budget(options, raw) == output_budget
    assert bench._context_capacity([case], options) == context + output_budget
    with pytest.raises(ValueError, match=f"needs {required} cache parents"):
        bench._cache_preflight(raw, [case], options)
    report = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert report["cases"][0]["peak_context"] == peak_context


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("drafts", [0, 3])
def test_untraced_cli_context_geometry_is_unchanged(overlap, drafts):
    from resin.bench import BenchCase

    options = dict(
        warmup_steps=3, measure_steps=12, draft_tokens=drafts, overlap=overlap
    )
    case = BenchCase(phase="decode", context_length=8192, batch_size=8)
    assert bench._context_capacity([case], options) == 8192 + 19 * (drafts + 1)
