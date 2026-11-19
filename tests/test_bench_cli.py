"""The benchmark CLI dispatches the same validated workload locally and on Modal."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import bench


def test_repeats_defer_all_profiles_and_keep_identical_headroom():
    from resin.bench import BenchCase

    cases = [
        BenchCase(phase="decode", context_length=32768, batch_size=batch)
        for batch in (1, 8, 32)
    ]
    seen, profiles = [], []

    def trace(case, step):
        profiles.append((len(seen), case.batch_size))

    def sweep(engine, selected, **kwargs):
        seen.append((selected, kwargs))
        for case in selected:
            kwargs["trace_step"](case, lambda: None)

    bench._repeat_sweeps(
        sweep,
        object(),
        cases,
        options=dict(repeats=3, trace_after_sweep=True, trace_batch_sizes=[1, 32]),
        metadata={"source": "frozen"},
        trace_step=trace,
        warmup_steps=8,
        measure_steps=128,
        on_result=object(),
    )
    assert profiles == [(4, 1), (4, 32)]
    assert [entry[1]["metadata"]["repeat"] for entry in seen[:3]] == [1, 2, 3]
    assert all(entry[1]["warmup_steps"] == 8 for entry in seen)
    assert all(entry[1]["measure_steps"] == 128 for entry in seen)
    assert all(entry[1]["trace_step"] is not None for entry in seen)
    assert seen[-1][1]["on_result"] is None
    assert seen[-1][1]["metadata"]["benchmark_role"] == "untimed_trace_cohort"


def test_capacity_only_probe_never_enters_capture_or_retained_cases(capsys):
    from resin import CacheGroupLayout, CacheLayout
    from resin.backend.capacity import plan_cache_capacity
    from resin.bench import BenchCase

    plan = plan_cache_capacity(100_000, 100_000, 0.9, 100)
    raw = SimpleNamespace(
        cache_capacity_plan=plan,
        cache_layout=CacheLayout(
            parent_capacity=plan.parent_capacity, groups=[CacheGroupLayout(128)]
        ),
        runner_config=SimpleNamespace(max_accepted_per_step=8),
        enqueue=lambda: None,
    )
    cases = [BenchCase(phase="decode", context_length=32768, batch_size=1)]
    options: dict = dict(
        warmup_steps=8,
        measure_steps=128,
        draft_tokens=7,
        trace_dir="traces",
        capacity_probe_batch_sizes=[64],
    )
    retained, skipped = bench._cache_preflight(raw, cases, options)
    assert retained == cases and not skipped
    probe = options["cache_capacity_probes"][0]
    assert probe["batch_size"] == 64
    assert probe["peak_context"] == 33905
    assert probe["fits"] is False and probe["measured"] is False
    assert '"probes"' in capsys.readouterr().out


@pytest.mark.parametrize("value", [0, -1])
def test_repeat_count_must_be_positive(value):
    args = bench.argument_parser().parse_args(
        ["--model", "inkling2", "--repeats", str(value)]
    )
    with pytest.raises(ValueError, match="repeats"):
        bench._configuration(args)


def test_resident_sweep_api_accepts_and_forwards_cli_repeat_options(monkeypatch):
    args = bench.argument_parser().parse_args(
        [
            "--model",
            "inkling2",
            "--repeats",
            "3",
            "--trace-after-sweep",
            "--trace-batch-sizes",
            "1,8",
            "--capacity-probe-batch-sizes",
            "64",
        ]
    )
    _, cases, topology, options = bench._configuration(args)
    checked = []

    def validate(model, cases, topology, forwarded):
        checked.append(forwarded)
        raise RuntimeError("CPU-only API boundary")

    monkeypatch.setattr(bench, "_validate_configuration", validate)
    with pytest.raises(RuntimeError, match="CPU-only API boundary"):
        bench.run_benchmark(
            "inkling2", "/unused", cases, parallelism=topology, **options
        )
    assert checked[0]["repeats"] == 3
    assert checked[0]["trace_after_sweep"] is True
    assert checked[0]["trace_batch_sizes"] == [1, 8]
    assert checked[0]["capacity_probe_batch_sizes"] == [64]


def test_fast_allreduce_reaches_runner_and_rejects_unsupported_models(monkeypatch):
    from resin.bench import BenchCase

    args = bench.argument_parser().parse_args(
        ["--model", "inkling2", "--fast-allreduce"]
    )
    spec, cases, topology, options = bench._configuration(args)
    monkeypatch.setattr(bench, "create_runner", lambda *args, **kwargs: kwargs)
    runner_options = bench._runner(
        spec.name,
        "/checkpoint",
        device="cuda:0",
        groups=None,
        cases=[BenchCase(**cases[0])],
        options=options,
        context_capacity=8192,
    )
    assert options["fast_allreduce"] is True
    assert runner_options["fast_allreduce"] is True
    assert topology.world_size == 2
    for model in ("gpt-oss-120b", "glm-5.3-flash"):
        args = bench.argument_parser().parse_args(
            ["--model", model, "--fast-allreduce"]
        )
        with pytest.raises(ValueError, match="does not support fast all-reduce"):
            bench._configuration(args)


@pytest.mark.parametrize("name", bench.MODEL_SPECS)
def test_model_defaults_build_valid_natural_sweeps(name):
    args = bench.argument_parser().parse_args(["--model", name])
    spec, cases, topology, options = bench._configuration(args)
    assert {case["phase"] for case in cases} == {"prefill", "decode"}
    assert topology.world_size == spec.world_size
    assert options["draft_tokens"] == spec.draft_tokens
    assert "accepted_drafts" not in options
    assert options["balanced_experts"] is False
    assert [
        (case["context_length"], case["batch_size"])
        for case in cases
        if case["phase"] == "prefill"
    ] == [(length, 1) for length in (4096, 8192, 16384, 32768, 65536)]
    assert [
        (case["context_length"], case["batch_size"])
        for case in cases
        if case["phase"] == "decode"
    ] == [(8192, batch) for batch in (1, 2, 4, 8, 16, 32, 64)]


def test_paired_topologies_keep_allocation_and_independent_metadata(monkeypatch):
    import subprocess

    from deploy import modal_bench

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "GPU0\nGPU1\n")
    seen = []

    def run(model, cases, topology, options, **kwargs):
        seen.append((topology, options["metadata"]["benchmark_sequence"]))
        return [{"topology": topology}]

    monkeypatch.setattr(modal_bench, "_run_sweep", run)
    topology = dict(world_size=2, attn_tp=2, attn_dp=1, ep=2, cp=1)
    variants = [{"name": "EP2"}, {"name": "ETP2", "topology": {"ep": 1}}]
    modal_bench.sweep_sequence.local("inkling2", [], topology, {}, variants)
    assert [item[0]["ep"] for item in seen] == [2, 1]
    assert all(item[0]["attn_tp"] == 2 for item in seen)
    assert [item[1]["name"] for item in seen] == ["EP2", "ETP2"]
    assert all(item[1]["gpu_uuid"] == "GPU0\nGPU1" for item in seen)
    assert topology["ep"] == 2
    with pytest.raises(ValueError, match="retain the GPU count"):
        modal_bench.sweep_sequence.local(
            "inkling2",
            [],
            topology,
            {},
            [{"name": "wrong", "topology": {"world_size": 4}}],
        )
    assert len(seen) == 2


def test_saved_custom_prompt_and_output_retention_are_forwarded():
    args = bench.argument_parser().parse_args(
        [
            "--model",
            "inkling2",
            "--prompt-source",
            "saved",
            "--prompt-corpus",
            "literal.zip",
            "--retain-outputs",
            "--context-lengths",
            "27",
            "--batch-sizes",
            "1",
        ]
    )
    _, cases, _, options = bench._configuration(args)
    assert options["prompt_source"] == "saved"
    assert options["retain_outputs"] is True
    assert str(options["prompt_corpus"]) == "literal.zip"
    assert all(case["context_length"] == 27 for case in cases)


def test_phase_specific_sweeps_override_shared_values():
    args = bench.argument_parser().parse_args(
        [
            "--model",
            "glm-5.3-flash",
            "--context-lengths",
            "2048",
            "--batch-sizes",
            "2",
            "--prefill-context-lengths",
            "4096,8192",
            "--prefill-batch-sizes",
            "1",
            "--decode-context-lengths",
            "8192",
            "--decode-batch-sizes",
            "1,4",
        ]
    )
    _, cases, _, options = bench._configuration(args)
    assert [
        (case["phase"], case["context_length"], case["batch_size"]) for case in cases
    ] == [
        ("prefill", 4096, 1),
        ("prefill", 8192, 1),
        ("decode", 8192, 1),
        ("decode", 8192, 4),
    ]
    assert options["prefill_chunk_size"] == 8192


def test_shared_sweep_options_remain_compatible():
    args = bench.argument_parser().parse_args(
        [
            "--model",
            "glm-5.3-flash",
            "--context-lengths",
            "1024,2048",
            "--batch-sizes",
            "1,2",
            "--prefill-chunk-size",
            "512",
        ]
    )
    _, cases, _, options = bench._configuration(args)
    assert [
        (case["phase"], case["context_length"], case["batch_size"]) for case in cases
    ] == [
        (phase, length, batch)
        for phase in ("prefill", "decode")
        for length in (1024, 2048)
        for batch in (1, 2)
    ]
    assert options["prefill_chunk_size"] == 512


@pytest.mark.parametrize("ep,expert_tp", [(4, 1), (2, 2), (1, 4)])
def test_glm_expert_tp_keeps_all_four_attention_replicas(ep, expert_tp):
    args = bench.argument_parser().parse_args(
        ["--model", "glm-5.3-flash", "--ep", str(ep), "--attn-dp", "4"]
    )
    _, _, topology, _ = bench._configuration(args)
    assert topology.attn_dp == topology.world_size == 4
    assert topology.ep == ep
    assert topology.expert_tp == expert_tp


def test_glm_rejects_attention_replication_without_four_independent_drivers():
    args = bench.argument_parser().parse_args(
        ["--model", "glm-5.3-flash", "--ep", "1", "--attn-dp", "2"]
    )
    with pytest.raises(ValueError, match="attention-DP=world size"):
        bench._configuration(args)


def test_deepseek_world_override_scales_replica_and_expert_axes():
    args = bench.argument_parser().parse_args(
        [
            "--model",
            "deepseek-v41",
            "--world-size",
            "8",
            "--draft-tokens",
            "4",
            "--balanced-experts",
        ]
    )
    _, _, topology, options = bench._configuration(args)
    assert topology.world_size == topology.attn_dp == topology.ep == 8
    assert topology.attn_tp == topology.cp == 1
    assert options["draft_tokens"] == 4 and options["balanced_experts"]


@pytest.mark.parametrize(
    "arguments,fragment",
    [
        (["--accepted-drafts", "3"], "unrecognized arguments: --accepted-drafts"),
        (["--draft-tokens", "-1"], "draft_tokens"),
        (["--measure-steps", "0"], "measure_steps"),
        (["--warmup-steps", "-1"], "warmup_steps"),
        (["--prefill-chunk-size", "0"], "prefill_chunk_size"),
        (["--phases", "decode,invalid"], "phases"),
        (["--gpu-memory-utilization", "1.5"], "gpu_memory_utilization"),
    ],
)
def test_invalid_options_fail_before_remote_dispatch(arguments, fragment, capsys):
    with pytest.raises(SystemExit) as error:
        bench.main(["--model", "deepseek-v41", "--backend", "modal", *arguments])
    assert error.value.code == 2
    assert fragment in capsys.readouterr().err


@pytest.mark.parametrize("action", ["--download-only", "--publish-only"])
def test_local_presets_have_no_modal_checkpoint_actions(action, capsys):
    with pytest.raises(SystemExit) as error:
        bench.main(["--model", "ornith-1.0-35b", "--backend", "modal", action])
    assert error.value.code == 2
    assert "runs locally" in capsys.readouterr().err


def test_local_preset_prefill_chunk_is_checked_before_loading(capsys):
    with pytest.raises(SystemExit) as error:
        bench.main(
            [
                "--model",
                "ornith-1.0-35b",
                "--checkpoint",
                "/missing",
                "--prefill-chunk-size",
                "8192",
            ]
        )
    assert error.value.code == 2
    assert "4096-token chunks" in capsys.readouterr().err


def test_local_requires_explicit_checkpoint(capsys):
    with pytest.raises(SystemExit):
        bench.main(["--model", "gpt-oss-20b"])
    assert "--checkpoint is required" in capsys.readouterr().err


def test_local_prefill_trace_uses_fresh_run_directory_and_appends_results(
    tmp_path, monkeypatch
):
    from deploy import models
    from scripts import merge_bench_traces

    output = tmp_path / "results.jsonl"
    output.write_text('{"prior":true}\n')
    checkpoint = tmp_path / "checkpoint"
    traces = tmp_path / "traces"
    seen = {}
    monkeypatch.setattr(bench, "source_metadata", lambda: {"git_commit": "fixture"})

    def require(spec, **options):
        assert spec.name == "gpt-oss-20b" and options["checkpoint"] == checkpoint
        assert options["draft_tokens"] == 0
        return checkpoint

    def run(model, path, cases, *, parallelism, **options):
        assert model == "gpt-oss-20b" and path == checkpoint
        assert parallelism.world_size == 1
        assert {case["phase"] for case in cases} == {"prefill"}
        assert options["metadata"] == {
            "git_commit": "fixture",
            "checkpoint_revision": "local",
        }
        seen.update(options)
        return [{"phase": "prefill", "input_tokens": 8192}]

    monkeypatch.setattr(models, "require_checkpoint", require)
    monkeypatch.setattr(bench, "run_benchmark", run)
    monkeypatch.setattr(
        merge_bench_traces, "main", lambda argv: seen.update(merged=argv)
    )
    bench.main(
        [
            "--model",
            "gpt-oss-20b",
            "--checkpoint",
            str(checkpoint),
            "--prompt-source",
            "synthetic",
            "--phases",
            "prefill",
            "--context-lengths",
            "4096,8192",
            "--batch-sizes",
            "1",
            "--trace-dir",
            str(traces),
            "--output",
            str(output),
        ]
    )
    assert Path(seen["trace_dir"]).parent == traces
    assert seen["merged"] == [seen["trace_dir"]]
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"prior": True},
        {"phase": "prefill", "input_tokens": 8192},
    ]


def test_modal_prefill_dispatch_and_trace_download_use_remote_paths(
    tmp_path, monkeypatch
):
    import modal

    from deploy import modal_bench

    seen = {}
    traces = tmp_path / "traces"
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(bench, "source_metadata", lambda: {"git_commit": "fixture"})
    monkeypatch.setattr(modal, "enable_output", nullcontext)
    monkeypatch.setattr(modal_bench, "app", SimpleNamespace(run=nullcontext))

    def remote(model, cases, topology, options):
        assert model == "deepseek-v41" and topology["world_size"] == 4
        assert {case["phase"] for case in cases} == {"prefill"}
        assert options["trace_dir"].startswith("/bench-traces/")
        assert str(tmp_path) not in options["trace_dir"]
        assert "accepted_drafts" not in options
        seen.update(options)
        return [{"phase": "prefill"}]

    def with_options(**options):
        assert options["gpu"] == "B300:4"
        return SimpleNamespace(remote=remote)

    def retrieve(source, destination):
        assert source == seen["trace_dir"] and destination == traces
        seen["downloaded"] = True
        return [str(traces / "prefill-all-ranks.json.gz")]

    monkeypatch.setattr(
        modal_bench, "sweep", SimpleNamespace(with_options=with_options)
    )
    monkeypatch.setattr(modal_bench, "download_traces", retrieve)
    bench.main(
        [
            "--model",
            "deepseek-v41",
            "--backend",
            "modal",
            "--prompt-source",
            "synthetic",
            "--phases",
            "prefill",
            "--trace-dir",
            str(traces),
            "--output",
            str(output),
        ]
    )
    assert seen["downloaded"]
    assert json.loads(output.read_text()) == {"phase": "prefill"}


def test_modal_download_is_cpu_only_and_uses_shared_registry(monkeypatch, capsys):
    import modal

    from deploy import modal_bench

    monkeypatch.setattr(modal, "enable_output", nullcontext)
    monkeypatch.setattr(modal_bench, "app", SimpleNamespace(run=nullcontext))
    monkeypatch.setattr(
        modal_bench,
        "download",
        SimpleNamespace(
            remote=lambda name, metadata: {"model": name, "metadata_only": metadata},
        ),
    )

    def cannot_allocate(args):
        raise AssertionError("download cannot allocate a benchmark")

    monkeypatch.setattr(bench, "_configuration", cannot_allocate)
    bench.main(["--model", "gemma4", "--backend", "modal", "--publish-only"])
    assert json.loads(capsys.readouterr().out) == {
        "model": "gemma4",
        "metadata_only": True,
    }


@pytest.mark.parametrize(
    "arguments,fragment",
    [
        (["--model", "inkling2", "--balanced-experts"], "balanced expert routing"),
        (["--model", "qwen3_5", "--gpu", "B300"], "GPU-specific kernel"),
    ],
)
def test_unsupported_model_options_fail_before_remote_allocation(
    arguments, fragment, capsys
):
    with pytest.raises(SystemExit) as error:
        bench.main(["--backend", "modal", *arguments])
    assert error.value.code == 2
    assert fragment in capsys.readouterr().err


@pytest.mark.parametrize("name", ["gpt-oss-20b", "gemma4"])
def test_model_graph_capacity_is_checked_before_remote_allocation(name, capsys):
    with pytest.raises(SystemExit) as error:
        bench.main(
            [
                "--backend",
                "modal",
                "--model",
                name,
                "--batch-sizes",
                "256",
                "--prefill-chunk-size",
                "1",
            ]
        )
    assert error.value.code == 2
    assert "decode graph batch capacity is 128" in capsys.readouterr().err
