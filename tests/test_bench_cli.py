"""The benchmark CLI dispatches the same validated workload locally and on Modal."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import bench


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
        (["--model", "inkling", "--balanced-experts"], "balanced expert routing"),
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
