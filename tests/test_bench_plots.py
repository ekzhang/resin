"""Chart normalization and compatibility checks use synthetic measurements only."""

import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.plot_bench import axes, load_records, main


def record(phase="decode", **changes):
    row = {
        "schema_version": 1,
        "phase": phase,
        "context_length": 4096,
        "batch_size": 8,
        "global_batch_size": 32,
        "gpu_count": 8,
        "attn_dp": 4,
        "prefill_tokens": 0,
        "input_tokens": 0,
        "draft_tokens": 4,
        "measured_accept_rate": 0.25,
        "balanced_experts": True,
        "warmup_steps": 1,
        "measure_steps": 2,
        "step_seconds": [0.1, 0.3],
        "window_seconds": 0.4,
        "committed_tokens": 128,
        "metadata": {
            "model": "fixture/model",
            "checkpoint_revision": "fixture-revision",
            "git_commit": "fixture-commit",
            "source_digest": "fixture-source",
            "hardware": "fixture GPU",
            "parallelism": {
                "world_size": 8,
                "attn_dp": 4,
                "attn_tp": 2,
                "cp": 1,
                "ep": 4,
            },
            "graphs": {"prefill": "breakable", "decode": "full"},
            "quant": {"experts": "mxfp4"},
            "expert_backend": "flashinfer",
        },
    }
    if phase == "prefill":
        row.update(input_tokens=40000, prefill_tokens=20000)
    row.update(changes)
    return row


def write(path: Path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_decode_projects_steps_without_relabeling_measured_acceptance(tmp_path):
    row = record()
    rows = load_records([write(tmp_path / "decode.jsonl", row)])
    before = deepcopy(rows)
    assert axes(rows, "decode") == {4096: [(10, 40, "8")]}
    assert axes(rows, "decode", accept_rate=0.75) == {4096: [(20, 80, "8")]}
    assert axes(rows, "decode", accept_rate=0) == {4096: [(5, 20, "8")]}
    assert rows == before
    assert rows[0]["measured_accept_rate"] == 0.25
    assert rows[0]["committed_tokens"] / 0.4 / 8 == 40


def test_legacy_natural_records_mix_with_current_measurements(tmp_path):
    path = write(tmp_path / "compatible.jsonl", record(accepted_drafts=None), record())
    rows = load_records([path])
    assert len(rows) == 1
    assert "accepted_drafts" not in rows[0]


def test_legacy_forced_acceptance_remains_labeled(tmp_path):
    source = write(tmp_path / "forced.jsonl", record(accepted_drafts=1))
    output = tmp_path / "plots"
    main([str(source), "--mode", "decode", "--output-dir", str(output)])
    assert "(fixed 1/4)" in (output / "decode.svg").read_text()


def test_prefill_uses_actual_input_counts_and_measured_step_latency(tmp_path):
    rows = load_records([write(tmp_path / "prefill.jsonl", record("prefill"))])
    assert axes(rows, "prefill") == {4096: [(200, 12500, "20,000")]}


def test_prefill_cohort_latency_includes_every_chunk(tmp_path):
    row = record("prefill", prefill_seconds=[0.4])
    rows = load_records([write(tmp_path / "cohort.jsonl", row)])
    assert axes(rows, "prefill") == {4096: [(400, 12500, "8")]}
    output = tmp_path / "plots"
    main(
        [
            str(tmp_path / "cohort.jsonl"),
            "--mode",
            "prefill",
            "--output-dir",
            str(output),
        ]
    )
    svg = (output / "prefill.svg").read_text()
    assert "Prefill latency (ms/cohort)" in svg
    assert "Latency covers every chunk of the requested context" in svg


def test_prefill_rejects_mixed_step_and_cohort_latency(tmp_path):
    path = write(
        tmp_path / "mixed.jsonl",
        record("prefill"),
        record("prefill", prefill_seconds=[0.4]),
    )
    with pytest.raises(ValueError, match="mixed .*configurations"):
        load_records([path])


@pytest.mark.parametrize("durations", [[], [0.1], [float("nan")]])
def test_prefill_cohort_timings_cover_the_measured_window(tmp_path, durations):
    path = write(tmp_path / "bad.jsonl", record("prefill", prefill_seconds=durations))
    with pytest.raises(ValueError, match="prefill_seconds"):
        load_records([path])


def test_cli_projection_is_explicit_and_uses_recorded_draft_count(tmp_path):
    source = write(tmp_path / "decode.jsonl", record())
    output = tmp_path / "plots"
    main(
        [
            str(source),
            "--mode",
            "decode",
            "--accept-rate",
            "0.75",
            "--output-dir",
            str(output),
        ]
    )
    svg = (output / "decode.svg").read_text()
    assert "Projected acceptance 75%, 4 drafts" in svg


def test_context_curves_and_latest_repeated_point_are_kept_separate(tmp_path):
    original = record()
    repeated = record(window_seconds=0.8)
    other_context = record(context_length=8192)
    another_batch = record(batch_size=16, global_batch_size=64)
    path = write(
        tmp_path / "sweep.jsonl", original, other_context, repeated, another_batch
    )
    rows = load_records([path])
    assert axes(rows, "decode") == {
        4096: [(5, 20, "8"), (5, 40, "16")],
        8192: [(10, 40, "8")],
    }


def test_older_records_keep_their_original_flashinfer_backend(tmp_path):
    old, current = record(), record()
    del old["metadata"]["expert_backend"]
    path = write(tmp_path / "compatible.jsonl", old, current)
    assert len(load_records([path])) == 1
    current["metadata"]["expert_backend"] = "deepgemm"
    write(path, old, current)
    with pytest.raises(ValueError, match="mixed .*configurations"):
        load_records([path])


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "checkpoint_revision",
        "git_commit",
        "source_digest",
        "hardware",
        "parallelism",
        "graphs",
        "quant",
        "expert_backend",
    ],
)
def test_mixed_fixed_metadata_is_rejected_before_duplicate_selection(tmp_path, field):
    first, different = record(), record()
    different["metadata"][field] = (
        {"world_size": 8, "attn_dp": 4, "cp": 2}
        if field == "parallelism"
        else "different"
    )
    path = write(tmp_path / "mixed.jsonl", first, different, first)
    with pytest.raises(ValueError, match="mixed .*configurations"):
        load_records([path])


@pytest.mark.parametrize(
    "changes",
    [{"draft_tokens": 3}, {"accepted_drafts": 2}, {"balanced_experts": False}],
)
def test_mixed_execution_conditions_are_not_connected(tmp_path, changes):
    path = write(tmp_path / "mixed.jsonl", record(), record(**changes))
    with pytest.raises(ValueError, match="mixed .*configurations"):
        load_records([path])


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"step_seconds": []},
        {"step_seconds": [0.1, float("nan")]},
        {"window_seconds": 0},
        {"global_batch_size": 8},
        {"gpu_count": 4},
        {"attn_dp": 2},
        {"accepted_drafts": 5},
        {"measured_accept_rate": 1.1},
        {"balanced_experts": 1},
        {"batch_size": True},
    ],
)
def test_invalid_schema_or_measurements_have_source_locations(tmp_path, changes):
    path = write(tmp_path / "bad.jsonl", record(**changes))
    with pytest.raises(ValueError, match=r"bad.jsonl:1:"):
        load_records([path])


def test_prefill_budget_is_not_accepted_as_actual_measured_batch(tmp_path):
    path = write(tmp_path / "bad.jsonl", record("prefill", prefill_tokens=32768))
    with pytest.raises(ValueError, match="measured mean input"):
        load_records([path])


@pytest.mark.parametrize("row", [None, [], "wrong"])
def test_nonobject_json_has_source_location(tmp_path, row):
    path = write(tmp_path / "bad.jsonl", row)
    with pytest.raises(ValueError, match=r"bad.jsonl:1:.*JSON object"):
        load_records([path])


@pytest.mark.parametrize("field", ["prefill_chunk_size", "max_batch_tokens"])
def test_changed_prefill_budget_configuration_is_rejected(tmp_path, field):
    first, second = record(), record()
    first["metadata"][field] = 4096
    second["metadata"][field] = 8192
    path = write(tmp_path / "mixed.jsonl", first, second)
    with pytest.raises(ValueError, match="mixed .*configurations"):
        load_records([path])


@pytest.mark.parametrize(
    "options",
    [
        {"draft_tokens": 8},
        {"draft_tokens": -1},
        {"accept_rate": -0.1},
        {"accept_rate": 1.1},
        {"accept_rate": float("nan")},
    ],
)
def test_projection_options_are_validated(options):
    with pytest.raises(ValueError):
        axes([record()], "decode", **options)


def test_natural_acceptance_measurements_allow_an_explicit_projection(tmp_path):
    row = record(measured_accept_rate=0.6)
    rows = load_records([write(tmp_path / "natural.jsonl", row)])
    assert axes(rows, "decode") == {4096: [(10, 40, "8")]}
    assert axes(rows, "decode", accept_rate=0.75) == {4096: [(20, 80, "8")]}
    assert rows[0]["measured_accept_rate"] == 0.6


def test_plotting_import_does_not_load_matplotlib_or_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.plot_bench; assert 'matplotlib' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("contexts", [(4096,), (4096, 8192)])
def test_cli_renders_two_pngs_and_two_svgs(tmp_path, capsys, contexts):
    rows = [
        record(phase, context_length=context)
        for context in contexts
        for phase in ("decode", "prefill")
    ]
    path = write(tmp_path / "synthetic.jsonl", *rows)
    output = tmp_path / "plots"
    main([str(path), "--output-dir", str(output)])
    assert {file.name for file in output.iterdir()} == {
        "decode.png",
        "decode.svg",
        "prefill.png",
        "prefill.svg",
    }
    assert all(file.stat().st_size > 1000 for file in output.iterdir())
    svg = (output / "decode.svg").read_text()
    assert "fixture GPU" in svg
    assert "Projected acceptance" not in svg
    assert "Measured output throughput; acceptance 25.0%" in svg
    for context in contexts:
        assert f"Context {context:,}" in svg
    assert "decode.png" in capsys.readouterr().out


def test_shared_prefix_geometry_cannot_mix_with_independent_points(tmp_path):
    path = tmp_path / "shared.jsonl"
    independent = record()
    shared = record(
        shared_prefix_tokens=2048, cached_tokens_min=2048, cached_tokens_max=2048
    )
    path.write_text(json.dumps(independent) + "\n" + json.dumps(shared) + "\n")
    with pytest.raises(ValueError, match="mixed model"):
        load_records([path])
    shared["cached_tokens_min"] = 1024
    path.write_text(json.dumps(shared) + "\n")
    with pytest.raises(ValueError, match="exact cached token counts"):
        load_records([path])


def topology_record(*, world=1, tp=1, ep=1, **changes):
    row = record(
        context_length=8192,
        gpu_count=world,
        attn_dp=1,
        global_batch_size=8,
        draft_tokens=0,
        measured_accept_rate=None,
        committed_tokens=16,
        **changes,
    )
    row["metadata"]["parallelism"].update(
        world_size=world, attn_dp=1, attn_tp=tp, ep=ep
    )
    return row


def test_explicit_comparison_keeps_meshes_and_latest_repeats_separate(tmp_path):
    first = topology_record()
    second = topology_record(world=2, tp=2)
    repeated = topology_record(window_seconds=0.8)
    path = write(tmp_path / "meshes.jsonl", first, second, repeated)
    with pytest.raises(ValueError, match="mixed .*configurations"):
        load_records([path])
    rows = load_records([path], compare_topologies=True)
    assert len(rows) == 2
    assert axes(rows, "decode", draft_tokens=0, compare_topologies=True) == {
        (8192, "TP1 EP1 ADP1 CP1 (1 GPU)"): [(2.5, 20, "8")],
        (8192, "TP2 EP1 ADP1 CP1 (2 GPUs)"): [(5, 20, "8")],
    }


@pytest.mark.parametrize(
    "field", ["model", "source_digest", "graphs", "quant", "expert_backend"]
)
def test_topology_comparison_still_rejects_different_implementations(tmp_path, field):
    first, second = topology_record(), topology_record(world=2, ep=2)
    second["metadata"][field] = "different"
    path = write(tmp_path / "incompatible.jsonl", first, second)
    with pytest.raises(ValueError, match="topology comparisons require matching"):
        load_records([path], compare_topologies=True)


def test_decode_comparison_allows_prefill_capacity_variation_only_between_meshes(
    tmp_path,
):
    first, second = topology_record(), topology_record(world=2, tp=2)
    first["metadata"].update(prefill_chunk_size=128, max_batch_tokens=4096)
    second["metadata"].update(prefill_chunk_size=128, max_batch_tokens=16384)
    path = write(tmp_path / "capacities.jsonl", first, second)
    assert len(load_records([path], compare_topologies=True)) == 2
    changed = deepcopy(second)
    changed["metadata"]["max_batch_tokens"] = 32768
    write(path, first, second, changed)
    with pytest.raises(ValueError, match="mixed .*configurations"):
        load_records([path], compare_topologies=True)


def test_prefill_comparison_preserves_equal_capacity_requirement(tmp_path):
    first, second = (
        topology_record(phase="prefill"),
        topology_record(phase="prefill", world=2, tp=2),
    )
    first["metadata"]["max_batch_tokens"] = 4096
    second["metadata"]["max_batch_tokens"] = 16384
    path = write(tmp_path / "prefill.jsonl", first, second)
    with pytest.raises(ValueError, match="topology comparisons require matching"):
        load_records([path], compare_topologies=True)


def test_decode_rejects_partial_measured_batches_instead_of_overstating_throughput():
    row = topology_record(
        measured_global_batch_size_min=4, measured_global_batch_size_max=8
    )
    with pytest.raises(ValueError, match="fully active requested batch"):
        axes([row], "decode", draft_tokens=0)


def test_comparison_cli_labels_meshes_contexts_and_unequal_prefill_capacities(tmp_path):
    first, second = topology_record(), topology_record(world=2, ep=2)
    for row, budget in ((first, 4096), (second, 16384)):
        row["metadata"].update(prefill_chunk_size=128, max_batch_tokens=budget)
        row.update(measured_context_start_min=8197, measured_context_end_max=8218)
    path = write(tmp_path / "comparison.jsonl", first, second)
    output = tmp_path / "plots"
    main(
        [
            str(path),
            "--compare-topologies",
            "--draft-tokens",
            "0",
            "--mode",
            "decode",
            "--output-dir",
            str(output),
        ]
    )
    assert {file.suffix for file in output.iterdir()} == {".png", ".svg"}
    svg = (output / "decode.svg").read_text()
    assert "TP1 EP1 ADP1 CP1 (1 GPU); context 8,192" in svg
    assert "TP1 EP2 ADP1 CP1 (2 GPUs); context 8,192" in svg
    assert "Measured context range: 8,197–8,218" in svg
    assert "Meshes reserve different prefill capacities" in svg


def test_shared_prefix_plot_labels_both_shared_and_private_context(tmp_path):
    row = topology_record(
        shared_prefix_tokens=4096, cached_tokens_min=4096, cached_tokens_max=4096
    )
    path = write(tmp_path / "shared.jsonl", row)
    output = tmp_path / "plots"
    main(
        [
            str(path),
            "--draft-tokens",
            "0",
            "--mode",
            "decode",
            "--output-dir",
            str(output),
        ]
    )
    svg = (output / "decode.svg").read_text()
    captions = " ".join(re.findall(r"<!-- (.*?) -->", svg))
    assert "shared prefix=4,096, private suffix=4,096 tokens" in captions


def test_batch_labels_stay_inside_axes_and_avoid_crowded_endpoints():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from scripts.plot_bench import _annotate_points

    figure, axis = plt.subplots(figsize=(9, 6.4))
    canvas = FigureCanvasAgg(figure)
    try:
        axis.set(xlim=(0, 10), ylim=(0, 10))
        points = [
            (5 + i * 0.01, 0.1, label, "blue")
            for i, label in enumerate(("1", "4", "16", "32", "64"))
        ]
        _annotate_points(axis, points)
        renderer = canvas.get_renderer()
        bounds = axis.get_window_extent(renderer)
        boxes = [text.get_window_extent(renderer).padded(2) for text in axis.texts]
        assert len(boxes) == 5
        for index, box in enumerate(boxes):
            assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
            assert not any(box.overlaps(other) for other in boxes[:index])
    finally:
        plt.close(figure)


def test_plot_window_note_reports_actual_mixed_step_counts(tmp_path):
    first = topology_record(warmup_steps=5)
    second = topology_record(
        warmup_steps=20,
        measure_steps=3,
        step_seconds=[0.1, 0.2, 0.3],
        window_seconds=0.6,
    )
    second.update(batch_size=16, global_batch_size=16)
    path = write(tmp_path / "windows.jsonl", first, second)
    output = tmp_path / "plots"
    main(
        [
            str(path),
            "--draft-tokens",
            "0",
            "--mode",
            "decode",
            "--output-dir",
            str(output),
        ]
    )
    svg = (output / "decode.svg").read_text()
    assert "Window per point: 5–20 warmup steps, 2–3 measured steps." in svg
