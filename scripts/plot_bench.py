"""Plot Resin JSONL sweeps as decode and prefill throughput/latency curves."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

FIXED_METADATA = (
    "model",
    "checkpoint_revision",
    "git_commit",
    "source_digest",
    "hardware",
    "parallelism",
    "graphs",
    "quant",
    "expert_backend",
)
FIXED_CASE = (
    "draft_tokens",
    "balanced_experts",
    "shared_prefix_tokens",
)


def _number(value, name, *, integer=False, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(
            f"{name} must be a finite {'integer' if integer else 'number'} >= {minimum}"
        )
    return value


def _validate(record):
    if not isinstance(record, dict):
        raise ValueError("benchmark record must be a JSON object")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("expected benchmark schema_version=1")
    if record.get("phase") not in {"decode", "prefill"}:
        raise ValueError("phase must be decode or prefill")
    for name in (
        "context_length",
        "batch_size",
        "global_batch_size",
        "gpu_count",
        "attn_dp",
        "measure_steps",
    ):
        _number(record[name], name, integer=True, minimum=1)
    for name in ("draft_tokens", "warmup_steps", "input_tokens", "committed_tokens"):
        _number(record[name], name, integer=True)
    shared = _number(
        record.setdefault("shared_prefix_tokens", 0),
        "shared_prefix_tokens",
        integer=True,
    )
    if shared >= record["context_length"]:
        raise ValueError("shared_prefix_tokens must be less than context_length")
    if shared and any(
        record.get(name) != shared
        for name in ("cached_tokens_min", "cached_tokens_max")
    ):
        raise ValueError("shared-prefix records must verify exact cached token counts")
    _number(record["prefill_tokens"], "prefill_tokens")
    _number(record["window_seconds"], "window_seconds", minimum=math.nextafter(0, 1))
    steps = record["step_seconds"]
    if not isinstance(steps, list) or len(steps) != record["measure_steps"]:
        raise ValueError("step_seconds must contain measure_steps measurements")
    for value in steps:
        _number(value, "step_seconds", minimum=math.nextafter(0, 1))
    # Older schema-1 results can contain a fixed acceptance workload.
    accepted = record.get("accepted_drafts")
    if (
        accepted is not None
        and _number(accepted, "accepted_drafts", integer=True) > record["draft_tokens"]
    ):
        raise ValueError("accepted_drafts must not exceed draft_tokens")
    measured = record["measured_accept_rate"]
    if measured is not None and _number(measured, "measured_accept_rate") > 1:
        raise ValueError("measured_accept_rate must be between 0 and 1")
    if type(record["balanced_experts"]) is not bool:
        raise ValueError("balanced_experts must be a boolean")
    if record["global_batch_size"] != record["batch_size"] * record["attn_dp"]:
        raise ValueError("global_batch_size must equal batch_size * attn_dp")
    metadata = record["metadata"]
    if isinstance(metadata, dict):
        # Earlier schema-1 measurements had only the FlashInfer expert path.
        metadata.setdefault("expert_backend", "flashinfer")
    if not isinstance(metadata, dict) or any(
        name not in metadata for name in FIXED_METADATA
    ):
        raise ValueError(f"metadata must include {', '.join(FIXED_METADATA)}")
    topology = metadata["parallelism"]
    if (
        not isinstance(topology, dict)
        or topology.get("world_size") != record["gpu_count"]
        or topology.get("attn_dp") != record["attn_dp"]
    ):
        raise ValueError("metadata.parallelism must agree with gpu_count and attn_dp")
    if record["phase"] == "prefill":
        if record["input_tokens"] <= 0 or not math.isclose(
            record["prefill_tokens"], record["input_tokens"] / len(steps), rel_tol=1e-9
        ):
            raise ValueError(
                "prefill_tokens must be the measured mean input tokens per step"
            )
        if "prefill_seconds" in record:
            cohorts = record["prefill_seconds"]
            if not isinstance(cohorts, list) or not cohorts:
                raise ValueError("prefill_seconds must contain complete cohort timings")
            for value in cohorts:
                _number(value, "prefill_seconds", minimum=math.nextafter(0, 1))
            if not math.isclose(sum(cohorts), record["window_seconds"], rel_tol=1e-9):
                raise ValueError("prefill_seconds must sum to window_seconds")
    elif (
        record["input_tokens"]
        or record["prefill_tokens"]
        or not record["committed_tokens"]
    ):
        raise ValueError(
            "decode requires positive committed_tokens and zero prefill input tokens"
        )


def _topology(record):
    return json.dumps(record["metadata"]["parallelism"], sort_keys=True)


def _configuration(record):
    fixed = {name: record[name] for name in FIXED_CASE}
    # Keep historical forced-acceptance runs separate from natural measurements.
    fixed["accepted_drafts"] = record.get("accepted_drafts")
    fixed.update({name: record["metadata"][name] for name in FIXED_METADATA})
    for name in ("prefill_chunk_size", "max_batch_tokens"):
        fixed[name] = record["metadata"].get(name)
    fixed.update(gpu_count=record["gpu_count"], attn_dp=record["attn_dp"])
    if record["phase"] == "prefill":
        fixed["prefill_timing"] = "cohort" if "prefill_seconds" in record else "step"
    return fixed


def _range(values):
    low, high = min(values), max(values)
    return f"{low:,}" if low == high else f"{low:,}–{high:,}"


def load_records(paths: list[Path], *, compare_topologies=False) -> list[dict]:
    """Validate every row before selecting the latest repeat of a sweep point."""
    records = []
    configs = {}
    comparisons = {}
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                _validate(record)
                fixed = _configuration(record)
                signature = json.dumps(fixed, sort_keys=True)
                curve = (
                    record["phase"],
                    _topology(record) if compare_topologies else None,
                )
                previous = configs.setdefault(curve, signature)
                if signature != previous:
                    raise ValueError(
                        "mixed model/topology/graphs/quant/kernels/balance/acceptance configurations; use separate files"
                    )
                if compare_topologies:
                    # A comparison permits independent meshes, never different
                    # implementations or workload policies within/between curves.
                    for name in ("parallelism", "gpu_count", "attn_dp"):
                        fixed.pop(name)
                    if record["phase"] == "decode":
                        # These capacities affect loading/prefill, outside timing.
                        for name in ("prefill_chunk_size", "max_batch_tokens"):
                            fixed.pop(name)
                    signature = json.dumps(fixed, sort_keys=True)
                    previous = comparisons.setdefault(record["phase"], signature)
                    if signature != previous:
                        raise ValueError(
                            "topology comparisons require matching model/source/graphs/quant/kernels/balance/acceptance configurations"
                        )
                records.append(record)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    if not records:
        raise ValueError("no benchmark records found")
    latest = {}
    for record in records:
        key = (
            record["phase"],
            record["context_length"],
            record["batch_size"],
            record["prefill_tokens"],
            _topology(record),
        )
        latest[key] = record
    return list(latest.values())


def axes(
    records, phase, *, draft_tokens=None, accept_rate=None, compare_topologies=False
):
    """Use measured tokens and latency unless an acceptance projection is requested."""
    if draft_tokens is not None:
        _number(draft_tokens, "draft_tokens", integer=True)
    if accept_rate is not None and _number(accept_rate, "accept_rate") > 1:
        raise ValueError("accept_rate must be between 0 and 1")
    result = {}
    for record in sorted(
        records,
        key=lambda r: (r["context_length"], r["prefill_tokens"], r["batch_size"]),
    ):
        if record["phase"] != phase:
            continue
        seconds, steps = record["window_seconds"], len(record["step_seconds"])
        if phase == "decode":
            if any(
                record.get(name, record["global_batch_size"])
                != record["global_batch_size"]
                for name in (
                    "measured_global_batch_size_min",
                    "measured_global_batch_size_max",
                )
            ):
                raise ValueError(
                    "decode throughput requires a fully active requested batch"
                )
            if draft_tokens is not None and record["draft_tokens"] != draft_tokens:
                raise ValueError(
                    "--draft-tokens must match the measured decode draft_tokens"
                )
            speed = (
                record["committed_tokens"] / seconds / record["global_batch_size"]
                if accept_rate is None
                else steps / seconds * (1 + record["draft_tokens"] * accept_rate)
            )
            point = (
                speed,
                speed * record["global_batch_size"] / record["gpu_count"],
                f"{record['batch_size']:,}",
            )
        else:
            cohorts = record.get("prefill_seconds")
            batch = record["batch_size"] if cohorts else record["prefill_tokens"]
            label = f"{batch:,.0f}" if batch == int(batch) else f"{batch:,.1f}"
            point = (
                seconds / (len(cohorts) if cohorts else steps) * 1000,
                record["input_tokens"] / seconds / record["gpu_count"],
                label,
            )
        key = record["context_length"]
        if compare_topologies:
            topology = record["metadata"]["parallelism"]
            mesh = " ".join(
                f"{label}{topology.get(name, 1)}"
                for label, name in (
                    ("TP", "attn_tp"),
                    ("EP", "ep"),
                    ("ADP", "attn_dp"),
                    ("CP", "cp"),
                )
            )
            gpu_label = "GPU" if record["gpu_count"] == 1 else "GPUs"
            key = key, f"{mesh} ({record['gpu_count']} {gpu_label})"
        result.setdefault(key, []).append(point)
    if not result:
        raise ValueError(f"no {phase} measurements found")
    return result


def _annotate_points(axis, points):
    """Keep batch labels inside the plot and separate crowded curve endpoints."""
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    bounds = axis.get_window_extent(renderer)
    placed = []
    if legend := axis.get_legend():
        placed.append(legend.get_window_extent(renderer))
    offsets = (
        (0, 9),
        (0, -16),
        (12, 9),
        (-12, 9),
        (12, -16),
        (-12, -16),
        (24, 18),
        (-24, 18),
        (24, -24),
        (-24, -24),
        (0, 27),
        (0, -34),
    )
    for x, y, label, color in sorted(points, key=lambda point: -point[1]):
        annotation = axis.annotate(
            label,
            (x, y),
            xytext=offsets[0],
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=color,
        )
        for offset in offsets:
            annotation.set_position(offset)
            box = annotation.get_window_extent(renderer).padded(2)
            if (
                bounds.contains(box.x0, box.y0)
                and bounds.contains(box.x1, box.y1)
                and not any(box.overlaps(other) for other in placed)
            ):
                break
        placed.append(box)


def plot(
    records,
    phase,
    output_dir,
    *,
    draft_tokens=None,
    accept_rate=None,
    compare_topologies=False,
):
    curves = axes(
        records,
        phase,
        draft_tokens=draft_tokens,
        accept_rate=accept_rate,
        compare_topologies=compare_topologies,
    )
    # Loading/validating JSONL and --help do not need Matplotlib installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    selected = [record for record in records if record["phase"] == phase]
    first = selected[0]
    metadata = first["metadata"]
    topology = metadata["parallelism"]
    mesh = " ".join(
        f"{name}={topology[name]}"
        for name in ("world_size", "attn_dp", "attn_tp", "cp", "ep")
        if name in topology
    )
    graphs = ", ".join(
        f"{name}: {policy}" for name, policy in metadata["graphs"].items()
    )
    layout = "Mesh comparison" if compare_topologies else mesh
    subtitle = f"{metadata['model']} | {metadata['hardware']} | {layout} | quant={metadata['quant']} | experts={metadata['expert_backend']} | graphs={graphs} | balanced experts={first['balanced_experts']}"
    if shared := first.get("shared_prefix_tokens", 0):
        private = first["context_length"] - shared
        subtitle += f" | shared prefix={shared:,}, private suffix={private:,} tokens"
    figure, axis = plt.subplots(figsize=(9, 6.4))
    try:
        annotations = []
        for context, points in sorted(curves.items()):
            xs, ys, labels = zip(*points, strict=True)
            label = (
                f"{context[1]}; context {context[0]:,}"
                if compare_topologies
                else f"Context {context:,}"
            )
            (line,) = axis.plot(xs, ys, "o-", linewidth=2, label=label)
            for x, y, label in points:
                annotations.append((x, y, label, line.get_color()))
        decode = phase == "decode"
        figure.suptitle(
            "Decode: output throughput vs. interactivity"
            if decode
            else "Prefill: input throughput vs. latency",
            x=0.08,
            ha="left",
            fontsize=15,
            weight="bold",
        )
        axis.set_title(
            textwrap.fill(subtitle, 105),
            loc="left",
            fontsize=9,
            color="#555555",
            pad=12,
        )
        axis.set_xlabel(
            "Per-user output (tokens/s)"
            if decode
            else "Prefill latency (ms/cohort)"
            if "prefill_seconds" in first
            else "Prefill latency (ms/step)"
        )
        axis.set_ylabel("Output tokens/s/GPU" if decode else "Input tokens/s/GPU")
        axis.set_xlim(0, axis.dataLim.xmax * 1.12)
        axis.set_ylim(0, axis.dataLim.ymax * 1.2)
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=9)
        if decode:
            measured = [
                r["measured_accept_rate"]
                for r in selected
                if r["measured_accept_rate"] is not None
            ]
            observed = (
                "unavailable"
                if not measured
                else f"{min(measured):.1%}–{max(measured):.1%}"
            )
            fixed = (
                "natural"
                if first.get("accepted_drafts") is None
                else f"fixed {first['accepted_drafts']}/{first['draft_tokens']}"
            )
            if accept_rate is None:
                note = f"Measured output throughput; acceptance {observed} ({fixed})."
            else:
                drafts = first["draft_tokens"]
                note = f"Projected acceptance {accept_rate:.0%}, {drafts} drafts → τ={1 + drafts * accept_rate:g}; measured acceptance {observed} ({fixed})."
            if not first["draft_tokens"]:
                note = "Measured ordinary decode (no drafts)."
            note += "\nPoint labels: requests per attention-DP replica."
        else:
            note = (
                "Latency covers every chunk of the requested context.\nPoint labels: requests per attention-DP replica."
                if "prefill_seconds" in first
                else "Legacy step timings; point labels: measured unique input tokens per global prefill step."
            )
        if all(
            "measured_context_start_min" in row and "measured_context_end_max" in row
            for row in selected
        ):
            start = min(row["measured_context_start_min"] for row in selected)
            end = max(row["measured_context_end_max"] for row in selected)
            note += f"\nMeasured context range: {start:,}–{end:,} tokens."
        if (
            compare_topologies
            and len(
                {
                    (
                        row["metadata"].get("prefill_chunk_size"),
                        row["metadata"].get("max_batch_tokens"),
                    )
                    for row in selected
                }
            )
            > 1
        ):
            note += "\nMeshes reserve different prefill capacities; decode timing excludes prefill."
        warmup = _range([row["warmup_steps"] for row in selected])
        measured = _range([row["measure_steps"] for row in selected])
        note += f"\nWindow per point: {warmup} warmup steps, {measured} measured steps."
        figure.text(0.5, 0.025, note, ha="center", fontsize=9, color="#555555")
        figure.tight_layout(rect=(0, 0.05 + 0.025 * (note.count("\n") + 1), 1, 0.94))
        _annotate_points(axis, annotations)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = tuple(output_dir / f"{phase}.{suffix}" for suffix in ("png", "svg"))
        for path in paths:
            figure.savefig(path, dpi=180)
        return paths
    finally:
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        nargs="+",
        help="benchmark JSONL files with matching fixed configurations",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("both", "decode", "prefill"), default="both")
    parser.add_argument(
        "--draft-tokens",
        type=int,
        help="optionally require this measured draft count (otherwise read from records)",
    )
    parser.add_argument(
        "--accept-rate",
        type=float,
        help="project decode throughput at this acceptance rate; default uses measured output",
    )
    parser.add_argument(
        "--compare-topologies",
        action="store_true",
        help="plot separate mesh curves with matching model and execution policies",
    )
    args = parser.parse_args(argv)
    try:
        records = load_records(args.results, compare_topologies=args.compare_topologies)
        phases = ("decode", "prefill") if args.mode == "both" else (args.mode,)
        output = args.output_dir or args.results[0].parent / "plots"
        for phase in phases:
            for path in plot(
                records,
                phase,
                output,
                draft_tokens=args.draft_tokens,
                accept_rate=args.accept_rate,
                compare_topologies=args.compare_topologies,
            ):
                print(path.resolve())
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
