"""Measure SGLang's committed decode intervals with the shared prompt artifact.

The scheduler observer is an ordinary SGLang plugin, enabled only by this CLI's
environment. It changes no scheduling or model computation and adds no device
synchronization. Its timestamps follow upstream's existing result commitment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


class DecodeWindow:
    """Pure host accounting; caller supplies already committed request counters."""

    def __init__(self, case):
        self.case = case
        self.ids = tuple(case["request_ids"])
        self.observations = []
        self.error = None
        self.complete = False
        self.saved = False
        self.max_decode_cohort = 0
        self.admission_excluded = False

    def observe(self, timestamp, rows, *, graph=True, prefill=False):
        if self.complete:
            return
        rows = {row["id"]: row for row in rows if row["id"] in self.ids}
        if not rows:
            return
        if not prefill:
            self.max_decode_cohort = max(self.max_decode_cohort, len(rows))
        if not self.observations and any(row["finished"] for row in rows.values()):
            self.error = "request retired before the complete cohort began decode"
            self.admission_excluded = True
            self.complete = True
            return
        if prefill or set(rows) != set(self.ids):
            if self.observations:
                self.error = (
                    "partial or re-prefilled cohort after full-batch decode began"
                )
                self.complete = True
            return
        ordered = [rows[rid] for rid in self.ids]
        if any(row["finished"] or row["cached_tokens"] for row in ordered):
            self.error = "request retired or reused a cached prefix inside the window"
            self.complete = True
            return
        counts = [row["tokens"] for row in ordered]
        if self.observations and any(
            count <= previous
            for count, previous in zip(
                counts, self.observations[-1]["tokens"], strict=True
            )
        ):
            self.error = "committed output did not advance for every request"
            self.complete = True
            return
        self.observations.append(
            dict(
                time=timestamp,
                tokens=counts,
                verify=[row["verify"] for row in ordered],
                accepted=[row["accepted"] for row in ordered],
                graph=bool(graph),
            )
        )
        self.complete = (
            len(self.observations)
            == self.case["warmup_steps"] + self.case["measure_steps"]
        )

    def result(self):
        if not self.complete:
            raise ValueError("incomplete decode window")
        if self.error:
            return dict(
                status="admission_excluded" if self.admission_excluded else "invalid",
                reason=self.error,
                case=self.case,
                **self.case,
                maximum_observed_decode_cohort=self.max_decode_cohort,
                observations=self.observations,
            )
        warmup = self.case["warmup_steps"]
        if warmup < 1:
            raise ValueError("at least one completed warmup interval is required")
        anchor = self.observations[warmup - 1]
        measured = self.observations[warmup:]
        end = measured[-1]
        elapsed = end["time"] - anchor["time"]
        if elapsed <= 0:
            raise ValueError("non-positive completion interval")
        tokens = [b - a for a, b in zip(anchor["tokens"], end["tokens"], strict=True)]
        verify = [b - a for a, b in zip(anchor["verify"], end["verify"], strict=True)]
        accepted = [
            b - a for a, b in zip(anchor["accepted"], end["accepted"], strict=True)
        ]
        previous = anchor
        steps, bursts = [], []
        for event in measured:
            steps.append(event["time"] - previous["time"])
            bursts.append(
                [
                    b - a
                    for a, b in zip(previous["tokens"], event["tokens"], strict=True)
                ]
            )
            previous = event
        context = self.case["context_length"]
        count = sum(tokens)
        drafts = self.case["draft_tokens"] * sum(verify)
        return dict(
            status="passed",
            phase="decode",
            backend="sglang",
            **self.case,
            batch_size=len(self.ids),
            global_batch_size=len(self.ids),
            window_seconds=elapsed,
            committed_tokens=count,
            per_request_committed_tokens=tokens,
            per_request_tokens_per_second=[value / elapsed for value in tokens],
            per_request_amortized_tpot_seconds=[elapsed / value for value in tokens],
            aggregate_tokens_per_second=count / elapsed,
            per_user_tokens_per_second=count / elapsed / len(self.ids),
            gpu_tokens_per_second=count / elapsed / self.case["gpu_count"],
            step_seconds=steps,
            committed_token_bursts=bursts,
            per_request_verify_passes=verify,
            accepted_drafts=sum(accepted),
            proposed_drafts=drafts,
            acceptance_rate=sum(accepted) / drafts if drafts else None,
            context_start=[context + value for value in anchor["tokens"]],
            context_end=[context + value for value in end["tokens"]],
            context_mean=sum(
                context + value for event in measured for value in event["tokens"]
            )
            / len(measured)
            / len(self.ids),
            graph_completions=sum(event["graph"] for event in measured),
            observations=self.observations,
            timing_boundary="continuous scheduler result-commit intervals; includes host gaps; excludes admission/prefill/retirement",
            interactivity="amortized per-user committed output rate/TPOT; speculative bursts are not individual SSE token arrivals",
        )


def capacity_preflight(
    case,
    capacity,
    *,
    page_size,
    physical_peak,
    prefill_chunk,
    window_tokens,
    output_budget,
):
    batch = len(case["request_ids"])
    full = batch * math.ceil(physical_peak / page_size) * page_size
    window = batch * math.ceil(window_tokens / page_size) * page_size + prefill_chunk
    failures = []
    # The worker's max_req_len already excludes its own reserved anchor;
    # scheduler admission reserves another token beyond the requested output.
    maximum_output = capacity["max_request_length"] - case["context_length"] - 1
    if output_budget > maximum_output:
        failures.append(
            f"output budget requires {output_budget}, admission permits {maximum_output}"
        )
    if full > capacity["full_tokens"]:
        failures.append(f"full KV requires {full}, capacity {capacity['full_tokens']}")
    if batch > capacity["max_running_requests"]:
        failures.append(
            f"request slots require {batch}, capacity {capacity['max_running_requests']}"
        )
    for name, value in capacity["window_tokens"].items():
        if window > value:
            failures.append(f"window {name} requires {window}, capacity {value}")
    return dict(
        status="capacity_exceeded" if failures else "fits",
        reasons=failures,
        full_tokens_required=full,
        window_tokens_required=window,
        capacity=capacity,
    )


def save_completed_case(path, record, output_budget):
    """Retain raw output even when its post-window validation fails."""
    path.write_text(json.dumps(record) + "\n")
    if record["status"] not in ("passed", "admission_excluded"):
        raise ValueError(f"invalid timed cohort: {record.get('reason')}")
    outputs = record["final_outputs"]
    actual = [item["meta_info"]["completion_tokens"] for item in outputs]
    expected = [output_budget] * len(record["request_ids"])
    if actual != expected:
        raise ValueError(f"final completion_tokens={actual}, expected={expected}")
    cached = [item["meta_info"].get("cached_tokens", 0) for item in outputs]
    if any(cached):
        raise ValueError(f"unexpected final cached_tokens={cached}")


def install_plugin():
    """Entry point loaded in each upstream scheduler process before startup."""
    if not (plan_path := os.environ.get("RESIN_SGLANG_BENCH_PLAN")):
        return
    Scheduler = importlib.import_module("sglang.srt.managers.scheduler").Scheduler
    SchedulerBatchResultProcessor = importlib.import_module(
        "sglang.srt.managers.scheduler_components.batch_result_processor"
    ).SchedulerBatchResultProcessor
    get_parallel = importlib.import_module("sglang.srt.runtime_context").get_parallel

    plan = json.loads(Path(plan_path).read_text())
    cases = plan["cases"] + plan.get("trace_cases", [])
    trace_ids = {case["id"] for case in plan.get("trace_cases", [])}
    windows = {case["id"]: DecodeWindow(case) for case in cases}
    request_cases = {rid: case["id"] for case in cases for rid in case["request_ids"]}
    output = Path(plan["output"])
    profiler = None

    def observe(batch, *, graph, prefill):
        # All values below are existing host-side request fields. No tensor
        # inspection, synchronization, file I/O or collective occurs in a window.
        ids = {request_cases[req.rid] for req in batch.reqs if req.rid in request_cases}
        timestamp = time.perf_counter()
        for case_id in ids:
            window = windows[case_id]
            if window.saved:
                continue
            rows = [
                dict(
                    id=req.rid,
                    tokens=len(req.output_ids),
                    verify=req.spec_verify_ct,
                    accepted=req.spec_num_correct_drafts,
                    cached_tokens=req.cached_tokens,
                    finished=req.finished(),
                )
                for req in batch.reqs
            ]
            window.observe(timestamp, rows, graph=graph, prefill=prefill)
            if window.complete:
                # The final timestamp has already been recorded. Persist only
                # outside the measured interval, retaining every observed gap.
                record = window.result()
                record["rank"] = get_parallel().tp_rank
                (output / f"{case_id}-rank-{get_parallel().tp_rank}.json").write_text(
                    json.dumps(record) + "\n"
                )
                window.saved = True
                if case_id in trace_ids and window.case.get("profile_after_window"):
                    if record["status"] != "passed" or profiler is None:
                        raise RuntimeError(
                            "profiling requires a complete diagnostic cohort"
                        )
                    ProfileReq = importlib.import_module(
                        "sglang.srt.managers.io_struct"
                    ).ProfileReq
                    # Arm on every rank after host commitments, not after a
                    # guessed number of prefill/chunk/retraction forwards.
                    armed_forward = profiler.get_forward_ct()
                    start_step = armed_forward + 1
                    response = profiler._profile(
                        ProfileReq(
                            output_dir=str(output / "profiles" / case_id),
                            start_step=start_step,
                            num_steps=3,
                            activities=["CPU", "GPU"],
                            profile_id=case_id,
                            profile_prefix="after-diagnostic-window",
                            with_stack=False,
                            record_shapes=False,
                        )
                    )
                    if not response.success:
                        raise RuntimeError(response.message)
                    (
                        output
                        / f"{case_id}-profile-start-rank-{get_parallel().tp_rank}.json"
                    ).write_text(
                        json.dumps(
                            dict(
                                start_forward=start_step,
                                armed_forward=armed_forward,
                                last_committed_forward=batch.forward_iter,
                                completed_decode_intervals=len(window.observations),
                                committed_context_at_arm=record["context_end"],
                                diagnostic_only=True,
                            )
                        )
                        + "\n"
                    )

    original_decode = SchedulerBatchResultProcessor.process_batch_result_decode
    original_prefill = SchedulerBatchResultProcessor.process_batch_result_prefill
    original_info = Scheduler.get_init_info

    def decode(self, batch, result):
        value = original_decode(self, batch, result)
        observe(batch, graph=result.can_run_cuda_graph, prefill=False)
        return value

    def prefill(self, batch, result):
        value = original_prefill(self, batch, result)
        observe(batch, graph=False, prefill=True)
        return value

    def info(self):
        nonlocal profiler
        value = original_info(self)
        profiler = self.profiler_manager
        capacity = dict(
            full_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            max_request_length=self.max_req_len,
            max_request_input_length=self.max_req_input_len,
            window_tokens=self.window_token_capacities(),
        )
        (output / f"capacity-rank-{get_parallel().tp_rank}.json").write_text(
            json.dumps(capacity, indent=2) + "\n"
        )
        return value

    SchedulerBatchResultProcessor.process_batch_result_decode = decode
    SchedulerBatchResultProcessor.process_batch_result_prefill = prefill
    Scheduler.get_init_info = info


def load_prompts(path):
    # The shared loader is standard-library-only; load its file without importing
    # Resin's native extension into SGLang's isolated dependency environment.
    source = Path(__file__).resolve().parents[1] / "resin/bench_prompts.py"
    spec = importlib.util.spec_from_file_location("resin_benchmark_prompts", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared prompt loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PromptArtifact(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-args", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    config = json.loads(args.server_args.read_text())
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts)
    assert prompts.digest == plan["prompt_sha256"]
    for case in plan["cases"]:
        prompts.cohort(case["context_length"], len(case["request_ids"]))
        assert case["warmup_steps"] >= 1 and case["measure_steps"] >= 1
        assert not case.get("profile_after_window"), (
            "primary windows cannot arm profiling"
        )
    assert not any(name.startswith("SGLANG_SIMULATE_ACC") for name in os.environ)
    os.environ["RESIN_SGLANG_BENCH_PLAN"] = str(args.plan.resolve())
    Engine = importlib.import_module("sglang.srt.entrypoints.engine").Engine
    ServerArgs = importlib.import_module("sglang.srt.server_args").ServerArgs

    server_args = ServerArgs(**config)
    manifest = dict(
        plan=plan,
        server_args=server_args.resolved_dict(),
        prompt_metadata=prompts.metadata,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version,
        cpu_affinity=sorted(getattr(os, "sched_getaffinity", lambda _: [])(0)),
        packages={
            package.metadata["Name"]: package.version
            for package in importlib.metadata.distributions()
        },
    )
    if source := plan.get("source_directory"):
        revision = subprocess.check_output(
            ["git", "-C", source, "rev-parse", "HEAD"], text=True
        ).strip()
        assert revision == plan["source_revision"]
        manifest["source_revision"] = revision
        manifest["source_status"] = subprocess.check_output(
            ["git", "-C", source, "status", "--short"], text=True
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    if args.preflight:
        plugins = importlib.import_module("sglang.srt.plugins")
        plugins.load_plugins()
        processor = importlib.import_module(
            "sglang.srt.managers.scheduler_components.batch_result_processor"
        ).SchedulerBatchResultProcessor
        assert processor.process_batch_result_decode.__module__ == "bench_sglang", (
            "benchmark observer plugin was not installed"
        )
        return
    hardware = {}
    for name, command in (
        (
            "gpu",
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,clocks.sm,clocks.mem,power.limit",
                "--format=csv",
            ],
        ),
        ("topology", ["nvidia-smi", "topo", "-m"]),
        ("cpu", ["lscpu"]),
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        hardware[name] = dict(
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    (output / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    started = time.perf_counter()
    engine = Engine(server_args=server_args)
    startup = time.perf_counter() - started
    records = []
    try:
        capacity = json.loads((output / "capacity-rank-0.json").read_text())
        for case in plan["cases"]:
            preflight = capacity_preflight(case, capacity, **plan["capacity_preflight"])
            if preflight["status"] != "fits":
                record = dict(**case, status="capacity_exceeded", preflight=preflight)
            elif len(case["request_ids"]) > max(plan["capture_batch_sizes"]):
                record = dict(
                    **case, status="deferred_capture_capacity", preflight=preflight
                )
            else:
                engine.flush_cache()
                sampling = dict(plan["sampling"], max_new_tokens=plan["output_budget"])
                begin = time.perf_counter()
                outputs = engine.generate(
                    input_ids=prompts.cohort(
                        case["context_length"], len(case["request_ids"])
                    ),
                    sampling_params=sampling,
                    rid=case["request_ids"],
                    cache_salt=case["request_ids"],
                    stream=False,
                )
                # Preserve endpoint evidence even if no scheduler window formed.
                (output / f"{case['id']}-outputs.json").write_text(
                    json.dumps(outputs) + "\n"
                )
                path = output / f"{case['id']}-rank-0.json"
                if not path.exists():
                    raise RuntimeError(f"no complete scheduler window for {case['id']}")
                record = json.loads(path.read_text())
                record.update(
                    startup_seconds=startup,
                    request_cohort_seconds=time.perf_counter() - begin,
                    preflight=preflight,
                    final_outputs=outputs,
                )
                save_completed_case(
                    output / f"{case['id']}.json", record, plan["output_budget"]
                )
            records.append(record)
            (output / f"{case['id']}.json").write_text(json.dumps(record) + "\n")
            print(
                "[bench-result] "
                + json.dumps(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in ("observations", "final_outputs")
                    }
                ),
                flush=True,
            )
        (output / "results.json").write_text(json.dumps(records) + "\n")
        # Separate diagnostic requests begin only after every primary timing
        # window. Their profiler overhead never enters a plotted curve point.
        fitted = {
            record["batch_size"] for record in records if record["status"] == "passed"
        }
        selected = {1, 8, max(fitted)} if fitted else set()
        for case in plan.get("trace_cases", []):
            batch = len(case["request_ids"])
            if batch not in fitted or batch not in selected:
                continue
            deadline = float(os.environ.get("RESIN_SGLANG_BENCH_DEADLINE", "inf"))
            if deadline - time.monotonic() < 180:
                (output / f"{case['id']}-skipped.json").write_text(
                    json.dumps(dict(status="budget_excluded", case=case)) + "\n"
                )
                continue
            engine.flush_cache()
            if not case.get("profile_after_window"):
                engine.start_profile(
                    output_dir=str(output / "profiles" / case["id"]),
                    profile_by_stage=True,
                    num_steps=3,
                    activities=["CPU", "GPU"],
                    profile_id=case["id"],
                    profile_prefix="after-primary-windows",
                    merge_profiles=False,
                    with_stack=False,
                    record_shapes=False,
                )
            outputs = engine.generate(
                input_ids=prompts.cohort(case["context_length"], batch),
                sampling_params=dict(
                    plan["sampling"], max_new_tokens=plan["output_budget"]
                ),
                rid=case["request_ids"],
                cache_salt=case["request_ids"],
                stream=False,
            )
            (output / f"{case['id']}-outputs.json").write_text(
                json.dumps(
                    dict(
                        diagnostic_only=True,
                        timing_excluded=True,
                        outputs=outputs,
                        scope=(
                            "three forwards after the complete diagnostic window; "
                            "see per-rank profile-start counters and committed contexts"
                            if case.get("profile_after_window")
                            else "upstream first three decode stages after fresh prefill; "
                            "inspect actual batch/context in retained trace"
                        ),
                    )
                )
                + "\n"
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
