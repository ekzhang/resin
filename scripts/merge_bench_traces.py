"""Merge each prefill/decode case's rank traces into a Chrome trace for Perfetto."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

TRACE_NAME = re.compile(
    r"(?P<case>(?:prefill|decode)-ctx\d+-b\d+(?:-repeat\d+)?)"
    r"-rank(?P<rank>\d+)\.json(?:\.gz)?$"
)


def _open(path: Path, mode: str):
    return (
        gzip.open(path, mode, encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(mode, encoding="utf-8")
    )


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _id_key(value):
    # Chrome flow IDs written as strings are hexadecimal, including strings
    # without a 0x prefix. Keep numeric and hex spellings of one ID connected.
    if isinstance(value, str):
        try:
            return _key(int(value, 16))
        except ValueError:
            pass
    return _key(value)


def merge_traces(paths: Iterable[str | Path], target: str | Path) -> Path:
    """Align rank clocks and isolate process/flow IDs without changing inputs.

    PyTorch's timestamps are microseconds relative to baseTimeNanoseconds.
    Unbased traces are accepted only when every input omits that base. Original
    top-level metadata from every rank is retained in resinTraceMerge.sources.
    """
    paths, target = [Path(path) for path in paths], Path(target)
    if not paths:
        raise ValueError("at least one trace is required")
    if target.resolve() in {path.resolve() for path in paths}:
        raise ValueError("the merged target must not overwrite a raw rank trace")
    traces, ranks = [], set()
    for index, path in enumerate(paths):
        with _open(path, "rt") as source:
            trace = json.load(source)
        if not isinstance(trace, dict) or not isinstance(
            trace.get("traceEvents"), list
        ):
            raise ValueError(f"{path}: expected an object containing traceEvents")
        match = TRACE_NAME.fullmatch(path.name)
        rank = int(match["rank"]) if match else int(trace.get("resin_rank", index))
        if rank in ranks:
            raise ValueError(f"duplicate trace for rank {rank}")
        ranks.add(rank)
        base = trace.get("baseTimeNanoseconds")
        if base is not None and (
            isinstance(base, bool)
            or not isinstance(base, (int, float))
            or not math.isfinite(base)
        ):
            raise ValueError(f"{path}: baseTimeNanoseconds must be a finite number")
        traces.append((rank, path, trace, base))
    based = [item[3] is not None for item in traces]
    if any(based) and not all(based):
        raise ValueError(
            "cannot align traces with mixed missing/present timestamp bases"
        )
    minimum_base = min(item[3] for item in traces) if all(based) else 0

    merged = {key: value for key, value in traces[0][2].items() if key != "traceEvents"}
    if all(based):
        merged["baseTimeNanoseconds"] = minimum_base
    events, sources, next_pid, next_id = [], [], 1, 1
    for rank, path, trace, base in sorted(traces, key=lambda item: item[0]):
        offset = (base - minimum_base) / 1000 if base is not None else 0
        pids, original_pids, named_pids, identifiers = {}, {}, set(), {}

        def identifier(value):
            nonlocal next_id
            key = _id_key(value)
            if key not in identifiers:
                identifiers[key] = next_id
                next_id += 1
            return identifiers[key]

        for event in trace["traceEvents"]:
            if not isinstance(event, dict):
                raise ValueError(f"{path}: traceEvents must contain event objects")
            if "pid" in event:
                key = _key(event["pid"])
                if key not in pids:
                    pids[key], original_pids[key] = next_pid, event["pid"]
                    next_pid += 1
                event["pid"] = pids[key]
            if offset and "ts" in event:
                event["ts"] += offset
            for field in ("id", "bind_id"):
                if field in event:
                    event[field] = identifier(event[field])
            if "id2" in event:
                event["id2"] = {
                    scope: identifier(value) for scope, value in event["id2"].items()
                }
            if event.get("ph") == "M" and event.get("name") == "process_name":
                args = event.setdefault("args", {})
                args["name"] = f"Rank {rank} / {args.get('name', 'Process')}"
                named_pids.add(event.get("pid"))
            events.append(event)
        for key, pid in pids.items():
            if pid not in named_pids:
                events.append(
                    {
                        "ph": "M",
                        "name": "process_name",
                        "pid": pid,
                        "tid": 0,
                        "args": {"name": f"Rank {rank} / PID {original_pids[key]}"},
                    }
                )
        sources.append(
            {
                "rank": rank,
                "path": str(path),
                "timestamp_offset_us": offset,
                "metadata": {
                    key: value for key, value in trace.items() if key != "traceEvents"
                },
            }
        )
    merged["traceEvents"] = events
    merged["resinTraceMerge"] = {"sources": sources}
    target.parent.mkdir(parents=True, exist_ok=True)
    with _open(target, "wt") as destination:
        json.dump(merged, destination, separators=(",", ":"))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory of raw per-rank traces")
    parser.add_argument(
        "--output-dir", type=Path, help="defaults to the input directory"
    )
    args = parser.parse_args(argv)
    try:
        cases = defaultdict(list)
        for path in args.directory.iterdir():
            if path.is_file() and (match := TRACE_NAME.fullmatch(path.name)):
                cases[match["case"]].append(path)
        if not cases:
            raise ValueError(
                f"no prefill or decode rank traces found in {args.directory}"
            )
        directory = args.output_dir or args.directory
        for case, paths in sorted(cases.items()):
            target = directory / f"{case}-all-ranks.json.gz"
            print(merge_traces(sorted(paths), target).resolve())
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
