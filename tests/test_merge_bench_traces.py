"""Cross-rank trace clocks, tracks and flow links use synthetic events only."""

import gzip
import json

import pytest

from scripts.merge_bench_traces import main, merge_traces


def write(path, trace):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as output:
        json.dump(trace, output)
    return path


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def test_aligns_nanosecond_bases_preserves_tracks_metadata_and_raw_files(tmp_path):
    paths = []
    for rank, base in ((3, 10**18 + 2500), (1, 10**18)):
        paths.append(
            write(
                tmp_path / f"decode-ctx4096-b8-rank{rank:03}.json.gz",
                {
                    "baseTimeNanoseconds": base,
                    "displayTimeUnit": "ms",
                    "deviceProperties": [{"id": rank, "name": "fixture GPU"}],
                    "resin_rank": rank,
                    "traceEvents": [
                        {
                            "ph": "M",
                            "name": "process_name",
                            "pid": 7,
                            "args": {"name": "Python", "extra": "keep"},
                        },
                        {
                            "ph": "M",
                            "name": "thread_name",
                            "pid": 0,
                            "tid": 9,
                            "args": {"name": "stream 9"},
                        },
                        {
                            "ph": "X",
                            "name": f"cpu{rank}",
                            "pid": 7,
                            "tid": 7,
                            "ts": 10,
                            "dur": 4.25,
                        },
                        {
                            "ph": "X",
                            "cat": "kernel",
                            "name": f"gpu{rank}",
                            "pid": 0,
                            "tid": 9,
                            "ts": 12.5,
                            "dur": 2,
                            "args": {"device": rank, "stream": 9, "correlation": 42},
                        },
                    ],
                },
            )
        )
    original = [path.read_bytes() for path in paths]
    target = merge_traces(iter(paths), tmp_path / "all.json.gz")
    merged = read(target)
    events = merged["traceEvents"]
    slices = {event["name"]: event for event in events if event["ph"] == "X"}
    assert merged["baseTimeNanoseconds"] == 10**18
    assert slices["cpu1"]["ts"] == 10
    assert slices["cpu3"]["ts"] == 12.5
    assert slices["gpu3"]["ts"] == 15
    assert slices["cpu3"]["dur"] == 4.25
    assert slices["gpu3"]["args"] == {"device": 3, "stream": 9, "correlation": 42}
    assert slices["gpu3"]["tid"] == 9
    assert len({event["pid"] for event in slices.values()}) == 4
    assert all(isinstance(event["pid"], int) for event in events)
    names = {
        event["pid"]: event["args"]["name"]
        for event in events
        if event["name"] == "process_name"
    }
    assert names[slices["cpu3"]["pid"]] == "Rank 3 / Python"
    assert names[slices["gpu1"]["pid"]] == "Rank 1 / PID 0"
    assert (
        next(event for event in events if event["name"] == "process_name")["args"][
            "extra"
        ]
        == "keep"
    )
    assert len([event for event in events if event["name"] == "thread_name"]) == 2
    sources = merged["resinTraceMerge"]["sources"]
    assert [source["rank"] for source in sources] == [1, 3]
    assert sources[1]["metadata"]["deviceProperties"] == [
        {"id": 3, "name": "fixture GPU"}
    ]
    assert [path.read_bytes() for path in paths] == original


def test_flow_and_async_identifiers_are_isolated_across_ranks(tmp_path):
    paths = []
    for rank in range(2):
        paths.append(
            write(
                tmp_path / f"decode-ctx8-b2-rank{rank:03}.json",
                {
                    "baseTimeNanoseconds": 1000,
                    "traceEvents": [
                        {
                            "ph": "s",
                            "name": f"start{rank}",
                            "pid": 0,
                            "tid": 1,
                            "ts": 1,
                            "id": 10,
                        },
                        {
                            "ph": "f",
                            "name": f"end{rank}",
                            "pid": 1,
                            "tid": 2,
                            "ts": 2,
                            "id": "0xa",
                        },
                        {
                            "ph": "b",
                            "name": f"global{rank}",
                            "pid": 0,
                            "ts": 3,
                            "id2": {"global": "20"},
                        },
                        {
                            "ph": "e",
                            "name": f"global-end{rank}",
                            "pid": 1,
                            "ts": 4,
                            "id2": {"global": 32},
                        },
                        {
                            "ph": "b",
                            "name": f"local{rank}",
                            "pid": 0,
                            "ts": 5,
                            "id2": {"local": "0x30"},
                        },
                        {
                            "ph": "e",
                            "name": f"local-end{rank}",
                            "pid": 0,
                            "ts": 6,
                            "id2": {"local": 48},
                        },
                        {
                            "ph": "X",
                            "name": f"bound{rank}",
                            "pid": 0,
                            "ts": 7,
                            "dur": 1,
                            "bind_id": 10,
                            "flow_out": True,
                        },
                    ],
                },
            )
        )
    events = {
        event["name"]: event
        for event in read(merge_traces(paths, tmp_path / "all.json.gz"))["traceEvents"]
        if event["ph"] != "M"
    }
    for rank in range(2):
        assert events[f"start{rank}"]["id"] == events[f"end{rank}"]["id"]
        assert events[f"bound{rank}"]["bind_id"] == events[f"start{rank}"]["id"]
        assert events[f"global{rank}"]["id2"] == events[f"global-end{rank}"]["id2"]
        assert events[f"local{rank}"]["id2"] == events[f"local-end{rank}"]["id2"]
    assert events["start0"]["id"] != events["start1"]["id"]
    assert events["global0"]["id2"] != events["global1"]["id2"]
    assert events["local0"]["id2"] != events["local1"]["id2"]


def test_cli_groups_cases_ignores_merged_outputs_and_retains_inputs(tmp_path, capsys):
    raw = []
    trace = {
        "baseTimeNanoseconds": 0,
        "traceEvents": [{"ph": "X", "name": "step", "pid": 0, "ts": 1, "dur": 2}],
    }
    for case, rank in (
        ("decode-ctx16-b4", 1),
        ("decode-ctx16-b4", 0),
        ("decode-ctx16-b4-repeat002", 1),
        ("decode-ctx16-b4-repeat002", 0),
        ("decode-ctx32-b2", 0),
        ("prefill-ctx16-b4", 0),
        ("prefill-ctx16-b4", 1),
    ):
        raw.append(write(tmp_path / f"{case}-rank{rank:03}.json.gz", trace))
    write(tmp_path / "decode-ctx16-b4-all-ranks.json.gz", {"old": "replace"})
    (tmp_path / "other.json").write_text("unrelated")
    main([str(tmp_path)])
    assert len(capsys.readouterr().out.splitlines()) == 4
    repeated = read(tmp_path / "decode-ctx16-b4-repeat002-all-ranks.json.gz")
    assert [source["rank"] for source in repeated["resinTraceMerge"]["sources"]] == [
        0,
        1,
    ]
    assert (
        len(
            read(tmp_path / "decode-ctx16-b4-all-ranks.json.gz")["resinTraceMerge"][
                "sources"
            ]
        )
        == 2
    )
    assert (
        len(
            read(tmp_path / "decode-ctx32-b2-all-ranks.json.gz")["resinTraceMerge"][
                "sources"
            ]
        )
        == 1
    )
    assert (
        len(
            read(tmp_path / "prefill-ctx16-b4-all-ranks.json.gz")["resinTraceMerge"][
                "sources"
            ]
        )
        == 2
    )
    assert all(path.exists() for path in raw)


def test_rejects_mixed_bases_and_overwriting_raw_trace(tmp_path):
    first = write(
        tmp_path / "first.json", {"baseTimeNanoseconds": 0, "traceEvents": []}
    )
    second = write(tmp_path / "second.json", {"traceEvents": []})
    with pytest.raises(ValueError, match="mixed missing/present"):
        merge_traces([first, second], tmp_path / "all.json.gz")
    with pytest.raises(ValueError, match="must not overwrite"):
        merge_traces([first], first)
