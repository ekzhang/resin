"""Host-only tests for external scheduler completion accounting."""

import json
from types import SimpleNamespace

import pytest

from scripts.bench_sglang import DecodeWindow, capacity_preflight, save_completed_case


def case():
    return dict(
        id="b2-r0",
        request_ids=["a", "b"],
        context_length=32768,
        warmup_steps=2,
        measure_steps=3,
        draft_tokens=7,
        gpu_count=2,
    )


def rows(tokens, verify, accepted=None, **overrides):
    accepted = accepted or [value - verify - 1 for value in tokens]
    return [
        dict(
            id=rid,
            tokens=value,
            verify=verify,
            accepted=correct,
            cached_tokens=0,
            finished=False,
            **overrides,
        )
        for rid, value, correct in zip(("a", "b"), tokens, accepted, strict=True)
    ]


def test_continuous_window_counts_actual_bursts_and_host_gap():
    window = DecodeWindow(case())
    for timestamp, tokens, verify in (
        (1.0, [3, 4], 1),
        (2.0, [7, 9], 2),
        (2.1, [10, 11], 3),
        (7.1, [11, 19], 4),
        (7.2, [17, 22], 5),
    ):
        window.observe(timestamp, rows(tokens, verify))
    result = window.result()
    assert result["window_seconds"] == pytest.approx(5.2)
    assert result["step_seconds"] == pytest.approx([0.1, 5.0, 0.1])
    assert result["committed_tokens"] == 23
    assert result["per_request_committed_tokens"] == [10, 13]
    assert result["committed_token_bursts"] == [[3, 2], [1, 8], [6, 3]]
    assert result["per_request_verify_passes"] == [3, 3]
    assert result["accepted_drafts"] == 17
    assert result["proposed_drafts"] == 42
    assert result["context_start"] == [32775, 32777]
    assert result["context_end"] == [32785, 32790]
    assert result["per_user_tokens_per_second"] == pytest.approx(23 / 5.2 / 2)
    assert result["per_request_amortized_tpot_seconds"] == pytest.approx([0.52, 0.4])
    # Already completed windows exclude retirement and all later work.
    window.observe(90, [dict(rows([18, 23], 6)[0], finished=True)])
    assert window.result() == result


def test_waits_for_full_cohort_before_warmup():
    window = DecodeWindow(case())
    window.observe(0, rows([1, 1], 0)[:1], prefill=True)
    window.observe(1, rows([2, 2], 1)[:1])
    assert not window.observations and not window.complete
    window.observe(2, rows([3, 3], 1))
    assert len(window.observations) == 1


def test_unformed_cohort_is_excluded_when_its_first_request_retires():
    window = DecodeWindow(case())
    window.observe(1, rows([3, 3], 1)[:1])
    window.observe(2, [dict(rows([4, 4], 2)[0], finished=True)])
    result = window.result()
    assert result["status"] == "admission_excluded"
    assert result["maximum_observed_decode_cohort"] == 1
    assert not result["observations"]
    assert "window_seconds" not in result


@pytest.mark.parametrize("fault", ["partial", "retract", "finished", "cached", "stale"])
def test_rejects_changed_cohort_and_stale_state(fault):
    window = DecodeWindow(case())
    window.observe(1, rows([3, 3], 1))
    next_rows = rows([4, 4], 2)
    if fault == "partial":
        next_rows.pop()
    elif fault == "finished":
        next_rows[0]["finished"] = True
    elif fault == "cached":
        next_rows[0]["cached_tokens"] = 128
    elif fault == "stale":
        next_rows[1]["tokens"] = 3
    window.observe(2, next_rows, prefill=fault == "retract")
    assert window.result()["status"] == "invalid"


def test_incomplete_window_is_not_a_point():
    with pytest.raises(ValueError, match="incomplete"):
        DecodeWindow(case()).result()


def test_actual_configured_pool_capacity_is_reported():
    capacity = dict(
        full_tokens=1246976,
        window_tokens={"target": 124672, "draft": 124672},
        max_running_requests=64,
        max_request_length=33899,
    )
    kwargs = dict(
        page_size=128,
        physical_peak=33907,
        prefill_chunk=4096,
        window_tokens=1040,
        output_budget=1130,
    )
    point = case() | dict(request_ids=list(map(str, range(32))))
    result = capacity_preflight(point, capacity, **kwargs)
    assert result["status"] == "fits"
    assert result["full_tokens_required"] == 1085440
    point["request_ids"] = list(map(str, range(64)))
    result = capacity_preflight(point, capacity, **kwargs)
    assert result["status"] == "capacity_exceeded"
    assert result["full_tokens_required"] == 2170880
    assert len(result["reasons"]) == 1


def test_worker_and_scheduler_each_reserve_an_admission_anchor():
    capacity = dict(full_tokens=1246976, window_tokens={}, max_running_requests=32)
    kwargs = dict(
        page_size=128,
        physical_peak=33907,
        prefill_chunk=4096,
        window_tokens=1040,
        output_budget=1130,
    )
    # Pinned worker reports context_len - 1; scheduler then reserves one more.
    rejected = capacity_preflight(
        case(), capacity | {"max_request_length": 33898}, **kwargs
    )
    assert rejected["status"] == "capacity_exceeded"
    assert rejected["reasons"] == [
        "output budget requires 1130, admission permits 1129"
    ]
    accepted = capacity_preflight(
        case(), capacity | {"max_request_length": 33899}, **kwargs
    )
    assert accepted["status"] == "fits"
    assert accepted["full_tokens_required"] == 2 * 33920


@pytest.mark.parametrize("final_count", [1129, 1130])
def test_final_outputs_are_saved_before_length_validation(tmp_path, final_count):
    record = dict(
        status="passed",
        request_ids=["a"],
        final_outputs=[
            dict(text="retained", meta_info=dict(completion_tokens=final_count))
        ],
    )
    path = tmp_path / "case.json"
    if final_count == 1130:
        save_completed_case(path, record, 1130)
    else:
        with pytest.raises(
            ValueError, match=r"completion_tokens=\[1129\], expected=\[1130\]"
        ):
            save_completed_case(path, record, 1130)
    assert json.loads(path.read_text()) == record


@pytest.mark.parametrize("diagnostic", [False, True])
def test_delayed_profile_requires_full_diagnostic_window(
    tmp_path, monkeypatch, diagnostic
):
    from scripts import bench_sglang

    requests = []
    profiler = SimpleNamespace(
        get_forward_ct=lambda: 913,
        _profile=lambda request: (
            requests.append(request) or SimpleNamespace(success=True)
        ),
    )

    class Scheduler:
        profiler_manager = profiler
        max_total_num_tokens = 100000
        max_running_requests = 2
        max_req_len = 34000
        max_req_input_len = 33999

        def get_init_info(self):
            return {}

        def window_token_capacities(self):
            return {}

    class Processor:
        def process_batch_result_decode(self, batch, result):
            pass

        def process_batch_result_prefill(self, batch, result):
            pass

    modules = {
        "sglang.srt.managers.scheduler": SimpleNamespace(Scheduler=Scheduler),
        "sglang.srt.managers.scheduler_components.batch_result_processor": SimpleNamespace(
            SchedulerBatchResultProcessor=Processor
        ),
        "sglang.srt.runtime_context": SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(tp_rank=1)
        ),
        "sglang.srt.managers.io_struct": SimpleNamespace(ProfileReq=SimpleNamespace),
    }
    monkeypatch.setattr(bench_sglang.importlib, "import_module", modules.__getitem__)
    plan = tmp_path / "plan.json"
    point = case() | {"profile_after_window": True}
    plan.write_text(
        json.dumps(
            dict(
                cases=[] if diagnostic else [point],
                trace_cases=[point] if diagnostic else [],
                output=str(tmp_path),
            )
        )
    )
    monkeypatch.setenv("RESIN_SGLANG_BENCH_PLAN", str(plan))
    bench_sglang.install_plugin()
    Scheduler().get_init_info()
    processor = Processor()
    result = SimpleNamespace(can_run_cuda_graph=True)

    def batch(step, ids=("a", "b")):
        return SimpleNamespace(
            forward_iter=step + 900,
            reqs=[
                SimpleNamespace(
                    rid=rid,
                    output_ids=[0] * (step + 1),
                    spec_verify_ct=step,
                    spec_num_correct_drafts=0,
                    cached_tokens=0,
                    finished=lambda: False,
                )
                for rid in ids
            ],
        )

    processor.process_batch_result_prefill(batch(0), result)
    processor.process_batch_result_decode(batch(1, ("a",)), result)
    for step in range(1, 5):
        processor.process_batch_result_decode(batch(step), result)
    assert not requests
    processor.process_batch_result_decode(batch(5), result)
    processor.process_batch_result_decode(batch(6), result)
    if diagnostic:
        assert len(requests) == 1
        assert requests[0].start_step == 914
        assert requests[0].num_steps == 3
        start = json.loads((tmp_path / "b2-r0-profile-start-rank-1.json").read_text())
        assert start["completed_decode_intervals"] == 5
        assert start["armed_forward"] == 913
        assert start["last_committed_forward"] == 905
        assert start["committed_context_at_arm"] == [32774, 32774]
    else:
        assert not requests
