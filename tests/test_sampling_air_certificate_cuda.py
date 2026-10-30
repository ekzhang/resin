"""Exact cutoff and replay checks for the AIR first-pass optimization.

Uses only the existing public air_top_p_cutoff API. Expected values come from
constant groups with a nucleus far from the boundary between groups; no changed
kernel is used as an oracle.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from resin.backend.cuda.sampling_native import air_top_p_cutoff

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


@dataclass
class Fixture:
    weights: torch.Tensor
    ps: torch.Tensor
    active: torch.Tensor
    mass: torch.Tensor
    guard: torch.Tensor
    expected: torch.Tensor


def _guard_row(fixture, row):
    """Independent exact full/top40 mass for the fixture's two-valued row."""
    vocab = fixture.weights.shape[1]
    values = fixture.weights[row].double()
    full = F.pad(values, (0, (-vocab) % 1024)).view(-1, 1024).sum(-1)
    kept = F.pad(values * (fixture.weights[row] >= 1.0),
                 (0, (-vocab) % 1024)).view(-1, 1024).sum(-1)
    fixture.mass[row, :, 0] = full
    fixture.mass[row, :, 1] = kept
    fixture.guard[row] = True
    # The supplied guard is a real inclusive top40 set, not invented scratch.
    assert kept.sum().item() < (fixture.ps[row].item() - 0.001) * full.sum().item()


def _fixture(vocab, kind):
    rows = 18  # Exercises the >=16-row native firstpass shape gate.
    parts = (vocab + 1023) // 1024
    fixture = Fixture(
        torch.zeros(rows, vocab, dtype=torch.float32, device="cuda"),
        torch.full((rows,), 0.83, device="cuda"),
        torch.ones(rows, dtype=torch.bool, device="cuda"),
        torch.full((rows, parts, 2), torch.nan, dtype=torch.float64, device="cuda"),
        torch.zeros(rows, dtype=torch.bool, device="cuda"),
        torch.zeros(rows, device="cuda"),
    )
    w, p, active, expected = fixture.weights, fixture.ps, fixture.active, fixture.expected
    if kind == "constant_groups":
        for row, exponent in enumerate((-126, -40, -3, 0, 3, 20)):
            value = 2.0**exponent
            w[row].fill_(value)
            expected[row] = value
        # Zero padding contributes no mass; all retained nonzero values tie.
        w[6, :37] = 0.125
        p[6] = 0.71
        expected[6] = 0.125
        # FP32 decimal constants are deliberately non-powers of two.
        # Total=3*.61+7*.23≈3.44; p=.73 crosses into the .23 group.
        w[7, :3] = 0.61
        w[7, 3:10] = 0.23
        p[7] = 0.73
        expected[7] = 0.23
        # Total=4+13*.3125=8.0625; p=.8 crosses into .3125.
        w[8, :4] = 1.0
        w[8, 4:17] = 0.3125
        p[8] = 0.8
        expected[8] = 0.3125
        w[9].fill_(1.0)
        active[9] = False
        # Inclusive top40 mass is40; tail mass>=20.11 for both vocab sizes.
        w[10].fill_(0.01)
        w[10, :40] = 1.0
        p[10] = 0.95
        _guard_row(fixture, 10)
        # Unnormalized finite power-of-two weights above1 are supported.
        # Total=3*8+7*2=38; p=.75 crosses into the2 group.
        w[11, :3] = 8.0
        w[11, 3:10] = 2.0
        p[11] = 0.75
        expected[11] = 2.0
        active[12] = False  # An inactive row need not have positive mass.
        for row, source in enumerate((0, 7, 11, 9, 10), start=13):
            w[row].copy_(w[source])
            p[row].copy_(p[source])
            active[row].copy_(active[source])
            expected[row].copy_(expected[source])
            fixture.guard[row].copy_(fixture.guard[source])
            fixture.mass[row].copy_(fixture.mass[source])
    elif kind == "non_power":
        w.fill_(0.3125)
        expected.fill_(0.3125)
        p.copy_(torch.tensor([0.11, 0.31, 0.73, 0.95, 0.999, 0.5] * 3, device="cuda"))
        # Certified→guard-skipped and then guard-skipped→certified transitions.
        w[10].fill_(0.01)
        w[10, :40] = 1.0
        p[10] = 0.95
        expected[10] = 0.0
        _guard_row(fixture, 10)
    elif kind == "inactive":
        active.fill_(False)
        p.fill_(0.42)
    elif kind == "power":
        for row in range(rows):
            value = 2.0**((-126, -40, -3, 0, 3, 20)[(row + 3) % 6])
            w[row].fill_(value)
            expected[row] = value
        p.copy_(torch.tensor([0.19, 0.41, 0.72, 0.96, 0.999, 0.6] * 3, device="cuda"))
    else:
        raise ValueError(kind)
    return fixture


def _call(fixture, *, out=None):
    return air_top_p_cutoff(
        fixture.weights, fixture.ps, fixture.active,
        guard_mass=fixture.mass, guard_active=fixture.guard, out=out,
    )


def _copy_fixture(destination, source):
    destination.weights.copy_(source.weights)
    destination.ps.copy_(source.ps)
    destination.active.copy_(source.active)
    destination.mass.copy_(source.mass)
    destination.guard.copy_(source.guard)


def _assert_exact(actual, fixture):
    torch.testing.assert_close(actual, fixture.expected, rtol=0, atol=0)


def _capture(fixture, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(fixture)
        _call(fixture)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = _call(fixture)
    return graph, output


@pytest.mark.parametrize("vocab", [2051, 262144])
@torch.inference_mode()
def test_air_certificate_exact_constant_group_cutoffs(vocab):
    for kind in ("constant_groups", "non_power", "inactive", "power"):
        fixture = _fixture(vocab, kind)
        _assert_exact(_call(fixture), fixture)


@pytest.mark.parametrize(
    "rows,vocab", [(15, 131073), (16, 131071), (16, 131072), (17, 131073)]
)
@torch.inference_mode()
def test_air_first_pass_shape_gate_and_odd_tail(rows, vocab):
    weights = torch.ones((rows, vocab), dtype=torch.float32, device="cuda")
    weights[1::4].fill_(0.3125)
    weights[2::4].zero_()
    weights[2::4, -1] = 2.0
    weights[3::4].fill_(0.3125)
    weights[3::4, -1] = 1.0
    ps = torch.full((rows,), 0.75, device="cuda")
    active = torch.ones(rows, dtype=torch.bool, device="cuda")
    expected = torch.ones(rows, device="cuda")
    expected[1::4] = 0.3125
    expected[2::4] = 2.0
    expected[3::4] = 0.3125
    inputs = tuple(value.clone() for value in (weights, ps, active))
    storage = torch.full((rows, 3), -99.0, device="cuda")
    out = storage[:, 1]
    assert air_top_p_cutoff(weights, ps, active, out=out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    untouched = storage[:, (0, 2)]
    torch.testing.assert_close(
        untouched, torch.full_like(untouched, -99.0), rtol=0, atol=0
    )
    for actual, original in zip((weights, ps, active), inputs, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("vocab", [2051, 262144])
@torch.inference_mode()
def test_air_certificate_marker_reset_across_80_changed_replays(vocab):
    kinds = ("constant_groups", "non_power", "inactive", "power")
    fixtures = [_fixture(vocab, kind) for kind in kinds]
    live = _fixture(vocab, "constant_groups")
    graph, output = _capture(live, torch.cuda.Stream())
    for iteration in range(80):
        source = fixtures[iteration % len(fixtures)]
        _copy_fixture(live, source)
        graph.replay()
        _assert_exact(output, source)


@torch.inference_mode()
def test_air_certificate_independent_graph_workspaces_on_two_streams():
    vocab = 262144
    fixtures = [_fixture(vocab, kind) for kind in
                ("constant_groups", "non_power", "inactive", "power")]
    live = [_fixture(vocab, "constant_groups"), _fixture(vocab, "power")]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    captured = [_capture(fixture, stream) for fixture, stream in zip(live, streams)]
    torch.cuda.synchronize()
    for iteration in range(12):
        selected = [fixtures[iteration % 4], fixtures[(iteration + 2) % 4]]
        for destination, source, stream, (graph, _) in zip(live, selected, streams, captured):
            with torch.cuda.stream(stream):
                _copy_fixture(destination, source)
                graph.replay()
        torch.cuda.synchronize()
        for (_, output), source in zip(captured, selected):
            _assert_exact(output, source)
