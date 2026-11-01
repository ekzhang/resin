"""Independent arithmetic and graph contracts for the shared CDF sampler."""

import gc
from dataclasses import fields, replace
from importlib import import_module

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


@pytest.fixture(scope="module")
def fused():
    return import_module("resin.backend.cuda.logits_sampling")


def _inputs(rows, vocab, *, dtype=torch.float32, strided=False):
    generator = torch.Generator(device="cuda").manual_seed(271)
    if strided:
        logits = torch.randn(
            rows * 2, vocab * 2 + 3, device="cuda", dtype=dtype, generator=generator
        )[::2, 1 : 2 * vocab + 1 : 2]
    else:
        logits = torch.randn(
            rows, vocab, device="cuda", dtype=dtype, generator=generator
        )
    row = torch.arange(rows, device="cuda", dtype=torch.int64)
    values = (
        0.5 + (row % 4).float() * 0.5,
        (1 << 62) + 19 + row * (1 << 32),
        (1 << 40) + 32 + row * 32,
        row % 3 != 0,
    )
    if strided:
        values = tuple(
            torch.stack((value, torch.zeros_like(value)), -1).flatten()[::2]
            for value in values
        )
    return (logits, *values)


def _philox(seed, offset, tokens):
    """Independent Random123 Philox4x32-10 counter arithmetic in NumPy."""
    mask = np.uint64(0xFFFFFFFF)
    c0 = np.asarray(tokens, dtype=np.uint64)
    c1 = np.zeros_like(c0)
    c2 = np.full_like(c0, offset & 0xFFFFFFFF)
    c3 = np.full_like(c0, (offset >> 32) & 0xFFFFFFFF)
    k0, k1 = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF
    for _ in range(10):
        a = np.uint64(0xD2511F53) * c0
        b = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = (
            (b >> np.uint64(32)) ^ c1 ^ np.uint64(k0),
            b & mask,
            (a >> np.uint64(32)) ^ c3 ^ np.uint64(k1),
            a & mask,
        )
        k0 = (k0 + 0x9E3779B9) & 0xFFFFFFFF
        k1 = (k1 + 0xBB67AE85) & 0xFFFFFFFF
    return c0, c1, c2, c3


def _reference(data):
    """Independent FP64 CDF with full 32-bit request-local Philox uniform."""
    logits, temperatures, seeds, offsets, mask = [value.cpu() for value in data]
    result = logits.argmax(-1)
    margins = np.full(len(result), np.inf)
    for row in range(len(result)):
        values = logits[row].double().numpy()
        temperature = float(temperatures[row])
        if (
            not mask[row]
            or not 0 < temperature < np.inf
            or np.isnan(values).any()
            or np.isposinf(values).any()
            or not np.isfinite(values).any()
        ):
            continue
        weights = np.exp((values - values.max()) / temperature)
        cumulative = np.cumsum(weights)
        random = int(
            _philox(int(seeds[row]), int(offsets[row]), np.array([0], dtype=np.uint64))[
                0
            ][0]
        )
        uniform = (random + 0.5) * 2**-32
        target = uniform * cumulative[-1]
        selected = int(np.searchsorted(cumulative, target, side="right"))
        result[row] = selected
        before = cumulative[selected - 1] if selected else 0.0
        margins[row] = (
            min(target - before, cumulative[selected] - target) / cumulative[-1]
        )
    return result, margins


def _check_reference(actual, expected, margins):
    # FP32 tile masses/CDF scans can move a draw within the measured rounding
    # band. Keep exact equality outside that band, and require meaningful
    # coverage rather than permitting a whole fixture to be excluded.
    stable = torch.from_numpy(margins > 2e-7)
    assert int(stable.sum()) >= max(1, len(stable) // 2), margins
    torch.testing.assert_close(actual.cpu()[stable], expected[stable], atol=0, rtol=0)
    assert torch.all((actual >= 0) & (actual < 2**31 - 1))


def test_independent_philox_matches_published_zero_counter_vector():
    values = _philox(0, 0, np.array([0], dtype=np.uint64))
    assert tuple(int(value[0]) for value in values) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )


def test_request_uniform_preserves_all_32_philox_bits(fused):
    import triton
    import triton.language as tl

    from resin.backend.cuda.logits_sampling import _request_uniform

    @triton.jit
    def uniforms(Seeds, Offsets, Output):
        row = tl.program_id(0)
        seed = tl.load(Seeds + row).to(tl.uint64)
        offset = tl.load(Offsets + row).to(tl.uint64)
        tl.store(Output + row, _request_uniform(seed, offset))

    rows = 64
    seeds = torch.arange(rows, device="cuda", dtype=torch.int64) + 2**62
    offsets = torch.arange(rows, device="cuda", dtype=torch.int64) * 32 - 2**63
    output = torch.empty(rows, device="cuda", dtype=torch.float64)
    uniforms[(rows,)](seeds, offsets, output)
    random = np.array(
        [
            int(_philox(int(seed), int(offset), np.array([0], dtype=np.uint64))[0][0])
            for seed, offset in zip(seeds.cpu(), offsets.cpu(), strict=True)
        ],
        dtype=np.uint64,
    )
    expected = (random.astype(np.float64) + 0.5) * 2**-32
    np.testing.assert_array_equal(output.cpu().numpy(), expected)
    old_precision = ((random >> np.uint64(9)).astype(np.float64) + 0.5) * 2**-23
    assert np.all(expected != old_precision)


def test_private_capture_scratch_survives_gc_and_signed_offset_wrap(fused):
    data = _inputs(8, 2051, strided=True)
    data[-1].fill_(True)
    data[3][0] = 2**63 - 17
    fused.sample_logits(*data)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fused.sample_logits(*data)
    gc.collect()
    for _ in range(3):
        data[3].add_(32)
        graph.replay()
        expected, margins = _reference(data)
        _check_reference(output, expected, margins)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_greedy_ties_nan_infinities_and_strided_inputs(fused, dtype):
    data = _inputs(8, 2051, dtype=dtype, strided=True)
    logits, _, _, _, mask = data
    mask.zero_()
    logits[0].fill_(-torch.inf)
    logits[1].fill_(1)
    logits[2, [4, 2049]] = torch.inf
    logits[3, [7, 1027]] = torch.nan
    logits[4].fill_(-5)
    logits[4, [1023, 1024, 2048]] = 10
    logits[5].zero_()
    logits[5, 0] = -0.0
    logits[6].fill_(torch.inf)
    logits[7, -1] = torch.nan
    expected = logits.argmax(-1)
    output = torch.empty(8, device="cuda", dtype=torch.int64)
    actual = fused.sample_logits(*data, out=output)
    assert actual is output
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("vocab", [1, 17, 1023, 1024, 1025, 2051])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_unfiltered_draw_matches_independent_philox_oracle(fused, vocab, dtype):
    data = _inputs(11, vocab, dtype=dtype, strided=True)
    data[0][:, ::7] = -torch.inf
    if vocab == 1:
        data[0].fill_(0.25)
    expected, margins = _reference(data)
    actual = fused.sample_logits(*data)
    _check_reference(actual, expected, margins)


@torch.inference_mode()
def test_stochastic_invalid_rows_retain_greedy_fallback(fused):
    data = _inputs(5, 2051)
    logits = data[0]
    data[-1].fill_(True)
    logits[0].fill_(-torch.inf)
    logits[1, [2, 2000]] = torch.inf
    logits[2, [3, 1999]] = torch.nan
    logits[3].fill_(torch.inf)
    logits[4].fill_(torch.nan)
    actual = fused.sample_logits(*data)
    torch.testing.assert_close(actual, logits.argmax(-1), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab", [2051, 201088])
@torch.inference_mode()
def test_request_reordering_cancellation_and_padding_preserve_stream(
    fused, dtype, vocab
):
    data = _inputs(17, vocab, dtype=dtype)
    data[-1].fill_(True)
    expected = fused.sample_logits(*data)
    order = torch.tensor([16, 0, 8, 2, 4, 12, 6], device="cuda")
    selected = tuple(value[order] for value in data)
    torch.testing.assert_close(
        fused.sample_logits(*selected),
        expected[order],
        atol=0,
        rtol=0,
    )
    padded = _inputs(32, vocab, dtype=dtype)
    for destination, source in zip(padded, data, strict=True):
        destination[:17].copy_(source)
    padded[-1][17:].zero_()
    torch.testing.assert_close(
        fused.sample_logits(*padded)[:17],
        expected,
        atol=0,
        rtol=0,
    )
    for row in (0, 7, 16):
        one = tuple(value[row : row + 1] for value in data)
        torch.testing.assert_close(
            fused.sample_logits(*one),
            expected[row : row + 1],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_graph_replay_reads_changed_values_and_reuses_owned_workspace(fused, dtype):
    data = _inputs(8, 154880, dtype=dtype, strided=True)
    workspace = fused._prepare_workspace(data[0])
    output = torch.empty(8, device="cuda", dtype=torch.int64)
    source = [value.clone() for value in data]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused.sample_logits(*data, workspace=workspace, out=output)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fused.sample_logits(*data, workspace=workspace, out=output)
    for old, new in zip(source, data, strict=True):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    for mode in (False, True, None):
        data[0].normal_()
        data[1].copy_(torch.linspace(0.13, 3.7, 8, device="cuda"))
        data[2].add_(1 << 40)
        data[3].add_(1 << 39)
        data[-1].copy_(
            torch.arange(8, device="cuda") % 2 == 0
            if mode is None
            else torch.full((8,), mode, device="cuda")
        )
        expected = fused.sample_logits(*data)
        for _ in range(3):
            graph.replay()
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
    # Scratch owns separate storage instead of borrowing mutable inputs.
    assert all(
        getattr(workspace, field.name).untyped_storage().data_ptr()
        not in {value.untyped_storage().data_ptr() for value in data}
        for field in fields(workspace)
        if isinstance(getattr(workspace, field.name), torch.Tensor)
    )


@torch.inference_mode()
def test_full_seed_and_counter_words_affect_draw(fused):
    data = _inputs(64, 17)
    data[0].zero_()
    data[1].fill_(1)
    data[-1].fill_(True)
    data[2].fill_(7)
    data[3].copy_(torch.arange(64, device="cuda") * 32)
    baseline = fused.sample_logits(*data)
    data[2].add_(1 << 62)
    assert not torch.equal(baseline, fused.sample_logits(*data))
    data[2].fill_(7)
    data[3].add_(1 << 40)
    assert not torch.equal(baseline, fused.sample_logits(*data))
    data[2].fill_(-(1 << 63) + 7)
    data[3].copy_(-(1 << 63) + torch.arange(64, device="cuda") * 32)
    expected, margins = _reference(data)
    _check_reference(fused.sample_logits(*data), expected, margins)


@pytest.mark.parametrize("temperature", [0.2, 0.7, 1.0, 2.5])
@torch.inference_mode()
def test_unfiltered_distribution_matches_fp64_softmax(fused, temperature):
    values = torch.tensor([-torch.inf, -2.6, -1.7, -0.9, -0.15, 0.4, 1.1, 1.9])
    count = 16384
    data = _inputs(count, len(values))
    data[0].copy_(values.cuda().expand(count, -1))
    data[1].fill_(temperature)
    data[2].fill_((1 << 62) + 41)
    data[-1].fill_(True)
    actual = fused.sample_logits(*data).cpu()
    expected = (values.double() / temperature).softmax(0)
    observed = torch.bincount(actual, minlength=len(values)).double() / count
    assert torch.all(observed[expected == 0] == 0)
    bounds = 6 * (expected * (1 - expected) / count).sqrt() + 2 / count
    assert torch.all((observed - expected).abs() <= bounds), (observed, expected)


@torch.inference_mode()
def test_large_common_logit_shift_preserves_draw(fused):
    data = _inputs(128, 2051)
    data[-1].fill_(True)
    data[1].fill_(8)
    # Integer multiples of eight survive this FP32 translation exactly.
    data[0].copy_(torch.randint(-8, 9, data[0].shape, device="cuda").float() * 8)
    original = fused.sample_logits(*data)
    data[0].add_(1e8)
    shifted = fused.sample_logits(*data)
    torch.testing.assert_close(shifted, original, atol=0, rtol=0)
    expected, margins = _reference(data)
    _check_reference(shifted, expected, margins)

    # Center before exponentiation so a large shared logit offset cannot
    # overflow the masses or bias selection among equal logits.
    equal = _inputs(16384, 7)
    equal[0].fill_(1e8)
    equal[1].fill_(1)
    equal[-1].fill_(True)
    sampled = fused.sample_logits(*equal).cpu()
    observed = torch.bincount(sampled, minlength=7).double() / 16384
    probability = 1 / 7
    bound = 6 * (probability * (1 - probability) / 16384) ** 0.5 + 2 / 16384
    assert torch.all((observed - probability).abs() <= bound), observed


@torch.inference_mode()
def test_request_values_rows_and_strides_do_not_add_jit_variants(fused):
    def entries():
        device = torch.cuda.current_device()
        return {
            name: set(value.device_caches[device][0])
            for name, value in vars(fused).items()
            if hasattr(value, "device_caches")
        }

    initial = _inputs(8, 2051)
    fused.sample_logits(*initial)
    torch.cuda.synchronize()
    baseline = entries()
    assert baseline and any(baseline.values())
    for rows, strided in ((1, False), (2, True), (16, True), (64, False)):
        data = _inputs(rows, 2051, strided=strided)
        data[1].fill_(0.137)
        data[2].add_(1 << 40)
        data[3].add_(1 << 44)
        for mode in (False, True):
            data[-1].fill_(mode)
            fused.sample_logits(*data)

    # CUDA allocations are usually aligned. Shift every input and each scratch
    # or output allocation to prove pointer alignment does not create variants.
    def shifted(value):
        storage = torch.empty(value.numel() + 1, device=value.device, dtype=value.dtype)
        result = storage[1:].view(value.shape)
        result.copy_(value)
        return result

    shifted_data = tuple(shifted(value) for value in initial)
    expected, margins = _reference(initial)
    workspace = fused._prepare_workspace(initial[0])
    workspace = replace(
        workspace,
        **{
            field.name: shifted(getattr(workspace, field.name))
            for field in fields(workspace)
            if isinstance(getattr(workspace, field.name), torch.Tensor)
        },
    )
    actual = fused.sample_logits(*shifted_data, workspace=workspace)
    _check_reference(actual, expected, margins)
    torch.cuda.synchronize()
    assert entries() == baseline


@torch.inference_mode()
def test_extreme_valid_temperatures_and_invalid_temperature_fallback(fused):
    data = _inputs(8, 17)
    data[0].copy_(torch.arange(17, device="cuda").expand(8, -1))
    data[0][:, ::7] = -torch.inf
    data[-1].fill_(True)
    data[1].copy_(
        torch.tensor(
            [
                torch.finfo(torch.float32).tiny,
                1e-40,
                1e-30,
                0.0,
                -1.0,
                torch.nan,
                torch.inf,
                -torch.inf,
            ],
            device="cuda",
        )
    )
    actual = fused.sample_logits(*data)
    torch.testing.assert_close(actual, data[0].argmax(-1), atol=0, rtol=0)


@torch.inference_mode()
def test_empty_batch_and_workspace_validation(fused):
    empty = _inputs(0, 17)
    result = fused.sample_logits(*empty)
    assert result.shape == (0,) and result.dtype == torch.int64
    data = _inputs(2, 17)
    workspace = fused._prepare_workspace(data[0])
    with pytest.raises(ValueError, match="geometry"):
        fused.sample_logits(*data, workspace=replace(workspace, shape=(2, 18)))
    with pytest.raises(ValueError):
        fused.sample_logits(data[0].double(), *data[1:])
    with pytest.raises(ValueError):
        fused.sample_logits(data[0], data[1].double(), *data[2:])
    with pytest.raises(ValueError):
        fused.sample_logits(
            *data,
            out=torch.empty(4, device="cuda", dtype=torch.int64)[::2],
        )
    with pytest.raises(ValueError):
        fused.sample_logits(
            *data,
            out=torch.empty(2, device="cuda", dtype=torch.int32),
        )


@pytest.mark.parametrize("vocab", [154880, 201088])
@pytest.mark.parametrize("capture", [False, True])
@torch.inference_mode()
def test_real_vocabulary_tiled_greedy_partials_and_final_winner(fused, vocab, capture):
    data = _inputs(64, vocab)
    data[-1].zero_()
    workspace = fused._prepare_workspace(data[0])
    actual = fused.sample_logits(*data, workspace=workspace)
    if capture:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused.sample_logits(*data, workspace=workspace)
        graph.replay()
    expected = data[0].argmax(-1)
    # Check each stage against an independent reduction, so a vocabulary-edge
    # regression identifies whether local selection or final reduction failed.
    parts = (vocab + 1023) // 1024
    padded = torch.nn.functional.pad(
        data[0], (0, parts * 1024 - vocab), value=-torch.inf
    )
    maxima, ids = padded.reshape(64, parts, 1024).max(-1)
    ids += torch.arange(parts, device="cuda")[None, :] * 1024
    torch.testing.assert_close(workspace.maxima, maxima, atol=0, rtol=0)
    torch.testing.assert_close(workspace.greedy.long(), ids, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("vocab", [154880, 201088])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_real_vocabulary_mixed_draw_matches_independent_oracle(fused, vocab, dtype):
    data = _inputs(8, vocab, dtype=dtype)
    expected, margins = _reference(data)
    actual = fused.sample_logits(*data)
    _check_reference(actual, expected, margins)


@pytest.mark.parametrize("stochastic", [False, True])
@torch.inference_mode()
def test_partial_tail_mask_preserves_the_first_128_tokens_in_every_tile(
    fused, stochastic
):
    # A vocabulary tail must not suppress the same lanes in complete tiles.
    # Every tile has a deliberate maximum at a different early lane per row.
    vocab = 201088
    data = _inputs(8, vocab)
    logits, _, _, _, mask = data
    mask.fill_(stochastic)
    logits.fill_(-100)
    offsets = torch.tensor([0, 1, 31, 63, 64, 127, 128, 255], device="cuda")
    rows = torch.arange(8, device="cuda")
    parts = (vocab + 1023) // 1024
    ids = torch.arange(parts, device="cuda")[None, :] * 1024 + offsets[:, None]
    logits[rows[:, None], ids] = 10 + rows[:, None].float()
    workspace = fused._prepare_workspace(logits)
    actual = fused.sample_logits(*data, workspace=workspace)
    expected_max = (10 + rows[:, None].float()).expand(-1, parts)
    torch.testing.assert_close(workspace.maxima, expected_max, atol=0, rtol=0)
    torch.testing.assert_close(workspace.greedy.long(), ids, atol=0, rtol=0)
    expected, margins = _reference(data)
    _check_reference(actual, expected, margins)


@torch.inference_mode()
def test_opposite_sign_finite_logits_preserve_high_temperature_distribution(fused):
    # The FP32 subtraction -2e38 - 2e38 overflows, although division by 3e38
    # makes the correct normalized difference finite. Include a cross-tile gap.
    data = _inputs(64, 2051)
    data[0].fill_(-2e38)
    data[0][:, :1024] = 2e38
    data[1].fill_(3e38)
    data[-1].fill_(True)
    expected, margins = _reference(data)
    assert torch.any(expected >= 1024)
    actual = fused.sample_logits(*data)
    _check_reference(actual, expected, margins)

    count = 16384
    distribution = _inputs(count, 2)
    values = torch.tensor([2e38, -2e38], device="cuda")
    distribution[0].copy_(values.expand(count, -1))
    distribution[1].fill_(3e38)
    distribution[-1].fill_(True)
    sampled = fused.sample_logits(*distribution).cpu()
    probabilities = (values.double().cpu() / float(distribution[1][0])).softmax(0)
    observed = torch.bincount(sampled, minlength=2).double() / count
    bounds = 6 * (probabilities * (1 - probabilities) / count).sqrt() + 2 / count
    assert torch.all((observed - probabilities).abs() <= bounds), observed


@torch.inference_mode()
def test_output_and_scratch_may_not_alias_inputs_or_each_other(fused):
    data = _inputs(8, 2051)
    workspace = fused._prepare_workspace(data[0])
    for output in (data[2], data[3]):
        with pytest.raises(ValueError):
            fused.sample_logits(*data, out=output)
    for overlapping in (
        replace(workspace, output=data[2]),
        replace(workspace, maxima=workspace.mass),
        replace(workspace, greedy=workspace.mass.view(torch.int32)),
        replace(
            workspace,
            maxima=data[0]
            .flatten()[: workspace.maxima.numel()]
            .view_as(workspace.maxima),
        ),
    ):
        with pytest.raises(ValueError):
            fused.sample_logits(*data, workspace=overlapping)
