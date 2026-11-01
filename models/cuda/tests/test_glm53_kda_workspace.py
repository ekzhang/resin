"""Reuse only owned KDA gather scratch after speculative acceptance is known."""

from types import SimpleNamespace

import pytest
import torch

from resin_models_cuda.glm5_3_flash import kernels
from resin_models_cuda.glm5_3_flash.speculative import SpeculativeTape

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def record(tape, layer):
    # Ownership checks are independent of projection values and run before CUDA.
    values = [torch.zeros(1)] * 8
    tape.record(layer, *values, -5.0)


def make_tape(initials, arena, *, reuse=True):
    layers = {
        index: SimpleNamespace(recurrent=value, convolution=torch.zeros(1))
        for index, value in enumerate(initials)
    }
    batch = SimpleNamespace(
        layers=layers, device_cache=SimpleNamespace(recurrent_states=arena)
    )
    return SpeculativeTape.attach(batch, 4, reuse_initial_state=reuse), layers


def test_owned_tape_rejects_engine_pages_before_recording():
    arena = torch.zeros(2, 3, 2, 8, 8)
    tape, layers = make_tape([arena[0]], arena)
    with pytest.raises(ValueError, match="Engine cache pages"):
        record(tape, layers[0])
    assert not tape.records


def test_owned_tape_requires_an_identifiable_engine_arena():
    with pytest.raises(ValueError, match="Engine cache arena"):
        SpeculativeTape.attach(SimpleNamespace(layers={}), 4, reuse_initial_state=True)


def test_owned_tape_accepts_disjoint_views_but_rejects_overlap():
    gathered = torch.zeros(3, 3, 2, 8, 8)
    arena = torch.zeros_like(gathered)
    tape, layers = make_tape([gathered[0], gathered[1], gathered[0]], arena)
    record(tape, layers[0])
    record(tape, layers[1])
    with pytest.raises(ValueError, match="disjoint layer states"):
        record(tape, layers[2])
    assert len(tape.records) == 2


@pytest.mark.parametrize("invalid", ["dtype", "strides"])
def test_owned_tape_rejects_unsupported_scratch(invalid):
    initial = torch.zeros(3, 2, 8, 8)
    initial = initial.bfloat16() if invalid == "dtype" else initial.transpose(-1, -2)
    tape, layers = make_tape([initial], torch.zeros_like(initial))
    with pytest.raises(ValueError, match="packed FP32"):
        record(tape, layers[0])


def inplace(data, bound, *, owned=True, out=None):
    from resin_models_cuda.glm5_3_flash.kda_state_cuda import accepted_kda_state

    return accepted_kda_state(
        data["k"],
        data["v"],
        data["g"],
        data["beta"],
        data["a"],
        data["bias"],
        data["initial"],
        data["offsets"],
        data["counts"],
        bound,
        out=data["initial"] if out is None else out,
        initial_is_owned=owned,
    )


@CUDA
@pytest.mark.parametrize("width", range(1, 9))
@pytest.mark.parametrize("bound", [None, -5.0])
@torch.inference_mode()
def test_reused_state_matches_full_reference_and_independent_output(width, bound):
    from test_glm53_kda_state import candidate, fixture, original

    data = fixture(9, width)
    before = data["initial"].clone()
    expected = original(data, bound)
    independent = candidate(data, bound)
    torch.testing.assert_close(data["initial"], before, rtol=0, atol=0)
    actual = inplace(data, bound)
    assert actual is data["initial"]
    assert independent.data_ptr() != actual.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, independent, rtol=0, atol=0)
    # The last sequence has an empty interval and keeps its initial state.
    torch.testing.assert_close(actual[-1], before[-1], rtol=0, atol=0)


@CUDA
@torch.inference_mode()
def test_low_level_reuse_requires_explicit_ownership_and_exact_tensor():
    from test_glm53_kda_state import fixture

    data = fixture(3, 4)
    with pytest.raises(ValueError, match="explicitly owned"):
        inplace(data, -5.0, owned=False)
    with pytest.raises(ValueError, match="explicitly owned"):
        inplace(data, -5.0, out=data["initial"].view_as(data["initial"]))


@CUDA
@pytest.mark.parametrize("width", [1, 4, 8])
@torch.inference_mode()
def test_verification_can_omit_final_state_without_changing_core(width):
    from test_glm53_kda_state import fixture

    data = fixture(5, width)
    before = data["initial"].clone()
    cache = SimpleNamespace(
        recurrent=data["initial"], cu_seqlens=data["offsets"], cu_seqlens_cpu=None
    )
    args = (
        data["q"],
        data["k"],
        data["v"],
        data["g"],
        data["beta"],
        data["a"],
        data["bias"],
        -5.0,
        cache,
    )
    expected, final = kernels.kda(*args)
    actual, omitted = kernels.kda(*args, output_final_state=False)
    assert final is not None and omitted is None
    # Tokens outside every sequence interval have no defined FLA output.
    active_tokens = int(data["offsets"][-1].item())
    torch.testing.assert_close(
        actual[:active_tokens], expected[:active_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(data["initial"], before, rtol=0, atol=0)


@CUDA
@torch.inference_mode()
def test_real_tape_reuses_disjoint_gathered_layers_and_preserves_pages():
    from test_glm53_kda_state import fixture, original

    rows, width, heads, dim = 5, 4, 4, 128
    data = [fixture(rows, width, seed=seed) for seed in (194, 711)]
    gathered = torch.stack([entry["initial"] for entry in data])
    arena = torch.full_like(gathered, 123)
    tape, layers = make_tape(list(gathered), arena)
    tape.batch.context_lengths = torch.arange(rows, device="cuda") + 8191
    counts = data[0]["counts"]
    hidden = torch.randn(rows * width, 16, device="cuda", dtype=torch.bfloat16)
    expected = []
    for index, entry in enumerate(data):
        entry["initial"] = gathered[index]
        expected.append(original(entry, -5.0))
        layer = layers[index]
        layer.cu_seqlens = entry["offsets"]
        history = torch.randn(
            rows, 3, heads * dim, 3, device="cuda", dtype=torch.bfloat16
        )
        layer.convolution = history
        raw = torch.randn(
            rows * width, 3, heads * dim, device="cuda", dtype=torch.bfloat16
        )
        tape.record(
            layer,
            raw,
            entry["q"],
            entry["k"],
            entry["v"],
            entry["g"],
            entry["beta"],
            entry["a"],
            entry["bias"],
            -5.0,
        )
    tape.commit(counts, hidden)
    for index, final in enumerate(expected):
        assert layers[index].recurrent.data_ptr() == gathered[index].data_ptr()
        torch.testing.assert_close(layers[index].recurrent, final, rtol=0, atol=0)
        assert layers[index].speculative_tape is None
    torch.testing.assert_close(arena, torch.full_like(arena, 123), rtol=0, atol=0)


@CUDA
@torch.inference_mode()
def test_graph_replay_refreshes_owned_gather_before_each_inplace_rollback():
    from test_glm53_cache_cuda import _fixture, _reference_gather
    from test_glm53_kda_state import fixture, original

    from resin_models_cuda.glm5_3_flash.cache import CacheGeometry
    from resin_models_cuda.glm5_3_flash.cache_cuda import gather_state_layer_major

    rows, width = 5, 4
    geometry = CacheGeometry(
        ("linear_attention", "deepseek_sparse_attention", "linear_attention"),
        hidden_size=32,
        latent_dim=8,
        index_dim=4,
        heads=4,
        head_dim=128,
        conv_history=3,
        state_tokens=64,
    )
    arena, cache = _fixture("cuda", rows, geometry=geometry)
    arena.recurrent_states.normal_()
    contexts = torch.full((rows,), 65, dtype=torch.int32, device="cuda")
    data = [fixture(rows, width, seed=seed) for seed in (194, 711)]

    def run():
        gathered, _, _ = gather_state_layer_major(cache, contexts)
        for index, entry in enumerate(data):
            inplace({**entry, "initial": gathered[index]}, -5.0)
        return gathered

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    assert (
        result.untyped_storage().data_ptr()
        != arena.storage.untyped_storage().data_ptr()
    )
    for turn, count in enumerate((4, 1, 3, 2)):
        arena.recurrent_states.normal_()
        contexts.fill_(64 if turn % 2 else 65)
        # Swap physical pages as well as context/acceptance values between replays.
        group = cache.groups[1]
        group.slab_ids.copy_(group.slab_ids.flip(0))
        before = arena.recurrent_states.clone()
        expected_initial, _, _ = _reference_gather(cache, contexts)
        expected = []
        for index, entry in enumerate(data):
            entry["counts"].fill_(count)
            for key in ("k", "v", "g", "beta"):
                entry[key].normal_()
            expected.append(
                original(
                    {**entry, "initial": expected_initial[:, index].contiguous()}, -5.0
                )
            )
        graph.replay()
        for index, final in enumerate(expected):
            torch.testing.assert_close(result[index], final, rtol=0, atol=0)
        torch.testing.assert_close(arena.recurrent_states, before, rtol=0, atol=0)
