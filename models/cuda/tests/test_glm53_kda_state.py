"""Accepted-prefix CUDA state updates versus the independent full FLA recurrence."""

from types import SimpleNamespace

import pytest
import torch

from resin_models_cuda.glm5_3_flash import kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def fixture(rows, width, *, heads=4, dim=128, seed=194):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", generator=generator).to(dtype)

    shape = (rows * width, heads, dim)
    q, k, v, g = [normal(shape) for _ in range(4)]
    beta = normal((rows * width, heads + 3))[:, 1 : heads + 1]
    initial = normal((rows, heads, dim, dim), torch.float32)
    a, bias = normal((heads,), torch.float32), normal((heads * dim,), torch.float32)
    counts = (torch.arange(rows, device="cuda") % width + 1).repeat_interleave(2)[::2]
    offsets = torch.arange(rows + 1, device="cuda", dtype=torch.int32) * width
    offsets[-1] = offsets[-2]  # Unwritten row must preserve its owned state.
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        a=a,
        bias=bias,
        initial=initial,
        offsets=offsets,
        counts=counts,
        width=width,
    )


def original(data, bound):
    accepted = (
        torch.arange(data["width"], device="cuda")[None] < data["counts"][:, None]
    ).flatten()
    masked_g = data["g"].masked_fill(~accepted[:, None, None], -torch.inf)
    masked_b = data["beta"].masked_fill(~accepted[:, None], -torch.inf)
    cache = SimpleNamespace(
        recurrent=data["initial"], cu_seqlens=data["offsets"], cu_seqlens_cpu=None
    )
    _, final = kernels.kda(
        data["q"],
        data["k"],
        data["v"],
        masked_g,
        masked_b,
        data["a"],
        data["bias"],
        bound,
        cache,
    )
    return final


def candidate(data, bound):
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
    )


@pytest.mark.parametrize("rows", [8, 64])
@torch.inference_mode()
def test_real_geometry_graph_changed_inputs(rows):
    data = fixture(rows, 4, heads=64)
    for _ in range(2):
        candidate(data, -5.0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = candidate(data, -5.0)
    for count in (4, 1, 3, 2):
        data["counts"].fill_(count)
        for name in ("k", "v", "g", "beta", "initial"):
            data[name].normal_()
        graph.replay()
        torch.testing.assert_close(output, original(data, -5.0), rtol=0, atol=0)


@torch.inference_mode()
def test_new_width_rows_and_counts_reuse_compilation():
    from resin_models_cuda.glm5_3_flash.kda_state_cuda import _accepted_state

    candidate(fixture(2, 1), -5.0)
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    before = set(_accepted_state.device_caches[device][0])
    for rows, width in ((3, 2), (7, 3), (8, 4), (9, 5), (16, 8)):
        data = fixture(rows, width)
        torch.testing.assert_close(
            candidate(data, -5.0), original(data, -5.0), rtol=0, atol=0
        )
    torch.cuda.synchronize()
    assert set(_accepted_state.device_caches[device][0]) == before
