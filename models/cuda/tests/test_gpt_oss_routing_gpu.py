"""Both GPT-OSS expert counts use the same fused routing path."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


@pytest.mark.parametrize("experts", [32, 128])
@pytest.mark.parametrize("rows", [1, 33, 1024])
def test_top4_matches_selected_bf16_softmax_and_replays_changed_routes(experts, rows):
    from resin_models_cuda.gpt_oss.cuda import _route
    from resin_models_cuda.gpt_oss.model import TopKRouter

    router = cast(TopKRouter, SimpleNamespace(top_k=4, balanced_routing=None))

    torch.manual_seed(71)
    # Unique representable BF16 scores make expert-ID equality meaningful;
    # PyTorch does not specify which indices it returns for ties.
    logits = torch.rand(rows, experts, device="cuda").argsort(-1).bfloat16() / 64
    logits *= (2.0 ** (torch.arange(rows, device="cuda") % 8))[:, None]

    def check(indices, weights):
        selected, expected_indices = logits.topk(4, dim=-1)
        torch.testing.assert_close(indices, expected_indices.int(), atol=0, rtol=0)
        torch.testing.assert_close(
            weights, selected.softmax(-1).float(), atol=0, rtol=0
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            indices, weights = _route(router, logits)
    stream.synchronize()
    check(indices, weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        indices, weights = _route(router, logits)
    torch.cuda.current_stream().wait_stream(stream)
    for shift in (1, 7, 17):
        logits.copy_(logits.roll(shift, dims=-1))
        graph.replay()
        check(indices, weights)

    logits.zero_()
    graph.replay()
    torch.testing.assert_close(weights, torch.full_like(weights, 0.25), atol=0, rtol=0)
    first_experts = torch.arange(4, device=indices.device, dtype=torch.int32)
    torch.testing.assert_close(
        indices, first_experts.expand_as(indices), atol=0, rtol=0
    )
