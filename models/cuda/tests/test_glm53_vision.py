"""Pinned upstream vision oracle and arbitrary prefill slice placement."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from resin.models.vision import span_slices
from resin.protocols import MediaTokenSpan
from resin_models_cuda.glm5_3_flash.vision import (
    GlmVisionConfig,
    GlmVisionModel,
)

FIXTURE = Path(__file__).parent / "fixtures/glm53_vision_hf.npz"


@pytest.fixture(params=[torch.float32, torch.bfloat16])
def vision(request):
    with np.load(FIXTURE, allow_pickle=False) as source:
        data = dict(source)
    metadata = json.loads(str(data.pop("metadata")))
    assert (
        metadata["transformers_revision"] == "50e35cc0ca18ec19b80653690de553e6decf74f5"
    )
    pixels, grids, expected = (
        torch.from_numpy(data.pop(k)) for k in ("pixels", "grids", "output")
    )
    expected_bf16 = torch.from_numpy(data.pop("output_bf16"))
    dtype = request.param
    if dtype == torch.bfloat16:
        expected = expected_bf16
    weights = {k: torch.from_numpy(v).to(dtype) for k, v in data.items()}
    return (
        GlmVisionModel(GlmVisionConfig.from_dict(metadata["config"]), weights),
        pixels.to(dtype),
        grids.tolist(),
        expected,
    )


def test_complete_vision_matches_pinned_hf_with_saturated_swiglu(vision):
    model, pixels, grids, expected = vision
    actual = model.encode(pixels, grids)
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=2e-6 if actual.dtype == torch.bfloat16 else 2e-8,
        rtol=3e-2 if actual.dtype == torch.bfloat16 else 3e-5,
    )
    # Packing images together cannot change either image's attention result.
    single = torch.cat((model(pixels[:16], grids[:1]), model(pixels[16:], grids[1:])))
    torch.testing.assert_close(actual, single)


def test_image_placement_matches_every_partial_prefill_slice_and_prefix(vision):
    model, pixels, grids, _ = vision
    features = model(pixels, grids)
    positions = (2, 3, 4, 5, 9, 10)
    original = torch.full((14, features.shape[-1]), -100.0, dtype=features.dtype)
    expected = original.clone()
    expected[list(positions)] = features
    for start in range(15):
        for end in range(start, 15):
            chunk = original[start:end].clone()
            for destination, source in span_slices(
                (MediaTokenSpan(2, 4), MediaTokenSpan(9, 2)), start, end
            ):
                chunk[destination] = features[source]
            torch.testing.assert_close(chunk, expected[start:end])


def test_vision_rejects_video_and_inconsistent_pixel_rows(vision):
    model, pixels, _, _ = vision
    for grid in ([[2, 2, 6]], [[1, 3, 8]], [[1, 4, 8]], []):
        with pytest.raises(ValueError, match="GLM"):
            model(pixels, grid)


def test_vision_weight_loading_requires_every_checkpoint_tensor(vision):
    model, *_ = vision
    weights = dict(model.state_dict())
    weights.pop("blocks.0.attn.q_norm.weight")
    with pytest.raises(RuntimeError, match="q_norm"):
        GlmVisionModel(model.config, weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA vision qualification")
def test_cuda_bf16_vision_oracle_and_changed_image_graph_replay():
    """All vision operators must capture and observe new pixels on replay."""
    with np.load(FIXTURE, allow_pickle=False) as source:
        data = dict(source)
    metadata = json.loads(str(data.pop("metadata")))
    pixels = torch.from_numpy(data.pop("pixels")).to(
        device="cuda", dtype=torch.bfloat16
    )
    grids = data.pop("grids").tolist()
    data.pop("output")
    expected = torch.from_numpy(data.pop("output_bf16")).cuda()
    weights = {
        name: torch.from_numpy(value).to(device="cuda", dtype=torch.bfloat16)
        for name, value in data.items()
    }
    model = GlmVisionModel(GlmVisionConfig.from_dict(metadata["config"]), weights)

    # Warm up cuDNN/SDPA outside capture; there must be no lazy initialization
    # or device-to-host geometry read inside the captured encoder.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            eager = model.encode(pixels, grids)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(eager.float(), expected, atol=5e-6, rtol=3e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        output = model.encode(pixels, grids)
    graph.replay()
    torch.testing.assert_close(output, eager, atol=2e-6, rtol=1e-2)
    first = output.clone()
    # Preserve geometry and addresses while changing every image's content.
    pixels.mul_(0.5).add_(0.4)
    with torch.inference_mode():
        changed_eager = model.encode(pixels, grids)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, changed_eager, atol=2e-6, rtol=1e-2)
    assert not torch.equal(output, first), (
        "vision graph replay ignored changed image pixels"
    )
