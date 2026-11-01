"""MXFP4 adapter boundaries use real tensors and preserve FP32 routed sums."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from resin_models_cuda.gpt_oss.flashinfer import (
    FlashInferSM90MXFP4Kernel,
    MXFP4Config,
    MXFP4Weights,
)


def _config(**changes):
    return replace(
        MXFP4Config(
            top_k=4,
            num_experts=4,
            hidden_size=32,
            intermediate_size=32,
            swiglu_limit=7.0,
        ),
        **changes,
    )


def test_preparation_preserves_paired_rows_and_neutral_padding(monkeypatch):
    kernel = FlashInferSM90MXFP4Kernel()
    monkeypatch.setattr(kernel, "_validate_sm90_tensor", lambda *args: None)
    monkeypatch.setattr(
        kernel,
        "_get_fused_moe",
        lambda: SimpleNamespace(
            interleave_moe_weights_for_sm90_mixed_gemm=lambda value, kind: value,
            interleave_moe_scales_for_sm90_mixed_gemm=lambda value, group: value,
        ),
    )
    rows = torch.arange(64, dtype=torch.uint8)[None, :, None]
    source = MXFP4Weights(
        gate_up=rows.expand(4, 64, 16).clone(),
        down=torch.ones(4, 32, 16, dtype=torch.uint8),
        gate_up_scales=rows.expand(4, 64, 1).clone(),
        down_scales=torch.full((4, 32, 1), 120, dtype=torch.uint8),
        gate_up_bias=torch.arange(64).float().expand(4, 64).clone(),
        down_bias=torch.ones(4, 32),
    )
    prepared = kernel.prepare_weights(source, _config())
    assert prepared.padded_hidden_size == prepared.padded_intermediate_size == 128
    expected = torch.zeros(4, 256, 64, dtype=torch.uint8)
    expected[:, :32, :16] = source.gate_up[:, 1::2]
    expected[:, 128:160, :16] = source.gate_up[:, ::2]
    torch.testing.assert_close(prepared.fc1_weights, expected)
    scales = torch.full((4, 256, 4), 127, dtype=torch.uint8)
    scales[:, :32, :1] = source.gate_up_scales[:, 1::2]
    scales[:, 128:160, :1] = source.gate_up_scales[:, ::2]
    torch.testing.assert_close(prepared.fc1_scales.view(torch.uint8), scales)
    bias = torch.zeros(4, 256, dtype=torch.bfloat16)
    bias[:, :32] = source.gate_up_bias[:, 1::2]
    bias[:, 128:160] = source.gate_up_bias[:, ::2]
    torch.testing.assert_close(prepared.fc1_bias, bias)
    torch.testing.assert_close(prepared.fc2_weights[:, :32, :16], source.down)
    assert not prepared.fc2_weights[:, 32:].any()
    assert not prepared.fc2_weights[:, :, 16:].any()
    assert not prepared.fc2_bias[:, 32:].any()


@pytest.mark.parametrize("change", [{"top_k": 2}, {"num_experts": 129}])
def test_adapter_rejects_unsupported_geometry(change):
    with pytest.raises(ValueError):
        FlashInferSM90MXFP4Kernel().validate(replace(_config(), **change))


def test_adapter_rejects_unsupported_device_and_api(monkeypatch):
    kernel = FlashInferSM90MXFP4Kernel()
    with pytest.raises(ValueError, match="CUDA"):
        kernel.validate_device(torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    with pytest.raises(ValueError, match="SM90"):
        kernel.validate_device(torch.device("cuda"))
    monkeypatch.setattr(kernel, "_fused_moe", SimpleNamespace(), raising=False)
    with pytest.raises(RuntimeError, match="pinned SM90 MXFP4 API"):
        kernel._get_fused_moe()


def test_inactive_route_sentinel_reaches_cutlass_without_rerouting(monkeypatch):
    import torch

    import resin_models_cuda.gpt_oss.flashinfer as adapter

    config = _config(num_experts=4, hidden_size=32, intermediate_size=32)
    kernel = adapter.FlashInferSM90MXFP4Kernel()
    monkeypatch.setattr(kernel, "_validate_sm90_tensor", lambda *args: None)
    indices = torch.tensor([[4, 0, 1, 2], [0, 4, 4, 4]], dtype=torch.int64)
    weights = torch.tensor([[0, 0.5, 0.25, 0.25], [1, 0, 0, 0]], dtype=torch.float32)
    hidden = torch.ones(2, 32, dtype=torch.bfloat16)

    def cutlass(**kwargs):
        actual_ids = kwargs["token_selected_experts"]
        actual_weights = kwargs["token_final_scales"]
        assert actual_ids.dtype == torch.int32
        assert torch.equal(actual_ids, indices.int())
        assert torch.equal(actual_weights, weights)
        active = actual_ids < config.num_experts
        scales = ((actual_ids + 1) * actual_weights * active).sum(1)
        kwargs["output"].copy_(hidden * scales[:, None])

    monkeypatch.setattr(
        kernel,
        "_get_fused_moe",
        lambda: SimpleNamespace(
            ActivationType=SimpleNamespace(Swiglu=object()),
            cutlass_fused_moe=cutlass,
        ),
    )
    prepared = adapter.PreparedFlashInferMXFP4Weights(
        config=config,
        fc1_weights=torch.empty(1),
        fc2_weights=torch.empty(1),
        fc1_scales=torch.empty(1, dtype=torch.int32),
        fc2_scales=torch.empty(1, dtype=torch.int32),
        fc1_bias=torch.zeros(4, 64, dtype=torch.bfloat16),
        fc2_bias=torch.zeros(4, 32, dtype=torch.bfloat16),
        swiglu_alpha=torch.full((4,), 1.702),
        swiglu_beta=torch.ones(4),
        swiglu_limit=torch.full((4,), config.swiglu_limit),
        padded_hidden_size=32,
        padded_intermediate_size=32,
        fc2_zero_bias=torch.zeros(4, 32, dtype=torch.bfloat16),
    )
    output = kernel.forward(hidden, indices, weights, prepared)
    torch.testing.assert_close(
        output[:, 0], torch.tensor([1.75, 1.0], dtype=torch.bfloat16)
    )


@pytest.mark.parametrize("limit", [2, 5, 8])
def test_fp32_partials_preserve_cancellation_and_chunk_expanded_routes(
    monkeypatch, limit
):
    import torch

    import resin_models_cuda.gpt_oss.flashinfer as adapter

    config = _config(num_experts=4, hidden_size=32, intermediate_size=32)
    kernel = adapter.FlashInferSM90MXFP4Kernel()
    monkeypatch.setattr(kernel, "max_tokens", limit)
    packed = torch.empty(1, dtype=torch.uint8)
    bias = torch.zeros(4, 32, dtype=torch.bfloat16)
    bias[0] = 0.75
    bias[1] = 1.25
    prepared = adapter.PreparedFlashInferMXFP4Weights(
        config=config,
        fc1_weights=packed,
        fc2_weights=packed,
        fc1_scales=packed,
        fc2_scales=packed,
        fc1_bias=torch.zeros(4, 64, dtype=torch.bfloat16),
        fc2_bias=bias,
        swiglu_alpha=torch.full((4,), 1.702),
        swiglu_beta=torch.ones(4),
        swiglu_limit=torch.full((4,), config.swiglu_limit),
        padded_hidden_size=32,
        padded_intermediate_size=32,
        fc2_zero_bias=torch.zeros_like(bias),
    )
    calls = []

    def unweighted(hidden, indices, routing_weights, weights):
        assert weights.config is config
        assert len(hidden) <= limit
        assert weights.fc1_weights is packed and weights.fc2_weights is packed
        assert weights.fc1_bias is prepared.fc1_bias
        assert not weights.fc2_bias.any()
        assert indices[:, 1:].eq(config.num_experts).all()
        active = indices[:, 0] < config.num_experts
        torch.testing.assert_close(routing_weights[:, 0], active.float())
        assert not routing_weights[:, 1:].any()
        calls.append(len(hidden))
        values = torch.tensor([1000, -1000, 4, -8, float("nan")], dtype=torch.bfloat16)
        return values[indices[:, 0].long(), None].expand(-1, 32)

    monkeypatch.setattr(kernel, "forward", unweighted)

    def dynamic_shape(*args, **kwargs):
        # pytest's _WithException wrapper obscures fail's callable signature.
        pytest.fail("partial execution must not compact routes dynamically")  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(torch.Tensor, "nonzero", dynamic_shape)
    hidden = torch.ones(4, 32, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 4, 4, 4], [1, 4, 4, 4], [0, 1, 2, 4], [0, 1, 2, 3]])
    scales = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0.25, 0.5, 0.25, 0],
            [0.25, 0.5, 0.125, 0.125],
        ]
    )
    output = kernel.forward_partial(hidden, ids, scales, prepared)
    assert output.dtype == torch.float32
    torch.testing.assert_close(
        output[:, 0], torch.tensor([1000.75, -998.75, -248.1875, -249.6875])
    )
    values = torch.tensor([1000.75, -998.75, 4, -8, 0])
    torch.testing.assert_close(
        output[:, 0], (values[ids] * scales).sum(1), rtol=0, atol=0
    )
    expected_calls = [min(limit, 16 - start) for start in range(0, 16, limit)]
    assert calls == expected_calls
    assert output[0, 0] + output[1, 0] == 2
    assert output[:2, 0].bfloat16().float().sum() == 0
    assert prepared.fc2_bias is bias and bias[0, 0] == 0.75
    empty = kernel.forward_partial(
        hidden[:0],
        ids[:0],
        scales[:0],
        prepared,
    )
    assert empty.shape == (0, 32) and empty.dtype == torch.float32
    assert calls == expected_calls

    # Replaying the same row shape with different active routes keeps all
    # kernel shapes and chunk counts stable, including entirely idle input.
    idle = kernel.forward_partial(
        hidden,
        ids,
        torch.zeros_like(scales),
        prepared,
    )
    assert idle.dtype == torch.float32
    assert torch.equal(idle, torch.zeros_like(idle))
    assert calls == expected_calls * 2
