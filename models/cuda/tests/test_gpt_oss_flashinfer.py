from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from resin_models_cuda.gpt_oss.flashinfer import MXFP4Config, MXFP4Weights


class _DType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


class _Device:
    def __init__(self, kind: str = "cuda", index: int = 0) -> None:
        self.type = kind
        self.index = index

    def __eq__(self, other) -> bool:
        return isinstance(other, _Device) and (self.type, self.index) == (
            other.type,
            other.index,
        )

    def __repr__(self) -> str:
        return f"{self.type}:{self.index}"


class _FakeView:
    def __init__(self, parent, key) -> None:
        self.parent = parent
        self.key = key
        self.dtype = parent.dtype
        self.device = parent.device
        self.shape = tuple(
            self._slice_size(size, selection)
            for size, selection in zip(parent.shape, key, strict=True)
        )

    @staticmethod
    def _slice_size(size, selection):
        if isinstance(selection, int):
            return None
        start, stop, step = selection.indices(size)
        return len(range(start, stop, step))

    def copy_(self, source):
        self.parent.copies.append((self.key, source))
        return self

    def contiguous(self):
        return _FakeTensor(self.shape, self.dtype, device=self.device)


class _FakeTensor:
    def __init__(
        self,
        shape,
        dtype,
        *,
        device: _Device | None = None,
        fill_value=None,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or _Device()
        self.fill_value = fill_value
        self.copies = []

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def contiguous(self):
        return self

    def to(self, *, device=None, dtype=None):
        return _FakeTensor(
            self.shape,
            self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            fill_value=self.fill_value,
        )

    def view(self, dtype):
        if self.dtype.name == "uint8" and dtype.name == "int32":
            if self.shape[-1] % 4:
                raise ValueError("last dimension is not int32-aligned")
            shape = (*self.shape[:-1], self.shape[-1] // 4)
        else:
            shape = self.shape
        return _FakeTensor(shape, dtype, device=self.device)

    def __getitem__(self, key):
        return _FakeView(self, key)


class _FakeCuda:
    def __init__(self, capability=(9, 0)) -> None:
        self.capability = capability

    def get_device_capability(self, device):
        del device
        return self.capability


class _FakeTorch:
    uint8 = _DType("uint8")
    int32 = _DType("int32")
    int64 = _DType("int64")
    float16 = _DType("float16")
    bfloat16 = _DType("bfloat16")
    float32 = _DType("float32")

    def __init__(self, capability=(9, 0)) -> None:
        self.cuda = _FakeCuda(capability)
        self.full_calls = []
        self.empty_calls = []

    def full(self, shape, *, fill_value, dtype, device):
        tensor = _FakeTensor(
            shape,
            dtype,
            device=device,
            fill_value=fill_value,
        )
        self.full_calls.append(tensor)
        return tensor

    def empty(self, shape, *, dtype, device):
        tensor = _FakeTensor(shape, dtype, device=device)
        self.empty_calls.append(tensor)
        return tensor


class _ActivationType:
    Swiglu = "swiglu-enum"


class _FakeFusedMoE:
    ActivationType = _ActivationType

    def __init__(self) -> None:
        self.weight_calls = []
        self.scale_calls = []
        self.forward_calls = []

    def interleave_moe_weights_for_sm90_mixed_gemm(self, tensor, quant_type):
        self.weight_calls.append((tensor, quant_type))
        return _FakeTensor(tensor.shape, tensor.dtype, device=tensor.device)

    def interleave_moe_scales_for_sm90_mixed_gemm(self, tensor, group_size):
        self.scale_calls.append((tensor, group_size))
        experts, rows, groups = tensor.shape
        return _FakeTensor(
            (experts, rows // 64, groups // 4, 16, 16),
            tensor.dtype,
            device=tensor.device,
        )

    def cutlass_fused_moe(self, **kwargs):
        self.forward_calls.append(kwargs)
        return [kwargs["output"], "auxiliary-output"]


def _config(**changes) -> MXFP4Config:
    values: dict[str, Any] = {
        "top_k": 4,
        "num_experts": 128,
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "swiglu_limit": 7.0,
    }
    values.update(changes)
    return MXFP4Config(**values)


def _tensor(runtime, shape, dtype=None, *, device=None):
    return _FakeTensor(
        shape,
        runtime.uint8 if dtype is None else dtype,
        device=device,
    )


def _weights(runtime, config=None):
    config = _config() if config is None else config
    experts = config.num_experts
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    return MXFP4Weights(
        gate_up=_tensor(runtime, (experts, 2 * intermediate, hidden // 2)),
        down=_tensor(runtime, (experts, hidden, intermediate // 2)),
        gate_up_scales=_tensor(runtime, (experts, 2 * intermediate, hidden // 32)),
        down_scales=_tensor(runtime, (experts, hidden, intermediate // 32)),
        gate_up_bias=_tensor(runtime, (experts, 2 * intermediate), runtime.float32),
        down_bias=_tensor(runtime, (experts, hidden), runtime.float32),
    )


class TestFlashInferMXFP4Adapter:
    def _kernel(self, monkeypatch, *, capability=(9, 0)):
        import resin_models_cuda.gpt_oss.flashinfer as flashinfer
        from resin_models_cuda.gpt_oss.flashinfer import (
            FlashInferSM90MXFP4Kernel,
        )

        torch = _FakeTorch(capability)
        fused_moe = _FakeFusedMoE()
        monkeypatch.setattr(flashinfer, "torch", torch)
        kernel = FlashInferSM90MXFP4Kernel()
        monkeypatch.setattr(kernel, "_fused_moe", fused_moe, raising=False)
        return kernel, torch, fused_moe

    def test_validate_accepts_supported_geometry_and_token_limits(self, monkeypatch):
        kernel, _, _ = self._kernel(monkeypatch)
        kernel.validate(_config(), num_tokens=8192)
        kernel.validate(
            _config(num_experts=4, hidden_size=32, intermediate_size=32), num_tokens=1
        )

    @pytest.mark.parametrize(
        "changes",
        [
            {"top_k": 1},
            {"top_k": 8},
            {"num_experts": 3},
            {"num_experts": 129},
            {"hidden_size": 0},
            {"hidden_size": 31},
            {"intermediate_size": -32},
            {"intermediate_size": 33},
            {"swiglu_limit": 0.0},
            {"swiglu_limit": -1.0},
            {"swiglu_limit": float("nan")},
            {"swiglu_limit": float("inf")},
        ],
    )
    def test_validate_rejects_unsupported_geometry(self, monkeypatch, changes):
        kernel, _, _ = self._kernel(monkeypatch)
        with pytest.raises(ValueError):
            kernel.validate(_config(**changes))

    @pytest.mark.parametrize("num_tokens", [0, 8193])
    def test_validate_rejects_unsupported_token_count(self, monkeypatch, num_tokens):
        kernel, _, _ = self._kernel(monkeypatch)
        with pytest.raises(ValueError):
            kernel.validate(_config(), num_tokens=num_tokens)

    def test_prepare_pads_2880_and_uses_exact_sm90_interleavers(
        self, monkeypatch
    ) -> None:
        kernel, torch, fused_moe = self._kernel(monkeypatch)
        prepared = kernel.prepare_weights(_weights(torch), _config())

        assert [(call[0].shape, call[1]) for call in fused_moe.weight_calls] == [
            ((128, 5888, 1472), "fp4"),
            ((128, 2944, 1472), "fp4"),
        ]
        fc1_before_interleave = fused_moe.weight_calls[0][0]
        assert len(fc1_before_interleave.copies) == 2
        assert [copy[0][1] for copy in fc1_before_interleave.copies] == [
            slice(0, 2880),
            slice(2944, 5824),
        ]
        assert [copy[1].key[1] for copy in fc1_before_interleave.copies] == [
            slice(1, None, 2),
            slice(0, None, 2),
        ]
        assert [(call[0].shape, call[1]) for call in fused_moe.scale_calls] == [
            ((128, 5888, 92), 32),
            ((128, 2944, 92), 32),
        ]
        fc1_scales_before_interleave = fused_moe.scale_calls[0][0]
        assert [copy[1].key[1] for copy in fc1_scales_before_interleave.copies] == [
            slice(1, None, 2),
            slice(0, None, 2),
        ]
        scale_padding = [
            tensor
            for tensor in torch.full_calls
            if tensor.fill_value == 127 and len(tensor.shape) == 3
        ]
        assert [tensor.shape for tensor in scale_padding] == [
            (128, 5888, 92),
            (128, 2944, 92),
        ]
        assert prepared.fc1_bias.shape == (128, 5888)
        assert prepared.fc2_bias.shape == (128, 2944)
        assert [copy[1].key[1] for copy in prepared.fc1_bias.copies] == [
            slice(1, None, 2),
            slice(0, None, 2),
        ]
        assert prepared.fc1_bias.dtype is torch.bfloat16
        assert prepared.fc2_bias.dtype is torch.bfloat16
        assert prepared.swiglu_alpha.shape == (128,)
        assert prepared.swiglu_alpha.fill_value == 1.702
        assert prepared.swiglu_beta.fill_value == 1.0
        assert prepared.swiglu_limit.fill_value == 7.0
        assert prepared.swiglu_alpha.dtype is torch.float32
        assert prepared.padded_hidden_size == 2944
        assert prepared.padded_intermediate_size == 2944

    def test_forward_uses_exact_low_level_api_types_and_scale_order(
        self, monkeypatch
    ) -> None:
        kernel, torch, fused_moe = self._kernel(monkeypatch)
        config = _config()
        prepared = kernel.prepare_weights(_weights(torch), config)
        hidden = _tensor(torch, (16, 2880), torch.bfloat16)
        indices = _tensor(torch, (16, 4), torch.int64)
        routing_weights = _tensor(torch, (16, 4), torch.bfloat16)

        output = kernel.forward(hidden, indices, routing_weights, prepared)
        assert output.shape == (16, 2880)
        assert output.dtype is torch.bfloat16
        assert len(fused_moe.forward_calls) == 1
        call = fused_moe.forward_calls[0]
        assert call["input"].shape == (16, 2944)
        assert call["token_selected_experts"].dtype is torch.int32
        assert call["token_final_scales"].dtype is torch.float32
        assert call["quant_scales"] == [prepared.fc1_scales, prepared.fc2_scales]
        assert prepared.fc1_scales.dtype is torch.int32
        assert prepared.fc2_scales.dtype is torch.int32
        assert prepared.fc1_scales.shape == (128, 92, 23, 16, 4)
        assert prepared.fc2_scales.shape == (128, 46, 23, 16, 4)
        assert call["output_dtype"] is torch.bfloat16
        assert call["output"].shape == (16, 2944)
        assert call["output"].dtype is torch.bfloat16
        assert call["fc1_expert_biases"].dtype is torch.bfloat16
        assert call["fc2_expert_biases"].dtype is torch.bfloat16
        assert call["swiglu_alpha"].shape == (128,)
        assert call["swiglu_beta"].shape == (128,)
        assert call["swiglu_limit"].shape == (128,)
        assert call["use_w4_group_scaling"]
        assert not call["use_packed_weights"]
        assert call["tune_max_num_tokens"] == 8192
        assert call["activation_type"] == _ActivationType.Swiglu

    def test_shape_and_architecture_fail_early(self, monkeypatch) -> None:
        kernel, torch, _ = self._kernel(monkeypatch)
        config = _config()
        invalid = _weights(torch)
        invalid = replace(
            invalid,
            gate_up=_tensor(torch, (128, 5760, 1439)),
        )
        with pytest.raises(ValueError, match="gate_up must have shape"):
            kernel.prepare_weights(invalid, config)

        wrong_arch_kernel, wrong_arch_torch, _ = self._kernel(
            monkeypatch, capability=(10, 0)
        )
        with pytest.raises(ValueError, match="requires SM90"):
            wrong_arch_kernel.prepare_weights(_weights(wrong_arch_torch), config)

    def test_inactive_route_sentinel_reaches_cutlass_without_rerouting(
        self, monkeypatch
    ):
        import torch

        import resin_models_cuda.gpt_oss.flashinfer as adapter

        config = _config(num_experts=4, hidden_size=32, intermediate_size=32)
        kernel = adapter.FlashInferSM90MXFP4Kernel()
        monkeypatch.setattr(kernel, "_validate_sm90_tensor", lambda *args: None)
        indices = torch.tensor([[4, 0, 1, 2], [0, 4, 4, 4]], dtype=torch.int64)
        weights = torch.tensor(
            [[0, 0.5, 0.25, 0.25], [1, 0, 0, 0]], dtype=torch.float32
        )
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
        self, monkeypatch, limit
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
            values = torch.tensor(
                [1000, -1000, 4, -8, float("nan")], dtype=torch.bfloat16
            )
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
