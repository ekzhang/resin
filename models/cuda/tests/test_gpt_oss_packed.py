from __future__ import annotations

import torch

from resin_models_cuda.gpt_oss.flashinfer import MXFP4Config


class TestGptOssPackedBridge:
    def test_official_block_layout_is_flattened_and_installed(self) -> None:
        from resin_models_cuda.gpt_oss.mxfp4 import KernelExperts

        config = MXFP4Config(
            top_k=2,
            num_experts=4,
            hidden_size=32,
            intermediate_size=32,
            swiglu_limit=7.0,
        )

        class Kernel:
            def __init__(self) -> None:
                self.prepared = []
                self.calls = []

            def prepare_weights(self, weights, actual_config):
                self.prepared.append((weights, actual_config))
                return weights

            def forward(self, hidden, indices, route_weights, weights):
                assert weights is self.prepared[0][0]
                self.calls.append((indices, route_weights))
                return hidden

        class Args:
            num_local_experts = 4
            hidden_size = 32
            intermediate_size = 32
            num_experts_per_tok = 2

        kernel = Kernel()
        module = KernelExperts(Args(), kernel, config)
        tensors = {
            "gate_up_proj_blocks": torch.zeros(4, 64, 1, 16, dtype=torch.uint8),
            "gate_up_proj_scales": torch.zeros(4, 64, 1, dtype=torch.uint8),
            "gate_up_proj_bias": torch.zeros(4, 64),
            "down_proj_blocks": torch.zeros(4, 32, 1, 16, dtype=torch.uint8),
            "down_proj_scales": torch.zeros(4, 32, 1, dtype=torch.uint8),
            "down_proj_bias": torch.zeros(4, 32),
        }
        module.load_weights(tensors)

        assert len(kernel.prepared) == 1
        weights, actual_config = kernel.prepared[0]
        assert actual_config is config
        assert weights.gate_up.shape == (4, 64, 16)
        assert weights.down.shape == (4, 32, 16)
        assert module.is_prepared

        hidden = torch.ones(3, 32, dtype=torch.bfloat16)
        indices = torch.tensor([[0, 1], [2, 3], [1, 2]])
        route_weights = torch.full((3, 2), 0.5)
        output = module(hidden, indices, route_weights)
        assert output is hidden
        assert len(kernel.calls) == 1
        assert kernel.calls[0][0] is indices
        assert kernel.calls[0][1] is route_weights
