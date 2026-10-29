from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from resin_models_cuda.gpt_oss.model import DenseExperts, MoEForCausalLM


def _tiny_args():
    from resin.models.gpt_oss.config import GptOssConfig

    return GptOssConfig(
        vocab_size=43,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        head_dim=4,
        sliding_window=3,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        initial_context_length=32,
        rope_scaling_factor=1.0,
        rope_beta_fast=32.0,
        rope_beta_slow=1.0,
        max_position_embeddings=64,
        swiglu_limit=7.0,
        attention_bias=True,
        layer_types=("sliding_attention", "full_attention"),
    )


def _config(args, *, quantization=None):
    config = {
        "model_type": "gpt_oss",
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "num_local_experts": args.num_local_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "head_dim": args.head_dim,
        "sliding_window": args.sliding_window,
        "rms_norm_eps": args.rms_norm_eps,
        "rope_theta": args.rope_theta,
        "rope_scaling": {
            "factor": args.rope_scaling_factor,
            "beta_fast": args.rope_beta_fast,
            "beta_slow": args.rope_beta_slow,
            "original_max_position_embeddings": args.initial_context_length,
            "rope_type": "yarn",
        },
        "max_position_embeddings": args.max_position_embeddings,
        "swiglu_limit": args.swiglu_limit,
        "attention_bias": args.attention_bias,
        "attention_dropout": args.attention_dropout,
        "initializer_range": args.initializer_range,
        "layer_types": list(args.layer_types),
        "tie_word_embeddings": False,
    }
    if quantization is not None:
        config["quantization_config"] = quantization
    return config


class TestGptOssImport:
    def test_configs_cover_both_released_model_shapes(self) -> None:
        from resin.models.gpt_oss.config import GptOssConfig

        base: dict[str, object] = {
            "model_type": "gpt_oss",
            "vocab_size": 201_088,
            "hidden_size": 2_880,
            "intermediate_size": 2_880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_experts_per_tok": 4,
            "head_dim": 64,
            "sliding_window": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 150_000,
            "rope_scaling": {
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4_096,
            },
            "max_position_embeddings": 131_072,
            "swiglu_limit": 7.0,
            "attention_bias": True,
            "attention_dropout": 0.0,
            "tie_word_embeddings": False,
        }
        for layers, experts in ((24, 32), (36, 128)):
            config = dict(base)
            config.update(
                num_hidden_layers=layers,
                num_local_experts=experts,
                layer_types=[
                    "sliding_attention" if index % 2 == 0 else "full_attention"
                    for index in range(layers)
                ],
            )
            args = GptOssConfig.from_dict(config)
            assert args.num_hidden_layers == layers
            assert args.num_local_experts == experts
            assert args.num_experts_per_tok == 4
            assert args.head_dim == 64

    def test_args_reject_non_gpt_oss_checkpoint(self) -> None:
        from resin.models.gpt_oss.config import GptOssConfig

        with pytest.raises(ValueError, match="expected a GPT-OSS"):
            GptOssConfig.from_dict({"model_type": "llama"})


class TestGptOssTorch:
    def setup_method(self) -> None:
        torch.manual_seed(11)
        self.args = _tiny_args()
        self.model = MoEForCausalLM(self.args).eval()

    def test_topk_moe_routes_and_weights_selected_experts(self) -> None:
        mlp = self.model.model.layers[0].mlp
        assert isinstance(mlp.experts, DenseExperts)
        with torch.no_grad():
            mlp.router.weight.zero_()
            mlp.router.bias.copy_(torch.tensor([10.0, 9.0, -10.0, -20.0]))
            mlp.experts.gate_up_proj.zero_()
            mlp.experts.gate_up_proj_bias.zero_()
            mlp.experts.down_proj.zero_()
            mlp.experts.down_proj_bias.zero_()
            mlp.experts.down_proj_bias[0].fill_(2.0)
            mlp.experts.down_proj_bias[1].fill_(4.0)

        hidden = torch.randn(1, 1, self.args.hidden_size)
        _, weights, indices = mlp.router(hidden.reshape(1, -1))
        assert indices.tolist() == [[0, 1]]
        expected_value = 2.0 * weights[0, 0] + 4.0 * weights[0, 1]
        with torch.inference_mode():
            actual = mlp(hidden)
        torch.testing.assert_close(
            actual,
            torch.full_like(actual, expected_value.item()),
            rtol=0,
            atol=1e-6,
        )

    def test_expert_boundary_receives_selected_routes_and_weights(self) -> None:
        calls = []

        class RecordingExperts(torch.nn.Module):
            def forward(self, hidden, indices, weights):
                calls.append((hidden.shape, indices.shape, weights.shape))
                return torch.zeros_like(hidden)

        model = MoEForCausalLM(self.args).eval()
        mlp = model.model.layers[0].mlp
        mlp.experts = RecordingExperts()
        hidden = torch.randn(1, 3, self.args.hidden_size)
        with torch.inference_mode():
            output = mlp(hidden)
        assert output.shape == hidden.shape
        assert calls == [
            (
                torch.Size([3, self.args.hidden_size]),
                torch.Size([3, self.args.num_experts_per_tok]),
                torch.Size([3, self.args.num_experts_per_tok]),
            )
        ]

    def test_mxfp4_checkpoint_has_precise_kernel_boundary_error(
        self, tmp_path: Path
    ) -> None:
        from resin_models_cuda.gpt_oss.checkpoint import load_model

        path = tmp_path
        (path / "config.json").write_text(
            json.dumps(_config(self.args, quantization={"quant_method": "mxfp4"}))
        )
        with pytest.raises(
            NotImplementedError,
            match="MXFP4 expert weights.*kernel_moe",
        ):
            load_model(path, device="cpu")

    def test_quantization_plan_validates_semantics_without_selecting_a_kernel(
        self,
    ) -> None:
        from resin.backend.quantization import (
            BF16,
            MXFP4,
            QuantizationConfig,
        )
        from resin_models_cuda.gpt_oss.checkpoint import _check_quantization

        config = _config(self.args, quantization={"quant_method": "mxfp4"})
        plan = QuantizationConfig({"transformer.experts": MXFP4})

        _check_quantization(config, quantization=plan, kernel_moe=object())
        with pytest.raises(ValueError, match="quantization config selected"):
            _check_quantization(
                config,
                quantization=QuantizationConfig(default=BF16),
                kernel_moe=object(),
            )

    def test_loader_rejects_non_mxfp4_checkpoints(self, tmp_path: Path) -> None:
        from resin_models_cuda.gpt_oss.checkpoint import load_model

        path = tmp_path
        (path / "config.json").write_text(json.dumps(_config(self.args)))
        with pytest.raises(ValueError, match="requires an MXFP4 checkpoint"):
            load_model(path, device="cpu", kernel_moe=object())
