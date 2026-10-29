from __future__ import annotations

import copy
from typing import Any

import pytest

from resin.models.qwen3_5.config import Qwen35Config


def _ornith_35b() -> dict[str, Any]:
    """Text geometry from ornith-ai/Ornith-1.0-35B at 5df2ed3."""

    layers = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(40)
    ]
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "dtype": "bfloat16",
            "vocab_size": 248320,
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
            "layer_types": layers,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "eos_token_id": 248044,
            "bos_token_id": 248044,
            "attn_output_gate": True,
            "tie_word_embeddings": False,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
        },
    }


def test_ornith_moe_geometry() -> None:
    config = Qwen35Config.from_hugging_face(_ornith_35b())

    assert config.is_moe
    assert config.intermediate_size == 0
    assert (config.num_experts, config.num_experts_per_tok) == (256, 8)
    assert config.moe_intermediate_size == 512
    assert config.shared_expert_intermediate_size == 512
    assert config.norm_topk_prob
    assert config.full_layer_count == 10
    assert config.linear_layer_count == 30
    assert config.rotary_dim == 64
    assert config.linear_conv_dim == 8192
    with pytest.raises(ValueError, match="Qwen3.5-MoE"):
        config.require_dense()


def test_moe_text_config_parses_without_root() -> None:
    config = Qwen35Config.from_hugging_face(_ornith_35b()["text_config"])
    assert config.is_moe


def test_moe_root_requires_matching_text_type() -> None:
    raw = _ornith_35b()
    raw["text_config"]["model_type"] = "qwen3_5_text"
    with pytest.raises(ValueError, match="qwen3_5_text"):
        Qwen35Config.from_hugging_face(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_experts", 0, "num_experts"),
        ("moe_intermediate_size", 0, "moe_intermediate_size"),
        ("num_experts_per_tok", 300, "experts per token"),
    ],
)
def test_invalid_moe_geometry(field: str, value: int, message: str) -> None:
    raw = copy.deepcopy(_ornith_35b())
    raw["text_config"][field] = value
    with pytest.raises(ValueError, match=message):
        Qwen35Config.from_hugging_face(raw)
