from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_cuda.qwen3_5.checkpoint import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    build_meta_model,
    load_config,
)


def _qwen38() -> dict[str, Any]:
    layers = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(64)
    ]
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "dtype": "bfloat16",
            "vocab_size": 248320,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
            },
            "layer_types": layers,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "eos_token_id": 248044,
            "bos_token_id": 248044,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "output_gate_type": "swish",
            "tie_word_embeddings": False,
        },
    }


def test_qwen38_27b_geometry() -> None:
    config = Qwen35Config.from_hugging_face(_qwen38())

    assert config.full_layer_ids == tuple(range(3, 64, 4))
    assert config.linear_layer_count == 48
    assert config.full_layer_count == 16
    assert config.rotary_dim == 64
    assert config.linear_conv_dim == 10240
    config.require_mtp()


def test_qwen38_accepts_interleaved_multimodal_rope_sections() -> None:
    values = copy.deepcopy(_qwen38())
    values["text_config"]["rope_parameters"]["mrope_section"] = [11, 11, 10]  # type: ignore[index]

    config = Qwen35Config.from_hugging_face(values)

    assert config.mrope_section == (11, 11, 10)


def test_qwen38_is_the_pinned_family_default() -> None:
    assert DEFAULT_MODEL == "RadixArk/Qwen3.8-27B-NVFP4"
    assert DEFAULT_REVISION == "554ebba9b5f1b79dc11246341960360e6ef05ef4"


def test_qwen35_parser_accepts_text_config_directly() -> None:
    values = _qwen38()
    config = Qwen35Config.from_hugging_face(values["text_config"])  # type: ignore[arg-type]
    assert config.vocab_size == 248320


def test_qwen35_rejects_incompatible_gate_semantics() -> None:
    values = copy.deepcopy(_qwen38())
    values["text_config"]["attn_output_gate"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="unsupported Qwen3.5 options"):
        Qwen35Config.from_hugging_face(values)


def test_cuda_loader_rejects_tied_embeddings(tmp_path) -> None:
    values = _qwen38()
    values["text_config"]["tie_word_embeddings"] = True
    (tmp_path / "config.json").write_text(json.dumps(values))

    with pytest.raises(ValueError, match="tied embeddings"):
        load_config(tmp_path)
    with pytest.raises(ValueError, match="tied embeddings"):
        build_meta_model(values, include_mtp=False)
