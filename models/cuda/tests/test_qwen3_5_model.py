from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_qwen3_5_config import _qwen38

from resin.models.qwen3_5.config import Qwen35Config
from resin_models_cuda.qwen3_5.checkpoint import _select_tensor_files
from resin_models_cuda.qwen3_5.model import (
    Qwen35ForConditionalGeneration,
    Qwen35GatedDeltaNet,
    Qwen35RMSNorm,
    Qwen35TextModel,
)


def _tiny_config() -> Qwen35Config:
    values = _qwen38()
    text = values["text_config"]  # type: ignore[assignment]
    text.update(  # type: ignore[union-attr]
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    return Qwen35Config.from_hugging_face(values)


def test_qwen_offset_rms_norm_uses_one_plus_checkpoint_weight() -> None:
    norm = Qwen35RMSNorm(4, 1e-6)
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        norm.weight.fill_(0.5)
    expected = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(norm(hidden), expected * 1.5)


@pytest.mark.parametrize("value_heads", [2, 4])
def test_delta_net_chunk_matches_recurrent_steps(value_heads: int) -> None:
    torch.manual_seed(8)
    module = Qwen35GatedDeltaNet(
        replace(_tiny_config(), linear_num_value_heads=value_heads)
    )
    hidden = torch.randn(2, 6, 32)

    expected, expected_recurrent, expected_convolution = module(hidden, None, None)
    recurrent = None
    convolution = None
    pieces = []
    for step in range(hidden.shape[1]):
        output, recurrent, convolution = module(
            hidden[:, step : step + 1], recurrent, convolution
        )
        pieces.append(output)

    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(recurrent, expected_recurrent, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(convolution, expected_convolution)


def test_prefill_ffn_regions_preserve_recurrent_continuation_and_changed_tokens():
    from test_cuda_runner_graph_policy import cpu_prefill_regions

    torch.manual_seed(28)
    # Isolate real recurrent layers so the test needs no paged attention backend.
    config = replace(
        _tiny_config(), num_hidden_layers=2, layer_types=("linear_attention",) * 2
    )
    model = Qwen35TextModel(config).to(torch.bfloat16)
    token_ids = torch.tensor([[1, 4, 9, 3, 8, 2, 11]])
    positions = torch.arange(7)[None]
    with torch.no_grad():
        expected, recurrent, convolution = model(token_ids, positions)
        runner, callbacks, replays = cpu_prefill_regions(
            (layer.feed_forward for layer in model.layers), config.hidden_size
        )
        for layer, callback in zip(model.layers, callbacks, strict=True):
            layer.prefill_ffn = callback

        # Outside prefill, installed callbacks execute the original operation.
        torch.testing.assert_close(model(token_ids, positions)[0], expected)
        assert replays == []
        with runner._prefill_graphs():
            first, state, conv = model(token_ids[:, :3], positions[:, :3])
            second, state, conv = model(
                token_ids[:, 3:],
                positions[:, 3:],
                recurrent_states=state,
                convolution_states=conv,
            )

    torch.testing.assert_close(torch.cat((first, second), dim=1), expected)
    torch.testing.assert_close(state, recurrent)
    torch.testing.assert_close(conv, convolution)
    assert replays == [(4, 1), (4, 2), (4, 1), (4, 2)]


def test_interleaved_mrope_selects_temporal_height_and_width_axes() -> None:
    values = _qwen38()
    text = values["text_config"]  # type: ignore[assignment]
    text.update(  # type: ignore[union-attr]
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        partial_rotary_factor=1.0,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    text["rope_parameters"] = {  # type: ignore[index]
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0,
        "mrope_section": [1, 1, 2],
    }
    model = Qwen35ForConditionalGeneration(
        Qwen35Config.from_hugging_face(values), include_mtp=False
    ).language_model
    positions = torch.tensor([[[2, 3]], [[5, 7]], [[11, 13]]])
    cos, _ = model.rotary(positions, torch.float32)
    axes = positions.float().unsqueeze(-1) * model.inv_freq
    expected = axes[0].clone()
    expected[..., 1] = axes[1, ..., 1]
    expected[..., 2] = axes[2, ..., 2]

    torch.testing.assert_close(cos[..., :4], expected.cos())


def test_target_and_optional_mtp_names_match_checkpoint() -> None:
    model = Qwen35ForConditionalGeneration(_tiny_config(), include_mtp=True)
    names = set(model.state_dict())

    assert "model.language_model.embed_tokens.weight" in names
    assert "model.language_model.layers.0.linear_attn.A_log" in names
    assert "model.language_model.layers.3.self_attn.q_proj.weight" in names
    assert "lm_head.weight" in names
    assert "mtp.fc.weight" in names
    assert "mtp.layers.0.self_attn.q_proj.weight" in names
    assert not any("mtp.embed_tokens" in name for name in names)


def _assert_optional_visual_tower_uses_checkpoint_names() -> None:
    vision_config = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "out_hidden_size": 32,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 16,
    }
    model = Qwen35ForConditionalGeneration(
        _tiny_config(), include_mtp=False, vision_config=vision_config
    )
    names = set(model.state_dict())

    assert "model.visual.patch_embed.proj.weight" in names
    assert "model.visual.blocks.0.attn.qkv.weight" in names
    assert "model.visual.merger.linear_fc2.weight" in names


def test_optional_visual_tower_uses_checkpoint_names() -> None:
    tests = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(tests)!r}); "
                "from test_qwen3_5_model import "
                "_assert_optional_visual_tower_uses_checkpoint_names as check; "
                "check()"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_radixark_quantized_target_keeps_mtp_bfloat16() -> None:
    model = Qwen35ForConditionalGeneration(
        _tiny_config(), include_mtp=True, quantized=True
    )
    state = model.state_dict()

    projection = "model.language_model.layers.3.self_attn.q_proj"
    assert state[f"{projection}.weight"].dtype is torch.float8_e4m3fn
    assert state[f"{projection}.input_scale"].shape == ()
    assert state[f"{projection}.weight_scale"].shape == ()
    mlp = "model.language_model.layers.3.mlp.gate_proj"
    assert state[f"{mlp}.weight"].dtype is torch.uint8
    assert state[f"{mlp}.weight_scale"].dtype is torch.float8_e4m3fn
    assert f"{mlp}.weight_scale_2" in state
    assert state["mtp.layers.0.self_attn.q_proj.weight"].dtype is torch.float32
    assert "mtp.layers.0.self_attn.q_proj.input_scale" not in state


def _assert_native_text_model_matches_transformers_reference() -> None:
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextModel as ReferenceModel,
    )

    values = _qwen38()["text_config"]  # type: ignore[index]
    values.update(  # type: ignore[union-attr]
        vocab_size=64,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    config = Qwen35Config.from_hugging_face(values)  # type: ignore[arg-type]
    native = Qwen35ForConditionalGeneration(config, include_mtp=False).language_model
    reference = ReferenceModel(Qwen3_5TextConfig.from_dict(values))  # type: ignore[arg-type]
    reference.load_state_dict(native.state_dict())
    token_ids = torch.tensor([[1, 7, 12, 4]])
    positions = torch.arange(4)[None]

    with torch.no_grad():
        actual = native(token_ids, positions)[0]
        expected = reference(
            token_ids,
            position_ids=positions,
            use_cache=False,
        ).last_hidden_state

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_native_text_model_matches_transformers_reference() -> None:
    tests = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(tests)!r}); "
                "from test_qwen3_5_model import "
                "_assert_native_text_model_matches_transformers_reference as check; "
                "check()"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_text_loader_ignores_vision_and_optionally_mtp() -> None:
    target = {
        "model.language_model.embed_tokens.weight": Path(__file__),
        "lm_head.weight": Path(__file__),
        "mtp.fc.weight": Path(__file__),
        "model.visual.blocks.0.attn.qkv.weight": Path(__file__),
    }
    selected = _select_tensor_files(
        target,
        {"model.language_model.embed_tokens.weight", "lm_head.weight"},
        include_mtp=False,
    )
    assert set(selected) == {
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    }


def test_multimodal_loader_selects_visual_tensors() -> None:
    target = {
        "model.language_model.embed_tokens.weight": Path(__file__),
        "lm_head.weight": Path(__file__),
        "model.visual.blocks.0.attn.qkv.weight": Path(__file__),
    }
    selected = _select_tensor_files(
        target,
        set(target),
        include_mtp=False,
        include_vision=True,
    )
    assert set(selected) == set(target)
