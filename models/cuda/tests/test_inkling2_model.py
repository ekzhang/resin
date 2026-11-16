"""Model geometry, numerical state transitions, and streamed checkpoint loading."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from _inkling2_model import (
    DenseCache,
    checkpoint_tensors,
    dense_attention,
    reference_experts,
    tiny_config,
)
from safetensors.torch import save_file

from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.models.inkling2.config import Inkling2Config
from resin_models_cuda.inkling2.kernels import (
    select_convolution_state,
)
from resin_models_cuda.inkling2.model import DecoderLayer, Inkling2Target
from resin_models_cuda.inkling2.weights import (
    load_target_weights,
    target_weight_slices,
)


def test_config_parses_checkpoint_and_rejects_unsupported_geometry():
    config = tiny_config()
    text = asdict(config)
    text.update(
        torch_dtype="bfloat16",
        q_bias=False,
        o_bias=False,
        pre_norm_pre_affine=False,
        final_norm_pre_affine=False,
        use_value_norm_no_gamma=True,
        n_shared_experts=0,
        use_gate_bias=True,
        n_routed_experts=config.num_routed_experts,
        num_experts_per_tok=config.experts_per_token,
        model_max_length=config.max_position_embeddings,
    )
    raw = dict(
        model_type="inkling2",
        architectures=["Inkling2ForConditionalGeneration"],
        text_config=text,
        eos_token_id=20,
    )
    assert Inkling2Config.from_hugging_face(raw) == config
    assert config.global_layer_ids == (2, 5)
    assert config.state_width(2) == 24
    with pytest.raises(ValueError, match="KV geometry"):
        replace(config, swa_head_dim=8).validate()
    with pytest.raises(ValueError, match="unique"):
        replace(config, local_layer_ids=(1, 1)).validate()
    with pytest.raises(ValueError, match="divide"):
        config.state_width(3)


def test_target_chunked_prefill_decode_and_speculative_rollback(monkeypatch):
    from resin_models_cuda.inkling2 import attention

    monkeypatch.setattr(attention, "paged_attention", dense_attention)
    torch.manual_seed(17)
    config = tiny_config()
    target = Inkling2Target(config, (1, 3, 5, 6))
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.normal_(0, 0.2)
        for layer in cast(Sequence[DecoderLayer], target.layers):
            experts = layer.mlp.experts
            monkeypatch.setattr(
                experts,
                "_local",
                lambda h, i, w, e=experts, max_rows_per_expert=None: reference_experts(
                    e, h, i, w
                ),
            )
        initial = torch.zeros(2, 6, 3, config.state_width(), dtype=torch.bfloat16)
        original = initial.clone()
        tokens = torch.randint(0, 21, (2, 9))
        positions = torch.arange(9).expand(2, -1)
        inputs = torch.empty(2, 9, 6, config.state_width(), dtype=torch.bfloat16)
        expected, taps, final, _ = target(
            tokens,
            positions,
            DenseCache(config, 2),
            initial,
            torch.tensor([9, 9]),
            32,
            full_logits=True,
            state_inputs_out=inputs,
        )
        torch.testing.assert_close(initial, original, rtol=0, atol=0)
        assert final.data_ptr() != initial.data_ptr()
        cache, state, outputs, captures = DenseCache(config, 2), initial, [], []
        for start, stop in ((0, 3), (3, 4), (4, 9)):
            result, captured, state, _ = target(
                tokens[:, start:stop],
                positions[:, start:stop],
                cache,
                state,
                torch.tensor([stop, stop]),
                32,
                full_logits=True,
            )
            outputs.append(result)
            captures.append(captured)
        torch.testing.assert_close(
            torch.cat(outputs, 1), expected, rtol=0.015, atol=0.002
        )
        torch.testing.assert_close(torch.cat(captures, 1), taps, rtol=0.015, atol=0.002)
        torch.testing.assert_close(state, final, rtol=0.015, atol=0.002)
        accepted = torch.tensor([2, 7])
        rolled = select_convolution_state(initial, inputs, accepted)
        for row, count in enumerate(accepted.tolist()):
            _, _, exact, _ = target(
                tokens[row : row + 1, :count],
                positions[row : row + 1, :count],
                DenseCache(config, 1),
                initial[row : row + 1],
                torch.tensor([count]),
                32,
                full_logits=True,
            )
            torch.testing.assert_close(
                rolled[row : row + 1], exact, rtol=0.015, atol=0.002
            )


@pytest.mark.parametrize("tp_rank,tp_size", [(0, 1), (0, 2), (1, 2)])
def test_checkpoint_loader_maps_every_tensor_and_rejects_extra_target(
    tmp_path, tp_rank, tp_size
):

    def unexpected_collective(_value):
        raise AssertionError("weight loading must not run collectives")

    parallel = SimpleNamespace(
        layout=TensorParallelLayout(tp_rank, tp_size),
        rank=tp_rank,
        size=tp_size,
        sum=unexpected_collective,
    )
    target = Inkling2Target(
        tiny_config(),
        (1, 3, 5, 6),
        tensor_parallel=cast(TensorParallelContext, parallel),
    )
    destinations = target_weight_slices(target)
    tensors = checkpoint_tensors(tiny_config())
    save_file(tensors, str(tmp_path / "model.safetensors"))
    index = {"weight_map": {name: "model.safetensors" for name in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert load_target_weights(target, tmp_path) == sum(
        tensors[name][destination.slices].numel() * tensors[name].element_size()
        for name, destination in destinations.items()
    )
    for name, destination in destinations.items():
        torch.testing.assert_close(
            destination.tensor,
            tensors[name][destination.slices].to(destination.tensor.dtype),
        )
    index["weight_map"]["model.llm.unknown.weight"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="unexpected"):
        load_target_weights(target, tmp_path)
