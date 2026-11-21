"""Model geometry, numerical state transitions, and streamed checkpoint loading."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from _inkling2_model import (
    DenseCache,
    checkpoint_tensors,
    dense_attention,
    reference_experts,
    tiny_config,
)
from safetensors.torch import save_file

from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout
from resin.models.inkling2.config import Inkling2Config, Inkling2LoopConfig
from resin_models_cuda.inkling2.kernels import (
    select_convolution_state,
)
from resin_models_cuda.inkling2.model import DecoderLayer, Inkling2Target
from resin_models_cuda.inkling2.weights import (
    load_target_weights,
    target_weight_slices,
)


def _looped_config() -> Inkling2Config:
    return replace(
        tiny_config(),
        num_hidden_layers=8,
        local_layer_ids=(0, 1, 3, 4, 6),
        loop=Inkling2LoopConfig(core_start=2, core_end=6, num_passes=3, gate_rank=2),
    )


def _reference_target(config, tap_ids, monkeypatch):
    from resin_models_cuda.inkling2 import attention

    monkeypatch.setattr(attention, "paged_attention", dense_attention)
    torch.manual_seed(17)
    target = Inkling2Target(config, tap_ids)
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
    return target


@pytest.mark.parametrize("looped", [False, True], ids=["linear", "looped"])
def test_config_parses_checkpoint_and_rejects_unsupported_geometry(looped):
    config = _looped_config() if looped else tiny_config()
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
    if config.loop is not None:
        text.pop("pre_norm_pre_affine")
        text.pop("final_norm_pre_affine")
        text["loop_config"] = {**asdict(config.loop), "is_varloop": False}
        raw.update(
            model_type="inkling2_loop",
            architectures=["Inkling2LoopForConditionalGeneration"],
        )
    assert Inkling2Config.from_hugging_face(raw) == config
    if config.loop is not None:
        assert config.global_layer_ids == (2, 5, 7)
        assert (
            config.execution_local_layer_count,
            config.execution_global_layer_count,
        ) == (9, 7)
        text["loop_config"]["is_varloop"] = True
        with pytest.raises(ValueError, match="fixed loops"):
            Inkling2Config.from_hugging_face(raw)
        with pytest.raises(ValueError, match="residual block boundaries"):
            replace(config, loop=replace(config.loop, core_end=5)).validate()
    else:
        assert config.global_layer_ids == (2, 5)
    assert config.state_width(2) == 24
    with pytest.raises(ValueError, match="KV geometry"):
        replace(config, swa_head_dim=8).validate()
    with pytest.raises(ValueError, match="unique"):
        replace(config, local_layer_ids=(1, 1)).validate()
    with pytest.raises(ValueError, match="divide"):
        config.state_width(3)


@pytest.mark.parametrize("looped", [False, True], ids=["linear", "looped"])
def test_target_chunked_prefill_decode_and_speculative_rollback(monkeypatch, looped):
    config = _looped_config() if looped else tiny_config()
    tap_ids = (1, 2, 3, 6, 7, 10, 11, 16) if looped else (1, 3, 5, 6)
    target = _reference_target(config, tap_ids, monkeypatch)
    layers = cast(Sequence[DecoderLayer], target.layers)
    assert len(layers) == config.num_hidden_layers
    visits, local_sources = [], []
    with torch.no_grad():
        for index, layer in enumerate(layers):
            layer.attn.register_forward_pre_hook(
                lambda _module, _inputs, index=index: visits.append(index)
            )
        if looped:
            layers[3].attn.register_forward_pre_hook(
                lambda _module, inputs: local_sources.append(inputs[0])
            )
        initial = torch.zeros(
            2,
            config.execution_layer_count,
            3,
            config.state_width(),
            dtype=torch.bfloat16,
        )
        original = initial.clone()
        tokens = torch.randint(0, 21, (2, 9))
        positions = torch.arange(9).expand(2, -1)
        inputs = torch.empty(
            2,
            9,
            config.execution_layer_count,
            config.state_width(),
            dtype=torch.bfloat16,
        )
        full_cache = DenseCache(config, 2)
        expected, taps, final, _ = target(
            tokens,
            positions,
            full_cache,
            initial,
            torch.tensor([9, 9]),
            32,
            full_logits=True,
            state_inputs_out=inputs,
        )
        if looped:
            assert visits == [0, 1, *([2, 3, 4, 5] * 3), 6, 7]
            assert [group for group, _ in full_cache.writes] == [
                1,
                1,
                *([0, 1, 1, 0] * 3),
                1,
                0,
            ]
            assert [slot for group, slot in full_cache.writes if group == 0] == list(
                range(7)
            )
            assert [slot for group, slot in full_cache.writes if group == 1] == list(
                range(9)
            )
            for group, first_layer, next_layer in ((0, 0, 2), (1, 2, 4)):
                count = full_cache.counts[group]
                first_pages = full_cache.pages(group, first_layer, count)
                next_pages = full_cache.pages(group, next_layer, count)
                assert not torch.equal(
                    full_cache.keys[first_pages], full_cache.keys[next_pages]
                )
            assert not torch.equal(final[:, 2], final[:, 6])
            assert not torch.equal(final[:, 3], final[:, 7])
            assert len(local_sources) == 3
            q, k, v = local_sources[0]
            torch.testing.assert_close(q, k, rtol=0, atol=0)
            torch.testing.assert_close(q, v, rtol=0, atol=0)
            for q, k, v in local_sources[1:]:
                assert not torch.equal(q, k)
                assert not torch.equal(q, v)
                assert not torch.equal(k, v)
            assert taps.shape[-1] == len(tap_ids) * config.hidden_size
            _, start1, next1, start2, next2, start3, next3, output = taps.split(
                config.hidden_size, -1
            )
            torch.testing.assert_close(start1, start2, rtol=0, atol=0)
            torch.testing.assert_close(start1, start3, rtol=0, atol=0)
            assert not torch.equal(next1, next2)
            assert not torch.equal(next2, next3)
            torch.testing.assert_close(target.logits(output), expected, rtol=0, atol=0)
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


@torch.no_grad()
def test_loop_messages_read_only_previous_pass_core_blocks(monkeypatch):
    config = _looped_config()
    target = _reference_target(config, (4, 5, 8, 9), monkeypatch)
    layers = cast(Sequence[DecoderLayer], target.layers)
    attention_outputs, mlp_outputs = [], []
    layers[5].attn_sconv.register_forward_hook(
        lambda _module, _inputs, result: attention_outputs.append(result[0])
    )
    layers[5].mlp_sconv.register_forward_hook(
        lambda _module, _inputs, result: mlp_outputs.append(result[0])
    )
    controllers = layers[3].controllers
    assert controllers is not None
    calls = {name: [] for name in controllers}
    for name, controller in controllers.items():
        controller.register_forward_hook(
            lambda _module, inputs, result, name=name: calls[name].append(
                (inputs, result)
            )
        )

    tokens = torch.tensor([[1, 2, 3, 5]])
    state = torch.zeros(
        1, config.execution_layer_count, 3, config.state_width(), dtype=torch.bfloat16
    )
    _, taps, _, _ = target(
        tokens,
        torch.arange(4)[None],
        DenseCache(config, 1),
        state,
        torch.tensor([4]),
        32,
        full_logits=True,
    )
    width = config.hidden_size
    first1, partial1, first2, partial2 = taps.split(width, -1)
    queries = target.middle_cycle_cross_pass_queries
    kv_queries = target.middle_cycle_cross_pass_kv_queries
    for pass_index, (first, partial) in enumerate(
        ((first1, partial1), (first2, partial2))
    ):
        last = (
            partial.float()
            + attention_outputs[pass_index].float()
            + mlp_outputs[pass_index].float()
        ).bfloat16()
        blocks = torch.stack((first, last), dim=-2).float()
        blocks = blocks * (blocks.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        for name, query in (
            ("attn", queries[2]),
            ("k", kv_queries[2]),
            ("v", kv_queries[3]),
            ("mlp", queries[3]),
        ):
            assert len(calls[name]) == 2
            (destination, message), result = calls[name][pass_index]
            scores = (blocks * query).sum(-1) * width**-0.5
            expected = (scores.softmax(-1)[..., None] * blocks).sum(-2)
            torch.testing.assert_close(message, expected)

            controller = controllers[name]
            pair = torch.stack((destination, message), dim=-2)
            pair = (
                pair * (pair.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            ).bfloat16()
            low_rank = F.linear(
                pair.flatten(-2), controller.get_parameter("down_weight")
            ).float()
            gain = F.linear(
                low_rank, controller.get_parameter("gain_weight").float()
            ).tanh()
            forget = F.linear(
                low_rank, controller.get_parameter("forget_weight").float()
            ).tanh()
            torch.testing.assert_close(
                result, destination * (1 - forget) + message * gain
            )


def test_empty_expert_forward_follows_loop_visits(monkeypatch):
    target = Inkling2Target(_looped_config(), (16,))
    visits = []

    def record_visit(index, hidden):
        assert hidden.shape == (0, target.config.hidden_size)
        visits.append(index)

    for index, layer in enumerate(cast(Sequence[DecoderLayer], target.layers)):
        monkeypatch.setattr(
            layer,
            "feed_forward",
            lambda hidden, index=index: record_visit(index, hidden),
        )
    target.forward_experts_empty()
    assert visits == [0, 1, *([2, 3, 4, 5] * 3), 6, 7]


@pytest.mark.parametrize(
    "tp_rank,tp_size,looped",
    [(0, 1, False), (0, 2, False), (1, 2, False), (1, 2, True)],
)
def test_checkpoint_loader_maps_every_tensor_and_rejects_extra_target(
    tmp_path, tp_rank, tp_size, looped
):
    def unexpected_collective(_value):
        raise AssertionError("weight loading must not run collectives")

    parallel = SimpleNamespace(
        layout=TensorParallelLayout(tp_rank, tp_size),
        rank=tp_rank,
        size=tp_size,
        sum=unexpected_collective,
    )
    config = _looped_config() if looped else tiny_config()
    target = Inkling2Target(
        config,
        (1, 3, 5, 6),
        tensor_parallel=cast(TensorParallelContext, parallel),
    )
    destinations = target_weight_slices(target)
    tensors = checkpoint_tensors(config)
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
