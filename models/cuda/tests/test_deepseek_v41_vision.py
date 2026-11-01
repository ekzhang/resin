"""Vision geometry, chunk ownership and language-model image semantics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from test_deepseek_v41_model import implementation as implementation
from torch import nn

from resin import BatchPhase, BatchPlan
from resin.backend.cuda.completion import CudaContinuation
from resin.backend.cuda.vision_cache import VisionFeatureCache
from resin.recipe import (
    DeepSeekV41Image,
    DeepSeekV41Media,
)
from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner
from resin_models_cuda.deepseek_v41.vision import Aligner, ViT, get_vision_cos_sin


def test_vision_rope_and_aligner_match_checkpoint_spatial_order():
    args = SimpleNamespace(
        vision_dim=16,
        vision_n_heads=2,
        vision_n_layers=1,
        vision_inter_dim=24,
        vision_patch_size=14,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        dim=8,
    )
    torch.manual_seed(12)
    vit, aligner = ViT(args), Aligner(args)
    patches = torch.randn(20, 3, 14, 14)
    hidden = vit(patches, 4, 5)
    assert hidden.shape == (20, 16) and hidden.isfinite().all()
    cos, sin = get_vision_cos_sin(4, 5, 4, 10000.0, hidden.device)
    inv = torch.tensor([1.0, 0.01])
    for row in range(4):
        for col in range(5):
            angles = torch.cat((row * inv, col * inv))
            torch.testing.assert_close(cos[row * 5 + col, 0], angles.cos())
            torch.testing.assert_close(sin[row * 5 + col, 0], angles.sin())
    # Check the channel-major 3x3 ordering and bottom/right zero padding against
    # explicit spatial slices, independent of the implementation's unfold call.
    padded = F.pad(hidden.reshape(4, 5, 16).permute(2, 0, 1), (0, 1, 0, 2))
    merged = torch.stack(
        [
            padded[:, row : row + 3, col : col + 3].flatten()
            for row in (0, 3)
            for col in (0, 3)
        ]
    )
    torch.testing.assert_close(
        aligner(hidden, 4, 5), aligner.w2(F.gelu(aligner.w1(merged)))
    )
    assert "blocks.0.attn.wqkv.bias" in vit.state_dict()
    assert "blocks.0.mlp.w1.weight" in vit.state_dict()


def media_fixture():
    types = (0, 1, 1, 2, 1, 1, 2, 3)
    image = DeepSeekV41Image(
        start=2,
        patches=np.zeros((16, 3, 14, 14), dtype=np.float32),
        n_vit_h=4,
        n_vit_w=4,
        types=types,
    )
    return DeepSeekV41Media(
        token_ids=(7, 8) + (129264,) * len(types) + (9, 10), images=(image,)
    )


def runner_fixture():
    runner = object.__new__(DeepSeekV41CudaRunner)
    runner._continuation = CudaContinuation()
    runner._media_features = {}
    runner._vision_feature_cache = VisionFeatureCache()
    runner.device = torch.device("cpu")
    runner.max_batch_size, runner.prefill_chunk_size = 8, 128
    calls = []

    def encode(patches, h, w):
        calls.append((patches, h, w))
        return torch.arange(12).reshape(4, 3).float()

    runner._model = SimpleNamespace(
        args=SimpleNamespace(image_token_id=129264),
        embed=lambda ids: ids.float()[..., None].expand(-1, -1, 3).clone(),
        vision=encode,
        aligner=lambda hidden, h, w: hidden,
        image_start=torch.full((3,), -1.0),
        image_end=torch.full((3,), -2.0),
        image_newline=torch.full((3,), -3.0),
    )
    return runner, calls


def plan_for(media, start, end, *, request=42):
    return BatchPlan(
        sequence_ids=[request],
        context_lengths=[start],
        token_ids=media.token_ids[start:end],
        token_indptr=[0, end - start],
        phase=BatchPhase.PREFILL,
        media=[media],
    )


def test_image_embeddings_reuse_features_across_chunks_and_prefix_restore():
    media = media_fixture()
    runner, calls = runner_fixture()
    expected = torch.tensor(
        [
            [7] * 3,
            [8] * 3,
            [-1] * 3,
            [0, 1, 2],
            [3, 4, 5],
            [-3] * 3,
            [6, 7, 8],
            [9, 10, 11],
            [-3] * 3,
            [-2] * 3,
            [9] * 3,
            [10] * 3,
        ]
    ).float()
    for start, end in ((0, 5), (5, 7), (7, 12)):
        plan = plan_for(media, start, end)
        runner._validate_plan(plan)
        result = runner._media_embeddings(
            plan, 0, torch.tensor([media.token_ids[start:end]])
        )
        torch.testing.assert_close(result[0], expected[start:end])
    assert len(calls) == 1
    runner.release([42])
    assert not runner._media_features
    # A prefix-cache hit can leave only the last part of an image to prefill.
    plan = plan_for(media, 7, 12)
    runner._validate_plan(plan)
    result = runner._media_embeddings(plan, 0, torch.tensor([media.token_ids[7:12]]))
    torch.testing.assert_close(result[0], expected[7:12])
    assert len(calls) == 1
    text = plan_for(media, 10, 12)
    assert runner._media_embeddings(text, 0, torch.tensor([[9, 10]])) is None
    changed = DeepSeekV41Media(
        token_ids=media.token_ids[:-1] + (11,), images=media.images
    )
    with pytest.raises(ValueError, match="changed its media or prompt"):
        runner._validate_plan(plan_for(changed, 0, 5))


def test_image_validation_rejects_unbacked_or_mismatched_prompt_tokens():
    runner, _ = runner_fixture()
    with pytest.raises(ValueError, match="require image media"):
        runner._validate_plan(
            BatchPlan(
                sequence_ids=[1],
                context_lengths=[0],
                token_ids=[129264],
                token_indptr=[0, 1],
                phase=BatchPhase.PREFILL,
            )
        )
    media = media_fixture()
    with pytest.raises(ValueError, match="does not match"):
        runner._validate_plan(
            BatchPlan(
                sequence_ids=[1],
                context_lengths=[0],
                token_ids=[11],
                token_indptr=[0, 1],
                phase=BatchPhase.PREFILL,
                media=[media],
            )
        )


def test_image_gate_bias_changes_expert_selection_only(implementation):
    gate = implementation.Gate(
        implementation.ModelArgs(
            dim=4,
            n_routed_experts=4,
            n_activated_experts=2,
            vision_n_layers=1,
        )
    )
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        gate.bias_vl.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    weights, indices = gate(torch.zeros(2, 4), torch.tensor([7, 129264]))
    assert indices.tolist() == [[0, 1], [3, 2]]
    torch.testing.assert_close(weights[0], weights[1])


def test_prefill_injects_embeddings_and_masks_engram_but_decode_stays_text(
    implementation,
):
    model = implementation.Transformer.__new__(implementation.Transformer)
    nn.Module.__init__(model)
    model.vision = nn.Identity()
    model.args = SimpleNamespace(image_token_id=129264)
    model.hc_mult, model.target_layer_ids = 1, set()
    model.layers, model.norm = nn.ModuleList(), nn.Identity()
    model.embed = lambda ids: torch.zeros(*ids.shape, 3)
    masks = []
    contributions = []
    projection_phases = []

    class Layer(nn.Module):
        layer_id = 0

        def engram(self, hidden, *, token_mask, embed_hook, prefill):
            contributions.append(token_mask)
            projection_phases.append(prefill)
            return hidden

        def forward(self, hidden, pre, ids, attention, **kwargs):
            return hidden, pre

    model.layers = nn.ModuleList([Layer()])
    model.engram_hash = lambda ids, positions, cache, mask: masks.append(mask) or {}
    model.engram_runtime = SimpleNamespace(fetch=lambda hashes: {0: lambda: None})
    cache = SimpleNamespace(
        prepare_metadata=lambda positions: SimpleNamespace(positions=positions)
    )
    ids, positions = torch.tensor([[7, 129264, 9]]), torch.arange(3)[None]
    embeddings = torch.randn(1, 3, 3)
    result, _ = model._run(
        ids, positions, 3, cache, prefill=True, inputs_embeds=embeddings
    )
    torch.testing.assert_close(result, embeddings)
    assert masks[0].tolist() == [[True, False, True]]
    assert contributions[0] is masks[0]
    model._run(ids[:, :1], positions[:, :1], 3, cache)
    assert masks[1] is None
    assert contributions[1] is None
    assert projection_phases == [True, False]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode", "verify", "draft"])
def test_generated_image_ids_keep_text_expert_routing(
    implementation, monkeypatch, lazy, phase
):
    """A sampled/draft image marker is text; only actual prompt spans use VL bias."""
    gate = implementation.Gate(
        implementation.ModelArgs(
            dim=4,
            n_routed_experts=4,
            n_activated_experts=2,
            vision_n_layers=1,
        )
    )
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        gate.bias_vl.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    routes = []

    class Routed(nn.Module):
        def forward(self, hidden, indices, weights):
            routes.append(indices)
            return torch.zeros_like(hidden)

    class Attention(nn.Module):
        def forward(self, hidden, *args, **kwargs):
            return hidden

    block = implementation.Block.__new__(implementation.Block)
    nn.Module.__init__(block)
    block.args, block.layer_id, block.engram = SimpleNamespace(hc_mult=1), 0, None
    block.attn_norm = block.ffn_norm = block.norm = nn.Identity()
    block.attn = Attention()
    block.lazy_hc, block.prefill_ffn = lazy, None
    block.ffn = implementation.MoE(
        SimpleNamespace(
            dim=4,
            gate=gate,
            shared_experts=nn.Identity(),
            decode_overlap_stream=None,
        ),
        Routed(),
    )
    block.ffn.forward_prepared = block.ffn.forward
    block.mixes = lambda hidden, _: (
        hidden.new_ones(*hidden.shape[:-1]),
        hidden.new_ones(*hidden.shape[:-1]),
        hidden.new_zeros(*hidden.shape[:-1], 1),
    )
    if lazy:

        def transition(layer, hidden, state, name, **kwargs):
            residual = implementation.materialize(hidden, state)
            pre, post, comb = block.mixes(residual, name)
            return residual.squeeze(2), implementation.HCLazyPost(
                residual, pre, post, comb
            )

        monkeypatch.setattr(implementation, "mega_transition", transition)
    block.markov_head = lambda ids: (torch.zeros(ids.numel(), 2), None)
    model = implementation.Transformer.__new__(implementation.Transformer)
    nn.Module.__init__(model)
    model.args, model.vision, model.parallel_groups = gate, nn.Identity(), None
    model.engram_hash, model.target_layer_ids, model.hc_mult = None, set(), 1
    model.layers, model.mtp, model.norm = (
        nn.ModuleList([block]),
        nn.ModuleList([block]),
        nn.Identity(),
    )
    model.dspark_block_size, model.dspark_noise_token_id = 3, 0
    model.embed = lambda ids: torch.ones(*ids.shape, 4)
    model.head = lambda hidden, **kwargs: hidden[..., :2]
    model._draft_attention = lambda attention, hidden, metadata, cache: hidden
    cache = SimpleNamespace(
        prepare_metadata=lambda positions, **kwargs: SimpleNamespace(
            positions=positions
        )
    )
    ids = torch.tensor([[129264]])
    if phase == "prefill":
        model.forward_prefill(ids, 0, cache=cache)
    elif phase == "decode":
        model.forward_decode(ids, torch.tensor([0]), 4, cache=cache)
    elif phase == "verify":
        model.forward_verify(ids.repeat(1, 3), torch.tensor([0]), 4, cache=cache)
    else:
        model.draft(ids, torch.tensor([0]), cache=cache)
    expected = [3, 2] if phase == "prefill" else [0, 1]
    assert routes[0].tolist() == [expected] * routes[0].shape[0]
