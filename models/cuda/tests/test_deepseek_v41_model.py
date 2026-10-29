"""Architecture regressions independent of the CUDA kernel stack."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn


@pytest.fixture
def implementation(monkeypatch):
    path = Path(__file__).parents[1] / "resin_models_cuda/deepseek_v41/model.py"
    spec = importlib.util.spec_from_file_location(
        "_resin_test_deepseek_v41_model", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fused", [False, True])
def test_shifted_hc_consumes_previous_pre_mix_and_returns_ffn_pre_mix(
    implementation, fused, monkeypatch
):
    block = implementation.Block.__new__(implementation.Block)
    nn.Module.__init__(block)
    block.args = SimpleNamespace(hc_mult=2)
    block.attn_norm = block.ffn_norm = nn.Identity()

    class FFN(nn.Module):
        def forward(self, hidden, _ids):
            return hidden

        forward_prepared = forward

    block.ffn, block.prefill_ffn, block.lazy_hc = FFN(), None, fused
    pre_values = {
        "attn": torch.tensor([[[2.0, 3.0]]]),
        "ffn": torch.tensor([[[5.0, 7.0]]]),
    }
    block.mixes = lambda hidden, name: (
        pre_values[name],
        torch.ones(1, 1, 2),
        torch.zeros(1, 1, 2, 2),
    )
    transitions = []
    if fused:

        def transition(layer, output, state, name, **kwargs):
            assert name == "ffn" and kwargs == {"prepare_ffn": True}
            transitions.append(state.pre)
            residual = implementation.materialize(output, state)
            next_pre, post, comb = layer.mixes(residual, name)
            return (
                layer.ffn_norm(layer.collapse(residual, state.pre)),
                implementation.HCLazyPost(residual, next_pre, post, comb),
            )

        monkeypatch.setattr(implementation, "mega_transition", transition)
    hidden = torch.tensor([[[[10.0, 20.0], [100.0, 200.0]]]])
    result, next_pre = block(
        hidden, torch.tensor([[[1.0, 0.0]]]), torch.tensor([[0]]), lambda value: value
    )
    if fused:
        result, next_pre = implementation.materialize(result, next_pre), next_pre.pre
    torch.testing.assert_close(result, torch.tensor([[[[50.0, 100.0], [50.0, 100.0]]]]))
    torch.testing.assert_close(next_pre, pre_values["ffn"])
    if fused:
        assert transitions == [pre_values["attn"]]


def test_deepgemm_preparation_enables_lazy_hc_across_both_sublayers(
    implementation,
):
    model = implementation.Transformer.__new__(implementation.Transformer)
    nn.Module.__init__(model)
    model.args = SimpleNamespace(dim=5120, hc_mult=4)
    model.engram_hash = None
    model.parallel_groups = SimpleNamespace(ep=object())
    block = implementation.Block.__new__(implementation.Block)
    nn.Module.__init__(block)
    block.lazy_hc = False
    model.layers = nn.ModuleList([block])
    model.prepare_weights(lambda value: value, expert_backend="deepgemm")
    assert block.lazy_hc


def test_hc_residual_combination_uses_official_source_destination_matrix_order(
    implementation,
):
    result = implementation.Block.expand(
        torch.zeros(1, 1, 2),
        torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]]),
        torch.ones(1, 1, 2),
        torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
    )
    torch.testing.assert_close(
        result, torch.tensor([[[[100.0, 140.0], [140.0, 200.0]]]])
    )


def test_dspark_collects_target_hidden_before_target_layer(implementation):
    model = implementation.Transformer.__new__(implementation.Transformer)
    nn.Module.__init__(model)
    model.engram_hash, model.hc_mult, model.target_layer_ids = None, 2, {1}
    model.embed = nn.Embedding(4, 2)
    model.embed.weight.data.fill_(1)
    model.norm = nn.Identity()

    class Layer(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer_id, self.engram = layer, None

        def forward(self, hidden, pre, ids, attention, *, prefill=False):
            return hidden + 10 * (self.layer_id + 1), pre

    model.layers = nn.ModuleList([Layer(0), Layer(1), Layer(2)])
    from test_deepseek_v41 import _cache

    _, cache = _cache()
    hidden, target = model._run(torch.tensor([[0]]), torch.tensor([[0]]), 1, cache)
    torch.testing.assert_close(target, torch.full((1, 1, 2), 11.0))
    torch.testing.assert_close(hidden, torch.full((1, 1, 2), 61.0))


def test_expert_tp4_balances_whole_fp4_kernel_blocks_without_dropping_columns(
    implementation,
):
    from resin.backend.cuda.parallel import TensorParallelLayout

    args = implementation.ModelArgs(moe_inter_dim=2304, expert_dtype="fp4")
    shards = [
        args.expert_intermediate_shard(TensorParallelLayout(rank=r, size=4))
        for r in range(4)
    ]
    assert [(s.start, s.stop) for s in shards] == [
        (0, 640),
        (640, 1280),
        (1280, 1792),
        (1792, 2304),
    ]
    assert all(s.size % 128 == 0 for s in shards)


@pytest.mark.parametrize("component", ["Attention", "Transformer"])
def test_single_kv_model_constructors_reject_tp_before_parameter_allocation(
    implementation, monkeypatch, component
):
    from resin.backend.cuda.parallel import TensorParallelContext, TensorParallelLayout

    module = implementation
    tp = TensorParallelContext(
        TensorParallelLayout(size=2), Mock(spec=torch.distributed.ProcessGroup)
    )

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "unsupported attention TP must fail before parameter allocation"
        )

    monkeypatch.setattr(torch, "empty", unexpected)
    with pytest.raises(ValueError, match="one KV head"):
        if component == "Transformer":
            module.Transformer(module.ModelArgs(), tensor_parallel=tp)
        else:
            module.Attention(0, module.ModelArgs(), tp)


def test_pinned_checkpoint_namespace_has_no_legacy_hc_head_or_draft_ownership(
    implementation,
):
    args = implementation.ModelArgs(
        vocab_size=128,
        dim=64,
        moe_inter_dim=128,
        n_layers=3,
        n_mtp_layers=2,
        n_heads=8,
        head_dim=512,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=1,
        compress_ratios=(0, 2, 1, 0, 0),
        kv_source_layers=(1, 2),
        index_source_layers=(1, 2),
        index_n_heads=8,
        n_routed_experts=8,
        n_activated_experts=6,
        dspark_n_routed_experts=4,
        dspark_n_activated_experts=3,
        dspark_target_layer_ids=(1, 2),
        dspark_block_size=4,
        expert_dtype="fp4",
        max_seq_len=128,
    )
    with torch.device("meta"):
        model = implementation.Transformer(args)
    names = set(model.state_dict())
    assert not any("hc_head" in name for name in names)
    assert "mtp.0.main_proj.weight" in names
    assert "mtp.1.confidence_head.proj.weight" in names
    assert "mtp.1.markov_head.embed.weight" in names
    assert "layers.1.attn.compressor.wgate.weight" in names
    assert "layers.2.attn.compressor.wgate.weight" not in names
    assert model.layers[1].attn.wq_a.scale.shape == (4, 2)
    assert model.layers[1].ffn.gate.topk == 6
    assert model.mtp[1].ffn.gate.topk == 3
    assert len(model.mtp[1].ffn.experts) == 4


@pytest.mark.parametrize(
    "ratio,layer,expected",
    [(1, 3, [129, 130, 131, 144, 144, 144]), (2, 1, [128, 129, 129, 144, 144, 144])],
)
def test_attention_bounds_sparse_active_lengths_for_short_queries(
    implementation, monkeypatch, ratio, layer, expected
):
    from test_deepseek_v41 import _cache

    _, cache = _cache()
    attention = implementation.Attention.__new__(implementation.Attention)
    nn.Module.__init__(attention)
    attention.layer_id, attention.ratio = layer, ratio
    attention.dim = attention.head_dim = 512
    attention.n_local_heads, attention.rope_head_dim, attention.window_size = 1, 64, 128
    attention.freqs_cis = torch.ones(256, 32, dtype=torch.complex64)
    attention.prefill_project, attention.prefill_query = None, None
    attention.kv_norm = attention.wq_b = nn.Identity()
    attention._project = lambda hidden: (hidden, hidden.clone())
    attention.prefill_finish = lambda output, positions: output.squeeze(2)
    attention.attn_sink, attention.attention_workspace = torch.zeros(1), torch.empty(0)
    attention.softmax_scale = 512**-0.5
    monkeypatch.setattr(
        implementation, "apply_rotary_emb", lambda values, *args, **kwargs: values
    )
    monkeypatch.setattr(implementation, "pack_kv", lambda value, **kwargs: value)
    monkeypatch.setattr(implementation, "store_kv", lambda *args: None)
    observed = {}

    def fused(attention, query, positions, window, indices, **kwargs):
        observed.update(
            positions=positions, indices=indices, extra=kwargs["extra_indices"]
        )
        return torch.zeros_like(query)

    monkeypatch.setattr(implementation, "fused_attention", fused)
    attention._finish = lambda output, positions: output.squeeze(2)
    positions = torch.tensor([[0, 1, 2, 127, 128, 129]])
    selected = torch.arange(16).expand(1, 6, -1)
    selected = selected.masked_fill(
        selected >= ((positions + 1) // ratio)[..., None], -1
    )
    attention(
        torch.zeros(1, 6, 512, dtype=torch.bfloat16),
        positions,
        256,
        cache,
        {"indices": selected, "index_source": layer},
    )
    assert (observed["extra"] >= 0).sum(-1).tolist() == [
        [value - 128 for value in expected]
    ]
    assert (observed["indices"] >= 0).sum(-1).tolist() == [[1, 2, 3, 128, 128, 128]]
    assert observed["indices"][0, 0, 0] >= 0
    assert (observed["indices"][0, 0, 1:] == -1).all()
    assert torch.equal(observed["positions"], positions)


@pytest.mark.parametrize("prefill,batch_size", [(False, 4), (True, 1)])
def test_batched_verification_bypasses_single_sequence_prefill_callbacks(
    implementation, monkeypatch, prefill, batch_size
):
    attention = implementation.Attention.__new__(implementation.Attention)
    nn.Module.__init__(attention)
    attention.layer_id, attention.ratio = 0, 0
    attention.dim = attention.head_dim = attention.rope_head_dim = 64
    attention.n_local_heads, attention.window_size = 1, 128
    attention.freqs_cis = torch.polar(
        torch.ones(32, 32), torch.arange(32)[:, None].expand(-1, 32).float() / 10
    )
    attention.kv_norm = attention.wq_b = nn.Identity()
    attention._project = lambda hidden: (hidden.clone(), hidden.clone())
    attention._finish = lambda output, positions: output.squeeze(2)
    attention.attn_sink, attention.attention_workspace = torch.zeros(1), None
    attention.softmax_scale = 64**-0.5
    calls = []

    # Match the runner's one-sequence adapters, including their batch removal
    # and insertion. Applying these to a real batch changes tensor rank.
    def project(hidden):
        calls.append("project")
        return tuple(
            value.squeeze(0).unsqueeze(0)
            for value in attention._project(hidden.squeeze(0)[None])
        )

    def query(hidden):
        calls.append("query")
        return attention.wq_b(hidden.squeeze(0))[None]

    def finish(output, positions):
        calls.append("finish")
        return attention._finish(output.squeeze(0)[None], positions.squeeze(0)[None])

    attention.prefill_project, attention.prefill_query, attention.prefill_finish = (
        project,
        query,
        finish,
    )
    monkeypatch.setattr(implementation, "pack_kv", lambda value, **kwargs: value)
    monkeypatch.setattr(implementation, "store_kv", lambda *args: None)
    monkeypatch.setattr(
        implementation,
        "fused_attention",
        lambda attention, query, *args, **kwargs: query,
    )
    monkeypatch.setattr(
        "resin_models_cuda.deepseek_v41.attention.gather_kv",
        lambda packed, indices: torch.zeros(indices.numel(), 64),
    )
    from test_deepseek_v41 import _cache

    _, cache = _cache(batch_size)
    hidden = torch.randn(batch_size, 5, 64)
    positions = torch.arange(5)[None].repeat(batch_size, 1)
    result = attention(hidden, positions, 5, cache, {}, prefill=prefill)
    torch.testing.assert_close(result, hidden)
    assert result.shape == (batch_size, 5, 64)
    assert calls == (["project", "query", "finish"] if prefill else [])


def test_only_prompt_forward_enables_prefill_projection_callbacks(implementation):
    model = implementation.Transformer.__new__(implementation.Transformer)
    nn.Module.__init__(model)
    model.parallel_groups = None
    model.head = lambda hidden, **kwargs: hidden
    modes = []

    def run(ids, positions, capacity, cache, cp=None, *, prefill=False, **kwargs):
        modes.append(prefill)
        return ids, None

    model._run = run
    model.forward_prefill(torch.zeros(1, 5, dtype=torch.long), 0, cache=None)
    model.forward_decode(
        torch.zeros(4, 1, dtype=torch.long), torch.arange(4), 16, cache=None
    )
    model.forward_verify(
        torch.zeros(4, 5, dtype=torch.long), torch.arange(4), 16, cache=None
    )
    assert modes == [True, False, False]


@pytest.mark.parametrize("replay_start,offset", [(132, 6), (200, 7)])
def test_replay_keeps_full_source_and_compacts_consumer_state(
    implementation, replay_start, offset
):
    from test_deepseek_v41 import _cache

    calls, sources = [], []

    class Attention(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def prepare_shared_kv(self, hidden, positions, cache, runtime):
            sources.append((hidden.clone(), positions.clone()))
            runtime["full_index_kv_stored"] = {self.layer}

        def forward(self, hidden, positions, capacity, cache, runtime, cp, **kwargs):
            frame = runtime["metadata"]
            calls.append((self.layer, positions.clone(), frame.slot_mapping_store_swa))
            return hidden + 1

    class Layer(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.layer_id, self.engram = index, None
            self.attn, self.attn_norm = Attention(index), nn.Identity()

        def forward(self, hidden, pre, ids, attention, **kwargs):
            value = implementation.Block.collapse(hidden, pre)
            return attention(value).unsqueeze(2), pre

    model = implementation.Transformer.__new__(implementation.Transformer)
    nn.Module.__init__(model)
    model.parallel_groups, model.engram_hash, model.vision = None, None, None
    model.hc_mult, model.prefill_boundary = 1, 1
    model.target_layer_ids = {2}
    model.embed = lambda ids: ids.float()[..., None].expand(-1, -1, 2)
    model.norm, model.head = nn.Identity(), nn.Identity()
    model.layers = nn.ModuleList(Layer(index) for index in range(3))
    _, cache = _cache()
    ids = torch.arange(8)[None]
    full, full_main = model.forward_prefill(
        ids, 126, cache=cache, replay_start=replay_start, compact_replay=False
    )
    calls.clear()
    compact, main = model.forward_prefill(
        ids, 126, cache=cache, replay_start=replay_start
    )
    assert [positions.shape[1] for _, positions, _ in calls] == [
        8,
        8 - offset,
        8 - offset,
    ]
    assert all(
        positions.tolist() == [list(range(126, 134))] for _, positions in sources
    )
    assert torch.equal(sources[0][0], sources[1][0])
    torch.testing.assert_close(compact, full[:, offset:])
    torch.testing.assert_close(main, full_main[:, offset:])
    for _, positions, stores in calls[1:]:
        assert positions.tolist() == [list(range(126 + offset, 134))]
        assert torch.equal(stores >= 0, positions >= replay_start)


@pytest.mark.parametrize(
    "mode,scored,context,count,retained",
    [
        ("compact", False, 0, 4096, 1),
        ("compact", False, 130900, 172, 140),
        ("mask", False, 130900, 172, 172),
        ("compact", True, 0, 4096, 4096),
        ("off", False, 0, 4096, 4096),
    ],
)
def test_runner_replay_policy_keeps_dspark_absolute_positions_and_full_scores(
    mode, scored, context, count, retained
):
    from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner

    runner = object.__new__(DeepSeekV41CudaRunner)
    runner._swa_replay_tokens, runner._swa_replay_mode = 140, mode
    runner.speculative_draft_tokens = 3
    stores = []

    def forward(
        ids, start, *, replay_start, compact_replay, return_hidden, cache, inputs_embeds
    ):
        assert inputs_embeds is None
        assert return_hidden == scored
        assert replay_start == (None if scored or mode == "off" else 130932)
        rows = retained
        return torch.zeros(1, rows, 2), torch.zeros(1, rows, 2)

    def commit(hidden, position, *, cache, valid_tokens, prefill):
        assert prefill is True
        stores.append((position, valid_tokens))

    runner._model = SimpleNamespace(
        prefill_boundary=20,
        layers=range(40),
        forward_prefill=forward,
        commit_dspark_main_block=commit,
    )
    plan = SimpleNamespace(
        needs_prompt_logprobs=scored,
        media=[None],
        context_lengths=[context],
        prefill_lengths=[131072],
    )
    positions = torch.arange(context, context + count)
    runner._prefill_row(
        plan, 0, torch.zeros(1, count, dtype=torch.long), positions, None
    )
    assert stores[0][0].tolist() == [context + count - retained]
    valid = stores[0][1]
    if scored or mode == "off":
        assert valid is None
    else:
        assert torch.equal(valid, positions[count - retained :] >= 130932)
    demands = runner._prefill_region_tokens(plan, ((0, count),))
    assert demands == ({} if mode == "off" else {range(20, 40): retained})
