from __future__ import annotations

from types import SimpleNamespace

import torch

from resin import BatchPhase
from resin.backend.cuda.paged_kv import DevicePagedAttentionBatch, PagedKvTable
from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner


def _runner():
    runner = GptOssCudaRunner.__new__(GptOssCudaRunner)
    runner.hidden_size, runner.num_query_heads, runner.num_kv_heads, runner.head_dim = (
        4,
        2,
        1,
        2,
    )
    setattr(runner, "tensor_parallel", SimpleNamespace(enabled=False))
    runner._prefill_graph_active = True
    runner._full_layer_slots = {0: 0}
    setattr(
        runner,
        "_kv",
        SimpleNamespace(
            keys=[torch.zeros(4, 4, 1, 2, dtype=torch.bfloat16)],
            values=[torch.zeros(4, 4, 1, 2, dtype=torch.bfloat16)],
        ),
    )
    runner.softmax_scale = 2**-0.5
    runner.sliding_window = 128
    attention = SimpleNamespace(
        qkv_proj=torch.nn.Linear(4, 8, bias=True, dtype=torch.bfloat16),
        o_proj=torch.nn.Linear(4, 4, bias=True, dtype=torch.bfloat16),
        sinks=torch.zeros(2, dtype=torch.bfloat16),
        layer_type="full_attention",
    )
    calls = []

    def project(hidden, cos, sin):
        calls.append(("project", hidden.shape[0]))
        return runner._attention_project(attention, hidden, cos, sin)

    runner._prefill_attention = (project,)

    def rotary(hidden, cos, sin):
        first, second = hidden.chunk(2, -1)
        cos, sin = cos.squeeze(0).unsqueeze(1), sin.squeeze(0).unsqueeze(1)
        hidden.copy_(
            torch.cat((first * cos - second * sin, second * cos + first * sin), -1)
        )
        return hidden

    setattr(runner, "_apply_yarn_rotary", rotary)
    return runner, attention, calls


def _batch(slots, lengths):
    group = PagedKvTable(
        page_table=torch.tensor([[1, 2]], dtype=torch.int32),
        lengths=torch.tensor(lengths, dtype=torch.int32),
        append_slots=torch.tensor(slots, dtype=torch.long),
        min_length=min(lengths),
        max_length=max(lengths),
    )
    return DevicePagedAttentionBatch(
        phase=BatchPhase.PREFILL,
        token_indices=torch.arange(len(slots)),
        sequence_indices=torch.tensor([0]),
        last_token_indices=torch.tensor([len(slots) - 1]),
        cu_seqlens_q=torch.tensor([0, len(slots)], dtype=torch.int32),
        max_seqlen_q=len(slots),
        groups=(("full", group),),
    )


def test_projection_regions_leave_actual_queries_and_page_writes_with_eager_attention():
    runner, attention, calls = _runner()
    attended = []

    def fa4(query, keys, values, **metadata):
        attended.append((query.shape[0], metadata))
        return query

    runner._fa4 = fa4
    expected_keys = runner._kv.keys[0].clone().view(-1, 1, 2)
    expected_values = runner._kv.values[0].clone().view(-1, 1, 2)
    with torch.inference_mode():
        for slots, context in (([4, 9, 12], 17), ([7], 35)):
            count = len(slots)
            hidden = torch.arange(count * 4, dtype=torch.bfloat16).view(1, count, 4) / 8
            cos = torch.full((1, count, 1), 0.5, dtype=torch.bfloat16)
            sin = torch.ones_like(cos)
            batch = _batch(slots, [context])
            projected = runner._attention_project(
                attention, hidden.squeeze(0), cos.squeeze(0), sin.squeeze(0)
            )
            query, key, value = projected.split((4, 2, 2), -1)
            expected_keys[slots] = key.view(count, 1, 2)
            expected_values[slots] = value.view(count, 1, 2)

            output = runner._attention(0, attention, hidden, (cos, sin), batch)

            torch.testing.assert_close(output, attention.o_proj(query).unsqueeze(0))
            assert calls[-1:] == [("project", count)]
            assert attended[-1][0] == count
            assert attended[-1][1]["cu_seqlens_q"] is batch.cu_seqlens_q
            assert attended[-1][1]["seqused_k"] is batch.group("full").lengths
            assert attended[-1][1]["page_table"] is batch.group("full").page_table
        torch.testing.assert_close(runner._kv.keys[0].view(-1, 1, 2), expected_keys)
        torch.testing.assert_close(runner._kv.values[0].view(-1, 1, 2), expected_values)
        assert not runner._kv.keys[0][0].count_nonzero()
        assert not runner._kv.values[0][0].count_nonzero()


def test_empty_context_query_shard_still_stores_gathered_real_kv():
    runner, attention, calls = _runner()
    runner._fa4 = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("no local Q")
    )
    gathered = []

    def gather(local):
        assert local.shape == (0, 1, 2)
        gathered.append(True)
        return torch.full((3, 1, 2), len(gathered), dtype=torch.bfloat16)

    with torch.inference_mode():
        empty = torch.empty(1, 0, 4, dtype=torch.bfloat16)
        rotary = torch.empty(1, 0, 1, dtype=torch.bfloat16)
        output = runner._attention(
            0,
            attention,
            empty,
            (rotary, rotary),
            _batch([4, 7, 12], [3]),
            context_parallel=SimpleNamespace(gather=gather),
        )

    assert output.shape == (1, 0, 4) and output.dtype == torch.bfloat16
    assert calls == [("project", 0)]
    assert len(gathered) == 2
    torch.testing.assert_close(
        runner._kv.keys[0].view(-1, 1, 2)[[4, 7, 12]],
        torch.ones(3, 1, 2, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        runner._kv.values[0].view(-1, 1, 2)[[4, 7, 12]],
        torch.full((3, 1, 2), 2, dtype=torch.bfloat16),
    )
