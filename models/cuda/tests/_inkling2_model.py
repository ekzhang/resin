"""Independent tiny-model geometry, dense cache, and checkpoint schema."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from resin.models.inkling2.config import Inkling2Config


def tiny_config() -> Inkling2Config:
    return Inkling2Config(
        vocab_size=24,
        unpadded_vocab_size=21,
        hidden_size=8,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=4,
        d_rel=2,
        rel_extent=8,
        local_layer_ids=(0, 1, 3, 4),
        sliding_window_size=4,
        sconv_kernel_size=4,
        n_layers_per_attn_res_block=2,
        attn_res_normalize_values=True,
        intermediate_size=12,
        num_routed_experts=4,
        experts_per_token=2,
        route_scale=2.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        qk_l2_clip_threshold=3.0,
        rel_l2_clip_threshold=2.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        log_scaling_n_floor=4,
        log_scaling_alpha=0.1,
        final_logit_softcapping=3.0,
        eos_token_id=20,
    )


class DenseCache:
    def __init__(self, config, batch, capacity=128):
        self.page_tokens, self.slab_units = capacity, 1
        self.counts = {
            0: config.execution_global_layer_count,
            1: config.execution_local_layer_count,
        }
        self.writes = []
        self.groups = [
            SimpleNamespace(first_blocks=torch.zeros(batch, dtype=torch.int64))
            for _ in range(2)
        ]
        self.keys = torch.zeros(
            batch * config.execution_layer_count,
            self.page_tokens,
            config.num_key_value_heads,
            config.head_dim,
            dtype=torch.bfloat16,
        )
        self.values = torch.zeros_like(self.keys)
        self.batch = batch

    def pages(self, group, layer, count):
        base = 0 if group == 0 else self.batch * self.counts[0]
        return (
            base + torch.arange(self.batch, device=self.keys.device) * count + layer
        )[:, None]

    def page_locations(self, *, group_index, layer, layer_count, positions):
        return self.pages(group_index, layer, layer_count).expand_as(
            positions
        ), positions

    def store_kv(self, keys, values, positions, *, group_index, layer, layer_count):
        assert layer_count == self.counts[group_index]
        self.writes.append((group_index, layer))
        pages, offsets = self.page_locations(
            group_index=group_index,
            layer=layer,
            layer_count=layer_count,
            positions=positions,
        )
        self.keys[pages, offsets], self.values[pages, offsets] = keys, values


def dense_attention(
    q,
    *,
    cache,
    group_index,
    layer,
    layer_count,
    positions,
    sequence_lengths,
    softmax_scale,
    window_left,
    relative_logits,
    **kwargs,
):
    outputs, stats = [], []
    for row in range(q.shape[0]):
        page = cache.pages(group_index, layer, layer_count)[row, 0]
        length = int(sequence_lengths[row])
        k, v = cache.keys[page, :length].float(), cache.values[page, :length].float()
        groups = q.shape[2] // k.shape[1]
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        logits = torch.einsum("thd,shd->ths", q[row].float(), k) * softmax_scale
        distance = positions[row, :, None] - torch.arange(length, device=q.device)[None]
        rel = relative_logits.view(*q.shape[:3], -1)[row]
        bias = rel.gather(
            2, distance.clamp(0, rel.shape[-1] - 1)[:, None].expand(-1, q.shape[2], -1)
        )
        logits += bias * ((distance >= 0) & (distance < rel.shape[-1]))[:, None]
        mask = distance >= 0
        if window_left is not None:
            mask &= distance <= window_left
        logits.masked_fill_(~mask[:, None], float("-inf"))
        outputs.append(torch.einsum("ths,shd->thd", logits.softmax(-1), v))
        stats.append(logits.logsumexp(-1))
    return torch.stack(outputs), torch.cat(stats, 0).t()


def reference_experts(experts, hidden, ids, weights):
    result = torch.zeros_like(hidden, dtype=torch.float32)
    for token in range(hidden.shape[0]):
        for slot in range(ids.shape[1]):
            expert = ids[token, slot]
            # Checkpoint W13 stores adjacent gate/up rows.
            gate, up = (
                F.linear(hidden[token], experts.w13_weight[expert])
                .float()
                .view(-1, 2)
                .unbind(-1)
            )
            activated = (
                4
                * (gate / 4).tanh()
                * gate.sigmoid()
                * 25
                * (up / 25).tanh()
                * weights[token, slot]
            ).bfloat16()
            result[token] += F.linear(activated, experts.w2_weight[expert]).float()
    return result


def checkpoint_tensors(config):
    """Independent published tensor schema, before projection packing or sharding."""
    h, d, r = config.hidden_size, config.head_dim, config.d_rel
    q, kv = config.num_attention_heads, config.num_key_value_heads
    shapes = {
        "embed.weight": (config.vocab_size, h),
        "embed_norm.weight": (h,),
        "norm.weight": (h,),
        "unembed.weight": (config.vocab_size, h),
        "attn_res.output_query": (h,),
    }
    if loop := config.loop:
        query_shape = (2 * (loop.core_end - loop.core_start), h)
        shapes["middle_cycle_cross_pass_queries"] = query_shape
        shapes["middle_cycle_cross_pass_kv_queries"] = query_shape
    for i in range(config.num_hidden_layers):
        local = i in config.local_layer_ids
        if i:
            shapes[f"attn_res.attn_queries.{i - 1}"] = (h,) if local else (3, h)
        shapes[f"attn_res.mlp_queries.{i}"] = (h,)
        prefix = f"layers.{i}."
        for name in (
            ("attn_norm",)
            if local
            else ("q_input_norm", "k_input_norm", "v_input_norm")
        ) + ("mlp_norm",):
            shapes[prefix + name + ".weight"] = (h,)
        for name, size in (
            ("wq_du", q * d),
            ("wk_dv", kv * d),
            ("wv_dv", kv * d),
            ("wr_du", q * r),
            ("gate_attn", q * d),
            ("attn_sink_proj", q),
        ):
            shapes[prefix + "attn." + name + ".weight"] = (size, h)
        shapes[prefix + "attn.wo_ud.weight"] = (h, q * d)
        shapes[prefix + "attn.q_l2_clip.scale"] = (q, d)
        shapes[prefix + "attn.rel_l2_clip.scale"] = (q, r)
        shapes[prefix + "attn.rel_logits_proj.proj"] = (
            r,
            config.sliding_window_size if local else config.rel_extent,
        )
        for name in ("k_sconv", "v_sconv"):
            if local:
                shapes[prefix + "attn." + name + ".weight"] = (kv * d, 1, 4)
            else:
                shapes[prefix + "attn." + name + ".static_w"] = (4, kv * d)
                shapes[prefix + "attn." + name + ".weight_proj"] = (kv * 4, h)
        for name in ("attn_sconv", "mlp_sconv"):
            shapes[prefix + name + ".weight"] = (h, 1, 4)
        shapes[prefix + "mlp.gate.weight"] = (config.num_routed_experts, h)
        shapes[prefix + "mlp.gate.bias"] = (config.num_routed_experts,)
        shapes[prefix + "mlp.experts.w13_weight"] = (
            config.num_routed_experts,
            2 * config.intermediate_size,
            h,
        )
        shapes[prefix + "mlp.experts.w2_weight"] = (
            config.num_routed_experts,
            h,
            config.intermediate_size,
        )
        if loop and loop.core_start <= i < loop.core_end:
            for stream in ("attn", "k", "v", "mlp"):
                controller = prefix + f"controllers.{stream}."
                shapes[controller + "down_weight"] = (loop.gate_rank, 2 * h)
                shapes[controller + "gain_weight"] = (h, loop.gate_rank)
                shapes[controller + "forget_weight"] = (h, loop.gate_rank)
    return {
        "model.llm." + name: torch.randn(
            shape,
            dtype=torch.float32
            if name.startswith(("attn_res.", "middle_cycle_cross_pass_"))
            or name.endswith("gate.bias")
            else torch.bfloat16,
        )
        for name, shape in shapes.items()
    }
