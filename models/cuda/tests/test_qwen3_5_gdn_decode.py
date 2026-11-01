import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def test_in_place_decode_matches_fla_recurrence():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    from resin_models_cuda.qwen3_5.gdn_decode import RecurrentStatePool, gdn_decode

    torch.manual_seed(0)
    B, H, HV, K, V, L = 5, 16, 32, 128, 128, 3
    slots = 2 * B + 1
    state_elems = L * HV * K * V
    arena = torch.randn(slots, state_elems, device="cuda", dtype=torch.float32) * 0.1
    states = arena.view(slots, L, HV, K, V)
    layer = 1
    source = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    # Row 2 crosses a state block: it writes a different slot than it reads.
    destination = torch.tensor([1, 2, 9, 4, 5], device="cuda")
    pool = RecurrentStatePool(
        states, source * state_elems, destination * state_elems, HV * K * V
    )
    mixed = torch.randn(B, 2 * H * K + HV * V, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(B, HV, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(B, HV, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.bfloat16)

    initial = states[source, layer].clone()
    q, k, v = torch.split(mixed, [H * K, H * K, HV * V], dim=-1)
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    ref_o, ref_state = fused_recurrent_gated_delta_rule(
        q.view(B, 1, H, K),
        k.view(B, 1, H, K),
        v.view(B, 1, HV, V),
        g=g[:, None],
        beta=beta[:, None],
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    untouched = states[6].clone()
    out = gdn_decode(
        mixed[:, None],
        a[:, None],
        b[:, None],
        A_log,
        dt_bias,
        pool,
        layer,
        num_key_heads=H,
        num_value_heads=HV,
        key_dim=K,
        value_dim=V,
    )
    torch.testing.assert_close(
        out[:, 0].float(), ref_o[:, 0].float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        states[destination, layer], ref_state, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(states[6], untouched)


@pytest.mark.parametrize("channels", [512, 2560, 8192])
def test_conv_decode_matches_torch_and_shifts_window(channels):
    from resin_models_cuda.qwen3_5.gdn_decode import RecurrentStatePool, conv_decode

    torch.manual_seed(1)
    B, C, W, L, slots = 4, channels, 4, 3, 9
    conv_states = torch.randn(slots, L, C, W - 1, device="cuda", dtype=torch.bfloat16)
    layer = 2
    source = torch.tensor([1, 2, 3, 4], device="cuda")
    destination = torch.tensor([1, 2, 7, 4], device="cuda")
    per_slot = L * C * (W - 1)
    states = torch.zeros(1, device="cuda")
    pool = RecurrentStatePool(
        states,
        source,
        destination,
        0,
        convolution_states=conv_states,
        convolution_source=source * per_slot,
        convolution_destination=destination * per_slot,
        convolution_layer_stride=C * (W - 1),
    )
    x = torch.randn(B, C, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(C, 1, W, device="cuda", dtype=torch.bfloat16)
    untouched_slot = conv_states[5].clone()
    untouched_layers = conv_states[:, :layer].clone()
    window = torch.cat((conv_states[source, layer], x[:, :, None]), dim=-1)
    ref = F.silu(F.conv1d(window, weight, groups=C))[:, :, 0]
    out = conv_decode(x, weight, pool, layer)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        conv_states[destination, layer], window[..., 1:], atol=0, rtol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = conv_decode(x, weight, pool, layer)
    x.copy_(torch.randn_like(x))
    window = torch.cat((conv_states[source, layer], x[:, :, None]), dim=-1)
    expected = F.silu(F.conv1d(window, weight, groups=C))[:, :, 0]
    graph.replay()
    torch.testing.assert_close(replayed, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        conv_states[destination, layer], window[..., 1:], atol=0, rtol=0
    )
    torch.testing.assert_close(conv_states[5], untouched_slot, atol=0, rtol=0)
    torch.testing.assert_close(conv_states[:, :layer], untouched_layers, atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [2, 4])
def test_verified_prefix_commit_matches_sequential_recurrence(tokens):
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    from resin_models_cuda.qwen3_5.gdn_decode import (
        RecurrentStatePool,
        gdn_commit,
        gdn_decode,
    )

    torch.manual_seed(4)
    B, H, HV, K, V, L, T = 4, 16, 32, 128, 128, 2, tokens
    state_elems = L * HV * K * V
    for accepted_value in range(T):
        arena = torch.randn(8, state_elems, device="cuda", dtype=torch.float32) * 0.1
        states = arena.view(8, L, HV, K, V)
        source = torch.tensor([0, 1, 2, 3], device="cuda")
        stored = torch.tensor([4, 5, 6, 7], device="cuda")
        committed = (
            stored if accepted_value == 0 else torch.tensor([4, 5, 6, 7], device="cuda")
        )
        pending = torch.empty(L, B, T - 1, HV, K + V + 2, device="cuda")
        pool = RecurrentStatePool(
            states,
            source * state_elems,
            stored * state_elems,
            HV * K * V,
            pending=pending,
        )
        mixed = torch.randn(
            B, T, 2 * H * K + HV * V, device="cuda", dtype=torch.bfloat16
        )
        a = torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16)
        A_log = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
        dt_bias = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
        layer = 1
        initial = states[source, layer].clone()
        out = gdn_decode(
            mixed,
            a,
            b,
            A_log,
            dt_bias,
            pool,
            layer,
            num_key_heads=H,
            num_value_heads=HV,
            key_dim=K,
            value_dim=V,
        )
        q, k, v = torch.split(mixed, [H * K, H * K, HV * V], dim=-1)
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
        ref_o, ref_state = fused_recurrent_gated_delta_rule(
            q.reshape(B, T, H, K),
            k.reshape(B, T, H, K),
            v.reshape(B, T, HV, V),
            g=g,
            beta=b.sigmoid(),
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        _, first_state = fused_recurrent_gated_delta_rule(
            q[:, :1].reshape(B, 1, H, K),
            k[:, :1].reshape(B, 1, H, K),
            v[:, :1].reshape(B, 1, HV, V),
            g=g[:, :1],
            beta=b[:, :1].sigmoid(),
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(out.float(), ref_o.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            states[stored, layer], first_state, atol=1e-4, rtol=1e-4
        )
        accepted = torch.full((B,), accepted_value, device="cuda", dtype=torch.long)
        gdn_commit(
            pool,
            L,
            pool.destination,
            committed * state_elems,
            accepted,
            num_value_heads=HV,
            key_dim=K,
            value_dim=V,
        )
        if accepted_value == T - 1:
            expected = ref_state
        elif accepted_value == 0:
            expected = first_state
        else:
            kept = accepted_value + 1
            _, expected = fused_recurrent_gated_delta_rule(
                q[:, :kept].reshape(B, kept, H, K),
                k[:, :kept].reshape(B, kept, H, K),
                v[:, :kept].reshape(B, kept, HV, V),
                g=g[:, :kept],
                beta=b[:, :kept].sigmoid(),
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        torch.testing.assert_close(
            states[committed, layer], expected, atol=1e-4, rtol=1e-4
        )


def _tiny_runner_config():
    from resin.models.qwen3_5.config import Qwen35Config

    config = Qwen35Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        max_position_embeddings=8704,
        rms_norm_eps=1e-6,
        rope_theta=10_000_000.0,
        partial_rotary_factor=1.0,
        mrope_section=(8, 12, 12),
        layer_types=("linear_attention", "full_attention"),
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        eos_token_id=0,
        bos_token_id=1,
    )
    config.validate()
    return config


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9,
    reason="the real Qwen runner requires Hopper or newer",
)
@pytest.mark.parametrize(
    "draft_tokens,temperature",
    [
        (0, 0.0),
        (1, 0.0),
        (3, 0.0),
        (3, 1.0),
    ],
)
def test_tiny_runner_overlap_matches_sync(
    monkeypatch, tmp_path, draft_tokens, temperature
):
    """Compare real decode paths using a small model and a padded graph batch."""
    import gc

    from resin import Engine, SamplingParams, SchedulerConfig
    from resin_models_cuda.qwen3_5 import cuda as runtime
    from resin_models_cuda.qwen3_5.model import Qwen3_5ForConditionalGeneration

    config = _tiny_runner_config()

    def load_random_model(_path, *, device, include_mtp, **_kwargs):
        # Fresh, identical CPU initialization for every independently bound runner.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(41)
            model = Qwen3_5ForConditionalGeneration(config, include_mtp=include_mtp)
        model = (
            model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        )
        model.language_model.materialize_buffers(torch.device(device))
        return model, config, {}

    monkeypatch.setattr(runtime, "resolve_model_path", lambda *args: tmp_path)
    monkeypatch.setattr(runtime, "load_model", load_random_model)
    modes = [
        ("disabled", False, draft_tokens, False),
        ("full", False, draft_tokens, False),
        ("full", True, draft_tokens, False),
    ]
    if draft_tokens and temperature == 0.0:
        modes.append(("full", False, 0, False))
    if draft_tokens in (0, 3):
        modes.append(("full", True, draft_tokens, True))
    outputs, statistics, cached_outputs = [], [], None
    for graph_mode, overlap, mode_drafts, prefix_cache in modes:
        runner = runtime.Qwen3_5CudaRunner(
            tmp_path,
            device="cuda",
            speculative_draft_tokens=mode_drafts,
            max_seq_len=config.max_position_embeddings,
            max_batch_size=4,
            cache_slab_capacity=1024,
            kv_cache_dtype="bfloat16",
            cuda_graph_prefill="disabled",
            cuda_graph_decode=graph_mode,
            cuda_graph_decode_shapes=(
                [(4, config.max_position_embeddings)] if graph_mode == "full" else []
            ),
        )
        runner.capture_cuda_graphs()
        engine = Engine(
            runner,
            max_running_requests=3,
            context_length=config.max_position_embeddings,
            scheduler_config=SchedulerConfig(prefill_chunk_size=4096),
            prefix_cache=prefix_cache,
            enable_overlap=overlap,
        )
        mode_outputs, mode_statistics = [], []

        def prompt(length, row):
            return [1] + [
                2 + (index * 17 + row * 11) % 62 for index in range(length - 1)
            ]

        try:
            if prefix_cache:
                warmup = [
                    engine.submit(
                        prompt(4096, row),
                        sampling_params=SamplingParams(
                            max_output_tokens=1,
                            stop_token_ids=[],
                            temperature=temperature,
                            top_p=0.95,
                            top_k=64,
                            seed=41,
                        ),
                    )
                    for row in range(3)
                ]
                while engine.has_work():
                    engine.step()
                assert all(len(lease.take_result()) == 1 for lease in warmup)
            for length in (4097,) if prefix_cache else (125, 4093, 4097):
                leases = [
                    engine.submit(
                        prompt(length, row),
                        sampling_params=SamplingParams(
                            max_output_tokens=40,
                            stop_token_ids=[],
                            temperature=temperature,
                            top_p=0.95,
                            top_k=64,
                            seed=41,
                        ),
                    )
                    for row in range(3)
                ]
                while engine.has_work():
                    engine.step()
                if prefix_cache:
                    assert all(lease.cached_tokens >= 4096 for lease in leases)
                values = [lease.take_result() for lease in leases]
                assert all(len(value) == 40 for value in values)
                mode_outputs.append(values)
                stats = dict(runner.speculative_stats)
                assert stats["drafts"] == mode_drafts * stats["blocks"]
                assert stats["emitted"] == stats["blocks"] + stats["accepted"]
                assert 0 <= stats["accepted"] <= stats["drafts"]
                if mode_drafts:
                    # Each final raw block can exceed the requested output by
                    # at most K tokens; prefill contributed one token per row.
                    emitted = len(mode_outputs) * 3 * 39
                    assert (
                        emitted
                        <= stats["emitted"]
                        <= emitted + len(mode_outputs) * 3 * mode_drafts
                    )
                mode_statistics.append(stats)
            if overlap:
                assert engine.overlap_stats[1] > 0
            if prefix_cache:
                cached_outputs = mode_outputs[0]
            else:
                outputs.append(mode_outputs)
                statistics.append(mode_statistics)
        finally:
            engine.close()
            runner.close_cuda_graphs()
            del engine, runner
            gc.collect()
            torch.cuda.empty_cache()
    assert outputs[1] == outputs[2]
    if temperature == 0.0:
        assert all(output == outputs[0] for output in outputs[1:])
    # Identical physical graph geometry must make exactly the same decisions.
    # The smaller eager batch can round an assistant logit tie differently;
    # Greedy verified output remains equal to the target-only sequence above.
    # Shared sampling tests cover the probability and RNG contract. BF16
    # target-logit or nucleus-cutoff ties can change seeded categorical draws
    # between eager/graph shapes or single-token/block verification.
    assert statistics[1] == statistics[2]
    if cached_outputs is not None:
        assert cached_outputs == outputs[1][-1]


@pytest.mark.parametrize(
    "tokens,kv_dtype",
    [(1, torch.bfloat16), (4, torch.bfloat16), (1, torch.float8_e4m3fn)],
)
def test_shifted_mtp_attention_masks_origin_and_replays_pages(tokens, kv_dtype):
    """Compare the real shifted-page kernel with an independent causal oracle."""
    from resin_models_cuda.qwen3_5.attention import (
        GraphAttentionSpec,
        GraphPagedAttentionRuntime,
    )
    from resin_models_cuda.qwen3_5.cache import CacheArena, CacheGeometry

    torch.manual_seed(17)
    config = _tiny_runner_config()
    geometry = CacheGeometry.from_config(
        config, include_mtp=True, kv_element_bytes=kv_dtype.itemsize
    )
    arena = CacheArena("cuda", geometry.layout(4), geometry, kv_dtype=kv_dtype)
    cache = arena.graph_cache(2, 512)
    for group in cache.groups:
        group.slab_ids.fill_(1)
        group.slotnums.copy_(torch.arange(group.slotnums.shape[1], device="cuda")[None])
        group.first_blocks.zero_()
    cache.groups[2].slab_ids[1].fill_(2)
    arena.keys.copy_(torch.randn_like(arena.keys, dtype=torch.bfloat16).to(kv_dtype))
    arena.values.copy_(
        torch.randn_like(arena.values, dtype=torch.bfloat16).to(kv_dtype)
    )
    # The unused origin would strongly alter the answer if it were attended.
    for row in range(2):
        physical = cache.row(row).rows(
            2, 0, 1, torch.zeros(1, 1, device="cuda", dtype=torch.long)
        )
        arena.values[
            physical.div(128, rounding_mode="floor"), physical.remainder(128)
        ] = 128
    queries = torch.randn(
        2,
        tokens,
        config.num_attention_heads,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.tensor([125, 127], device="cuda")
    runtime = GraphPagedAttentionRuntime(
        torch.device("cuda"),
        cache,
        (GraphAttentionSpec(2, 1, tokens, tokens, 1),),
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        workspace=torch.empty(64 * 1024**2, device="cuda", dtype=torch.uint8),
    )

    def run():
        runtime.reset()
        runtime.refresh(positions)
        prepared = runtime.prepare(
            cache,
            group_index=2,
            layer_count=1,
            sequence_lengths=positions + tokens + 1,
            tokens=tokens,
            softmax_scale=config.head_dim**-0.5,
        )
        return prepared.run(queries, 0)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for starts in ((125, 127), (127, 129), (255, 256)):
        positions.copy_(torch.tensor(starts, device="cuda"))
        queries.normal_()
        graph.replay()
        for row, start in enumerate(starts):
            length = start + tokens
            physical = cache.row(row).rows(
                2,
                0,
                1,
                torch.arange(1, length + 1, device="cuda")[None],
            )[0]
            key = (
                arena.keys.flatten(0, 1)[physical]
                .to(queries.dtype)
                .repeat_interleave(2, dim=1)
            )
            value = (
                arena.values.flatten(0, 1)[physical]
                .to(queries.dtype)
                .repeat_interleave(2, dim=1)
            )
            mask = torch.arange(length, device="cuda")[None] <= (
                torch.arange(tokens, device="cuda")[:, None] + length - tokens
            )
            expected = F.scaled_dot_product_attention(
                queries[row].transpose(0, 1)[None],
                key.transpose(0, 1)[None],
                value.transpose(0, 1)[None],
                attn_mask=mask,
            )[0].transpose(0, 1)
            torch.testing.assert_close(actual[row], expected, atol=2e-2, rtol=2e-2)
