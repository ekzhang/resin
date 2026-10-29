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


def test_two_token_verify_and_commit_match_sequential_recurrence():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    from resin_models_cuda.qwen3_5.gdn_decode import (
        RecurrentStatePool,
        gdn_commit,
        gdn_decode,
    )

    torch.manual_seed(4)
    B, H, HV, K, V, L, T = 4, 16, 32, 128, 128, 2, 2
    state_elems = L * HV * K * V
    for accepted_value in (0, 1):
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
        expected = ref_state if accepted_value else first_state
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
@pytest.mark.parametrize("draft_tokens,forced", [(0, None), (1, None), (1, 0), (1, 1)])
def test_tiny_runner_overlap_matches_sync(monkeypatch, tmp_path, draft_tokens, forced):
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
    modes = ((False, 0), (True, 136))
    outputs, statistics = [], []
    for overlap, reservation in modes:
        runner = runtime.Qwen3_5CudaRunner(
            tmp_path,
            device="cuda",
            speculative_tokens=draft_tokens,
            max_seq_len=config.max_position_embeddings,
            max_batch_size=4,
            cache_slab_capacity=1024,
            kv_cache_dtype="bfloat16",
            cuda_graph_prefill="disabled",
            cuda_graph_decode="full",
            cuda_graph_decode_shapes=[(4, config.max_position_embeddings)],
            cache_write_reservation_tokens=reservation,
        )
        runner._speculation.benchmark_accepted_drafts = forced
        runner.capture_cuda_graphs()
        engine = Engine(
            runner,
            max_running_requests=3,
            context_length=config.max_position_embeddings,
            scheduler_config=SchedulerConfig(prefill_chunk_size=4096),
            prefix_cache=False,
            enable_overlap=overlap,
        )
        mode_outputs, mode_statistics = [], []
        try:
            for length in (125, 4093, 8192):
                leases = [
                    engine.submit(
                        [1]
                        + [
                            2 + (index * 17 + row * 11) % 62
                            for index in range(length - 1)
                        ],
                        sampling_params=SamplingParams(
                            max_output_tokens=160,
                            stop_token_ids=[],
                            temperature=0.0,
                            seed=41,
                        ),
                    )
                    for row in range(3)
                ]
                while engine.has_work():
                    engine.step()
                values = [lease.take_result() for lease in leases]
                assert all(len(value) == 160 for value in values)
                mode_outputs.append(values)
                mode_statistics.append(dict(runner.speculative_stats))
            if overlap:
                assert engine.overlap_stats[1] > 0
            outputs.append(mode_outputs)
            statistics.append(mode_statistics)
        finally:
            engine.close()
            runner.close_cuda_graphs()
            del engine, runner
            gc.collect()
            torch.cuda.empty_cache()
    assert outputs[0] == outputs[1]
    assert statistics[0] == statistics[1]
