"""Representative Inkling2 cache ownership, rollback and DSpark recurrence checks."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from resin.models.inkling2.dspark import Inkling2DSparkConfig
from resin_models_cuda.inkling2.cache import CacheArena, CacheGeometry
from resin_models_cuda.inkling2.cuda import Inkling2CudaRunner
from resin_models_cuda.inkling2.dspark import DSparkMarkovHead
from resin_models_cuda.inkling2.kernels import (
    copy_state_checkpoints,
    select_convolution_state,
    short_convolution,
)


def _geometry():
    return CacheGeometry(2, 3, 1, 2, 2, 8, 32, 3, 128, 128, prefill_chunk_size=128)


def test_snapshot_preserves_state_and_draft_features_without_aliasing():
    geometry = _geometry()
    arena = CacheArena("cpu", geometry.layout(8), geometry)
    assert arena.draft_features is not None
    state, features = arena.states[1, 0], arena.draft_features[1, 0]
    state.fill_(3)
    features.fill_(7)
    block = arena.state_blocks[geometry.state_group]
    assert block[1, 0].numel() == state.numel() + features.numel()
    # The ordinary snapshot copier must include both fields.
    block[2, 1].copy_(block[1, 0])
    state.zero_()
    features.zero_()
    assert arena.states[2, 1].eq(3).all()
    assert arena.draft_features[2, 1].eq(7).all()
    arena.draft_features[2, 1].fill_(11)
    assert arena.states[2, 1].eq(3).all()
    assert features.eq(0).all()


def test_state_alignment_is_independent_of_prefill_chunk_size():
    geometry = CacheGeometry(2, 3, 1, 2, 2, 8, 32, 3, 128, 128)
    assert geometry.prefill_chunk_size == 4096
    assert geometry.layout(64).groups[geometry.state_group].block_tokens == 128


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_checkpoints_inside_accepted_verification_preserve_exact_state(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="requires CUDA")
    torch.manual_seed(39)
    initial = torch.randn(2, 2, 3, 16, device=device).bfloat16()
    inputs = torch.randn(2, 8, 2, 16, device=device).bfloat16()
    features = torch.randn(2, 8, 13, device=device).bfloat16()
    width = initial[0].numel() + features.shape[-1]
    arena = torch.full((8, 2 * width + 17), -99, device=device).bfloat16()
    storage = arena.as_strided((8, 2, width), (arena.stride(0), width, 1))
    addresses = torch.tensor([[0, 0, item + 1, 1] for item in range(6)], device=device)
    candidates = torch.tensor(
        [[0, 256], [1, 512], [0, 260], [0, 261], [0, 253], [1, 517]],
        device=device,
    )
    positions = torch.tensor([253, 509], device=device)
    ends = torch.tensor([260, 511], device=device)

    def operation():
        return copy_state_checkpoints(
            storage, addresses, candidates, ends, initial, inputs, features, positions
        )

    def check(flags):
        history = torch.cat((initial.transpose(1, 2), inputs), dim=1)
        expected = torch.full_like(arena, -99)
        expected_slots = expected.as_strided(storage.shape, storage.stride())
        ready = []
        for (_, _, slab, slot), (row, prefix) in zip(
            addresses.tolist(), candidates.tolist(), strict=True
        ):
            count = prefix - int(positions[row])
            valid = 0 < count <= 8 and prefix <= int(ends[row])
            ready.append(valid)
            if valid:
                state = history[row, count : count + 3].transpose(0, 1).flatten()
                expected_slots[slab, slot] = torch.cat(
                    (state, features[row, count - 1])
                )
        assert flags.tolist() == ready
        torch.testing.assert_close(arena, expected, rtol=0, atol=0)

    check(operation())
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flags = operation()
        for stop in (512, 517):
            arena.fill_(-99)
            ends[1] = stop
            initial.add_(1)
            inputs.sub_(2)
            features.add_(3)
            graph.replay()
            check(flags)


def test_convolution_rollback_uses_raw_inputs_at_each_accepted_boundary():
    torch.manual_seed(741)
    initial = torch.randn(2, 1, 3, 16).bfloat16()
    inputs = torch.randn(2, 8, 1, 16).bfloat16()
    weight = torch.randn(16, 4)
    original = initial.clone()
    raw = torch.full_like(inputs, -512)
    destination = torch.empty_like(initial)
    actual, state = short_convolution(
        inputs[:, :, 0],
        initial[:, 0],
        weight,
        None,
        state_out=destination[:, 0],
        inputs_out=raw[:, :, 0],
    )
    expected, expected_state = short_convolution(
        inputs[:, :, 0], initial[:, 0], weight, None
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
    torch.testing.assert_close(destination[:, 0], expected_state, atol=0, rtol=0)
    torch.testing.assert_close(raw, inputs, atol=0, rtol=0)
    torch.testing.assert_close(initial, original, atol=0, rtol=0)
    history = torch.cat((initial, inputs.transpose(1, 2)), 2)
    # The two requests accept different counts, including zero and the full block.
    for count in (0, 1, 3, 8):
        accepted = torch.tensor([count, min(count + 1, 8)])
        wanted = torch.stack(
            [history[row, :, n : n + 3] for row, n in enumerate(accepted)]
        )
        torch.testing.assert_close(
            select_convolution_state(initial, raw, accepted), wanted, atol=0, rtol=0
        )


def test_initial_state_owns_storage_and_zeros_only_new_sequences():
    positions = [0, 1, 9]
    g = _geometry()
    arena = CacheArena("cpu", g.layout(8), g)
    live = arena.states[1:4, 0]
    original = torch.arange(live.numel()).reshape(live.shape).to(live.dtype)
    live.copy_(original)
    positions = torch.tensor(positions)

    def state_at(indices):
        torch.testing.assert_close(indices[:, 0], (positions - 1).clamp_min(0))
        # Model the strongest aliasing contract: a view of the actual live slab.
        # This keeps the test valid if state_at later avoids its gather copy.
        return live[:, None]

    runner = object.__new__(Inkling2CudaRunner)
    initial = runner._initial_state(SimpleNamespace(state_at=state_at), positions)
    expected = original.clone()
    expected[positions == 0] = 0
    torch.testing.assert_close(initial, expected, rtol=0, atol=0)
    torch.testing.assert_close(live, original, rtol=0, atol=0)
    scratch = torch.empty_like(live)
    captured = runner._initial_state(
        SimpleNamespace(state_at=state_at), positions, out=scratch
    )
    assert captured.data_ptr() == scratch.data_ptr()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)

    live.fill_(-7)  # A later target forward overwrites the physical live slots.
    torch.testing.assert_close(initial, expected, rtol=0, atol=0)
    initial.fill_(11)
    assert live.eq(-7).all()


def _draft_config() -> Inkling2DSparkConfig:
    return Inkling2DSparkConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        rope_theta=1_000_000,
        sliding_window_size=4,
        block_size=8,
        markov_rank=3,
        local_layer_ids=(0,),
        target_layer_ids=(0, 2),
    )


def test_recurrent_markov_matches_gated_reference_and_folded_target_head():
    torch.manual_seed(23)
    config = _draft_config()
    head = DSparkMarkovHead(config)
    lm_head = torch.randn(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
    head.prepare(lm_head)
    expected_weight = (lm_head.double() @ head.hidden_out.weight.double()).bfloat16()
    expected_bias = (lm_head.double() @ head.hidden_out.bias.double()).bfloat16()
    torch.testing.assert_close(head.fold_weight, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(head.fold_bias, expected_bias, rtol=0, atol=0)
    hidden = torch.randn(2, 7, config.hidden_size, dtype=torch.bfloat16)
    f_rnn = torch.randn(2, config.hidden_size, dtype=torch.bfloat16)
    state = hidden.new_zeros(2, config.markov_rank)
    previous = torch.tensor([1, 2])
    for step in range(7):
        joint = torch.cat((state, head.markov_w1(previous), hidden[:, step], f_rnn), -1)
        gate, candidate, out = F.linear(
            joint, head.rnn_proj.weight, head.rnn_proj.bias
        ).chunk(3, -1)
        expected_state = (
            gate.sigmoid() * state + (1 - gate.sigmoid()) * candidate.tanh()
        )
        expected = F.linear(out.tanh(), expected_weight, expected_bias)
        actual, state = head.step(state, previous, hidden[:, step], f_rnn)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        previous = actual.argmax(-1)


def _draft_case(rows):
    from resin_models_cuda.inkling2.dspark_kernels import (
        mix_rmsnorm,
        rnn_gates,
        rnn_inputs,
    )

    state = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    embedding = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(rows, 7, 4608, device="cuda", dtype=torch.bfloat16)[:, 3]
    features = torch.randn(rows, 4608, device="cuda", dtype=torch.bfloat16)
    tokens = torch.randint(32, (rows,), device="cuda")
    projection = torch.randn(rows, 768, device="cuda", dtype=torch.bfloat16)
    targets = torch.randn(rows, 8, 4608, device="cuda", dtype=torch.bfloat16)
    alpha = torch.randn(8, device="cuda").softmax(-1).bfloat16()
    weight = torch.randn(4608, device="cuda", dtype=torch.bfloat16)

    def operation():
        return (
            rnn_inputs(state, tokens, embedding, hidden, features),
            *rnn_gates(projection, state),
            mix_rmsnorm(targets, alpha, weight, 1e-6),
        )

    def reference():
        gate, candidate, latent = projection.chunk(3, -1)
        mixed = (targets.float() * alpha.float()[None, :, None]).sum(1)
        return (
            torch.cat((state, embedding[tokens], hidden, features), -1),
            gate.sigmoid() * state + (1 - gate.sigmoid()) * candidate.tanh(),
            latent.tanh(),
            F.rms_norm(mixed, (4608,), weight.float(), 1e-6).bfloat16(),
        )

    def change():
        for value in (state, embedding, hidden, features, projection, targets, weight):
            value.copy_(torch.randn_like(value))
        tokens.copy_(torch.randint(32, tokens.shape, device="cuda"))
        alpha.copy_(torch.randn_like(alpha).float().softmax(-1).bfloat16())

    return operation, reference, change


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_recurrent_kernels_changed_graph_replay(monkeypatch):
    from triton import knobs

    factory = _draft_case
    operation, reference, change = factory(3)
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = operation()
    compiled = []
    previous = knobs.runtime.jit_post_compile_hook
    names = {
        "_rnn_inputs",
        "_rnn_gates",
        "_mix_rmsnorm",
    }

    def post_compile(**event):
        if event["fn"].name in names:
            compiled.append((event["fn"].name, str(event["key"])))
        if previous is not None:
            return previous(**event)

    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", post_compile)

    def check(actual, expected):
        for index, (value, wanted) in enumerate(zip(actual, expected, strict=True)):
            exact = index < 3
            torch.testing.assert_close(
                value,
                wanted,
                rtol=0 if exact else 0.008,
                atol=0 if exact else 0.00003,
            )

    for _ in range(3):
        change()
        graph.replay()
        check(captured, reference())
    operation, reference, change = factory(17)
    change()
    check(operation(), reference())
    assert compiled == [], f"request data compiled new kernels: {compiled}"
