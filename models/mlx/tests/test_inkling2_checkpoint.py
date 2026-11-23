"""Inkling 2 Mini on its real checkpoint against the CUDA package's model on CPU.

Set ``RESIN_TEST_CHECKPOINT`` to a local thinkingmachines/Inkling-2-Mini
snapshot (revision ed40e7da625f594fc9c4990f5f66be24873c22d7). Both models load
(about 50 GB of memory); the run takes about a minute on an M4 Max.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")

CHECKPOINT = os.environ.get("RESIN_TEST_CHECKPOINT")
pytestmark = pytest.mark.skipif(
    not CHECKPOINT, reason="RESIN_TEST_CHECKPOINT is not set"
)


def _fast_experts():
    """The reference's per-route arithmetic, grouped by expert for speed."""
    from test_inkling2_model import _activations, _weight

    modules = {}

    @lru_cache(maxsize=640)
    def weight(module, name, expert):
        found = modules[module]
        return _weight(found, name, found.precision[name], expert).float()

    def experts(module, hidden, ids, weights):
        modules[id(module)] = module
        mode = module.precision["w13"]
        rows, top_k = ids.shape
        down = torch.zeros(rows, top_k, hidden.shape[1], dtype=torch.bfloat16)

        def matmul(x, name, expert):
            base, amax = _activations(x, mode)
            output = base @ weight(id(module), name, expert).t()
            if mode == "nvfp4":
                denominator = getattr(module, name + "_weight_outer_scale_denominator")
                weight_amax = getattr(module, name + "_weight_amax")[expert]
                factor = (amax * weight_amax) * (1.0 / (1536.0 * denominator))
                output = output * factor[:, None]
            return output.to(torch.bfloat16)

        for expert in torch.unique(ids).tolist():
            token, slot = (ids == expert).nonzero(as_tuple=True)
            gate_up = (
                matmul(hidden[token], "w13", expert).float().view(len(token), -1, 2)
            )
            gate, up = gate_up[..., 0], gate_up[..., 1]
            activated = (
                ((4 * (gate / 4).tanh()) * gate.sigmoid()) * (25 * (up / 25).tanh())
            ) * weights[token, slot][:, None]
            down[token, slot] = matmul(activated.to(torch.bfloat16), "w2", expert)
        total = down[:, 0].float()
        for slot in range(1, top_k):
            total = total + down[:, slot].float()
        return total

    return experts


def test_real_checkpoint_matches_cpu_reference(monkeypatch):
    from test_inkling2_model import (
        DenseStore,
        ReferenceCache,
        _record,
        _relative,
        _to_mlx,
        reference_attention,
    )
    from tokenizers import Tokenizer

    from resin_models_cuda.inkling2 import attention
    from resin_models_cuda.inkling2.checkpoint import load_precision
    from resin_models_cuda.inkling2.model import Inkling2Target
    from resin_models_cuda.inkling2.weights import load_target_weights
    from resin_models_mlx.inkling2 import model as mlx_model
    from resin_models_mlx.inkling2.checkpoint import load_model
    from resin_models_mlx.inkling2.model import Inkling2Model

    assert CHECKPOINT is not None
    root = Path(CHECKPOINT).expanduser()
    config, weights = load_model(root)
    monkeypatch.setattr(attention, "paged_attention", reference_attention)
    target = Inkling2Target(config, (5,), precision=load_precision(root))
    load_target_weights(target, root)
    target.prepare()
    experts = _fast_experts()
    for layer in target.layers:
        module = layer.mlp.experts  # ty: ignore[unresolved-attribute]
        monkeypatch.setattr(
            module,
            "_local",
            lambda h, i, w, e=module, max_rows_per_expert=None: experts(e, h, i, w),
        )
    model = Inkling2Model(config, weights)
    text = (root / "README.md").read_text()
    # 80 tokens (640 routes, five per expert) take MLX's grouped expert GEMM.
    prompt = Tokenizer.from_file(str(root / "tokenizer.json")).encode(text).ids[:80]

    # Teacher-forced: each visit's attention and experts from the reference's inputs.
    records = _record(target)
    errors = {"attention": 0.0, "mlp": 0.0}
    cursor = {"attention": 0, "mlp": 0}
    attend, sparse = Inkling2Model._attention, mlx_model.sparse_mlp

    def forced_attention(self, layer, sources, state, tau, visit, slot, store):
        index = cursor["attention"]
        cursor["attention"] += 1
        forced = tuple(_to_mlx(x) for x in records["sources"][index])
        output, k_history, v_history = attend(
            self, layer, forced, state, tau, visit, slot, store
        )
        expected = records["attention"][index]
        errors["attention"] = max(errors["attention"], _relative(output, expected))
        return _to_mlx(expected), k_history, v_history

    route_ties = []

    def forced_mlp(x, layer, config):
        index = cursor["mlp"]
        cursor["mlp"] += 1
        forced = _to_mlx(records["mlp_input"][index]).reshape(x.shape)
        output = sparse(forced, layer, config)
        expected = records["mlp"][index]
        # A token whose eighth and ninth biased router scores tie in FP32 may
        # pick the other expert under a different summation order.
        rows = forced.reshape(-1, x.shape[-1]).astype(mx.float32)
        scores = np.array(mx.sigmoid(rows @ layer.router.T) + layer.router_bias)
        ranked = -np.sort(-scores, axis=-1)
        top_k = config.experts_per_token
        tied = ranked[:, top_k - 1] - ranked[:, top_k] < 1e-5
        route_ties.extend((index, int(row)) for row in np.flatnonzero(tied))
        keep = np.flatnonzero(~tied)
        width = expected.shape[-1]
        kept = _relative(
            output.reshape(-1, width)[mx.array(keep)],
            expected.reshape(-1, width)[torch.from_numpy(keep)],
        )
        errors["mlp"] = max(errors["mlp"], kept)
        return _to_mlx(expected).reshape(x.shape)

    tokens = torch.tensor([prompt])
    positions = torch.arange(len(prompt))[None]
    state = torch.zeros(
        1, config.execution_layer_count, 3, config.state_width(), dtype=torch.bfloat16
    )
    with torch.no_grad():
        expected, _, _, _ = target(
            tokens,
            positions,
            ReferenceCache(config, 1, 128),
            state,
            positions[:, -1] + 1,
            len(prompt),
            full_logits=True,
            prefill=True,
        )
    with monkeypatch.context() as patch:
        patch.setattr(Inkling2Model, "_attention", forced_attention)
        patch.setattr(mlx_model, "sparse_mlp", forced_mlp)
        store = DenseStore(config, 1, 128)
        store.positions = mx.array(positions.numpy())
        actual, _ = model(
            mx.array(tokens.numpy()),
            store.positions,
            model.initial_state(1),
            store,
            last_only=False,
        )
    # Real-checkpoint visits agree within about one BF16 step (2^-8 relative).
    assert errors["attention"] < 1e-2 and errors["mlp"] < 1e-2, errors
    # Three of the 3,840 token visits tie at this prompt.
    assert len(route_ties) <= 0.002 * len(prompt) * config.execution_layer_count, (
        route_ties
    )
    assert _relative(actual, expected) < 1e-3

    # Free-running: prefill and three greedy decode steps choose the same tokens.
    cache, store = ReferenceCache(config, 1, 128), DenseStore(config, 1, 128)
    reference_state = state
    mlx_state = model.initial_state(1)
    current, start = tokens, 0
    for step in range(4):
        count = current.shape[1]
        positions = torch.arange(start, start + count)[None]
        with torch.no_grad():
            expected, _, reference_state, _ = target(
                current,
                positions,
                cache,
                reference_state,
                positions[:, -1] + 1,
                start + count,
                full_logits=False,
                prefill=step == 0,
            )
        store.positions = mx.array(positions.numpy())
        actual, mlx_state = model(
            mx.array(current.numpy()), store.positions, mlx_state, store
        )
        token = int(expected[0, -1].float().argmax())
        assert int(np.argmax(np.array(actual[0, -1]))) == token, step
        current, start = torch.tensor([[token]]), start + count
