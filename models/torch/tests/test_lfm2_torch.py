from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _tiny_args():
    from resin_models_torch.lfm2.config import Lfm2Config

    return Lfm2Config(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        norm_eps=1e-5,
        conv_bias=False,
        conv_cache_length=3,
        intermediate_size=32,
        multiple_of=8,
        auto_adjust_ff_dim=False,
        ffn_dim_multiplier=1.0,
        rope_theta=10_000.0,
        layer_types=("conv", "full_attention"),
    )


class TestLfm2TorchImport:
    def test_import_loads_torch_but_not_checkpoint_stacks(self) -> None:
        script = """
import sys
sys.path.insert(0, 'models/torch')
import resin_models_torch.lfm2
assert 'torch' in sys.modules
for name in ('transformers', 'tokenizers', 'huggingface_hub', 'safetensors'):
    assert name not in sys.modules
"""
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_args_reject_non_lfm_checkpoint(self) -> None:
        from resin_models_torch.lfm2.config import Lfm2Config

        with pytest.raises(ValueError, match="expected an LFM2"):
            Lfm2Config.from_dict({"model_type": "llama"})


class TestLfm2RMSNorm:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("strided", [False, True])
    def test_preserves_input_and_matches_reference(self, dtype, strided) -> None:
        from resin_models_torch.lfm2.model import RMSNorm

        storage = torch.arange(1, 17, dtype=dtype).reshape(2, 8)
        hidden = storage[:, ::2] if strided else storage
        original = storage.clone()
        norm = RMSNorm(hidden.shape[-1], 1e-5).to(dtype=dtype)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, hidden.shape[-1]))

        reference = hidden.float().clone()
        variance = reference.square().mean(-1, keepdim=True)
        expected = (reference / torch.sqrt(variance + norm.eps)).to(dtype)
        expected = norm.weight.detach() * expected

        with torch.inference_mode():
            actual = norm(hidden)

        torch.testing.assert_close(storage, original, rtol=0, atol=0)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected)


class TestLfm2Torch:
    def setup_method(self) -> None:
        from resin_models_torch.lfm2.model import Lfm2ForCausalLM

        torch.manual_seed(7)
        self.args = _tiny_args()
        self.model = Lfm2ForCausalLM(self.args).eval()

    def test_fp32_logits_match_transformers_reference(self) -> None:
        from transformers.models.lfm2.configuration_lfm2 import (
            Lfm2Config as ReferenceConfig,
        )
        from transformers.models.lfm2.modeling_lfm2 import (
            Lfm2ForCausalLM as ReferenceModel,
        )

        from resin_models_torch.lfm2.model import RMSNorm

        config = ReferenceConfig(
            vocab_size=self.args.vocab_size,
            hidden_size=self.args.hidden_size,
            intermediate_size=self.args.intermediate_size,
            num_hidden_layers=self.args.num_hidden_layers,
            num_attention_heads=self.args.num_attention_heads,
            num_key_value_heads=self.args.num_key_value_heads,
            norm_eps=self.args.norm_eps,
            conv_bias=self.args.conv_bias,
            conv_L_cache=self.args.conv_cache_length,
            block_multiple_of=self.args.multiple_of,
            block_auto_adjust_ff_dim=self.args.auto_adjust_ff_dim,
            block_ffn_dim_multiplier=self.args.ffn_dim_multiplier,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": self.args.rope_theta,
            },
            layer_types=list(self.args.layer_types),
            pad_token_id=self.args.pad_token_id,
            tie_word_embeddings=True,
        )
        config._attn_implementation = "eager"
        reference = ReferenceModel(config).eval()
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, RMSNorm):
                    module.weight.copy_(torch.linspace(0.5, 1.5, module.weight.numel()))

        # Resin projects logits with the embedding weight directly; HF exposes
        # the same tied tensor under an additional lm_head.weight state key.
        reference.load_state_dict(
            {
                **self.model.state_dict(),
                "lm_head.weight": self.model.model.embed_tokens.weight,
            }
        )
        assert reference.lm_head.weight is reference.model.embed_tokens.weight
        tokens = torch.tensor([[1, 2, 3, 4]])

        with torch.inference_mode():
            expected = reference(tokens, use_cache=False).logits
            actual = self.model(tokens)

        assert actual.dtype == expected.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_cached_chunks_match_full_forward(self) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.inference_mode():
            full = self.model(tokens, self.model.make_cache())
            for split in (1, 2, 3):
                cache = self.model.make_cache()
                self.model(tokens[:, :split], cache)
                cached = self.model(tokens[:, split:], cache)
                torch.testing.assert_close(
                    full[:, split:], cached, rtol=1e-5, atol=1e-5
                )

    def test_batched_decode_matches_independent_cached_forwards(self) -> None:
        independent_states = [self.model.make_cache(), self.model.make_cache()]
        batched_states = [self.model.make_cache(), self.model.make_cache()]
        prefixes = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])]

        with torch.inference_mode():
            for prefix, independent, batched in zip(
                prefixes, independent_states, batched_states, strict=True
            ):
                self.model(prefix, independent)
                self.model(prefix, batched)

            for tokens in (torch.tensor([[6], [7]]), torch.tensor([[8], [9]])):
                expected = torch.cat(
                    [
                        self.model(tokens[index : index + 1], state)
                        for index, state in enumerate(independent_states)
                    ]
                )
                actual = self.model.forward_batch(tokens, batched_states)
                torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)

        assert [state.length for state in batched_states] == [5, 4]
        for independent, batched in zip(
            independent_states, batched_states, strict=True
        ):
            for expected_layer, actual_layer in zip(
                independent.layers, batched.layers, strict=True
            ):
                expected_tensors = vars(expected_layer).values()
                actual_tensors = vars(actual_layer).values()
                for expected_tensor, actual_tensor in zip(
                    expected_tensors, actual_tensors, strict=True
                ):
                    torch.testing.assert_close(
                        expected_tensor, actual_tensor, rtol=1e-5, atol=1e-5
                    )

    def test_engine_scores_match_full_context_model_logits(self) -> None:
        from resin import Engine, SamplingParams
        from resin_models_torch.lfm2 import Lfm2TorchRunner

        runner = Lfm2TorchRunner.__new__(Lfm2TorchRunner)
        runner.device = torch.device("cpu")
        runner._model = self.model
        runner.vocab_size = self.args.vocab_size
        runner._cache = {}
        runner.forward_calls = 0
        engine = Engine(runner)
        tokens = [1, 2, 3]
        request = engine.submit(
            tokens,
            sampling_params=SamplingParams(
                max_output_tokens=2, temperature=0, top_logprobs=3
            ),
        )
        for _ in range(2):
            with torch.inference_mode():
                expected = self.model(torch.tensor([tokens]), self.model.make_cache())[
                    0, -1
                ].log_softmax(-1)
            (event,) = engine.step()
            assert event.token_id == expected.argmax().item()
            assert event.logprobs is not None
            assert event.logprobs.logprob == pytest.approx(
                expected[event.token_id].item(), abs=1e-5
            )
            assert [token for token, _ in event.logprobs.top_logprobs] == expected.topk(
                3
            ).indices.tolist()
            tokens.append(event.token_id)
        assert request.take_result() == tokens[-2:]
        assert runner.live_cache_count == 0

    def test_checkpoint_parameter_paths_match_the_released_layout(self) -> None:
        keys = set(self.model.state_dict())
        assert "model.layers.0.conv.in_proj.weight" in keys
        assert "model.layers.1.self_attn.q_layernorm.weight" in keys
        assert "model.layers.1.feed_forward.w1.weight" in keys
        assert "model.embedding_norm.weight" in keys
        assert not any(".operator." in key for key in keys)

    @pytest.mark.parametrize("chunk", [1, 2, 8])
    def test_prompt_scores_match_full_context_without_extra_forwards(self, chunk):
        from resin import Engine, SamplingParams, SchedulerConfig
        from resin_models_torch.lfm2 import Lfm2TorchRunner

        runner = Lfm2TorchRunner.__new__(Lfm2TorchRunner)
        runner.device = torch.device("cpu")
        runner._model = self.model
        runner.vocab_size = self.args.vocab_size
        runner._cache = {}
        runner.forward_calls = 0
        tokens = [1, 2, 3, 4, 5]
        with torch.inference_mode():
            logits = self.model(torch.tensor([tokens]), self.model.make_cache())[0]
            distributions = logits[:-1].log_softmax(-1)
            expected = distributions[range(len(tokens) - 1), tokens[1:]]
        engine = Engine(
            runner, scheduler_config=SchedulerConfig(prefill_chunk_size=chunk)
        )
        request = engine.submit(
            tokens,
            sampling_params=SamplingParams(max_output_tokens=0, prompt_top_logprobs=3),
            prompt_target_logprobs=[[], [7, 2, 7], [], [63], [4, 4]],
        )
        while engine.has_work():
            assert engine.step() == []
        scores = request.prompt_logprobs
        assert scores is not None and scores[0] is None
        assert [
            score.logprob for score in scores[1:] if score is not None
        ] == pytest.approx(expected.tolist(), abs=1e-5)
        for position, target_ids in [(1, [7, 2, 7]), (2, []), (3, [63]), (4, [4, 4])]:
            score = scores[position]
            assert score is not None
            distribution = distributions[position - 1]
            assert [token for token, _ in score.top_logprobs] == distribution.topk(
                3
            ).indices.tolist()
            assert [value for _, value in score.top_logprobs] == pytest.approx(
                distribution.topk(3).values.tolist(), abs=1e-5
            )
            assert [token for token, _ in score.target_logprobs] == target_ids
            assert [value for _, value in score.target_logprobs] == pytest.approx(
                distribution[target_ids].tolist(), abs=1e-5
            )
        assert runner.forward_calls == (len(tokens) + chunk - 1) // chunk
        assert runner.live_cache_count == 0

    def test_runner_consumes_batch_plan_and_releases_per_request_cache(self) -> None:
        from resin import BatchPhase, BatchPlan
        from resin_models_torch.lfm2 import Lfm2TorchRunner

        runner = Lfm2TorchRunner.__new__(Lfm2TorchRunner)
        runner.device = torch.device("cpu")
        runner._model = self.model
        runner.config = {"vocab_size": self.args.vocab_size}
        runner.eos_token_id = None
        runner.vocab_size = self.args.vocab_size
        runner._cache = {}
        runner.forward_calls = 0

        prefill = BatchPlan(
            sequence_ids=[17, 23],
            context_lengths=[0, 0],
            token_ids=[1, 2, 8],
            token_indptr=[0, 2, 3],
        )
        first = runner.forward(prefill)
        assert first.token_indptr.tolist() == [0, 1, 2]
        assert len(first.token_ids) == 2
        assert runner.live_cache_count == 2

        decode = BatchPlan(
            sequence_ids=[17, 23],
            context_lengths=[2, 1],
            token_ids=first.token_ids,
            token_indptr=[0, 1, 2],
            phase=BatchPhase.DECODE,
        )
        second = runner.forward(decode)
        assert second.token_indptr.tolist() == [0, 1, 2]
        assert len(second.token_ids) == 2
        assert all(0 <= token < self.args.vocab_size for token in second.token_ids)
        runner.release([17, 23])
        assert runner.live_cache_count == 0

    @pytest.mark.skipif(
        not importlib.util.find_spec("safetensors"),
        reason="safetensors is not installed",
    )
    def test_direct_safetensors_loading_without_transformers(
        self, tmp_path: Path
    ) -> None:
        from safetensors.torch import save_file

        from resin_models_torch.lfm2.reference import load_model

        config = {
            "model_type": "lfm2",
            "vocab_size": self.args.vocab_size,
            "hidden_size": self.args.hidden_size,
            "num_hidden_layers": self.args.num_hidden_layers,
            "num_attention_heads": self.args.num_attention_heads,
            "num_key_value_heads": self.args.num_key_value_heads,
            "norm_eps": self.args.norm_eps,
            "conv_bias": self.args.conv_bias,
            "conv_L_cache": self.args.conv_cache_length,
            "intermediate_size": self.args.intermediate_size,
            "block_ff_dim": self.args.intermediate_size,
            "block_multiple_of": self.args.multiple_of,
            "block_auto_adjust_ff_dim": self.args.auto_adjust_ff_dim,
            "block_ffn_dim_multiplier": self.args.ffn_dim_multiplier,
            "rope_parameters": {
                "rope_theta": self.args.rope_theta,
                "rope_type": "default",
            },
            "layer_types": list(self.args.layer_types),
        }
        tokens = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            expected = self.model(tokens, self.model.make_cache())

        path = tmp_path
        (path / "config.json").write_text(json.dumps(config))
        save_file(self.model.state_dict(), path / "model.safetensors")
        loaded, loaded_config = load_model(path, device="cpu")
        # Recipe tests can import these libraries earlier in this process.
        # Check the loader's import boundary in a fresh interpreter.
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
sys.path.insert(0, 'models/torch')
from resin_models_torch.lfm2.reference import load_model
from pathlib import Path
load_model(Path(sys.argv[1]), device='cpu')
assert 'transformers' not in sys.modules
assert 'tokenizers' not in sys.modules
""",
                str(path),
            ],
            check=True,
        )
        with torch.inference_mode():
            actual = loaded(tokens, loaded.make_cache())
        assert loaded_config["model_type"] == "lfm2"
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
