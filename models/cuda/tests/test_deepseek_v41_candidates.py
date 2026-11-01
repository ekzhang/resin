"""Candidate wrapper allocates aligned scores for runtime context widths."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from resin_models_cuda.deepseek_v41.upstream import cuda


@pytest.mark.parametrize("block_size", [8, 16])
def test_candidate_output_alignment_and_valid_counts(monkeypatch, block_size):
    def factory(size):
        assert size == block_size

        def launch(logits, lengths, scores, counts):
            blocks = (logits.shape[1] + size - 1) // size
            assert scores.shape == (5, blocks)
            assert scores.stride(0) * scores.element_size() % 1024 == 0
            assert counts.numel() == 8
            counts[:5] = (lengths + size - 1) // size

        return launch

    operations = SimpleNamespace(candidate_block_max_tl=factory)
    monkeypatch.setattr(cuda, "candidate_blocks", operations, raising=False)
    path = (
        Path(__file__).parents[1]
        / "resin_models_cuda/deepseek_v41/upstream/candidate_blocks.py"
    )
    spec = importlib.util.spec_from_file_location("candidate_wrapper_test", path)
    assert spec is not None and spec.loader is not None
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    for width in (17, 16385):
        storage = torch.zeros(5, (width + 15) // 16 * 16, dtype=torch.bfloat16)
        lengths = torch.tensor(
            [0, 1, block_size + 1, width - 1, width], dtype=torch.int32
        )
        scores, counts = wrapper.candidate_block_max(
            storage[:, :width], lengths, block_size
        )
        blocks = (width + block_size - 1) // block_size
        assert scores.shape == (5, (blocks + 511) // 512 * 512)
        torch.testing.assert_close(counts, (lengths + block_size - 1) // block_size)
