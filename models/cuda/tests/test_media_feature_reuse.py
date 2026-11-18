"""Cross-request vision reuse without retaining request identity or placement."""

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from resin import BatchPhase, BatchPlan
from resin.backend.cuda.completion import CudaContinuation
from resin.backend.cuda.vision_cache import VisionFeatureCache
from resin.recipe import Media


@pytest.fixture(params=["deepseek", "glm", "qwen", "gemma", "inkling2"])
def case(request):
    return _case(request.param)


def _case(name):
    from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner
    from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner
    from resin_models_cuda.glm5_3_flash.cuda import Glm53FlashCudaRunner
    from resin_models_cuda.inkling2.cuda import Inkling2CudaRunner
    from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner

    cls = {
        "deepseek": DeepSeekV41CudaRunner,
        "glm": Glm53FlashCudaRunner,
        "qwen": Qwen3_5CudaRunner,
        "gemma": Gemma4CudaRunner,
        "inkling2": Inkling2CudaRunner,
    }[name]
    # These are deliberately partial runners with mock model/encoder state.
    runner: Any = object.__new__(cls)
    runner.device = torch.device("cpu")
    runner.config = SimpleNamespace(hidden_size=3, image_token_id=8, video_token_id=9)
    (
        runner.vision_patch_size,
        runner.spatial_merge_size,
        runner.vision_temporal_patch_size,
    ) = 2, 2, 1
    runner._media_features = {}
    runner._target_hidden = {}
    if name in ("deepseek", "glm", "qwen", "inkling2"):
        runner._continuation = CudaContinuation()
    if name == "qwen":
        runner._drafts, runner._draft_hidden = {}, {}
    if name == "gemma":
        runner._hidden_bank = None
        runner._hidden_free_slots, runner._hidden_slot_generation = [], 0
        runner._hidden_index_cache, runner._hidden_input_buffers = {}, {}
    runner._vision_feature_cache = VisionFeatureCache(max_entries=1)
    runner.max_batch_size, runner.prefill_chunk_size = 8, 128
    state = SimpleNamespace(calls=0, buffers={})

    def encode(pixels, rows, kind="image"):
        state.calls += 1
        buffer = state.buffers.setdefault(kind, torch.empty(12, 3))
        buffer.fill_(float(pixels.float().mean()))
        return buffer[:rows]  # Deliberately reuse encoder-owned storage.

    runner._model = SimpleNamespace(
        args=SimpleNamespace(image_token_id=129264),
        embed=lambda ids: ids.float()[..., None].expand(-1, -1, 3).clone(),
        embedding=torch.zeros(10, 3),
        vision=(
            SimpleNamespace(
                encode=lambda pixels, grid: encode(pixels, 4),
                config=SimpleNamespace(
                    patch_size=2, spatial_merge_size=2, temporal_patch_size=1
                ),
            )
            if name == "glm"
            else lambda pixels, *args: encode(pixels, 1 if name == "deepseek" else 2)
        ),
        visual=lambda pixels, grid: SimpleNamespace(pooler_output=encode(pixels, 4)),
        aligner=lambda features, *args: features,
        image_start=torch.full((3,), -1.0),
        image_end=torch.full((3,), -2.0),
        image_newline=torch.full((3,), -3.0),
        get_image_features=lambda pixels, positions: encode(pixels, 2),
        get_video_features=lambda pixels, positions: encode(pixels, 2, "video"),
    )

    runner._vision = runner._model.vision
    if name == "inkling2":
        runner._vision.patch_size, runner._vision.merge_size = 2, 1
    runner._model.transformers_config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=1, pooling_kernel_size=1)
    )

    def media(value=0.25, *, followup=False, geometry=False, shift=0):
        start = 1 + shift
        if name == "deepseek":
            shape = (28, 14, 3) if geometry else (14, 28, 3)
            return (Media(span=(start, 4), image=np.full(shape, value, np.float32)),)
        if name in ("qwen", "glm"):
            shape = (4, 16, 3) if geometry else (8, 8, 3)
            return (Media(span=(start, 4), image=np.full(shape, value, np.float32)),)
        if name == "inkling2":
            shape = (4, 2, 3) if geometry else (2, 4, 3)
            return (Media(span=(start, 2), image=np.full(shape, value, np.float32)),)
        shape = (2, 1, 3) if geometry else (1, 2, 3)
        return (
            Media(span=(start, 2), image=np.full(shape, value, np.float32)),
            Media(
                span=(start + 2, 2),
                video=(np.full(shape, value + 1, np.float32),),
                video_frame_spans=((start + 2, 2),),
                video_fps=1.0,
            ),
        )

    def tokens(payload):
        ids = [7] * (max(sum(item.span) for item in payload) + 1)
        for item in payload:
            for start, length in item.video_frame_spans or (item.span,):
                ids[start : start + length] = [
                    129264 if name == "deepseek" else 9 if item.video is not None else 8
                ] * length
        return tuple(ids)

    def features(sequence_id, payload, *, start=0):
        if name in ("qwen", "inkling2"):
            return (runner._features_for(sequence_id, payload),)
        if name == "gemma":
            result = runner._features_for(sequence_id, payload)
            return result.image, result.video
        plan = BatchPlan(
            sequence_ids=[sequence_id],
            context_lengths=[start],
            token_ids=tokens(payload)[start:],
            token_indptr=[0, len(tokens(payload)) - start],
            phase=BatchPhase.PREFILL,
            media=[payload],
        )
        if name == "deepseek":
            runner._validate_plan(plan)
            runner._media_embeddings(plan, 0, torch.tensor([tokens(payload)[start:]]))
            return (runner._media_features[sequence_id][1][0],)
        runner._media_embeddings(plan, torch.tensor(tokens(payload)[start:]))
        return (runner._media_features[sequence_id][1],)

    return SimpleNamespace(
        name=name,
        runner=runner,
        state=state,
        media=media,
        features=features,
        tokens=tokens,
    )


def test_followup_reuses_owned_features_after_request_release(case):
    payload = case.media()
    refs = sys.getrefcount(payload)
    first = case.features(1, payload)
    expected = tuple(t.clone() for t in first)
    calls = case.state.calls
    case.runner.release([1])
    assert not case.runner._media_features
    assert sys.getrefcount(payload) == refs
    with torch.inference_mode():
        for buffer in case.state.buffers.values():
            buffer.fill_(99)
    followup = case.media(followup=True)
    second = case.features(2, followup, start=2 if case.name == "deepseek" else 0)
    assert case.state.calls == calls
    for old, new, value in zip(first, second, expected, strict=True):
        assert old is new
        torch.testing.assert_close(new, value)


@pytest.mark.parametrize("change", ["pixels", "geometry"])
def test_changed_content_misses_and_eviction_preserves_live_features(case, change):
    first = case.features(1, case.media())
    expected = tuple(t.clone() for t in first)
    calls = case.state.calls
    changed = case.media(0.75) if change == "pixels" else case.media(geometry=True)
    case.features(2, changed)
    assert case.state.calls == 2 * calls
    assert len(case.runner._vision_feature_cache) == 1
    for value, saved in zip(first, expected, strict=True):
        torch.testing.assert_close(value, saved)
    case.features(3, case.media())
    assert case.state.calls == 3 * calls


def test_changed_pixels_cannot_replace_a_live_request(case):
    case.features(1, case.media())
    with pytest.raises(ValueError, match="changed"):
        case.features(1, case.media(0.75))


def test_inkling2_only_encodes_chunks_with_image_positions():
    case = _case("inkling2")
    payload = case.media()
    case.runner._model.embed_tokens = case.runner._model.embed
    case.runner._model.embed_norm = lambda hidden: hidden

    def chunk(sequence_id, start, stop):
        plan = BatchPlan(
            sequence_ids=[sequence_id],
            context_lengths=[start],
            token_ids=case.tokens(payload)[start:stop],
            token_indptr=[0, stop - start],
            phase=BatchPhase.PREFILL,
            media=[payload],
        )
        return case.runner._media_embeddings(
            plan, 0, torch.tensor([case.tokens(payload)[start:stop]])
        )

    assert chunk(1, 0, 1) is None
    assert chunk(2, 3, 4) is None  # Restored prefix already includes the image.
    assert case.state.calls == 0 and not case.runner._media_features
    embeddings = chunk(1, 1, 3)
    torch.testing.assert_close(embeddings, torch.full((1, 2, 3), 0.25))
    assert case.state.calls == 1
    assert chunk(1, 3, 4) is None
    assert case.state.calls == 1


def test_gemma_rebuilds_current_request_positions():
    case = _case("gemma")
    original = case.media()
    shifted = case.media(followup=True, shift=3)
    first = case.features(1, original)
    second = case.features(2, shifted)
    assert case.state.calls == 2  # One image and one video encoder invocation.
    assert all(a is b for a, b in zip(first, second, strict=True))
    record = case.runner._media_features[2]
    assert record.media is shifted
    tokens = case.tokens(shifted)
    hidden = torch.full((len(tokens), 3), 99.0)
    batch = SimpleNamespace(
        token_indptr=np.array([0, len(tokens)]), positions=np.arange(len(tokens))
    )
    native = SimpleNamespace(sequence_ids=(2,), media=(shifted,))
    case.runner._merge_media_embeddings(native, batch, hidden)
    torch.testing.assert_close(hidden[4:6], torch.full((2, 3), 0.25))
    torch.testing.assert_close(hidden[6:8], torch.full((2, 3), 1.25))
    assert (hidden[:4] == 99).all()
    batch.phase = BatchPhase.PREFILL
    batch.local = SimpleNamespace(first_token_positions=[0], kv_lengths=[len(tokens)])
    case.runner.sliding_window = 16
    mask = case.runner._local_custom_mask(batch, {0: record}).reshape(
        len(tokens), len(tokens)
    )
    expected = torch.ones_like(mask).tril()
    expected[4:6, 4:6] = True
    expected[6:8, 6:8] = True
    torch.testing.assert_close(mask, expected)
    with pytest.raises(ValueError, match="prefill chunk boundary"):
        case.runner._validate_unsplit_vision_blocks(shifted, 5, len(tokens))


def test_deepseek_keeps_image_indices_distinct():
    case = _case("deepseek")
    case.runner._vision_feature_cache = VisionFeatureCache(max_entries=2)
    first_image = case.media()[0]
    second_image = Media(span=(6, 4), image=np.full((14, 28, 3), 0.75, np.float32))

    def payload():
        return (first_image, second_image)

    case.features(1, payload())
    features = case.runner._media_features[1][1]
    assert features[0][1, 0] == 0.25
    assert features[1][1, 0] == 0.75
    case.runner.release([1])
    case.features(2, payload())
    assert case.state.calls == 2
    assert case.runner._media_features[2][1][0] is features[0]
    assert case.runner._media_features[2][1][1] is features[1]


def test_cached_features_reject_wrong_placeholder_lengths(case):
    media = case.media()
    case.features(1, media)
    item = media[0]
    changed = (
        Media(span=(item.span[0], item.span[1] - 1), image=item.image),
        *media[1:],
    )
    assert changed[0].digest == item.digest
    with pytest.raises(ValueError, match="geometry"):
        case.features(2, changed)
