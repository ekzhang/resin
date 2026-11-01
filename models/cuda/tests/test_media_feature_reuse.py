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
from resin.recipe import (
    DeepSeekV41Image,
    DeepSeekV41Media,
    Gemma4Media,
    Glm53FlashMedia,
    InklingMedia,
    Qwen35Media,
)


@pytest.fixture(params=["deepseek", "glm", "qwen", "gemma", "inkling"])
def case(request):
    return _case(request.param)


def _case(name):
    from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner
    from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner
    from resin_models_cuda.glm5_3_flash.cuda import Glm53FlashCudaRunner
    from resin_models_cuda.inkling.cuda import InklingCudaRunner
    from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner

    cls = {
        "deepseek": DeepSeekV41CudaRunner,
        "glm": Glm53FlashCudaRunner,
        "qwen": Qwen3_5CudaRunner,
        "gemma": Gemma4CudaRunner,
        "inkling": InklingCudaRunner,
    }[name]
    # These are deliberately partial runners with mock model/encoder state.
    runner: Any = object.__new__(cls)
    runner.device = torch.device("cpu")
    runner.config = SimpleNamespace(hidden_size=3, image_token_id=8, video_token_id=9)
    runner._media_features = {}
    runner._target_hidden = {}
    if name in ("deepseek", "glm", "qwen"):
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
            SimpleNamespace(encode=lambda pixels, grid: encode(pixels, 4))
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

    def media(value=0.25, *, followup=False, geometry=False, shift=0):
        prefix = (7,) * (1 + shift)
        suffix = (6, 5) if followup else (6,)
        if name == "deepseek":
            h, w = (2, 1) if geometry else (1, 2)
            types = (0, 1, 2, 3)
            image = DeepSeekV41Image(
                start=len(prefix),
                patches=np.full((2, 3, 14, 14), value, dtype=np.float32),
                n_vit_h=h,
                n_vit_w=w,
                types=types,
            )
            return DeepSeekV41Media(
                token_ids=prefix + (129264,) * len(types) + suffix, images=(image,)
            )
        if name in ("qwen", "glm"):
            return (Qwen35Media if name == "qwen" else Glm53FlashMedia)(
                token_ids=prefix + (8,) * 4 + suffix,
                image_positions=tuple(range(len(prefix), len(prefix) + 4)),
                pixel_values=np.full((16, 12), value, dtype=np.float32),
                image_grid_thw=np.array(
                    [[1, 2, 8] if geometry else [1, 4, 4]], dtype=np.int64
                ),
                spatial_merge_size=2,
                image_token_id=8,
            )
        if name == "inkling":
            return InklingMedia(
                token_ids=prefix + (8, 8) + suffix,
                image_token_id=8,
                pixel_values=np.full(
                    (2, 1, 4, 2, 3) if geometry else (2, 1, 2, 4, 3),
                    value,
                    dtype=np.float32,
                ),
            )
        return Gemma4Media(
            token_ids=prefix + (8, 8, 9, 9) + suffix,
            mm_token_type_ids=(0,) * len(prefix) + (1, 1, 2, 2) + (0,) * len(suffix),
            pixel_values=np.full((1, 4, 3), value, dtype=np.float32),
            image_position_ids=np.full((1, 4, 2), int(geometry), dtype=np.int64),
            pixel_values_videos=np.full((1, 1, 4, 3), value + 1, dtype=np.float32),
            video_position_ids=np.zeros((1, 1, 4, 2), dtype=np.int64),
        )

    def features(sequence_id, payload, *, start=0):
        if name in ("qwen", "inkling"):
            return (runner._features_for(sequence_id, payload),)
        if name == "gemma":
            result = runner._features_for(sequence_id, payload)
            return result.image, result.video
        plan = BatchPlan(
            sequence_ids=[sequence_id],
            context_lengths=[start],
            token_ids=payload.token_ids[start:],
            token_indptr=[0, len(payload.token_ids) - start],
            phase=BatchPhase.PREFILL,
            media=[payload],
        )
        if name == "deepseek":
            runner._validate_plan(plan)
            runner._media_embeddings(plan, 0, torch.tensor([payload.token_ids[start:]]))
            return (runner._media_features[sequence_id][1][0],)
        runner._media_embeddings(plan, torch.tensor(payload.token_ids[start:]))
        return (runner._media_features[sequence_id][1],)

    return SimpleNamespace(
        name=name, runner=runner, state=state, media=media, features=features
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
    assert record.positions == {
        4: ("image", 0),
        5: ("image", 1),
        6: ("video", 0),
        7: ("video", 1),
    }


def test_deepseek_keeps_image_indices_distinct():
    case = _case("deepseek")
    case.runner._vision_feature_cache = VisionFeatureCache(max_entries=2)
    first_image = case.media().images[0]
    second_image = DeepSeekV41Image(
        start=6,
        patches=np.full((2, 3, 14, 14), 0.75, dtype=np.float32),
        n_vit_h=1,
        n_vit_w=2,
        types=first_image.types,
    )

    def payload():
        return DeepSeekV41Media(
            token_ids=(7,) + (129264,) * 4 + (6,) + (129264,) * 4 + (6,),
            images=(first_image, second_image),
        )

    case.features(1, payload())
    features = case.runner._media_features[1][1]
    assert features[0][1, 0] == 0.25
    assert features[1][1, 0] == 0.75
    case.runner.release([1])
    case.features(2, payload())
    assert case.state.calls == 2
    assert case.runner._media_features[2][1][0] is features[0]
    assert case.runner._media_features[2][1][1] is features[1]
