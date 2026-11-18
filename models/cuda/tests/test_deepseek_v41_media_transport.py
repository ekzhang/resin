"""Registered media stays stable across follower prefill/decode plans."""

import pytest
import torch
from test_deepseek_v41_vision import TOKENS, media_fixture, plan_for, runner_fixture

from resin import BatchPhase, BatchPlan
from resin.backend.plan_transport import PlanTransport
from resin.recipe import Media


def test_registered_prefill_chunks_and_decode_reuse_image_features():
    media = media_fixture()
    leader, _ = runner_fixture()
    follower, calls = runner_fixture()
    sender, receiver = PlanTransport(), PlanTransport()
    registered_media = None
    for start, end in ((0, 5), (5, 7), (7, len(TOKENS))):
        original = plan_for(media, start, end)
        packet = sender.prepare(original)
        assert len(packet.registrations) == (1 if registered_media is None else 0)
        restored = receiver.restore(packet)
        current = restored.media[0]
        assert current is not None
        assert all(isinstance(item, Media) for item in current)
        assert current is not media
        assert current[0].digest == media[0].digest
        if registered_media is None:
            registered_media = current
        else:
            assert current is registered_media
        follower._validate_plan(restored)
        input_ids = torch.tensor([TOKENS[start:end]])
        torch.testing.assert_close(
            follower._media_embeddings(restored, 0, input_ids),
            leader._media_embeddings(original, 0, input_ids),
        )
    assert len(calls) == 1
    cached = follower._media_features[42]
    for position in range(len(TOKENS), len(TOKENS) + 2):
        packet = sender.prepare(
            BatchPlan(
                sequence_ids=[42],
                context_lengths=[position],
                token_ids=[21],
                token_indptr=[0, 1],
                phase=BatchPhase.DECODE,
                media=[media],
            )
        )
        assert packet.registrations == ()
        decode = receiver.restore(packet)
        assert decode.media[0] is cached[0] is registered_media
        assert follower._validate_plan(decode) == ((0, 1),)
        assert follower._media_embeddings(decode, 0, torch.tensor([[21]])) is None
        assert follower._media_features[42] is cached
    assert len(calls) == 1
    follower.release([42])
    sender.release([42])
    receiver.release([42])
    assert not follower._media_features
    packet = sender.prepare(plan_for(media, 0, 5))
    assert len(packet.registrations) == 1
    assert receiver.restore(packet).media[0] is not registered_media


@pytest.mark.parametrize("change", ["pixels", "placement"])
@pytest.mark.parametrize("phase", [BatchPhase.PREFILL, BatchPhase.DECODE])
def test_registered_plans_reject_changed_media_or_prompt(change, phase):
    media = media_fixture()
    follower, _ = runner_fixture()
    sender, receiver = PlanTransport(), PlanTransport()
    initial = receiver.restore(sender.prepare(plan_for(media, 0, 5)))
    follower._validate_plan(initial)
    follower._media_embeddings(initial, 0, torch.tensor([TOKENS[:5]]))
    original = media[0]
    changed = (
        Media(span=original.span, image=original.image + 1)
        if change == "pixels"
        else Media(span=(1, original.span[1]), image=original.image),
    )
    plan = (
        plan_for(changed, 5, 7)
        if phase == BatchPhase.PREFILL
        else BatchPlan(
            sequence_ids=[42],
            context_lengths=[len(TOKENS)],
            token_ids=[21],
            token_indptr=[0, 1],
            phase=phase,
            media=[changed],
        )
    )
    with pytest.raises(ValueError, match="changed.*media"):
        sender.prepare(plan)
