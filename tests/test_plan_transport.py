from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from resin import (
    BatchPhase,
    BatchPlan,
    CacheGroupPlan,
    CacheStatePlan,
    SamplingParams,
)
from resin.backend.plan_transport import PlanTransport, RequestRegistration
from resin.recipe import InklingMedia


def media(value=0.5):
    return InklingMedia(
        token_ids=[7, 200054, 9],
        image_token_id=200054,
        pixel_values=np.full((1, 2, 2, 2, 3), value, dtype=np.float32),
    )


def rich_plan(batch_key):
    params = SamplingParams(
        max_output_tokens=100,
        stop_token_ids=[3, 7, 2**32 - 1],
        temperature=0.75,
        top_p=0.9,
        top_k=40,
        seed=2**63 - 1,
        logprobs=True,
        top_logprobs=3,
        prompt_logprobs=True,
        prompt_top_logprobs=2,
        extra={"penalty": 0.1, "nested": {"tokens": [1, 2], "enabled": True}},
    )
    return BatchPlan(
        sequence_ids=[2**64 - 1, 2**63 + 13],
        context_lengths=[6, 9],
        token_ids=[10, 11, 12],
        token_indptr=[0, 2, 3],
        prefill_lengths=[20, 30],
        batch_id=2**64 - 1,
        batch_key=batch_key,
        cache_groups=[
            CacheGroupPlan(
                group_index=index,
                block_indptr=[0, 2, 3],
                block_slab_ids=[1 + index * 10, 2 + index * 10, 3 + index * 10],
                block_slotnums=[0, 1, 0],
                block_table_start_indices=[4, 7],
                append_slab_ids=[2 + index * 10, 2 + index * 10, 3 + index * 10],
                append_slotnums=[1, 1, 0],
                append_block_offsets=[2, 3, 1],
                blocks_per_slab=2,
            )
            for index in range(2)
        ]
        + [
            CacheGroupPlan(
                group_index=2,
                blocks_per_slab=2,
                block_indptr=[0, 1, 2],
                block_slab_ids=[31, 32],
                block_slotnums=[1, 0],
                block_table_start_indices=[0, 0],
                append_slab_ids=[31, 31, 32],
                append_slotnums=[1, 1, 0],
                append_block_offsets=[0, 0, 0],
                state=CacheStatePlan(
                    restore_copies=[(0, 36, 0)],
                    checkpoint_slots=[(2**63 + 7, 0, 8, 41, 1)],
                ),
            )
        ],
        sampling_mask=[False, True],
        sampling_params=[params, SamplingParams(temperature=0)],
        prompt_score_indices=[0, 1, 2],
        prompt_score_token_ids=[11, 12, 13],
        prompt_score_indptr=[0, 2, 3],
        prompt_score_top_counts=[3, 0, 2],
        prompt_target_indptr=[0, 2, 2, 3],
        prompt_target_token_ids=[21, 22, 23],
    )


@pytest.mark.parametrize("batch_key", [None, 0, 2**64 - 1])
def test_packet_roundtrips_all_plan_fields_and_full_width_ids(batch_key):
    original = rich_plan(batch_key)
    packet = PlanTransport().prepare(original)
    restored = PlanTransport().restore(packet)
    assert packet.data.dtype == np.uint8 and packet.data.flags.c_contiguous
    assert len(packet.registrations) == 2
    assert restored.batch_id == original.batch_id
    assert restored.batch_key == batch_key
    assert restored.phase == BatchPhase.PREFILL
    assert restored.media == (None, None)
    for name in (
        "sequence_ids",
        "context_lengths",
        "prefill_lengths",
        "token_ids",
        "token_indptr",
        "positions",
        "sampling_mask",
        "prompt_score_indices",
        "prompt_score_token_ids",
        "prompt_score_indptr",
        "prompt_score_top_counts",
        "prompt_target_indptr",
        "prompt_target_token_ids",
    ):
        actual, expected = getattr(restored, name), getattr(original, name)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype and not actual.flags.writeable
        assert not np.shares_memory(actual, packet.data)
    for actual, expected in zip(restored.cache_groups, original.cache_groups):
        assert actual.group_index == expected.group_index
        assert actual.blocks_per_slab == expected.blocks_per_slab
        for name in (
            "block_indptr",
            "block_slab_ids",
            "block_slotnums",
            "block_table_start_indices",
            "append_slab_ids",
            "append_slotnums",
            "append_block_offsets",
        ):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name)
            )
            assert not np.shares_memory(getattr(actual, name), packet.data)
            assert not getattr(actual, name).flags.writeable
        if expected.state is None:
            assert actual.state is None
        else:
            assert actual.state is not None
            for name in ("restore_copies", "checkpoint_slots"):
                value = getattr(actual.state, name)
                assert isinstance(value, tuple)
                assert all(isinstance(item, tuple) for item in value)
                assert value == getattr(expected.state, name)
    for actual, expected in zip(restored.sampling_params, original.sampling_params):
        for name in (
            "max_output_tokens",
            "stop_token_ids",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "logprobs",
            "top_logprobs",
            "prompt_logprobs",
            "prompt_top_logprobs",
            "extra",
        ):
            assert getattr(actual, name) == getattr(expected, name)


def test_request_registration_is_reused_across_prefill_and_decode(monkeypatch):
    sender, receiver = PlanTransport(), PlanTransport()
    payload = media()
    params = SamplingParams(temperature=0.25, extra={"unchanged": True})
    first = BatchPlan(
        sequence_ids=[1],
        context_lengths=[0],
        token_ids=[7],
        token_indptr=[0, 1],
        prefill_lengths=[3],
        sampling_params=[params],
        media=[payload],
        sampling_mask=[False],
    )
    restored = receiver.restore(sender.prepare(first))
    received_media = restored.media[0]
    assert received_media is not payload

    def unexpected(*args, **kwargs):
        raise AssertionError("request metadata was serialized again")

    monkeypatch.setattr(json, "dumps", unexpected)
    for context, token, phase in (
        (1, 200054, BatchPhase.PREFILL),
        (2, 9, BatchPhase.PREFILL),
        (3, 12, BatchPhase.DECODE),
    ):
        plan = BatchPlan(
            sequence_ids=[1],
            context_lengths=[context],
            token_ids=[token],
            token_indptr=[0, 1],
            phase=phase,
            sampling_params=[params],
            media=[payload],
        )
        packet = sender.prepare(plan)
        assert packet.registrations == ()
        decoded = receiver.restore(packet)
        assert decoded.phase == phase and decoded.media[0] is received_media
        assert decoded.sampling_params[0].temperature == 0.25
        assert decoded.sampling_params[0].extra == {"unchanged": True}


def test_restore_owns_step_arrays_and_registration_buffers():
    packet = PlanTransport().prepare(
        BatchPlan(
            sequence_ids=[9],
            context_lengths=[0],
            token_ids=[7, 200054, 9],
            token_indptr=[0, 3],
            media=[media()],
        )
    )
    packet = replace(
        packet,
        registrations=tuple(
            replace(item, buffers=tuple(array.copy() for array in item.buffers))
            for item in packet.registrations
        ),
    )
    receiver = PlanTransport()
    restored = receiver.restore(packet)
    receiver.clear()
    packet.data.fill(0)
    for registration in packet.registrations:
        for array in registration.buffers:
            array.fill(0)
    assert restored.sequence_ids.tolist() == [9]
    assert restored.token_ids.tolist() == [7, 200054, 9]
    received = restored.media[0]
    assert isinstance(received, InklingMedia)
    assert received.token_ids == (7, 200054, 9)
    np.testing.assert_array_equal(received.pixel_values, 0.5)


def test_release_allows_new_request_with_same_id_without_invalidating_old_plan():
    sender, receiver = PlanTransport(), PlanTransport()
    original = media(0.25)
    first = receiver.restore(
        sender.prepare(
            BatchPlan(
                sequence_ids=[4],
                context_lengths=[0],
                token_ids=[7],
                token_indptr=[0, 1],
                media=[original],
            )
        )
    )
    replacement = media(0.75)
    second_plan = BatchPlan(
        sequence_ids=[4],
        context_lengths=[0],
        token_ids=[7],
        token_indptr=[0, 1],
        media=[replacement],
        sampling_params=[SamplingParams(temperature=0)],
    )
    with pytest.raises(ValueError, match="changed its media object"):
        sender.prepare(second_plan)
    sender.release([4, 999])
    receiver.release([4, 999])
    second_packet = sender.prepare(second_plan)
    assert len(second_packet.registrations) == 1
    second = receiver.restore(second_packet)
    assert first.media[0] is not second.media[0]
    assert isinstance(first.media[0], InklingMedia)
    assert isinstance(second.media[0], InklingMedia)
    np.testing.assert_array_equal(first.media[0].pixel_values, 0.25)
    np.testing.assert_array_equal(second.media[0].pixel_values, 0.75)
    assert second.sampling_params[0].temperature == 0


def test_prepare_is_atomic_when_later_request_has_unsupported_media():
    class CustomMedia:
        token_ids = (8,)
        cache_key = (8,)
        token_spans = ()

    sender = PlanTransport()
    payload = media()
    invalid = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[0, 0],
        token_ids=[7, 8],
        token_indptr=[0, 1, 2],
        media=[payload, CustomMedia()],
    )
    with pytest.raises(TypeError, match="media"):
        sender.prepare(invalid)
    valid = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[0, 0],
        token_ids=[7, 8],
        token_indptr=[0, 1, 2],
        media=[payload, None],
    )
    assert len(sender.prepare(valid).registrations) == 2


@pytest.mark.parametrize(
    "extra",
    [
        (1, 2),
        {1: "integer key"},
        {None: "null key"},
        {"nested": [{"tuple": (1, 2)}]},
        {"nested": [{2: "integer key"}]},
        {"nested": object()},
        {"nested": np.int64(3)},
    ],
)
def test_non_json_extra_is_rejected_without_registering_earlier_rows(extra):
    sender = PlanTransport()
    invalid = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[0, 0],
        token_ids=[7, 8],
        token_indptr=[0, 1, 2],
        sampling_params=[SamplingParams(), SamplingParams(extra=extra)],
    )
    with pytest.raises(TypeError, match=r"distributed SamplingParams.extra.*JSON"):
        sender.prepare(invalid)
    valid = BatchPlan(
        sequence_ids=[1, 2],
        context_lengths=[0, 0],
        token_ids=[7, 8],
        token_indptr=[0, 1, 2],
    )
    assert len(sender.prepare(valid).registrations) == 2


@pytest.mark.parametrize(
    "extra",
    [
        None,
        [None, True, False, 0, 2**80, 0.125, "text", [], {}],
        {"nested": [{"tokens": [1, 2], "optional": None}, {"enabled": True}]},
    ],
)
def test_json_native_extra_retains_values_and_container_types(extra, monkeypatch):
    sender, receiver = PlanTransport(), PlanTransport()
    plan = BatchPlan(
        sequence_ids=[1],
        context_lengths=[0],
        token_ids=[7],
        token_indptr=[0, 1],
        sampling_params=[SamplingParams(extra=extra)],
    )
    restored = receiver.restore(sender.prepare(plan))
    assert restored.sampling_params[0].extra == extra
    assert type(restored.sampling_params[0].extra) is type(extra)

    def unexpected(*args):
        raise AssertionError("registered extra was validated again")

    import resin.backend.plan_transport as transport

    monkeypatch.setattr(transport, "_is_json_extra", unexpected)
    packet = sender.prepare(plan)
    assert packet.registrations == ()
    assert receiver.restore(packet).sampling_params[0].extra == extra


@pytest.mark.parametrize("extra", [float("nan"), {"nested": [float("inf")]}])
def test_nonfinite_extra_remains_rejected_by_json_encoder(extra):
    sender = PlanTransport()
    plan = BatchPlan(
        sequence_ids=[1],
        context_lengths=[0],
        token_ids=[7],
        token_indptr=[0, 1],
        sampling_params=[SamplingParams(extra=extra)],
    )
    with pytest.raises(ValueError, match="JSON"):
        sender.prepare(plan)
    assert (
        len(
            sender.prepare(
                BatchPlan(
                    sequence_ids=[1],
                    context_lengths=[0],
                    token_ids=[7],
                    token_indptr=[0, 1],
                )
            ).registrations
        )
        == 1
    )


def test_restore_is_atomic_and_requires_registration():
    sender, receiver = PlanTransport(), PlanTransport()
    packet = sender.prepare(
        BatchPlan(
            sequence_ids=[1], context_lengths=[0], token_ids=[7], token_indptr=[0, 1]
        )
    )
    with pytest.raises(ValueError, match="has not been registered"):
        receiver.restore(replace(packet, registrations=()))
    malformed = RequestRegistration(b'{"version":99}')
    with pytest.raises(ValueError, match="registration version"):
        receiver.restore(
            replace(packet, registrations=(*packet.registrations, malformed))
        )
    assert receiver.restore(packet).sequence_ids.tolist() == [1]
    with pytest.raises(ValueError, match="already registered"):
        receiver.restore(packet)
    receiver.clear()
    assert receiver.restore(packet).sequence_ids.tolist() == [1]


@pytest.mark.parametrize(
    "damage", ["dtype", "strided", "magic", "version", "truncated", "trailing"]
)
def test_malformed_packet_cannot_commit_registrations(damage):
    packet = PlanTransport().prepare(
        BatchPlan(
            sequence_ids=[1], context_lengths=[0], token_ids=[7], token_indptr=[0, 1]
        )
    )
    receiver = PlanTransport()
    match damage:
        case "dtype":
            data = packet.data.astype(np.int64)
        case "strided":
            storage = np.zeros(packet.data.size * 2, dtype=np.uint8)
            storage[::2] = packet.data
            data = storage[::2]
        case "magic":
            data = packet.data.copy()
            data[0] ^= 0xFF
        case "version":
            data = packet.data.copy()
            data[8] ^= 0xFF
        case "truncated":
            data = packet.data[:-1]
        case _:
            data = np.append(packet.data, np.uint8(0))
    with pytest.raises((ValueError, TypeError)):
        receiver.restore(replace(packet, data=data))
    assert receiver.restore(packet).token_ids.tolist() == [7]


def test_archive_accepts_unaligned_transport_storage():
    packet = PlanTransport().prepare(rich_plan(None))
    storage = np.zeros(packet.data.size + 1, dtype=np.uint8)
    storage[1:] = packet.data
    assert storage[1:].ctypes.data % 8 != 0
    restored = PlanTransport().restore(replace(packet, data=storage[1:]))
    np.testing.assert_array_equal(restored.token_ids, [10, 11, 12])
    storage.fill(0)
    np.testing.assert_array_equal(restored.cache_groups[0].block_indptr, [0, 2, 3])


def test_clear_requires_registering_live_requests_again():
    sender = PlanTransport()
    plan = BatchPlan(
        sequence_ids=[1], context_lengths=[0], token_ids=[7], token_indptr=[0, 1]
    )
    assert len(sender.prepare(plan).registrations) == 1
    assert sender.prepare(plan).registrations == ()
    sender.clear()
    assert len(sender.prepare(plan).registrations) == 1
