"""Packed step metadata with request-scoped sampling and native media ownership."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from resin import BatchPlan, SamplingParams
from resin.recipe import Media

_VERSION = 1
_SAMPLING_FIELDS = (
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
)


@dataclass(frozen=True, slots=True)
class RequestRegistration:
    """One-time JSON policy metadata and separate native media arrays."""

    metadata: bytes
    buffers: tuple[npt.NDArray[Any], ...] = ()


@dataclass(frozen=True, slots=True)
class PlanPacket:
    """Native plan archive and any newly admitted request state."""

    data: npt.NDArray[np.uint8]
    registrations: tuple[RequestRegistration, ...] = ()


@dataclass(frozen=True, slots=True)
class _Request:
    sampling_params: SamplingParams
    media: tuple[Media, ...] | None


def _is_json_extra(value: Any) -> bool:
    if type(value) is dict:
        return all(
            type(key) is str and _is_json_extra(item) for key, item in value.items()
        )
    if type(value) is list:
        return all(_is_json_extra(item) for item in value)
    return value is None or type(value) in (bool, int, float, str)


def _register(sequence_id: int, request: _Request) -> RequestRegistration:
    sampling = {
        name: getattr(request.sampling_params, name) for name in _SAMPLING_FIELDS
    }
    if not _is_json_extra(sampling["extra"]):
        raise TypeError(
            "distributed SamplingParams.extra requires JSON-native values: "
            "string-keyed dicts, lists, and None/bool/int/float/str scalars"
        )
    media_metadata = None
    buffers = ()
    if request.media is not None:
        from resin._resin import _media_to_buffers

        encoded, buffers = _media_to_buffers(request.media)
        media_metadata = encoded.decode("utf-8")
    metadata = json.dumps(
        {
            "version": _VERSION,
            "sequence_id": sequence_id,
            "sampling": sampling,
            "media": media_metadata,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return RequestRegistration(metadata, buffers)


def _restore_registration(registration: RequestRegistration) -> tuple[int, _Request]:
    metadata = json.loads(registration.metadata)
    if not isinstance(metadata, dict) or metadata.get("version") != _VERSION:
        raise ValueError("unsupported request registration version")
    sequence_id = metadata["sequence_id"]
    if type(sequence_id) is not int or not 0 <= sequence_id < 2**64:
        raise ValueError("request sequence_id must be a uint64")
    sampling = SamplingParams(**metadata["sampling"])
    media_metadata = metadata["media"]
    if media_metadata is None:
        if registration.buffers:
            raise ValueError("request without media cannot contain media buffers")
        media = None
    else:
        from resin._resin import _media_from_buffers

        media = _media_from_buffers(
            media_metadata.encode("utf-8"), registration.buffers
        )
    return sequence_id, _Request(sampling, media)


class PlanTransport:
    """Keep request state once while transporting only typed arrays each step.

    Sender and receiver each own a registry. Releasing a request drops only the
    registry reference; previously restored plans retain their media and arrays.
    """

    def __init__(self) -> None:
        self._requests: dict[int, _Request] = {}

    def prepare(self, plan: BatchPlan) -> PlanPacket:
        pending: dict[int, _Request] = {}
        registrations = []
        sampling = None
        for row, (raw_id, media) in enumerate(zip(plan.sequence_ids, plan.media)):
            sequence_id = int(raw_id)
            existing = self._requests.get(sequence_id)
            if existing is not None:
                if existing.media is not media:
                    raise ValueError(f"request {sequence_id} changed its media object")
                continue
            if sampling is None:
                sampling = plan.sampling_params
            request = _Request(sampling[row], media)
            registrations.append(_register(sequence_id, request))
            pending[sequence_id] = request
        packet = PlanPacket(plan._to_wire(), tuple(registrations))
        self._requests.update(pending)
        return packet

    def restore(self, packet: PlanPacket) -> BatchPlan:
        pending: dict[int, _Request] = {}
        for registration in packet.registrations:
            sequence_id, request = _restore_registration(registration)
            if sequence_id in self._requests or sequence_id in pending:
                raise ValueError(f"request {sequence_id} was already registered")
            pending[sequence_id] = request
        # The native plan defines its wire schema and validates the restored
        # arrays. Request objects retain their identity; step arrays are owned
        # independently of the receive buffer and earlier in-flight plans.
        plan = BatchPlan._from_wire(packet.data, self._requests, pending)
        self._requests.update(pending)
        return plan

    def release(self, sequence_ids: Iterable[int]) -> None:
        for sequence_id in sequence_ids:
            self._requests.pop(int(sequence_id), None)

    def clear(self) -> None:
        self._requests.clear()
