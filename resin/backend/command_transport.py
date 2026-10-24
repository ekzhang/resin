"""Typed CPU broadcasts for replica commands and prepared execution plans."""

from __future__ import annotations

import json

import numpy as np
import torch

from resin.backend.plan_transport import PlanPacket, RequestRegistration


class CommandTransport:
    """Two broadcasts per forward: a fixed header and packed plan buffers.

    Request registrations follow only on admission. Their descriptors use JSON
    and their arrays travel as raw buffers, never as serialized Python objects.
    Received packets own their storage through plan reconstruction, including
    while an earlier asynchronous forward is still using its own plan.
    """

    _operations = (
        "bind",
        "release",
        "forward",
        "finish",
        "close",
        "metrics",
        "metrics_snapshot",
    )
    _version = 3
    _header_size = 13

    def __init__(self, distributed, group, *, source=0):
        self.distributed, self.group, self.source = distributed, group, source

    def _broadcast(self, array):
        # All traffic uses bytes: Gloo need not support every NumPy scalar dtype.
        # Native prompt-token views are immutable; PyTorch requires writable
        # storage even on the source rank. Pixel buffers already satisfy this.
        if not array.flags.writeable or not array.flags.c_contiguous:
            array = array.copy(order="C")
        # Gloo writes on its own thread, outside the caller's inference scope.
        with torch.inference_mode(False):
            tensor = torch.from_numpy(array.reshape(-1).view(np.uint8))
        self.distributed.broadcast(tensor, src=self.source, group=self.group)

    def send(self, command):
        packet = command.get("payload")
        sizes = np.asarray(command.get("sizes", ()), dtype=np.uint64).reshape(-1)
        ids = np.asarray(command.get("ids", ()), dtype=np.uint64)
        error = (command.get("error") or "").encode()
        header = np.array(
            [
                self._version,
                self._operations.index(command["operation"]),
                command.get("step_id", 0),
                int(command.get("phase") == "decode"),
                int(command.get("enqueue", False)),
                int(command.get("feedback", False)),
                command.get("cache_pool_id", 0),
                sizes.size // 2,
                ids.size,
                len(error),
                0 if packet is None else packet.data.size,
                0 if packet is None else len(packet.registrations),
                int(command.get("metrics", False)),
            ],
            dtype=np.uint64,
        )
        pieces = [
            sizes.view(np.uint8),
            ids.view(np.uint8),
            np.frombuffer(error, np.uint8),
        ]
        if packet is not None:
            pieces.append(packet.data)
        body = np.concatenate(pieces)
        self._broadcast(header)
        if body.size:
            self._broadcast(body)
        if packet is not None:
            for registration in packet.registrations:
                self._send_registration(registration)

    def receive(self):
        header = np.empty(self._header_size, dtype=np.uint64)
        self._broadcast(header)
        (
            version,
            operation,
            step,
            phase,
            enqueue,
            feedback,
            cache_pool_id,
            size_count,
            id_count,
            error_size,
            plan_size,
            registrations,
            metrics,
        ) = map(int, header)
        if version != self._version or operation >= len(self._operations):
            raise ValueError("unsupported replica command format")
        details_size = (size_count * 2 + id_count) * 8
        body = np.empty(details_size + error_size + plan_size, np.uint8)
        if body.size:
            self._broadcast(body)
        details = body[:details_size].view(np.uint64)
        offset = details_size + error_size
        admitted = tuple(self._receive_registration() for _ in range(registrations))
        return {
            "operation": self._operations[operation],
            "step_id": step,
            "phase": "decode" if phase else "prefill",
            "enqueue": bool(enqueue),
            "feedback": bool(feedback),
            "metrics": bool(metrics),
            "cache_pool_id": cache_pool_id,
            "sizes": tuple(
                map(tuple, details[: size_count * 2].reshape(-1, 2).tolist())
            ),
            "ids": details[size_count * 2 :].tolist(),
            "error": body[details_size:offset].tobytes().decode() or None,
            "payload": PlanPacket(body[offset:], admitted) if plan_size else None,
        }

    def _send_registration(self, registration):
        descriptor = json.dumps(
            {
                "metadata": registration.metadata.decode(),
                "buffers": [
                    (array.dtype.str, array.shape) for array in registration.buffers
                ],
            },
            separators=(",", ":"),
        ).encode()
        self._broadcast(np.array([len(descriptor)], dtype=np.uint64))
        self._broadcast(np.frombuffer(descriptor, np.uint8).copy())
        for array in registration.buffers:
            if array.size:
                self._broadcast(array)

    def _receive_registration(self):
        size = np.empty(1, np.uint64)
        self._broadcast(size)
        descriptor = np.empty(int(size[0]), np.uint8)
        self._broadcast(descriptor)
        fields = json.loads(descriptor.tobytes())
        buffers = []
        for dtype, shape in fields["buffers"]:
            dtype = np.dtype(dtype)
            if dtype.kind not in "biuf" or dtype.hasobject:
                raise ValueError("request buffers must contain numeric data")
            array = np.empty(shape, dtype)
            if array.size:
                self._broadcast(array)
            buffers.append(array)
        return RequestRegistration(fields["metadata"].encode(), tuple(buffers))
