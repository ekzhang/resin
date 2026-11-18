"""Wire fixtures for the private native distributed-HTTP packet boundary."""

import json
import time

from resin._resin import _HttpEvents, _HttpIngress, _HttpSession


def ingress(requests=(), *, canceled=()):
    return _HttpIngress.from_bytes(
        json.dumps({"requests": requests, "canceled": canceled}).encode()
    )


def admit_and_wait(session: _HttpSession, packet: _HttpIngress, *, expected: int = 1):
    processed = session.admit(packet)
    deadline = time.monotonic() + 5
    empty = ingress()
    while processed < expected:
        assert time.monotonic() < deadline, (
            f"HTTP admission processed {processed} of {expected} requests"
        )
        time.sleep(0.001)
        processed += session.admit(empty)
    assert processed == expected


def decoded_events(packet: _HttpEvents):
    return json.loads(packet.to_bytes() or b'{"events":[]}')["events"]
