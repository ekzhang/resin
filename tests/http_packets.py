"""Wire fixtures for the private native distributed-HTTP packet boundary."""

import json

from resin._resin import _HttpEvents, _HttpIngress


def ingress(requests=(), *, canceled=()):
    return _HttpIngress.from_bytes(
        json.dumps({"requests": requests, "canceled": canceled}).encode()
    )


def decoded_events(packet: _HttpEvents):
    return json.loads(packet.to_bytes() or b'{"events":[]}')["events"]
