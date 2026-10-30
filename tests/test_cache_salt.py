"""Cache namespaces preserve prompt identities through native serving and storage."""

import hashlib
import json
from contextlib import closing
from dataclasses import dataclass
from time import sleep

import numpy as np
import pytest
from http_packets import decoded_events, ingress
from test_cache_storage import make_engine
from test_flat_cache import RecordingCachedRunner

from resin import CacheGroupLayout, CacheLayout, Engine, SamplingParams, recipe
from resin._cache_storage import _PrefixStorage
from resin._resin import _HttpIngress, _HttpSession
from resin.cache_storage import MemoryCacheStorage


@dataclass
class Media:
    cache_key: tuple[int, ...] = (101, 102, 103, 104, 105)
    token_ids = (1, 2, 3, 4, 5)
    token_spans = ()


def complete(engine, **kwargs):
    request = engine.submit(
        sampling_params=SamplingParams(max_output_tokens=1), **kwargs
    )
    while not request.done:
        engine.step()
    assert request.take_result() == [6]
    return request.cached_tokens


@pytest.mark.parametrize("media", [False, True])
def test_same_prompt_reuses_only_its_salt_namespace(media):
    runner = RecordingCachedRunner(
        CacheLayout(parent_capacity=32, groups=[CacheGroupLayout(2)])
    )
    prompt = {"media": Media()} if media else {"token_ids": [1, 2, 3, 4, 5]}
    with Engine(runner, prefix_cache=True) as engine:
        for salt in (None, "", "alice", "bob", "λ\x00namespace"):
            assert complete(engine, cache_salt=salt, **prompt) == 0
            assert complete(engine, cache_salt=salt, **prompt) == 4
            assert runner.plans[-1].token_ids.tolist() == [5]
        if media:
            assert (
                complete(
                    engine, cache_salt="alice", media=Media((201, 202, 203, 204, 205))
                )
                == 0
            )
            assert complete(engine, cache_salt="alice", media=Media()) == 4
        # Omitting the argument preserves the None namespace.
        assert complete(engine, **prompt) == 4


@pytest.mark.parametrize("salt", [0, True, b"alice", [], {}])
def test_engine_rejects_non_string_cache_salts(salt):
    runner = RecordingCachedRunner(
        CacheLayout(parent_capacity=4, groups=[CacheGroupLayout(2)])
    )
    with Engine(runner, prefix_cache=True) as engine:
        with pytest.raises(TypeError, match="cache_salt"):
            engine.submit([1, 2, 3], cache_salt=salt)
        assert engine.active_request_count == 0


@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("policy", ["write_through", "write_through_selective"])
def test_storage_namespace_survives_reads_writes_and_engine_recreation(media, policy):
    prompt = {"media": Media()} if media else {"token_ids": [1, 2, 3, 4, 5]}
    with closing(MemoryCacheStorage(max_bytes=4096)) as store:
        for salt, expected in (
            ("alice", 0),
            ("alice", 4),
            ("bob", 0),
            ("bob", 4),
            (None, 0),
            (None, 4),
            ("", 0),
            ("", 4),
        ):
            engine, _, arena = make_engine(
                store, fill=73 if expected == 0 else 0, policy=policy
            )
            with engine:
                request = engine.submit(
                    sampling_params=SamplingParams(max_output_tokens=1),
                    cache_salt=salt,
                    **prompt,
                )
                for _ in range(1000):
                    if request.done:
                        break
                    engine.step()
                    sleep(0.001)
                assert request.done
                assert request.cached_tokens == expected
                assert request.take_result() == [6]
                if expected:
                    assert bool((arena == 73).any())
                else:
                    # The default selective policy backs up a checkpoint on reuse.
                    assert complete(engine, cache_salt=salt, **prompt) == 4


def test_unsalted_storage_keys_are_unchanged_and_disjoint_from_salted_keys():
    storage = object.__new__(_PrefixStorage)
    storage.alignment = 2
    tokens = [1, 2, 3, 4]
    encoded = np.asarray(tokens, dtype="<u4").tobytes()
    expected = [hashlib.sha256(encoded[:end]).digest() for end in (8, 16)]
    assert storage._keys(tokens) == storage._keys(tokens, None) == expected
    keys = [storage._keys(tokens, salt) for salt in (None, "", "alice", "bob")]
    assert all(
        set(left).isdisjoint(right)
        for i, left in enumerate(keys)
        for right in keys[i + 1 :]
    )


def test_distributed_http_ingress_preserves_cache_namespace_at_admission():
    class Recipe:
        tokenizer = None
        stop_token_ids = []

        def render(self, messages, *, tools=(), options=None):
            return recipe.RenderedPrompt([1, 2, 3, 4, 5])

        def create_parser(self, prompt):
            class Parser:
                def feed(self, ids):
                    return []

            return Parser()

    runner = RecordingCachedRunner(
        CacheLayout(parent_capacity=32, groups=[CacheGroupLayout(2)])
    )
    with Engine(runner, prefix_cache=True) as engine:
        session = _HttpSession(engine, Recipe())
        for index, (salt, expected) in enumerate(
            (("alice", 0), ("alice", 4), ("bob", 0), (None, 0), ("", 0))
        ):
            packet = ingress(
                [
                    {
                        "id": str(index),
                        "model": "test",
                        "messages": [],
                        "tools": [],
                        "max_output_tokens": 1,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": None,
                        "seed": None,
                        "logprobs": False,
                        "top_logprobs": 0,
                        "prompt_logprobs": False,
                        "routed_dp_rank": None,
                        "stop": [],
                        "reasoning_effort": None,
                        "extra": {},
                        "cache_salt": salt,
                    }
                ]
            )
            encoded = packet.to_bytes()
            assert json.loads(encoded)["requests"][0]["cache_salt"] == salt
            session.admit(_HttpIngress.from_bytes(encoded))
            events = []
            while engine.has_work():
                events.extend(decoded_events(session.process(engine.step())))
            final = next(event for _, event in events if event["type"] == "finished")
            assert final["data"]["cached_tokens"] == expected
        session.close()
