"""Cache namespaces preserve prompt identities through native serving and storage."""

import hashlib
import json
from contextlib import closing
from dataclasses import dataclass
from time import sleep
from types import SimpleNamespace

import numpy as np
import pytest
from test_cache_storage import make_engine
from test_flat_cache import RecordingCachedRunner
from test_http_lifecycle import post, request_body, running_server

from resin import CacheGroupLayout, CacheLayout, Engine, SamplingParams, recipe
from resin._cache_storage import _PrefixStorage
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
            ("λ\x00namespace", 0),
            ("λ\x00namespace", 4),
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
    assert storage._keys(([], tokens)) == expected
    namespaces = [[], [2**32 - 1], [97, 2**32 - 1], [97, 98, 99, 2**32 - 1]]
    keys = [storage._keys((namespace, tokens)) for namespace in namespaces]
    assert all(
        set(left).isdisjoint(right)
        for i, left in enumerate(keys)
        for right in keys[i + 1 :]
    )
    # An extended salt cannot absorb token words from a shorter namespace.
    assert (
        storage._keys((namespaces[2], [98, 99, 1, 2]))[-1]
        != storage._keys((namespaces[3], [1, 2]))[-1]
    )


def test_http_requests_preserve_cache_namespace_at_admission():
    prompt_recipe = SimpleNamespace(
        tokenizer=None,
        stop_token_ids=[],
        render=lambda *args, **kwargs: recipe.RenderedPrompt([1, 2, 3, 4, 5]),
        create_parser=lambda prompt: SimpleNamespace(feed=lambda ids: []),
    )
    runner = RecordingCachedRunner(
        CacheLayout(parent_capacity=32, groups=[CacheGroupLayout(2)])
    )
    with running_server(
        runner, prompt_recipe, engine_options={"prefix_cache": True}
    ) as (_, _, _, errors, base, _):
        for salt, expected in (
            ("alice", 0),
            ("alice", 4),
            ("bob", 0),
            (None, 0),
            ("", 0),
        ):
            body = {**json.loads(request_body(max_tokens=1)), "cache_salt": salt}
            with post(base, json.dumps(body).encode()) as response:
                usage = json.load(response)["usage"]
            assert usage["prompt_tokens_details"]["cached_tokens"] == expected
        assert errors == []
