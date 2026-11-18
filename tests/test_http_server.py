from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from output_helpers import forward_output

import resin
from resin import Engine
from resin._resin import _HttpServer
from resin.server import HttpServer


class IncrementingRunner:
    @staticmethod
    def forward(plan):
        return forward_output(
            [
                plan.token_ids[end - 1] + 1
                for _, end in map(plan.token_range, range(plan.num_sequences))
            ]
        )


class _Recipe:
    tokenizer = None
    stop_token_ids = []

    def render(self, messages, *, tools=(), options=None):
        return resin.recipe.RenderedPrompt([1])

    def create_parser(self, prompt):
        return SimpleNamespace(feed=lambda ids: [])


def test_http_server_validates_route_configuration() -> None:
    with Engine(IncrementingRunner()) as engine:
        with pytest.raises(ValueError, match="requires enable_openai"):
            HttpServer(engine, recipe=_Recipe(), model="test", enable_openai=False)
        with pytest.raises(ValueError, match="flags must be booleans"):
            HttpServer(
                engine,
                recipe=_Recipe(),
                model="test",
                enable_priority_scheduling=1,  # ty: ignore[invalid-argument-type]
            )


@pytest.mark.parametrize(
    "setting,value",
    [
        ("stream_interval", 0),
        ("stream_interval", True),
        ("request_queue_capacity", -1),
        ("max_active_requests", 2**32),
        ("render_workers", 1.5),
        ("admissions_per_step", None),
        ("admissions_per_step", "2"),
    ],
)
def test_http_server_requires_positive_integer_settings(setting, value) -> None:
    with Engine(IncrementingRunner()) as engine:
        with pytest.raises(ValueError, match=f"{setting}.*positive integer"):
            HttpServer(engine, recipe=_Recipe(), model="test", **{setting: value})
        assert not engine.closed


@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "stream_interval": 4,
            "request_queue_capacity": 2**32 - 1,
            "render_workers": 32,
            "admissions_per_step": 4,
            "enable_priority_scheduling": True,
        },
    ],
    ids=["defaults", "custom"],
)
def test_http_server_passes_configuration_to_native_server(options):
    with Engine(IncrementingRunner(), max_running_requests=192) as engine:
        server = HttpServer(engine, recipe=_Recipe(), model="test", **options)
        assert server.model == "test"
        native = repr(server._native)
        for name, default in (
            ("stream_interval", 1),
            ("request_queue_capacity", 1024),
            ("admissions_per_step", 32),
            ("enable_priority_scheduling", False),
        ):
            expected = options.get(name, default)
            assert f"{name}={str(expected).lower()}," in native
        workers = re.search(r"render_workers=(\d+),", native)
        assert workers is not None
        render_workers = int(workers[1])
        if options:
            assert render_workers == options["render_workers"]
        else:
            assert 1 <= render_workers <= 16
        assert f"max_active_requests={192 + render_workers}," in native


def test_native_http_server_validates_settings_without_public_wrapper():
    with Engine(IncrementingRunner()) as engine:
        with pytest.raises(ValueError, match="stream_interval must be positive"):
            _HttpServer(engine, _Recipe(), model="test", stream_interval=0)
