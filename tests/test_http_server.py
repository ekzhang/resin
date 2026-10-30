from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from output_helpers import forward_output

import resin
from resin import Engine
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
    with pytest.raises(ValueError, match="requires enable_openai"):
        HttpServer(
            Engine(IncrementingRunner()),
            recipe=_Recipe(),
            model="test",
            enable_openai=False,
            enable_demo=True,
        )


@pytest.mark.parametrize(
    "setting,value",
    [
        (setting, value)
        for setting in (
            "stream_interval",
            "request_queue_capacity",
            "max_active_requests",
            "render_workers",
            "admissions_per_step",
        )
        for value in (0, -1, 2**32, True, False, 1.5, "2", None)
        if setting not in ("max_active_requests", "render_workers") or value is not None
    ],
)
def test_http_server_requires_positive_integer_settings(setting, value) -> None:
    engine = Engine(IncrementingRunner())
    with pytest.raises(ValueError, match=f"{setting}.*positive integer"):
        HttpServer(engine, recipe=_Recipe(), model="test", **{setting: value})
    assert not engine.closed
    engine.close()


@pytest.mark.parametrize(
    "setting,default",
    [
        ("stream_interval", 1),
        ("request_queue_capacity", 1024),
        ("max_active_requests", None),
        ("render_workers", None),
        ("admissions_per_step", 32),
    ],
)
@pytest.mark.parametrize("value", [None, 4, 2**32 - 1])
def test_http_server_forwards_default_and_custom_settings(
    setting, default, value, monkeypatch
):
    import resin.server as serving

    created = []

    def native(*args, **kwargs):
        created.append(kwargs)
        return SimpleNamespace(running=False)

    monkeypatch.setattr(serving, "_HttpServer", native)
    engine = Engine(IncrementingRunner())
    options = {} if value is None else {setting: value}
    server = HttpServer(engine, recipe=_Recipe(), model="test", **options)
    assert server.model == "test"
    assert created[0][setting] == (default if value is None else value)
    engine.close()


@pytest.mark.parametrize("render_workers", [3, 32])
def test_native_http_server_derives_active_limit_from_engine_and_render_workers(
    render_workers,
):
    from resin._resin import _HttpServer

    engine = Engine(IncrementingRunner(), max_running_requests=192)
    server = _HttpServer(engine, _Recipe(), model="test", render_workers=render_workers)
    assert f"render_workers={render_workers}" in repr(server)
    assert f"max_active_requests={192 + render_workers}" in repr(server)
    engine.close()


def test_native_http_server_resolves_automatic_render_workers():
    from resin._resin import _HttpServer

    engine = Engine(IncrementingRunner())
    server = _HttpServer(engine, _Recipe(), model="test")
    workers = re.search(r"render_workers=(\d+)", repr(server))
    assert workers is not None
    render_workers = int(workers[1])
    assert 1 <= render_workers <= 16
    assert (
        f"max_active_requests={engine.max_running_requests + render_workers}"
        in repr(server)
    )
    engine.close()


@pytest.mark.parametrize(
    "setting",
    [
        "stream_interval",
        "request_queue_capacity",
        "max_active_requests",
        "render_workers",
        "admissions_per_step",
    ],
)
def test_native_http_server_rejects_zero_settings(setting):
    from resin._resin import _HttpServer

    engine = Engine(IncrementingRunner())
    with pytest.raises(ValueError, match=f"{setting}.*positive"):
        _HttpServer(
            engine,
            _Recipe(),
            model="test",
            stream_interval=0 if setting == "stream_interval" else 1,
            request_queue_capacity=0 if setting == "request_queue_capacity" else 1024,
            max_active_requests=0 if setting == "max_active_requests" else None,
            render_workers=0 if setting == "render_workers" else None,
            admissions_per_step=0 if setting == "admissions_per_step" else 32,
        )
    engine.close()


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_http_server_requires_boolean_priority_setting(value):
    engine = Engine(IncrementingRunner())
    with pytest.raises(ValueError, match="flags must be booleans"):
        HttpServer(
            engine, recipe=_Recipe(), model="test", enable_priority_scheduling=value
        )
    engine.close()


@pytest.mark.parametrize("value", [None, False, True])
def test_http_server_forwards_priority_opt_in(value, monkeypatch):
    import resin.server as serving

    created = []
    monkeypatch.setattr(
        serving, "_HttpServer", lambda *args, **kwargs: created.append(kwargs)
    )
    engine = Engine(IncrementingRunner())
    options = {} if value is None else {"enable_priority_scheduling": value}
    HttpServer(engine, recipe=_Recipe(), model="test", **options)
    assert created[0]["enable_priority_scheduling"] is (
        False if value is None else value
    )
    engine.close()
