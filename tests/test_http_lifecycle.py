from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, contextmanager

import pytest
from output_helpers import forward_output
from test_recipe import _byte_tokenizer, _FastTokenizer, _token_ids

import resin
from resin import BatchPhase, Engine
from resin.backend.cuda.failures import PostAcceptCudaError
from resin.recipe import RenderedPrompt, Tokenizer, get_native_recipe
from resin.server import HttpServer


class Recipe:
    tokenizer = None
    stop_token_ids = []

    def render(self, messages, *, tools=(), options=None):
        slow = messages[0].content[0].text == "slow"
        return RenderedPrompt([1 if slow else 2])

    def create_parser(self, prompt):
        delta = "x" * 65_536 if prompt.token_ids.tolist() == [1] else "ok"

        class Parser:
            def feed(self, token_ids):
                return [resin.recipe.TextDelta(delta)]

        return Parser()


class Runner:
    def __init__(self):
        self.counts = {}
        self.slow_id = None
        self.started = threading.Event()
        self.slow_released = threading.Event()

    def forward(self, plan):
        for row, sequence_id in enumerate(plan.sequence_ids):
            self.counts[int(sequence_id)] = self.counts.get(int(sequence_id), 0) + 1
            if (
                plan.phase == BatchPhase.PREFILL
                and plan.token_ids[plan.token_indptr[row]] == 1
            ):
                self.slow_id = int(sequence_id)
                self.started.set()
        return forward_output([3] * plan.num_sequences)

    def release(self, sequence_ids):
        if self.slow_id in sequence_ids:
            self.slow_released.set()


def request_body(content="fast", *, stream=False, max_tokens=2):
    return json.dumps(
        {
            "model": "test",
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
            "max_tokens": max_tokens,
        }
    ).encode()


@contextmanager
def running_server(runner, recipe=None, *, engine_options=None, **options):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    engine = Engine(runner, **(engine_options or {}))
    server = HttpServer(engine, recipe=recipe or Recipe(), model="test", **options)
    errors = []

    def serve():
        try:
            server.serve(port=port)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
    while True:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=0.2) as response:
                assert response.status == 200
                break
        except (OSError, urllib.error.URLError):
            if errors or time.monotonic() >= deadline:
                server.shutdown()
                thread.join(timeout=2)
                raise AssertionError(f"server did not become ready: {errors}")
            time.sleep(0.01)
    try:
        yield server, engine, thread, errors, base, port
    finally:
        server.shutdown()
        thread.join(timeout=7)
        assert not thread.is_alive(), (
            "shutdown left the inference actor or listener running"
        )


def post(base, body):
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=5)


def test_programmatic_shutdown_stops_idle_server_and_closes_engine():
    with running_server(Runner()) as (server, engine, thread, errors, _base, port):
        server.shutdown()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not server.running
        assert engine.closed
        assert errors == []
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.2)
        server.shutdown()  # Shutdown remains harmless after serving has stopped.


def test_configured_intake_queue_rejects_overflow_then_drains_and_recovers():
    rendering = threading.Event()
    resume = threading.Event()

    class BlockingRecipe(Recipe):
        def render(self, messages, *, tools=(), options=None):
            if not rendering.is_set():
                rendering.set()
                assert resume.wait(timeout=5), "test did not release the first render"
            return super().render(messages, tools=tools, options=options)

    with running_server(
        Runner(), BlockingRecipe(), request_queue_capacity=2, max_active_requests=1
    ) as (
        _server,
        engine,
        _thread,
        errors,
        base,
        _port,
    ):
        with ExitStack() as responses:
            streams = []
            try:
                streams.append(
                    responses.enter_context(post(base, request_body(stream=True)))
                )
                assert rendering.wait(timeout=2)
                # The actor holds the first job; exactly two more fit in ingress.
                for _ in range(2):
                    streams.append(
                        responses.enter_context(post(base, request_body(stream=True)))
                    )
                with pytest.raises(urllib.error.HTTPError) as caught:
                    post(base, request_body())
                with caught.value as response:
                    assert response.code == 429
                    assert "queue is full" in json.load(response)["error"]["message"]
            finally:
                resume.set()
            for response in streams:
                payloads = [
                    line.removeprefix("data: ")
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: ")
                ]
                assert response.status == 200
                assert payloads[-1] == "[DONE]"
                choices = [json.loads(value)["choices"][0] for value in payloads[:-1]]
                assert (
                    "".join(
                        (choice["delta"].get("content") or "") for choice in choices
                    )
                    == "okok"
                )
                assert choices[-1]["finish_reason"] == "length"
        with post(base, request_body()) as response:
            assert json.load(response)["choices"][0]["message"]["content"] == "okok"
        assert not engine.poisoned
        assert errors == []


@pytest.mark.parametrize("max_active_requests,render_workers", [(2, 4), (4, 2)])
def test_configured_active_and_render_limits_bound_inflight_prompts(
    max_active_requests, render_workers
):
    rendering = threading.Condition()
    resume = threading.Event()
    started = 0
    expected = min(max_active_requests, render_workers)

    class BlockingRecipe(Recipe):
        def render(self, messages, *, tools=(), options=None):
            nonlocal started
            with rendering:
                started += 1
                rendering.notify_all()
            assert resume.wait(timeout=5), "test did not release prompt rendering"
            return super().render(messages, tools=tools, options=options)

    with running_server(
        Runner(),
        BlockingRecipe(),
        max_active_requests=max_active_requests,
        render_workers=render_workers,
    ) as (_server, _engine, _thread, errors, base, _port):
        with ExitStack() as responses:
            try:
                streams = [
                    responses.enter_context(post(base, request_body(stream=True)))
                    for _ in range(5)
                ]
                with rendering:
                    assert rendering.wait_for(lambda: started >= expected, timeout=2)
                    assert not rendering.wait_for(
                        lambda: started > expected, timeout=0.1
                    )
            finally:
                resume.set()
            for response in streams:
                assert response.read().decode().endswith("data: [DONE]\n\n")
        assert started == 5
        assert errors == []


@pytest.mark.parametrize("max_active_requests", [None, 129])
def test_configured_active_limit_fills_engine_above_128_requests(max_active_requests):
    rendering = threading.Event()
    resume = threading.Event()
    full_batch = threading.Event()

    class BlockingRecipe(Recipe):
        def render(self, messages, *, tools=(), options=None):
            rendering.set()
            assert resume.wait(timeout=5), "test did not release prompt rendering"
            return super().render(messages, tools=tools, options=options)

    class BatchRunner(Runner):
        def forward(self, plan):
            if plan.num_sequences == 129:
                full_batch.set()
            return super().forward(plan)

    with running_server(
        BatchRunner(),
        BlockingRecipe(),
        engine_options={"max_running_requests": 129},
        **(
            {}
            if max_active_requests is None
            else {"max_active_requests": max_active_requests}
        ),
        render_workers=1,
    ) as (_server, _engine, _thread, errors, base, _port):
        with ExitStack() as responses:
            try:
                for _ in range(129):
                    responses.enter_context(
                        post(base, request_body(stream=True, max_tokens=512))
                    )
                assert rendering.wait(timeout=2)
            finally:
                resume.set()
            assert full_batch.wait(timeout=5), "HTTP admission never filled the engine"
        assert errors == []


def test_configured_admissions_limit_bounds_new_requests_between_steps():
    first_step = threading.Event()
    start_rendering = threading.Event()
    resume = threading.Event()
    rendering = threading.Condition()
    rendered = 0
    first_render = True
    admissions = []

    class CountingRecipe(Recipe):
        def render(self, messages, *, tools=(), options=None):
            nonlocal rendered, first_render
            with rendering:
                first = first_render
                first_render = False
            gate = start_rendering if first else first_step
            assert gate.wait(timeout=5), "test did not release prompt rendering"
            prompt = super().render(messages, tools=tools, options=options)
            with rendering:
                rendered += 1
                rendering.notify_all()
            return prompt

    class BlockingRunner(Runner):
        def forward(self, plan):
            admissions.append(sum(plan.context_lengths == 0))
            if not first_step.is_set():
                first_step.set()
                assert resume.wait(timeout=5), "test did not release the first step"
            return super().forward(plan)

    with running_server(BlockingRunner(), CountingRecipe(), admissions_per_step=1) as (
        _server,
        _engine,
        _thread,
        errors,
        base,
        _port,
    ):
        with ExitStack() as responses:
            try:
                streams = [
                    responses.enter_context(post(base, request_body(stream=True)))
                ]
                for _ in range(5):
                    streams.append(
                        responses.enter_context(post(base, request_body(stream=True)))
                    )
                start_rendering.set()
                assert first_step.wait(timeout=2)
                with rendering:
                    assert rendering.wait_for(lambda: rendered == 6, timeout=2)
            finally:
                start_rendering.set()
                resume.set()
            for response in streams:
                assert response.read().decode().endswith("data: [DONE]\n\n")
        assert sum(admissions) == 6
        assert max(admissions) == 1
        assert errors == []


def test_actor_failure_stops_listener_and_propagates_from_serve():
    class FailingRunner(Runner):
        def forward(self, plan):
            raise PostAcceptCudaError("runner failed after updating state")

    with running_server(FailingRunner()) as (
        server,
        engine,
        thread,
        errors,
        base,
        port,
    ):
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(base, request_body())
        assert caught.value.code == 500
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert not server.running
        assert engine.poisoned
        assert len(errors) == 1
        assert "runner failed after updating state" in str(errors[0])
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.2)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("completion_reason", ["stop", "length"])
def test_parser_finalization_deltas_precede_completion(stream, completion_reason):
    finished = []

    class FinalizingRecipe(Recipe):
        stop_token_ids = [3] if completion_reason == "stop" else []

        def create_parser(self, prompt):
            class Parser:
                def feed(self, token_ids):
                    return [resin.recipe.TextDelta("before")]

                def finish(self, reason):
                    finished.append(reason)
                    return [resin.recipe.TextDelta("-after")]

            return Parser()

    with running_server(Runner(), FinalizingRecipe()) as (
        _server,
        _engine,
        _thread,
        errors,
        base,
        _port,
    ):
        with post(base, request_body(stream=stream, max_tokens=1)) as response:
            if stream:
                payloads = [
                    line.removeprefix("data: ")
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: ")
                ]
                assert payloads[-1] == "[DONE]"
                choices = [
                    json.loads(payload)["choices"][0] for payload in payloads[:-1]
                ]
                deltas = [
                    choice["delta"]["content"]
                    for choice in choices
                    if choice["delta"].get("content")
                ]
                assert "".join(deltas) == "before-after"
                assert choices[-1]["finish_reason"] == completion_reason
                assert all(choice["finish_reason"] is None for choice in choices[:-1])
            else:
                choice = json.load(response)["choices"][0]
                assert choice["message"]["content"] == "before-after"
                assert choice["finish_reason"] == completion_reason
        assert finished == [completion_reason]
        assert errors == []


def test_parser_without_finalization_remains_supported():
    with running_server(Runner()) as (_server, engine, _thread, errors, base, _port):
        with post(base, request_body(max_tokens=1)) as response:
            choice = json.load(response)["choices"][0]
            assert choice["message"]["content"] == "ok"
            assert choice["finish_reason"] == "length"
        assert not engine.poisoned
        assert errors == []


@pytest.mark.parametrize("stream", [False, True])
def test_custom_native_tool_events_preserve_arguments_and_identity(stream):
    class ToolRecipe(Recipe):
        def create_parser(self, prompt):
            class Parser:
                def feed(self, token_ids):
                    return [
                        resin.recipe.ReasoningDelta("Looking it up."),
                        resin.recipe.ToolCallStart(0, "call-1", "lookup"),
                        resin.recipe.ToolCallArgumentsDelta(0, '{"city":"Paris"}'),
                    ]

            return Parser()

    with running_server(Runner(), ToolRecipe()) as (_, engine, _, errors, base, _):
        with post(base, request_body(stream=stream, max_tokens=1)) as response:
            if stream:
                chunks = [
                    json.loads(line[6:])
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                deltas = [
                    choice["delta"] for chunk in chunks for choice in chunk["choices"]
                ]
                assert (
                    "".join(delta.get("reasoning_content", "") for delta in deltas)
                    == "Looking it up."
                )
                calls = [
                    call for delta in deltas for call in delta.get("tool_calls", [])
                ]
                assert calls[0]["id"] == "call-1"
                assert calls[0]["function"]["name"] == "lookup"
                assert all(call["index"] == 0 for call in calls)
                assert (
                    "".join(call["function"].get("arguments", "") for call in calls)
                    == '{"city":"Paris"}'
                )
            else:
                message = json.load(response)["choices"][0]["message"]
                assert message["reasoning_content"] == "Looking it up."
                assert message["tool_calls"] == [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"city":"Paris"}'},
                    },
                ]
        assert not engine.poisoned and errors == []


def test_dictionary_parser_events_are_rejected_without_poisoning_engine():
    class InvalidRecipe(Recipe):
        invalid = True

        def create_parser(self, prompt):
            owner = self

            class Parser:
                def feed(self, token_ids):
                    if owner.invalid:
                        return [{"type": "text_delta", "delta": "legacy"}]
                    return [resin.recipe.TextDelta("healthy")]

            return Parser()

    recipe = InvalidRecipe()
    with running_server(Runner(), recipe) as (_, engine, _, errors, base, _):
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(base, request_body(max_tokens=1))
        assert caught.value.code == 500
        assert "parser" in caught.value.read().decode()
        recipe.invalid = False
        with post(base, request_body(max_tokens=1)) as response:
            assert json.load(response)["choices"][0]["message"]["content"] == "healthy"
        assert not engine.poisoned and not engine.has_work() and errors == []


@pytest.mark.parametrize("stream", [False, True])
def test_native_parser_flushes_partial_utf8_in_http_actor(stream):
    backend = _byte_tokenizer()
    markers = _token_ids(backend)
    # The final two UTF-8 bytes never arrive because generation reaches its
    # token limit. Only parser finalization can emit the replacement character.
    output = [
        markers["<|content_text|>"],
        *backend.encode("A🦀", add_special_tokens=False).ids[:3],
    ]
    tokenizer = Tokenizer(_FastTokenizer(backend))
    assert tokenizer.is_native

    native_recipe = get_native_recipe("inkling", tokenizer)

    class ByteRunner:
        def __init__(self):
            self.step = 0

        def forward(self, plan):
            assert plan.num_sequences == 1
            token = output[self.step]
            self.step += 1
            return forward_output([token])

    with running_server(ByteRunner(), native_recipe) as (
        _server,
        engine,
        _thread,
        errors,
        base,
        _port,
    ):
        with post(
            base, request_body(stream=stream, max_tokens=len(output))
        ) as response:
            if stream:
                payloads = [
                    line.removeprefix("data: ")
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: ")
                ]
                assert payloads[-1] == "[DONE]"
                choices = [
                    json.loads(payload)["choices"][0] for payload in payloads[:-1]
                ]
                assert [
                    choice["delta"]["content"]
                    for choice in choices
                    if choice["delta"].get("content")
                ] == ["A", "\ufffd"]
                assert choices[-1]["finish_reason"] == "length"
                assert all(choice["finish_reason"] is None for choice in choices[:-1])
            else:
                choice = json.load(response)["choices"][0]
                assert choice["message"]["content"] == "A\ufffd"
                assert choice["finish_reason"] == "length"
        assert not engine.poisoned
        assert errors == []


@pytest.mark.parametrize("stream", [False, True])
def test_parser_finalization_failure_does_not_poison_other_requests(stream):
    class FinalizingRecipe(Recipe):
        def render(self, messages, *, tools=(), options=None):
            return RenderedPrompt([99 if messages[0].content[0].text == "fail" else 2])

        def create_parser(self, prompt):
            fail = prompt.token_ids.tolist() == [99]

            class Parser:
                def feed(self, token_ids):
                    return []

                def finish(self, reason):
                    if fail:
                        raise ValueError("intentional finalization failure")
                    return [resin.recipe.TextDelta("success")]

            return Parser()

    with running_server(Runner(), FinalizingRecipe()) as (
        server,
        engine,
        thread,
        errors,
        base,
        _port,
    ):
        body = request_body("fail", stream=stream, max_tokens=1)
        if stream:
            with post(base, body) as response:
                payload = response.read().decode()
            assert "intentional finalization failure" in payload
            assert '"finish_reason":"length"' not in payload
        else:
            with pytest.raises(urllib.error.HTTPError) as caught:
                post(base, body)
            assert caught.value.code == 500
            assert "intentional finalization failure" in caught.value.read().decode()

        with post(base, request_body(max_tokens=1)) as response:
            choice = json.load(response)["choices"][0]
            assert choice["message"]["content"] == "success"
            assert choice["finish_reason"] == "length"
        assert server.running
        assert thread.is_alive()
        assert not engine.poisoned
        assert errors == []


def test_slow_client_is_cancelled_without_blocking_other_clients():
    runner = Runner()
    with running_server(runner) as (_server, _engine, _thread, errors, base, port):
        with socket.socket() as slow:
            slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
            slow.settimeout(2)
            slow.connect(("127.0.0.1", port))
            body = request_body("slow", stream=True, max_tokens=2000)
            slow.sendall(
                f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            assert runner.started.wait(timeout=3)
            # The slow socket is deliberately never read. Another client must
            # finish while the actor cancels the blocked stream's generation.
            with post(base, request_body()) as response:
                assert json.load(response)["choices"][0]["message"]["content"] == "okok"
            assert runner.slow_released.wait(timeout=3)
            assert runner.counts[runner.slow_id] < 2000
            assert errors == []


def test_admission_burst_yields_to_inference_before_draining_queue():
    class GatedRecipe(Recipe):
        def __init__(self):
            self.rendering = threading.Event()
            self.resume = threading.Event()
            self.admitted = 0

        def render(self, messages, *, tools=(), options=None):
            if not self.admitted:
                self.rendering.set()
                assert self.resume.wait(timeout=3)
            self.admitted += 1
            return super().render(messages, tools=tools, options=options)

    recipe = GatedRecipe()
    first_step = threading.Event()
    counts_at_step = []

    class ObservingRunner(Runner):
        def forward(self, plan):
            counts_at_step.append(plan.num_sequences)
            first_step.set()
            return super().forward(plan)

    connections = []
    with running_server(ObservingRunner(), recipe, admissions_per_step=4) as (
        _server,
        _engine,
        _thread,
        errors,
        _base,
        port,
    ):
        try:
            body = request_body(stream=True, max_tokens=1)
            wire = (
                f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            for index in range(13):
                connection = socket.create_connection(("127.0.0.1", port), timeout=2)
                connections.append(connection)
                connection.sendall(wire)
                if index == 0:
                    assert recipe.rendering.wait(timeout=2)
                # A response header acknowledges the request was queued, even
                # while its recipe has not yet been admitted by the actor.
                assert connection.recv(1024).startswith(b"HTTP/1.1 200")
            recipe.resume.set()
            assert first_step.wait(timeout=3)
            assert 1 <= counts_at_step[0] <= 4
            assert errors == []
        finally:
            recipe.resume.set()
            for connection in connections:
                connection.close()
