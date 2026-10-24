"""Published native metrics remain readable independently of inference progress."""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request

import test_http_lifecycle as lifecycle
from output_helpers import forward_output

import resin


def scrape(base):
    with urllib.request.urlopen(f"{base}/metrics", timeout=1) as response:
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("text/plain; version=0.0.4")
        return response.read().decode()


def sample(text, name):
    lines = [line for line in text.splitlines() if line.startswith(f"resin:{name}{{")]
    assert len(lines) == 1, f"expected one {name} series, got {lines}"
    return float(lines[0].rsplit(" ", 1)[1])


def wait_for_sample(base, name, expected):
    deadline = time.monotonic() + 5
    while True:
        text = scrape(base)
        if sample(text, name) == expected:
            return text
        assert time.monotonic() < deadline, f"{name} did not reach {expected}"
        time.sleep(0.05)


def test_http_metrics_include_rendering_and_ignore_parser_stream_buffering():
    class SlowRecipe(lifecycle.Codec):
        def render(self, messages, *, tools=(), options=None):
            time.sleep(0.12)
            return super().render(messages, tools=tools, options=options)

        def create_parser(self, prompt):
            class Parser:
                def feed(self, token_ids):
                    return []

                def finish(self, reason):
                    return [resin.recipe.TextDelta("delayed parser output")]

            return Parser()

    class SlowRunner:
        calls = 0

        def forward(self, plan):
            self.calls += 1
            if self.calls > 1:
                time.sleep(0.03)
            return forward_output([3] * plan.num_sequences)

    with lifecycle.running_server(SlowRunner(), SlowRecipe(), stream_interval=4) as (
        _,
        _,
        _,
        errors,
        base,
        _,
    ):
        with lifecycle.post(
            base, lifecycle.request_body(stream=True, max_tokens=5)
        ) as response:
            body = response.read().decode()
        assert "delayed parser output" in body
        assert '"type":"first_token"' not in body
        text = wait_for_sample(base, "num_requests_total", 1)
        assert sample(text, "http_request_latency_seconds_count") == 1
        assert sample(text, "http_time_to_first_token_seconds_count") == 1
        first_token = sample(text, "http_time_to_first_token_seconds_sum")
        latency = sample(text, "http_request_latency_seconds_sum")
        assert first_token >= 0.1  # Includes recipe rendering before submit.
        assert latency - first_token >= 0.08  # Excludes parser/coalescing delay.
        assert errors == []


def test_scrape_reads_published_state_while_runner_cannot_make_progress():
    entered, resume = threading.Event(), threading.Event()

    class BlockingRunner:
        def forward(self, plan):
            entered.set()
            assert resume.wait(timeout=5), "test did not release inference"
            return forward_output([3] * plan.num_sequences)

    with lifecycle.running_server(BlockingRunner()) as (_, _, _, errors, base, _):
        assert sample(scrape(base), "num_requests_total") == 0
        response = lifecycle.post(
            base, lifecycle.request_body(stream=True, max_tokens=1)
        )
        try:
            assert entered.wait(timeout=2)
            # The actor holds its engine borrow inside forward. A scrape must
            # not access it or wait for completion to collect fresh metrics.
            published = scrape(base)
            assert scrape(base) == published
            assert sample(published, "num_requests_total") == 0
            assert not resume.is_set()
        finally:
            resume.set()
            response.read()
            response.close()
        wait_for_sample(base, "num_requests_total", 1)
        assert errors == []
