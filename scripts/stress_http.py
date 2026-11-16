# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28,<1"]
# ///
"""Bounded chat-completions load, protocol, disconnect, and recovery checks.

uv run scripts/stress_http.py URL --dp-size 4 --output artifacts/http-stress.json
Requests are closed-loop: at most --concurrency requests run at once. Reported
throughput includes network and serving overhead, and is not a kernel benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True, kw_only=True)
class Case:
    name: str
    body: dict[str, Any] = field(default_factory=dict)
    raw_body: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)
    expected_status: tuple[int, ...] = (200,)
    disconnect: bool = False


class Completion:
    """Accumulate the same checks for JSON responses and incremental SSE."""

    def __init__(self, *, scored: bool, top_k: int):
        self.scored, self.top_k = scored, top_k
        self.identifier = None
        self.finish_reason = None
        self.usage = None
        self.raw_scores = 0
        self.visible_bytes = 0
        self.events = 0
        self.preview = ""
        self.content = ""
        self.reasoning_content = ""

    def add(self, value: dict[str, Any], *, stream: bool) -> bool:
        if "error" in value:
            raise ValueError(f"response error: {value['error']}")
        identifier = value["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("missing or invalid response ID")
        if self.identifier is not None and identifier != self.identifier:
            raise ValueError("response ID changed midstream")
        self.identifier = identifier
        self.events += 1
        self.usage = value.get("usage") or self.usage
        meaningful = False
        for score in value.get("resin", {}).get("generation_logprobs", []):
            if score["token_index"] != self.raw_scores:
                raise ValueError("noncontiguous generation logprob token indices")
            alternatives = score.get("top_logprobs", [])
            if len(alternatives) != self.top_k:
                raise ValueError("incorrect number of top logprobs")
            for token in [score, *alternatives]:
                logprob = token["logprob"]
                if not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
                    raise ValueError("non-finite logprob")
                if logprob > 1e-4:
                    raise ValueError("positive logprob")
                if type(token["token_id"]) is not int:
                    raise ValueError("invalid token ID")
            self.raw_scores += 1
            meaningful = True
        for choice in value["choices"]:
            if choice["index"] != 0:
                raise ValueError("unexpected choice index")
            message = choice["delta" if stream else "message"]
            self.content += message.get("content") or ""
            self.reasoning_content += message.get("reasoning_content") or ""
            text = (message.get("content") or "") + (
                message.get("reasoning_content") or ""
            )
            self.visible_bytes += len(text.encode())
            self.preview = (self.preview + text)[:160]
            meaningful |= bool(text or message.get("tool_calls"))
            if reason := choice.get("finish_reason"):
                if self.finish_reason is not None:
                    raise ValueError("duplicate finish event")
                self.finish_reason = reason
        return meaningful

    def finish(self, max_tokens: int) -> dict[str, Any]:
        if self.finish_reason not in ("stop", "length", "tool_calls"):
            raise ValueError("missing or unexpected finish reason")
        if self.usage is None:
            raise ValueError("missing usage")
        prompt, completion = (
            self.usage["prompt_tokens"],
            self.usage["completion_tokens"],
        )
        if not (
            type(prompt) is int
            and type(completion) is int
            and prompt > 0
            and 0 <= completion <= max_tokens
        ):
            raise ValueError("invalid token usage")
        if self.usage["total_tokens"] != prompt + completion:
            raise ValueError("inconsistent total token usage")
        if self.scored and self.raw_scores != completion:
            raise ValueError("generation logprobs do not cover completion tokens")
        result = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "finish_reason": self.finish_reason,
            "raw_logprobs": self.raw_scores,
            "events": self.events,
            "visible_bytes": self.visible_bytes,
            "preview": self.preview,
            "content": self.content,
            "reasoning_content": self.reasoning_content,
        }
        details = self.usage.get("prompt_tokens_details") or {}
        if "cached_tokens" in details:
            cached = details["cached_tokens"]
            if type(cached) is not int or not 0 <= cached <= prompt:
                raise ValueError("invalid cached prompt token usage")
            result["cached_prompt_tokens"] = cached
        return result


async def sse_data(response: httpx.Response):
    """Handle comments, CRLF, and multiline SSE data without buffering the body."""
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        raise ValueError("SSE stream ended inside an event")


async def run_case(client: httpx.AsyncClient, case: Case, timeout: float):
    started = time.perf_counter()
    result: dict[str, Any] = {"case": case.name, "ok": False, "ttft_ms": None}
    body = case.body
    streaming = bool(body.get("stream"))
    result.update(
        stream=streaming,
        disconnect=case.disconnect,
        logprobs=bool(body.get("logprobs")),
        routed_dp_rank=case.headers.get(
            "resin-routed-dp-rank", body.get("routed_dp_rank")
        ),
        route_source="header"
        if "resin-routed-dp-rank" in case.headers
        else "body"
        if "routed_dp_rank" in body
        else "automatic",
    )
    tracker = Completion(
        scored=bool(body.get("logprobs")), top_k=body.get("top_logprobs", 0)
    )
    try:
        async with asyncio.timeout(timeout):
            async with client.stream(
                "POST",
                "chat/completions",
                follow_redirects=True,
                content=(
                    case.raw_body
                    if case.raw_body is not None
                    else json.dumps(body).encode()
                ),
                headers={"Content-Type": "application/json", **case.headers},
            ) as response:
                result["status"] = response.status_code
                if response.status_code not in case.expected_status:
                    error = (await response.aread()).decode(errors="replace")[:1000]
                    raise ValueError(f"unexpected HTTP {response.status_code}: {error}")
                if response.status_code != 200:
                    result["error_response"] = (await response.aread()).decode(
                        errors="replace"
                    )[:500]
                elif streaming:
                    if "text/event-stream" not in response.headers.get(
                        "content-type", ""
                    ):
                        raise ValueError("streaming response is not SSE")
                    done = False
                    async for data in sse_data(response):
                        if data == "[DONE]":
                            done = True
                            break
                        meaningful = tracker.add(json.loads(data), stream=True)
                        if meaningful and result["ttft_ms"] is None:
                            result["ttft_ms"] = (time.perf_counter() - started) * 1000
                        if case.disconnect and meaningful:
                            result.update(disconnected=True, events=tracker.events)
                            break
                    if not result.get("disconnected"):
                        if case.disconnect:
                            raise ValueError(
                                "stream finished before client could disconnect"
                            )
                        if not done:
                            raise ValueError("SSE stream ended without [DONE]")
                        result.update(tracker.finish(body["max_tokens"]))
                else:
                    tracker.add(json.loads(await response.aread()), stream=False)
                    result.update(tracker.finish(body["max_tokens"]))
                result["ok"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    return result


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1)]


async def run_stage(
    client,
    name,
    cases,
    concurrency,
    timeout,
    *,
    event_path: Path | None = None,
    validate_result=None,
):
    semaphore = asyncio.Semaphore(concurrency)
    finished = 0

    def record(event):
        if event_path is not None:
            with event_path.open("a") as journal:
                journal.write(
                    json.dumps(
                        {
                            "at": datetime.now(timezone.utc).isoformat(),
                            "stage": name,
                            **event,
                        }
                    )
                    + "\n"
                )

    async def limited(case):
        nonlocal finished
        async with semaphore:
            record(
                {
                    "event": "start",
                    "case": case.name,
                    "body": case.body,
                    "routed_rank_header": case.headers.get("resin-routed-dp-rank"),
                }
            )
            result = await run_case(client, case, timeout)
            if result["ok"] and validate_result is not None:
                try:
                    validate_result(result)
                except Exception as error:
                    result.update(ok=False, error=f"{type(error).__name__}: {error}")
            record({"event": "finish", **result})
            finished += 1
            if not result["ok"]:
                print(json.dumps({"stage": name, **result}), flush=True)
            elif finished % 16 == 0:
                print(
                    json.dumps(
                        {"stage": name, "finished": finished, "requests": len(cases)}
                    ),
                    flush=True,
                )
            return result

    started = time.perf_counter()
    results = await asyncio.gather(*(limited(case) for case in cases))
    elapsed = time.perf_counter() - started
    completed = [r for r in results if r["ok"] and "completion_tokens" in r]
    latencies = [r["latency_ms"] for r in completed]
    ttfts = [r["ttft_ms"] for r in completed if r["ttft_ms"] is not None]
    tokens = sum(r["completion_tokens"] for r in completed)
    summary = {
        "name": name,
        "concurrency": concurrency,
        "requests": len(results),
        "passed": sum(r["ok"] for r in results),
        "elapsed_s": elapsed,
        "completed_requests_per_s": len(completed) / elapsed,
        "completion_tokens": tokens,
        "completion_tokens_per_s": tokens / elapsed,
        "status_counts": dict(
            Counter(str(r.get("status", "transport_error")) for r in results)
        ),
        "errors": dict(Counter(r["error"] for r in results if "error" in r)),
        "latency_ms": {
            f"p{int(p * 100)}": percentile(latencies, p) for p in (0.5, 0.95, 0.99)
        },
        "ttft_ms": {
            f"p{int(p * 100)}": percentile(ttfts, p) for p in (0.5, 0.95, 0.99)
        },
    }
    print(json.dumps(summary), flush=True)
    return {**summary, "results": results}


def chat_case(model: str, index: int, dp_size: int, max_tokens: int) -> Case:
    # Four prompt lengths, distinct suffixes, and repeatable shared prefixes.
    context = (
        "The observatory tracks weather and records clear skies each evening. "
        * (1, 8, 32, 128)[(index // (4 * dp_size)) % 4]
    )
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"{context}\nIn two sentences, describe why observing the sky is interesting. Request {index}.",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "seed": index,
        "stream": (index // dp_size) % 2 == 0,
    }
    if body["stream"]:
        body["stream_options"] = {"include_usage": True}
    if (index // (2 * dp_size)) % 2 == 0:
        body.update(logprobs=True, top_logprobs=3)
    headers = {}
    if index % 3 == 1:
        body["routed_dp_rank"] = index % dp_size
    elif index % 3 == 2:
        headers["resin-routed-dp-rank"] = str(index % dp_size)
    return Case(name=f"chat-{index}", body=body, headers=headers)


def robustness_cases(model: str, dp_size: int) -> list[Case]:
    base = dict(chat_case(model, 1, dp_size, 16).body)
    mutations = [
        ("bad-temperature", {"temperature": -1}),
        ("bad-top-p", {"top_p": 0}),
        ("bad-top-logprobs", {"top_logprobs": -1}),
        ("bad-max-tokens", {"max_tokens": 0}),
        ("bad-rank", {"routed_dp_rank": dp_size}),
        ("negative-rank", {"routed_dp_rank": -1}),
    ]
    cases = [
        Case(
            name=name,
            body={**base, **change},
            # A negative unsigned count may fail JSON deserialization first.
            expected_status=(400, 422) if name == "bad-top-logprobs" else (400,),
        )
        for name, change in mutations
    ]
    cases.extend(
        [
            Case(name="bad-json", raw_body=b'{"messages":', expected_status=(400, 422)),
            Case(
                name="unknown-model",
                body={**base, "model": "resin-stress-nonexistent-model"},
                expected_status=(404,),
            ),
            Case(
                name="bad-rank-header",
                body=base,
                headers={"resin-routed-dp-rank": "bad"},
                expected_status=(400,),
            ),
            Case(
                name="header-overrides-body",
                body={**base, "routed_dp_rank": dp_size + 1},
                headers={"resin-routed-dp-rank": "0"},
            ),
        ]
    )
    for rank in range(dp_size):
        case = chat_case(model, rank * 2, dp_size, 256)
        cases.append(
            Case(
                name=f"disconnect-rank-{rank}",
                body={
                    **case.body,
                    "routed_dp_rank": rank,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                disconnect=True,
            )
        )
    return cases


async def main(args):
    base_url = args.url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    headers = (
        {"Authorization": f"Bearer {os.environ[args.api_key_env]}"}
        if args.api_key_env
        else {}
    )
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "url": base_url,
        "configuration": {
            key: value for key, value in vars(args).items() if key != "output"
        },
        "ttft_definition": "First nonempty content, reasoning, tool, or raw-token SSE event; includes network and queueing. JSON requests have no TTFT sample.",
        "stages": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    event_path = args.output.with_suffix(".requests.jsonl")
    event_path.write_text("")

    async with httpx.AsyncClient(
        base_url=base_url + "/",
        headers=headers,
        timeout=args.timeout,
        limits=httpx.Limits(
            max_connections=max(args.concurrency),
            max_keepalive_connections=max(args.concurrency),
        ),
    ) as client:
        model = args.model
        if model is None:
            response = await client.get("models")
            response.raise_for_status()
            model = response.json()["data"][0]["id"]
        report["model"] = model
        stages = [
            (
                "warmup",
                [chat_case(model, i, args.dp_size, 16) for i in range(args.dp_size)],
                1,
            )
        ]
        stages.extend(
            (
                f"load-c{c}",
                [
                    chat_case(model, i, args.dp_size, args.max_tokens)
                    for i in range(args.requests)
                ],
                c,
            )
            for c in args.concurrency
        )
        if not args.skip_robustness:
            stages.append(
                (
                    "robustness",
                    robustness_cases(model, args.dp_size),
                    min(8, max(args.concurrency)),
                )
            )
        stages.append(
            (
                "recovery",
                [
                    Case(
                        name=f"recovery-rank-{i}",
                        body={
                            **dict(chat_case(model, i, args.dp_size, 32).body),
                            "routed_dp_rank": i,
                        },
                    )
                    for i in range(args.dp_size)
                ],
                args.dp_size,
            )
        )
        for name, cases, concurrency in stages:
            report["stages"].append(
                await run_stage(
                    client,
                    name,
                    cases,
                    concurrency,
                    args.timeout,
                    event_path=event_path,
                )
            )
            report["ok"] = all(
                stage["passed"] == stage["requests"] for stage in report["stages"]
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if not report["ok"]:
                break
    return report["ok"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--model")
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument(
        "--requests", type=int, default=128, help="Requests per concurrency stage"
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--api-key-env", help="Environment variable containing an optional bearer token"
    )
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        min(
            arguments.dp_size,
            arguments.requests,
            arguments.max_tokens,
            *arguments.concurrency,
        )
        < 1
        or arguments.timeout <= 0
    ):
        parser.error(
            "DP size, requests, token budget, concurrency, and timeout must be positive"
        )
    raise SystemExit(0 if asyncio.run(main(arguments)) else 1)
