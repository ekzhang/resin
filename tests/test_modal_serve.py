"""The parameterized deployment preserves driver, follower and media lifecycles."""

from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from deploy import modal_serve as deployment
from deploy.models import get_model
from resin import Engine
from resin.backend import distributed
from resin.backend.topology import ParallelGroups
from resin.recipe import Media
from resin.server import HttpServer


@pytest.mark.parametrize(
    "name,rank,failure",
    [
        ("gpt-oss-20b", 0, None),
        ("gpt-oss-120b", 0, None),
        ("gpt-oss-120b", 1, None),
        ("deepseek-v41", 0, None),
        ("deepseek-v41", 3, None),
        ("glm53-flash", 3, None),
        ("gemma4-31b", 0, None),
        ("qwen38-27b", 0, None),
        ("inkling2-small", 0, None),
        ("gpt-oss-20b", 0, "warmup"),
        ("deepseek-v41", 0, "serve"),
    ],
)
def test_rank_warms_then_serves_and_closes_on_failure(monkeypatch, name, rank, failure):
    spec = get_model(name)
    calls = []
    groups = SimpleNamespace(is_driver=rank == 0 or spec.attn_dp > 1)
    runner = SimpleNamespace(
        device=f"cuda:{rank}",
        capture_cuda_graphs=lambda: calls.append("capture"),
        runner_config=SimpleNamespace(
            max_batch_tokens=spec.max_batch_size * spec.prefill_chunk_size
        ),
    )
    media = (Media((1, 1), image=np.zeros((1, 1, 3), dtype=np.float32)),)
    recipe = SimpleNamespace(
        stop_token_ids=[7, 8],
        render=lambda messages: SimpleNamespace(
            token_ids=[4, 5], media=media if len(messages[0].content) > 1 else None
        ),
    )

    def initialize(backend, **options):
        assert backend == "nccl"
        assert options["rank"] == rank and options["world_size"] == spec.world_size
        assert options["device_id"] == torch.device("cuda", rank)
        calls.append("initialize")

    def make_groups(topology):
        assert (
            topology.world_size,
            topology.attn_dp,
            topology.attn_tp,
            topology.ep,
        ) == (spec.world_size, spec.attn_dp, spec.attn_tp, spec.ep)
        return groups

    def make_runner(model, **options):
        assert model is spec
        assert options["draft_tokens"] == spec.draft_tokens
        assert options["context_length"] == spec.context_length + (
            spec.family == "gpt_oss"
        )
        assert options["groups"] is (groups if spec.world_size > 1 else None)
        return runner

    class Worker:
        def __init__(self, value, rank_groups):
            assert value is runner and rank_groups is groups

        def serve(self):
            calls.append("follower")

    class RequestEngine:
        closed = False

        def __init__(self, owner, **options):
            signature(Engine).bind(owner, **options)
            assert options["enable_overlap"] and options["prefix_cache"]
            assert options["context_length"] == spec.context_length
            self.remaining = 2
            calls.append("engine")

        def submit(self, **options):
            signature(Engine.submit).bind(self, **options)
            assert options["sampling_params"].stop_token_ids == (7, 8)
            assert options["sampling_params"].temperature == 0
            assert options["token_ids"] == [4, 5]
            assert rank == 0 or (spec.image_warmup and options["media"] is media)
            if options["media"] is not None:
                assert options["media"] is media and spec.image_warmup
                calls.append("media")
            else:
                calls.append("text")
            self.remaining = 2
            return SimpleNamespace(take_result=lambda: calls.append("warmup_done"))

        def has_work(self):
            return self.remaining > 0

        def step(self):
            if failure == "warmup":
                raise RuntimeError("injected warmup failure")
            self.remaining -= 1
            calls.append("step")

        def close(self):
            assert not self.closed
            self.closed = True
            calls.append("close")

    class Server:
        def __init__(self, engine, **options):
            signature(HttpServer.__init__).bind(self, engine, **options)
            assert options["recipe"] is recipe
            assert options["model"] == spec.repo_id
            assert options["enable_anthropic"] == spec.enable_anthropic
            assert engine.remaining == 0
            calls.append("http")

        def serve(self, *, host, port):
            assert (host, port) == ("0.0.0.0", 8000)
            calls.append("serve")
            if failure == "serve":
                raise RuntimeError("injected serve failure")

    monkeypatch.setattr(torch.cuda, "set_device", lambda value: None)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: calls.append("synchronize")
    )
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(dist, "init_process_group", initialize)
    monkeypatch.setattr(ParallelGroups, "create", make_groups)
    monkeypatch.setattr(deployment, "prepare_runtime", lambda spec: None)
    monkeypatch.setattr(deployment, "create_runner", make_runner)
    monkeypatch.setattr(deployment, "create_recipe", lambda spec, runner: recipe)
    monkeypatch.setattr(distributed, "DistributedWorker", Worker)
    monkeypatch.setattr(distributed, "DistributedEngine", RequestEngine)
    monkeypatch.setattr("resin.Engine", RequestEngine)
    monkeypatch.setattr("resin.server.HttpServer", Server)
    monkeypatch.setattr(
        deployment,
        "kernel_volume",
        SimpleNamespace(commit=lambda: calls.append("commit")),
    )
    monkeypatch.setattr(
        deployment.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )
    if failure:
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            deployment._serve_rank(rank, 12345, name)
    elif spec.world_size > 1:
        with pytest.raises(SystemExit) as exit:
            deployment._serve_rank(rank, 12345, name)
        assert exit.value.code == 0
    else:
        deployment._serve_rank(rank, 12345, name)
    if not groups.is_driver:
        assert "follower" in calls and "engine" not in calls
    else:
        assert calls.count("close") == 1
        assert calls.index("capture") < calls.index("engine")
        assert ("http" in calls) == (failure != "warmup")
        if rank == 0 and failure != "warmup":
            assert calls.count("warmup_done") == 1 + spec.image_warmup
            assert calls.count("media") == spec.image_warmup
            assert calls.index("warmup_done") < calls.index("http")
        if rank != 0:
            assert "text" not in calls
            assert calls.count("media") == spec.image_warmup
            assert calls.count("warmup_done") == spec.image_warmup


def test_selected_model_owns_download_and_serving_volume(monkeypatch):
    calls = []
    monkeypatch.setattr(
        deployment,
        "model_volume",
        SimpleNamespace(
            reload=lambda: calls.append("reload"), commit=lambda: calls.append("commit")
        ),
    )
    monkeypatch.setattr(
        deployment,
        "download_checkpoint",
        lambda model, metadata_only: {
            "model": model.name,
            "metadata_only": metadata_only,
        },
    )
    with pytest.warns(UserWarning, match="executing locally"):
        report = deployment.download.local(True)
    assert report == {"model": deployment.MODEL.name, "metadata_only": True}
    assert calls == ["reload", "commit"]
