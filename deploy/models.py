"""Pinned model configuration shared by serving and benchmark launchers.

This module has no Modal or CUDA imports. Model-specific constructor differences
live here; launchers share checkpoint handling and runtime lifecycle code.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deploy.checkpoint_cache import (
    checkpoint_inventory,
    checkpoint_is_complete,
    publish_checkpoint,
)


@dataclass(frozen=True, kw_only=True)
class Checkpoint:
    repo_id: str
    revision: str
    directory: str
    required_files: tuple[str, ...] = ("config.json", "tokenizer.json")
    patterns: tuple[str, ...] = (
        "*.json",
        "*.jinja",
        "*.model",
        "tokenizer*",
        "model*.safetensors",
        "merges.txt",
        "vocab.json",
    )


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    name: str
    family: str
    checkpoint: Checkpoint
    model_mount: str
    model_volume: str
    gpu: str
    label: str
    app_name: str | None = None
    model_volume_version: int = 1
    kernel_mount: str | None = None
    kernel_volume: str | None = None
    kernel_volume_version: int = 1
    assistant: Checkpoint | None = None
    world_size: int = 1
    attn_dp: int = 1
    attn_tp: int = 1
    ep: int = 1
    context_length: int = 131_072
    max_batch_size: int = 128
    prefill_chunk_size: int = 4096
    draft_tokens: int = 0
    gpu_memory_utilization: float = 0.9
    expert_backend: str = "flashinfer"
    memory: int = 256 * 1024
    enable_anthropic: bool = True
    image_warmup: bool = False
    recipe: str | None = None

    @property
    def repo_id(self) -> str:
        return self.checkpoint.repo_id

    @property
    def revision(self) -> str:
        return self.checkpoint.revision

    @property
    def path(self) -> Path:
        return Path(self.model_mount) / self.checkpoint.directory / self.revision

    @property
    def assistant_path(self) -> Path | None:
        if self.assistant is None:
            return None
        return (
            Path(self.model_mount) / self.assistant.directory / self.assistant.revision
        )


MODEL_SPECS = {
    f"gpt-oss-{size}": ModelSpec(
        name=f"gpt-oss-{size}",
        family="gpt_oss",
        checkpoint=Checkpoint(
            repo_id=f"openai/gpt-oss-{size}", revision=revision, directory=size
        ),
        model_mount="/models/resin-gpt-oss",
        model_volume="resin-gpt-oss-models-v2",
        model_volume_version=2,
        kernel_mount="/cache/resin-kernels",
        kernel_volume="resin-kernel-cache-v2",
        kernel_volume_version=2,
        gpu="H200",
        world_size=world_size,
        attn_tp=world_size,
        label=f"gpt-oss-{size}",
        prefill_chunk_size=8192,
        gpu_memory_utilization=0.95,
    )
    for size, revision, world_size in (
        ("20b", "6cee5e81ee83917806bbde320786a8fb61efebee", 1),
        ("120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a", 2),
    )
}
MODEL_SPECS.update(
    {
        "gemma4-e2b": ModelSpec(
            name="gemma4-e2b",
            family="gemma4",
            checkpoint=Checkpoint(
                repo_id="google/gemma-4-E2B-it",
                revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                directory="bf16",
                required_files=(
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ),
            ),
            assistant=Checkpoint(
                repo_id="google/gemma-4-E2B-it-assistant",
                revision="2d874ef7d29f9a30599a1e4b3c1cbc9595f005df",
                directory="assistant",
                required_files=("config.json",),
                patterns=("config.json", "model.safetensors"),
            ),
            model_mount="/models/resin-gemma4-e2b",
            model_volume="resin-gemma4-e2b-models-v1",
            kernel_mount="/cache/resin-gemma4-e2b",
            kernel_volume="resin-gemma4-e2b-kernels-v1",
            gpu="A100-80GB",
            label="gemma4-e2b-bf16",
            context_length=131072,
            max_batch_size=256,
            prefill_chunk_size=8192,
            draft_tokens=3,
            gpu_memory_utilization=0.90,
            memory=128 * 1024,
        ),
        "glm-5.3-flash": ModelSpec(
            name="glm-5.3-flash",
            family="glm5_3_flash",
            checkpoint=Checkpoint(
                repo_id="nvidia/GLM-5.3-Flash-NVFP4",
                revision="da920bb0b9f4a06727223a349e55468e38352348",
                directory="nvfp4",
                required_files=(
                    "config.json",
                    "hf_quant_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "generation_config.json",
                ),
            ),
            model_mount="/models/resin-glm5-3-flash",
            model_volume="resin-glm5-3-flash-nvfp4-models-v1",
            kernel_mount="/kernel-cache/resin-glm5-3-flash",
            kernel_volume="resin-glm5-3-flash-kernels-v1",
            gpu="B200",
            label="glm-5.3-flash-nvfp4",
            world_size=4,
            attn_dp=4,
            ep=4,
            # Full decode graphs scan this sparse-index capacity even for short
            # requests. Keep a practical serving window with room for an 8K
            # prefix checkpoint and its continuation, rather than allocating
            # the checkpoint's 1M architectural limit at startup.
            context_length=16384,
            max_batch_size=8,
            prefill_chunk_size=8192,
            gpu_memory_utilization=0.65,
            image_warmup=True,
        ),
        "deepseek-v41": ModelSpec(
            name="deepseek-v41",
            family="deepseek_v41",
            checkpoint=Checkpoint(
                repo_id="deepseek-ai/DeepSeek-V4.1-Flash-1017",
                revision="1bae0e1088993b9b82de08f59de5be5a34e42170",
                directory="flash-1017",
                required_files=(
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "inference/config.json",
                ),
                patterns=("*.json", "model*.safetensors", "inference/config.json"),
            ),
            model_mount="/models/deepseek",
            model_volume="resin-deepseek-v4-models-v1",
            kernel_mount="/kernel-cache/.cache",
            kernel_volume="resin-deepseek-v4-kernels-v2",
            gpu="B300",
            world_size=4,
            attn_dp=4,
            ep=4,
            label="deepseek-v41",
            context_length=1_048_576,
            max_batch_size=32,
            prefill_chunk_size=8192,
            draft_tokens=3,
            gpu_memory_utilization=0.85,
            image_warmup=True,
            expert_backend="deepgemm",
        ),
        "gemma4": ModelSpec(
            name="gemma4",
            family="gemma4",
            checkpoint=Checkpoint(
                repo_id="nvidia/Gemma-4-31B-IT-NVFP4",
                revision="4135a98a9b728a548947683219633b25682223ac",
                directory="",
                required_files=(
                    "config.json",
                    "generation_config.json",
                    "hf_quant_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "processor_config.json",
                    "chat_template.jinja",
                ),
            ),
            assistant=Checkpoint(
                repo_id="google/gemma-4-31B-it-assistant",
                revision="627c5ec1458b9086b841a91e0512fd31fd2fbbf1",
                directory="assistant",
                required_files=("config.json",),
                patterns=("config.json", "model.safetensors"),
            ),
            model_mount="/models/resin-gemma4",
            model_volume="resin-gemma4-models-v1",
            gpu="RTX-PRO-6000",
            label="gemma4-31b",
            app_name="resin-gemma4-nvfp4",
            context_length=262_144,
            prefill_chunk_size=8192,
            draft_tokens=3,
            gpu_memory_utilization=0.95,
            memory=128 * 1024,
            image_warmup=True,
        ),
        "qwen3_5": ModelSpec(
            name="qwen3_5",
            family="qwen3_5",
            checkpoint=Checkpoint(
                repo_id="RadixArk/Qwen3.8-27B-NVFP4",
                revision="554ebba9b5f1b79dc11246341960360e6ef05ef4",
                directory="",
                required_files=(
                    "config.json",
                    "generation_config.json",
                    "hf_quant_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "preprocessor_config.json",
                    "video_preprocessor_config.json",
                ),
            ),
            model_mount="/models/resin-qwen3-5",
            model_volume="resin-qwen3-5-nvfp4-models-v1",
            kernel_mount="/kernel-cache/resin-qwen3-5",
            kernel_volume="resin-qwen3-5-kernels-v2",
            gpu="B200",
            label="qwen3-8-27b",
            context_length=32_768,
            max_batch_size=32,
            draft_tokens=2,
            gpu_memory_utilization=0.70,
            memory=128 * 1024,
            image_warmup=True,
        ),
        "ornith-1.0-35b": ModelSpec(
            name="ornith-1.0-35b",
            family="qwen3_5_moe",
            checkpoint=Checkpoint(
                repo_id="ornith-ai/Ornith-1.0-35B",
                revision="5df2ed3f675c7beaa490328cc70bb573b65fb660",
                directory="",
                required_files=(
                    "config.json",
                    "chat_template.jinja",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "preprocessor_config.json",
                ),
            ),
            # Apple Silicon (MLX) runner: BF16 weights quantized to 8-bit at load.
            model_mount="/models/resin-ornith",
            model_volume="resin-ornith-models-v1",
            gpu="M4-Max",
            label="ornith-1.0-35b-mlx",
            context_length=32_768,
            max_batch_size=32,
            # Unified memory is shared with the OS and other applications.
            gpu_memory_utilization=0.70,
            recipe="ornith",
        ),
        "inkling": ModelSpec(
            name="inkling",
            family="inkling",
            checkpoint=Checkpoint(
                repo_id="thinkingmachines/Inkling-Small-NVFP4",
                revision="b6a99534467840620d411e4cd4ad5819b2610d9c",
                directory="target",
                required_files=(
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ),
            ),
            assistant=Checkpoint(
                repo_id="RadixArk/Inkling-Small-DSpark",
                revision="736501c3901cfc6bbb53ba382781eb0e5d9ad66a",
                directory="draft",
                required_files=("config.json",),
                patterns=("config.json", "model.safetensors"),
            ),
            model_mount="/models/resin-inkling",
            model_volume="resin-inkling-models-v1",
            kernel_mount="/kernel-cache/resin-inkling",
            kernel_volume="resin-inkling-kernels-v2",
            gpu="B300",
            label="inkling-small",
            draft_tokens=5,
        ),
    }
)


def get_model(model: str | ModelSpec) -> ModelSpec:
    if isinstance(model, ModelSpec):
        return model
    try:
        return MODEL_SPECS[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; choose from {', '.join(MODEL_SPECS)}"
        ) from None


def validate_gpu(model: str | ModelSpec, gpu: str | None = None) -> str:
    """Keep hardware overrides within the preset's compiled-kernel architecture."""
    spec = get_model(model)
    selected = spec.gpu if gpu is None else gpu
    supported = ("H100", "H200") if spec.family == "gpt_oss" else (spec.gpu,)
    if selected not in supported:
        raise ValueError(
            f"{spec.name} uses a GPU-specific kernel configuration; "
            f"choose {', '.join(supported)} (got {selected!r})"
        )
    return selected


def checkpoint_parts(spec: ModelSpec):
    yield spec.checkpoint, spec.path
    if spec.assistant is not None:
        yield spec.assistant, spec.assistant_path


def download_checkpoint(model: str | ModelSpec, *, metadata_only: bool = False) -> dict:
    """Download missing files and publish identical inventories for every launcher.

    Complete existing volumes are adopted without downloading their weight bodies.
    The caller owns Modal volume reload/commit, so this also works on local paths.
    """
    spec = get_model(model)
    started, inventory, cache_hit = time.perf_counter(), {}, True
    for checkpoint, path in checkpoint_parts(spec):
        try:
            checkpoint_inventory(path, required_files=checkpoint.required_files)
        except RuntimeError:
            if metadata_only:
                raise
            from huggingface_hub import snapshot_download

            cache_hit = False
            path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                checkpoint.repo_id,
                revision=checkpoint.revision,
                local_dir=path,
                allow_patterns=list(checkpoint.patterns),
            )
        files = publish_checkpoint(
            path,
            checkpoint.repo_id,
            checkpoint.revision,
            required_files=checkpoint.required_files,
        )
        inventory.update(
            {
                str(path.relative_to(spec.model_mount) / name): size
                for name, size in files.items()
            }
        )
    return dict(
        model=spec.name,
        repo_id=spec.repo_id,
        revision=spec.revision,
        path=str(spec.path),
        cache_hit=cache_hit,
        metadata_only=metadata_only,
        files=len(inventory),
        bytes=sum(inventory.values()),
        elapsed_s=time.perf_counter() - started,
    )


def require_checkpoint(
    model: str | ModelSpec,
    *,
    checkpoint: str | Path | None = None,
    assistant_checkpoint: str | Path | None = None,
    draft_tokens: int | None = None,
) -> Path:
    """Require a published volume, or validate explicitly supplied local weights."""
    spec = get_model(model)
    for part, path in checkpoint_parts(spec):
        if part is spec.assistant and draft_tokens == 0:
            continue
        override = assistant_checkpoint if part is spec.assistant else checkpoint
        if override is not None:
            checkpoint_inventory(Path(override), required_files=part.required_files)
        elif not checkpoint_is_complete(
            path, part.repo_id, part.revision, required_files=part.required_files
        ):
            raise FileNotFoundError(
                f"incomplete {spec.name} checkpoint: run the download command (or --publish-only for existing files)"
            )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return spec.path if checkpoint is None else Path(checkpoint)


def image_environment(model: str | ModelSpec) -> dict[str, str]:
    spec = get_model(model)
    if spec.family == "gpt_oss":
        from deploy.jit_cache import KERNEL_CACHE_ENV

        return {
            **KERNEL_CACHE_ENV,
            "MAX_JOBS": "12",
            "CMAKE_BUILD_PARALLEL_LEVEL": "12",
        }
    if spec.family == "gemma4":
        if spec.name == "gemma4-e2b":
            return {
                "TORCH_CUDA_ARCH_LIST": "8.0",
                "FLASHINFER_CUDA_ARCH_LIST": "8.0",
                "TRITON_CACHE_DIR": f"{spec.kernel_mount}/triton",
                "FLASHINFER_WORKSPACE_BASE": f"{spec.kernel_mount}/flashinfer",
            }
        return {"TORCH_CUDA_ARCH_LIST": "12.0", "FLASHINFER_CUDA_ARCH_LIST": "12.0"}
    root = spec.kernel_mount
    env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TRITON_CACHE_DIR": f"{root}/triton",
        "CUDA_TILE_CACHE_DIR": f"{root}/cutile-python",
    }
    if spec.family == "deepseek_v41":
        return {
            **env,
            "DG_JIT_CACHE_DIR": f"{root}/deep_gemm",
            "TILELANG_CACHE_DIR": f"{root}/tilelang",
            "FLASHINFER_WORKSPACE_BASE": "/tmp/resin-flashinfer-build",
        }
    arch = "10.3a" if spec.family == "inkling" else "10.0a"
    env.update(
        {
            "TORCH_CUDA_ARCH_LIST": arch,
            "FLASHINFER_CUDA_ARCH_LIST": arch,
            "FLASHINFER_WORKSPACE_BASE": f"{root}/flashinfer",
            "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
            "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": f"{root}/flash-attention-4",
            "FLASH_ATTENTION_ARCH": "sm_" + arch.replace(".", ""),
            "CUTE_DSL_CACHE_DIR": f"{root}/cute-dsl",
            "QUACK_CACHE_DIR": f"{root}/quack",
        }
    )
    if spec.family == "inkling":
        env.update(CUTE_DSL_ARCH="sm_103a", QUACK_CACHE_ENABLED="1")
    else:
        env.update(
            CUDA_CACHE_PATH=f"{root}/cuda",
            CUDA_TILE_CACHE_DIR=f"{root}/cuda-tile",
            TORCHINDUCTOR_CACHE_DIR=f"{root}/torchinductor",
        )
    return env


def prepare_runtime(model: str | ModelSpec) -> None:
    spec = get_model(model)
    if spec.family == "gpt_oss":
        from deploy.jit_cache import (
            ensure_kernel_cache_dirs,
            install_flashinfer_hash_cache,
        )

        ensure_kernel_cache_dirs()
        install_flashinfer_hash_cache()
    elif spec.kernel_mount:
        for key, value in image_environment(spec).items():
            if value.startswith(spec.kernel_mount + "/"):
                Path(value).mkdir(parents=True, exist_ok=True)


def create_runner(
    model: str | ModelSpec,
    checkpoint: str | Path | None = None,
    *,
    device,
    groups=None,
    context_length: int,
    max_batch_size: int,
    prefill_chunk_size: int,
    draft_tokens: int,
    prefill_graph: str = "breakable",
    decode_graph: str = "full",
    gpu_memory_utilization: float | None = None,
    decode_batch_sizes=None,
    expert_backend: str | None = None,
    profile_graph_phases: bool = False,
    fast_allreduce: bool = False,
    assistant_checkpoint: str | Path | None = None,
):
    """Construct one runner with model-specific cache geometry and draft support."""
    spec = get_model(model)
    checkpoint = spec.path if checkpoint is None else Path(checkpoint)
    assistant = (
        spec.assistant_path
        if assistant_checkpoint is None
        else Path(assistant_checkpoint)
    )
    memory = (
        spec.gpu_memory_utilization
        if gpu_memory_utilization is None
        else gpu_memory_utilization
    )
    if spec.family == "qwen3_5_moe":
        from resin_models_mlx.qwen3_5_moe import Qwen3_5MoeMlxRunner
        from resin_models_mlx.qwen3_5_moe.cache import PREFILL_CHUNK_SIZE

        if draft_tokens:
            raise ValueError(
                "Ornith checkpoints ship no MTP weights; use draft_tokens=0"
            )
        if groups is not None:
            raise ValueError("the MLX runner executes on one Apple Silicon device")
        if prefill_chunk_size != PREFILL_CHUNK_SIZE:
            raise ValueError(
                f"the MLX Qwen3.5-MoE runner uses {PREFILL_CHUNK_SIZE}-token chunks"
            )
        return Qwen3_5MoeMlxRunner(checkpoint, memory_utilization=memory)
    common: dict[str, Any] = dict(
        device=device, cuda_graph_prefill=prefill_graph, cuda_graph_decode=decode_graph
    )
    page_tokens = {"gpt_oss": 128, "gemma4": 64}.get(spec.family, 1)
    width = (context_length + page_tokens - 1) // page_tokens
    if spec.family == "inkling":
        width += draft_tokens
    if decode_batch_sizes is None:
        from resin.backend.cuda.graphs import default_decode_graph_shapes

        shapes = default_decode_graph_shapes(
            width,
            speculative=bool(draft_tokens) and spec.family != "glm5_3_flash",
            max_batch_size=max_batch_size,
        )
    else:
        shapes = tuple((batch, width) for batch in sorted(set(decode_batch_sizes)))
    common["cuda_graph_decode_shapes"] = shapes if decode_graph != "disabled" else ()
    if spec.family == "gpt_oss":
        from resin_models_cuda.gpt_oss.cuda import GptOssCudaRunner
        from resin_models_cuda.gpt_oss.flashinfer import FlashInferSM90MXFP4Kernel

        if draft_tokens:
            raise ValueError("GPT-OSS has no draft model; use draft_tokens=0")

        class Runner(GptOssCudaRunner):
            pass

        Runner.prefill_chunk_size = prefill_chunk_size
        return Runner(
            checkpoint,
            parallel_groups=groups,
            kernel_moe=FlashInferSM90MXFP4Kernel(),
            fast_allreduce=fast_allreduce,
            gpu_memory_utilization=memory,
            **common,
        )
    if spec.family == "deepseek_v41":
        from resin_models_cuda.deepseek_v41.cuda import DeepSeekV41CudaRunner

        return DeepSeekV41CudaRunner(
            checkpoint,
            parallel_groups=groups,
            max_seq_len=context_length,
            max_batch_size=max_batch_size,
            speculative_draft_tokens=draft_tokens,
            prefill_chunk_size=prefill_chunk_size,
            expert_backend=expert_backend or spec.expert_backend,
            profile_graph_phases=profile_graph_phases,
            fast_allreduce=fast_allreduce,
            gpu_memory_utilization=memory,
            **common,
        )
    if spec.family == "glm5_3_flash":
        from resin_models_cuda.glm5_3_flash.cuda import Glm53FlashCudaRunner

        return Glm53FlashCudaRunner(
            checkpoint,
            parallel_groups=groups,
            max_seq_len=context_length,
            max_batch_size=max_batch_size,
            prefill_chunk_size=prefill_chunk_size,
            speculative_draft_tokens=draft_tokens,
            gpu_memory_utilization=memory,
            profile_graph_phases=profile_graph_phases,
            **common,
        )
    if spec.family == "gemma4":
        if spec.name == "gemma4-e2b":
            from resin_models_cuda.gemma4.edge_cuda import Gemma4EdgeCudaRunner

            return Gemma4EdgeCudaRunner(
                checkpoint,
                parallel_groups=groups,
                assistant_model=assistant if draft_tokens else None,
                speculative_draft_tokens=draft_tokens,
                prefill_chunk_size=prefill_chunk_size,
                static_memory_fraction=memory,
                **common,
            )
        from resin_models_cuda.gemma4.cuda import Gemma4CudaRunner

        return Gemma4CudaRunner(
            checkpoint,
            parallel_groups=groups,
            assistant_model=assistant if draft_tokens else None,
            speculative_draft_tokens=draft_tokens,
            prefill_chunk_size=prefill_chunk_size,
            static_memory_fraction=memory,
            **common,
        )
    if spec.family == "qwen3_5":
        from resin_models_cuda.qwen3_5.cuda import Qwen3_5CudaRunner

        return Qwen3_5CudaRunner(
            checkpoint,
            parallel_groups=groups,
            max_seq_len=context_length,
            max_batch_size=max_batch_size,
            speculative_draft_tokens=draft_tokens,
            gpu_memory_utilization=memory,
            kv_cache_dtype="fp8",
            **common,
        )
    if spec.family == "inkling":
        from resin_models_cuda.inkling.cuda import InklingCudaRunner

        if draft_tokens != 5:
            raise ValueError("Inkling requires its fixed five-token DSpark draft")
        if groups is not None and groups.topology.world_size != 1:
            raise ValueError("Inkling currently supports a single GPU")
        if assistant is None:
            raise ValueError("Inkling requires a draft checkpoint")
        return InklingCudaRunner(
            checkpoint,
            assistant,
            max_seq_len=context_length,
            max_batch_size=max_batch_size,
            prefill_chunk_size=prefill_chunk_size,
            gpu_memory_utilization=memory,
            **common,
        )
    raise ValueError(f"unsupported runner family: {spec.family}")


def create_recipe(
    model: str | ModelSpec, checkpoint: str | Path | None = None, *, runner=None
):
    from resin.recipe import get_native_recipe

    spec = get_model(model)
    path = spec.path if checkpoint is None else Path(checkpoint)
    if spec.family in {"deepseek_v41", "glm5_3_flash"}:
        from tokenizers import Tokenizer

        return get_native_recipe(
            spec.family, Tokenizer.from_file(str(path / "tokenizer.json"))
        )
    if spec.family == "gpt_oss":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        return get_native_recipe(spec.family, tokenizer, reasoning_effort="medium")
    if spec.name == "gemma4-e2b":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        options = (
            {"stop_token_ids": runner.config.eos_token_ids}
            if runner is not None
            else {}
        )
        return get_native_recipe(
            "gemma4", tokenizer, empty_thought_prefix=False, **options
        )
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    options = (
        {"stop_token_ids": runner.config.eos_token_ids}
        if spec.family == "gemma4" and runner is not None
        else {}
    )
    name = spec.recipe or ("qwen3_8" if spec.family == "qwen3_5" else spec.family)
    return get_native_recipe(name, processor, **options)
