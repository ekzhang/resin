"""Validate owned CUDA graph timing generations without model checkpoints."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

from deploy.modal_image import build_modal_image

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = "/opt/resin/validation/test_cuda_metrics.py"
image = (
    build_modal_image(ROOT)
    .uv_pip_install("pytest>=8,<9")
    .add_local_file(ROOT / "tests/test_cuda_metrics.py", TEST_PATH, copy=True)
)
app = modal.App("resin-metrics", image=image)


@app.function(gpu="H100", cpu=4, memory=8 * 1024, timeout=10 * 60)
def validate() -> dict:
    import torch

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-c", "/dev/null", "-q", TEST_PATH],
        cwd="/tmp",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=9 * 60,
        check=False,
    )
    print(result.stdout, flush=True)
    if result.returncode or "skipped" in result.stdout:
        raise RuntimeError("GPU metrics validation did not pass every case")
    return {
        "gpu": torch.cuda.get_device_name(),
        "tests": result.stdout.strip(),
    }
