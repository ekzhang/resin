from __future__ import annotations

import subprocess
import sys


class TestImport:
    def test_import_resin_does_not_import_accelerator_stacks(self) -> None:
        code = (
            "import sys; sys.path[:0] = ['models/cuda', 'models/torch']; "
            "import resin; import resin.backend; "
            "from resin import Parallelism; Parallelism(world_size=4, attn_dp=4); "
            "import resin.backend.cuda; import resin_models_cuda; "
            "import resin_models_torch; "
            "print(*(int(name in sys.modules) for name in "
            "('torch', 'flashinfer', 'flash_attn')))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        )
        assert result.stdout.strip() == "0 0 0"
