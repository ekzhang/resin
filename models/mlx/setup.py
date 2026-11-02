"""Build the Apple Silicon extension with the matching MLX toolchain."""

import os
import sys
from importlib.metadata import distribution, version

from setuptools import setup

extensions = []
commands = {}
if sys.platform == "darwin":
    from mlx import extension

    class NativeBuildExtension(extension.BuildExtension):
        def build_extension(self, ext):
            # Ambient CPATH can contain another MLX installation. Imported
            # targets are otherwise -isystem, which searches after CPATH.
            previous = os.environ.get("CMAKE_ARGS")
            os.environ["CMAKE_ARGS"] = (
                f"{previous or ''} -DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON"
            )
            try:
                super().build_extension(ext)
            finally:
                if previous is None:
                    os.environ.pop("CMAKE_ARGS", None)
                else:
                    os.environ["CMAKE_ARGS"] = previous

    if version("mlx") != "0.32.4":
        raise RuntimeError("The native cache extension requires MLX 0.32.4")
    extensions = [
        extension.MetalExtension(
            "resin_models_mlx._native_cache",
            sources=[
                "resin_models_mlx/native/cache.cpp",
                "resin_models_mlx/native/cache.metal",
                "resin_models_mlx/native/attention.cpp",
                "resin_models_mlx/native/attention.metal",
            ],
            include_dirs=[str(distribution("mlx").locate_file("mlx/include"))],
        )
    ]
    # Keep the checked-in interface: automatic stub generation omits private
    # attention entrypoints used by the Python adapter.
    commands = {"build_ext": NativeBuildExtension.with_options(generate_stubs=False)}

setup(ext_modules=extensions, cmdclass=commands, zip_safe=False)
