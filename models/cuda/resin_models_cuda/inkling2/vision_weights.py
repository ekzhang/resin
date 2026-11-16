"""Load the native BF16 vision tensors from the Inkling2 checkpoint index."""

from pathlib import Path

import torch

from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin_models_cuda.inkling2.vision import Inkling2VisionEncoder


@torch.no_grad()
def load_vision_weights(model_path: str | Path, model: Inkling2VisionEncoder):
    files = indexed_tensor_files(
        Path(model_path).expanduser().resolve(), require_index=True
    )
    tensors = {
        "model.visual." + name: value for name, value in model.named_parameters()
    }
    expected = {name for name in files if name.startswith("model.visual.")}
    if tensors.keys() != expected:
        raise ValueError(
            "Inkling2 vision checkpoint tensors do not match its configuration"
        )
    loaded_bytes = 0
    with SafetensorReader() as reader:
        for name in sorted(tensors, key=lambda key: (str(files[key]), key)):
            source, destination = reader.read(files[name], name), tensors[name]
            if source.shape != destination.shape or source.dtype != torch.bfloat16:
                raise ValueError(
                    f"{name}: invalid Inkling2 vision tensor shape or dtype"
                )
            destination.copy_(source)
            loaded_bytes += source.numel() * source.element_size()
    return {"tensors": len(tensors), "bytes": loaded_bytes}
