"""Zero-copy helpers for model-declared checkpoint byte ranges."""


def live_rows(prefix_length, logical_block, block_tokens, read_window=None):
    first = 0 if read_window is None else max(0, prefix_length - read_window)
    base = logical_block * block_tokens
    return slice(
        max(0, min(block_tokens, first - base)),
        max(0, min(block_tokens, prefix_length - base)),
    )


def byte_regions(tensors):
    import torch

    return {
        name: tensor.view(torch.uint8).view(-1)
        for name, tensor in tensors.items()
        if tensor.numel()
    }
