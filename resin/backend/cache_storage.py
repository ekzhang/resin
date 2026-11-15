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


def layered_cache_buffers(arena, groups, *, page_fields, page_layers, state_fields):
    """Export layered pages and complete state fields from one shared slab arena."""
    pages = tuple((name, getattr(arena, name)) for name in page_fields)
    buffers = []
    slab_units = arena.geometry.slab_units
    for group, (_, addresses) in enumerate(groups):
        layers = page_layers[group]
        for slab, slot in addresses:
            if layers is None:
                tensors = {
                    name: getattr(arena, name)[slab, slot] for name in state_fields
                }
            else:
                base = slab * slab_units + slot * layers
                tensors = {
                    f"{name}/{layer}": values[base + layer]
                    for name, values in pages
                    for layer in range(layers)
                }
            buffers.append(byte_regions(tensors))
    return tuple(buffers)
