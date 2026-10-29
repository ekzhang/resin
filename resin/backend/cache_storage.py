"""Borrow exact child-block byte spans from model-owned cache arenas."""


def slab_buffers(groups, storage, layout, block_bytes):
    if hasattr(storage, "element_size"):
        import torch

        raw = storage.view(torch.uint8).reshape(-1)
        stride = storage[0].numel() * storage.element_size()
    else:
        raw = memoryview(storage).cast("B")
        stride = raw.nbytes // len(storage)
    buffers = []
    for (_, addresses), group, size in zip(
        groups, layout.groups, block_bytes, strict=True
    ):
        if size <= 0 or size * group.blocks_per_slab > stride:
            raise ValueError("cache child byte span exceeds its slab")
        for slab, slot in addresses:
            if (
                not 0 < slab <= layout.parent_capacity
                or not 0 <= slot < group.blocks_per_slab
            ):
                raise ValueError("cache storage address is outside the managed arena")
            offset = slab * stride + slot * size
            buffers.append((raw[offset : offset + size],))
    return tuple(buffers)
