"""Borrow model-declared child spans; addresses are authenticated by the Engine."""


def slab_buffers(groups, storage, sizes):
    import torch

    raw = storage.view(torch.uint8).reshape(-1)
    stride = storage[0].numel() * storage.element_size()
    return tuple(
        (raw[slab * stride + slot * size : slab * stride + (slot + 1) * size],)
        for (_, addresses), size in zip(groups, sizes, strict=True)
        for slab, slot in addresses
    )
