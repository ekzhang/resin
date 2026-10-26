"""Borrow exact model-owned cache bytes without changing physical placement."""

from __future__ import annotations

import json

from resin.cache_storage import CacheBuffer, _torch_tensor


def cache_signature(kind, layout, block_bytes, geometry=()):
    """Identify a byte ABI without device addresses or arena capacity."""
    return json.dumps(
        {
            "format": kind,
            "version": 1,
            "block_bytes": block_bytes,
            "geometry": geometry,
            "groups": [
                (
                    group.block_tokens,
                    group.blocks_per_slab,
                    int(group.family),
                    int(group.retention),
                    group.window_tokens,
                    group.read_window_tokens,
                )
                for group in layout.groups
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()


def _addresses(transfer, layout):
    previous = -1
    for group in transfer.groups:
        index = group.group_index
        if not previous < index < len(layout.groups):
            raise ValueError("cache transfer groups must be ordered and in range")
        previous = index
        if group.first_block < 0 or len(group.slab_ids) != len(group.slotnums):
            raise ValueError("invalid cache transfer block table")
        packing = layout.groups[index].blocks_per_slab
        for slab, slot in zip(group.slab_ids, group.slotnums, strict=True):
            if not 0 < slab <= layout.parent_capacity or not 0 <= slot < packing:
                raise ValueError("cache transfer address is outside the managed arena")
            yield index, int(slab), int(slot)


def slab_buffers(transfer, storage, layout, block_bytes):
    """One borrowed contiguous child span per group-major transfer entry."""
    if len(block_bytes) != len(layout.groups):
        raise ValueError("cache byte geometry must cover every group")
    torch = _torch_tensor(storage)
    if torch is not None:
        if not storage.is_contiguous():
            raise ValueError("cache slab storage must be contiguous")
        slab_bytes = storage[0].numel() * storage.element_size()
        raw = storage.view(torch.uint8).reshape(-1)
        device = str(storage.device)
    else:
        raw = memoryview(storage).cast("B")
        slab_bytes = raw.nbytes // len(storage)
        device = "cpu"
    buffers = []
    for group, slab, slot in _addresses(transfer, layout):
        size = block_bytes[group]
        if size <= 0 or size * layout.groups[group].blocks_per_slab > slab_bytes:
            raise ValueError("cache child byte geometry exceeds its slab")
        offset = slab * slab_bytes + slot * size
        buffers.append(CacheBuffer((raw[offset : offset + size],), size, device))
    return tuple(buffers)


def paged_kv_buffers(transfer, storage, layout):
    """K/V-by-layer storage is scatter/gather across its physical page axis."""
    torch = _torch_tensor(storage)
    if torch is None:
        raise TypeError("paged KV cache storage requires tensor-owned regions")
    buffers = []
    for group, slab, slot in _addresses(transfer, layout):
        if slot or layout.groups[group].blocks_per_slab != 1:
            raise ValueError("paged KV arena does not support packed child slots")
        regions = tuple(
            storage[kv, layer, slab].view(torch.uint8).reshape(-1)
            for kv in range(storage.shape[0])
            for layer in range(storage.shape[1])
        )
        buffers.append(
            CacheBuffer(
                regions, sum(region.numel() for region in regions), str(storage.device)
            )
        )
    return tuple(buffers)


__all__ = ["cache_signature", "slab_buffers", "paged_kv_buffers"]
