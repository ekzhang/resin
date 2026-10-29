"""Graph-owned Qwen attention metadata; imported only on CUDA execution."""

from typing import Any, cast

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["table_width"])
def _lengths(
    positions,
    slabs,
    first_blocks,
    indptr,
    last,
    table_width,
    ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    PAGE: tl.constexpr,
    TOKENS: tl.constexpr,
    OFFSET: tl.constexpr,
):
    row = tl.arange(0, BLOCK_ROWS)
    valid = row < ROWS
    position = tl.load(positions + row, valid, other=0)
    first = tl.load(first_blocks + row, valid, other=0)
    live = tl.load(slabs + row * table_width, valid, other=0) != 0
    # Null graph rows still execute Q queries against a valid null KV prefix.
    used = tl.where(live, position + OFFSET - first * PAGE, TOKENS)
    count = tl.where(valid, (used + PAGE - 1) // PAGE, 0).to(tl.int32)
    end = tl.cumsum(count)
    tl.store(indptr + row, end - count, valid)
    tl.store(indptr + ROWS, tl.sum(count, 0))
    tl.store(last + row, (used - 1) % PAGE + 1, valid)


@triton.jit(do_not_specialize=["table_width"])
def _pages(
    slabs,
    slots,
    indptr,
    base_pages,
    table_width,
    SLAB_UNITS: tl.constexpr,
    LAYERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(indptr + row)
    count = tl.load(indptr + row + 1) - start
    valid = (column < count) & (column < table_width)
    address = row * table_width + column
    slab = tl.load(slabs + address, valid, other=0)
    slot = tl.load(slots + address, valid, other=0)
    tl.store(base_pages + start + column, slab * SLAB_UNITS + slot * LAYERS, valid)


def refresh(
    positions,
    group,
    indptr,
    last,
    base_pages,
    *,
    page,
    tokens,
    offset,
    slab_units,
    layers,
):
    rows, width = group.slab_ids.shape
    cast(Any, _lengths)[(1,)](
        positions,
        group.slab_ids,
        group.first_blocks,
        indptr,
        last,
        width,
        rows,
        triton.next_power_of_2(rows),
        page,
        tokens,
        offset,
        num_warps=4,
    )
    cast(Any, _pages)[(rows, triton.cdiv(width, 128))](
        group.slab_ids,
        group.slotnums,
        indptr,
        base_pages,
        width,
        slab_units,
        layers,
        128,
        num_warps=4,
    )
