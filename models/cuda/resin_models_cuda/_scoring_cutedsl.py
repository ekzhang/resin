"""Private sparse CuTe scoring kernels and their graph-owned workspaces.

This is a much faster backend for computing logprobs, top-k logprobs, and target
logprobs than the reference PyTorch. It's been tested on SM80+.
"""

from __future__ import annotations

import hashlib
import math
import operator
import os
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass._mlir.dialects import arith, vector
from cutlass.cutlass_dsl import dsl_user_op
from quack.cache import jit_cache
from quack.copy_utils import copy, predicate_k, tiled_copy_2d
from quack.reduce import row_reduce
from quack.sort.bitonic_sort import bitonic_sort, bitonic_topk_merge

_TYPES = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
    torch.bool: Uint8,
    torch.uint64: Uint64,
}
_SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_ABI = (
    str(torch.__version__),
    torch.version.cuda,
    version("nvidia-cutlass-dsl-libs-base"),
)


def _pointer(dtype, address=0, alignment=None):
    return cute.runtime.make_ptr(
        dtype,
        address,
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8) if alignment is None else alignment,
    )


@dsl_user_op
def _maximum_uint(values, *, loc=None, ip=None):
    # The pinned TensorSSA.reduce(MAX) uses a floating opcode for integers too.
    dtype = values.element_type
    return dtype(
        vector.reduction(
            dtype.mlir_type,
            vector.CombiningKind.MAXUI,
            values,
            acc=dtype(0).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _less_uint64(left, right, *, loc=None, ip=None):
    return Boolean(
        arith.cmpi(
            arith.CmpIPredicate.ult,
            Uint64(left).ir_value(loc=loc, ip=ip),
            Uint64(right).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _policy(p, g, row, flags: cutlass.Constexpr):
    width = Int32(g[1])
    request, column = row // width, row % width
    enabled = True
    if cutlass.const_expr(flags[0]):
        mask = cute.make_tensor(
            p[2], cute.make_layout((g[0] // g[1], g[1]), stride=(g[7], g[8]))
        )
        enabled = mask[request, column] != 0
    if cutlass.const_expr(flags[1]):
        counts = cute.make_tensor(p[3], cute.make_layout(g[0] // g[1], stride=g[9]))
        enabled = enabled & (column < counts[request])
    return enabled, request, column


@cute.jit
def _requested(p, g, request, column, capacity, flags: cutlass.Constexpr):
    count = capacity
    if cutlass.const_expr(flags[2]):
        counts = cute.make_tensor(
            p[4], cute.make_layout((g[0] // g[1], g[1]), stride=(g[10], g[11]))
        )
        count = min(count, Int64(counts[request, column]))
    return max(count, Int64(0))


@cute.jit
def _row(
    p, g, request, column, vocab: cutlass.Constexpr[int], vec: cutlass.Constexpr[int]
):
    s0, s1, sv = g[2], g[3], g[4]
    if cutlass.const_expr(vec > 1):
        s0, s1 = cute.assume(s0, divby=vec), cute.assume(s1, divby=vec)
        sv = Int64(1)
    return cute.make_tensor(
        p[0] + request * s0 + column * s1, cute.make_layout((1, vocab), stride=(0, sv))
    )


@cute.jit
def _select32(keys: cute.Tensor, count, reduction: cute.Tensor, scratch: cute.Tensor):
    dtype = keys.element_type
    tid, _, _ = cute.arch.thread_idx()
    lane, warp = tid % 32, tid // 32
    if count <= 8:
        if tid < 32:
            scratch[0, tid] = dtype(0)
        cute.arch.barrier()
        for rank in cutlass.range(count):
            local = _maximum_uint(keys.load())
            maximum = row_reduce(
                local, cute.ReductionOp.MAX, 128, reduction, init_val=dtype(0)
            )
            if tid == 0:
                scratch[0, rank] = maximum
            for index in cutlass.range_constexpr(cute.size(keys)):
                keys[index] = cutlass.select_(
                    keys[index] == maximum, dtype(0), keys[index]
                )
            cute.arch.barrier()
    else:
        best = cute.make_rmem_tensor(32, dtype)
        for index in cutlass.range_constexpr(32):
            best[index] = (
                keys[index] if cutlass.const_expr(index < cute.size(keys)) else dtype(0)
            )
        bitonic_sort(best, ascending=False)
        for stage in cutlass.range_constexpr(5):
            other = cute.make_rmem_tensor(32, dtype)
            for index in cutlass.range_constexpr(32):
                other[index] = cute.arch.shuffle_sync_bfly(
                    best[index], offset=1 << stage
                )
            bitonic_topk_merge(best, other, k=32, ascending=False)
        if lane == 0:
            for index in cutlass.range_constexpr(32):
                scratch[warp, index] = best[index]
        cute.arch.barrier()
        for stage in cutlass.range_constexpr(2):
            if warp % (2 << stage) == 0:
                other = cute.make_rmem_tensor(32, dtype)
                for index in cutlass.range_constexpr(32):
                    other[index] = scratch[warp + (1 << stage), index]
                bitonic_topk_merge(best, other, k=32, ascending=False)
                if lane == 0:
                    for index in cutlass.range_constexpr(32):
                        scratch[warp, index] = best[index]
            cute.arch.barrier()


@cute.jit
def _heads(source, cache, row, group, groups, count):
    lane, _, _ = cute.arch.thread_idx()
    for index in cutlass.range_constexpr(16):
        local = lane + index * 32
        flat = group * 512 + local
        key = Uint64(0)
        if flat < groups * 32:
            key = Uint64(source[row, flat // 32, flat % 32])
        cache[local // 32, local % 32] = key
    cute.arch.sync_warp()
    head, result, position = Uint64(0), Uint64(0), Int32(0)
    if lane < 16:
        head = Uint64(cache[lane, 0])
    for rank in cutlass.range(count):
        high = Uint32(head >> 32)
        high_max = Uint32(cute.arch.warp_redux_sync(high, "umax"))
        low = cutlass.select_(high == high_max, Uint32(head), Uint32(0))
        low_max = Uint32(cute.arch.warp_redux_sync(low, "umax"))
        maximum = (Uint64(high_max) << 32) | Uint64(low_max)
        if lane == rank:
            result = maximum
        if (head == maximum) & (head != 0):
            position += 1
            head = Uint64(0)
            if position < 32:
                head = Uint64(cache[lane, position])
    return result


class _Kernels:
    def __init__(self, dtype, vocab, vec, flags, top, original, kind):
        self.dtype, self.vocab, self.vec = dtype, vocab, vec
        self.flags, self.top, self.original, self.kind = flags, top, original, kind
        self.tile = 2048 if top else (4096 if original else 8192)
        self.parts = (vocab + self.tile - 1) // self.tile

    @cute.jit
    def __call__(self, p: tuple, g: tuple, o: tuple, q: tuple, stream: cuda.CUstream):
        if cutlass.const_expr(self.kind == "partial"):
            self.partial(p, g, o, q).launch(
                grid=(g[0], self.parts, 1), block=(128, 1, 1), stream=stream
            )
        elif cutlass.const_expr(self.kind == "finish"):
            threads = 128 if cutlass.const_expr(self.original) else 32
            self.finish(p, g, o, q).launch(
                grid=(g[0], 1, 1), block=(threads, 1, 1), stream=stream
            )
        elif cutlass.const_expr(self.kind == "merge"):
            self.merge(p, g, o, q).launch(
                grid=(g[0], cute.ceil_div(q[2], 16), 1), block=(32, 1, 1), stream=stream
            )
        elif cutlass.const_expr(self.kind == "targets"):
            self.targets(p, g, o, q).launch(
                grid=(cute.ceil_div(q[3], 128), 1, 1), block=(128, 1, 1), stream=stream
            )
        else:
            self.native(p, g, o, q).launch(
                grid=(g[0], cute.ceil_div(q[0], 128), 1),
                block=(128, 1, 1),
                stream=stream,
            )

    @cute.kernel
    def partial(self, p, g, o, q):
        tid, _, _ = cute.arch.thread_idx()
        row, part, _ = cute.arch.block_idx()
        enabled, request, column = _policy(p, g, row, self.flags)
        count = _requested(p, g, request, column, q[0], self.flags)
        first = True
        active = enabled
        if cutlass.const_expr(self.top):
            first = q[1] == 0
            active = enabled & (first | (count > q[1]))
        maximum, mass = Float32(-Float32.inf), Float32(0)
        allocator = cutlass.utils.SmemAllocator()
        stages = 3 if cutlass.const_expr(self.original or self.top) else 2
        reduction = allocator.allocate_tensor(
            Float32, cute.make_ordered_layout((1, (4, 1), stages), order=(1, 0, 2))
        )
        if cutlass.const_expr(self.top):
            key_dtype = Uint32 if cutlass.const_expr(self.dtype.width == 16) else Uint64
            key_reduction = allocator.allocate_tensor(
                key_dtype, cute.make_layout((1, (4, 1)))
            )
            scratch = allocator.allocate_tensor(
                key_dtype, cute.make_layout((4, 32), stride=(32, 1))
            )
        if active:
            x = _row(p, g, request, column, self.vocab, self.vec)
            identity = cute.make_identity_tensor(x.shape)
            gx = cute.local_tile(x, (1, self.tile), (0, part))
            cx = cute.local_tile(identity, (1, self.tile), (0, part))
            layout = tiled_copy_2d(self.dtype, 128, 128, self.vec)
            thread = layout.get_slice(tid)
            source, coords = thread.partition_S(gx), thread.partition_S(cx)
            rx = cute.make_rmem_tensor_like(source)
            rx.fill(-Float32.inf)
            copy(source, rx, pred=predicate_k(coords, self.vocab))
            if first:
                values = cute.make_rmem_tensor_like(rx, Float32)
                weights = cute.make_rmem_tensor_like(rx, Float32)
                if cutlass.const_expr(self.original or self.top):
                    invalid = cute.make_rmem_tensor_like(rx, Float32)
                for index in cutlass.range_constexpr(cute.size(rx)):
                    value = Float32(rx[index])
                    bad = (value != value) | (value == Float32.inf)
                    finite = (value > -Float32.inf) & (value < Float32.inf)
                    if cutlass.const_expr(self.original or self.top):
                        values[index] = cutlass.select_(finite, value, -Float32.inf)
                        invalid[index] = Float32(bad)
                    else:
                        values[index] = cutlass.select_(bad, Float32.inf, value)
                maximum = row_reduce(
                    values.load(),
                    cute.ReductionOp.MAX,
                    128,
                    reduction[None, None, 0],
                    init_val=-Float32.inf,
                )
                for index in cutlass.range_constexpr(cute.size(rx)):
                    value = Float32(rx[index])
                    finite = (value > -Float32.inf) & (value < Float32.inf)
                    weight = cute.math.exp2(
                        (value - maximum) * math.log2(math.e), fastmath=True
                    )
                    weights[index] = cutlass.select_(finite, weight, Float32(0))
                mass = row_reduce(
                    weights.load(),
                    cute.ReductionOp.ADD,
                    128,
                    reduction[None, None, 1],
                    init_val=0.0,
                )
                if cutlass.const_expr(self.original or self.top):
                    bad = row_reduce(
                        invalid.load(),
                        cute.ReductionOp.MAX,
                        128,
                        reduction[None, None, 2],
                        init_val=0.0,
                    )
                    maximum = cutlass.select_(bad > 0, Float32(float("nan")), maximum)
                else:
                    maximum = cutlass.select_(
                        maximum == Float32.inf, Float32(float("nan")), maximum
                    )
            if cutlass.const_expr(self.top):
                if count > q[1]:
                    threshold = Uint64(0)
                    if q[1] > 0:
                        thresholds = cute.make_tensor(o[6], cute.make_layout(g[0]))
                        threshold = Uint64(thresholds[row])
                    keys = cute.make_rmem_tensor(cute.size(rx), key_dtype)
                    for index in cutlass.range_constexpr(cute.size(rx)):
                        value, token = Float32(rx[index]), Int32(coords[index][1])
                        bits = Uint32(value.bitcast(Uint32))
                        ordered = cutlass.select_(
                            (bits & Uint32(0x80000000)) != 0,
                            ~bits,
                            bits ^ Uint32(0x80000000),
                        )
                        full_key = (Uint64(ordered) << 32) | Uint64(
                            Uint32(0xFFFFFFFF) - Uint32(token)
                        )
                        keep = (token < self.vocab) & (value == value)
                        if q[1] > 0:
                            keep = keep & _less_uint64(full_key, threshold)
                        if cutlass.const_expr(self.dtype.width == 16):
                            key = (ordered & Uint32(0xFFFFF800)) | (
                                Uint32(2047) - (Uint32(token) & Uint32(2047))
                            )
                        else:
                            key = full_key
                        keys[index] = cutlass.select_(keep, key, key_dtype(0))
                    _select32(
                        keys,
                        Int32(min(count - q[1], Int64(32))),
                        key_reduction,
                        scratch,
                    )
                    if tid < 32:
                        selected_key = scratch[0, tid]
                        if cutlass.const_expr(self.dtype.width == 16):
                            output_token = (
                                Uint32(part * 2048)
                                + Uint32(2047)
                                - (selected_key & Uint32(2047))
                            )
                            output_ordered = (
                                selected_key & Uint32(0xFFFFF800)
                            ) | cutlass.select_(
                                (selected_key & Uint32(0x80000000)) == 0,
                                Uint32(2047),
                                Uint32(0),
                            )
                            expanded = (Uint64(output_ordered) << 32) | Uint64(
                                Uint32(0xFFFFFFFF) - output_token
                            )
                            output_key = cutlass.select_(
                                selected_key != 0, expanded, Uint64(0)
                            )
                        else:
                            output_key = Uint64(selected_key)
                        candidates = cute.make_tensor(
                            o[4],
                            cute.make_layout(
                                (g[0], self.parts, 32), stride=(self.parts * 32, 32, 1)
                            ),
                        )
                        candidates[row, part, tid] = output_key
        if (tid == 0) & first:
            maxima = cute.make_tensor(
                o[0], cute.make_layout((g[0], self.parts), stride=(self.parts, 1))
            )
            masses = cute.make_tensor(
                o[1], cute.make_layout((g[0], self.parts), stride=(self.parts, 1))
            )
            maxima[row, part], masses[row, part] = maximum, mass

    @cute.kernel
    def merge(self, p, g, o, q):
        lane, _, _ = cute.arch.thread_idx()
        row, group, _ = cute.arch.block_idx()
        enabled, request, column = _policy(p, g, row, self.flags)
        count = _requested(p, g, request, column, q[0], self.flags)
        allocator = cutlass.utils.SmemAllocator()
        cache = allocator.allocate_tensor(
            Uint64, cute.make_layout((16, 32), stride=(33, 1))
        )
        if enabled & (count > q[1]):
            source = cute.make_tensor(
                o[4], cute.make_layout((g[0], q[2], 32), stride=(q[2] * 32, 32, 1))
            )
            result = _heads(
                source, cache, row, group, q[2], Int32(min(count - q[1], Int64(32)))
            )
            groups = cute.ceil_div(q[2], 16)
            output = cute.make_tensor(
                o[5], cute.make_layout((g[0], groups, 32), stride=(groups * 32, 32, 1))
            )
            output[row, group, lane] = result

    @cute.kernel
    def finish(self, p, g, o, q):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        enabled, request, column = _policy(p, g, row, self.flags)
        count = _requested(p, g, request, column, q[0], self.flags)
        first = True
        if cutlass.const_expr(self.top):
            first = q[1] == 0
        maximum, log_mass, chosen = Float32(0), Float32(0), Float32(0)
        key = Uint64(0)
        stats = cute.make_tensor(o[2], cute.make_layout((g[0], 2), stride=(2, 1)))
        allocator = cutlass.utils.SmemAllocator()
        threads = 128 if cutlass.const_expr(self.original) else 32
        if cutlass.const_expr(self.original):
            reduction = allocator.allocate_tensor(
                Float32, cute.make_ordered_layout((1, (4, 1), 3), order=(1, 0, 2))
            )
        if cutlass.const_expr(self.top):
            cache = allocator.allocate_tensor(
                Uint64, cute.make_layout((16, 32), stride=(33, 1))
            )
        if enabled:
            if first:
                maxima = cute.make_tensor(
                    o[0], cute.make_layout((g[0], self.parts), stride=(self.parts, 1))
                )
                masses = cute.make_tensor(
                    o[1], cute.make_layout((g[0], self.parts), stride=(self.parts, 1))
                )
                local_max, local_bad = Float32(-Float32.inf), Float32(0)
                for index in cutlass.range_constexpr(
                    cute.ceil_div(self.parts, threads)
                ):
                    part = tid + index * threads
                    if part < self.parts:
                        value = maxima[row, part]
                        replacement = (
                            -Float32.inf
                            if cutlass.const_expr(self.original)
                            else Float32.inf
                        )
                        local_max = max(
                            local_max,
                            cutlass.select_(value == value, value, replacement),
                        )
                        local_bad = max(local_bad, Float32(value != value))
                if cutlass.const_expr(self.original):
                    maximum = row_reduce(
                        local_max,
                        cute.ReductionOp.MAX,
                        128,
                        reduction[None, None, 0],
                        init_val=-Float32.inf,
                    )
                else:
                    maximum = cute.arch.warp_reduction(local_max, cute.arch.fmax)
                local_mass = Float32(0)
                for index in cutlass.range_constexpr(
                    cute.ceil_div(self.parts, threads)
                ):
                    part = tid + index * threads
                    if part < self.parts:
                        value = maxima[row, part]
                        finite = (
                            (value > -Float32.inf)
                            & (value < Float32.inf)
                            & (maximum < Float32.inf)
                        )
                        weight = masses[row, part] * cute.math.exp2(
                            (value - maximum) * math.log2(math.e), fastmath=True
                        )
                        local_mass += cutlass.select_(finite, weight, Float32(0))
                if cutlass.const_expr(self.original):
                    mass = row_reduce(
                        local_mass,
                        cute.ReductionOp.ADD,
                        128,
                        reduction[None, None, 1],
                        init_val=0.0,
                    )
                    bad = row_reduce(
                        local_bad,
                        cute.ReductionOp.MAX,
                        128,
                        reduction[None, None, 2],
                        init_val=0.0,
                    )
                    valid = (bad == 0) & (maximum > -Float32.inf) & (mass > 0)
                else:
                    mass = cute.arch.warp_reduction(local_mass, operator.add)
                    valid = (
                        (maximum > -Float32.inf) & (maximum < Float32.inf) & (mass > 0)
                    )
                log_mass = cutlass.select_(
                    valid, cute.math.log(mass, fastmath=True), Float32(float("nan"))
                )
                maximum = cutlass.select_(
                    maximum == Float32.inf, Float32(float("nan")), maximum
                )
                if tid == 0:
                    ids = cute.make_tensor(
                        p[1],
                        cute.make_layout((g[0] // g[1], g[1]), stride=(g[5], g[6])),
                    )
                    token = ids[request, column]
                    chosen = Float32(float("nan"))
                    if (token >= 0) & (token < self.vocab):
                        x = _row(p, g, request, column, self.vocab, self.vec)
                        chosen = (Float32(x[0, token]) - maximum) - log_mass
            else:
                maximum, log_mass = stats[row, 0], stats[row, 1]
            if cutlass.const_expr(self.top):
                if count > q[1]:
                    source = cute.make_tensor(
                        o[4],
                        cute.make_layout((g[0], q[2], 32), stride=(q[2] * 32, 32, 1)),
                    )
                    if q[2] == 1:
                        key = Uint64(source[row, 0, tid])
                    else:
                        key = _heads(
                            source,
                            cache,
                            row,
                            Int32(0),
                            q[2],
                            Int32(min(count - q[1], Int64(32))),
                        )
        if (tid == 0) & first:
            output = cute.make_tensor(o[3], cute.make_layout(g[0]))
            output[row] = chosen
            stats[row, 0], stats[row, 1] = maximum, log_mass
        if cutlass.const_expr(self.top):
            thresholds = cute.make_tensor(o[6], cute.make_layout(g[0]))
            if tid == 31:
                thresholds[row] = key
            slot = q[1] + tid
            if slot < q[0]:
                token_out, result = Int64(-1), Float32(0)
                if enabled & (slot < min(count, Int64(self.vocab))):
                    result = Float32(float("nan"))
                    if key != 0:
                        token_out = Int64(Uint32(0xFFFFFFFF) - Uint32(key))
                        ordered = Uint32(key >> 32)
                        bits = cutlass.select_(
                            (ordered & Uint32(0x80000000)) != 0,
                            ordered ^ Uint32(0x80000000),
                            ~ordered,
                        )
                        result = (Uint32(bits).bitcast(Float32) - maximum) - log_mass
                ids = cute.make_tensor(
                    o[7], cute.make_layout((g[0], q[0]), stride=(q[0], 1))
                )
                scores = cute.make_tensor(
                    o[8], cute.make_layout((g[0], q[0]), stride=(q[0], 1))
                )
                ids[row, slot], scores[row, slot] = token_out, result

    @cute.kernel
    def targets(self, p, g, o, q):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = block * 128 + tid
        if index < q[3]:
            rows = cute.make_tensor(o[9], cute.make_layout(q[3], stride=q[4]))
            ids = cute.make_tensor(o[10], cute.make_layout(q[3], stride=q[5]))
            output = cute.make_tensor(o[11], cute.make_layout(q[3]))
            row, token = Int64(rows[index]), Int64(ids[index])
            result = Float32(float("nan"))
            if (row >= 0) & (row < g[0]):
                enabled, request, column = _policy(p, g, Int32(row), self.flags)
                result = Float32(0)
                if enabled:
                    result = Float32(float("nan"))
                    if (token >= 0) & (token < self.vocab):
                        stats = cute.make_tensor(
                            o[2], cute.make_layout((g[0], 2), stride=(2, 1))
                        )
                        x = _row(p, g, request, column, self.vocab, self.vec)
                        result = (Float32(x[0, token]) - stats[row, 0]) - stats[row, 1]
            output[index] = result

    @cute.kernel
    def native(self, p, g, o, q):
        tid, _, _ = cute.arch.thread_idx()
        row, block, _ = cute.arch.block_idx()
        slot = block * 128 + tid
        if slot < q[0]:
            enabled, request, column = _policy(p, g, row, self.flags)
            count = _requested(p, g, request, column, q[0], self.flags)
            token, result = Int64(-1), Float32(0)
            if enabled & (slot < min(count, q[6])):
                native_ids = cute.make_tensor(
                    o[12], cute.make_layout((g[0], q[6]), stride=(q[6], 1))
                )
                values = cute.make_tensor(
                    o[13], cute.make_layout((g[0], q[6]), stride=(q[6], 1))
                )
                stats = cute.make_tensor(
                    o[2], cute.make_layout((g[0], 2), stride=(2, 1))
                )
                token = Int64(native_ids[row, slot])
                result = (Float32(values[row, slot]) - stats[row, 0]) - stats[row, 1]
            ids = cute.make_tensor(
                o[7], cute.make_layout((g[0], q[0]), stride=(q[0], 1))
            )
            scores = cute.make_tensor(
                o[8], cute.make_layout((g[0], q[0]), stride=(q[0], 1))
            )
            ids[row, slot], scores[row, slot] = token, result


def _build(stamp, types, vocab, vec, flags, top, original, kind):
    p = tuple(
        _pointer(t, alignment=vec * t.width // 8 if index == 0 else None)
        for index, t in enumerate(types[:5])
    )
    out_types = (
        Float32,
        Float32,
        Float32,
        Float32,
        Uint64,
        Uint64,
        Uint64,
        Int64,
        Float32,
        types[5],
        types[6],
        Float32,
        Int32,
        types[0],
    )
    o = tuple(_pointer(t) for t in out_types)
    return cute.compile(
        _Kernels(types[0], vocab, vec, flags, top, original, kind),
        p,
        (Int64(0),) * 12,
        o,
        (Int64(0),) * 7,
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=None)
def _context_cache(context):
    # Independent in-memory executors per context; stable keys share .o files.
    return jit_cache(_build)


def _row_strides(value, shape, name):
    if value is None:
        return 0, 0
    if tuple(value.shape) == shape:
        return (value.stride(0), 0) if len(shape) == 1 else value.stride()
    if len(shape) == 2 and tuple(value.shape) == shape[:1]:
        return value.stride(0), 0
    raise ValueError(f"{name} must match the logits rows or request batch")


def score_logits(
    logits: torch.Tensor,
    selected_ids: torch.Tensor,
    *,
    score_mask: torch.Tensor | None = None,
    valid_counts: torch.Tensor | None = None,
    top_counts: torch.Tensor | None = None,
    max_top_k: int = 0,
    target_rows: torch.Tensor | None = None,
    target_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact raw selected, alternative-ID, alternative and target scores."""
    if (
        not logits.is_cuda
        or logits.ndim not in (2, 3)
        or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError("score logits require a CUDA FP16/BF16/FP32 matrix or grid")
    shape, vocab = tuple(logits.shape[:-1]), logits.shape[-1]
    if not 0 < vocab < 2**31 or tuple(selected_ids.shape) != shape:
        raise ValueError(
            "selected IDs must match a positive int32-sized score vocabulary"
        )
    if selected_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected IDs must use int32 or int64")
    if type(max_top_k) is not int or max_top_k < 0:
        raise ValueError("maximum top count must be a nonnegative integer")
    if (target_rows is None) != (target_ids is None):
        raise ValueError("target rows and IDs must be supplied together")
    inputs = [selected_ids]
    for value, dtype, name in (
        (score_mask, (torch.bool,), "score mask"),
        (top_counts, (torch.int32, torch.int64), "top counts"),
        (valid_counts, (torch.int32, torch.int64), "valid counts"),
    ):
        if value is not None:
            if value.dtype not in dtype:
                raise ValueError(f"invalid {name} dtype")
            inputs.append(value)
    if valid_counts is not None and tuple(valid_counts.shape) != shape[:1]:
        raise ValueError("valid counts must contain one integer per request")
    if target_rows is not None and target_ids is not None:
        if (
            target_rows.ndim != 1
            or target_rows.shape != target_ids.shape
            or target_rows.dtype not in (torch.int32, torch.int64)
            or target_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("target rows and IDs must be matching integer vectors")
        inputs.extend((target_rows, target_ids))
    if any(
        value.device != logits.device or any(s < 0 for s in value.stride())
        for value in inputs
    ):
        raise ValueError("score inputs must share a device and use nonnegative strides")
    if any(s < 0 for s in logits.stride()):
        raise ValueError("score logits strides must be nonnegative")
    mask_strides = _row_strides(score_mask, shape, "score mask")
    top_strides = _row_strides(top_counts, shape, "top counts")
    rows, width = selected_ids.numel(), 1 if logits.ndim == 2 else logits.shape[1]
    selected = torch.empty(shape, dtype=torch.float32, device=logits.device)
    ids = torch.empty((*shape, max_top_k), dtype=torch.int64, device=logits.device)
    scores = torch.empty((*shape, max_top_k), dtype=torch.float32, device=logits.device)
    targets = torch.empty(
        0 if target_ids is None else target_ids.numel(),
        dtype=torch.float32,
        device=logits.device,
    )
    if not rows:
        if targets.numel():
            raise ValueError("target scores require at least one logits row")
        return selected, ids, scores, targets
    s0, s1, sv = (
        (logits.stride(0), 0, logits.stride(1)) if logits.ndim == 2 else logits.stride()
    )
    i0, i1 = (
        (selected_ids.stride(0), 0) if selected_ids.ndim == 1 else selected_ids.stride()
    )
    vec = 1
    if sv == 1:
        vec = math.gcd(
            vocab,
            16 // logits.element_size(),
            logits.data_ptr() // logits.element_size(),
            s0,
            s1,
        )
    flags = (score_mask is not None, valid_counts is not None, top_counts is not None)
    meta = (
        logits,
        selected_ids,
        score_mask,
        valid_counts,
        top_counts,
        target_rows,
        target_ids,
    )
    types = tuple(
        _TYPES[value.dtype] if value is not None else default
        for value, default in zip(
            meta,
            (_TYPES[logits.dtype], Int64, Uint8, Int32, Int32, Int64, Int64),
            strict=True,
        )
    )
    p = tuple(
        _pointer(
            t,
            0 if value is None else value.data_ptr(),
            vec * t.width // 8 if index == 0 else None,
        )
        for index, (value, t) in enumerate(zip(meta[:5], types[:5], strict=True))
    )
    g = (
        rows,
        width,
        s0,
        s1,
        sv,
        i0,
        i1,
        *mask_strides,
        0 if valid_counts is None else valid_counts.stride(0),
        *top_strides,
    )
    with torch.cuda.device(logits.device):
        status, handle = cuda.cuCtxGetCurrent()
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("cannot read the current CUDA scoring context")
        context = (torch.cuda.current_device(), int(handle))
        native = 32 < max_top_k <= 2048
        top = max_top_k > 0 and not native
        capability = torch.cuda.get_device_capability(logits.device)
        major = capability[0]
        stamp = (_SOURCE, capability, os.getenv("CUTE_DSL_ARCH"), _ABI)
        # SM100 scalar FP32 layouts need the verified smaller reduction geometry.
        original = (
            not top
            and logits.dtype == torch.float32
            and ((vocab >= 262144 and major in (9, 10)) or (major == 10 and vec == 1))
        )
        tile = 2048 if top else (4096 if original else 8192)
        parts = (vocab + tile - 1) // tile
        maxima = torch.empty((rows, parts), dtype=torch.float32, device=logits.device)
        masses = torch.empty_like(maxima)
        stats = torch.empty((rows, 2), dtype=torch.float32, device=logits.device)
        threshold = (
            torch.empty(rows, dtype=torch.uint64, device=logits.device) if top else None
        )
        out = [
            maxima,
            masses,
            stats,
            selected,
            None,
            None,
            threshold,
            ids,
            scores,
            target_rows,
            target_ids,
            targets,
            None,
            None,
        ]
        out_types = (
            Float32,
            Float32,
            Float32,
            Float32,
            Uint64,
            Uint64,
            Uint64,
            Int64,
            Float32,
            types[5],
            types[6],
            Float32,
            Int32,
            types[0],
        )
        q = [
            max_top_k,
            0,
            parts,
            targets.numel(),
            1 if target_rows is None else target_rows.stride(0),
            1 if target_ids is None else target_ids.stride(0),
            0,
        ]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        def launch(kind):
            pointers = tuple(
                _pointer(t, 0 if value is None else value.data_ptr())
                for value, t in zip(out, out_types, strict=True)
            )
            _context_cache(context)(
                stamp, types, vocab, vec, flags, top, original, kind
            )(p, g, pointers, tuple(q), stream)

        if top:
            candidates = torch.empty(
                (rows, parts, 32), dtype=torch.uint64, device=logits.device
            )
            buffers, groups = [], parts
            while groups > 16:
                groups = (groups + 15) // 16
                buffers.append(
                    torch.empty(
                        (rows, groups, 32), dtype=torch.uint64, device=logits.device
                    )
                )
            for start in range(0, max_top_k, 32):
                q[1], q[2], out[4] = start, parts, candidates
                launch("partial")
                for merged in buffers:
                    out[5] = merged
                    launch("merge")
                    out[4], q[2] = merged, merged.shape[1]
                launch("finish")
        else:
            launch("partial")
            launch("finish")
            if native:
                # The public wrapper owns a process-global mutable workspace.
                # This pinned radix primitive instead receives graph-owned state.
                from flashinfer.topk import get_topk_module

                native_k = min(max_top_k, vocab)
                values = torch.empty(
                    (rows, native_k), dtype=logits.dtype, device=logits.device
                )
                workspace = torch.zeros(
                    1024 * 1024, dtype=torch.uint8, device=logits.device
                )
                native_ids = get_topk_module().radix_topk(
                    logits.contiguous().view(rows, vocab),
                    native_k,
                    True,
                    True,
                    1,
                    workspace,
                    values,
                    True,
                )
                out[12], out[13], q[6] = native_ids, values, native_k
                launch("native")
        if targets.numel():
            launch("targets")
    return selected, ids, scores, targets
