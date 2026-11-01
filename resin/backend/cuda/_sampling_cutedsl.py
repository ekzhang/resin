"""Graph-safe nucleus cutoffs from deterministic integer mass histograms.

Positive FP32 weights are selected by exponent, then mantissa. Integer sums
make bucket accumulation independent of CTA order; FP64 reconstructs their
mass. Workspaces belong to each invocation and captured graph.
"""

from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass import Float32, Float64, Int32, Int64, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from quack.cache import jit_cache
from quack.reduce import row_reduce

from resin.backend.cuda.logits_sampling import _scaled_difference

_TYPES = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: Float32,
    torch.float64: Float64,
    torch.int32: Int32,
    torch.int64: Int64,
    torch.bool: Uint8,
}
_SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_ABI = (
    str(torch.__version__),
    torch.version.cuda,
    version("nvidia-cutlass-dsl-libs-base"),
)


@triton.jit(
    do_not_specialize=["s0", "s1", "st"],
    do_not_specialize_on_alignment=[
        "Logits",
        "Temperature",
        "State",
        "ActiveK",
        "ActiveP",
        "GuardMass",
    ],
)
def _joint_mass(
    Logits,
    Temperature,
    State,
    ActiveK,
    ActiveP,
    GuardMass,
    s0,
    s1,
    st,
    V: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if tl.load(ActiveK + row) & tl.load(ActiveP + row):
        tile = tl.program_id(1)
        index = tile * BLOCK + tl.arange(0, BLOCK)
        valid = index < V
        value = tl.load(
            Logits + row * s0 + index.to(tl.int64) * s1,
            valid,
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.load(State + row * 3 + 2)
        temperature = tl.load(Temperature + row * st)
        weight = tl.exp(_scaled_difference(value, maximum, temperature))
        weight = tl.where(valid & (value > -float("inf")), weight, 0.0)
        cutoff = tl.load(State + row * 3)
        # This is only a skip certificate. Even a sequential 1024-term FP32
        # sum has <6.1e-5 relative error; the ratio error stays well inside
        # the 0.001 guard margin. Combine these small partials in FP64 below.
        slot = (row * PARTS + tile) * 2
        tl.store(GuardMass + slot, tl.sum(weight, 0))
        tl.store(
            GuardMass + slot + 1, tl.sum(tl.where(value >= cutoff, weight, 0.0), 0)
        )


def _ptr(dtype, address=0):
    return cute.runtime.make_ptr(
        dtype, address, cute.AddressSpace.gmem, assumed_align=dtype.width // 8
    )


@dsl_user_op
def _divide(a, b, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _weight(value, maximum, temperature):
    difference = value - maximum
    scaled = _divide(difference, temperature)
    if (
        (maximum >= Float32(2.0**103))
        & (difference == -Float32.inf)
        & (value > -Float32.inf)
        & (value < Float32.inf)
    ):
        scaled = _divide(value, temperature) - _divide(maximum, temperature)
    return cute.math.exp2(scaled * math.log2(math.e), fastmath=True)


@cute.jit
def _bucket_mass(significands, count, exponent):
    # Normal FP32 weights have mass (mantissa + 2**23) * 2**(exponent-150).
    # Subnormals omit the implicit bit and use 2**-149. The integer numerator
    # fits Int64 even for the largest supported vocabulary.
    scale_bits = Int64(cutlass.select_(exponent > 0, exponent + 873, Int32(874))) << 52
    numerator = significands
    if exponent > 0:
        numerator += Int64(count) << 23
    return Float64(numerator) * scale_bits.bitcast(Float64)


@cute.jit
def _histograms(p, rows):
    layout = cute.make_layout((rows, 256), stride=(256, 1))
    return (
        cute.make_tensor(p[5], layout),
        cute.make_tensor(p[6], layout),
        cute.make_tensor(p[7], layout),
    )


class _Nucleus:
    def __init__(self, vocab, stage, kind):
        self.vocab, self.stage, self.kind = vocab, stage, kind
        self.bins = 128 if stage in (0, 3) else 256
        self.items = self.bins // 128
        self.shift = (23, 15, 7, 0)[stage]
        # Positive weights <= 1 use 30 key bits. Earlier passes fix the high
        # digits; the remaining low digits are ignored until their pass.
        remaining_bits = self.shift + (self.bins - 1).bit_length()
        self.prefix_mask = (0x3FFFFFFF >> remaining_bits) << remaining_bits

    @cute.jit
    def __call__(self, p: tuple, g: tuple, stream: cuda.CUstream):
        if cutlass.const_expr(self.kind == "init"):
            self.initialize(p, g).launch(
                grid=(g[0], Int64(1), Int64(1)), block=(128, 1, 1), stream=stream
            )
        elif cutlass.const_expr(self.kind == "hist"):
            self.histogram(p, g).launch(
                grid=(g[0], g[1], Int64(1)), block=(128, 1, 1), stream=stream
            )
        else:
            self.finish(p, g).launch(
                grid=(g[0], Int64(1), Int64(1)), block=(128, 1, 1), stream=stream
            )

    @cute.kernel
    def initialize(self, p, g):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        allocator = cutlass.utils.SmemAllocator()
        reduction = allocator.allocate_tensor(
            Float64, cute.make_ordered_layout((1, (4, 1), 2), order=(1, 0, 2))
        )
        active = cute.make_tensor(p[4], cute.make_layout(g[0]))
        active_k = cute.make_tensor(p[3], cute.make_layout(g[0]))
        ps = cute.make_tensor(p[11], cute.make_layout(g[0]))
        control = cute.make_tensor(p[8], cute.make_layout((g[0], 2), stride=(2, 1)))
        enabled = active[row] != 0
        redundant = cutlass.Boolean(False)
        if enabled & (active_k[row] != 0) & (ps[row] > Float32(0.001)):
            # A nucleus beyond the inclusive top-k set cannot remove any of
            # those tokens. Keep the previous selector's conservative margin.
            partial = cute.make_tensor(
                p[10], cute.make_layout((g[0], g[5], 2), stride=(g[5] * 2, 2, 1))
            )
            total, retained = Float64(0), Float64(0)
            tile = Int64(tid)
            while tile < g[5]:
                total += Float64(partial[row, tile, 0])
                retained += Float64(partial[row, tile, 1])
                tile += Int64(128)
            total = row_reduce(
                total,
                cute.ReductionOp.ADD,
                128,
                reduction[None, None, 0],
                init_val=Float64(0),
            )
            retained = row_reduce(
                retained,
                cute.ReductionOp.ADD,
                128,
                reduction[None, None, 1],
                init_val=Float64(0),
            )
            redundant = (
                (total > 0)
                & (retained >= 0)
                & (retained < (Float64(ps[row]) - Float64(0.001)) * total)
            )
        enabled &= ~redundant
        if tid == 0:
            control[row, 0] = Int32(enabled)
            control[row, 1] = 0
        sums, counts, minima = _histograms(p, g[0])
        for offset in cutlass.range_constexpr(2):
            slot = tid + offset * 128
            sums[row, slot], counts[row, slot], minima[row, slot] = (
                Int64(0),
                Int32(0),
                Int32(2147483647),
            )

    @cute.kernel
    def histogram(self, p, g):
        tid, _, _ = cute.arch.thread_idx()
        row, part, _ = cute.arch.block_idx()
        allocator = cutlass.utils.SmemAllocator()
        shared_s = allocator.allocate_tensor(Int64, cute.make_layout(4 * self.bins))
        shared_c = allocator.allocate_tensor(Int32, cute.make_layout(4 * self.bins))
        shared_lo = allocator.allocate_tensor(Int32, cute.make_layout(4 * self.bins))
        control = cute.make_tensor(p[8], cute.make_layout((g[0], 2), stride=(2, 1)))
        if control[row, 0] != 0:
            x = cute.make_tensor(
                p[0], cute.make_layout((g[0], self.vocab), stride=(g[2], g[3]))
            )
            temperatures = cute.make_tensor(p[1], cute.make_layout(g[0], stride=g[4]))
            state = cute.make_tensor(p[2], cute.make_layout((g[0], 3), stride=(3, 1)))
            for offset in cutlass.range_constexpr(cute.ceil_div(4 * self.bins, 128)):
                index = tid + offset * 128
                shared_s[index], shared_c[index] = Int64(0), Int32(0)
                shared_lo[index] = Int32(2147483647)
            cute.arch.barrier()
            maximum, temperature = state[row, 2], temperatures[row]
            prefix = Int32(control[row, 1])
            # All lanes participate in match/redux, including the invalid tail.
            # One leader adds each equal-digit warp group's exact integer mass.
            base = Int64(part) * 512
            while base < Int64(self.vocab):
                for item in cutlass.range_constexpr(4):
                    token = Int64(base) + Int64(tid) * 4 + Int64(item)
                    valid = cutlass.Boolean(token < self.vocab)
                    value = Float32(-Float32.inf)
                    if valid:
                        value = Float32(x[row, token])
                    weight = Float32(_weight(value, maximum, temperature))
                    key = weight.bitcast(Int32)
                    tag = cutlass.select_(
                        valid & (weight > 0) & ((key & self.prefix_mask) == prefix),
                        Int32((key >> self.shift) & Int32(self.bins - 1)),
                        Int32(self.bins),
                    )
                    members = Int32(cute.arch.match_sync(0xFFFFFFFF, tag, kind="any"))
                    group_sum = cute.arch.warp_redux_sync(
                        key & Int32(0x7FFFFF), "add", mask_and_clamp=members
                    )
                    group_count = cute.arch.popc(members)
                    group_low = cute.arch.warp_redux_sync(
                        key, "min", mask_and_clamp=members
                    )
                    previous = Int32(cute.arch.lanemask_lt())
                    if (tag < self.bins) & ((members & previous) == 0):
                        cute.arch.atomic_add(
                            shared_s.iterator + tid // 32 * self.bins + tag,
                            Int64(group_sum),
                            scope="cta",
                        )
                        cute.arch.atomic_add(
                            shared_c.iterator + tid // 32 * self.bins + tag,
                            Int32(group_count),
                            scope="cta",
                        )
                        cute.arch.atomic_min(
                            shared_lo.iterator + tid // 32 * self.bins + tag,
                            Int32(group_low),
                            scope="cta",
                        )
                base += g[1] * 512
            cute.arch.barrier()
            sums, counts, minima = _histograms(p, g[0])
            for offset in cutlass.range_constexpr(cute.ceil_div(self.bins, 128)):
                digit = tid + offset * 128
                total_s, total_c = Int64(0), Int32(0)
                low = Int32(2147483647)
                for warp in cutlass.range_constexpr(4):
                    index = warp * self.bins + digit
                    total_s += shared_s[index]
                    total_c += shared_c[index]
                    low = min(low, shared_lo[index])
                if total_c > 0:
                    index = row * 256 + digit
                    cute.arch.atomic_add(sums.iterator + index, total_s, scope="gpu")
                    cute.arch.atomic_add(counts.iterator + index, total_c, scope="gpu")
                    cute.arch.atomic_min(minima.iterator + index, low, scope="gpu")

    @cute.kernel
    def finish(self, p, g):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        allocator = cutlass.utils.SmemAllocator()
        warp_totals = allocator.allocate_tensor(Float64, cute.make_layout(4))
        int_reduce = allocator.allocate_tensor(
            Int32, cute.make_ordered_layout((1, (4, 1), 2), order=(1, 0, 2))
        )
        float_reduce = allocator.allocate_tensor(Float64, cute.make_layout((1, (4, 1))))
        control = cute.make_tensor(p[8], cute.make_layout((g[0], 2), stride=(2, 1)))
        active = control[row, 0] != 0
        cute.arch.barrier()
        if active:
            sums, counts, minima = _histograms(p, g[0])
            remaining = cute.make_tensor(p[9], cute.make_layout(g[0]))
            state = cute.make_tensor(p[2], cute.make_layout((g[0], 3), stride=(3, 1)))
            ps = cute.make_tensor(p[11], cute.make_layout(g[0]))
            prefix = Int32(control[row, 1])
            exponent = prefix >> 23
            first = Int32(self.bins - 1 - tid * self.items)
            count0 = Int32(counts[row, first])
            exp0 = first if cutlass.const_expr(self.stage == 0) else exponent
            mass0 = _bucket_mass(sums[row, first], count0, exp0)
            mass1, count1 = Float64(0), Int32(0)
            if cutlass.const_expr(self.items == 2):
                count1 = Int32(counts[row, first - 1])
                mass1 = _bucket_mass(sums[row, first - 1], count1, exponent)
            inclusive = mass0 + mass1
            for level in cutlass.range_constexpr(5):
                delta = 1 << level
                other = Float64(
                    cute.arch.shuffle_sync_up(inclusive, offset=delta, mask_and_clamp=0)
                )
                if lane >= delta:
                    inclusive += other
            if lane == 31:
                warp_totals[warp] = inclusive
            cute.arch.barrier()
            before_warp, full = Float64(0), Float64(0)
            for index in cutlass.range_constexpr(4):
                full += warp_totals[index]
                if warp > index:
                    before_warp += warp_totals[index]
            previous = Float64(
                cute.arch.shuffle_sync_up(inclusive, offset=1, mask_and_clamp=0)
            )
            before = before_warp + cutlass.select_(lane > 0, previous, Float64(0))
            cdf0, cdf1 = before + mass0, before + mass0 + mass1
            target = Float64(0)
            if cutlass.const_expr(self.stage == 0):
                target = Float64(ps[row]) * full
            else:
                target = Float64(remaining[row])
            rank = Int32(self.bins)
            last = Int32(-1)
            if count0 > 0:
                last = Int32(tid * self.items)
                if cdf0 >= target:
                    rank = Int32(tid * self.items)
            if cutlass.const_expr(self.items == 2):
                if count1 > 0:
                    last = Int32(tid * self.items + 1)
                    if (rank == self.bins) & (cdf1 >= target):
                        rank = Int32(tid * self.items + 1)
            chosen_rank = row_reduce(
                rank,
                cute.ReductionOp.MIN,
                128,
                int_reduce[None, None, 0],
                init_val=Int32(self.bins),
            )
            last_rank = row_reduce(
                last,
                cute.ReductionOp.MAX,
                128,
                int_reduce[None, None, 1],
                init_val=Int32(-1),
            )
            chosen_rank = Int32(
                cutlass.select_(chosen_rank < self.bins, chosen_rank, last_rank)
            )
            local_before = Float64(0)
            if chosen_rank == tid * self.items:
                local_before = before
            if cutlass.const_expr(self.items == 2):
                if chosen_rank == tid * self.items + 1:
                    local_before = cdf0
            chosen_before = row_reduce(
                local_before,
                cute.ReductionOp.ADD,
                128,
                float_reduce,
                init_val=Float64(0),
            )
            if tid == 0:
                selected = Int32(self.bins - 1 - chosen_rank)
                low = Int32(minima[row, selected])
                count = Int32(counts[row, selected])
                # One exponent per bin means every mantissa is at least
                # the minimum; equality proves all keys are identical.
                uniform = (count > 0) & (
                    sums[row, selected] == Int64(count) * Int64(low & Int32(0x7FFFFF))
                )
                if uniform | cutlass.const_expr(self.stage == 3):
                    state[row, 1] = low.bitcast(Float32)
                    control[row, 0] = 0
                else:
                    control[row, 1] = prefix | (selected << self.shift)
                    remaining[row] = max(Float64(0), target - chosen_before)
            cute.arch.barrier()
            if cutlass.const_expr(self.stage < 3):
                for offset in cutlass.range_constexpr(2):
                    slot = tid + offset * 128
                    sums[row, slot], counts[row, slot] = Int64(0), Int32(0)
                    minima[row, slot] = Int32(2147483647)


def _build(stamp, dtype, vocab, stage, kind):
    types = (
        dtype,
        Float32,
        Float32,
        Uint8,
        Uint8,
        Int64,
        Int32,
        Int32,
        Int32,
        Float64,
        Float32,
        Float32,
    )
    pointers = tuple(_ptr(dtype) for dtype in types)
    return cute.compile(
        _Nucleus(vocab, stage, kind),
        pointers,
        (Int64(0),) * 6,
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=None)
def _context_cache(context):
    return jit_cache(_build)


def nucleus_cutoffs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    state: torch.Tensor,
    active_k: torch.Tensor,
    active_p: torch.Tensor,
    ps: torch.Tensor,
) -> None:
    """Write inclusive top-p cutoffs into state[:, 1], preserving raw top-k.

    Policies and geometry are runtime arguments. Only dtype and vocabulary
    specialize kernels, and the persistent cache includes source and ABI.
    """
    rows, vocab = logits.shape
    device = logits.device
    mass_parts = (vocab + 1023) // 1024
    histogram_parts = min(
        (vocab + 511) // 512,
        128,
        max(
            32,
            2 * torch.cuda.get_device_properties(device).multi_processor_count // rows,
        ),
    )
    sums = torch.empty((rows, 256), dtype=torch.int64, device=device)
    counts = torch.empty((rows, 256), dtype=torch.int32, device=device)
    minima = torch.empty((rows, 256), dtype=torch.int32, device=device)
    control = torch.empty((rows, 2), dtype=torch.int32, device=device)
    remaining = torch.empty(rows, dtype=torch.float64, device=device)
    guard_mass = torch.empty((rows, mass_parts, 2), dtype=torch.float32, device=device)
    _joint_mass.run(
        logits,
        temperatures,
        state,
        active_k,
        active_p,
        guard_mass,
        logits.stride(0),
        logits.stride(1),
        temperatures.stride(0),
        vocab,
        mass_parts,
        1024,
        grid=(rows, mass_parts),
        warmup=False,
        num_warps=4,
    )
    values = (
        logits,
        temperatures,
        state,
        active_k,
        active_p,
        sums,
        counts,
        minima,
        control,
        remaining,
        guard_mass,
        ps,
    )
    pointers = tuple(_ptr(_TYPES[value.dtype], value.data_ptr()) for value in values)
    geometry = (
        rows,
        histogram_parts,
        logits.stride(0),
        logits.stride(1),
        temperatures.stride(0),
        mass_parts,
    )
    with torch.cuda.device(device):
        status, handle = cuda.cuCtxGetCurrent()
        assert status == cuda.CUresult.CUDA_SUCCESS
        compile = _context_cache((torch.cuda.current_device(), int(handle)))
        stamp = (
            _SOURCE,
            torch.cuda.get_device_capability(device),
            os.environ.get("CUTE_DSL_ARCH"),
            _ABI,
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compile(stamp, _TYPES[logits.dtype], vocab, 0, "init")(
            pointers, geometry, stream
        )
        for stage in range(4):
            compile(stamp, _TYPES[logits.dtype], vocab, stage, "hist")(
                pointers, geometry, stream
            )
            compile(stamp, _TYPES[logits.dtype], vocab, stage, "finish")(
                pointers, geometry, stream
            )
