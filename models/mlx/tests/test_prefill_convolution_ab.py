"""One-off paired serial/tiled measurements on the Apple CI GPU."""

import statistics
import time
from functools import cache

import mlx.core as mx

from resin_models_mlx.inkling2.kernels import short_convolution as tiled
from resin_models_mlx.metal import constant

_CONVOLUTION_SOURCE = r"""
    // One thread per (row, channel). State holds three raw inputs, oldest first.
    const uint index = thread_position_in_grid.x;
    const uint channel = index % C;
    const uint row = index / C;
    if (row >= uint(geometry[0])) return;
    const int tokens = geometry[1];
    float history[3];
    for (int i = 0; i < 3; ++i)
        history[i] = float(state[(ulong(row) * 3 + i) * C + channel]);
    for (int t = 0; t < tokens; ++t) {
        const ulong at = (ulong(row) * tokens + t) * C + channel;
        const float current = float(x[at]);
        float output = 0.0f;
        if (DYNAMIC) {
            // Tap zero is the current token; coefficients stay FP32.
            const uint head = channel / (C / HEADS);
            const ulong taps = ((ulong(row) * tokens + t) * HEADS + head) * 4;
            const float values[4] = {current, history[2], history[1], history[0]};
            for (int tap = 0; tap < 4; ++tap) {
                const float coefficient =
                    float(weight[tap * C + channel]) + float(dynamic[taps + tap]);
                output = metal::fma(values[tap], coefficient, output);
            }
        } else {
            // Chronological taps rounded to the input dtype; tap three is current.
            const float values[4] = {history[0], history[1], history[2], current};
            for (int tap = 0; tap < 4; ++tap) {
                const float coefficient = float(T(weight[channel * 4 + tap]));
                output = metal::fma(values[tap], coefficient, output);
            }
        }
        // FP32 roundings of 1 / sqrt(9) and 1 / sqrt(5), as the reference uses.
        const float scale = DYNAMIC ? 0.3333333432674408f : 0.4472135901451111f;
        y[at] = T((output + current) * scale);
        history[0] = history[1];
        history[1] = history[2];
        history[2] = current;
    }
    for (int i = 0; i < 3; ++i)
        next_state[(ulong(row) * 3 + i) * C + channel] = T(history[i]);
"""


@cache
def _convolution_kernel():
    return mx.fast.metal_kernel(
        name="resin_inkling2_short_convolution_serial",
        input_names=["x", "state", "weight", "dynamic", "geometry"],
        output_names=["y", "next_state"],
        source=_CONVOLUTION_SOURCE,
    )


def serial(
    x: mx.array,
    state: mx.array,
    weight: mx.array,
    dynamic: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Four residual taps over three raw history inputs; output and new history.

    ``x`` is ``[B, T, C]`` and ``state`` ``[B, 3, C]`` (oldest first), both BF16.
    Static ``weight`` is ``[C, 4]`` in chronological order; dynamic convolutions
    pass FP32 ``weight [4, C]`` (tap zero current) and ``dynamic [B, T, H, 4]``.
    """
    batch, tokens, channels = x.shape
    is_dynamic = dynamic is not None
    heads = dynamic.shape[-2] if dynamic is not None else 1
    if dynamic is None:
        dynamic = constant((), "float32")
    geometry = constant((batch, tokens), "int32")
    threads = batch * channels
    return tuple(
        _convolution_kernel()(
            inputs=[x, state, weight, dynamic, geometry],
            template=[
                ("C", channels),
                ("HEADS", heads),
                ("DYNAMIC", is_dynamic),
                ("T", x.dtype),
            ],
            grid=(-(-threads // 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape, (batch, 3, channels)],
            output_dtypes=[x.dtype, x.dtype],
        )
    )  # type: ignore[return-value]


def test_prefill_convolution_ab(capsys):
    with capsys.disabled():
        print(mx.device_info(), flush=True)
        mx.random.seed(0)
        for batch, tokens in [(1, 1), (8, 1), (1, 128), (1, 2048), (1, 8192)]:
            for width, dynamic in [(1024, False), (1024, True), (2304, False)]:
                bf16 = mx.bfloat16
                x = mx.random.normal((batch, tokens, width)).astype(bf16)
                state = mx.random.normal((batch, 3, width)).astype(bf16)
                if dynamic:
                    weight = mx.random.normal((4, width))
                    taps = mx.random.normal((batch, tokens, 8, 4)).astype(bf16)
                    mx.eval(taps)
                else:
                    weight = mx.random.normal((width, 4)).astype(bf16)
                    taps = None
                mx.eval(x, state, weight)
                inputs = x, state, weight, taps
                a, b = serial(*inputs), tiled(*inputs)
                mx.eval(*a, *b)
                assert all(
                    mx.array_equal(v.view(mx.uint16), w.view(mx.uint16)).item()
                    for v, w in zip(a, b, strict=True)
                )
                times = {f: [] for f in (serial, tiled)}
                for i in range(7):
                    for f in (tiled, serial) if i % 2 else (serial, tiled):
                        begin = time.perf_counter_ns()
                        mx.eval(*f(*inputs))
                        times[f].append((time.perf_counter_ns() - begin) / 1e6)
                old, new = (statistics.median(times[f]) for f in (serial, tiled))
                print(
                    f"CONV_AB {batch=} {tokens=} {width=} {dynamic=} {old:.3f} -> {new:.3f} ms, {old / new:.2f}x",
                    flush=True,
                )
