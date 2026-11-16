"""Small native-weight fixtures and independent packed expert arithmetic."""

from types import SimpleNamespace

import torch


def make_experts(mode):
    from resin_models_cuda.inkling2.experts import Inkling2Experts

    config = SimpleNamespace(
        hidden_size=256,
        intermediate_size=128,
        num_routed_experts=4,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
    )
    precision = {"w13": mode, "w2": mode}
    module = Inkling2Experts(
        config, precision=precision, expert_backend="cute-dsl"
    ).cuda()
    for name in precision:
        weight = getattr(module, name + "_weight")
        if mode == "nvfp4":
            weight.random_(0, 256)
            getattr(module, name + "_weight_scale").fill_(0.25)
            getattr(module, name + "_weight_amax").uniform_(0.5, 1)
            getattr(module, name + "_weight_outer_scale_denominator").fill_(16)
        else:
            weight.copy_(torch.randn(weight.shape, device="cuda").to(weight.dtype))
            getattr(module, name + "_weight_scale").fill_(123)
    setattr(
        module,
        "_reference_scales",
        {name: getattr(module, name + "_weight_scale").clone() for name in precision},
    )
    module.prepare()
    return module


def _fp4(x):
    levels = x.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    distance = (x.abs()[..., None] - levels).abs()
    # IEEE ties choose the even encoding, including boundaries 0.25 and 5.
    nearest = distance.min(-1, keepdim=True).values
    codes = torch.arange(8, device=x.device)
    priority = torch.where(codes % 2 == 0, codes, codes + 8)
    index = torch.where(distance == nearest, priority, 100).argmin(-1)
    index = index | (torch.signbit(x).long() << 3)
    return levels[index & 7] * torch.where(index & 8 != 0, -1, 1), index


def nvfp4_reference(x):
    x = x.float()
    amax = x.abs().amax(-1)
    encoder = torch.div(torch.full_like(amax, 1536.0), amax).clamp(
        max=torch.finfo(torch.float32).max
    )
    decoder = torch.div(torch.ones_like(encoder), encoder)
    grouped = x.reshape(*x.shape[:-1], -1, 16)
    maximum = grouped.abs().amax(-1)
    candidates = []
    for limit in (6.0, 4.0):
        scale = (
            (maximum * (encoder[..., None] * (1.0 / limit)))
            .clamp(max=448)
            .to(torch.float8_e4m3fn)
            .float()
        )
        inverse = torch.div(torch.ones_like(scale), scale * decoder[..., None]).clamp(
            max=torch.finfo(torch.float32).max
        )
        values, codes = _fp4(grouped * inverse[..., None])
        reconstruction = (
            values * scale[..., None] * (amax[..., None, None] * (1.0 / 1536.0))
        )
        error = (reconstruction - grouped).abs()
        for _ in range(4):
            pairs = error.reshape(*error.shape[:-1], -1, 2)
            error = pairs[..., 0] + pairs[..., 1]
        error = error.squeeze(-1)
        candidates.append((scale, codes, reconstruction, error))
    a, b = candidates
    choose4 = b[3] < a[3]
    scales = torch.where(choose4, b[0], a[0]).to(torch.float8_e4m3fn)
    codes = torch.where(choose4[..., None], b[1], a[1]).flatten(-2)
    packed = (codes[..., ::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return packed, scales, amax


def dequant(values, scales, amax=None, denominator=1536.0):
    if amax is None:
        factors = torch.exp2(scales.float() - 127)
        factors = torch.where(scales == 0, 2.0**-127, factors)
        return values.float() * factors.repeat_interleave(32, -1)
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=values.device)
    codes = torch.stack((values & 15, values >> 4), -1).flatten(-2).long()
    decoded = table[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    return (
        decoded
        * scales.float().repeat_interleave(16, -1)
        * (amax / denominator)[..., None]
    )


def expert_reference(module, hidden, ids, weights):
    from resin_models_cuda.inkling2.expert_quantization import quantize

    def dequant_activation(x, mode):
        if mode == "bf16":
            return x.float(), None
        values, scales, amax = quantize(x, mode)
        base = (
            dequant(values, scales, torch.ones_like(amax), 1)
            if amax is not None
            else dequant(values, scales)
        )
        return base, amax

    def dequant_weight(name):
        weight = getattr(module, name + "_weight")
        mode = module.precision[name]
        if mode == "bf16":
            return weight.float()
        native_scales = getattr(module, "_reference_scales", None)
        scale = (
            native_scales[name]
            if native_scales is not None
            else getattr(module, name + "_weight_scale")
        )
        if mode == "nvfp4":
            return dequant(
                weight, scale, torch.ones((weight.shape[0], 1), device=weight.device), 1
            )
        return dequant(weight, scale)

    def project(name, weight, x, amax, expert):
        acc = weight[expert] @ x
        if amax is not None:
            # NVFP4 accumulates the block-scaled product first. The two outer
            # scales are multiplied after GEMM, before BF16 rounding (SGLang
            # nvfp4._scale_output / moe._nvfp4_row_scale).
            weight_amax = getattr(module, name + "_weight_amax")[expert]
            denominator = 1536.0 * getattr(
                module, name + "_weight_outer_scale_denominator"
            )
            outer = (amax * weight_amax) * torch.div(
                torch.ones_like(denominator), denominator
            )
            acc = acc * outer
        return acc.bfloat16().float()

    w13, w2 = dequant_weight("w13"), dequant_weight("w2")
    x, x_amax = dequant_activation(hidden, module.precision["w13"])
    out = torch.zeros_like(hidden, dtype=torch.float32)
    for token in range(hidden.shape[0]):
        for k in range(ids.shape[1]):
            expert = int(ids[token, k]) - module.expert_start
            if not 0 <= expert < module.expert_count:
                continue
            projected = project(
                "w13",
                w13,
                x[token],
                x_amax[token] if x_amax is not None else None,
                expert,
            )
            gate, up = projected[::2], projected[1::2]
            activation = (4 * torch.tanh(gate / 4)) * torch.sigmoid(gate)
            activation = (activation * (25 * torch.tanh(up / 25))) * weights[token, k]
            activation, activation_amax = dequant_activation(
                activation.bfloat16()[None], module.precision["w2"]
            )
            out[token] += project(
                "w2",
                w2,
                activation[0],
                activation_amax[0] if activation_amax is not None else None,
                expert,
            )
    return out
