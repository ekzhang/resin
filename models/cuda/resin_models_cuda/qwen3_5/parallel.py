"""Checkpoint preparation for Qwen's full-attention and DeltaNet head shards."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from resin.backend.cuda.parallel import (
    TensorParallelContext,
    TensorParallelLayout,
    TensorShard,
)
from resin.models.qwen3_5.config import Qwen35Config
from resin_models_cuda.dense_parallel import TensorParallelSum, shard_linear
from resin_models_cuda.qwen3_5.model import (
    Qwen35Attention,
    Qwen35ForConditionalGeneration,
    Qwen35GatedDeltaNet,
)
from resin_models_cuda.qwen3_5.quantization import ModelOptFp8Linear


def shard_attention_config(
    config: Qwen35Config, layout: TensorParallelLayout
) -> Qwen35Config:
    """Partition complete KV and recurrent heads; never replicate missing shards."""
    counts = {
        name: layout.even_shard(getattr(config, name), name=name).size
        for name in (
            "num_attention_heads",
            "num_key_value_heads",
            "linear_num_key_heads",
            "linear_num_value_heads",
        )
    }
    return replace(config, **counts)


def _projection(
    source: nn.Module,
    shard: TensorShard,
    parallel: TensorParallelContext,
    *,
    output: bool,
) -> nn.Module:
    if isinstance(source, nn.Linear):
        return shard_linear(source, shard, output=output, rank=parallel.rank)
    if not isinstance(source, ModelOptFp8Linear):
        raise TypeError("Qwen attention TP requires BF16 or ModelOpt FP8 projections")
    with torch.device("meta"):
        local = ModelOptFp8Linear(
            source.in_features if output else shard.size,
            shard.size if output else source.out_features,
        )
    local.weight = source.weight.narrow(
        0 if output else 1, shard.start, shard.size
    ).clone(memory_format=torch.contiguous_format)
    local.input_scale, local.weight_scale = source.input_scale, source.weight_scale
    local.train(source.training)
    return local


def _packed_rows(
    tensor: Tensor, widths: tuple[int, ...], layout: TensorParallelLayout
) -> Tensor:
    """Keep the local portion of each Q/K/V component, preserving their order."""
    return torch.cat(
        [
            component.narrow(0, shard.start, shard.size)
            for component, shard in zip(
                tensor.split(widths, dim=0),
                (
                    layout.even_shard(width, name="packed projection width")
                    for width in widths
                ),
                strict=True,
            )
        ],
        dim=0,
    )


def _full_attention(module: Qwen35Attention, parallel: TensorParallelContext) -> None:
    config = module.config
    layout = parallel.layout
    # q_proj stores [head, query/gate, channel], so contiguous head intervals
    # retain each head's query and output gate together.
    for name, width in (
        ("q_proj", config.num_attention_heads * config.head_dim * 2),
        ("k_proj", config.num_key_value_heads * config.head_dim),
        ("v_proj", config.num_key_value_heads * config.head_dim),
    ):
        setattr(
            module,
            name,
            _projection(
                getattr(module, name),
                layout.even_shard(width, name=name),
                parallel,
                output=True,
            ),
        )
    module.o_proj = TensorParallelSum(
        _projection(
            module.o_proj,
            layout.even_shard(
                config.num_attention_heads * config.head_dim,
                name="attention output width",
            ),
            parallel,
            output=False,
        ),
        parallel,
    )
    module.config = shard_attention_config(config, layout)


def _delta_net(module: Qwen35GatedDeltaNet, parallel: TensorParallelContext) -> None:
    config, layout = module.config, parallel.layout
    widths = (config.linear_key_dim, config.linear_key_dim, config.linear_value_dim)
    projection = module.in_proj_qkv
    if not isinstance(projection, (nn.Linear, ModelOptFp8Linear)):
        raise TypeError("Qwen DeltaNet TP requires BF16 or ModelOpt FP8 projections")
    weight = _packed_rows(projection.weight, widths, layout)
    if isinstance(projection, nn.Linear):
        projection.weight = nn.Parameter(
            weight, requires_grad=projection.weight.requires_grad
        )
    else:
        projection.weight = weight
    projection.out_features = weight.shape[0]
    for name, width in (
        ("in_proj_z", config.linear_value_dim),
        ("in_proj_a", config.linear_num_value_heads),
        ("in_proj_b", config.linear_num_value_heads),
    ):
        setattr(
            module,
            name,
            _projection(
                getattr(module, name),
                layout.even_shard(width, name=name),
                parallel,
                output=True,
            ),
        )
    local = shard_attention_config(config, layout)
    conv = module.conv1d
    conv.weight = nn.Parameter(
        _packed_rows(conv.weight, widths, layout),
        requires_grad=conv.weight.requires_grad,
    )
    conv.in_channels = conv.out_channels = conv.groups = local.linear_conv_dim
    value_heads = layout.even_shard(
        config.linear_num_value_heads, name="linear value heads"
    )
    for name in ("A_log", "dt_bias"):
        weight = getattr(module, name)
        setattr(
            module,
            name,
            nn.Parameter(
                weight[value_heads.as_slice()].clone(),
                requires_grad=weight.requires_grad,
            ),
        )
    module.out_proj = TensorParallelSum(
        _projection(
            module.out_proj,
            layout.even_shard(config.linear_value_dim, name="DeltaNet output width"),
            parallel,
            output=False,
        ),
        parallel,
    )
    module.config = local
    module.key_dim, module.value_dim, module.conv_dim = (
        local.linear_key_dim,
        local.linear_value_dim,
        local.linear_conv_dim,
    )


def prepare_attention_parallel(
    model: Qwen35ForConditionalGeneration, parallel: TensorParallelContext
) -> None:
    """Shard loaded target and MTP projections before any graph or KV allocation.

    The checkpoint config remains global on model.config. Execution modules own
    their local head geometry; hidden width and normalization weights are shared.
    """
    if not parallel.enabled:
        return
    local = shard_attention_config(model.config, parallel.layout)
    for module in tuple(model.modules()):
        if isinstance(module, Qwen35Attention):
            _full_attention(module, parallel)
        elif isinstance(module, Qwen35GatedDeltaNet):
            _delta_net(module, parallel)
    model.language_model.config = local
    if model.mtp is not None:
        model.mtp.config = shard_attention_config(model.mtp.config, parallel.layout)
