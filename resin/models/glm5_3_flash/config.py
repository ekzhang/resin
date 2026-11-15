"""Validated geometry for GLM-5.3-Flash's hybrid KDA/DSA architecture."""

from dataclasses import dataclass, fields
from math import isfinite

from resin.models.hf_config import PositiveInt, invalid_dimensions


@dataclass(frozen=True, slots=True)
class Glm53Config:
    hidden_size: PositiveInt = 4096
    num_hidden_layers: PositiveInt = 45
    vocab_size: PositiveInt = 154880
    intermediate_size: PositiveInt = 12288
    moe_intermediate_size: PositiveInt = 2048
    n_routed_experts: PositiveInt = 288
    n_shared_experts: PositiveInt = 1
    num_experts_per_tok: PositiveInt = 8
    first_k_dense_replace: int = 3
    num_attention_heads: PositiveInt = 64
    q_lora_rank: PositiveInt = 1536
    kv_lora_rank: PositiveInt = 512
    qk_head_dim: PositiveInt = 256
    qk_nope_head_dim: PositiveInt = 256
    qk_rope_head_dim: int = 0
    v_head_dim: PositiveInt = 256
    linear_num_heads: PositiveInt = 64
    linear_head_dim: PositiveInt = 128
    linear_conv_kernel_dim: PositiveInt = 4
    linear_lower_bound: float = -5.0
    index_n_heads: PositiveInt = 32
    index_head_dim: PositiveInt = 128
    index_topk: PositiveInt = 2048
    index_kpool: PositiveInt = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True
    index_share_for_mtp_iteration: bool = True
    hc_mult: PositiveInt = 4
    hc_sinkhorn_iters: PositiveInt = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-5
    routed_scaling_factor: float = 2.5
    swiglu_limit: float = 10.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    max_position_embeddings: PositiveInt = 1048576
    num_nextn_predict_layers: int = 1
    layer_types: tuple[str, ...] = tuple(
        "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
        for i in range(45)
    )
    eos_token_id: tuple[int, ...] = (154820, 154827, 154829)

    @classmethod
    def from_dict(cls, raw):
        if raw.get("model_type") not in {"glm5_next", "glm5_next_text"}:
            raise ValueError("expected a GLM-5.3-Flash glm5_next checkpoint")
        text = raw.get("text_config", raw)
        if not text.get("mhc", True) or not text.get("mla_use_nope", True):
            raise ValueError("GLM-5.3-Flash requires mHC and NoPE MLA")
        if text.get("scoring_func", "sigmoid") != "sigmoid":
            raise ValueError("GLM router requires sigmoid scoring")
        if text.get("hidden_act", "silu") != "silu" or text.get(
            "attention_bias", False
        ):
            raise ValueError(
                "GLM requires SwiGLU and attention without projection bias"
            )
        allowed = {field.name for field in fields(cls)}
        values = {key: value for key, value in text.items() if key in allowed}
        for name in ("layer_types", "eos_token_id"):
            if name in values:
                value = values[name]
                values[name] = (value,) if isinstance(value, int) else tuple(value)
        result = cls(**values)
        result.validate()
        mlp_types = text.get("mlp_layer_types")
        if mlp_types is not None and tuple(mlp_types) != tuple(
            "dense" if layer < result.first_k_dense_replace else "sparse"
            for layer in range(result.num_hidden_layers)
        ):
            raise ValueError("mlp_layer_types disagrees with the dense layer count")
        if any(kind != "full" for kind in text.get("indexer_types", ())):
            raise ValueError("only the full GLM sparse indexer is supported")
        return result

    def validate(self):
        for name in invalid_dimensions(self, strict=True):
            raise ValueError(f"{name} must be a positive integer")
        if len(self.layer_types) != self.num_hidden_layers or any(
            kind not in {"linear_attention", "deepseek_sparse_attention"}
            for kind in self.layer_types
        ):
            raise ValueError("layer_types must describe every KDA/DSA layer")
        if self.qk_rope_head_dim or self.qk_head_dim != self.qk_nope_head_dim:
            raise ValueError("only NoPE MLA is supported")
        if self.n_group != 1 or self.topk_group != 1:
            raise ValueError("GLM-5.3-Flash expects one routing group")
        if not 0 <= self.first_k_dense_replace <= self.num_hidden_layers:
            raise ValueError("invalid dense layer count")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("top-k exceeds the expert count")
        if self.hidden_size % 16 or self.moe_intermediate_size % 16:
            raise ValueError("NVFP4 dimensions must align to 16 elements")
        for name in ("rms_norm_eps", "hc_eps", "routed_scaling_factor", "swiglu_limit"):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.linear_lower_bound) or self.linear_lower_bound >= 0:
            raise ValueError("linear_lower_bound must be finite and negative")

    @property
    def kda_layers(self):
        return tuple(
            i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"
        )

    @property
    def dsa_layers(self):
        return tuple(
            i
            for i, kind in enumerate(self.layer_types)
            if kind == "deepseek_sparse_attention"
        )

    @property
    def linear_dim(self):
        return self.linear_num_heads * self.linear_head_dim


__all__ = ["Glm53Config"]
