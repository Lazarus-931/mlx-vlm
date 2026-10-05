from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5Model,
)
from .config import ModelConfig, TextConfig


class AgnesMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, parallel_hidden_dim: int = 0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.parallel_ffn = (
            Qwen3_5MLP(dim, parallel_hidden_dim) if parallel_hidden_dim else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        y = self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
        if self.parallel_ffn is not None:
            y = y + self.parallel_ffn(x)
        return y


class AgnesDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        if args.layer_types is not None:
            self.is_linear = args.layer_types[layer_idx] == "agnes_delta_attention"
        else:
            self.is_linear = (layer_idx + 1) % args.global_attention_interval != 0

        # Submodule names stay linear_attn/self_attn (reused qwen3_5 machinery
        # indexes them by those names); the Agnes delta_attn/global_attn keys are
        # remapped in sanitize.
        if self.is_linear:
            self.linear_attn = Qwen3_5GatedDeltaNet(args)
        else:
            self.self_attn = Qwen3_5Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = AgnesMLP(
            args.hidden_size,
            args.intermediate_size,
            args.parallel_ffn_intermediate_size,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(
                self.input_layernorm(x),
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class AgnesModel(Qwen3_5Model):
    def __init__(self, args: TextConfig):
        nn.Module.__init__(self)
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            AgnesDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = next((i for i, l in enumerate(self.layers) if l.is_linear), 0)
        self.fa_idx = next((i for i, l in enumerate(self.layers) if not l.is_linear), 0)


class LanguageModel(Qwen3_5LanguageModel):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = AgnesModel(args)
        self._position_ids = None
        self._rope_deltas = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
