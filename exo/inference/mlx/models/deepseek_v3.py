from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v3 import ModelArgs, DeepseekV3DecoderLayer
from .base import IdentityBlock
from exo.inference.shard import Shard


@dataclass
class ModelArgs(ModelArgs):
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

    def __post_init__(self):
        if isinstance(self.shard, Shard):
            return
        if not isinstance(self.shard, dict):
            raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

        self.shard = Shard(**self.shard)


class DeepseekV3Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        if self.args.shard.is_first_layer():
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = []
        for i in range(self.num_hidden_layers):
            if self.args.shard.start_layer <= i <= self.args.shard.end_layer:
                self.layers.append(DeepseekV3DecoderLayer(config, i))
            else:
                self.layers.append(IdentityBlock())

        if self.args.shard.is_last_layer():
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        if self.args.shard.is_first_layer():
            h = self.embed_tokens(x)
        else:
            h = x

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None]*len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
            # Protect against overflow for fp16
            if h.dtype == mx.float16:
                h = mx.clip(h, a_min=None, a_max=mx.finfo(mx.float16).max - 1000)

        if self.args.shard.is_last_layer():
            h = self.norm(h)
        return h


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV3Model(config)
        if self.args.shard.is_last_layer():
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[KVCache] = None,
    ):
        out = self.model(inputs, cache)
        if self.args.shard.is_last_layer():
            return self.lm_head(out)
        return out

    def sanitize(self, weights):
        shard_state_dict = {}

        for key, value in weights.items():
            if key.startswith('model.layers.'):
                layer_num = int(key.split('.')[2])
                if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
                    shard_state_dict[key] = value
            elif self.args.shard.is_first_layer() and key.startswith('model.embed_tokens'):
                shard_state_dict[key] = value
            elif self.args.shard.is_last_layer() and (key.startswith('model.norm') or key.startswith('lm_head')):
                shard_state_dict[key] = value

        return shard_state_dict
