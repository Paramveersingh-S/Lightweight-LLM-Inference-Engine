import torch
import torch.nn as nn
from .attention import LlamaAttention, ForwardBatch
from .mlp import LlamaMLP
from .norm import RMSNorm

class LlamaConfig:
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.intermediate_size = kwargs.get("intermediate_size", 11008)
        self.vocab_size = kwargs.get("vocab_size", 128256) # LLaMA 3
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-5)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8192)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, batch: ForwardBatch):
        # Self Attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, batch)
        x = residual + x
        
        # Fully Connected
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, batch: ForwardBatch):
        x = self.embed_tokens(batch.input_ids)
        for layer in self.layers:
            x = layer(x, batch)
        x = self.norm(x)
        return x
