import torch
from .block_manager import KVBlockManager

class TokenAttentionCache:
    def __init__(self, config):
        self.config = config
        # Assuming config has llama_config attribute attached
        try:
            head_dim = config.llama_config.hidden_size // config.llama_config.num_attention_heads
            num_kv_heads = config.llama_config.num_key_value_heads
        except AttributeError:
            head_dim = 128
            num_kv_heads = 8
            
        self.block_manager = KVBlockManager(
            num_blocks=getattr(config, "kv_cache_blocks", 4096),
            block_size=getattr(config, "block_size", 16),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=getattr(torch, config.dtype)
        )
        
    def write(self, k, v, batch):
        pass
