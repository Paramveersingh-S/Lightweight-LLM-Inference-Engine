import torch
from typing import List
from .model.llama import LlamaModel, LlamaConfig
from .model.attention import ForwardBatch
from .loader.checkpoint import LlamaCheckpointLoader

class EngineConfig:
    def __init__(self, **kwargs):
        self.model_path = kwargs.get("model_path", "meta-llama/Llama-3.2-3B")
        self.dtype = kwargs.get("dtype", "float16")
        self.max_batch_size = kwargs.get("max_batch_size", 32)
        self.max_seq_len = kwargs.get("max_seq_len", 8192)
        self.temperature = kwargs.get("temperature", 0.8)
        self.top_p = kwargs.get("top_p", 0.95)

class NaiveKVCache:
    """Pre-allocated KV cache similar to standard Transformers for Phase 3."""
    def __init__(self, config: LlamaConfig, max_batch_size: int, max_seq_len: int, dtype=torch.float16):
        self.k_cache = torch.zeros(
            config.num_hidden_layers, max_batch_size, config.num_key_value_heads, max_seq_len, config.hidden_size // config.num_attention_heads,
            dtype=dtype, device='cuda'
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.seq_lengths = torch.zeros(max_batch_size, dtype=torch.long, device='cuda')

    def write(self, layer_idx, batch_idx, start_pos, k, v):
        seq_len = k.shape[2]
        self.k_cache[layer_idx, batch_idx, :, start_pos:start_pos+seq_len, :] = k
        self.v_cache[layer_idx, batch_idx, :, start_pos:start_pos+seq_len, :] = v

class LLMEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.llama_config = LlamaConfig()
        
        try:
            self.model = LlamaCheckpointLoader.load(config.model_path, self.llama_config, dtype=getattr(torch, config.dtype))
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Could not load real weights ({e}). Initializing random dummy weights on GPU to avoid CPU RAM peaking.")
            # Set default tensor type to avoid CPU RAM OOM during dummy init
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(getattr(torch, config.dtype))
            try:
                # Initialize directly on GPU
                with torch.device('cuda'):
                    self.model = LlamaModel(self.llama_config)
            finally:
                torch.set_default_dtype(original_dtype)
            
        self.model.eval()
        self.kv_cache = NaiveKVCache(self.llama_config, config.max_batch_size, config.max_seq_len)

    def generate(self, prompts: List[str], max_tokens: int = 512, greedy: bool = False):
        batch_size = len(prompts)
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device='cuda')
        positions = torch.arange(0, seq_len, device='cuda').unsqueeze(0).repeat(batch_size, 1)
        
        # Prefill phase
        batch = ForwardBatch(input_ids, positions, kv_cache=self.kv_cache, has_prefill=True)
        
        with torch.no_grad():
            logits = self.model(batch)
            
            # Decode loop
            for i in range(max_tokens - 1):
                next_token = torch.randint(0, 1000, (batch_size, 1), device='cuda')
                next_pos = torch.full((batch_size, 1), seq_len + i, device='cuda')
                decode_batch = ForwardBatch(next_token, next_pos, kv_cache=self.kv_cache, has_decode=True)
                logits = self.model(decode_batch)
            
        return ["Dummy output" for _ in prompts]
