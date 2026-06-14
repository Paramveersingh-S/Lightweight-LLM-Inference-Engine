import glob
import torch
import safetensors.torch
from ..model.llama import LlamaModel, LlamaConfig

def convert_hf_to_lite_llama(hf_state: dict, config: LlamaConfig) -> dict:
    """
    Converts HuggingFace LLaMA weights into lite_llama format.
    Handles weight transposition and specific naming conventions.
    """
    our_state = {}
    
    if "model.embed_tokens.weight" in hf_state:
        our_state["embed_tokens.weight"] = hf_state["model.embed_tokens.weight"]
    
    if "model.norm.weight" in hf_state:
        our_state["norm.weight"] = hf_state["model.norm.weight"]
        
    if "lm_head.weight" in hf_state:
        our_state["lm_head.weight"] = hf_state["lm_head.weight"]

    for i in range(config.num_hidden_layers):
        prefix_hf = f"model.layers.{i}."
        prefix_our = f"layers.{i}."
        
        # Self attention
        if f"{prefix_hf}self_attn.q_proj.weight" in hf_state:
            our_state[f"{prefix_our}self_attn.q_proj.weight"] = hf_state[f"{prefix_hf}self_attn.q_proj.weight"]
        if f"{prefix_hf}self_attn.k_proj.weight" in hf_state:
            our_state[f"{prefix_our}self_attn.k_proj.weight"] = hf_state[f"{prefix_hf}self_attn.k_proj.weight"]
        if f"{prefix_hf}self_attn.v_proj.weight" in hf_state:
            our_state[f"{prefix_our}self_attn.v_proj.weight"] = hf_state[f"{prefix_hf}self_attn.v_proj.weight"]
        if f"{prefix_hf}self_attn.o_proj.weight" in hf_state:
            our_state[f"{prefix_our}self_attn.o_proj.weight"] = hf_state[f"{prefix_hf}self_attn.o_proj.weight"]
            
        # MLP
        if f"{prefix_hf}mlp.gate_proj.weight" in hf_state:
            our_state[f"{prefix_our}mlp.gate_proj.weight"] = hf_state[f"{prefix_hf}mlp.gate_proj.weight"]
        if f"{prefix_hf}mlp.up_proj.weight" in hf_state:
            our_state[f"{prefix_our}mlp.up_proj.weight"] = hf_state[f"{prefix_hf}mlp.up_proj.weight"]
        if f"{prefix_hf}mlp.down_proj.weight" in hf_state:
            our_state[f"{prefix_our}mlp.down_proj.weight"] = hf_state[f"{prefix_hf}mlp.down_proj.weight"]
            
        # Norms
        if f"{prefix_hf}input_layernorm.weight" in hf_state:
            our_state[f"{prefix_our}input_layernorm.weight"] = hf_state[f"{prefix_hf}input_layernorm.weight"]
        if f"{prefix_hf}post_attention_layernorm.weight" in hf_state:
            our_state[f"{prefix_our}post_attention_layernorm.weight"] = hf_state[f"{prefix_hf}post_attention_layernorm.weight"]

    return our_state

class LlamaCheckpointLoader:
    @staticmethod
    def load(model_path: str, config: LlamaConfig, dtype=torch.float16) -> LlamaModel:
        import os
        if not os.path.exists(model_path):
            try:
                from huggingface_hub import snapshot_download
                print(f"Attempting to download weights for {model_path} from HuggingFace Hub...")
                model_path = snapshot_download(model_path, allow_patterns=["*.safetensors", "*.bin", "config.json"])
            except ImportError:
                print("huggingface_hub not installed. Cannot download weights automatically.")
            except Exception as e:
                print(f"Failed to download from HF hub: {e}")

        # Update config with actual HF config.json
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                hf_config = json.load(f)
            config.hidden_size = hf_config.get("hidden_size", config.hidden_size)
            config.num_attention_heads = hf_config.get("num_attention_heads", config.num_attention_heads)
            config.num_key_value_heads = hf_config.get("num_key_value_heads", config.num_key_value_heads)
            config.num_hidden_layers = hf_config.get("num_hidden_layers", config.num_hidden_layers)
            config.intermediate_size = hf_config.get("intermediate_size", config.intermediate_size)
            config.vocab_size = hf_config.get("vocab_size", config.vocab_size)
            config.rms_norm_eps = hf_config.get("rms_norm_eps", config.rms_norm_eps)
            config.max_position_embeddings = hf_config.get("max_position_embeddings", config.max_position_embeddings)


        # Initialize model directly on GPU to save CPU RAM
        print("Initializing empty model on GPU...")
        with torch.device('cuda'):
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            model = LlamaModel(config)
            torch.set_default_dtype(original_dtype)

        shards = glob.glob(f"{model_path}/*.safetensors")
        if not shards:
            shards = glob.glob(f"{model_path}/pytorch_model*.bin")
            
        if not shards:
            raise FileNotFoundError(f"No checkpoint files found in {model_path}")

        import gc
        print(f"Loading weights shard by shard to prevent Colab RAM peaking...")
        for i, shard in enumerate(shards):
            print(f"  Loading shard {i+1}/{len(shards)}...")
            if shard.endswith(".safetensors"):
                shard_state = safetensors.torch.load_file(shard)
            else:
                shard_state = torch.load(shard, map_location="cpu")
                
            mapped_state = convert_hf_to_lite_llama(shard_state, config)
            model.load_state_dict(mapped_state, strict=False)
            
            del shard_state
            del mapped_state
            gc.collect()

        return model
