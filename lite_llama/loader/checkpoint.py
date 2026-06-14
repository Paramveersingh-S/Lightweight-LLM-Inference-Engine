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
                model_path = snapshot_download(model_path, allow_patterns=["*.safetensors", "*.bin"])
            except ImportError:
                print("huggingface_hub not installed. Cannot download weights automatically.")
            except Exception as e:
                print(f"Failed to download from HF hub: {e}")

        hf_state = {}
        shards = glob.glob(f"{model_path}/*.safetensors")
        if not shards:
            shards = glob.glob(f"{model_path}/pytorch_model*.bin")
            for shard in shards:
                hf_state.update(torch.load(shard, map_location="cpu"))
        else:
            for shard in shards:
                hf_state.update(safetensors.torch.load_file(shard))
                
        if not hf_state:
            raise FileNotFoundError(f"No checkpoint files found in {model_path}")

        model = LlamaModel(config)
        our_state = convert_hf_to_lite_llama(hf_state, config)
        model.load_state_dict(our_state, strict=False)
        return model.to(dtype).cuda()
