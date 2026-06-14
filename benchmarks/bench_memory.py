import torch
from lite_llama.engine import LLMEngine, EngineConfig

def run_memory_benchmark():
    print("Running Memory Overhead Benchmark...")
    MODEL = "unsloth/Llama-3.2-3B-Instruct"
    config = EngineConfig(model_path=MODEL, max_batch_size=1)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    engine = LLMEngine(config)
    allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f"Peak VRAM Allocated: {allocated_mb:.2f} MB")
    print("This includes model weights + dynamic token KV cache blocks.")

if __name__ == "__main__":
    run_memory_benchmark()
