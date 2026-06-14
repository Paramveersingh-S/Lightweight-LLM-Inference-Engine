import time
import torch
from lite_llama.engine import LLMEngine, EngineConfig

def run_ttft_benchmark():
    print("Running Time-To-First-Token (TTFT) Benchmark...")
    MODEL = "unsloth/Llama-3.2-3B-Instruct"
    config = EngineConfig(model_path=MODEL, max_batch_size=1)
    engine = LLMEngine(config)
    
    # Approx 2048 tokens context
    prompt = "Transformers are amazing " * 1000 
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # Just generate 1 token to measure prefill phase
    engine.generate([prompt], max_tokens=1, greedy=True)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"TTFT (2048 prompt): {elapsed * 1000:.2f} ms")

if __name__ == "__main__":
    run_ttft_benchmark()
