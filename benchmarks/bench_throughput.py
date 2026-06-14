import time
import torch
from lite_llama.engine import LLMEngine, EngineConfig

def run_throughput_benchmark():
    print("Running Throughput Benchmark Across Batch Sizes...")
    MODEL = "unsloth/Llama-3.2-3B-Instruct"  # Safe default to avoid Auth issues on auto-runs
    config = EngineConfig(model_path=MODEL, max_batch_size=8)
    engine = LLMEngine(config)
    
    batch_sizes = [1, 2, 4, 8]
    prompt = "Write a detailed explanation of transformers " * 20
    
    for bs in batch_sizes:
        prompts = [prompt for _ in range(bs)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Generate 100 tokens per sequence
        engine.generate(prompts, max_tokens=100, greedy=True)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        total_tokens = bs * 100
        print(f"Batch Size: {bs:2d} | Tokens: {total_tokens:4d} | Time: {elapsed:.2f}s | Throughput: {total_tokens/elapsed:.2f} tokens/sec")

if __name__ == "__main__":
    run_throughput_benchmark()
