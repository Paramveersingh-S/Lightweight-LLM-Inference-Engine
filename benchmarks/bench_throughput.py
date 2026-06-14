import time
import torch
from lite_llama.engine import LLMEngine, EngineConfig

def run_throughput_benchmark():
    print("Running Throughput Benchmark Across Batch Sizes...")
    MODEL = "unsloth/Llama-3.2-3B-Instruct"  # Safe default to avoid Auth issues on auto-runs
    config = EngineConfig(model_path=MODEL, max_batch_size=8)
    engine = LLMEngine(config)
    
    batch_sizes = [1, 2, 4, 8]
    throughputs = []
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
        tps = total_tokens/elapsed
        throughputs.append(tps)
        print(f"Batch Size: {bs:2d} | Tokens: {total_tokens:4d} | Time: {elapsed:.2f}s | Throughput: {tps:.2f} tokens/sec")

    generate_throughput_graph(batch_sizes, throughputs)

def generate_throughput_graph(batch_sizes, throughputs):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        os.makedirs("docs", exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=batch_sizes, y=throughputs, marker='o', color='b', linewidth=2.5)
        plt.title('Token Throughput vs Batch Size', fontsize=16)
        plt.xlabel('Batch Size', fontsize=12)
        plt.ylabel('Tokens / Second', fontsize=12)
        plt.xticks(batch_sizes)
        plt.savefig('docs/throughput_scaling.png')
        print("Graph saved to docs/throughput_scaling.png")
    except ImportError:
        pass

if __name__ == "__main__":
    run_throughput_benchmark()
