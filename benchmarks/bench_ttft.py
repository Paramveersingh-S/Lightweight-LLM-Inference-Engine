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
    generate_ttft_graph(elapsed * 1000)

def generate_ttft_graph(ttft_ms):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        os.makedirs("docs", exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(6, 6))
        
        models = ['lite_llama']
        latency = [ttft_ms]
        
        ax = sns.barplot(x=models, y=latency, hue=models, legend=False, width=0.4)
        plt.title('Time-To-First-Token (TTFT) - 2048 prompt', fontsize=16)
        plt.ylabel('Latency (ms)', fontsize=12)
        plt.axhline(y=300, color='r', linestyle='--', label='Target (300ms)')
        plt.legend()
        
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.1f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
                        
        plt.savefig('docs/ttft_latency.png')
        print("Graph saved to docs/ttft_latency.png")
    except ImportError:
        pass

if __name__ == "__main__":
    run_ttft_benchmark()
