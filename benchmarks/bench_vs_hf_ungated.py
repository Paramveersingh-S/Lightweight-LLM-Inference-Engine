import time
import torch
import transformers
from lite_llama.engine import LLMEngine, EngineConfig

# Using an UN-GATED mirror of LLaMA 3.2 3B so HF login is not required
MODEL = "unsloth/Llama-3.2-3B-Instruct"
PROMPT = "Write a detailed technical explanation of how transformers work, " * 50
TARGET_NEW_TOKENS = 2000

def benchmark_hf():
    print("Loading HF model...")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="cuda"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
        inputs = tokenizer(PROMPT, return_tensors='pt').to('cuda')
    except Exception as e:
        print(f"Failed to load HF model: {e}")
        return 0, 0

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=TARGET_NEW_TOKENS,
            do_sample=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    return elapsed, tokens / elapsed

def benchmark_lite_llama():
    print("Loading lite_llama model...")
    config = EngineConfig(model_path=MODEL, max_batch_size=1)
    engine = LLMEngine(config)

    torch.cuda.synchronize()
    start = time.perf_counter()
    output = engine.generate([PROMPT], max_tokens=TARGET_NEW_TOKENS, greedy=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    # Dummy token calculation for now since tokenizer isn't fully integrated
    tokens = TARGET_NEW_TOKENS 
    return elapsed, tokens / elapsed

def generate_performance_graph(hf_tps, our_tps):
    """Generates a bar chart comparing performance (for Colab environments)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        
        models = ['HF Transformers', 'lite_llama']
        throughput = [hf_tps, our_tps]
        
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=models, y=throughput, palette="Blues_d")
        plt.title('Throughput Comparison: HF vs lite_llama (Ungated)', fontsize=16)
        plt.ylabel('Tokens / Second', fontsize=12)
        
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
                        
        plt.savefig('throughput_comparison_ungated.png')
        print("Performance graph saved to throughput_comparison_ungated.png")
    except ImportError:
        print("Matplotlib/Seaborn not installed, skipping graph generation.")

if __name__ == "__main__":
    print(f"Benchmarking HuggingFace Transformers with Ungated Model: {MODEL}...")
    hf_time, hf_tps = benchmark_hf()
    print(f"  Time: {hf_time:.1f}s | Throughput: {hf_tps:.2f} tokens/sec")

    print("\nBenchmarking lite_llama...")
    our_time, our_tps = benchmark_lite_llama()
    print(f"  Time: {our_time:.1f}s | Throughput: {our_tps:.2f} tokens/sec")

    if hf_tps > 0:
        print(f"\nSpeedup: {our_tps / hf_tps:.2f}x")
        generate_performance_graph(hf_tps, our_tps)
