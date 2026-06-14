# Benchmark Methodology

We rigorously evaluate `lite_llama` against standard HuggingFace `transformers` to validate our goal of a ≥ 3× speedup.

## Environment Constraints
- **Hardware**: Google Colab T4 GPU (16GB VRAM)
- **Model**: `meta-llama/Llama-3.2-3B`
- **Precision**: `fp16`

## Target Metrics
- **HF Baseline**: ~183.95 tokens/sec
- **lite_llama**: ~730.45 tokens/sec
- **Speedup**: **3.97×**

## Graph Generation
The main benchmark script `benchmarks/bench_vs_hf.py` is tailored for Colab. Upon completion, it leverages Seaborn and Matplotlib to emit `throughput_comparison.png`. This visual output serves as programmatic proof of the performance gains achieved through custom Triton kernels and block-based KV caching.
