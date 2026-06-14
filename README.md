<p align="center">
  <img src="docs/logo.png" alt="LiteLlama Logo" width="200">
</p>

# LiteLlama Inference Engine 🦙⚡

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Triton](https://img.shields.io/badge/Triton-Custom_Kernels-00C5C5.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**LiteLlama** is a high-performance, from-scratch Large Language Model inference engine tailored for the LLaMA-3 model family. Engineered with dynamic memory management, custom Triton fused kernels, and Grouped Query Attention (GQA), LiteLlama drastically outperforms standard HuggingFace architectures on memory efficiency and sheer throughput.

## 🔥 Key Features

- **Blazing Fast Throughput**: Achieves ~4x speedup over HuggingFace Transformers using memory-bound fused Triton kernels (SwiGLU, RMSNorm).
- **Dynamic TokenAttention KV Cache**: Implements a block-based paged memory pool (similar to vLLM's PagedAttention) to eliminate memory fragmentation and prevent OOMs during batch inference.
- **Grouped Query Attention (GQA)**: Custom CUDA-level kernel mapping for LLaMA-3's asymmetric attention heads.
- **Enterprise Benchmarking**: Built-in suites to measure precise Tokens/Second scaling, Peak VRAM allocations, and Time-To-First-Token (TTFT).

## 🚀 Performance Metrics

### Throughput vs HuggingFace
By stripping away massive Python abstraction layers and wiring custom Triton kernels, LiteLlama scales drastically better.

<p align="center">
  <img src="docs/throughput_comparison.png" alt="Throughput Comparison" width="600">
</p>

### Batch Scaling
Throughput remains incredibly stable and scales efficiently up to 32 concurrent requests thanks to our Continuous Batching Scheduler.

<p align="center">
  <img src="docs/throughput_scaling.png" alt="Throughput Scaling" width="600">
</p>

### Memory Efficiency
Our TokenAttention dynamically allocates exact memory block sizes without PyTorch's padding bloat, allowing massive sequence lengths without crashing the Colab T4 limits.

<p align="center">
  <img src="docs/memory_overhead.png" alt="Memory Overhead" width="600">
</p>

### Time-To-First-Token
Latency target successfully hit for large 2048-token context windows.

<p align="center">
  <img src="docs/ttft_latency.png" alt="TTFT Latency" width="600">
</p>


## 🛠️ Usage

This project is built explicitly to be run on Google Colab's T4 GPUs for consistency. 

1. **Clone and Install**
```bash
git clone https://github.com/Paramveersingh-S/Lightweight-LLM-Inference-Engine.git
cd Lightweight-LLM-Inference-Engine
pip install -e .
```

2. **Authenticate with HuggingFace**
```bash
# You must have access to LLaMA 3.2 gated repos!
huggingface-cli login
```

3. **Run the Benchmarks**
```bash
python benchmarks/bench_vs_hf.py
python benchmarks/bench_throughput.py
python benchmarks/bench_memory.py
python benchmarks/bench_ttft.py
```

## 🧠 Architecture Flow

```mermaid
graph TD;
    User[User Prompt] --> Scheduler[Batch Scheduler];
    Scheduler --> Engine[LiteLlama Engine];
    Engine --> Prefill[Prefill Pass];
    Engine --> Decode[Decode Loop];
    
    Prefill --> TokenCache[(Token Attention KV Cache)];
    Decode --> TokenCache;
    
    Prefill --> Output[First Token];
    Decode --> Sampler[Multinomial Sampler & Logits Processor];
    Sampler --> OutputToken[Next Token];
    OutputToken --> Decode;
```

---
*Built from scratch for maximum ML Systems mastery.*
