# TokenAttention and Paged Memory

Standard transformer implementations allocate a contiguous block of memory for the maximum possible sequence length. This wastes significant VRAM (especially in Colab's 16GB T4 limit) and prevents optimal batching.

## Our Solution
We implement a block-based paged KV cache (inspired by vLLM's PagedAttention). 
- **Physical memory** is a pool of `16-token` blocks.
- **Logical memory** maps sequences to physical block indices via block tables.
- **Zero fragmentation**: Blocks are dynamically allocated as needed, allowing maximum concurrent batch sizes on low-memory hardware.

## Memory Math
For LLaMA-3.2-3B on a T4 GPU (16GB):
- Weights: ~6GB (fp16)
- Available for KV: ~8GB
- `num_blocks = 8GB / (2 × 16 × 8 × 128 × 2 bytes) = ~122,000 blocks`
