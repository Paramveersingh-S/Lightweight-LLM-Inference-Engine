# Grouped Query Attention (GQA)

LLaMA-3 uses GQA to drastically reduce the KV cache memory footprint while maintaining the expressive power of Multi-Head Attention (MHA). 

## The Math
If we have 32 query heads (`Hq`) and 8 KV heads (`Hkv`), our group size `G = 4`. 
Each KV head serves 4 query heads:
- KV head 0 → Q heads 0, 1, 2, 3
- KV head 1 → Q heads 4, 5, 6, 7

## Kernel Mapping
In our custom Triton kernels, the mathematical mapping is handled at the hardware grid level:
```python
q_head = tl.program_id(1)
kv_head = q_head // (Hq // Hkv)
```
This entirely avoids duplicating KV cache data in memory (which standard PyTorch `repeat_interleave` would do), resulting in massive bandwidth savings during decoding.
