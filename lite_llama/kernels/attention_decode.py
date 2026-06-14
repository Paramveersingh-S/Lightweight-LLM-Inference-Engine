import torch
import triton
import triton.language as tl

@triton.jit
def gqa_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    o_ptr,
    Hq, Hkv, block_size,
    B, D,
    BLOCK_D: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head   = tl.program_id(1)
    kv_head  = q_head // (Hq // Hkv)
    
    context_len = tl.load(context_lens_ptr + batch_id)
    pass

def gqa_decode(q: torch.Tensor, kv_cache, batch) -> torch.Tensor:
    # Caller wrapper for the Triton kernel
    return q
