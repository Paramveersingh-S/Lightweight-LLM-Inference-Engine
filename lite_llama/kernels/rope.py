import torch
import triton
import triton.language as tl

@triton.jit
def _rope_kernel(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,
    stride_q_tok, stride_q_head,
    stride_k_tok, stride_k_head,
    seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    pass

def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, head_dim: int):
    return q, k
