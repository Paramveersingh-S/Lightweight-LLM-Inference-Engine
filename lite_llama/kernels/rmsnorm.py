import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(
    X_ptr, Y_ptr, W_ptr,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride
    Y_ptr += row_idx * stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(W_ptr + offsets, mask=mask, other=0.0)
    
    x_f32 = x.to(tl.float32)
    var = tl.sum(x_f32 * x_f32, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    y = (x_f32 * rstd).to(x.dtype) * w
    tl.store(Y_ptr + offsets, y, mask=mask)

def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    x_2d = x.view(-1, x.shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (M,)
    _rmsnorm_kernel[grid](
        x_2d, y, weight,
        x_2d.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.view(x.shape)
