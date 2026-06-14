import torch
import triton
import triton.language as tl

@triton.jit
def _silu_and_mul_kernel(
    GATE_ptr, UP_ptr, Y_ptr,
    stride, N,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    GATE_ptr += row_idx * stride
    UP_ptr += row_idx * stride
    Y_ptr += row_idx * stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    gate = tl.load(GATE_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(UP_ptr + offsets, mask=mask, other=0.0)
    
    gate_f32 = gate.to(tl.float32)
    silu = gate_f32 * tl.sigmoid(gate_f32)
    
    y = (silu.to(gate.dtype) * up)
    tl.store(Y_ptr + offsets, y, mask=mask)

def fused_silu_and_mul(gate: torch.Tensor, up: torch.Tensor):
    gate_2d = gate.view(-1, gate.shape[-1])
    up_2d = up.view(-1, up.shape[-1])
    M, N = gate_2d.shape
    y_2d = torch.empty_like(gate_2d)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    
    _silu_and_mul_kernel[grid](
        gate_2d, up_2d, y_2d,
        gate_2d.stride(0), N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y_2d.view(gate.shape)
