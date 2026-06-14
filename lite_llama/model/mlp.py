import torch
import torch.nn as nn
from ..kernels.silu_and_mul import fused_silu_and_mul
import torch.nn.functional as F

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # Removed nn.SiLU() — we now use fused Triton kernel

    def forward(self, x):
        # Route to Triton kernel to skip intermediate tensor allocations
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(fused_silu_and_mul(gate, up))
