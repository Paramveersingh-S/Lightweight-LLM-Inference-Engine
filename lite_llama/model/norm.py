import torch
import torch.nn as nn
from ..kernels.rmsnorm import fused_rmsnorm

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Use our low-level memory-bound Triton kernel
        return fused_rmsnorm(x, self.weight, self.eps)
