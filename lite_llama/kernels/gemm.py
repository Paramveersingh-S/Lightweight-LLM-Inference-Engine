import torch

def hgemm_wrapper(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Placeholder for custom Triton Half-Precision GEMM.
    Currently delegates to torch.matmul which calls cuBLAS in fp16.
    """
    return torch.matmul(a, b)
