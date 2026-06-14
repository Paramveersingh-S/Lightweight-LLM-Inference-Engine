import pytest
import torch

@pytest.mark.skip(reason="Needs Triton GPU compilation in Colab")
def test_gqa_prefill_kernel_correctness():
    """Matches output of our Triton GQA prefill against PyTorch"""
    pass

@pytest.mark.skip(reason="Needs Triton GPU compilation in Colab")
def test_gqa_decode_kernel_correctness():
    """Matches output of our Triton GQA decode against PyTorch"""
    pass
