import pytest
import torch
from lite_llama.sampling.sampler import Sampler

def test_greedy_sampler():
    sampler = Sampler()
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    token = sampler.sample(logits, greedy=True)
    assert token.item() == 1

def test_multinomial_sampler():
    sampler = Sampler(temperature=1.0)
    logits = torch.tensor([[1.0, 1.0, 1.0]])
    token = sampler.sample(logits, greedy=False)
    assert token.item() in [0, 1, 2]
