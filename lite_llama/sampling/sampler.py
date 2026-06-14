import torch
from .logits_processor import LogitsProcessor

class Sampler:
    def __init__(self, temperature: float = 0.8, top_p: float = 0.95, top_k: int = 50):
        self.processor = LogitsProcessor(temperature, top_p, top_k)

    def sample(self, logits: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        """
        Samples the next token from logits.
        logits: [batch_size, vocab_size]
        """
        if greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
            
        probs = self.processor(logits)
        return torch.multinomial(probs, num_samples=1)
