import torch
from typing import List, Dict

class KVBlockManager:
    def __init__(self, num_blocks: int, block_size: int, 
                 num_kv_heads: int, head_dim: int, dtype=torch.float16):
        self.k_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device='cuda'
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        self.free_blocks = list(range(num_blocks))
        self.allocated: Dict[int, List[int]] = {}

    def allocate(self, seq_id: int, num_blocks: int) -> List[int]:
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError("KV cache OOM — too many concurrent sequences")
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        self.allocated[seq_id] = blocks
        return blocks

    def append_block(self, seq_id: int) -> int:
        block_id = self.free_blocks.pop()
        self.allocated[seq_id].append(block_id)
        return block_id

    def free(self, seq_id: int):
        self.free_blocks.extend(self.allocated.pop(seq_id))

    def get_block_table(self, seq_id: int) -> torch.Tensor:
        return torch.tensor(self.allocated[seq_id], dtype=torch.int32, device='cuda')
