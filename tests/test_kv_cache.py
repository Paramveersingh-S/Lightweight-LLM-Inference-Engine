import pytest
import torch
from lite_llama.kv_cache.block_manager import KVBlockManager

def test_block_manager_allocation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping KV block test")
        
    manager = KVBlockManager(num_blocks=100, block_size=16, num_kv_heads=8, head_dim=128)
    
    # Allocate 10 blocks for seq 0
    blocks = manager.allocate(seq_id=0, num_blocks=10)
    assert len(blocks) == 10
    assert len(manager.free_blocks) == 90
    
    # Append 1 block for seq 0
    new_block = manager.append_block(seq_id=0)
    assert len(manager.free_blocks) == 89
    assert len(manager.allocated[0]) == 11
    
    # Free seq 0
    manager.free(seq_id=0)
    assert len(manager.free_blocks) == 100
    assert 0 not in manager.allocated
