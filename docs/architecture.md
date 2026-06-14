# Engine Design Overview

`lite_llama` is built around a lean, continuous batching architecture. It minimizes the abstraction overhead of frameworks like HuggingFace by providing direct kernel bindings for memory and attention operations.

## Request Lifecycle
1. User Request enters `LLMEngine`.
2. `BatchScheduler` adds the request to the active queue.
3. If memory is available, it allocates blocks via the `KVBlockManager`.
4. **Prefill Phase**: Computes the full context via the `gqa_prefill_kernel`.
5. **Decode Loop**: Iteratively appends tokens using `gqa_decode_kernel`.

## Core Data Flow
```python
# Pseudo-code representation
logits = model.forward(batch)
tokens = sampler.sample(logits)
scheduler.update(batch, tokens)
```
