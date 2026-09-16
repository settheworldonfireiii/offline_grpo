---
branch: pinned-memory
status: applied
---
#branch/pinned-memory #status/applied

# pin_memory neutered

**File:** `src/verl/verl/mix_src/mix_fsdp_worker.py`, line 50 — module scope, before any
class definition

```python
torch.Tensor.pin_memory = lambda self, *a, **k: self
```

## Why

`cudaHostAlloc` fails on this host. Any call to `.pin_memory()` raises

```
RuntimeError: CUDA error: invalid argument
```

Page-locked host allocation is used by the PyTorch DataLoader, by FSDP's CPU-offload
staging buffers, and by vLLM's swap space. One of them hits it on every run.

## What it achieves

Replaces the method with identity at class level, so every pinned allocation in the
process silently becomes a normal pageable tensor. The failure disappears everywhere at
once rather than needing a call-site-by-call-site hunt.

## The tax it imposes

`non_blocking=True` **only** has effect on pinned memory. Every transfer in
[[Offloading]] — `load_fsdp_optimizer` and `offload_fsdp_optimizer` at
`verl/utils/fsdp_utils.py:127-144` — passes `non_blocking=True` and now gets a fully
**synchronous, pageable** copy instead. Effective bandwidth drops roughly 2x and nothing
overlaps with compute.

At 4 mini-batches per training step that is `4 x 2 x 7.09` = **56.7 GiB** of synchronous
PCIe traffic per rank per step. This is the known price of the fix and the reason
wall-clock, not memory, is the next thing to watch.

Up: [[Pinned Memory]]
