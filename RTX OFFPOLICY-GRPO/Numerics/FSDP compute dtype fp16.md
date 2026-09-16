---
branch: numerics
status: applied
---
#branch/numerics #status/applied

# FSDP compute dtype fp16

**File:** `src/verl/verl/mix_src/mix_fsdp_worker.py`, lines 257-261

```python
mixed_precision = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=reduce_dtype,
    buffer_dtype=buffer_dtype,
)
```

`param_dtype` is hardcoded rather than read from the computed variable above it.

## Why

sm_75 has **no bf16 tensor cores**. cuBLAS refuses a `CUDA_R_16BF` gemm below sm_80. The
upstream default of bf16 aborts at the first matmul. fp16 is the only 16-bit format with
tensor-core support on Turing.

## What it achieves

Three dtypes are in play and they are not the same thing:

| tensor | dtype | set by |
|---|---|---|
| flat parameter, grad, AdamW state | fp32 | `torch_dtype` at construction — [[Actor master weights in fp32]] |
| all-gathered compute copy (`_mp_shard`) | fp16 | `param_dtype` here |
| gradient reduce-scatter | fp32 | `reduce_dtype` |

So forward and backward run on tensor cores at fp16 speed while the master weights stay
numerically safe. `reduce_dtype=fp32` also keeps the gradient all-reduce from
underflowing across 8 ranks.

Confirmed live: `[LEDGER] fsdp _mp_shard float16 0.000 GiB` — the fp16 copy is transient
per layer, never resident.

Up: [[Numerics and Precision]]
