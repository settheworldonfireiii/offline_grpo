---
branch: instrumentation
status: applied
---
#branch/instrumentation #status/applied

# probe_fsdp_mem

**New file:** `src/verl/verl/utils/probe_fsdp_mem.py`, 170 lines. Standalone:

```bash
cd /mnt/data2/radke149/offline-grpo/src/verl && \
  VERL_MEM_LEDGER=1 torchrun --nproc_per_node=8 verl/utils/probe_fsdp_mem.py
```

## What it does

Rebuilds the actor exactly as `_build_model_optimizer` does — fp32 construction,
`sdpa`, gradient checkpointing with `use_reentrant=False`, `FSDP(cpu_offload=None,
use_orig_params=False, sharding_strategy=FULL_SHARD, MixedPrecision(fp16, fp32, fp32),
forward_prefetch=False)`, 1-D device mesh — then runs **two** forward/backward passes
around a single `optimizer.step()`, printing [[mem_ledger]] at six boundaries.

Pass 1 runs before AdamW state exists. Pass 2 runs with it on the host. The difference
between them is the entire question.

## Why

To answer "will the fix work" **before** committing a multi-hour run to it. The one
assumption the whole design rested on — that `offload_fsdp_optimizer` genuinely frees GPU
memory rather than leaving a copy — had never been measured.

## What it produced

| boundary | GiB |
|---|---|
| after fsdp init | 3.57 |
| pass 1 backward peak | **13.36** |
| after backward 1 | 7.16 |
| after step | 14.29 |
| after offload | 7.27 |
| pass 2 backward peak | **13.36** |

and decisively:

```
[OFFLOAD after-step r0] state_on_gpu 7.093 -> 0.000 GiB | allocated 14.295 -> 7.266 | freed 7.029 GiB
```

The 13.36 predicted the live `[MEM-C] micro-batch peak = 13.61` on the first micro-batch to
within 0.25 GiB.

## Honest limits

No vLLM in the process and no `param_offload`, so its absolute floor differs from training.
It measures the **shape** of the update step, not the full budget. Its synthetic loss is
NaN (`torch.randint(0, 100, ...)` token ids through the fp16-autocast fused head), which
does not affect memory — shapes and dtypes drive allocation, not values — but does mean the
probe validated memory only, never numerics.

Up: [[Instrumentation]]
