---
branch: instrumentation
status: applied
---
#branch/instrumentation #status/applied

# MEM probes in update path

**File:** `src/verl/verl/mix_src/mix_fsdp_worker.py`, lines 500-503 — inside `update_actor`

```python
                torch.cuda.reset_peak_memory_stats()
                print("[MEM-A] floor = %.2f GiB" % (torch.cuda.memory_allocated() / 2**30), flush=True)
                metrics = self.actor.update_policy(data=data)
                print("[MEM] update_actor peak=%.2f GiB of 23.64" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
```

**File:** `src/verl/verl/mix_src/mix_actor.py`, lines 347-348 — inside the micro-batch loop,
immediately after `loss.backward()`

```python
                    print("[MEM-C] micro-batch peak = %.2f GiB" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
                    torch.cuda.reset_peak_memory_stats()
```

## What each actually measures — this is not obvious

`[MEM-C]`'s reset at `mix_actor.py:348` fires **128 times per training step**, which
overwrites the single reset at `mix_fsdp_worker.py:500`. The consequence:

| probe | window | live value |
|---|---|---|
| `[MEM-A] floor` | instantaneous, before any reset | **7.24** — params 3.546 + grads 3.546 |
| `[MEM-C]` mid-mini-batch | one micro-batch's forward+backward | **15.94** — the real backward peak |
| `[MEM-C]` first of a mini-batch | includes the preceding `_optimizer_step` | **17.90** — the true worst case |
| `[MEM]` at line 503 | only since the 128th reset, i.e. the last `_optimizer_step` | ~18, **not** the update peak |

So `[MEM]` is the misleading one. The number that corresponds to the probe's 13.36 is
`[MEM-C]`, and the number that represents the ceiling is the periodic **17.90** appearing
once per mini-batch boundary.

## Why the 17.90 appears exactly 4 times per step

`_optimizer_step` runs at the end of each mini-batch (`mix_actor.py:358`), the local batch
of 128 splits into 4 mini-batches, and the reset at line 348 happens **after** the print —
so the first micro-batch of each new mini-batch reports a window containing the previous
load → step → offload. See [[Gradient accumulation coupling]].

## What it settled

`22.70 -> 15.94` backward, worst case `17.90 of 23.64`, **~5.7 GiB headroom**. This is the
measurement that closed [[Optimizer state offload around step]].

Up: [[Instrumentation]]
