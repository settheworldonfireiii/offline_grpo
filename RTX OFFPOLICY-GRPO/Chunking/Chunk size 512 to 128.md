---
branch: chunking
status: applied
---
#branch/chunking #status/applied

# Chunk size 512 to 128

**File:** `src/verl/verl/workers/actor/dp_actor.py`, line 45

```python
_FUSED_LINEAR_PPO = FusedLinearForPPO(chunk_size=128)
```

Was `chunk_size=512`.

## Why

At 512 the per-chunk logits are `[512, 152064]` fp16 = **148 MiB**, plus the fp32
log-softmax intermediate and the transposed weight view. That was the largest single live
allocation in the LM-head region.

## What it achieves

Per-chunk logits drop 4x to `[128, 152064]` = **37 MiB**.

Verified by the change in failure signature: after this edit the OOM traceback **stopped
passing through** `verl/utils/torch_functional.py`, and the failing allocation moved from
**298 MiB to 334 MiB** — the LM head stopped being the peak and the failure relocated to
the transformer body. That relocation is what redirected the investigation towards
[[Offloading]].

Tradeoff: 4x more chunk iterations per forward. Negligible against an 8192-token sequence.

## This literal is the only control

`FusedLinearForPPO.__init__(self, chunk_size: int = 128, ...)` takes the value as a
constructor argument and the sole construction in the tree is this line. The first
attempt at this fix set `+actor_rollout_ref.actor.logprob_chunk_size=128` on the command
line and changed nothing, because no code reads that key and no yaml interpolates it.
**To change the chunk size, change this line.**

Up: [[Chunking]]
