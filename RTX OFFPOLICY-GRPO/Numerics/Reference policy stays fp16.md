---
branch: numerics
status: applied
---
#branch/numerics #status/applied

# Reference policy stays fp16

**File:** `src/verl/verl/mix_src/mix_fsdp_worker.py`, line 173 — the `else` arm

```python
torch_dtype = torch.float32 if role == "actor"  else torch.float16
```

## Why

`KL_COEF=0.2` so `use_kl_loss=True` and `ref.use_ref=True`: a second full 7.62B model
exists. The NaN argument in [[Actor master weights in fp32]] does **not** apply to it —
the reference policy has **no optimizer**, runs entirely under `torch.no_grad()`, and only
produces `ref_log_prob`. There is no `exp_avg_sq` to underflow.

## What it achieves

Halves the reference model's footprint. Combined with the upstream line 279:

```python
cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
```

the reference shard (1.9 GiB per rank) lives on the **host**, streamed to GPU one FSDP
unit at a time during `compute_ref_log_prob`. It contributes nothing to the resident GPU
budget during `update_actor`.

Structural note: `Role.RefPolicy` maps to a second `MIXActorRolloutRefWorker` instance
co-located in the same process via `create_colocated_worker_cls`, constructed with
`role="ref"` — so `_is_actor` is False there and it never builds an actor or optimizer.

Up: [[Numerics and Precision]]
