---
branch: offloading
status: applied
---
#branch/offloading #status/applied

# Eager optimizer load removed

**File:** `src/verl/verl/mix_src/mix_fsdp_worker.py`, lines 487-490 — in `update_actor`

```python
        #if self._is_offload_optimizer:
        #    load_fsdp_optimizer(
        #        optimizer=self.actor_optimizer, device_id=torch.cuda.current_device()
        #    )
```

Commented out. The matching offload at lines 532-533 is **kept**:

```python
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
```

## Why

Upstream pulls the whole optimizer state onto the GPU at the **top** of `update_actor`,
before the first forward, and leaves it there for the entire 128-micro-batch run. That
places all 7.09 GiB in residence for exactly the phase where the peak occurs, defeating
[[Optimizer state offload around step]] before it can act.

This line and the one it pairs with are a matched set: remove the eager load, and the
per-step load in `_optimizer_step` becomes the only path onto the GPU.

## What it achieves

Makes `optim_gpu` zero on entry to the update. The retained offload at 532 is the
idempotent backstop — after the final `_optimizer_step` the state is already on the host,
so it is a no-op, but it guarantees the invariant if the loop ever exits early.

Commented rather than deleted so the upstream shape stays legible and the pairing with
`_is_offload_optimizer` is obvious to the next reader.

Up: [[Offloading]]
