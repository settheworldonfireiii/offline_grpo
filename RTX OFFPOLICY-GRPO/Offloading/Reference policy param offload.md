---
branch: offloading
status: applied
---
#branch/offloading #status/applied

# Reference policy param offload

**File:** `exp_scripts/train_openthinker3.sh`, line 96

```bash
actor_rollout_ref.ref.fsdp_config.param_offload=True \
```

Backed by `mix_fsdp_worker.py:279`, the upstream line this flag makes meaningful:

```python
cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
```

## Why

`KL_COEF=0.2` makes `use_kl_loss=True` and `ref.use_ref=True`, so a **second** 7.62B model
exists in the job. At fp16 ([[Reference policy stays fp16]]) its shard is 1.9 GiB per
rank — small, but it would be resident for the entire run while being used in exactly one
phase, `compute_ref_log_prob`.

The comment above line 279 in upstream verl notes CPUOffload is forced off for the actor
because it produces incorrect results with gradient accumulation. The reference policy has
no gradients, so the restriction does not apply to it.

## What it achieves

FSDP's native `CPUOffload` keeps the reference shard on the host and streams it to the GPU
one FSDP unit at a time during the forward, freeing each unit immediately after. Peak GPU
contribution is one unwrapped decoder layer, not the model.

The reference policy therefore contributes **nothing** to the resident budget during
`update_actor`, which is where the ceiling binds.

## Structural note

`Role.RefPolicy` maps to a second `MIXActorRolloutRefWorker`, co-located in the same
process by `create_colocated_worker_cls` (`mix_trainer.py:325`) and constructed with
`role="ref"` (`mix_trainer.py:302`). Because `_is_actor` is False on that instance, the
`if self._is_actor or self._is_rollout` guard at `mix_fsdp_worker.py:392` is skipped and it
never builds an actor, an optimizer, or a vLLM engine. One process per GPU, two models,
one of them entirely on the host.

Up: [[Offloading]]
