---
branch: offloading
status: applied
---
#branch/offloading #status/applied

# Actor param and grad offload

**File:** `exp_scripts/train_openthinker3.sh`, lines 86-88

```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.grad_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
```

These drive the existing verl machinery in `mix_fsdp_worker.py`:

- init, lines 421-431 — `offload_fsdp_grad` and `offload_fsdp_optimizer` right after build
- `update_actor`, 481-486 — `load_fsdp_param_and_grad` in, 528-531 — offload out
- `generate_sequences`, 544-549 / 571-575 — same bracket around rollout
- `compute_log_prob`, 584-587 / 613-614 — same bracket

Implementations at `verl/utils/fsdp_utils.py:107-144`.

## Why

Three phases per training step each need the parameters on the GPU, and none of them needs
them simultaneously: generation (vLLM holds its own fp16 copy), log-prob recompute, and
the update. Between phases the fp32 shard is 3.546 GiB of idle residency.

`optimizer_offload=True` is what makes `self._is_offload_optimizer` true and therefore
what arms [[Eager optimizer load removed]]'s surviving half.

## What it achieves

Each phase brackets itself: load at entry, offload at exit, `torch.cuda.empty_cache()`
after. The GPU holds one model's worth of parameters at a time instead of three.

Measured at the boundary: `[MEM-A] floor = 7.24 GiB` — params 3.546 plus gradients 3.546
plus 0.15 of residue, and nothing else.

## Known secondary cost

`load_fsdp_param_and_grad` and its partner reassign `param.data` and `param._local_shard`
with fresh `.to()` allocations on every call rather than copying into a reused buffer. At
four phase transitions per step across 8 ranks this churns host memory and contributes to
the ~20 GiB per-rank host residency that makes RAM, not VRAM, the next constraint to watch.

Up: [[Offloading]]
