---
branch: instrumentation
status: applied
---
#branch/instrumentation #status/applied

# DBG probes in loss path

Five print sites, added while hunting the NaN that
[[Actor master weights in fp32]] eventually fixed.

**`src/verl/verl/mix_src/mix_core_alg.py`, lines 180-189** — `[DBG1]`, guarded by
`_DBGN[0] <= 3`. For each of `log_prob`, `old_log_prob`, `ratio`, `advantages`: reports
`absmax` if finite, or the count of nan/posinf/neginf plus the first bad index and whether
that index is inside `eos_mask` / `prefix_mask`. Then
`[DBG1] shape=... eos_sum=... prefix_sum=...`.

**lines 205-208** — `[DBG2] on_pg_losses_nf=... on_pg_loss=... on_mask_sum=...`

**lines 327-329** — `[DBG4] pg_loss=... off_pg_loss=... on_pg_loss=... ppo_kl=...`

**`src/verl/verl/mix_src/mix_actor.py`, line 345** — `[DBG5] policy_loss=... entropy_loss=...
loss=...`, immediately before `loss.backward()`.

**`mix_actor.py`, line 131** — `print("MICROBATCH STEP")`.

## What they established

The NaN is in the **metric**, not the gradient. Live:

```
[DBG1] shape=(1, 8192) eos_sum=8192 prefix_sum=8192
[DBG2] on_pg_losses_nf=0 on_pg_loss=nan on_mask_sum=0
[DBG4] pg_loss=-0.5827569961547852 off_pg_loss=-0.5827569961547852 on_pg_loss=nan
[DBG5] policy_loss=-0.42423132061958313 entropy_loss=1.0908203125 loss=-0.013257228769361973
```

With `n_prefix = n = 8` and `prefix_ratio = 1.0`, `prefix_mask` covers the whole response,
so `(~prefix_mask) * eos_mask` sums to zero and `masked_mean` at `mix_core_alg.py:203`
divides 0/0. `off_pg_loss` has a NaN guard at line 298; `on_pg_loss` does not.

It does **not** reach `backward()`: line 303 combines the two element-wise under
`prefix_mask`, and `loss_remove_token_mean=True` divides by the constant 8192 at line 322.
`pg_loss` and `loss` are finite throughout. `[DBG1] advantages absmax` varies 0.125-0.875
across micro-batches, so the groups carry real signal rather than collapsing.

## Cost, and why they should now be gated

`[DBG1]`, `[DBG2]`, `[DBG4]` self-limit to the first 3 calls. `[DBG5]`, `MICROBATCH STEP`,
and the unconditional `print(f"no token mean: ...")` at `mix_core_alg.py:323` do not — they
fire on **every** micro-batch. At 128 micro-batches x 8 ranks that is several thousand
lines per step through Ray's log forwarder, and `[DBG5]`'s three `.item()` calls force
three device syncs each, ~384 per rank per step.

The NaN they were added to find is fixed. Gating the three unconditional sites behind the
existing `_DBGN` counter is the obvious cleanup.

Up: [[Instrumentation]]
