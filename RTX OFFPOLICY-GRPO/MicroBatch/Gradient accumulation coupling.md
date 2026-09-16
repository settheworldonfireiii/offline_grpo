---
branch: microbatch
status: latent-hazard
---
#branch/microbatch #status/latent-hazard

# Gradient accumulation coupling

**File:** `src/verl/verl/mix_src/mix_actor.py`, lines 70-73 and 343

```python
assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
self.gradient_accumulation = (
    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
)
```

```python
loss = policy_loss / self.gradient_accumulation
```

## The arithmetic

After normalisation at `mix_fsdp_worker.py:122-130`:

```
ppo_mini_batch_size  = 32 * 8 / 8 = 32
ppo_micro_batch_size =  1 * 8 / 8 =  1
gradient_accumulation = 32 / 1 = 32
```

Local batch per rank is `train_batch_size(128) * n(8) / world(8)` = **128 sequences**, so
`batch.split(32)` gives **4 mini-batches per training step**, each of 32 micro-batches.
128 backward passes and 4 optimizer steps per step.

## The hazard

With `use_dynamic_bsz=True` the real micro-batch count comes from
`rearrange_micro_batches`, **not** from `ppo_micro_batch_size`. The divisor at line 343 is
a fixed 32 either way. They agree only because
[[One sequence per micro-batch]] forces the count to exactly 32.

Drop `VERL_MAX_SEQS_PER_MICRO_BATCH` and the loss is still divided by 32 while the true
count might be 7 — gradients silently scaled ~4.5x too small. **No crash, no warning, a
quietly wrong learning rate.**

Upstream verl handles this by scaling with `len(micro_batch) / mini_batch_size` instead;
`mix_actor` does not.

## Suggested guard (not applied)

```python
if self.config.use_dynamic_bsz:
    assert len(micro_batches) == self.gradient_accumulation, (
        f"dynamic bsz produced {len(micro_batches)} micro-batches but "
        f"gradient_accumulation is {self.gradient_accumulation}")
```

Up: [[Micro-batch Sizing]]
