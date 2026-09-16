---
branch: numerics
status: applied
---
#branch/numerics #status/applied

# Actor master weights in fp32

**File:** `src/verl/verl/mix_src/mix_fsdp_worker.py`, lines 171-175

```python
torch_dtype = fsdp_config.get("model_dtype", None)
if torch_dtype is None:
    torch_dtype = torch.float32 if role == "actor"  else torch.float16
else:
    torch_dtype = PrecisionType.to_dtype(torch_dtype)
```

Upstream verl builds the actor in fp16 here. The `role == "actor"` test is ours.

## Why

The loss went NaN. Not from the model, from **AdamW**.

`from_pretrained(torch_dtype=...)` sets the dtype the FSDP flat parameter is built in,
and AdamW allocates its state lazily inside the first `optimizer.step()` with
`torch.zeros_like(p)` — so the state inherits that dtype. In fp16:

- `exp_avg_sq` accumulates `grad**2`. With gradients around `1e-4`, `grad**2 ~ 1e-8`.
- fp16's smallest subnormal is `5.96e-8`. `1e-8` flushes to **zero**.
- `eps = 1e-8` also flushes to zero.
- The update is `m / (sqrt(v) + eps)` = `0 / 0` = **NaN**.

This is a property of the exponent range, not of the model or the data. No amount of
gradient clipping or loss scaling reaches it.

## What it achieves

The flat parameter, its gradient, and both AdamW moments are all fp32. Compute still
happens in fp16 via [[FSDP compute dtype fp16]] — `param_dtype` governs the compute copy
only.

Verified: micro-batch 33 went from
`[DBG1] log_prob BAD n=8192 nan=8192` to `[DBG1] log_prob ok absmax=18.2568`.

## Cost

Doubles the parameter shard: 952.5M params per rank x 4 bytes = **3.546 GiB** instead of
1.773, and the optimizer state to 7.09 GiB. That cost is what [[Offloading]] then had to
pay for.

Up: [[Numerics and Precision]]
