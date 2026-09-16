---
branch: offloading
status: applied
---
#branch/offloading #status/applied

# Optimizer state offload around step

**File:** `src/verl/verl/mix_src/mix_actor.py`, lines 366-383 —
`MIXDataParallelPPOActor._optimizer_step`

```python
def _optimizer_step(self):
    assert self.config.grad_clip is not None

    if isinstance(self.actor_module, FSDP):
        grad_norm = self.actor_module.clip_grad_norm_(
            max_norm=self.config.grad_clip
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_module.parameters(), max_norm=self.config.grad_clip
        )

    from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer
    load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())
    self.actor_optimizer.step()
    offload_fsdp_optimizer(optimizer=self.actor_optimizer)
```

The three added lines are the import, the load, and the offload. **This is the change that
fixed the OOM.**

## Why

The budget, per rank, for a 7.62B model sharded 8 ways (952.5M params per rank):

| tensor | bytes/param | GiB |
|---|---|---|
| fp32 parameter shard | 4 | 3.546 |
| fp32 gradient shard | 4 | 3.546 |
| AdamW `exp_avg` + `exp_avg_sq` | 8 | **7.093** |
| **resident floor** | | **14.19** |

Measured backward transient on top of the floor: **~6.2 GiB**. `14.19 + 6.2 = 20.4`, plus
~2.3 GiB of vLLM allocator residue and loss-path overhead = **22.7** — exactly where it
died.

The AdamW state is needed **only** inside `.step()`. It is dead weight during the forward
and backward, which is where the peak lives.

## Why not split it across GPUs instead

FSDP `FULL_SHARD` is ZeRO-3: it already shards parameters, gradients **and** optimizer
state across the 8 ranks. The 7.09 GiB is already one eighth of the 56.7 GiB total
(`2 states x 4 bytes x 7.62B`). There is no unsharded copy left to distribute and no idle
GPU to distribute it to. The only remaining axis is **time**.

## What it achieves

Proven by the probe before deployment and confirmed in the live run:

```
[OFFLOAD after-step r0] state_on_gpu 7.093 -> 0.000 GiB | allocated 14.295 -> 7.266 | freed 7.029 GiB
```

all eight ranks. And in production:

```
[LEDGER pre-update r0] allocated=7.24 GiB of 23.64
   == fsdp_gpu 3.546 | optim_gpu 0.000 | OTHER 3.692 GiB
```

`optim_gpu 0.000` entering the update. Backward peak fell from 22.70 to **15.94**.

## Cost

`_optimizer_step` sits inside the **mini-batch** loop (`mix_actor.py:358`), and the local
batch of 128 sequences splits into 4 mini-batches of 32 — so this runs **4 times per
training step**. `4 x 2 x 7.09` = **56.7 GiB** of PCIe traffic per rank per step, made
synchronous by [[pin_memory neutered]].

## Available refinement, not applied

AdamW is elementwise: once the gradient is complete, parameter *i*'s update depends on
nothing but parameter *i*. The state could be streamed one FSDP flat-param at a time
(~30 units, MLPs being 5.70B of the 7.62B), dropping peak optimizer residency from 7.09 to
~0.24 GiB and allowing the next transfer to overlap the current step —
**bit-identical arithmetic**. Not applied because the binding peak is the backward
(15.94), not the step, and 5.7 GiB of headroom remains.

Up: [[Offloading]]
