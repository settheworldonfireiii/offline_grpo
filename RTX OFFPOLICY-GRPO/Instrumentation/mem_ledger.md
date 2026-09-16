---
branch: instrumentation
status: applied
---
#branch/instrumentation #status/applied

# mem_ledger

**New file:** `src/verl/verl/utils/mem_ledger.py`. Env-gated by `VERL_MEM_LEDGER=1`
(set at `exp_scripts/train_openthinker3.sh:60`), rank selected by `VERL_MEM_LEDGER_RANK`,
default 0.

## What it does

Attributes every CUDA byte to a named owner instead of reporting one opaque total. Walks
`FSDP.fsdp_modules(module)` to each unit's `_handle.flat_param` and reads:

```python
_FLAT_PARAM_BUFFERS = ("_local_shard", "_saved_grad_shard", "_mp_shard",
                       "_full_param_padded", "_full_prec_full_param_padded", "grad")
```

deduplicating by `untyped_storage().data_ptr()` so aliased views are counted once, then
sums optimizer state by device, and reports the remainder as `OTHER`.

`offload_check(optimizer, offload_fn, tag)` takes **weakrefs before** calling the offload
and runs `gc.get_referrers` on whatever survives, so a failed offload is distinguishable
from a successful one that merely copied.

## Why it was needed

Three consecutive diagnoses were wrong. `torch.cuda.memory_allocated()` says 22.70 GiB and
nothing about whose 22.70. The decision — whether the optimizer state was the thing worth
moving — could not be made from a scalar.

## What it produced

```
[LEDGER pre-update r0] allocated=7.24 GiB of 23.64
   fsdp _local_shard   float32   3.546 GiB
   fsdp _mp_shard      float16   0.000 GiB
   == fsdp_gpu 3.546 | optim_gpu 0.000 | OTHER 3.692 GiB
```

`optim_gpu 0.000` is the single line that confirms [[Optimizer state offload around step]]
works in production. `_mp_shard 0.000` confirms [[FSDP compute dtype fp16]] materialises
the fp16 copy only transiently.

## Caveat learned the hard way

The buffer names are correct for torch 2.4.0, but at `after-fsdp-init` the walk attributes
**nothing** — FSDP has not populated `_local_shard` yet, and the 3.57 GiB present at that
moment is the `flat_param` itself, which the walk does not count. Lazy initialisation, not
a naming error. The ledger's own `_self_test()` monkeypatches `torch.Tensor.is_cuda`, so it
validates the printing path and not the attribute lookup — it cannot catch this class of
mistake.

Up: [[Instrumentation]]
