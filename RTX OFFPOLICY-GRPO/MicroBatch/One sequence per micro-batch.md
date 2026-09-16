---
branch: microbatch
status: applied
---
#branch/microbatch #status/applied

# One sequence per micro-batch

**File:** `src/verl/verl/utils/seqlen_balancing.py`, lines 247-250

```python
_cap = int(os.environ.get("VERL_MAX_SEQS_PER_MICRO_BATCH", "0"))
if _cap > 0:
    num_micro_batches = max(num_micro_batches,
                            ceildiv(seq_len_effective.numel(), _cap))
```

**File:** `exp_scripts/train_openthinker3.sh`, line 5

```bash
export VERL_MAX_SEQS_PER_MICRO_BATCH=1
```

## Why

`use_dynamic_bsz=True` packs sequences until the token budget is reached. It uses the
**effective** length (`attention_mask.sum`), not the padded length. A mini-batch of 32
sequences averaging 2000 real tokens packs into `ceil(64000 / 9216)` = 7 micro-batches —
but because of [[Padded attention path]] each of those still occupies a **padded** 9216
slot per sequence. The allocator sees 7x the activation cost the budget intended.

## What it achieves

Forces `num_micro_batches >= number_of_sequences`, i.e. exactly one sequence per
micro-batch. `get_seqlen_balanced_partitions(seq_len, 32, equal_size=False)` with 32 items
into 32 partitions puts one in each.

This is what makes the measured `[MEM-C] micro-batch peak = 15.94 GiB` a **stable**
number instead of a lottery. Confirmed live: `[DBG1] shape=(1, 8192)`.

Implemented as an env var rather than a config key so it survives Hydra's struct-mode
schema without needing a `+` override, and so it applies identically to
`update_policy` and `compute_log_prob`, which both route through `rearrange_micro_batches`.

## Load-bearing beyond memory

See [[Gradient accumulation coupling]] — this cap is also what keeps the loss scaling
correct, by accident rather than by design.

Up: [[Micro-batch Sizing]]
