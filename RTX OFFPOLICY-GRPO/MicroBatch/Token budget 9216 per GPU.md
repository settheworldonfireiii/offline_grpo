---
branch: microbatch
status: constraint
---
#branch/microbatch #status/constraint

# Token budget 9216 per GPU

**File:** `exp_scripts/train_openthinker3.sh`, lines 71-79

```bash
data.max_prompt_length=1024 \
data.max_response_length=8192 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9216 \
```

## Not a modification — a fixed constraint

`max_response_length=8192` and `ppo_max_token_len_per_gpu=9216` are **the condition of the
whole work** and were never changed. Every other note in this vault exists to make these
two numbers fit in 23.64 GiB. Recorded here so it is visible in the graph as the thing
everything else orbits.

## Propagation

`9216 = 1024 + 8192`, exactly one full padded sequence. The value reaches three places:

- `actor.ppo_max_token_len_per_gpu` — the training forward/backward
- `rollout.log_prob_max_token_len_per_gpu` — interpolated at
  `mix_ppo_trainer.yaml:102` as `${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}`
- `ref.log_prob_max_token_len_per_gpu` — interpolated the same way at line 125

So the recompute and reference passes inherit 9216 rather than the 16384 default. This was
checked explicitly; had it not interpolated, both log-prob passes would have run at nearly
double the intended budget.

## Zero margin

`verl/utils/seqlen_balancing.py:240-242`:

```python
assert (
    max_token_len >= max_seq_len
), f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
```

`max_seq_len` is `attention_mask.shape[-1]` = 9216. The assert passes at exactly
`9216 >= 9216`. Any increase to prompt or response length trips it immediately.

Up: [[Micro-batch Sizing]]
