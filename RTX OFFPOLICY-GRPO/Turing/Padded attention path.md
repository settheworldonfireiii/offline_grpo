---
branch: turing
status: applied
---
#branch/turing #status/applied

# Padded attention path

**File:** `exp_scripts/train_openthinker3.sh`, line 75

```bash
actor_rollout_ref.model.use_remove_padding=False \
```

Pinned explicitly. It matches the config default at `mix_ppo_trainer.yaml:31`, but is
stated in the script so it cannot drift.

## Why

verl's remove-padding path calls `unpad_input` / `pad_input` / `index_first_axis` — all
from `flash_attn.bert_padding`. Unavailable here, see [[SDPA attention for HF model]].

## What it achieves

Takes the `else` branch of `_forward_micro_batch` at `dp_actor.py:194-201`, which feeds
rectangular `[batch, seqlen]` tensors straight to the model.

**This is the reason the memory arithmetic is so clean and so unforgiving.** Every
micro-batch is padded to exactly 1024 + 8192 = **9216** tokens regardless of the real
sequence length, so the activation cost per micro-batch is constant. One full-sequence
hidden state is `9216 x 3584 x 2 bytes` = **exactly 64 MiB** — the allocation size that
appeared in every OOM traceback.

It also means [[Token budget 9216 per GPU]] sits at exactly the assert boundary.

Up: [[Turing Compatibility]]
