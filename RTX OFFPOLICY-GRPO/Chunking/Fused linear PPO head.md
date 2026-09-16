---
branch: chunking
status: applied
---
#branch/chunking #status/applied

# Fused linear PPO head

**File:** `src/verl/verl/workers/actor/dp_actor.py`, lines 41-45

```python
# --- backport of verl's fused LM head: verl/utils/experimental/torch_functional.py ---
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from verl.utils.experimental.torch_functional import FusedLinearForPPO

_FUSED_LINEAR_PPO = FusedLinearForPPO(chunk_size=128)
```

Sourced from upstream verl (`verl/utils/experimental/torch_functional.py`); this pin of
verl predates its use in `dp_actor`, so it is imported and wired in here.

## Why

The unfused path computes full logits, then log-softmax, then gathers the label
log-probs, then computes entropy. With vocab 152,064 and a padded sequence of 8192
(see [[Padded attention path]]):

| tensor | size |
|---|---|
| logits fp16 `[1, 8192, 152064]` | **2.32 GiB** |
| log-softmax fp32 intermediate | **4.64 GiB** |
| entropy intermediate | another pass over the same |

All of it kept alive for backward. This alone exceeds the activation budget.

## What it achieves

Slices the sequence into 128-token chunks, and for each chunk computes
`hidden @ lm_head.weight.T`, the log-prob of the label, and the entropy — then discards
that chunk's logits before moving to the next. Only `[128, 152064]` = **37 MiB** of
logits is ever live. Backward recomputes per chunk.

The output is exactly the same two tensors the loss needs — `log_probs` and `entropy`,
both `[batch, response_len]` — so nothing downstream changes.

Wired in by [[Qwen2 forward monkeypatch]]. Chunk value set by [[Chunk size 512 to 128]].

Up: [[Chunking]]
