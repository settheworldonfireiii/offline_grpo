---
branch: turing
status: applied
---
#branch/turing #status/applied

# XFORMERS backend for vLLM

**File:** `exp_scripts/train_openthinker3.sh`, line 18

```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
```

## Why

vLLM 0.6.3 auto-selects FlashAttention when it thinks it can. FlashAttention-2 kernels
require **sm_80**; on sm_75 the selector either refuses or the kernel launch fails.

## What it achieves

Forces vLLM's memory-efficient xformers attention path, which has Turing kernels.
Confirmed in the run log:

```
INFO 09-15 23:37:31 selector.py:115] Using XFormers backend.
```

The HF-side equivalent is [[SDPA attention for HF model]] — the two engines need
separate instructions.

Up: [[Turing Compatibility]]
