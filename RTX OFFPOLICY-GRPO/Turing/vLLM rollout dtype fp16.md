---
branch: turing
status: applied
---
#branch/turing #status/applied

# vLLM rollout dtype fp16

**File:** `src/verl/verl/mix_src/config/mix_ppo_trainer.yaml`, line 114

```yaml
dtype: float16 # should align with FSDP
```

Passed through at `verl/workers/rollout/vllm_rollout/vllm_rollout.py:121` as
`dtype=config.dtype`.

## Why

`OpenThinker3-7B`'s config declares `"torch_dtype": "bfloat16"`. Left alone, vLLM would
try to run bf16 on hardware without bf16 tensor cores. It must also match what
[[FSDP compute dtype fp16]] hands over during weight sync, or the DTensor load mismatches.

## What it achieves

vLLM casts on load. Confirmed in the run log:

```
WARNING 09-15 23:37:25 config.py:1674] Casting torch.bfloat16 to torch.float16.
```

fp16 weights at TP=4 are ~3.8 GiB per GPU. See [[Tensor parallel size 4]].

Up: [[Turing Compatibility]]
