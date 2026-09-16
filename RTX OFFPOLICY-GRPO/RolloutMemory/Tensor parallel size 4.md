---
branch: rollout
status: applied
---
#branch/rollout #status/applied

# Tensor parallel size 4

**File:** `exp_scripts/train_openthinker3.sh`, line 89

```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
```

Config default is `2` (`mix_ppo_trainer.yaml:120`).

## Why

fp16 weights for 7.62B params are **15.2 GiB**. At TP=2 each GPU carries 7.6 GiB of vLLM
weights on top of the actor shard — before any KV cache. At TP=4 it is **3.8 GiB**.

## What it achieves

With `world_size=8` and `infer_tp=4`, `_build_rollout` (`mix_fsdp_worker.py:333-340`)
builds a `(dp=2, infer_tp=4)` device mesh: two vLLM replicas of four GPUs each.

Consequences that propagate:

- `FSDPVLLMShardingManager.preprocess_data` all-gathers the prompt batch across the TP
  group of 4, so each replica sees 64 prompts; `postprocess_data` chunks the generated
  batch back by `tp_size`, returning 128 sequences per rank. That 128 is exactly the local
  batch that [[Gradient accumulation coupling]] then splits into 4 mini-batches.
- Weight sync stays sharded rather than gathered — `load_format: dummy_dtensor` means
  `full_params = "hf" in load_format` is False, so the manager uses `SHARDED_STATE_DICT`
  (`fsdp_vllm.py:66-70`) and never materialises a full fp32 state dict.

Up: [[Rollout Memory]]
