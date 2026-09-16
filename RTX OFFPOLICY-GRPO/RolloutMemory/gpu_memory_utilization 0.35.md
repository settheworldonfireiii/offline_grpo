---
branch: rollout
status: applied
---
#branch/rollout #status/applied

# gpu_memory_utilization 0.35

**File:** `exp_scripts/train_openthinker3.sh`, line 93

```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
```

Config default is `0.5` (`mix_ppo_trainer.yaml:115`). Consumed at
`verl/workers/rollout/vllm_rollout/vllm_rollout.py:123`.

## Why

vLLM 0.6.3 sizes its KV cache as
`total_memory * gpu_memory_utilization - (memory already in use after a profiling run)`.
At the moment the engine is built, the actor's fp32 shard (3.546 GiB) is **already
resident** and counts against that budget. At 0.5 the arithmetic leaves too little margin
alongside the fp16 weights.

Measured at build time from the run log:

```
before init cache memory allocated: 7.63 GB, reserved: 7.70 GB
after  init cache memory allocated: 13.06 GB, reserved: 13.11 GB
```

so the cache engine itself claims ~5.4 GiB, and the engine settles at
`After building vllm rollout, memory allocated 8.61 GB, reserved 12.21 GB`.

## What it achieves

Leaves enough of the 23.64 GiB for the actor shard to coexist with vLLM during
`generate_sequences`.

## Explicitly not a lever for the update-step OOM

Established and re-confirmed: `FSDPVLLMShardingManager.__exit__`
(`verl/workers/sharding_manager/fsdp_vllm.py:117-129`) calls
`self.inference_engine.offload_model_weights()` followed by `torch.cuda.empty_cache()`
**before** `update_actor` ever runs. vLLM holds no weights during the update. Lowering
this number further buys nothing for the backward peak and only starves generation.

Up: [[Rollout Memory]]
