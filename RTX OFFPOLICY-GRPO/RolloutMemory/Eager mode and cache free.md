---
branch: rollout
status: load-bearing-default
---
#branch/rollout #status/applied

# Eager mode and cache free

**File:** `src/verl/verl/mix_src/config/mix_ppo_trainer.yaml`, lines 117-118

```yaml
enforce_eager: True
free_cache_engine: True
```

Config defaults, left unchanged — recorded because they are load-bearing and would break
the memory budget if flipped, and because `vllm_rollout.py:86-88` asserts they stay
consistent:

```python
assert not (
    not config.enforce_eager and config.free_cache_engine
), "disable CUDA graph (enforce_eager = False) if free cache engine"
```

## What each does

**`enforce_eager: True`** — no CUDA graph capture. Graph capture would reserve a static
memory pool for the captured shapes and hold it for the process lifetime. Confirmed in the
log: `WARNING config.py:380] To see benefits of async output processing, enable CUDA
graph. Since, enforce-eager is enabled, async output processor cannot be used`. We trade
generation throughput for a reclaimable allocator.

**`free_cache_engine: True`** — the KV cache blocks are released between rollout phases
(`vllm_rollout.py:290-291`) rather than held across the step. This is what lets the
~5.4 GiB cache engine coexist with a training phase that needs 15.94 GiB.

## Related: chunked prefill

`enable_chunked_prefill: True` (line 127) resolves what would otherwise be a hard failure:
`max_num_batched_tokens` is 8192 while `max_model_len` is `1024 + 8192 = 9216`. Without
chunked prefill vLLM rejects sequences longer than the batch budget. Confirmed:
`INFO config.py:1005] Chunked prefill is enabled with max_num_batched_tokens=8192.`

This matters specifically because of teacher forcing: with `prefix_ratio = 1.0` the prompt
handed to vLLM is the original prompt **plus the entire target**, so prefills routinely
approach the full 9216.

Up: [[Rollout Memory]]
