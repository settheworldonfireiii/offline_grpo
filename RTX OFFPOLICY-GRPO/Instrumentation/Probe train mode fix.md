---
branch: instrumentation
status: applied
---
#branch/instrumentation #status/applied

# Probe train mode fix

**File:** `src/verl/verl/utils/probe_fsdp_mem.py`, lines 102-106 (in `build()`, immediately
before the AdamW construction) and line 123 (first line of `one_pass`)

```python
# verl reaches update_actor through fsdp_vllm.FSDPVLLMShardingManager.__exit__,
# which calls self.module.train().  from_pretrained leaves the model in eval
# mode, and HF only applies gradient checkpointing when self.training is True,
# so without this the probe stores all 28 layers of activations.
fsdp_module.train()
```

```python
assert model.training, "model is in eval mode: gradient checkpointing is inactive"
```

+7 lines total.

## The bug this fixed

The first probe run OOM'd at **23.07 GiB in the forward pass** — the wrong failure. The
real run dies in the backward at 22.70. The probe was measuring a different program.

`AutoModelForCausalLM.from_pretrained` returns a model in **eval** mode. HuggingFace gates
checkpointing on `self.gradient_checkpointing and self.training`, so calling
`gradient_checkpointing_enable()` alone is not sufficient — all 28 decoder layers stored
full activations.

## How it was diagnosed

From the two tracebacks, not from memory of library internals:

- Probe: `modeling_qwen2.py", line 895, in forward / layer_outputs = decoder_layer(` — the
  direct, **non**-checkpointed branch.
- Real run: `torch/utils/checkpoint.py:1399 ... recompute_context` — `checkpoint.py` is in
  the frame stack.

The failing allocation was `9216 x 3584 x 2 bytes` = **exactly 64.00 MiB**, one full
hidden state, confirming the shape.

## Why the real pipeline does not have this bug

`MIXDataParallelPPOActor.update_policy` calls `self.actor_module.train()` at
`mix_actor.py:68`, and `DataParallelPPOActor.compute_log_prob` restores train mode at
`dp_actor.py:287` after its `eval()` at line 244. The probe, not going through the
sharding manager, inherited neither.

The assert exists so the failure can never recur silently.

Up: [[Instrumentation]]
