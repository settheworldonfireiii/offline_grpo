---
branch: turing
status: applied
---
#branch/turing #status/applied

# SDPA attention for HF model

**File:** `src/verl/verl/mix_src/mix_fsdp_worker.py`, line 215

```python
actor_module = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=local_path,
    torch_dtype=torch_dtype,
    config=actor_model_config,
    attn_implementation="sdpa",
    trust_remote_code=trust_remote_code,
)
```

## Why

Transformers would otherwise pick `flash_attention_2` when the package is importable, and
that fails on sm_75 for the same reason as [[XFORMERS backend for vLLM]].

## What it achieves

Routes both the actor and the reference model through
`torch.nn.functional.scaled_dot_product_attention`, whose memory-efficient backend has
Turing support. This is the attention used in every training forward/backward and in
`compute_log_prob` / `compute_ref_log_prob`.

Up: [[Turing Compatibility]]
