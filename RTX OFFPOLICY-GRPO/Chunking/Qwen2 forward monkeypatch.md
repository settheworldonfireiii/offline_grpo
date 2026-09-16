---
branch: chunking
status: applied
---
#branch/chunking #status/applied

# Qwen2 forward monkeypatch

**File:** `src/verl/verl/workers/actor/dp_actor.py`, lines 48-68

```python
def _qwen2_forward_fused_ppo(self, input_ids=None, attention_mask=None,
                             position_ids=None, ppo_labels=None,
                             temperature=1.0, **kwargs):
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    response_length = ppo_labels.shape[1]
    hidden = outputs.last_hidden_state[:, -response_length - 1 : -1, :]
    return _FUSED_LINEAR_PPO(
        hidden_states=hidden,
        vocab_weights=self.lm_head.weight,
        input_ids=ppo_labels,
        temperature=temperature,
    )


Qwen2ForCausalLM.forward = _qwen2_forward_fused_ppo
```

And the call site, `dp_actor.py:195-201`:

```python
else:  # not using rmpad and no ulysses sp
    log_probs, entropy = self.actor_module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        ppo_labels=micro_batch["responses"],
        temperature=temperature,
    )
```

## Why

`FusedLinearForPPO` needs the **hidden states**, not the logits. HF's
`Qwen2ForCausalLM.forward` applies `lm_head` internally and only returns logits — by then
the 2.32 GiB is already allocated. The only way to intercept is to replace `forward`.

## What it achieves

- Calls `self.model` (the bare `Qwen2Model`) so the LM head is never applied eagerly.
- Slices `[-response_length - 1 : -1]` — the standard next-token shift, so hidden state at
  position *t* predicts token *t+1*.
- Returns `(log_probs, entropy)` instead of `CausalLMOutputWithPast`.

The patch is applied at **class level, at import time**, so it covers the actor, the
reference policy (which also goes through `DataParallelPPOActor._forward_micro_batch`),
and any other holder of a `Qwen2ForCausalLM`. It does **not** touch vLLM, which uses its
own model implementation under `verl/third_party/vllm`.

`modeling_qwen2.py:895` — the direct, non-checkpointed `layer_outputs = decoder_layer(`
call — is the frame that appears in tracebacks from this path; its presence or absence is
how [[Probe train mode fix]] was diagnosed.

Up: [[Chunking]]
