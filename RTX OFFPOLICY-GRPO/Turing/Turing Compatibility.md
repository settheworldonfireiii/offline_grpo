---
branch: turing
---
#branch/turing

# Turing Compatibility

sm_75 lacks two things every modern LLM stack assumes: **bf16 tensor cores** and
**FlashAttention-2**. Every note here is a substitution forced by one of those absences.

- [[XFORMERS backend for vLLM]]
- [[SDPA attention for HF model]]
- [[Padded attention path]]
- [[vLLM rollout dtype fp16]]

Up: [[RTX OFFPOLICY-GRPO]]
