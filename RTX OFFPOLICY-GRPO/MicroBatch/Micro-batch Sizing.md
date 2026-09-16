---
branch: microbatch
---
#branch/microbatch

# Micro-batch Sizing

How many sequences enter one forward/backward. Distinct from [[Chunking]]: chunking
splits *within* a sequence's vocabulary axis, this splits *across* sequences.

- [[One sequence per micro-batch]]
- [[Token budget 9216 per GPU]]
- [[Gradient accumulation coupling]]

Up: [[RTX OFFPOLICY-GRPO]]
