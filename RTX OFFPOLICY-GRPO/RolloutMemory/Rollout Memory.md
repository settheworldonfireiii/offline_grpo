---
branch: rollout
---
#branch/rollout

# Rollout Memory

vLLM's share of each GPU during generation. Established early as **not** the lever for
the update-step OOM — the weights are already offloaded before `update_actor` runs — but
these settings are what let generation itself fit.

- [[gpu_memory_utilization 0.35]]
- [[Tensor parallel size 4]]
- [[Eager mode and cache free]]

Up: [[RTX OFFPOLICY-GRPO]]
