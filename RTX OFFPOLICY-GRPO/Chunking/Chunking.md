---
branch: chunking
---
#branch/chunking

# Chunking

Never materialise the full `[1, 8192, 152064]` logits tensor. The vocabulary is
152,064 wide; one full-sequence logits tensor in fp16 is **2.32 GiB**, and the softmax
and entropy intermediates multiply that.

- [[Fused linear PPO head]]
- [[Qwen2 forward monkeypatch]]
- [[Chunk size 512 to 128]]

Up: [[RTX OFFPOLICY-GRPO]]
