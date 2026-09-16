---
branch: numerics
---
#branch/numerics

# Numerics and Precision

Which dtype each tensor lives in. Separate from [[Turing Compatibility]], which is about
*which kernels exist*; this branch is about *which numbers survive*.

The whole branch exists because of one failure: AdamW cannot run with fp16 master weights.

- [[Actor master weights in fp32]]
- [[FSDP compute dtype fp16]]
- [[Reference policy stays fp16]]

Up: [[RTX OFFPOLICY-GRPO]]
