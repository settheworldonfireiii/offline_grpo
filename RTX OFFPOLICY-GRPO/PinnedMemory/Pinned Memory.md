---
branch: pinned-memory
---
#branch/pinned-memory

# Pinned Memory

`cudaHostAlloc` fails on this box. One line neutralises every pinned allocation in the
process. Kept as its own branch because it is a host-side driver defect, not a memory
budget decision — but it silently taxes [[Offloading]].

- [[pin_memory neutered]]

Up: [[RTX OFFPOLICY-GRPO]]
