---
branch: offloading
---
#branch/offloading

# Offloading

**All** movement of state between GPU and host lives here, whatever the state is —
optimizer, parameters, gradients, reference weights. This is the branch that fixed the
OOM.

The insight: FSDP `FULL_SHARD` already shards optimizer state 8 ways, so 7.09 GiB per
rank *is* the shard. It cannot be split further across GPUs. The only remaining axis is
time — hold it on the host except during `optimizer.step()`.

- [[Optimizer state offload around step]]
- [[Eager optimizer load removed]]
- [[Actor param and grad offload]]
- [[Reference policy param offload]]

Taxed by [[pin_memory neutered]] — every transfer here is synchronous and pageable.

Up: [[RTX OFFPOLICY-GRPO]]
