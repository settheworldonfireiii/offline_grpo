---
branch: allocator
status: parsed-but-refused
---
#branch/allocator #status/inert

# expandable_segments is inert

**File:** `exp_scripts/train_openthinker3.sh`, line 4

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Why it was set

The intent was to let the caching allocator grow a segment in place rather than reserve
fixed-size blocks, reducing fragmentation between the 64 MiB activation allocations that
dominate the backward.

## What actually happens

It **is** read — PyTorch parses the variable, evaluates support, and refuses. Every run
prints, once per worker:

```
[W915 23:36:21.919866224 CUDAAllocatorConfig.h:28] Warning: expandable_segments
not supported on this platform (function operator())
```

so this node is included: the code path executes. It simply has **no effect** on this
driver/GPU combination, and the allocator falls back to its default behaviour.

## Consequence

Fragmentation had to be solved structurally rather than by allocator tuning — by keeping
every allocation small and predictable. That is what [[One sequence per micro-batch]],
[[Chunk size 512 to 128]] and [[Offloading]] between them accomplish.

Left in the script because it is harmless and self-documenting; it costs one warning line
per worker at startup.

Up: [[Allocator]]
