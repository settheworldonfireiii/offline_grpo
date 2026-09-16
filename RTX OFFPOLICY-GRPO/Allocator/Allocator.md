---
branch: allocator
---
#branch/allocator

# Allocator

Caching-allocator tuning. One setting applied, parsed at runtime, and refused by the
platform.

- [[expandable_segments is inert]]

**Ruled out, never applied:** `max_split_size_mb:256`. It would cap the size of blocks the
allocator is willing to split, which collides directly with the 37 MiB chunk allocations
from [[Chunk size 512 to 128]] — capping splits would force fresh segments for exactly the
tensor the chunking exists to keep small. Rejected on that argument, not tested.

Up: [[RTX OFFPOLICY-GRPO]]
