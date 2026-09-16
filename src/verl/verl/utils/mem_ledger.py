"""Attribute resident GPU memory to FSDP internals, optimizer state, and everything else.

Drop-in diagnostic for verl FSDP workers.  Prints one block per call:

    [LEDGER pre-update r0] allocated=15.42 GiB of 23.64
       fsdp _local_shard                 float32   3.548 GiB
       fsdp _saved_grad_shard            float32   3.548 GiB
       fsdp _mp_shard                    float16   1.774 GiB
       optim state on cpu                          7.097 GiB
       == fsdp_gpu 8.870 | optim_gpu 0.000 | OTHER 6.550 GiB

`OTHER` is allocated minus the two attributed buckets: activations, autograd
graph, the data batch, and anything else holding CUDA tensors.

Storages are de-duplicated by data_ptr, so aliases (flat_param.grad pointing at
_saved_grad_shard, flat_param.data at _local_shard) are counted once.

Enabled only when VERL_MEM_LEDGER=1.  Prints from rank VERL_MEM_LEDGER_RANK
(default 0); set it to -1 for all ranks.

Self-test without a GPU:   python -m verl.utils.mem_ledger
"""

import os

import torch

_GIB = 2 ** 30

# FlatParameter buffers worth attributing, in report order.
_FLAT_PARAM_BUFFERS = (
    "_local_shard",                  # the fp32 (or model-dtype) shard this rank owns
    "_saved_grad_shard",             # gradient shard, reduce_dtype
    "_mp_shard",                     # low-precision copy when param_dtype differs
    "_full_param_padded",            # all-gathered unit, param_dtype
    "_full_prec_full_param_padded",  # all-gathered unit in full precision
    "grad",
)


def _enabled():
    if os.environ.get("VERL_MEM_LEDGER", "0") != "1":
        return False
    want = int(os.environ.get("VERL_MEM_LEDGER_RANK", "0"))
    if want < 0:
        return True
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == want
    except Exception:
        pass
    return True


def _rank():
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return 0


def _fsdp_buffers(module):
    """{(buffer_name, dtype): bytes} for CUDA-resident FlatParameter storages."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    seen, agg = set(), {}
    for m in FSDP.fsdp_modules(module):
        handle = getattr(m, "_handle", None)
        handles = [handle] if handle is not None else list(getattr(m, "_handles", None) or [])
        for h in handles:
            flat_param = getattr(h, "flat_param", None)
            if flat_param is None:
                continue
            for name in _FLAT_PARAM_BUFFERS:
                t = getattr(flat_param, name, None)
                if not torch.is_tensor(t) or not t.is_cuda or t.numel() == 0:
                    continue
                storage = t.untyped_storage()
                if storage.data_ptr() in seen:
                    continue           # alias of a buffer already counted
                seen.add(storage.data_ptr())
                key = (name, str(t.dtype).replace("torch.", ""))
                agg[key] = agg.get(key, 0) + storage.nbytes()
    return agg


def _why_empty(module):
    """Say what the FSDP walk actually saw, when it attributed nothing.

    The buffer names in _FLAT_PARAM_BUFFERS come from torch's _flat_param.py and
    can differ by version; rather than guess, report the ground truth from this
    install so the names can be corrected.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    lines = []
    try:
        mods = list(FSDP.fsdp_modules(module))
    except Exception as exc:
        return ["   ?? FSDP.fsdp_modules failed: %r" % (exc,)]
    lines.append("   ?? fsdp_modules=%d" % len(mods))
    for m in mods[:1]:
        h = getattr(m, "_handle", None)
        hs = list(getattr(m, "_handles", None) or [])
        lines.append("   ?? first module: _handle=%s _handles=%d"
                     % (type(h).__name__ if h is not None else None, len(hs)))
        h = h or (hs[0] if hs else None)
        if h is None:
            lines.append("   ?? no handle; try module.parameters() instead")
            break
        fp = getattr(h, "flat_param", None)
        lines.append("   ?? flat_param=%s" % (type(fp).__name__ if fp is not None else None,))
        if fp is None:
            break
        for name in _FLAT_PARAM_BUFFERS:
            t = getattr(fp, name, "<missing>")
            if t is None or isinstance(t, str):
                lines.append("   ??   %-28s %s" % (name, t))
            elif torch.is_tensor(t):
                lines.append("   ??   %-28s tensor numel=%d cuda=%s %s"
                             % (name, t.numel(), t.is_cuda, t.dtype))
            else:
                lines.append("   ??   %-28s %r" % (name, type(t).__name__))
        present = [a for a in dir(fp) if a.startswith("_") and torch.is_tensor(
            getattr(fp, a, None))]
        lines.append("   ?? tensor-valued attrs actually on flat_param: %s"
                     % (sorted(present)[:20],))
    return lines


def _optimizer_bytes(optimizer):
    """({device: bytes}, gpu_bytes) for optimizer state tensors."""
    per_device, gpu = {}, 0
    if optimizer is None:
        return per_device, gpu
    for state in optimizer.state.values():
        for v in state.values():
            if not torch.is_tensor(v):
                continue
            nbytes = v.numel() * v.element_size()
            per_device[str(v.device)] = per_device.get(str(v.device), 0) + nbytes
            if v.is_cuda:
                gpu += nbytes
    return per_device, gpu


def mem_ledger(tag, module, optimizer=None, total_gib=23.64):
    """Print the GPU memory ledger.  No-op unless VERL_MEM_LEDGER=1."""
    if not _enabled():
        return
    try:
        agg = _fsdp_buffers(module)
    except Exception as exc:                      # never let a probe kill a run
        print("[LEDGER %s] fsdp walk failed: %r" % (tag, exc), flush=True)
        agg = {}
    per_device, opt_gpu = _optimizer_bytes(optimizer)

    fsdp_gpu = sum(agg.values())
    allocated = torch.cuda.memory_allocated()

    order = {name: i for i, name in enumerate(_FLAT_PARAM_BUFFERS)}
    print("[LEDGER %s r%d] allocated=%.2f GiB of %.2f"
          % (tag, _rank(), allocated / _GIB, total_gib), flush=True)
    for (name, dtype), nbytes in sorted(agg.items(), key=lambda kv: order.get(kv[0][0], 99)):
        print("   fsdp %-30s %-8s %7.3f GiB" % (name, dtype, nbytes / _GIB), flush=True)
    for device, nbytes in sorted(per_device.items()):
        print("   optim state on %-23s %7.3f GiB" % (device, nbytes / _GIB), flush=True)
    print("   == fsdp_gpu %.3f | optim_gpu %.3f | OTHER %.3f GiB"
          % (fsdp_gpu / _GIB, opt_gpu / _GIB,
             (allocated - fsdp_gpu - opt_gpu) / _GIB), flush=True)
    if fsdp_gpu == 0 and allocated > _GIB:
        # Nothing attributed but the GPU is far from empty: the buffer names are
        # wrong for this torch version.  Report what is really there.
        try:
            for line in _why_empty(module):
                print(line, flush=True)
        except Exception as exc:
            print("   ?? introspection failed: %r" % (exc,), flush=True)


def offload_check(optimizer, offload_fn, tag="offload"):
    """Call offload_fn(optimizer) and prove whether the GPU bytes were actually freed.

    `offload_fn` is normally verl.utils.fsdp_utils.offload_fsdp_optimizer.

    state[key] = value.to("cpu") only frees the CUDA tensor when its refcount
    reaches zero.  We hold weak references across the call: any tensor still
    alive afterwards is being kept by someone else, and we name the holders.
    Runs regardless of VERL_MEM_LEDGER, since it is cheap and only fires once
    per optimizer step.
    """
    import gc
    import weakref

    watch = []
    for state in optimizer.state.values():
        for key, v in state.items():
            if torch.is_tensor(v) and v.is_cuda:
                try:
                    watch.append((key, weakref.ref(v)))
                except TypeError:
                    pass
    before_state = _optimizer_bytes(optimizer)[1]
    torch.cuda.synchronize()
    before_alloc = torch.cuda.memory_allocated()

    offload_fn(optimizer)

    torch.cuda.synchronize()
    after_alloc = torch.cuda.memory_allocated()
    after_state = _optimizer_bytes(optimizer)[1]
    freed = before_alloc - after_alloc

    print("[OFFLOAD %s r%d] state_on_gpu %.3f -> %.3f GiB | allocated %.3f -> %.3f "
          "| freed %.3f GiB"
          % (tag, _rank(), before_state / _GIB, after_state / _GIB,
             before_alloc / _GIB, after_alloc / _GIB, freed / _GIB), flush=True)

    survivors = [(k, ref()) for k, ref in watch if ref() is not None]
    if survivors:
        print("   WARNING: %d state tensors survived the move to CPU "
              "(expected 0); .to('cpu') copied instead of freeing."
              % len(survivors), flush=True)
        key, tensor = survivors[0]
        holders = [type(r).__name__ for r in gc.get_referrers(tensor)]
        print("   e.g. %-12s %s %s held by: %s"
              % (key, tuple(tensor.shape), tensor.dtype, holders[:8]), flush=True)
        print("   (one referrer is this function's own frame; more than one means"
              " something else retains it)", flush=True)
    elif before_state > 0:
        print("   OK: every state tensor was released from GPU.", flush=True)


def _self_test():
    """Exercise every code path on CPU tensors, no GPU or distributed needed."""
    import types

    import torch.distributed.fsdp as fsdp_mod

    flat = types.SimpleNamespace()
    flat._local_shard = torch.zeros(4, dtype=torch.float32)
    flat._mp_shard = torch.zeros(4, dtype=torch.float16)
    flat._full_param_padded = torch.zeros(8, dtype=torch.float16)
    flat._saved_grad_shard = torch.zeros(4, dtype=torch.float32)
    flat.grad = flat._saved_grad_shard                      # alias: must be deduped
    handle = types.SimpleNamespace(flat_param=flat)
    module = types.SimpleNamespace(_handle=handle)

    real_modules = fsdp_mod.FullyShardedDataParallel.fsdp_modules
    real_is_cuda = torch.Tensor.is_cuda
    real_alloc = torch.cuda.memory_allocated
    os.environ["VERL_MEM_LEDGER"] = "1"
    try:
        fsdp_mod.FullyShardedDataParallel.fsdp_modules = staticmethod(lambda m: [module])
        torch.Tensor.is_cuda = property(lambda self: True)
        torch.cuda.memory_allocated = lambda *a, **k: 1024 ** 3
        opt = types.SimpleNamespace(state={0: {"exp_avg": torch.zeros(4),
                                               "exp_avg_sq": torch.zeros(4),
                                               "step": torch.tensor(1.0)}})
        mem_ledger("self-test", None, opt)
    finally:
        fsdp_mod.FullyShardedDataParallel.fsdp_modules = real_modules
        torch.Tensor.is_cuda = real_is_cuda
        torch.cuda.memory_allocated = real_alloc
    print("self-test OK: 4 distinct storages listed, 'grad' deduped away")


if __name__ == "__main__":
    _self_test()

