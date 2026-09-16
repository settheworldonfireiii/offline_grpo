"""Standalone rehearsal of one update_actor micro-step, without Ray, vLLM, or data.

Reproduces exactly what mix_fsdp_worker._build_model_optimizer does — same dtypes,
same wrap policy, same MixedPrecision, same FULL_SHARD mesh, same AdamW — then runs
two forward/backward passes around one optimizer step and prints the memory ledger
at every boundary.  Answers, in ~2 minutes instead of a 15-minute training run:

  * how big _mp_shard and _full_param_padded actually are
  * how much of the peak is activations ("OTHER")
  * whether offload_fsdp_optimizer really frees the GPU bytes
  * what the backward peak is with optimizer state on CPU

Run from the verl source root so `import verl` works:

    cd /mnt/data2/radke149/offline-grpo/src/verl
    VERL_MEM_LEDGER=1 VERL_MEM_LEDGER_RANK=0 \
    torchrun --nproc_per_node=8 verl/utils/probe_fsdp_mem.py

Env knobs: MODEL_PATH, PROMPT_LEN (1024), RESP_LEN (8192), TOTAL_GIB (23.64).
"""

import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoConfig, AutoModelForCausalLM

# Importing dp_actor installs the chunked fused LM head onto Qwen2ForCausalLM.forward.
import verl.workers.actor.dp_actor  # noqa: F401
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    offload_fsdp_optimizer,
)
from verl.utils.mem_ledger import mem_ledger, offload_check

MODEL_PATH = os.environ.get("MODEL_PATH", "open-thoughts/OpenThinker3-7B")
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "1024"))
RESP_LEN = int(os.environ.get("RESP_LEN", "8192"))
TOTAL_GIB = float(os.environ.get("TOTAL_GIB", "23.64"))
LR = float(os.environ.get("LR", "1e-7"))


def log(msg):
    if dist.get_rank() == 0:
        print("== %s" % msg, flush=True)


def build():
    """Mirror mix_fsdp_worker._build_model_optimizer for role='actor'."""
    world_size = dist.get_world_size()

    model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=False)
    init_ctx = get_init_weight_context_manager(
        use_meta_tensor=not model_config.tie_word_embeddings
    )
    torch_dtype = torch.float32                      # role == "actor"

    with init_ctx():
        module = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_PATH,
            torch_dtype=torch_dtype,
            config=model_config,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        module.to(torch_dtype)
        module.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    dist.barrier()

    auto_wrap_policy = get_fsdp_wrap_policy(
        module=module, config=OmegaConf.create({"min_num_params": 0})
    )
    # create_device_mesh(world_size, fsdp_size=-1) + get_sharding_strategy(1-D mesh)
    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

    fsdp_module = FSDP(
        module,
        cpu_offload=None,                            # actor
        param_init_fn=init_fn,
        use_orig_params=False,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        sync_module_states=True,
        device_mesh=device_mesh,
        forward_prefetch=False,
    )
    # verl reaches update_actor through fsdp_vllm.FSDPVLLMShardingManager.__exit__,
    # which calls self.module.train().  from_pretrained leaves the model in eval
    # mode, and HF only applies gradient checkpointing when self.training is True,
    # so without this the probe stores all 28 layers of activations.
    fsdp_module.train()

    optimizer = optim.AdamW(
        fsdp_module.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-2
    )
    return fsdp_module, optimizer


def one_pass(model, tag):
    """One forward+backward on a synthetic full-length sequence; returns peak GiB."""
    seq = PROMPT_LEN + RESP_LEN
    dev = torch.cuda.current_device()
    input_ids = torch.randint(0, 100, (1, seq), device=dev)
    attention_mask = torch.ones((1, seq), dtype=torch.long, device=dev)
    position_ids = torch.arange(seq, device=dev).unsqueeze(0)
    responses = input_ids[:, -RESP_LEN:]

    assert model.training, "model is in eval mode: gradient checkpointing is inactive"
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        log_probs, entropy = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            ppo_labels=responses,
            temperature=1.0,
        )
    loss = -log_probs.mean() - 0.001 * entropy.mean()
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    log("%s: loss=%.6f  backward peak=%.2f GiB of %.2f" % (tag, loss.item(), peak, TOTAL_GIB))
    return peak


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    os.environ.setdefault("VERL_MEM_LEDGER", "1")

    log("building model  path=%s  seq=%d+%d  world=%d"
        % (MODEL_PATH, PROMPT_LEN, RESP_LEN, dist.get_world_size()))
    model, optimizer = build()
    mem_ledger("after-fsdp-init", model, optimizer, TOTAL_GIB)

    one_pass(model, "pass 1 (no optimizer state yet)")
    mem_ledger("after-backward-1", model, optimizer, TOTAL_GIB)

    model.clip_grad_norm_(max_norm=1.0)
    optimizer.step()
    mem_ledger("after-step (state just created on GPU)", model, optimizer, TOTAL_GIB)

    offload_check(optimizer, offload_fsdp_optimizer, tag="after-step")
    mem_ledger("after-offload", model, optimizer, TOTAL_GIB)

    optimizer.zero_grad(set_to_none=True)
    one_pass(model, "pass 2 (optimizer state on CPU)")
    mem_ledger("after-backward-2", model, optimizer, TOTAL_GIB)

    log("done")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

