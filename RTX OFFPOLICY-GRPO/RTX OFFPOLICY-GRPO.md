# RTX OFFPOLICY-GRPO

Port of `krafton-ai/Offline-GRPO` (verl 0.1 + `mix_src` off-policy GRPO, vLLM 0.6.3)
onto **cs-u-iteb361**: 8x Quadro RTX 6000, 24 GiB each, **sm_75 (Turing)**.

Fork: `settheworldonfireiii/offline_grpo`, branch `RTX6000`, HEAD `d800bd9`.
Repo root on box: `/mnt/data2/radke149/offline-grpo`. Env: conda `offline-grpo`, torch 2.4.0.

## Fixed experimental constraints (never modified)

- `data.max_response_length=8192`, `actor.ppo_max_token_len_per_gpu=9216`
- Pure teacher forcing: `n_prefix = rollout.n = 8`, `min_prefix_ratio = max_prefix_ratio = 1.0`
- Every scientific knob in `exp_scripts/train_openthinker3.sh` is the author's

## Outcome

| stage | peak GiB of 23.64 |
|---|---|
| OOM before the work | 22.70 (died in backward) |
| backward, first micro-batch | 13.61 |
| backward, steady | 15.94 |
| backward + optimizer step window | **17.90** |

Headroom after the work: **~5.7 GiB**.

## Inclusion rule

Every node is a change that **executes** in this pipeline
(`python3 -m verl.mix_src.main_mix_ppo`, fsdp strategy, grpo estimator).

Two changes were written and are deliberately **absent from this graph**, each only after
every counter-argument for their being live was constructed and closed:

- **`[GUARD]` non-finite grad_norm skip**, `dp_actor.py:218-221`. Unreachable:
  `MIXDataParallelPPOActor` overrides `_optimizer_step`, the reference policy never calls
  `update_policy` and is built with `actor_optimizer=None`, `use_critic` is False, and the
  entry points that would reach it (`main_ppo`, `main_generation`, `main_ppo_new_reward`,
  megatron) are not the module we launch.
- **`logprob_chunk_size` command-line keys**. No code reads them and no yaml interpolates
  them; the real control is the literal at `dp_actor.py:45`. Recorded as prose inside
  [[Chunk size 512 to 128]] because it is a live trap, not a live change.

Levers evaluated and ruled out — Ulysses sequence parallel, `max_split_size_mb`, lowering
vLLM `gpu_memory_utilization`, reducing batch size — are recorded inside the branch hub
they belong to, not as nodes, since nothing was modified.

## Branches

One branch per aspect. All offloading lives in a single branch regardless of what is
being offloaded; chunking, precision, and instrumentation are each their own.

- [[Numerics and Precision]]
- [[Turing Compatibility]]
- [[Pinned Memory]]
- [[Chunking]]
- [[Micro-batch Sizing]]
- [[Offloading]]
- [[Rollout Memory]]
- [[Instrumentation]]
- [[Allocator]]
