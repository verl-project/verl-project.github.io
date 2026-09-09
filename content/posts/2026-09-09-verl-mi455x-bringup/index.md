---
title: "Bringing End-to-End LLM Reinforcement Learning to AMD Instinct MI455X with verl"
date: 2026-09-09
authors:
  - "Yuhan Yang"
  - "Liz Li"
  - "Xibin Wu"
  - "Yanyuan Qin"
  - "Fuwei Yang"
  - "Yuankai Chen"
  - "Andy Luo"
  - "Xinyu Kang"
  - "Yao Fu"
summary: "A complete GRPO workflow using verl, Qwen3-30B-A3B, SGLang, and PyTorch FSDP ran for 100 consecutive steps on four AMD Instinct MI455X GPUs, establishing an end-to-end RL baseline for CDNA 5."
tags:
  - amd
  - rocm
  - training
  - sglang
  - fsdp
  - case-study
  - mi455x
  - cdna5
math: false
toc: true
---

AMD Instinct™ MI455X is the first accelerator based on the fifth-generation AMD
CDNA™ architecture. Bringing up a new GPU generation, however, goes beyond
hardware enablement: real-world AI workloads depend on the entire software
stack—from frameworks and optimized kernels to communication libraries—working
together reliably and efficiently end to end.

In this post, we walk through the enablement of an end-to-end Group Relative
Policy Optimization (GRPO) workflow on MI455X, integrating
[verl](https://github.com/verl-project/verl), Qwen3-30B-A3B, SGLang, and PyTorch
FSDP. Our goal is to validate the complete reinforcement learning loop—including
rollout generation, policy training, and weight synchronization—on the new CDNA
5 platform, and to identify and resolve the cross-stack challenges required to
make the workflow run reliably.

## Result at a glance

- **End-to-end execution:** 100 consecutive GRPO steps completed on four MI455X
  GPUs.
- **Full workflow:** every step covered rollout generation, actor-side
  probability computation and optimization, weight synchronization, and
  transition into the next rollout.
- **Numerical consistency:** rollout-to-actor sampled-token probability
  correlation was approximately `0.9965`, with no visible drift.
- **Runtime stability:** the final configuration had no out-of-memory error,
  kernel fault, or collective timeout, and peak allocated memory remained flat
  after warm-up.

## AMD Instinct MI455X: bringing RL to CDNA 5

AMD launched the Instinct MI400 Series at Advancing AI 2026. MI455X is the first
Instinct accelerator based on AMD CDNA 5 and sits at the center of the 72-GPU
AMD Helios™ rack-scale system.[^1][^2]

| Specification         | AMD Instinct MI355X | AMD Instinct MI455X |
| :-------------------: | :-----------------: | :-----------------: |
| Architecture          | AMD CDNA 4          | AMD CDNA 5          |
| High-bandwidth memory | 288 GB HBM3E        | 432 GB HBM4         |
| Peak memory bandwidth | 8 TB/s              | Up to 23.3 TB/s     |

*Table 1. Public specifications for MI355X and MI455X. Sources:* [^2][^3]

Compared with MI355X, MI455X increases per-GPU HBM capacity by 50% and peak
memory bandwidth by about 2.9×. At the package level, eight 3D hybrid-bonded
CDNA 5 compute chiplets are paired with two fabric-and-cache dies, 192 MB of
global L2 cache, and 12 HBM4 stacks. CDNA 5 also changes the execution and
data-movement model exposed to software. MI455X provides 256 Work Group
Processors (WGPs) with Wave32 execution. A Tensor Data Mover supports
asynchronous global-memory-to-LDS transfers without register staging, while L2
multicast and WGP clustering reduce redundant traffic and coordination
overhead.

For end-to-end RL, memory capacity is only part of the story. Model states,
training activations, rollout KV caches, and updated weights must flow
efficiently across kernels, process groups, and framework boundaries. On a new
`gfx1250` platform, each of these paths must be enabled and validated as part of
a complete system.

The end-to-end RL loop therefore provides a practical test of software readiness
on MI455X. It exercises the full workflow—from SGLang rollout generation and
PyTorch FSDP actor training to weight synchronization between the rollout and
training engines—exposing integration and performance challenges that may not
surface when individual components are tested in isolation.

## What the end-to-end RL loop exercises

{{< figure src="verl-rl-loop.svg" alt="End-to-end verl GRPO loop on four AMD Instinct MI455X GPUs" caption="Figure 1. The end-to-end RL loop exercised during each training step." width="100%" >}}

One training step crosses two execution stacks. SGLang uses tensor parallelism
to generate multiple responses per prompt and returns sampled-token log
probabilities. verl scores the responses and computes GRPO group-relative
advantages;[^4][^5] PyTorch FSDP then recomputes actor probabilities and applies
the sharded update. Before the next rollout, the updated weights are broadcast
back into the SGLang schedulers. This tests two boundaries that isolated
inference or training does not: numerical agreement between the rollout and
actor paths, and model-state transfer across their process groups.

## What it took to run on MI455X

Enabling the full verl workflow on MI455X required work across rollout kernels,
actor training, checkpoint loading, and weight synchronization.

### Rollout: route kernels per operation

SGLang required selecting kernel paths per operation on `gfx1250`. AITER provided
MoE top-k routing, while Triton handled attention and expert GEMMs, enabling
stable generation and CUDA graph execution.

### Actor: enable the CDNA 5 training stack

We enabled the actor stack for `gfx1250` by building Transformer Engine and Apex
with CDNA 5 targets, then selecting PyTorch SDPA for attention and Triton for
grouped GEMM. This configuration provided the BF16 execution path required for
stable FSDP actor training.

### Checkpoint loading: reduce host-memory pressure

Checkpoint loading initially materialized the model in FP32, causing excessive
host-memory usage. Loading directly in BF16 substantially reduced memory pressure
and enabled reliable initialization.

### Weight synchronization: complete the RL loop

Actor-to-rollout weight synchronization required two adjustments: selecting the
RCCL path by disabling MSCCL/MSCCL++, and adapting verl to SGLang's new
session-based weight-update API. With these changes, updated actor weights
synchronized reliably after every training step, completing the end-to-end GRPO
loop.

## Reproducing the MI455X run

### Workload configuration

- **Hardware:** 4 × AMD Instinct MI455X (`gfx1250`), 432 GB HBM4 per GPU, single
  node, driver `7.1.1.31300009`
- **Model / data:** Qwen3-30B-A3B, a 128-expert MoE model,[^6] on
  DAPO-Math-17k[^7]
- **Algorithm:** GRPO, learning rate `1e-6`, 8 prompts × 8 responses per step,
  8,192-token maximum response
- **Engines:** SGLang 0.5.19 rollout at TP=4; PyTorch FSDP actor across the same
  four GPUs
- **Software:** ROCm 10.0.0, PyTorch 2.11.0+rocm10.0.0, Transformer Engine
  2.17.0, Megatron-LM 0.19, verl 0.10.0.dev
- **Run:** 100 training steps, 8.0 hours wall clock

### Launch command

The image carries the `gfx1250` environment and both compatibility patches; the
Dockerfile, launch script, and patches are published in full.[^8]

```bash
docker pull amdagi/verl-dev:verl-rocm10-mi45x

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NCCL_IB_DISABLE=1 \
  -v $MODELS:/root/models -v $DATA:/root/datasets \
  amdagi/verl-dev:verl-rocm10-mi45x \
  bash examples/grpo_trainer/run_qwen3_30b_a3b_mi45x.sh
```

### MI455X-specific image configuration

These are compatibility settings, not tuning:

- **Rollout:** `SGLANG_USE_AITER=1`, `SGLANG_ATTENTION_BACKEND=triton`, and
  `moe_runner_backend=triton`.
- **Training:** `NVTE_USE_GROUPED_GEMM_TRITON=1`, `model_dtype=bf16`,
  `attn_implementation=sdpa`, and `ENABLE_CK=0`.
- **Collectives and memory:** set `RCCL_MSCCL_ENABLE=0`,
  `RCCL_MSCCLPP_ENABLE=0`,
  `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0`, `GPU_MAX_HW_QUEUES=2`, and
  `PYTORCH_ALLOC_CONF=expandable_segments:True`; clear `NCCL_MIN_NCHANNELS`.

`NCCL_IB_DISABLE=1` is supplied only at run time for this single-node job.

## Validation results

### Rollout-to-actor probability consistency

Rollout generation and actor training evaluate the same sampled tokens through
different execution paths. To verify that the two paths remained numerically
aligned, we compared their sampled-token probabilities at every training step.

{{< figure src="rollout-actor-diagnostics.jpg" alt="Rollout-to-actor probability diagnostics across 100 steps" caption="Figure 2. Rollout-to-actor diagnostics over 100 steps: (a) Pearson correlation, mean `0.9965` (σ `0.0003`), (b) mean absolute sampled-token probability difference, mean `0.0077` over a `0.0063`–`0.0094` range, and (c) sampled forward KL, mean `0.00183`." width="100%" >}}

Across 100 steps, all three metrics remained within a narrow range with no
visible drift, indicating stable numerical agreement between the rollout and
actor paths. These measurements provide a sampled-token consistency baseline
for tracking future kernel and framework changes.

### Training stability

{{< figure src="training-stability.jpg" alt="Actor gradient norm and peak memory allocation across 100 steps" caption="Figure 3. (a) Actor gradient norm, mean `0.124`, and (b) peak allocated memory per card, flat at `75.6 GB` from step 11 onward." width="100%" >}}

The gradient norm remained bounded throughout the run, while peak allocated
memory stabilized at `75.6 GB` per GPU from step 11 onward.

### Where a step spends its time

| Phase                  | Share of step wall clock |
| :--------------------: | :----------------------: |
| Rollout generation     | 48.1%                    |
| Actor update           | 39.1%                    |
| Log-probability pass   | 10.9%                    |
| Weight synchronization | 1.8%                     |

*Table 2. Share of step wall clock by phase, averaged over 100 steps.*

## Conclusion

This work marks the first end-to-end enablement of verl on AMD Instinct™ MI455X,
demonstrating a complete GRPO workflow on the new CDNA 5 platform. A 30B-class
MoE model can generate rollouts with SGLang, train with PyTorch FSDP, and
synchronize updated weights every step, with stable sampled-token diagnostics
across 100 updates.

This milestone establishes a functional RL baseline on MI455X rather than a
claim about peak performance or scaling. It provides a foundation for broader
model coverage, longer runs, multi-node scaling, and continued performance
optimization on CDNA 5.

## References

[^1]: AMD Newsroom, "[AAI 2026: AMD Launches AMD Instinct MI400 Series GPUs for Frontier AI, HPC](https://newsroom.amd.com/news/aai-2026-mi400-instinct-update/)," 2026.

[^2]: AMD, "[AMD CDNA™ Architecture](https://www.amd.com/en/technologies/cdna.html)."

[^3]: AMD, "[AMD Instinct™ MI355X GPUs](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)."

[^4]: G. Sheng et al., "[HybridFlow: A Flexible and Efficient RLHF Framework](https://doi.org/10.1145/3689031.3696075)," *EuroSys '25*, 2025.

[^5]: Z. Shao et al., "[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Large Language Models](https://arxiv.org/abs/2402.03300)," arXiv:2402.03300, 2024.

[^6]: A. Yang et al., "[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)," arXiv:2505.09388, 2025.

[^7]: Q. Yu et al., "[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)," arXiv:2503.14476, 2025.

[^8]: "[MI455X verl bring-up source: Dockerfile, launch scripts, and compatibility patches](https://github.com/lizamd/verl/commit/d74323771ca870ade550df358aadac3a4d63ad01)," commit `d743237`.
