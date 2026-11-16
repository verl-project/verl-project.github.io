---
title: "verl-SpeCo v0.2.0: Faster Standalone Training and Broader Online Co-Training"
date: 2026-11-16
authors:
  - "PeiXiang Tan"
  - "Shuang Li"
  - "Shuai Yuan"
  - "RunPing Chen"
  - "Jihuan Li"
  - "YiRan Xiong"
author_links:
  "PeiXiang Tan": "https://github.com/tpx818"
  "Shuang Li": "https://github.com/755651978"
  "Shuai Yuan": "https://github.com/crp0128"
  "RunPing Chen":  "https://github.com/ys2025-AI"
  "Jihuan Li": "https://github.com/vx120"
  "YiRan Xiong": "https://github.com/xxyyrr598"
summary: "verl-SpeCo v0.2.0 pipelines standalone drafter training, extends online DSpark co-training to V1, OPD, and SGLang, and adds controls for when and on what to train."
image: "cover.svg"
tags:
  - dspark
  - dflash
  - speculative decoding
  - co-train
math: false
toc: true
---

## 1. Overview

verl-SpeCo v0.2.0 improves drafter training efficiency and expands online co-training. Pipelining accelerates standalone training, while online co-training extends to the V1 trainer, OPD, and the SGLang rollout backend. Idle-window scheduling and drafter freezing reduce the impact of drafter training and feature collection on the main training loop while preserving acceptance behavior. Sampling low-acceptance-length requests further improves mean acceptance length.

**Release:** [v0.2.0 release notes](https://github.com/verl-project/verl-SpeCo/releases/tag/v0.2.0) · **Code:** [verl-SpeCo](https://github.com/verl-project/verl-SpeCo) · **Previous release:** [v0.1.0 blog](https://verl-project.github.io/posts/2026-08-13-verl-speco-release/)

### Expanded Standalone and Co-Training Support

- **Standalone Drafter Training with Parallel Feature Generation.** Overlap feature generation and training to reduce waiting. Compared with Speculators, data preparation time is about **40% lower** and mean step time about **5% lower**.
- **Online Co-Training with the V1 Trainer.** Support Synchronous, Colocated Async, and Separate Async execution. In the 16-NPU MRV2 experiment, DSpark online co-training recorded roughly **40%–55% lower step times** than the synchronous no-drafter baseline.
- **OPD Co-Training.** Update the drafter online during synchronous OPD so it adapts as the student changes. The Qwen3-8B experiment increased mean acceptance length by **7.1%** and reduced generation time by about **1.0%**.
- **DSpark Co-Training Integration with SGLang.** Publish updated DSpark drafter weights to SGLang for subsequent rollouts. In the Qwen3-4B experiment, mean acceptance length increased by **5.0%** and generation time decreased by **3.0%**.

### Training Scheduling and Sampling Optimizations

- **Rollout Idle-Window Training (Bubble Time).** Train the drafter during idle rollout windows to reduce waiting in the main training loop. In the Qwen3-8B + DSpark experiment, mean step time decreased by about **4.0%**, and mean logged critical-path drafter time decreased by **79.2%**.
- **Drafter Training Freeze.** Pause training and feature collection when further updates offer limited gains. In a 200-step experiment, freezing triggered at step **126**; drafter training time then fell to **zero**, with mean acceptance length remaining stable through step **200**.
- **Sampling Low-Acceptance-Length Requests (Hard Sampling).** Increase their share of training under the same collection budget. Mean acceptance length pulled ahead of the baseline from around step **40**.

## 2. Expanded Standalone and Co-Training Support

For standalone training, the Ray reference pipeline overlaps feature generation, data preparation, and drafter training to reduce waiting. Online DSpark co-training now extends to the V1 trainer, OPD, and the SGLang rollout backend, covering more training workflows and resource layouts.

### 2.1 Standalone Drafter Training with Parallel Feature Generation

verl-SpeCo v0.2.0 pipelines feature generation, data preparation, and drafter training. Ray schedules the feature producer and the training worker group independently, so training can begin as soon as the first complete batch is ready. While that batch trains, the producer can generate more features and training workers can load later batches, reducing time spent waiting for data.

- **Pass references and prepare data on the training workers.** The producer sends batches of feature references and request metadata. Training workers use those references to read hidden states from files or Mooncake, then align, crop, and assemble batches in parallel. This removes hidden-state loading and tensor encoding from the producer's workload.
- **Use bounded prefetching to keep training supplied.** Workers prefetch later batches while the current batch trains. The scheduler uses queue watermarks to apply backpressure when work builds up and dispatches complete batches to the training worker group. This overlaps data preparation with training while limiting queued work. The subprocess-based TransferQueue path remains available.

**Standalone training performance.** SpeCo used the Ray reference path in the comparison with Speculators. Estimates from the plotted points put mean data preparation time at **0.09 s versus 0.15 s**, about **40% lower**. Mean training step time was **0.37 s versus 0.39 s**, about **5% lower**, with smaller fluctuations across the plotted steps. [PR #103](https://github.com/verl-project/verl-SpeCo/pull/103) · [PR #129](https://github.com/verl-project/verl-SpeCo/pull/129).

{{< figure src="standalone-speculators-comparison.png" alt="SpeCo and Speculators compared on standalone drafter training step time and data preparation time" caption="Standalone drafter training with SpeCo using the Ray reference path and Speculators. The upper panel shows training step time and the lower panel shows data preparation time. The reported averages are estimates from the plotted points." >}}

### 2.2 Online Co-Training with the V1 Trainer

V1 is verl's unified RL trainer for synchronous and asynchronous execution. verl-SpeCo v0.2.0 integrates online DSpark co-training into V1, with three modes for different training and rollout resource layouts.

- **Synchronous.** Update the drafter at a training synchronization point.
- **Colocated Async.** Training and rollout share devices, and the drafter updates asynchronously.
- **Separate Async.** Training and rollout use separate resource pools that can be configured independently.

The integration covers rollout feature collection, batch preparation, drafter training, weight publication, and checkpointing. When assembling training batches, V1 aligns left-padded prompts, right-padded responses, and their corresponding features. vLLM exposes native speculative-decoding acceptance metrics to track drafter performance. Both async modes have MRV2 launch examples.

**NPU results.** The 16-NPU MRV2 comparison covers 10 training steps, including a synchronous no-drafter baseline, synchronous no-train and co-train DSpark configurations, and online co-training in Colocated Async and Separate Async modes. The figure shows mean acceptance length, step time, and mean training reward. Reading the majority of plotted points gives roughly **40%–55% lower step times** for DSpark online co-training than the synchronous no-drafter baseline. [PR #85](https://github.com/verl-project/verl-SpeCo/pull/85).

{{< figure src="v1-npu-comparison.png" alt="V1 acceptance length, step time, and mean training reward on 16 NPUs" caption="16-NPU MRV2 comparison over 10 training steps." >}}

**GPU results.** Qwen3-8B with DSpark is compared under Synchronous and Colocated Async execution. The configurations are **Baseline** (no drafter), **No-train** (fixed drafter), and **Co-train** (online drafter training). Values average the first 10 steps, with changes relative to the corresponding Synchronous run.

| Drafter configuration | Synchronous step time | Colocated Async step time | Time reduction | Synchronous acceptance length | Colocated Async acceptance length | Acceptance length change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 703.8 s | 602.8 s | 14.3% | — | — | — |
| No-train | 569.8 s | 468.4 s | 17.8% | 2.954 | 3.244 | +0.290 (+9.8%) |
| Co-train (freeze enabled) | 574.6 s | 464.8 s | 19.1% | 3.008 | 3.286 | +0.278 (+9.2%) |

### 2.3 On-Policy Distillation (OPD)

OPD has two learning paths with different inputs. Teacher log probabilities supervise the student, while the drafter learns from the student features that correspond to the model it will draft for. The integration keeps teacher supervision out of the drafter payload and checks at startup that the run uses FSDP2 and synchronous drafter execution.

Synchronous DSpark co-training now works with OPD on FSDP2. The student learns from teacher log probabilities, while the drafter trains on detached student features collected during the old-logprob forward pass.

In one paired experiment with a Qwen3-8B student, a Qwen3-32B teacher, and vLLM on GSM8K and MATH, co-training increased mean acceptance length by **7.1%** and reduced mean generation time by about **1.0%** relative to a fixed drafter (no-train). The table reports averages over steps 1–200 from one run per condition. [PR #124](https://github.com/verl-project/verl-SpeCo/pull/124).

| Metric | No-train | Co-train |
| --- | ---: | ---: |
| Mean acceptance length | 2.960 | 3.171 |
| Mean generation time | 17.89 s | 17.71 s |

{{< figure src="opd-acceptance.png" alt="OPD acceptance length over 200 steps for fixed and online DSpark drafters" caption="Mean acceptance length over 200 OPD steps." >}}

### 2.4 DSpark Co-Training Integration with SGLang

verl-SpeCo integrates online DSpark co-training with the SGLang rollout backend, supporting training-feature collection and drafter weight updates. The adapter maps parameter names, configures draft-token counts from the DSpark verification length, and exposes speculative-decoding metrics. Updated drafter weights are published to SGLang for subsequent rollouts.

The GRPO comparison uses Qwen3-4B with identical data and training settings and a DSpark block size of seven. Over the first 38 steps, co-train increased mean acceptance length from **3.72 to 3.91** relative to no-train and reduced mean generation time from **319.3 s to 309.7 s**, a **3.0%** reduction. [PR #107](https://github.com/verl-project/verl-SpeCo/pull/107).

{{< figure src="sglang-acceptance.png" alt="Mean acceptance length for online and frozen DSpark drafters on SGLang" caption="Qwen3-4B on SGLang. Mean acceptance length with online co-training and a frozen drafter." >}}

{{< figure src="sglang-generation.png" alt="Per-step generation time for online and frozen DSpark drafters on SGLang" caption="Qwen3-4B on SGLang. Generation time per step for the same comparison." >}}

## 3. Training Scheduling and Sampling Optimizations

This release improves when the drafter trains, when updates are worth continuing, and which samples it learns from. Bubble Time uses idle rollout windows, Drafter Freeze pauses training and feature collection when gains plateau, and Hard Sampling prioritizes low-acceptance requests so more training goes toward samples with room for improvement.

### 3.1 Bubble Time

In synchronous co-training, running drafter updates immediately after rollout blocks the next main-model training step. **Bubble Time runs some drafter updates during idle rollout windows**, reducing the time the main training loop spends waiting.

**Train during idle windows and make up any shortfall.** Bubble Time prioritizes idle rollout windows. When they are too short, the scheduler tracks the remaining optimizer steps and completes them at an appropriate stage of the main training loop. This preserves the planned amount of drafter training while reducing waiting. If drafter training or weight publication is still in progress, the scheduler defers checkpointing until the training quota is met and publication is confirmed.

**Qwen3-8B + DSpark.** This experiment used four GPUs and rollout tensor parallelism of two. Over steps 1–45, including warmup, mean step time fell from **204.11 s to 195.87 s**, about **4.0%**. Mean logged critical-path drafter training time fell from **9.24 s to 1.93 s**, a **79.2%** reduction. [PR #114](https://github.com/verl-project/verl-SpeCo/pull/114).

{{< figure src="bubble-time-dspark.png" alt="Per-step timings for synchronous and Bubble Time DSpark runs on Qwen3-8B" caption="Qwen3-8B + DSpark on four GPUs with rollout tensor parallelism of two. The curves show matched RL steps 1–45, including warm-up." >}}

### 3.2 Drafter Freeze

Online co-training helps the DSpark drafter adapt as the policy changes. Once speculative-decoding gains level off, training the drafter, collecting hidden states, and publishing new weights still incur costs. Drafter Freeze uses rollout performance to decide when to pause drafter updates while policy training and speculative decoding continue.

- **Pause training and feature collection together.** The policy tracks acceptance length and decoding performance across drafter versions. When gains plateau, the scheduler pauses drafter updates and stops collecting hidden states during the old-logprob forward pass, saving work at both stages.
- **Resume when performance declines.** The policy monitors recent rollouts while frozen and can resume training and collection if acceptance declines persistently. Freeze state is saved with the checkpoint. Shadow mode records freeze decisions while training and collection continue.
- **Optionally measure whether an update pays off.** Paired probes use the same prompts before and after an update to estimate future generation savings and compare them with the cost of training and publishing the update.

**Results.** The policy triggered at step **126** in the reported 200-step experiment. The active and shadow runs continued from the same checkpoint through step 200. With active freezing, **drafter training time fell to zero**, hidden-state collection overhead decreased, and **mean acceptance length held steady**. [PR #104](https://github.com/verl-project/verl-SpeCo/pull/104).

{{< figure src="drafter-freeze.png" alt="Drafter training time, feature collection overhead, and mean acceptance length for active freeze and shadow runs" caption="Active freeze and shadow continuations from the same step-126 checkpoint through step 200." >}}

### 3.3 Sampling Low-Acceptance-Length Requests (Hard Sampling)

In speculative decoding, mean acceptance length varies across rollout requests. A lower value indicates that fewer draft tokens are accepted per verification round. Hard Sampling uses this signal to prioritize difficult requests for feature collection under the same collection budget and increase their share of drafter training batches.

- **Select hard samples across the full rollout batch.** The scheduler ranks eligible requests by mean acceptance length after merging data-parallel shards. It builds the candidate pool from the lowest-ranked requests and retains their labels through feature storage and batch assembly.
- **Set the candidate pool and training share separately.** The candidate ratio determines which requests enter the hard pool, and the sample ratio sets the target hard-sample share per training batch. By default, if there are too few hard samples, distinct ordinary requests fill the remaining slots.

**Results.** With matched collection budgets and training settings, the experiment used the lowest-acceptance **40%** of eligible requests as candidates and targeted **10%** hard samples per training batch. From around step **40**, mean acceptance length pulled ahead of the baseline while task accuracy remained comparable. The acceptance-length metric covers all valid scored rollout requests. [PR #120](https://github.com/verl-project/verl-SpeCo/pull/120).

{{< figure src="hard-sampling-acceptance.png" alt="Mean request acceptance length over training steps with and without hard sampling" caption="Mean acceptance length across valid scored rollout requests under matched collection budgets." >}}

{{< figure src="hard-sampling-accuracy.png" alt="Task accuracy over training steps with and without hard sampling" caption="Task accuracy under the same evaluation protocol for the hard-sampling run and baseline." >}}

## 4. Unified Drafter Scheduler

The unified Drafter Scheduler supports online co-training and standalone training. The Trainer supplies lifecycle events and runtime context. The scheduler turns that context into explicit plans, and workers validate and execute those plans.

- **Collect features.** The scheduler selects the feature source and sampling budget. Workers stage and validate features before committing them, and failed collections can be rolled back.
- **Train the drafter.** On the synchronous path, the scheduler prepares a training plan before the actor update. After the actor update, it starts distributed drafter training only once all participating workers pass preflight checks for data versions and batch availability.
- **Publish weights.** After successful training, the scheduler obtains a weight snapshot from a single designated worker and updates the rollout drafter after the actor weights have been updated.

**Standalone training.** The Ray workflow in Section 2.1 also uses the unified Drafter Scheduler. Its queue-based collection plan controls when the feature producer runs and applies backpressure as the queue fills. Its training plan selects a complete batch of feature references and dispatches it to the consumer worker group. Both standalone and online workflows use explicit collection and training plans, with execution adapted to their respective runtimes.

Centralizing these decisions gives the trainer and workers a shared view of which data and model versions each update uses. Explicit plan and execution interfaces also let new collection sources and execution strategies reuse the same validation and publication safeguards.

## 5. Getting Started and Configuration Notes

Choose a [v0.2.0 example](https://github.com/verl-project/verl-SpeCo/tree/v0.2.0/examples) for your model, task, and hardware, then configure the matching runtime environment.

- **Dependencies.** Install upstream verl according to `REQUIRED_VERL.txt`. Native V1 training uses `release/v0.9.0`. Select a matching vLLM, SGLang, or vLLM-Ascend runtime for your drafter and device.
- **Standalone training.** Launch `run_qwen3-8b_drafter_dspark_separate_training.sh` after setting the target model, drafter, dataset, and output paths.
- **V1 trainer.** Set `trainer.use_v1=true` and choose `sync`, `colocate_async`, or `separate_async` through `trainer.v1.trainer_mode`.
- **OPD co-training.** Use `run_qwen3-8b_opd_actor_fsdp2_drafter_dspark_vllm.sh`, configure the student, teacher, and drafter paths, and use FSDP2 with `scheduler.execution.strategy=sync`.
- **Bubble Time.** Set `scheduler.execution.strategy=rollout_idle_worker` and configure training quotas and top-up rules under `scheduler.training_quota`.
- **Drafter Freeze.** Set `drafter_convergence_freeze.enabled=true`. Use `mode=shadow` to record decisions or `mode=active` to pause training and feature collection.
- **Hard Sampling.** Set `dspark_hard_candidate_ratio` for the candidate pool and `dspark_hard_sample_ratio` for the target sampling share. Both must be positive to enable prioritization.

Scheduler, freeze, and sampling options above are under `actor_rollout_ref.rollout.drafter.training`. End-to-end examples include the following.

| Scenario | Example |
| --- | --- |
| DSpark standalone training | [Standalone launcher](https://github.com/verl-project/verl-SpeCo/blob/v0.2.0/examples/run_qwen3-8b_drafter_dspark_separate_training.sh) |
| OPD co-training | [OPD launcher](https://github.com/verl-project/verl-SpeCo/blob/v0.2.0/examples/run_qwen3-8b_opd_actor_fsdp2_drafter_dspark_vllm.sh) |
| Bubble Time | [Bubble/Sync launcher](https://github.com/verl-project/verl-SpeCo/blob/v0.2.0/examples/bubble_time/run_qwen3-8b_drafter_dspark_vllm.sh) |
| V1, Hard Sampling, and Freeze | [Combined NPU launcher](https://github.com/verl-project/verl-SpeCo/blob/v0.2.0/examples/run_qwen3-8b_v1_hard_freeze_longrun_drafter_dspark_vllm_npu.sh) |
