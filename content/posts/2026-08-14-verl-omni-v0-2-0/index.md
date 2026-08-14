---
title: "VeRL-Omni v0.2.0: Faster Diffusion RL and Stable Omni Training"
date: 2026-08-14
authors:
  - "Yongxiang Huang"
  - "Didan Deng"
  - "Kawai Cheung"
  - "Nguyen Long"
  - "Jingan Zhou"
  - "Yingshu Chen"
  - "Ziyang Hong"
summary: "A release focused on higher-throughput diffusion rollout, reusable omni adapters, and broader recipe coverage."
image: "cover.png"
tags:
  - release
  - performance
  - recipes
  - training
  - rollout
math: false
toc: true
---

VeRL-Omni `v0.2.0` establishes a stronger foundation for production-grade
omni-modal reinforcement learning. This release improves the training stack
across rollout performance, model integration, reward support, hardware
coverage, and documentation, with two changes carrying the most impact:

- Faster diffusion RL, centered on higher-throughput Qwen-Image FlowGRPO rollout
  and V1 trainer support.
- Stable omni training, built around the omni V1 trainer, reusable model
  adapters, FSDP2, and vLLM-Omni rollout.

{{< figure src="verl_omni_v0_2_0_blog_overview.png" alt="VeRL-Omni v0.2.0 release overview" caption="VeRL-Omni v0.2.0 release overview" width="90%" >}}

## 1. Faster Diffusion RL

Diffusion RL is expensive, but not in the same way as autoregressive
language-model RL. A single rollout carries many denoising steps, large latent
tensors, prompt embeddings, optional classifier-free guidance, reward-model
scoring, old-log-prob recomputation, and policy-weight synchronization. For
Qwen-Image FlowGRPO, there is no single villain in the profile. Step time is
shaped by rollout generation, old-log-prob computation, reward scoring, actor
update, and LoRA weight sync together.

### Key Features

The faster diffusion RL work has two main features.

- Request-level batching leads the rollout side. For supported diffusion
  adapters, it becomes the default batching path. Instead of sending diffusion
  generations through a serial loop, the engine now acts more like a traffic
  controller, with explicit concurrency knobs for scheduling rollout work.

- The trainer path matters just as much. Diffusion now has a V1 trainer path,
  bringing diffusion RL closer to the modern trainer architecture used elsewhere
  in VeRL-Omni and laying the groundwork for decoupled rollout and training
  execution.

Faster rollout only matters if the generated trajectories and log-probs still
describe the same policy. This release fixes several correctness-sensitive
areas: request-batched diffusion log-probs, async rollout semantics, rank-local
LoRA weight-update routes, and the hooks used by optional rollout-correction
recipes.

### Newly Support

The [rollout batching guide](https://verl-omni.readthedocs.io/en/latest/start/rollout_batching.html)
explains both diffusion batching modes, how to enable them, and when to choose
each mode. Current faster diffusion RL support is organized around these
recipes:

| Model | Algorithm | Script | Acceleration / support | W&B run |
|---|---|---|---|---|
| Qwen-Image | FlowGRPO LoRA | [`run_qwen_image_ocr_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh) | request-level batching | [v0.2.0 runs](https://wandb.ai/mikecheung/flow_grpo/runs/1vsrnhbd) |
| Qwen-Image | FlowGRPO LoRA | [`run_qwen_image_ocr_lora_async_reward.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_async_reward.sh) | request-level batching, async reward | - |
| Qwen-Image | FlowGRPO LoRA | [`run_qwen_image_ocr_lora_rollout_corr.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_rollout_corr.sh) | request-level batching, rollout correction bypass | - |
| Qwen-Image | FlowGRPO full model | [`run_qwen_image_ocr.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr.sh) | step-wise continuous batching, full-model training | [full model](https://wandb.ai/andyzhou/VeRL-Omni-demo/runs/8p8y9olb) |
| SD3.5 Medium | FlowGRPO LoRA | [`run_sd35_medium_ocr_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh) | request-level batching | [v0 trainer](https://wandb.ai/mikecheung/flow_grpo/runs/9ylk6e5f) |
| SD3.5 Medium | FlowGRPO LoRA, V1 trainer | [`run_sd35_medium_ocr_lora_v1.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh) | V1 trainer sync mode | [v1 trainer](https://wandb.ai/mikecheung/flow_grpo/runs/h04p15jr) |
| SD3.5 Medium | FlowGRPO LoRA, V1 trainer | [`run_sd35_medium_ocr_lora_v1_separate_async.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async.sh) | V1 trainer `separate_async`, dedicated rollout workers | - |

A full diffusion post training support table in VeRL-Omni is available at
[README.md](https://github.com/verl-project/verl-omni#model-and-algorithm-support-).

### Recipe and Benchmark

The Qwen-Image LoRA OCR recipe is a good place to see the change. In the v0.1
line, rollout was the core bottleneck: each request effectively ran as serial
`B≈1` DiT forwards, with 10 denoising steps and True-CFG doubling each step
into two forwards. GPU utilization hovered around `80%`, not because the model
was small, but because the engine could not keep enough diffusion work packed
together.

In `v0.2.0`, request-level packing changes that shape. Multiple complete
requests are packed into one transformer forward, GPU utilization rises to about
`100%`, and isolated generation time drops from `226s` to `108s`, a `52%`
reduction. The same story shows up in per-image generation latency, which falls
with the packed rollout path. Reference runs:
[Qwen-Image OCR LoRA v0.1](https://wandb.ai/mikecheung/flow_grpo/runs/o7x44yrr)
and
[Qwen-Image OCR LoRA v0.2](https://wandb.ai/mikecheung/flow_grpo/runs/1vsrnhbd).

In the charts below, the blue curve is `v0.1` and the green curve is `v0.2`.

{{< figure src="qwen-image-gpu-utilization.svg" alt="Qwen-Image FlowGRPO GPU utilization" caption="GPU utilization rises after request-level packing." >}}

{{< figure src="qwen-image-timing-gen.svg" alt="Qwen-Image FlowGRPO generation time" caption="Generation time drops from the v0.1 path to the v0.2 path." >}}

{{< figure src="qwen-image-timing-step.svg" alt="Qwen-Image FlowGRPO step time" caption="Step time follows the same trend." >}}

The production-style Qwen-Image FlowGRPO recipes enable request-level batching by
default. The main entry points are `run_qwen_image_ocr_lora.sh` for the
baseline OCR reward setup, `run_qwen_image_ocr_lora_async_reward.sh` for async
reward on a dedicated resource pool, and `run_qwen_image_ocr_lora_rollout_corr.sh`
for rollout-correction bypass mode. The request-level rollout knobs are
intentionally small and explicit:

```bash
actor_rollout_ref.rollout.step_execution=false
++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=32
```

For Qwen-Image LoRA with True-CFG at 512 px, a practical tuning range is
`max_num_seqs=8` to `32`; larger values can run into HBM pressure. SD3.5 has a
lighter request-level memory shape and can use `max_num_seqs=256`.

The recipe-level step-time numbers line up with that story: the baseline
Qwen-Image FlowGRPO LoRA run is about `420s` per step on 4 × H800, while the
async reward variant reaches about `360s` per step on 5 GPUs.

SD3.5 FlowGRPO shows the trainer side of the same release. The repository
includes `run_sd35_medium_ocr_lora_v1.sh` for the V1 trainer in sync mode and
`run_sd35_medium_ocr_lora_v1_separate_async.sh` for V1 `separate_async`
rollout. In the current benchmark, v0 and v1 are roughly tied on step time,
but the V1 run has a cleaner stability story as reward rises through training.
Reference runs:
[SD3.5 Medium OCR LoRA v0 trainer](https://wandb.ai/mikecheung/flow_grpo/runs/9ylk6e5f)
and
[SD3.5 Medium OCR LoRA v1 trainer](https://wandb.ai/mikecheung/flow_grpo/runs/h04p15jr).

{{< figure src="sd3.5-m-timing-step.svg" alt="SD3.5 FlowGRPO v0 versus v1 step time" caption="SD3.5 v0 and V1 trainer step time are currently close." >}}

{{< figure src="sd3.5-m-training-rewards.svg" alt="SD3.5 FlowGRPO v0 versus v1 training rewards" caption="Training rewards rise steadily on the V1 path." >}}

{{< figure src="sd3.5-m-val-rewards.svg" alt="SD3.5 FlowGRPO v0 versus v1 validation rewards" caption="Validation rewards track the same stability story." >}}

## 2. Stable Omni Training

The other half of the release is stable omni training. Omni models are not just
bigger language models; they are small systems with processors, modality-specific
towers, trainable stages, and rollout-time behavior that has to stay aligned
with the actor. `v0.2.0` moves the project from model-specific integrations
toward a reusable omni training stack, so multimodal autoregressive training
fits more naturally into VeRL-Omni's trainer, adapter, rollout, and recipe
structure.

### Key Features

Here, the release pulls on two levers.

One lever is the `verl` V1 trainer architecture. Omni recipes get clearer
worker orchestration, standard configuration overrides, and better alignment
with vLLM-Omni rollout.

The other is the reusable omni model adapter layer. Instead of wiring each
architecture as a one-off path, the trainer can rely on a shared interface for
model setup, processor setup, trainable-stage selection, FSDP preparation, and
rollout alignment.

### Newly Support

The current Qwen3-Omni adapter supports thinker-only training by redirecting
training to the target component, stripping unused modules such as Talker and
codec-related components, and working with FSDP/FSDP2 wrapping. Current stable
omni training support is organized around these recipes:

| Model | Training mode | Algorithm / data | Script | Support | W&B run |
|---|---|---|---|---|---|
| Qwen3-Omni Thinker | text -> text | GSPO on GSM8K | [`run_qwen3_omni_thinker_gspo_lora_v1.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh) | V1 trainer, reusable omni adapter, FSDP2, vLLM-Omni rollout | [gsm8k](https://wandb.ai/mikecheung/gspo/runs/j5mro1tn) |
| Qwen3-Omni Thinker | image -> text | GSPO on MMK12 | [`run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh) | V1 trainer, multimodal data, actor-rollout consistency signals | [MMK12](https://wandb.ai/mikecheung/gspo/runs/2j8hxr36) |
| Qwen3-Omni Thinker | text + image + audio -> text | GSPO on AVQA-R1-6K | [`run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh) | V1 trainer, NPU recipe, multimodal inputs | - |
| Qwen3-Omni Thinker | offline multimodal preference | Omni DPO on Omni-Preference | [`run_qwen3_omni_omni_preference_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/dpo_trainer/qwen3_omni/qwen3_omni/run_qwen3_omni_omni_preference_lora.sh) | Offline MLLM DPO dataset, `OmniDPOLoss`, modality-grouped batches | [w&b report](https://api.wandb.ai/links/didan/iumxl2zr) |

A full omni post training support table in VeRL-Omni is available at
[README.md](https://github.com/verl-project/verl-omni#model-and-algorithm-support-).

### Recipe and Benchmark

The best single recipe to highlight is **MMK12**. It exercises the new stable
Qwen3-Omni path with real multimodal input: image plus text prompt, text answer,
GSPO optimization, FSDP actor training, and vLLM-Omni rollout.

**MMK12 anchor recipe.** `run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh` trains
Qwen3-Omni on K12 visual math reasoning (`image -> text`) with GSPO, LoRA rank
32, and colocated actor-rollout workers on 4 × H800 80GB. The rollout shape is
128 prompts × 16 responses, or 2048 samples per rollout. After training, the
run reaches `0.833` validation reward, `0.998` actor-rollout Pearson correlation,
and about `59 GB` GPU memory usage. See some training results in the reference
run: [`MMK12 (wandb)`](https://wandb.ai/mikecheung/gspo/runs/2j8hxr36).

{{< figure src="mmk12_training_rewards.svg" alt="MMK12 training rewards mean scores" caption="MMK12 training rewards mean scores." >}}

{{< figure src="mmk12_val_rewards.svg" alt="MMK12 validation rewards mean scores" caption="MMK12 validation rewards mean scores." >}}

The MMK12 data pipeline converts raw MMK12 parquet shards into the verl RL
parquet layout. Each row carries the image bytes inline and uses a prompt format
that asks the model to produce a structured answer. The reward combines
`math_verify` accuracy with a progressive format reward on the
`<answer>...\boxed{}...</answer>` template.

To run the recipe:

```bash
python examples/gspo_trainer/data_process/mmk12.py \
    --local_dataset_path /path/to/mmk12/ \
    --local_save_dir ~/data/mmk12

TRAIN_FILE=$HOME/data/mmk12/train.parquet \
VAL_FILE=$HOME/data/mmk12/test.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh
```

This anchors the `v0.2.0` stability story: Qwen3-Omni training is no longer
just a model-specific launch path. It is a V1 trainer recipe with a reusable
omni adapter, multimodal data handling, actor-rollout consistency metrics, and
a documented image-to-text benchmark.

## Briefly: Other Updates

The release also expands the broader VeRL-Omni model and algorithm surface:

| Model / family | Category | Modality | Algorithm / recipe | Update |
|---|---|---|---|---|
| [LTX2.3](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/ltx2/README.md) | Diffusion generator | Text -> Video + Audio | FlowGRPO | Adds text-to-video+audio training with CLAP and ImageBind rewards. |
| [Qwen-Image-Edit](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image_edit/README.md) | Diffusion image editor | Text + Image -> Image | FlowGRPO | Adds image-editing data preparation and a general edit-training interface. |
| [BAGEL](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/bagel/README.md) | Unified understand + generation model | Text + Image | FlowGRPO | Adds full-parameter and LoRA recipes with OCR and PickScore rewards. |
| [SD3.5 + DiNa-LRM](https://verl-omni.readthedocs.io/en/latest/examples/flowgrpo_trainer_sd35_drm.html) | Diffusion generator | Text -> Image | FlowGRPO with latent reward model | Scores clean diffusion latents directly, avoiding VAE decode during reward scoring. |
| [Flow-DPPO](https://verl-omni.readthedocs.io/en/latest/algo/flowdppo.html) | Diffusion generator algorithm | Text/Image -> Image | Flow-DPPO | Adds an alternative policy-optimization recipe for Qwen-Image style diffusion RL. |
| [Wan2.2](https://github.com/verl-project/verl-omni/blob/main/examples/dancegrpo_trainer/README.md) | Diffusion video generator | Text -> Video | DanceGRPO | Adds video-generation RL recipe coverage. |

Outside the model-algorithm matrix, `v0.2.0` also adds Ascend NPU Dockerfiles
and install guidance.

## Looking Ahead

The road after `v0.2.0` is fairly clear.

For diffusion RL, the next step is to keep moving from "works end-to-end" to
"measurably efficient and diagnosable". The repository now has the features for
this: batching modes, profiling recipes, and rollout-correction paths. Future
benchmark suites should separate rollout, log-prob computation, reward scoring,
weight sync, and actor update costs.

For omni model training, the adapter design should become a reusable
model-integration pattern. It points toward more omni models with
stage-specific training, for example, minicpm-o.

Correctness signals should keep getting tighter. For both diffusion and omni
training, high throughput is only useful when log-probs, rollout trajectories,
actor weights, and reward signals remain aligned. Metrics such as
actor-rollout Pearson correlation, log-prob consistency, precision dumps, and
step-level timing should become standard parts of every serious recipe.

Reproducibility should also become a first-class release goal. Future recipes
should make it easier to rerun the same setup with pinned model versions, fixed
data preprocessing, explicit seeds, stable configuration overrides, and
published reference metrics. For long-running RL jobs, reproducibility is not
only about getting the same final score; it is about making reward curves,
timing breakdowns, memory usage, and validation outputs comparable across
machines and releases.
