---
title: "VeRL-Omni v0.2.0: Faster Diffusion RL and Stable Omni Training"
date: 2026-08-17
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

{{< figure src="verl_omni_v0_2_0_blog_overview.png" alt="VeRL-Omni v0.2.0 release overview" caption="VeRL-Omni v0.2.0 release overview" width="100%" >}}

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

<table>
  <colgroup>
    <col style="width: 40%;">
    <col style="width: 36%;">
    <col style="width: 12%;">
    <col style="width: 12%;">
  </colgroup>
  <thead>
    <tr>
      <th>Model x Algorithm</th>
      <th>Acceleration / support</th>
      <th>Script</th>
      <th>W&amp;B run</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen-Image x FlowGRPO LoRA</td>
      <td><strong>request-level batching</strong></td>
      <td><a href="https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh">script</a></td>
      <td><a href="https://wandb.ai/mikecheung/flow_grpo/runs/1vsrnhbd">w&amp;b run</a></td>
    </tr>
    <tr>
      <td>Qwen-Image x FlowGRPO full model</td>
      <td>step-wise continuous batching</td>
      <td><a href="https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr.sh">script</a></td>
      <td><a href="https://wandb.ai/andyzhou/VeRL-Omni-demo/runs/8p8y9olb">w&amp;b run</a></td>
    </tr>
    <tr>
      <td>SD3.5 Medium x FlowGRPO LoRA, <strong>V1 trainer</strong></td>
      <td><strong>request-level batching</strong>, sync mode</td>
      <td><a href="https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh">script</a></td>
      <td><a href="https://wandb.ai/mikecheung/flow_grpo/runs/h04p15jr">w&amp;b run</a></td>
    </tr>
    <tr>
      <td>SD3.5 Medium x FlowGRPO LoRA, <strong>V1 trainer</strong></td>
      <td><strong>request-level batching</strong>, <code>separate_async</code></td>
      <td><a href="https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async.sh">script</a></td>
      <td><a href="https://api.wandb.ai/links/didan/kk5uxbmh">w&amp;b run</a></td>
    </tr>
  </tbody>
</table>

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

{{< figure src="qwen-image-gpu-utilization.svg" alt="Qwen-Image FlowGRPO GPU utilization" caption="GPU utilization rises after request-level packing. Blue: v0.1; green: v0.2." >}}

{{< figure src="qwen-image-timing-gen.svg" alt="Qwen-Image FlowGRPO generation time" caption="Generation time drops from the v0.1 path to the v0.2 path. Blue: v0.1; green: v0.2." >}}

{{< figure src="qwen-image-timing-step.svg" alt="Qwen-Image FlowGRPO step time" caption="Step time follows the same trend. Blue: v0.1; green: v0.2." >}}

The production-style Qwen-Image FlowGRPO recipes enable request-level batching by
default. The main entry points are `run_qwen_image_ocr_lora.sh` for the
baseline OCR reward setup. The request-level rollout batching is enabled by default:

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

The repository-level call flow is roughly:

{{< figure src="omni-ppo-adapter-flow.svg" alt="Omni PPO trainer and OmniModelBase adapter call flow" caption="Omni PPO trainer and OmniModelBase adapter call flow." >}}

The module boundary is intentionally narrow. `main_omni.py` only decides that an
online omni job should enter the verl PPO V1 path. The PPO trainer then owns the
generic RL loop: rollout scheduling, advantage computation, and policy updates.
When the actor model is built, the FSDP omni engine loads the Hugging Face model
and asks `OmniModelBase` to resolve the adapter for the configured architecture
and stage. That adapter is where model-specific work lives. For Qwen3-Omni
thinker training, `Qwen3OmniThinkerAdapter` strips inactive modules, redirects
`forward` to the thinker component, and prepares the processor and rollout
alignment hooks before control returns to the PPO loop.

### Newly Support

The current Qwen3-Omni adapter supports thinker-only training by redirecting
training to the target component, stripping unused modules such as Talker and
codec-related components, and working with FSDP/FSDP2 wrapping. Current stable
omni training support is organized around these recipes:

<table>
  <colgroup>
    <col style="width: 32%;">
    <col style="width: 18%;">
    <col style="width: 28%;">
    <col style="width: 11%;">
    <col style="width: 11%;">
  </colgroup>
  <thead>
    <tr>
      <th>Model x Algorithm</th>
      <th>Modality / dataset</th>
      <th>Support</th>
      <th>Script</th>
      <th>W&amp;B run</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen3-Omni Thinker x GSPO</td>
      <td>text -&gt; text / GSM8K</td>
      <td><strong>V1 trainer</strong>, reusable omni adapter, FSDP2, vLLM-Omni rollout</td>
      <td><a href="https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh">script</a></td>
      <td><a href="https://wandb.ai/mikecheung/gspo/runs/j5mro1tn">w&amp;b run</a></td>
    </tr>
    <tr>
      <td>Qwen3-Omni Thinker x GSPO</td>
      <td>image -&gt; text / MMK12</td>
      <td><strong>V1 trainer</strong>, multimodal data, actor-rollout consistency signals</td>
      <td><a href="https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh">script</a></td>
      <td><a href="https://wandb.ai/mikecheung/gspo/runs/2j8hxr36">w&amp;b run</a></td>
    </tr>
    <tr>
      <td>Qwen3-Omni Thinker x GSPO</td>
      <td>text + image + audio -&gt; text / AVQA-R1-6K</td>
      <td><strong>V1 trainer</strong>, NPU recipe, multimodal inputs</td>
      <td><a href="https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh">script</a></td>
      <td>-</td>
    </tr>
    <tr>
      <td>Qwen3-Omni Thinker x DPO</td>
      <td>multimodal -&gt; preference / Omni-Preference</td>
      <td><code>OmniDPOLoss</code>, modality-grouped batches</td>
      <td><a href="https://github.com/verl-project/verl-omni/blob/main/examples/dpo_trainer/qwen3_omni/qwen3_omni/run_qwen3_omni_omni_preference_lora.sh">script</a></td>
      <td><a href="https://api.wandb.ai/links/didan/iumxl2zr">w&amp;b report</a></td>
    </tr>
  </tbody>
</table>

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

## Model and Algorithm Extensions

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

## Future Plan

- Optimize omni-modal models via fully async training.
- Extend new models and algorithms, such as MiniMax-H3, MiniCPM-o models, and
  OPD/M-OPD trainers.
- Make video diffusion model training more efficient via batching, TQ, and the
  V1 trainer.
- Harden diffusion and omni-modal rollout code for async training.
- Support agentic RL with multi-stage and multi-turn generation.
