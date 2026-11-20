---
title: "Laminar: Scalable RL Through Full Decoupling, Fine-grained Asynchrony, and Fault Tolerance"
date: 2026-11-20
authors:
  - "Yuxuan Tong"
  - "Guangming Sheng"
  - "Wang Zhang"
summary: "Laminar scales RL post-training for LLMs and long-horizon agents with full decoupling, fine-grained asynchrony, and fault tolerance. To our knowledge, it is the first open-source framework to enable efficient version-consistent rollout."
image: "cover.png"
tags:
  - release
  - agents
  - rollout
  - training
  - performance
  - recipes
math: true
toc: true
---

<p style="text-align: center; margin-bottom: 1.5rem;"><a href="https://github.com/verl-project/laminar" rel="noopener" class="external"><strong>GitHub</strong></a> · <a href="https://doi.org/10.1145/3767295.3803580" rel="noopener" class="external"><strong>Paper</strong></a> · <a href="https://github.com/verl-project/laminar/tree/main/recipes" rel="noopener" class="external"><strong>Recipes</strong></a></p>

We are releasing <a href="https://github.com/verl-project/laminar" rel="noopener" class="external"><strong>Laminar</strong></a>, an open-source reinforcement learning (RL) post-training framework for large language models (LLMs) and long-horizon agents, accompanying our EuroSys &rsquo;26 paper, <a href="https://doi.org/10.1145/3767295.3803580" rel="noopener" class="external"><em>Laminar: A Scalable Asynchronous RL Post-Training Framework</em></a>.

Laminar supports flexible rollout execution across large GPU clusters—becoming, to our knowledge, the **first open-source framework to enable efficient version-consistent rollout** alongside version-mixed partial rollout, while also enabling concurrent multi-checkpoint evaluation.

To scale RL post-training reliably across long-horizon reasoning and agent workloads, Laminar is built around three core system principles:

- **Full decoupling:** Separating rollout, experience buffering, training, and weight transfer into independent loops across dedicated data and weight planes.
- **Fine-grained asynchrony:** Allowing individual trajectories and individual inference replicas to execute, complete, and update weights independently without batch-level or cluster-wide barriers.
- **Fault tolerance:** Isolating local component faults and recovering cleanly so single-replica or single-node errors do not stall the entire RL system.

For example, on our SWE agent training recipe with `Qwen3-Coder-30B-A3B-Instruct`, Laminar improves the **SWE-bench Verified** resolve rate from **46.6% to 52.2%**, surpassing the **51.4%** baseline peak while using **33% fewer trained samples** and **64% fewer GPU-hours**.

## Efficient Version-Consistent Rollout

To accommodate weight synchronization boundaries without waiting for long-running trajectories to finish, existing asynchronous RL systems adopt **version-mixed partial rollout**—interrupting all in-flight trajectories at each weight update and resuming generation under the newly broadcast policy. Because their weight paths rely on cluster-wide broadcasts that force every replica onto a single latest policy version, maintaining version consistency across trajectories was previously impractical without severe GPU idling.

Laminar removes this constraint by allowing inference replicas to hold and update policy versions on independent schedules rather than behind a global weight-switch barrier. A scalable RL system no longer needs to dictate a single rollout semantics: alongside native support for version-mixed partial rollout, Laminar enables **efficient version-consistent rollout**—completing each trajectory entirely under a single policy version at high cluster utilization.

{{< figure src="fig1-rollout-semantics.png" alt="Comparison between version-mixed partial rollout and version-consistent rollout across inference replicas" caption="**Figure 1.** Version-mixed partial rollout interrupts in-flight trajectories at global sync boundaries and incurs repeated re-prefill across mixed policy versions (top), whereas version-consistent rollout completes each trajectory under a single policy version without mid-trajectory re-prefill as replicas update independently (bottom)." width="100%" >}}

### Why Explore Version-Consistent Rollout?

Compared with switching weights mid-episode, keeping each trajectory on a single consistent policy version offers distinct algorithmic and system advantages:

- **Algorithmic clarity and exploration:** Each trajectory is sampled from a single well-defined behavior policy rather than a mixture of policy versions, fitting cleanly into standard off-policy importance sampling formulations and avoiding mid-episode behavioral shifts. Supporting both version-mixed and version-consistent rollout within the same runtime allows controlled empirical comparisons between the two.
- **Eliminating weight-switch re-prefill:** Switching weights mid-trajectory invalidates cached KV states and forces the new policy to re-prefill the entire accumulated context—an overhead that scales cubically with sequence length under frequent updates. Routing all steps of a trajectory to a replica serving its assigned version preserves its prefix KV cache from start to finish, prefilling only newly appended context.

### Overcoming the Old-Version Tail with Repack

The central system challenge of version-consistent rollout under asynchronous updates is tail under-utilization: once newer weights \(v_{k+1}\) become available, replicas still serving a few slow \(v_k\) stragglers cannot overwrite their weights without breaking version consistency, leaving those GPUs under-utilized on stale weights.

{{< figure src="fig2-repack.png" alt="Repack consolidating sparse old-version trajectories onto fewer same-version replicas" caption="**Figure 2.** Repack consolidates sparse old-version trajectories from lightly loaded replicas onto fewer same-version replicas, freeing drained replicas to load newer weights immediately." width="100%" >}}

Laminar resolves this tail bottleneck through <a href="https://github.com/verl-project/laminar/blob/main/src/laminar/multi_version_rollout/repack.py" rel="noopener" class="external"><strong>Repack</strong></a>. By continuously monitoring replica KV-cache occupancy, Laminar consolidates remaining trajectories from lightly loaded \(v_k\) replicas onto fewer destination replicas running the **same version** \(v_k\) (retaining already-generated token prefixes on interrupted steps), freeing drained replicas to load \(v_{k+1}\) immediately and return to full utilization.

### Fully Utilizing GPUs in Multi-Checkpoint Evaluation

Multi-version replica scheduling and Repack also power standalone checkpoint evaluation via `laminar.eval`. Sequential evaluation of long-horizon benchmarks suffers from severe tail idle bubbles at the end of every checkpoint. `laminar.eval` evaluates **multiple checkpoints concurrently** across resident inference replicas, consolidates each checkpoint's long tail onto fewer GPUs via Repack, and reloads new checkpoints in place—supporting both **offline batch evaluation** and **online streaming evaluation** alongside active training runs.

## Building for Scalability

Supporting flexible rollout strategies is part of Laminar's broader architectural direction: scaling RL post-training reliably across large clusters as reasoning horizons and agent environments grow in complexity. While recent LLM RL systems have mitigated synchronous batch bubbles through partial rollout and disaggregated experience buffers, scaling further still encounters three systemic bottlenecks: **synchronization on the weight path** from coordinated trainer-to-rollout broadcasts, **coarse-grained lockstep** across batches or replica pools, and **cluster-wide failure blast radii** when trainer ranks and inference replicas share collective groups or global barriers.

To provide a general, extensible foundation for large-scale post-training, Laminar is built around three system design principles: **full decoupling**, **fine-grained asynchrony**, and **fault tolerance**.

### Full Decoupling

Prior asynchronous RL systems typically disaggregate rollout and training onto separate GPU pools and buffer the data path, yet keep the weight path coupled through direct, cluster-wide trainer-to-rollout broadcasts. Laminar achieves **full decoupling** across both the data plane and the weight plane, separating the pipeline into four independently operating components so rollout and training progress without blocking each other:

{{< figure src="fig3-architecture.png" alt="Laminar decoupled architecture across Rollout Loop, Experience Buffer, Training Loop, and Weight Relay" caption="**Figure 3.** Laminar decouples the Rollout Loop and Training Loop through an asynchronous Experience Buffer (data plane) and Weight Relay (weight plane)." width="100%" >}}

- **Rollout Loop:** Continuously drives agent interactions through an agent runtime and serves generation across inference replicas.
- [**Experience Buffer**](https://github.com/verl-project/laminar/tree/main/src/laminar/train/experience_buffer) **(Data Plane):** Decouples *trajectory production* from *training consumption*, ingesting completed trajectories asynchronously and serving ready batches on demand.
- **Training Loop:** Continuously samples batches from the buffer, computes advantages and off-policy corrections, updates parameters on distributed training backends, and periodically publishes weights.
- [**Weight Relay**](https://github.com/verl-project/laminar/tree/main/src/laminar/train/relay) **(Weight Plane):** Decouples *weight publication* from *weight loading*. The trainer stages updated parameters into per-node shared CPU memory and proceeds immediately to the next optimization step, while inference replicas pull local updates asynchronously when ready.

### Fine-grained Asynchrony

Even with fully decoupled training and rollout loops, coarse-grained batch waves or cluster-wide weight switches still leave GPUs waiting on stragglers. Laminar schedules execution with fine-grained asynchrony—operating at the granularity of individual trajectories and independent replicas rather than synchronous batches.

Each trajectory advances through generation steps and environment calls independently. As soon as a short trajectory finishes, its experience enters the Experience Buffer and its concurrency slot is immediately released to admit new prompts under KV-cache utilization control. Long-tail trajectories continue executing in the background without delaying other replicas and join subsequent training batches upon completion.

### Fault Tolerance

At cluster scale, hardware and runtime failures are inevitable; what matters is preventing a local fault from becoming a global interruption. Conventional fault tolerance focuses on whole-job recoverability—restarting an entire run from a checkpoint after failure—which still wastes substantial compute and discards all in-flight trajectories whenever a single rollout server fails. Laminar achieves fault tolerance by combining strict **fault isolation** with targeted recovery across three failure domains:

{{< figure src="fig4-fault-tolerance.png" alt="Fault isolation and targeted recovery in Laminar when a rollout replica fails" caption="**Figure 4.** Example of fault tolerance in Laminar: a failed inference replica is fenced by the router, restarted in place from the local Weight Relay, and resumed from its generated token prefix while healthy components continue running." width="100%" >}}

- **Rollout instance and inference replica failures:** Inference replicas operate behind Laminar's <a href="https://github.com/verl-project/laminar/tree/main/src/laminar/global_request_router" rel="noopener" class="external">router</a>, outside the trainer's process group. When a replica fails, [lifecycle recovery](https://github.com/verl-project/laminar/blob/main/src/laminar/multi_version_rollout/lifecycle.py) fences only the affected replica while healthy replicas, the Experience Buffer, and the Training Loop continue making progress. The isolated replica restarts in place, reloads weights from the local Weight Relay (or checkpoint storage during evaluation) without interrupting the trainer, and resumes any interrupted generation from its already-generated token prefix.
- **Weight synchronization failures:** Broadcasting weights over a single shared collective across trainer and rollout GPUs allows one unresponsive worker to stall the entire cluster. Laminar's [Weight Relay](https://github.com/verl-project/laminar/tree/main/src/laminar/train/relay) decouples weight transfer over point-to-point connections by default. If a replica fails while loading weights, it releases its read lock cleanly and recovers locally without blocking subsequent trainer publications; version fences ensure uncommitted weights are never exposed.
- **Trainer failures:** Decoupling keeps checkpoints aligned across model/optimizer state, prompt cursors, and published weight versions, allowing runs to resume cleanly from the latest saved checkpoint—and, combined with asynchronous per-step checkpointing, minimizing training progress lost upon failure.

## Evaluation: SWE Agent Training

We demonstrate Laminar end-to-end on our <a href="https://github.com/verl-project/laminar/blob/main/recipes/swe/README.md" rel="noopener" class="external">SWE agent training recipe</a>, post-training `Qwen3-Coder-30B-A3B-Instruct` on **SWE-rebench** and evaluating checkpoints on **SWE-bench Verified** under a unified greedy evaluation protocol.

{{< figure src="fig5-swe-benchmark.png" alt="SWE-bench Verified resolve rate versus trained samples and GPU-hours comparing Laminar and baseline" caption="**Figure 5.** SWE-bench Verified resolve rate (%) versus trained samples (left) and GPU-hours (right), comparing the Laminar SWE recipe (64×H800, version-consistent rollout) with the baseline recipe (64×H100, version-mixed rollout)." width="100%" >}}

<table style="font-size: 0.81rem; line-height: 1.38;">
  <thead>
    <tr>
      <th style="padding: 0.45rem 0.42rem;">Run</th>
      <th style="padding: 0.45rem 0.42rem;">Cluster</th>
      <th style="padding: 0.45rem 0.42rem;">Trajectory Version</th>
      <th style="padding: 0.45rem 0.42rem;">Global Batch</th>
      <th style="padding: 0.45rem 0.42rem;">Mini Batch</th>
      <th style="padding: 0.45rem 0.42rem;">Weight Sync</th>
      <th style="padding: 0.45rem 0.42rem;">Trained Samples</th>
      <th style="padding: 0.45rem 0.42rem;">GPU-Hours</th>
      <th style="padding: 0.45rem 0.42rem;">SWE-bench Verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 0.45rem 0.42rem;"><strong>Base Model</strong></td>
      <td style="padding: 0.45rem 0.42rem;">—</td>
      <td style="padding: 0.45rem 0.42rem;">—</td>
      <td style="padding: 0.45rem 0.42rem;">—</td>
      <td style="padding: 0.45rem 0.42rem;">—</td>
      <td style="padding: 0.45rem 0.42rem;">—</td>
      <td style="padding: 0.45rem 0.42rem;">0</td>
      <td style="padding: 0.45rem 0.42rem;">—</td>
      <td style="padding: 0.45rem 0.42rem;">46.6% (233/500)</td>
    </tr>
    <tr>
      <td style="padding: 0.45rem 0.42rem;"><strong>Baseline</strong> (Step 150, peak)</td>
      <td style="padding: 0.45rem 0.42rem;">64×H100</td>
      <td style="padding: 0.45rem 0.42rem;">Mixed</td>
      <td style="padding: 0.45rem 0.42rem;">512</td>
      <td style="padding: 0.45rem 0.42rem;">128</td>
      <td style="padding: 0.45rem 0.42rem;">1 step (512 traj.)</td>
      <td style="padding: 0.45rem 0.42rem;">76.4k</td>
      <td style="padding: 0.45rem 0.42rem;">5,085</td>
      <td style="padding: 0.45rem 0.42rem;">51.4% (257/500)</td>
    </tr>
    <tr>
      <td style="padding: 0.45rem 0.42rem;"><strong>Laminar</strong> (Step 320, match)</td>
      <td style="padding: 0.45rem 0.42rem;">64×H800</td>
      <td style="padding: 0.45rem 0.42rem;">Consistent</td>
      <td style="padding: 0.45rem 0.42rem;">128</td>
      <td style="padding: 0.45rem 0.42rem;">128</td>
      <td style="padding: 0.45rem 0.42rem;">4 steps (512 traj.)</td>
      <td style="padding: 0.45rem 0.42rem;">41.0k (−46%)</td>
      <td style="padding: 0.45rem 0.42rem;">1,450 (−71%)</td>
      <td style="padding: 0.45rem 0.42rem;">51.4% (257/500)</td>
    </tr>
    <tr>
      <td style="padding: 0.45rem 0.42rem;"><strong>Laminar</strong> (Step 400, peak)</td>
      <td style="padding: 0.45rem 0.42rem;"><strong>64×H800</strong></td>
      <td style="padding: 0.45rem 0.42rem;"><strong>Consistent</strong></td>
      <td style="padding: 0.45rem 0.42rem;"><strong>128</strong></td>
      <td style="padding: 0.45rem 0.42rem;"><strong>128</strong></td>
      <td style="padding: 0.45rem 0.42rem;"><strong>4 steps (512 traj.)</strong></td>
      <td style="padding: 0.45rem 0.42rem;"><strong>51.2k (−33%)</strong></td>
      <td style="padding: 0.45rem 0.42rem;"><strong>1,843 (−64%)</strong></td>
      <td style="padding: 0.45rem 0.42rem;"><strong>52.2% (261/500, +5.6%)</strong></td>
    </tr>
  </tbody>
</table>

Comparing the two end-to-end configurations, Laminar improves the resolve rate on SWE-bench Verified from **46.6% to 52.2%**, surpassing the **51.4%** baseline peak while requiring **33% fewer trained samples** (51.2k vs. 76.4k) and **64% fewer GPU-hours** (1,843 vs. 5,085). Controlled component ablations and microbenchmarks are reported in our <a href="https://doi.org/10.1145/3767295.3803580" rel="noopener" class="external">paper</a>.

## Getting Started

Laminar serves as the asynchronous system layer within a modular RL post-training stack:

- **Compute Backends (<a href="https://github.com/verl-project/verl" rel="noopener" class="external">verl</a>):** Distributed model training and inference engines.
- **Agent Runtime (<a href="https://github.com/verl-project/uni-agent" rel="noopener" class="external">Uni-Agent</a>):** Agent execution, tool calling, and sandbox integration.
- **Scalable RL Orchestration (<a href="https://github.com/verl-project/laminar" rel="noopener" class="external">Laminar</a>):** Cluster-scale orchestration with full decoupling across data and weight planes, fine-grained asynchrony, and fault-tolerant, version-aware execution for both training and evaluation.

Visit the [Laminar GitHub repository](https://github.com/verl-project/laminar) for setup instructions, training and evaluation recipes, and contribution guidelines!
