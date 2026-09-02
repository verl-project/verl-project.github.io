---
title: "verl-vla v0.1.0: A Unified Cloud-Edge Post-Training Loop for Large-Scale VLA"
date: 2026-09-02
authors:
  - "Jincheng Liu"
  - "Haiquan Chen"
  - "Rui Zhang"
  - "Yujie Wang"
  - "Tao Li"
  - "Xiao Liang"
  - "Weihua Zhang"
  - "Yi Shen"
summary: "verl-vla is an open-source framework built on verl that unifies human-in-the-loop data collection, fine-tuning, evaluation, and reinforcement learning for vision-language-action policies across cloud and edge resources."
image: "cover.png"
tags:
  - release
  - vla
  - robotics
  - post-training
  - reinforcement-learning
math: false
toc: true
draft: true
---

As large-scale vision-language-action (VLA) models such as Isaac GR00T and Pi0.5 advance, robot policies are gaining stronger multimodal understanding and cross-task generalization. Turning those capabilities into reliable task performance, however, still requires continuous cycles of data collection, model fine-tuning, policy evaluation, and reinforcement learning post-training. Organizing these stages into a sustainable, iterative loop is a central challenge for bringing large-scale VLAs into practice.

[verl-vla](https://github.com/verl-project/verl-vla) is an open-source VLA post-training framework built on verl. It addresses this challenge by bringing physical robots, simulated environments, teleoperation devices, and compute resources into one unified loop. This post introduces its architecture and shows how distributed training clusters, web-based data collection, and composable workflows connect data collection, fine-tuning, evaluation, and reinforcement learning, while reproducible recipes accelerate VLA deployment.

With v0.1.0, the verl-vla community also worked with NVIDIA across policy models, simulation environments, and reinforcement learning recipes: integrating Isaac GR00T with unified training and evaluation interfaces, bringing Isaac Lab Arena into an environment layer that can be scheduled across nodes, and packaging methods such as RECAP and DSRL as reproducible recipes for online post-training. These open-source contributions let developers complete GPU-parallel simulation, data collection, VLA integration, and policy optimization within one framework, forming an end-to-end loop.

## Why Large-Scale VLA Post-Training Needs a Cloud-Edge System

Robot learning has traditionally revolved around local workflows: developers collect demonstrations beside a robot arm, train policies on a local workstation, and deploy the resulting model back to the robot. For policies of manageable scale, such as ACT and Diffusion Policy, the robot, data, model, and GPUs typically reside in the same environment, enabling rapid iteration across data collection, imitation learning, deployment, and human intervention.

As models scale to large VLAs, however, this local loop begins to encounter new constraints. Model inference and fine-tuning often depend on cloud GPUs or multi-node clusters; high-fidelity simulators may run on dedicated servers; and physical robots and teleoperation devices remain at the edge. Post-training must also accommodate different environments, input devices, and training paradigms, including supervised fine-tuning, retraining on corrective data, and offline or online reinforcement learning. Rebuilding the workflow whenever any one of these elements changes quickly makes the engineering cost exceed the algorithmic cost.

Large-scale VLA post-training therefore needs more than a collection of disconnected tools. It needs a unified cloud-edge system that brings data collection, policy optimization, and evaluation into the same loop. The figure below shows the unified architecture that verl-vla builds toward this goal.

{{< figure src="image1.png" alt="Unified cloud-edge architecture for VLA post-training in verl-vla" caption="Figure 1. Unified cloud-edge architecture for VLA post-training in verl-vla. Workflows orchestrate data collection, fine-tuning, and reinforcement learning; TrainCluster coordinates training, rollout, environment interaction, and resources; and models, simulators, physical robots, and teleoperation devices can be connected to the same post-training loop as needed." width="100%" >}}

## TrainCluster: A Unified Training Cluster Across Cloud, Edge, and Heterogeneous Resources

As described above, distributed resources are the first challenge in cloud-edge VLA post-training. Models, environments, training jobs, and human inputs no longer share a single runtime, so deployment, scheduling, and coordination become concerns that every workflow must address. Solving this problem begins with bringing those distributed resources into one system. TrainCluster is the execution foundation that verl-vla provides for that purpose.

As Figure 1 shows, TrainCluster brings training, rollout, environment interaction, evaluation, data recording, and checkpoint management behind a unified interface. Higher-level workflows only describe the operations required at each stage; they do not need to manage Ray, resource pools, or simulator processes directly. TrainCluster launches the required workers from the cluster configuration, schedules resources, and manages their lifecycles. Resources are organized by role rather than location: environment workers can run on edge devices or simulator nodes, rollout workers can share a machine with training or run on dedicated GPUs, and actor workers can scale across multiple GPUs and nodes. Data collection, evaluation, and online reinforcement learning workflows simply compose the roles they need.

APIs such as `start()`, `record()`, `rollout()`, `train()`, `eval()`, and `shutdown()` expose these capabilities consistently. In disaggregated training-and-serving deployments, TrainCluster also synchronizes weights between the training and rollout models. Developers can validate a workflow on a single machine first, then distribute environments, rollout, and training across nodes as needed, without reimplementing post-training logic for each deployment topology.

## Web-Based Data Collection and Human Intervention for Distributed Cloud-Edge Deployments

TrainCluster brings distributed compute and environment resources into one system, but human-in-the-loop data collection still faces a separation between devices and environments: an operator's keyboard, gamepad, or XR controller is connected locally, while the environment may run in a cloud simulator or on an edge-side robot. Traditional teleoperation binds an input device directly to an environment process, which does not adapt well to this cross-device, cross-network setting.

To address this, verl-vla starts a web service alongside the environment. Operators can open the teleoperation interface from any device and connect their own local input hardware. Whether the environment is LIBERO or Isaac Lab Arena in the cloud, or a robot at the edge, operators use the same interface to observe state, send control commands, and record trajectories. Teleoperation adapters map device signals to environment-specific actions, so adding a new environment only requires adapter logic rather than rebuilding device connections, web visualization, or the data-collection path. Figure 2 shows this unified web interface across different environments.

{{< figure-row src1="teleop-arena.webp" src2="teleop-libero.webp" src3="teleop-piper.webp" alt="Unified web-based teleoperation interfaces for Isaac Lab Arena, LIBERO, and Piper" caption="Figure 2. A unified web-based teleoperation interface. The same browser interface can be reused across LIBERO, Isaac Lab Arena, and edge-side robot environments. Operators can connect local input devices from any machine to observe state, teleoperate, and record trajectories." >}}

DAgger also needs a different interaction pattern in cloud-edge deployments. Smaller local models can infer at high frequency, allowing teleoperation actions to replace model actions step by step. Large VLAs, however, face inference and cloud-edge communication latency, so real deployments usually rely on action chunks to keep edge-side execution smooth. verl-vla therefore switches control at the trajectory level: as Figure 3 shows, an operator can interrupt autonomous execution at any time, insert a recovery or correction trajectory of arbitrary length, and then return control to the policy. The complete trajectory and its control state are recorded together and passed into subsequent fine-tuning and reinforcement learning workflows.

{{< figure src="image5.png" alt="Trajectory-level human intervention in cloud-edge deployments" caption="Figure 3. Trajectory-level human intervention in cloud-edge deployments. A cloud policy produces and sends action chunks for edge-side autonomous execution; an operator can interrupt at any time, insert a recovery or correction trajectory of arbitrary length, and then return control to the policy." width="100%" >}}

## Orchestrating Complete Post-Training Workflows

TrainCluster unifies cloud-edge resources, and the web environment loop unifies data collection and human intervention. Workflow then orchestrates these capabilities into complete post-training procedures. It defines the execution order and data flow across stages, while TrainCluster handles resource scheduling and worker lifecycles and Trainers focus on algorithmic updates. This separation decouples workflow orchestration, distributed execution, and algorithm implementation.

The RECAP recipe illustrates this design. It reuses existing evaluation, trajectory collection, SFT training, and rollout capabilities to perform evaluation, data collection, return computation, value-model training, advantage annotation, and policy updates in sequence. The RECAP-specific pieces are limited to data processing for returns, advantages, and the value model; existing components provide the rest. Workflow connects these stages into an executable, recoverable iteration process.

Once a model and environment have been integrated, developers can pass data, models, and checkpoints between data collection, fine-tuning, evaluation, and different reinforcement learning workflows, then compose new post-training procedures as needed. Implementing a new algorithm like RECAP only requires its algorithm-specific data flow and a small amount of logic, without reintegrating the cloud-edge infrastructure.

## Reproducible Recipes for Bringing VLA Systems into Practice

Unified infrastructure solves the problem of building workflows, but developers still need to know how to choose configurations and reproduce results for a specific model, environment, and post-training objective. verl-vla packages models, environments, data, resource topologies, training procedures, and evaluation protocols into end-to-end recipes. Developers can run the reference configurations directly or swap in their own models, environments, or algorithms within the same workflow.

The table below lists representative recipes validated so far. They cover web-based teleoperation and demonstration collection, supervised fine-tuning, and reinforcement learning post-training with methods such as RECAP and DSRL.

| Recipe | Model and environment | Workflow | Reference results |
| --- | --- | --- | --- |
| ACT Quick Start | ACT / LIBERO Spatial Task 0 | Web-based demonstration collection → SFT → TD3+BC | Success rate improves from about 40% to 80% |
| Pi0.5 SFT | Pi0.5 / LIBERO Spatial | Supervised fine-tuning and evaluation across all tasks | 100/100 successes across 10 tasks, for a 100% success rate |
| Pi0.5 RECAP | Pi0.5 / LIBERO-10 Task 8 | 10 demonstrations → 3 RECAP iterations | 16% to 46%; plain SFT on the same data pool reaches 12% |
| Isaac GR00T RECAP | Isaac GR00T / Isaac Lab Arena GR1 | One RECAP iteration on a two-node cloud-edge cluster | Long-horizon task success improves from 12% to 34% |
| Pi0.5 DSRL | Pi0.5 / LIBERO Spatial | Online DSRL | Task 9: 60% → 82%; Task 2: 74% → 88% |
| Isaac GR00T DSRL | Isaac GR00T / Isaac Lab Arena LIBERO | Freeze the base VLA and optimize a lightweight steering module online | 15.2% average improvement across 10 Arena LIBERO Spatial Suite tasks |

These recipes provide reusable starting points for VLAs of different scales, environments, and post-training paradigms, helping developers close the loop from environment integration to policy validation more quickly.

{{< figure src="image6.GIF" alt="Online post-training of Isaac GR00T with DSRL on the Isaac Lab Arena LIBERO Spatial Suite" caption="Figure 4. Online post-training of Isaac GR00T with DSRL on the Isaac Lab Arena LIBERO Spatial Suite. This recipe freezes the base VLA and optimizes a lightweight steering module with online rewards, achieving an average 15.2% success-rate improvement across 10 spatial manipulation tasks. The result shows that, without retraining the GR00T base model, reinforcement learning post-training can continue to improve policy performance through verl-vla's cloud-edge loop and the Isaac Lab simulator." width="70%" >}}

## verl-vla and NVIDIA: Connecting Policies, Simulation, and Reinforcement Learning

With v0.1.0, we worked with NVIDIA across three key layers of VLA post-training: VLA policy models, GPU-parallel high-fidelity simulation environments, and lightweight reinforcement learning recipes. We connected these capabilities through verl-vla's unified interfaces so they can be composed into end-to-end post-training workflows as needed.

**Isaac Lab Arena: Connecting to the Community Ecosystem of Embodied AI Benchmarks.**

At the environment layer, we integrated Isaac Lab Arena into verl-vla. Built on NVIDIA Isaac Lab, it provides GPU-parallel, high-fidelity simulation and is continuously extended by the community to cover embodied AI benchmarks across robots, scenes, and skills.

Community embodied AI benchmarks built on Isaac Lab Arena can directly reuse verl-vla's environment integration, parallel rollout, data recording, evaluation, and post-training workflows. As the community contributes new scenes, robots, and tasks, a single integration continuously expands benchmark coverage.

**Isaac GR00T: A New VLA Policy Model.**

At the model layer, we integrated Isaac GR00T as a new VLA policy model in verl-vla, supporting fine-tuning, evaluation, and reinforcement learning post-training in LIBERO and Isaac Lab Arena.

Developers can reuse Isaac GR00T checkpoints through the unified interface and combine environments, training procedures, and GPU resources as needed, without maintaining separate scripts.

**RECAP/DSRL: New VLA Reinforcement Learning Recipes.**

At the algorithm layer, we integrated RECAP and DSRL into Isaac Lab Arena pipelines for Isaac GR00T and Pi0/Pi0.5. RECAP combines autonomous policy experience with human correction data, using return and advantage signals to train an advantage-conditioned policy that better reuses high-quality actions and learns continuously from failure. DSRL retains the capabilities of the base VLA, training only a lightweight steering module and optimizing actions with online rewards.

Through this collaboration, we connect data collection, policy evaluation, and distributed post-training into reproducible recipes, while GPU-parallel simulation continuously accelerates the training and optimization of base policy models. Isaac GR00T, Isaac Lab Arena, RECAP, and DSRL now form reusable combinations within verl-vla. Starting from existing checkpoints and simulation tasks, community developers can use the same APIs for evaluation, data collection, fine-tuning, and reinforcement learning, then extend the framework with new models, environments, and recipes.

## Announcing verl-vla v0.1.0

verl-vla v0.1.0 is now available. As a unified VLA post-training framework built on verl, it lets models, environments, and training algorithms integrate independently and be composed as needed across the full post-training lifecycle: from human demonstrations and supervised fine-tuning through evaluation, reinforcement learning, and human intervention.

The current release supports policy models including ACT, Gaussian Actor, Pi0.5, and Isaac GR00T; covers the LIBERO and Isaac Lab Arena environments; and provides post-training workflows for SFT, SAC-style off-policy training, DSRL, and RECAP. Developers can also teleoperate and intervene through keyboards, gamepads, or XR controllers while observing environment state and execution in real time through the browser.

Next, we will continue expanding the supported models, environments, input devices, post-training methods, and end-to-end recipes. Bringing physical-robot training into the unified cloud-edge post-training loop is an important direction: enabling closer coordination between real robots, cloud inference, data collection, human intervention, and continuous optimization. Visit the [verl-vla GitHub repository](https://github.com/verl-project/verl-vla) and [documentation](https://verl-vla.readthedocs.io/en/latest/) to explore the current capabilities and contribute.
