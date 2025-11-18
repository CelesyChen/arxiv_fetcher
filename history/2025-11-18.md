# ArXiv 新论文更新（2025-11-18)

### [Tiny Chiplets Enabled by Packaging Scaling: Opportunities in ESD Protection and Signal Integrity](https://arxiv.org/abs/2511.10760)
**作者**：Emad Haque, Pragnya Sudershan Nalla, Jeff Zhang, Sachin S. Sapatnekar, Chaitali Chakrabarti, Yu Cao

The scaling of advanced packaging technologies provides abundant interconnection resources for 2.5D/3D heterogeneous integration (HI), thereby enabling the construction of larger-scale VLSI systems with higher energy efficiency in data movement. However, conventional I/O circuitry, including electrostatic discharge (ESD) protection and signaling, introduces significant area overhead. Prior studies have identified this overhead as a major constraint in reducing chiplet size below 100 mm2. In this study, we revisit reliability requirements from the perspective of chiplet interface design. Through parasitic extraction and SPICE simulations, we demonstrate that ESD protection and inter-chiplet signaling can be substantially simplified in future 2.5D/3D packaging technologies. Such simplification, in turn, paves the road for further chiplet miniaturization and improves the composability and reusability of tiny chiplets.


#### cs.AR

### [T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup](https://arxiv.org/abs/2511.11248)
**作者**：Jianyu Wei, Qingtao Li, Shijie Cao, Lingxiao Ma, Zixu Hao, Yanyong Zhang, Xiaoyan Hu, Ting Cao

Large language models (LLMs) are increasingly deployed on customer devices. To support them, current devices are adopting SoCs (System on Chip) with NPUs (Neural Processing Unit) installed. Although high performance is expected, LLM inference on NPUs is slower than its CPU counterpart. The reason is that NPUs have poor performance on computations other than GEMM, like dequantization. Current works either disaggregate prefill on the NPUs and decoding on the CPUs, or put both on the NPUs but with an accuracy loss. To solve this issue, based on the insight that low-bit can enable target computation encoded within an acceptably sized table, we propose table lookup to subsume hardware operations otherwise unsupported. To realize this, we overcome the conflicting hardware behavior of prefill and decoding to design a unified table layout and tiling through (1) fused two-level table-based dequantization and (2) concurrency-hierarchy-guided tiling. Based on that, we implement the prefill phase by three-stage pipeline and map the table-lookup-based decoding to NPU's vector units. Results show 1.4x and 3.1x speedup for prefill and decoding respectively, and 84% energy savings compared to the baseline NPU methods. The code is available at https://github.com/microsoft/T-MAC/tree/main/t-man.


#### cs.AR

### [LIMINAL: Exploring The Frontiers of LLM Decode Performance](https://arxiv.org/abs/2507.14397)
**作者**：Michael Davies, Neal Crago, Karthikeyan Sankaralingam, Christos Kozyrakis

The rapid advancement of Large Language Models (LLMs) necessitates a deep understanding of their fundamental performance limits. This paper investigates the limits of LLM inference, focusing on hardware-imposed bottlenecks in auto-regressive decoding. We develop LIMINAL, an analytical performance model that abstracts application requirements and hardware capabilities to systematically explore performance and efficiency across a wide range of current, near-future, and hypothetical hardware. We find LIMINAL is accurate when comparing to LLMs executing on existing hardware, achieving a mean absolute error of $7.6\%$. Our analysis spans from current HBM3 memory technology used in AI accelerators like GPUs and TPUs to systems based on advanced HBM4 and advanced 3D-stacked DRAM technology. We identify five non-negotiable challenges for LLM inference hardware, establishing compute, memory capacity, bandwidth and collective communication as primary barriers to performance. These findings suggest that achieving significant performance gains beyond 10,000 tokens-per-second will require not just hardware evolution but also fundamental algorithmic advances.


#### cs.AR

### [MMA-Sim: Bit-Accurate Reference Model of Tensor Cores and Matrix Cores](https://arxiv.org/abs/2511.10909)
**作者**：Peichen Xie, Yang Wang, Fan Yang, Mao Yang

The rapidly growing computation demands of deep neural networks (DNNs) have driven hardware vendors to integrate matrix multiplication accelerators (MMAs), such as NVIDIA Tensor Cores and AMD Matrix Cores, into modern GPUs. However, due to distinct and undocumented arithmetic specifications for floating-point matrix multiplication, some MMAs can lead to numerical imprecision and inconsistency that can compromise the stability and reproducibility of DNN training and inference.  This paper presents MMA-Sim, the first bit-accurate reference model that reveals the detailed arithmetic behaviors of the MMAs from ten GPU architectures (eight from NVIDIA and two from AMD). By dissecting the MMAs using a combination of targeted and randomized tests, our methodology derives nine arithmetic algorithms to simulate the floating-point matrix multiplication of the MMAs. Large-scale validation confirms bitwise equivalence between MMA-Sim and the real hardware. Using MMA-Sim, we investigate arithmetic behaviors that affect DNN training stability, and identify undocumented behaviors that could lead to significant errors.


#### cs.AR, cs.LG, cs.NA, math.NA

### [FengHuang: Next-Generation Memory Orchestration for AI Inferencing](https://arxiv.org/abs/2511.10753)
**作者**：Jiamin Li, Lei Qu, Tao Zhang, Grigory Chirkov, Shuotao Xu, Peng Cheng, Lidong Zhou

This document presents a vision for a novel AI infrastructure design that has been initially validated through inference simulations on state-of-the-art large language models. Advancements in deep learning and specialized hardware have driven the rapid growth of large language models (LLMs) and generative AI systems. However, traditional GPU-centric architectures face scalability challenges for inference workloads due to limitations in memory capacity, bandwidth, and interconnect scaling. To address these issues, the FengHuang Platform, a disaggregated AI infrastructure platform, is proposed to overcome memory and communication scaling limits for AI inference. FengHuang features a multi-tier shared-memory architecture combining high-speed local memory with centralized disaggregated remote memory, enhanced by active tensor paging and near-memory compute for tensor operations. Simulations demonstrate that FengHuang achieves up to 93% local memory capacity reduction, 50% GPU compute savings, and 16x to 70x faster inter-GPU communication compared to conventional GPU scaling. Across workloads such as GPT-3, Grok-1, and QWEN3-235B, FengHuang enables up to 50% GPU reductions while maintaining end-user performance, offering a scalable, flexible, and cost-effective solution for AI inference infrastructure. FengHuang provides an optimal balance as a rack-level AI infrastructure scale-up solution. Its open, heterogeneous design eliminates vendor lock-in and enhances supply chain flexibility, enabling significant infrastructure and power cost reductions.


#### cs.DC, cs.AR

### [A Compilation Framework for Quantum Circuits with Mid-Circuit Measurement Error Awareness](https://arxiv.org/abs/2511.10921)
**作者**：Ming Zhong, Zhemin Zhang, Xiangyu Ren, Chenghong Zhu, Siyuan Niu, Zhiding Liang

Mid-circuit measurement (MCM) provides the capability for qubit reuse and dynamic control in quantum processors, enabling more resource-efficient algorithms and supporting error-correction procedures. However, MCM introduces several sources of error, including measurement-induced crosstalk, idling-qubit decoherence, and reset infidelity, and these errors exhibit pronounced qubit-dependent variability within a single device. Since existing compilers such as the Qiskit-compiler and QR-Map (the state-of-art qubit reuse compiler) do not account for this variability, circuits with frequent MCM operations often experience substantial fidelity loss.  In thie paper, we propose MERA, a compilation framework that performs MCM-error-aware layout, routing, and scheduling. MERA leverages lightweight profiling to obtain a stable per-qubit MCM error distribution, which it uses to guide error-aware qubit mapping and SWAP insertions. To further mitigate MCM-related decoherence and crosstalk, MERA augments as-late-as-possible scheduling with context-aware dynamic decoupling. Evaluated on 27 benchmark circuits, MERA achieves 24.94% -- 52.00% fidelity improvement over the Qiskit compiler (optimization level 3) without introducing additional overhead. On QR-Map-generated circuits, it improves fidelity by 29.26% on average and up to 122.58% in the best case, demonstrating its effectiveness for dynamic circuits dominated by MCM operations.


#### quant-ph, cs.AR

### [Views: a hardware-friendly graph database model for storing semantic information](https://arxiv.org/abs/2508.18123)
**作者**：Yanjun Yang, Adrian Wheeldon, Yihan Pan, Themis Prodromakis, Alex Serb

The graph database (GDB) is an increasingly common storage model for data involving relationships between entries. Beyond its widespread usage in database industries, the advantages of GDBs indicate a strong potential in constructing symbolic artificial intelligences (AIs) and retrieval-augmented generation (RAG), where knowledge of data inter-relationships takes a critical role in implementation. However, current GDB models are not optimised for hardware acceleration, leading to bottlenecks in storage capacity and computational efficiency. In this paper, we propose a hardware-friendly GDB model, called Views. We show its data structure and organisation tailored for efficient storage and retrieval of graph data and demonstrate its functional equivalence and storage performance advantage compared to represent traditional graph representations. We further demonstrate its symbolic processing abilities in semantic reasoning and cognitive modelling with practical examples and provide a short perspective on future developments.


#### cs.DB, cs.AR, cs.DC, cs.SC

### [HPCAgentTester: A Multi-Agent LLM Approach for Enhanced HPC Unit Test Generation](https://arxiv.org/abs/2511.10860)
**作者**：Rabimba Karanjai, Lei Xu, Weidong Shi

Unit testing in High-Performance Computing (HPC) is critical but challenged by parallelism, complex algorithms, and diverse hardware. Traditional methods often fail to address non-deterministic behavior and synchronization issues in HPC applications. This paper introduces HPCAgentTester, a novel multi-agent Large Language Model (LLM) framework designed to automate and enhance unit test generation for HPC software utilizing OpenMP and MPI. HPCAgentTester employs a unique collaborative workflow where specialized LLM agents (Recipe Agent and Test Agent) iteratively generate and refine test cases through a critique loop. This architecture enables the generation of context-aware unit tests that specifically target parallel execution constructs, complex communication patterns, and hierarchical parallelism. We demonstrate HPCAgentTester's ability to produce compilable and functionally correct tests for OpenMP and MPI primitives, effectively identifying subtle bugs that are often missed by conventional techniques. Our evaluation shows that HPCAgentTester significantly improves test compilation rates and correctness compared to standalone LLMs, offering a more robust and scalable solution for ensuring the reliability of parallel software systems.


#### cs.DC, cs.AI, cs.SE

### [UFO$^3$: Weaving the Digital Agent Galaxy](https://arxiv.org/abs/2511.11332)
**作者**：Chaoyun Zhang, Liqun Li, He Huang, Chiming Ni, Bo Qiao, Si Qin, Yu Kang, Minghua Ma, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang

Large language model (LLM)-powered agents are transforming digital devices from passive tools into proactive intelligent collaborators. However, most existing frameworks remain confined to a single OS or device, making cross-device workflows brittle and largely manual. We present UFO$^3$, a system that unifies heterogeneous endpoints, desktops, servers, mobile devices, and edge, into a single orchestration fabric. UFO$^3$ models each user request as a mutable TaskConstellation: a distributed DAG of atomic subtasks (TaskStars) with explicit control and data dependencies (TaskStarLines). The TaskConstellation continuously evolves as results stream in from distributed devices, enabling asynchronous execution, adaptive recovery, and dynamic optimization. A Constellation Orchestrator} executes tasks safely and asynchronously while applying dynamic DAG updates, and the Agent Interaction Protocol (AIP) provides persistent, low-latency channels for reliable task dispatch and result streaming. These designs dissolve the traditional boundaries between devices and platforms, allowing agents to collaborate seamlessly and amplify their collective intelligence.  We evaluate UFO$^3$ on NebulaBench, a benchmark of 55 cross-device tasks across 5 machines and 10 categories. UFO$^3$ achieves 83.3% subtask completion, 70.9% task success, exposes parallelism with an average width of 1.72, and reduces end-to-end latency by 31% relative to a sequential baseline. Fault-injection experiments demonstrate graceful degradation and recovery under transient and permanent agent failures. These results show that UFO$^3$ achieves accurate, efficient, and resilient task orchestration across heterogeneous devices, uniting isolated agents into a coherent, adaptive computing fabric that extends across the landscape of ubiquitous computing.


#### cs.DC, cs.MA

### [Beyond Exascale: Dataflow Domain Translation on a Cerebras Cluster](https://arxiv.org/abs/2511.11542)
**作者**：Tomas Oppelstrup, Nicholas Giamblanco, Delyan Z. Kalchev, Ilya Sharapov, Mark Taylor, Dirk Van Essendelft, Sivasankaran Rajamanickam, Michael James

Simulation of physical systems is essential in many scientific and engineering domains. Commonly used domain decomposition methods are unable to deliver high simulation rate or high utilization in network computing environments. In particular, Exascale systems deliver only a small fraction their peak performance for these workloads. This paper introduces the novel \algorithmpropernoun{} algorithm, designed to overcome these limitations. We apply this method and show simulations running in excess of 1.6 million time steps per second and simulations achieving 84 PFLOP/s. Our implementation can achieve 90\% of peak performance in both single-node and clustered environments. We illustrate the method by applying the shallow-water equations to model a tsunami following an asteroid impact at 460m-resolution on a planetary scale running on a cluster of 64 Cerebras CS-3 systems.


#### cs.DC, physics.comp-ph

### [EarthSight: A Distributed Framework for Low-Latency Satellite Intelligence](https://arxiv.org/abs/2511.10834)
**作者**：Ansel Kaplan Erol, Seungjun Lee, Divya Mahajan

Low-latency delivery of satellite imagery is essential for time-critical applications such as disaster response, intelligence, and infrastructure monitoring. However, traditional pipelines rely on downlinking all captured images before analysis, introducing delays of hours to days due to restricted communication bandwidth. To address these bottlenecks, emerging systems perform onboard machine learning to prioritize which images to transmit. However, these solutions typically treat each satellite as an isolated compute node, limiting scalability and efficiency. Redundant inference across satellites and tasks further strains onboard power and compute costs, constraining mission scope and responsiveness. We present EarthSight, a distributed runtime framework that redefines satellite image intelligence as a distributed decision problem between orbit and ground. EarthSight introduces three core innovations: (1) multi-task inference on satellites using shared backbones to amortize computation across multiple vision tasks; (2) a ground-station query scheduler that aggregates user requests, predicts priorities, and assigns compute budgets to incoming imagery; and (3) dynamic filter ordering, which integrates model selectivity, accuracy, and execution cost to reject low-value images early and conserve resources. EarthSight leverages global context from ground stations and resource-aware adaptive decisions in orbit to enable constellations to perform scalable, low-latency image analysis within strict downlink bandwidth and onboard power budgets. Evaluations using a prior established satellite simulator show that EarthSight reduces average compute time per image by 1.9x and lowers 90th percentile end-to-end latency from first contact to delivery from 51 to 21 minutes compared to the state-of-the-art baseline.


#### cs.LG, cs.DC

### [Cascading Bandits With Feedback](https://arxiv.org/abs/2511.10938)
**作者**：R Sri Prakash, Nikhil Karamchandani, Sharayu Moharir

Motivated by the challenges of edge inference, we study a variant of the cascade bandit model in which each arm corresponds to an inference model with an associated accuracy and error probability. We analyse four decision-making policies-Explore-then-Commit, Action Elimination, Lower Confidence Bound (LCB), and Thompson Sampling-and provide sharp theoretical regret guarantees for each. Unlike in classical bandit settings, Explore-then-Commit and Action Elimination incur suboptimal regret because they commit to a fixed ordering after the exploration phase, limiting their ability to adapt. In contrast, LCB and Thompson Sampling continuously update their decisions based on observed feedback, achieving constant O(1) regret. Simulations corroborate these theoretical findings, highlighting the crucial role of adaptivity for efficient edge inference under uncertainty.


#### cs.LG, cs.DC

### [SMART: A Surrogate Model for Predicting Application Runtime in Dragonfly Systems](https://arxiv.org/abs/2511.11111)
**作者**：Xin Wang, Pietro Lodi Rizzini, Sourav Medya, Zhiling Lan

The Dragonfly network, with its high-radix and low-diameter structure, is a leading interconnect in high-performance computing. A major challenge is workload interference on shared network links. Parallel discrete event simulation (PDES) is commonly used to analyze workload interference. However, high-fidelity PDES is computationally expensive, making it impractical for large-scale or real-time scenarios. Hybrid simulation that incorporates data-driven surrogate models offers a promising alternative, especially for forecasting application runtime, a task complicated by the dynamic behavior of network traffic. We present \ourmodel, a surrogate model that combines graph neural networks (GNNs) and large language models (LLMs) to capture both spatial and temporal patterns from port level router data. \ourmodel outperforms existing statistical and machine learning baselines, enabling accurate runtime prediction and supporting efficient hybrid simulation of Dragonfly networks.


#### cs.LG, cs.DC

### [SemanticNN: Compressive and Error-Resilient Semantic Offloading for Extremely Weak Devices](https://arxiv.org/abs/2511.11038)
**作者**：Jiaming Huang, Yi Gao, Fuchang Pan, Renjie Li, Wei Dong

With the rapid growth of the Internet of Things (IoT), integrating artificial intelligence (AI) on extremely weak embedded devices has garnered significant attention, enabling improved real-time performance and enhanced data privacy. However, the resource limitations of such devices and unreliable network conditions necessitate error-resilient device-edge collaboration systems. Traditional approaches focus on bit-level transmission correctness, which can be inefficient under dynamic channel conditions. In contrast, we propose SemanticNN, a semantic codec that tolerates bit-level errors in pursuit of semantic-level correctness, enabling compressive and resilient collaborative inference offloading under strict computational and communication constraints. It incorporates a Bit Error Rate (BER)-aware decoder that adapts to dynamic channel conditions and a Soft Quantization (SQ)-based encoder to learn compact representations. Building on this architecture, we introduce Feature-augmentation Learning, a novel training strategy that enhances offloading efficiency. To address encoder-decoder capability mismatches from asymmetric resources, we propose XAI-based Asymmetry Compensation to enhance decoding semantic fidelity. We conduct extensive experiments on STM32 using three models and six datasets across image classification and object detection tasks. Experimental results demonstrate that, under varying transmission error rates, SemanticNN significantly reduces feature transmission volume by 56.82-344.83x while maintaining superior inference accuracy.


#### cs.CV, cs.AI, cs.DC

### [A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication](https://arxiv.org/abs/2511.11560)
**作者**：Angelo Rodio, Giovanni Neglia, Zheng Chen, Erik G. Larsson

In semi-decentralized federated learning, devices primarily rely on device-to-device communication but occasionally interact with a central server. Periodically, a sampled subset of devices uploads their local models to the server, which computes an aggregate model. The server can then either (i) share this aggregate model only with the sampled clients (sampled-to-sampled, S2S) or (ii) broadcast it to all clients (sampled-to-all, S2A). Despite their practical significance, a rigorous theoretical and empirical comparison of these two strategies remains absent. We address this gap by analyzing S2S and S2A within a unified convergence framework that accounts for key system parameters: sampling rate, server aggregation frequency, and network connectivity. Our results, both analytical and experimental, reveal distinct regimes where one strategy outperforms the other, depending primarily on the degree of data heterogeneity across devices. These insights lead to concrete design guidelines for practical semi-decentralized FL deployments.


#### cs.LG, cs.AI, cs.DC

### [When Federated Learning Meets Quantum Computing: Survey and Research Opportunities](https://arxiv.org/abs/2504.08814)
**作者**：Aakar Mathur, Ashish Gupta, Sajal K. Das

Quantum Federated Learning (QFL) is an emerging field that harnesses advances in Quantum Computing (QC) to improve the scalability and efficiency of decentralized Federated Learning (FL) models. This paper provides a systematic and comprehensive survey of the emerging problems and solutions when FL meets QC, from research protocol to a novel taxonomy, particularly focusing on both quantum and federated limitations, such as their architectures, Noisy Intermediate Scale Quantum (NISQ) devices, and privacy preservation, so on. With the introduction of two novel metrics, qubit utilization efficiency and quantum model training strategy, we present a thorough analysis of the current status of the QFL research. This work explores key developments and integration strategies, along with the impact of QC on FL, keeping a sharp focus on hybrid quantum-classical approaches. The paper offers an in-depth understanding of how the strengths of QC, such as gradient hiding, state entanglement, quantum key distribution, quantum security, and quantum-enhanced differential privacy, have been integrated into FL to ensure the privacy of participants in an enhanced, fast, and secure framework. Finally, this study proposes potential future directions to address the identified research gaps and challenges, aiming to inspire faster and more secure QFL models for practical use.


#### cs.DC, cs.ET, cs.LG, cs.NE

### [To Offload or Not To Offload: Model-driven Comparison of Edge-native and On-device Processing In the Era of Accelerators](https://arxiv.org/abs/2504.15162)
**作者**：Nathan Ng, David Irwin, Ananthram Swami, Don Towsley, Prashant Shenoy

Computational offloading is a promising approach for overcoming resource constraints on client devices by moving some or all of an application's computations to remote servers. With the advent of specialized hardware accelerators, client devices can now perform fast local processing of specific tasks, such as machine learning inference, reducing the need for offloading computations. However, edge servers with accelerators also offer faster processing for offloaded tasks than was previously possible. In this paper, we present an analytic and experimental comparison of on-device processing and edge offloading for a range of accelerator, network, multi-tenant, and application workload scenarios, with the goal of understanding when to use local on-device processing and when to offload computations. We present models that leverage analytical queuing results to derive explainable closed-form equations for the expected end-to-end latencies of both strategies, which yield precise, quantitative performance crossover predictions that guide adaptive offloading. We experimentally validate our models across a range of scenarios and show that they achieve a mean absolute percentage error of 2.2% compared to observed latencies. We further use our models to develop a resource manager for adaptive offloading and show its effectiveness under variable network conditions and dynamic multi-tenant edge settings.


#### cs.DC

### [Mapple: A Domain-Specific Language for Mapping Distributed Programs](https://arxiv.org/abs/2507.17087)
**作者**：Anjiang Wei, Rohan Yadav, Hang Song, Wonchan Lee, Ke Wang, Alex Aiken

Optimizing parallel programs for distributed systems is a complex task, often requiring significant code modifications. Task-based programming systems improve modularity by separating performance decisions from application logic, but their mapping interfaces are low-level. We introduce Mapple, a high-level, declarative programming interface for mapping distributed applications. Mapple provides transformation primitives to resolve dimensionality mismatches between task and processor spaces, including a key primitive, decompose, that helps minimize communication volume. We implement Mapple on top of the Legion runtime by translating Mapple mappers into its low-level C++ interface. Across nine applications, including six matrix multiplication algorithms and three scientific computing workloads, Mapple reduces mapper code size by 14x and enables performance improvements of up to 1.34x over expert-written C++ mappers. In addition, the decompose primitive achieves up to 1.83x improvement over existing dimensionality-resolution heuristics.


#### cs.DC, cs.PL

### [Quantum resources in resource management systems](https://arxiv.org/abs/2506.10052)
**作者**：Utz Bacher, Mark Birmingham, Christopher D. Carothers, Andrew Damin, Carlos D. Gonzalez Calaza, Ashwin Kumar Karnad, Stefano Mensa, Matthieu Moreau, Aurelien Nober, Munetaka Ohtani, Max Rossmannek, Philippa Rubin, M. Emre Sahin, Oscar Wallis, Amir Shehata, Iskandar Sitdikov, Aleksander Wennersteen

Quantum computing resources are increasingly being incorporated into high-performance computing (HPC) environments as co-processors for hybrid workloads. To support this paradigm, quantum devices must be treated as schedulable first-class resources within existing HPC infrastructure. This enables consistent workload management, unified resource visibility, and support for hybrid quantum-classical job execution models.  This paper presents a reference architecture and implementation for the integration of quantum computing resources, both on-premises and cloud-hosted into HPC centers via standard workload managers. We introduce a Slurm plugin designed to abstract and control quantum backends, enabling seamless resource scheduling, minimizing queue duplication, and supporting job co-scheduling with classical compute nodes. The architecture supports heterogeneous quantum resources and can be extended to any workload (and container) management systems.


#### quant-ph, cs.DC, cs.ET, cs.SE

### [How Exclusive are Ethereum Transactions? Evidence from non-winning blocks](https://arxiv.org/abs/2509.16052)
**作者**：Vabuk Pahari, Andrea Canidio

We analyze 15,097 blocks proposed for inclusion in Ethereum's blockchain over an eight-minute window on December 3, 2024, during which 38 blocks were added to the chain. We classify transactions as exclusive -- appearing only in blocks from a single builder -- or private -- absent from the public mempool but included in blocks from multiple builders. We find that, depending on the methodology, exclusive transactions account for between 77.2% and 84% of the total fees paid by transactions in winning blocks. Moreover, we show that exclusivity cannot be fully attributed to persistent relationships between senders and builders: only between 7% and 8.4% of all on-chain exclusive transaction value originates from senders who route exclusively to one builder. Finally, we observe that transaction exclusivity is dynamic. Some transactions are exclusive at the start of a bidding cycle but later appear in blocks from multiple builders. Other transactions remain exclusive to a losing builder for two or three cycles before appearing in the public mempool. These transactions are therefore delayed and then exposed to potential attacks.


#### cs.CR, cs.DC, econ.GN, q-fin.EC
