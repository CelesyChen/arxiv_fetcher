# ArXiv 新论文更新（2025-10-20)

### [Impact of AI-Triage on Radiologist Report Turnaround Time: Real-World Time-Savings and Insights from Model Predictions](https://arxiv.org/abs/2510.15237)
**作者**：Yee Lam Elim Thompson, Jonathan Fergus, Jonathan Chung, Jana G. Delfino, Weijie Chen, Gary M. Levine, Frank W. Samuelson

Objective: To quantify the impact of workflow parameters on time-savings in report turnaround time (TAT) due to an AI-triage device that prioritized pulmonary embolism (PE) in chest CT pulmonary angiography (CTPA) exams. Methods: This retrospective study analyzed 11252 adult CTPA exams conducted for suspected PE at a single tertiary academic medical center. Data was divided into two periods: pre-AI and post-AI. For PE-positive exams, TAT - defined as the duration from patient scan completion to the first preliminary report completion - was compared between the two periods. Time-savings were reported separately for work-hour and off-hour cohorts. To characterize radiologist workflow, 527234 records were retrieved from the PACS and workflow parameters such as exam inter-arrival time and radiologist read-time extracted. These parameters were input into a computational model to predict time-savings following deployment of an AI-triage device and to study the impact of workflow parameters. Results: The pre-AI dataset included 4694 chest CTPA exams with 13.3% being PE-positive. The post-AI dataset comprised 6558 exams with 16.2% being PE-positive. The mean TAT for pre-AI and post-AI during work hours are 68.9 [95% CI" 55.0, 82.8] and 46.7 [38.1, 55.2] minutes respectively, and those during off-hours are 44.8 [33.7, 55.9] and 42.0 [33.6, 50.3] minutes. Clinically-observed time-savings during work hours (22.2 [95% CI: 5.85, 38.6] minutes) were significant (p=0.004), while off-hour (2.82 [-11.1, 16.7] minutes) were not (p=0.345). Observed time-savings aligned with model predictions (29.6 [95% range: 23.2, 38.1] minutes for work hours; 2.10 [1.76, 2.58] minutes for off-hours). Discussion: Consideration and quantification of clinical workflow contribute to an accurate assessment of the expected time-savings in TAT following deployment of an AI-triage device.


#### cs.PF

### [Bridging Simplicity and Sophistication using GLinear: A Novel Architecture for Enhanced Time Series Prediction](https://arxiv.org/abs/2501.01087)
**作者**：Syed Tahir Hussain Rizvi, Neel Kanwal, Muddasar Naeem

Time Series Forecasting (TSF) is an important application across many fields. There is a debate about whether Transformers, despite being good at understanding long sequences, struggle with preserving temporal relationships in time series data. Recent research suggests that simpler linear models might outperform or at least provide competitive performance compared to complex Transformer-based models for TSF tasks. In this paper, we propose a novel data-efficient architecture, \textit{Gaussian-activated Linear model (GLinear)}, for multivariate TSF that exploits periodic patterns to provide better accuracy. It achieves higher prediction accuracy while requiring less historical data than other state-of-the-art linear predictors. Four different datasets (ETTh1, Electricity, Traffic, and Weather) are used to evaluate the performance of the proposed predictor. A performance comparison with state-of-the-art linear architectures (such as NLinear, DLinear, and RLinear) and transformer-based time series predictors (Autoformer) shows that the GLinear, despite being data efficient, outperforms the existing architectures in most cases of multivariate TSF while being competitive in others. We hope that the proposed GLinear model opens new fronts of research and development of simpler and more sophisticated architectures for data and computationally efficient time-series analysis. The source code is publicly available on GitHub.


#### cs.LG, cs.CV, cs.ET, cs.LO, cs.PF

### [An Experimental Study of Real-Life LLM-Proposed Performance Improvements](https://arxiv.org/abs/2510.15494)
**作者**：Lirong Yi, Gregory Gay, Philipp Leitner

Large Language Models (LLMs) can generate code, but can they generate fast code? In this paper, we study this question using a dataset of 65 real-world tasks mined from open-source Java programs. We specifically select tasks where developers achieved significant speedups, and employ an automated pipeline to generate patches for these issues using two leading LLMs under four prompt variations. By rigorously benchmarking the results against the baseline and human-authored solutions, we demonstrate that LLM-generated code indeed improves performance over the baseline in most cases. However, patches proposed by human developers outperform LLM fixes by a statistically significant margin, indicating that LLMs often fall short of finding truly optimal solutions. We further find that LLM solutions are semantically identical or similar to the developer optimization idea in approximately two-thirds of cases, whereas they propose a more original idea in the remaining one-third. However, these original ideas only occasionally yield substantial performance gains.


#### cs.SE, cs.AI, cs.PF

### [Cleaning up the Mess](https://arxiv.org/abs/2510.15744)
**作者**：Haocong Luo, Ataberk Olgun, Maria Makeenkova, F. Nisa Bostanci, Geraldo F. Oliveira, A. Giray Yaglikci, Onur Mutlu

A MICRO 2024 best paper runner-up publication (the Mess paper) with all three artifact badges awarded (including "Reproducible") proposes a new benchmark to evaluate real and simulated memory system performance. In this paper, we demonstrate that the Ramulator 2.0 simulation results reported in the Mess paper are incorrect and, at the time of the publication of the Mess paper, irreproducible. We find that the authors of Mess paper made multiple trivial human errors in both the configuration and usage of the simulators. We show that by correctly configuring Ramulator 2.0, Ramulator 2.0's simulated memory system performance actually resembles real system characteristics well, and thus a key claimed contribution of the Mess paper is factually incorrect. We also identify that the DAMOV simulation results in the Mess paper use wrong simulation statistics that are unrelated to the simulated DRAM performance. Moreover, the Mess paper's artifact repository lacks the necessary sources to fully reproduce all the Mess paper's results.  Our work corrects the Mess paper's errors regarding Ramulator 2.0 and identifies important issues in the Mess paper's memory simulator evaluation methodology. We emphasize the importance of both carefully and rigorously validating simulation results and contacting simulator authors and developers, in true open source spirit, to ensure these simulators are used with correct configurations and as intended. We encourage the computer architecture community to correct the Mess paper's errors. This is necessary to prevent the propagation of inaccurate and misleading results, and to maintain the reliability of the scientific record. Our investigation also opens up questions about the integrity of the review and artifact evaluation processes. To aid future work, our source code and scripts are openly available at https: //github.com/CMU-SAFARI/ramulator2/tree/mess.


#### cs.AR, cs.PF

### [Scaling Multi Agent Reinforcement Learning for Underwater Acoustic Tracking via Autonomous Vehicles](https://arxiv.org/abs/2505.08222)
**作者**：Matteo Gallici, Ivan Masmitja, Mario Mart\'in

Autonomous vehicles (AV) offer a cost-effective solution for scientific missions such as underwater tracking. Recently, reinforcement learning (RL) has emerged as a powerful method for controlling AVs in complex marine environments. However, scaling these techniques to a fleet--essential for multi-target tracking or targets with rapid, unpredictable motion--presents significant computational challenges. Multi-Agent Reinforcement Learning (MARL) is notoriously sample-inefficient, and while high-fidelity simulators like Gazebo's LRAUV provide 100x faster-than-real-time single-robot simulations, they offer no significant speedup for multi-vehicle scenarios, making MARL training impractical. To address these limitations, we propose an iterative distillation method that transfers high-fidelity simulations into a simplified, GPU-accelerated environment while preserving high-level dynamics. This approach achieves up to a 30,000x speedup over Gazebo through parallelization, enabling efficient training via end-to-end GPU acceleration. Additionally, we introduce a novel Transformer-based architecture (TransfMAPPO) that learns multi-agent policies invariant to the number of agents and targets, significantly improving sample efficiency. Following large-scale curriculum learning conducted entirely on GPU, we perform extensive evaluations in Gazebo, demonstrating that our method maintains tracking errors below 5 meters over extended durations, even in the presence of multiple fast-moving targets. This work bridges the gap between large-scale MARL training and high-fidelity deployment, providing a scalable framework for autonomous fleet control in real-world sea missions.


#### cs.RO, cs.AI, cs.DC, cs.PF

### [msf-CNN: Patch-based Multi-Stage Fusion with Convolutional Neural Networks for TinyML](https://arxiv.org/abs/2505.11483)
**作者**：Zhaolan Huang, Emmanuel Baccelli

AI spans from large language models to tiny models running on microcontrollers (MCUs). Extremely memory-efficient model architectures are decisive to fit within an MCU's tiny memory budget e.g., 128kB of RAM. However, inference latency must remain small to fit real-time constraints. An approach to tackle this is patch-based fusion, which aims to optimize data flows across neural network layers. In this paper, we introduce msf-CNN, a novel technique that efficiently finds optimal fusion settings for convolutional neural networks (CNNs) by walking through the fusion solution space represented as a directed acyclic graph. Compared to previous work on CNN fusion for MCUs, msf-CNN identifies a wider set of solutions. We published an implementation of msf-CNN running on various microcontrollers (ARM Cortex-M, RISC-V, ESP32). We show that msf-CNN can achieve inference using 50% less RAM compared to the prior art (MCUNetV2 and StreamNet). We thus demonstrate how msf-CNN offers additional flexibility for system designers.


#### cs.LG, cs.PF

### [Lookup multivariate Kolmogorov-Arnold Networks](https://arxiv.org/abs/2509.07103)
**作者**：Sergey Pozdnyakov, Philippe Schwaller

High-dimensional linear mappings, or linear layers, dominate both the parameter count and the computational cost of most modern deep-learning models. We introduce a general-purpose drop-in replacement, lookup multivariate Kolmogorov-Arnold Networks (lmKANs), which deliver a substantially better trade-off between capacity and inference cost. Our construction expresses a general high-dimensional mapping through trainable low-dimensional multivariate functions. These functions can carry dozens or hundreds of trainable parameters each, and yet it takes only a few multiplications to compute them because they are implemented as spline lookup tables. Empirically, lmKANs reduce inference FLOPs by up to 6.0x while matching the flexibility of MLPs in general high-dimensional function approximation. In another feedforward fully connected benchmark, on the tabular-like dataset of randomly displaced methane configurations, lmKANs enable more than 10x higher H100 throughput at equal accuracy. Within frameworks of Convolutional Neural Networks, lmKAN-based CNNs cut inference FLOPs at matched accuracy by 1.6-2.1x and by 1.7x on the CIFAR-10 and ImageNet-1k datasets, respectively. Our code, including dedicated CUDA kernels, is available online at https://github.com/schwallergroup/lmkan.


#### cs.LG, cs.AI, cs.PF, cs.SE

### [Fine-Tuning Small Language Models for Domain-Specific AI: An Edge AI Perspective](https://arxiv.org/abs/2503.01933)
**作者**：Rakshit Aralimatti, Syed Abdul Gaffar Shakhadri, Kruthika KR, Kartik Basavaraj Angadi

Deploying large scale language models on edge devices faces inherent challenges such as high computational demands, energy consumption, and potential data privacy risks. This paper introduces the Shakti Small Language Models (SLMs) Shakti-100M, Shakti-250M, and Shakti-500M which target these constraints headon. By combining efficient architectures, quantization techniques, and responsible AI principles, the Shakti series enables on-device intelligence for smartphones, smart appliances, IoT systems, and beyond. We provide comprehensive insights into their design philosophy, training pipelines, and benchmark performance on both general tasks (e.g., MMLU, Hellaswag) and specialized domains (healthcare, finance, and legal). Our findings illustrate that compact models, when carefully engineered and fine-tuned, can meet and often exceed expectations in real-world edge-AI scenarios.


#### cs.LG, cs.AR, cs.CL, cs.CV

### [Hummingbird: A Smaller and Faster Large Language Model Accelerator on Embedded FPGA](https://arxiv.org/abs/2507.03308)
**作者**：Jindong Li, Tenglong Li, Ruiqi Chen, Guobin Shen, Dongcheng Zhao, Qian Zhang, Yi Zeng

Deploying large language models (LLMs) on embedded devices remains a significant research challenge due to the high computational and memory demands of LLMs and the limited hardware resources available in such environments. While embedded FPGAs have demonstrated performance and energy efficiency in traditional deep neural networks, their potential for LLM inference remains largely unexplored. Recent efforts to deploy LLMs on FPGAs have primarily relied on large, expensive cloud-grade hardware and have only shown promising results on relatively small LLMs, limiting their real-world applicability. In this work, we present Hummingbird, a novel FPGA accelerator designed specifically for LLM inference on embedded FPGAs. Hummingbird is smaller, targeting embedded FPGAs such as the KV260 and ZCU104 with 67% LUT, 39% DSP, and 42% power savings over existing research. Hummingbird is stronger, targeting LLaMA3-8B and supporting longer contexts, overcoming the typical 4GB memory constraint of embedded FPGAs through offloading strategies. Finally, Hummingbird is faste, achieving 4.8 tokens/s and 8.6 tokens/s for LLaMA3-8B on the KV260 and ZCU104 respectively, with 93-94% model bandwidth utilization, outperforming the prior 4.9 token/s for LLaMA2-7B with 84% bandwidth utilization baseline. We further demonstrate the viability of industrial applications by deploying Hummingbird on a cost-optimized Spartan UltraScale FPGA, paving the way for affordable LLM solutions at the edge.


#### cs.AR

### [Hive Hash Table: A Warp-Cooperative, Dynamically Resizable Hash Table for GPUs](https://arxiv.org/abs/2510.15095)
**作者**：Md Sabbir Hossain Polak, David Troendle, Byunghyun Jang

Hash tables are essential building blocks in data-intensive applications, yet existing GPU implementations often struggle with concurrent updates, high load factors, and irregular memory access patterns. We present Hive hash table, a high-performance, warp-cooperative and dynamically resizable GPU hash table that adapts to varying workloads without global rehashing.  Hive hash table makes three key contributions. First, a cache-aligned packed bucket layout stores key-value pairs as 64-bit words, enabling coalesced memory access and atomic updates via single-CAS operations. Second, warp-synchronous concurrency protocols - Warp-Aggregated-Bitmask-Claim (WABC) and Warp-Cooperative Match-and-Elect (WCME) - reduce contention to one atomic operation per warp while ensuring lock-free progress. Third, a load-factor-aware dynamic resizing strategy expands or contracts capacity in warp-parallel K-bucket batches using linear hashing, maintaining balanced occupancy. To handle insertions under heavy contention, Hive hash table employs a four-step strategy: replace, claim-and-commit, bounded cuckoo eviction, and overflow-stash fallback. This design provides lock-free fast paths and bounded recovery cost under contention determined by a fixed eviction depth, while eliminating ABA hazards during concurrent updates.  Experimental evaluation on an NVIDIA RTX 4090 shows Hive hash table sustains load factors up to 95% while delivering 1.5-2x higher throughput than state-of-the-art GPU hash tables (Slab-Hash, DyCuckoo, WarpCore) under mixed insert-delete-lookup workloads. On balanced workload, Hive hash table reaches 3.5 billion updates/s and nearly 4 billion lookups/s, demonstrating scalability and efficiency for GPU-accelerated data processing.


#### cs.DC

### [NEMO: Faster Parallel Execution for Highly Contended Blockchain Workloads (Full version)](https://arxiv.org/abs/2510.15122)
**作者**：Fran\c{c}ois Ezard, Can Umut Ileri, J\'er\'emie Decouchant

Following the design of more efficient blockchain consensus algorithms, the execution layer has emerged as the new performance bottleneck of blockchains, especially under high contention. Current parallel execution frameworks either rely on optimistic concurrency control (OCC) or on pessimistic concurrency control (PCC), both of which see their performance decrease when workloads are highly contended, albeit for different reasons. In this work, we present NEMO, a new blockchain execution engine that combines OCC with the object data model to address this challenge. NEMO introduces four core innovations: (i) a greedy commit rule for transactions using only owned objects; (ii) refined handling of dependencies to reduce re-executions; (iii) the use of incomplete but statically derivable read/write hints to guide execution; and (iv) a priority-based scheduler that favors transactions that unblock others. Through simulated execution experiments, we demonstrate that NEMO significantly reduces redundant computation and achieves higher throughput than representative approaches. For example, with 16 workers NEMO's throughput is up to 42% higher than the one of Block-STM, the state-of-the-art OCC approach, and 61% higher than the pessimistic concurrency control baseline used.


#### cs.DC

### [An Elastic Job Scheduler for HPC Applications on the Cloud](https://arxiv.org/abs/2510.15147)
**作者**：Aditya Bhosale, Kavitha Chandrasekar, Laxmikant Kale, Sara Kokkila-Schumacher

The last few years have seen an increase in adoption of the cloud for running HPC applications. The pay-as-you-go cost model of these cloud resources has necessitated the development of specialized programming models and schedulers for HPC jobs for efficient utilization of cloud resources. A key aspect of efficient utilization is the ability to rescale applications on the fly to maximize the utilization of cloud resources. Most commonly used parallel programming models like MPI have traditionally not supported autoscaling either in a cloud environment or on supercomputers. While more recent work has been done to implement this functionality in MPI, it is still nascent and requires additional programmer effort. Charm++ is a parallel programming model that natively supports dynamic rescaling through its migratable objects paradigm. In this paper, we present a Kubernetes operator to run Charm++ applications on a Kubernetes cluster. We then present a priority-based elastic job scheduler that can dynamically rescale jobs based on the state of a Kubernetes cluster to maximize cluster utilization while minimizing response time for high-priority jobs. We show that our elastic scheduler, with the ability to rescale HPC jobs with minimal overhead, demonstrates significant performance improvements over traditional static schedulers.


#### cs.DC

### [Spatiotemporal Traffic Prediction in Distributed Backend Systems via Graph Neural Networks](https://arxiv.org/abs/2510.15215)
**作者**：Zhimin Qiu, Feng Liu, Yuxiao Wang, Chenrui Hu, Ziyu Cheng, Di Wu

This paper addresses the problem of traffic prediction in distributed backend systems and proposes a graph neural network based modeling approach to overcome the limitations of traditional models in capturing complex dependencies and dynamic features. The system is abstracted as a graph with nodes and edges, where node features represent traffic and resource states, and adjacency relations describe service interactions. A graph convolution mechanism enables multi order propagation and aggregation of node features, while a gated recurrent structure models historical sequences dynamically, thus integrating spatial structures with temporal evolution. A spatiotemporal joint modeling module further fuses graph representation with temporal dependency, and a decoder generates future traffic predictions. The model is trained with mean squared error to minimize deviations from actual values. Experiments based on public distributed system logs construct combined inputs of node features, topology, and sequences, and compare the proposed method with mainstream baselines using MSE, RMSE, MAE, and MAPE. Results show that the proposed method achieves stable performance and low error across different prediction horizons and model depths, significantly improving the accuracy and robustness of traffic forecasting in distributed backend systems and verifying the potential of graph neural networks in complex system modeling.


#### cs.DC

### [Cloud-Enabled Virtual Prototypes](https://arxiv.org/abs/2510.15355)
**作者**：Tim Kraus, Axel Sauer, Ingo Feldner

The rapid evolution of embedded systems, along with the growing variety and complexity of AI algorithms, necessitates a powerful hardware/software co-design methodology based on virtual prototyping technologies. The market offers a diverse range of simulation solutions, each with its unique technological approach and therefore strengths and weaknesses. Additionally, with the increasing availability of remote on-demand computing resources and their adaptation throughout the industry, the choice of the host infrastructure for execution opens even more new possibilities for operational strategies. This work explores the dichotomy between local and cloud-based simulation environments, focusing on the trade-offs between scalability and privacy. We discuss how the setup of the compute infrastructure impacts the performance of the execution and security of data involved in the process. Furthermore, we highlight the development workflow associated with embedded AI and the critical role of efficient simulations in optimizing these algorithms. With the proposed solution, we aim to sustainably improve trust in remote simulations and facilitate the adoption of virtual prototyping practices.


#### cs.DC

### [Retrofitting Service Dependency Discovery in Distributed Systems](https://arxiv.org/abs/2510.15490)
**作者**：Diogo Landau, Gijs Blanken, Jorge Barbosa, Nishant Saurabh

Modern distributed systems rely on complex networks of interconnected services, creating direct or indirect dependencies that can propagate faults and cause cascading failures. To localize the root cause of performance degradation in these environments, constructing a service dependency graph is highly beneficial. However, building an accurate service dependency graph is impaired by complex routing techniques, such as Network Address Translation (NAT), an essential mechanism for connecting services across networks. NAT obfuscates the actual hosts running the services, causing existing run-time approaches that passively observe network metadata to fail in accurately inferring service dependencies. To this end, this paper introduces XXXX, a novel run-time system for constructing process-level service dependency graphs. It operates without source code instrumentation and remains resilient under complex network routing mechanisms, including NAT. XXXX implements a non-disruptive method of injecting metadata onto a TCP packet's header that maintains protocol correctness across host boundaries. In other words, if no receiving agent is present, the instrumentation leaves existing TCP connections unaffected, ensuring non-disruptive operation when it is partially deployed across hosts. We evaluated XXXX extensively against three state-of-the-art systems across nine scenarios, involving three network configurations (NAT-free, internal-NAT, external-NAT) and three microservice benchmarks. XXXX was the only approach that performed consistently across networking configurations. With regards to correctness, it performed on par with, or better than, the state-of-the-art with precision and recall values of 100% in the majority of the scenarios.


#### cs.DC

### [PRISM: Probabilistic Runtime Insights and Scalable Performance Modeling for Large-Scale Distributed Training](https://arxiv.org/abs/2510.15596)
**作者**：Alicia Golden, Michael Kuchnik, Samuel Hsia, Zachary DeVito, Gu-Yeon Wei, David Brooks, Carole-Jean Wu

Large model training beyond tens of thousands of GPUs is an uncharted territory. At such scales, disruptions to the training process are not a matter of if, but a matter of when -- a stochastic process degrading training productivity. Dynamic runtime variation will become increasingly more frequent as training scales up and GPUs are operated in increasingly power-limited and thermally-stressed environments. At the 64k GPU scale, we already observed 9% GPU time variability for frontier foundation model training. To understand potential causes of variability, we analyze GPU microbenchmarks at scale across a variety of platforms, showing up to 14% variation in GPU performance on GEMM workloads depending on training hardware and deployed environment.  Motivated by our analysis and the large design space around performance variability, we present PRISM -- a performance modeling framework that considers the stochastic nature of the large-scale distributed training. The core of PRISM is the statistical method that provides a quantifiable measure for probabilistic guarantees on training time. Using PRISM, we explore the design and optimization space of distributed training, from parallelization methods to next-generation training systems. PRISM is validated with real-system measurement, showing training time prediction accuracy with 20.8% Kolmogorov-Smirnov distance. Using PRISM, we demonstrate that, depending on computation node placement, up to 1.26x performance improvement potential is available if we factor in sensitivities of parallelization strategies to variation. In addition, we use PRISM to identify kernels to optimize for reducing performance variability and predict probability of slow-down for large-scale jobs where variation is magnified. We find optimizing communication kernels, such as AllGather and ReduceScatter, contribute most to minimizing variability in training step time.


#### cs.DC

### [A Post-Quantum Lower Bound for the Distributed Lov\'asz Local Lemma](https://arxiv.org/abs/2510.15698)
**作者**：Sebastian Brandt, Tim G\"ottlicher

In this work, we study the Lov\'asz local lemma (LLL) problem in the area of distributed quantum computing, which has been the focus of attention of recent advances in quantum computing [STOC'24, STOC'25, STOC'25]. We prove a lower bound of $2^{\Omega(\log^* n)}$ for the complexity of the distributed LLL in the quantum-LOCAL model. More specifically, we obtain our lower bound already for a very well-studied special case of the LLL, called sinkless orientation, in a stronger model than quantum-LOCAL, called the randomized online-LOCAL model. As a consequence, we obtain the same lower bounds for sinkless orientation and the distributed LLL also in a variety of other models studied across different research communities.  Our work provides the first superconstant lower bound for sinkless orientation and the distributed LLL in all of these models, addressing recently stated open questions. Moreover, to obtain our results, we develop an entirely new lower bound technique that we believe has the potential to become the first generic technique for proving post-quantum lower bounds for many of the most important problems studied in the context of locality.


#### cs.DC

### [The ArborX library: version 2.0](https://arxiv.org/abs/2507.23700)
**作者**：Andrey Prokopenko, Daniel Arndt, Damien Lebrun-Grandi\'e, Bruno Turcksin

This paper provides an overview of the 2.0 release of the ArborX library, a performance portable geometric search library based on Kokkos. We describe the major changes in ArborX 2.0 including a new interface for the library to support a wider range of user problems, new search data structures (brute force, distributed), support for user functions to be executed on the results (callbacks), and an expanded set of the supported algorithms (ray tracing, clustering).


#### cs.DC

### [BeLLMan: Controlling LLM Congestion](https://arxiv.org/abs/2510.15330)
**作者**：Tella Rajashekhar Reddy, Atharva Deshmukh, Karan Tandon, Rohan Gandhi, Anjaly Parayil, Debopam Bhattacherjee

Large language model (LLM) applications are blindfolded to the infrastructure underneath and generate tokens autoregressively, indifferent to the system load, thus risking inferencing latency inflation and poor user experience. Our first-cut controller, named beLLMan, enables the LLM infrastructure to actively and progressively signal the first-party LLM application to adjust the output length in response to changing system load. On a real testbed with H100 GPUs, beLLMan helps keep inferencing latency under control (upto 8X lower end-to-end latency) and reduces energy consumption by 25% (while serving 19% more requests) during periods of congestion for a summarization workload.


#### cs.DC, cs.AI, cs.CL, cs.NI

### [(Almost) Perfect Discrete Iterative Load Balancing](https://arxiv.org/abs/2510.15473)
**作者**：Petra Berenbrink, Robert Els\"asser, Tom Friedetzky, Hamed Hosseinpour, Dominik Kaaser, Peter Kling, Thomas Sauerwald

We consider discrete, iterative load balancing via matchings on arbitrary graphs. Initially each node holds a certain number of tokens, defining the load of the node, and the objective is to redistribute the tokens such that eventually each node has approximately the same number of tokens. We present results for a general class of simple local balancing schemes where the tokens are balanced via matchings. In each round the process averages the tokens of any two matched nodes. If the sum of their tokens is odd, the node to receive the one excess token is selected at random. Our class covers three popular models: in the matching model a new matching is generated randomly in each round, in the balancing circuit model a fixed sequence of matchings is applied periodically, and in the asynchronous model the load is balanced over a randomly chosen edge.  We measure the quality of a load vector by its discrepancy, defined as the difference between the maximum and minimum load across all nodes. As our main result we show that with high probability our discrete balancing scheme reaches a discrepancy of $3$ in a number of rounds which asymptotically matches the spectral bound for continuous load balancing with fractional load.  This result improves and tightens a long line of previous works, by not only achieving a small constant discrepancy (instead of a non-explicit, large constant) but also holding for arbitrary instead of regular graphs. The result also demonstrates that in the general model we consider, discrete load balancing is no harder than continuous load balancing.


#### cs.DC, math.PR

### [Balancing Fairness and Performance in Multi-User Spark Workloads with Dynamic Scheduling (extended version)](https://arxiv.org/abs/2510.15485)
**作者**：D\=avis Ka\v{z}emaks, Laurens Versluis, Burcu Kulahcioglu Ozkan, J\'er\'emie Decouchant

Apache Spark is a widely adopted framework for large-scale data processing. However, in industrial analytics environments, Spark's built-in schedulers, such as FIFO and fair scheduling, struggle to maintain both user-level fairness and low mean response time, particularly in long-running shared applications. Existing solutions typically focus on job-level fairness which unintentionally favors users who submit more jobs. Although Spark offers a built-in fair scheduler, it lacks adaptability to dynamic user workloads and may degrade overall job performance. We present the User Weighted Fair Queuing (UWFQ) scheduler, designed to minimize job response times while ensuring equitable resource distribution across users and their respective jobs. UWFQ simulates a virtual fair queuing system and schedules jobs based on their estimated finish times under a bounded fairness model. To further address task skew and reduce priority inversions, which are common in Spark workloads, we introduce runtime partitioning, a method that dynamically refines task granularity based on expected runtime. We implement UWFQ within the Spark framework and evaluate its performance using multi-user synthetic workloads and Google cluster traces. We show that UWFQ reduces the average response time of small jobs by up to 74% compared to existing built-in Spark schedulers and to state-of-the-art fair scheduling algorithms.


#### cs.DC, cs.DB, cs.SY, eess.SY

### [GOGH: Correlation-Guided Orchestration of GPUs in Heterogeneous Clusters](https://arxiv.org/abs/2510.15652)
**作者**：Ahmad Raeisi, Mahdi Dolati, Sina Darabi, Sadegh Talebi, Patrick Eugster, Ahmad Khonsari

The growing demand for computational resources in machine learning has made efficient resource allocation a critical challenge, especially in heterogeneous hardware clusters where devices vary in capability, age, and energy efficiency. Upgrading to the latest hardware is often infeasible, making sustainable use of existing, mixed-generation resources essential. In this paper, we propose a learning-based architecture for managing machine learning workloads in heterogeneous clusters. The system operates online, allocating resources to incoming training or inference requests while minimizing energy consumption and meeting performance requirements. It uses two neural networks: the first provides initial estimates of how well a new model will utilize different hardware types and how it will affect co-located models. An optimizer then allocates resources based on these estimates. After deployment, the system monitors real performance and uses this data to refine its predictions via a second neural network. This updated model improves estimates not only for the current hardware but also for hardware not initially allocated and for co-location scenarios not yet observed. The result is an adaptive, iterative approach that learns over time to make more effective resource allocation decisions in heterogeneous deep learning clusters.


#### cs.DC, cs.LG

### [SYMI: Efficient Mixture-of-Experts Training via Model and Optimizer State Decoupling](https://arxiv.org/abs/2504.19925)
**作者**：Athinagoras Skiadopoulos, Mark Zhao, Swapnil Gandhi, Thomas Norrie, Shrijeet Mukherjee, Christos Kozyrakis

Mixture-of-Experts (MoE) models have become a widely-adopted solution to continue scaling model sizes without a corresponding linear increase in compute. During MoE model training, each input token is dynamically routed to a subset of experts -- sparsely-activated feed-forward networks -- within each transformer layer. The distribution of tokens assigned to each expert varies widely and rapidly over the course of training. To handle the wide load imbalance across experts, current systems are forced to either drop tokens assigned to popular experts, degrading convergence, or frequently rebalance resources allocated to each expert based on popularity, incurring high state migration overheads.  To break this performance-accuracy tradeoff, we introduce SYMI, an adaptive MoE training system. The key insight of SYMI is to decouple the placement of expert parameters from their large optimizer state. SYMI statically partitions the optimizer of each expert across all training nodes. Meanwhile, SYMI dynamically adjusts the placement of expert parameters by repurposing existing weight updates, avoiding migration overheads. In doing so, SYMI right-sizes the GPU resources allocated to each expert, on a per-iteration basis, with minimal overhead. Compared to state-of-the-art MoE training systems, DeepSpeed and FlexMoE, SYMI is able to achieve a 30.5% and 25.9% faster time-to-convergence, respectively.


#### cs.DC, cs.LG

### [Funky: Cloud-Native FPGA Virtualization and Orchestration](https://arxiv.org/abs/2510.15755)
**作者**：Atsushi Koshiba, Charalampos Mainas, Pramod Bhatotia

The adoption of FPGAs in cloud-native environments is facing impediments due to FPGA limitations and CPU-oriented design of orchestrators, as they lack virtualization, isolation, and preemption support for FPGAs. Consequently, cloud providers offer no orchestration services for FPGAs, leading to low scalability, flexibility, and resiliency.  This paper presents Funky, a full-stack FPGA-aware orchestration engine for cloud-native applications. Funky offers primary orchestration services for FPGA workloads to achieve high performance, utilization, scalability, and fault tolerance, accomplished by three contributions: (1) FPGA virtualization for lightweight sandboxes, (2) FPGA state management enabling task preemption and checkpointing, and (3) FPGA-aware orchestration components following the industry-standard CRI/OCI specifications.  We implement and evaluate Funky using four x86 servers with Alveo U50 FPGA cards. Our evaluation highlights that Funky allows us to port 23 OpenCL applications from the Xilinx Vitis and Rosetta benchmark suites by modifying 3.4% of the source code while keeping the OCI image sizes 28.7 times smaller than AMD's FPGA-accessible Docker containers. In addition, Funky incurs only 7.4% performance overheads compared to native execution, while providing virtualization support with strong hypervisor-enforced isolation and cloud-native orchestration for a set of distributed FPGAs. Lastly, we evaluate Funky's orchestration services in a large-scale cluster using Google production traces, showing its scalability, fault tolerance, and scheduling efficiency.


#### cs.DC, cs.OS

### [Targeted Attacks and Defenses for Distributed Federated Learning in Vehicular Networks](https://arxiv.org/abs/2510.15109)
**作者**：Utku Demir, Tugba Erpek, Yalin E. Sagduyu, Sastry Kompella, Mengran Xue

In emerging networked systems, mobile edge devices such as ground vehicles and unmanned aerial system (UAS) swarms collectively aggregate vast amounts of data to make machine learning decisions such as threat detection in remote, dynamic, and infrastructure-constrained environments where power and bandwidth are scarce. Federated learning (FL) addresses these constraints and privacy concerns by enabling nodes to share local model weights for deep neural networks instead of raw data, facilitating more reliable decision-making than individual learning. However, conventional FL relies on a central server to coordinate model updates in each learning round, which imposes significant computational burdens on the central node and may not be feasible due to the connectivity constraints. By eliminating dependence on a central server, distributed federated learning (DFL) offers scalability, resilience to node failures, learning robustness, and more effective defense strategies. Despite these advantages, DFL remains vulnerable to increasingly advanced and stealthy cyberattacks. In this paper, we design sophisticated targeted training data poisoning and backdoor (Trojan) attacks, and characterize the emerging vulnerabilities in a vehicular network. We analyze how DFL provides resilience against such attacks compared to individual learning and present effective defense mechanisms to further strengthen DFL against the emerging cyber threats.


#### cs.NI, cs.AI, cs.DC, cs.LG, eess.SP

### [Grassroots Logic Programs: A Secure, Multiagent, Concurrent, Logic Programming Language](https://arxiv.org/abs/2510.15747)
**作者**：Ehud Shapiro

Grassroots platforms are distributed applications run by\linebreak cryptographically-identified people on their networked personal devices, where multiple disjoint platform instances emerge independently and coalesce when they interoperate. Their foundation is the grassroots social graph, upon which grassroots social networks, grassroots cryptocurrencies, and grassroots democratic federations can be built.  Grassroots platforms have yet to be implemented, the key challenge being faulty and malicious participants: without secure programming support, correct participants cannot reliably identify each other, establish secure communication, or verify each other's code integrity.  We present Grassroots Logic Programs (GLP), a secure, multiagent, concurrent, logic programming language for implementing grassroots platforms. GLP extends logic programs with paired single-reader/single-writer (SRSW) logic variables, providing secure communication channels among cryptographically-identified people through encrypted, signed and attested messages, which enable identity and code integrity verification. We present GLP progressively: logic programs, concurrent GLP, multiagent GLP, augmenting it with cryptographic security, and providing smartphone implementation-ready specifications. We prove safety properties including that GLP computations are deductions, SRSW preservation, acyclicity, and monotonicity. We prove multiagent GLP is grassroots and that GLP streams achieve blockchain security properties. We present a grassroots social graph protocol establishing authenticated peer-to-peer connections and demonstrate secure grassroots social networking applications.


#### cs.PL, cs.CR, cs.DC, cs.LO, cs.MA

### [Personalized Semi-Supervised Federated Learning for Human Activity Recognition](https://arxiv.org/abs/2104.08094)
**作者**：Riccardo Presotto, Gabriele Civitarese, Claudio Bettini

One of the major open problems in sensor-based Human Activity Recognition (HAR) is the scarcity of labeled data. Among the many solutions to address this challenge, semi-supervised learning approaches represent a promising direction. However, their centralised architecture incurs in the scalability and privacy problems that arise when the process involves a large number of users. Federated Learning (FL) is a promising paradigm to address these problems. However, the FL methods that have been proposed for HAR assume that the participating users can always obtain labels to train their local models (i.e., they assume a fully supervised setting). In this work, we propose FedAR: a novel hybrid method for HAR that combines semi-supervised and federated learning to take advantage of the strengths of both approaches. FedAR combines active learning and label propagation to semi-automatically annotate the local streams of unlabeled sensor data, and it relies on FL to build a global activity model in a scalable and privacy-aware fashion. FedAR also includes a transfer learning strategy to fine-tune the global model on each user. We evaluated our method on two public datasets, showing that FedAR reaches recognition rates and personalization capabilities similar to state-of-the-art FL supervised approaches. As a major advantage, FedAR only requires a very limited number of annotated data to populate a pre-trained model and a small number of active learning questions that quickly decrease while using the system, leading to an effective and scalable solution for the data scarcity problem of HAR.


#### cs.LG, cs.DC

### [MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production](https://arxiv.org/abs/2505.11432)
**作者**：Chao Jin, Ziheng Jiang, Zhihao Bai, Zheng Zhong, Juncai Liu, Xiang Li, Ningxin Zheng, Xi Wang, Cong Xie, Qi Huang, Wen Heng, Yiyuan Ma, Wenlei Bao, Size Zheng, Yanghua Peng, Haibin Lin, Xuanzhe Liu, Xin Jin, Xin Liu

We present MegaScale-MoE, a production system tailored for the efficient training of large-scale mixture-of-experts (MoE) models. MoE emerges as a promising architecture to scale large language models (LLMs) to unprecedented sizes, thereby enhancing model performance. However, existing MoE training systems experience a degradation in training efficiency, exacerbated by the escalating scale of MoE models and the continuous evolution of hardware.  Recognizing the pivotal role of efficient communication in enhancing MoE training, MegaScale-MoE customizes communication-efficient parallelism strategies for attention and FFNs in each MoE layer and adopts a holistic approach to overlap communication with computation at both inter- and intra-operator levels. Additionally, MegaScale-MoE applies communication compression with adjusted communication patterns to lower precision, further improving training efficiency. When training a 352B MoE model on 1,440 NVIDIA Hopper GPUs, MegaScale-MoE achieves a training throughput of 1.41M tokens/s, improving the efficiency by 1.88$\times$ compared to Megatron-LM. We share our operational experience in accelerating MoE training and hope that by offering our insights in system design, this work will motivate future research in MoE systems.


#### cs.LG, cs.DC

### [Maratona Linux a tale of upgrading from Ubuntu 20.04 to 22.04](https://arxiv.org/abs/2510.15263)
**作者**：Davi Ant\^onio da Silva Santos, Bruno C\'esar Ribas

Maratona Linux is the development environment used since 2016 on the ``Maratona de Programa\c{c}\~ao'', ICPC's South American regional contest. It consists of Debian packages that modify a standard Ubuntu installation in order to make it suitable for the competition, installing IDEs, documentation, compilers, debuggers, interpreters, and enforcing network restrictions. The project, which began based on Ubuntu 16.04, has been successfully migrated from 20.04 to 22.04, the current Long-term Support (LTS) version. The project has also been improved by adding static analyzers, updating the package dependency map, splitting large packages, and enhancing the packaging pipeline.


#### cs.OS
