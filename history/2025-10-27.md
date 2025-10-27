# ArXiv 新论文更新（2025-10-27)

### [xMem: A CPU-Based Approach for Accurate Estimation of GPU Memory in Deep Learning Training Workloads](https://arxiv.org/abs/2510.21048)
**作者**：Jiabo Shi, Dimitrios Pezaros, Yehia Elkhatib

The global scarcity of GPUs necessitates more sophisticated strategies for Deep Learning jobs in shared cluster environments. Accurate estimation of how much GPU memory a job will require is fundamental to enabling advanced scheduling and GPU sharing, which helps prevent out-of-memory (OOM) errors and resource underutilization. However, existing estimation methods have limitations. Approaches relying on static analysis or historical data with machine learning often fail to accurately capture runtime dynamics. Furthermore, direct GPU analysis consumes scarce resources, and some techniques require intrusive code modifications. Thus, the key challenge lies in precisely estimating dynamic memory requirements, including memory allocator nuances, without consuming GPU resources and non-intrusive code changes. To address this challenge, we propose xMem, a novel framework that leverages CPU-only dynamic analysis to accurately estimate peak GPU memory requirements a priori. We conducted a thorough evaluation of xMem against state-of-the-art solutions using workloads from 25 different models, including architectures like Convolutional Neural Networks and Transformers. The analysis of 5209 runs, which includes ANOVA and Monte Carlo results, highlights xMem's benefits: it decreases the median relative error by 91% and significantly reduces the probability of estimation failure as safe OOM thresholds by 75%, meaning that the estimated value can often be used directly without causing OOM. Ultimately, these improvements lead to a 368% increase in memory conservation potential over current solutions.


#### cs.PF, cs.DC, cs.LG

### [Accelerating Mobile Inference through Fine-Grained CPU-GPU Co-Execution](https://arxiv.org/abs/2510.21081)
**作者**：Zhuojin Li, Marco Paolieri, Leana Golubchik

Deploying deep neural networks on mobile devices is increasingly important but remains challenging due to limited computing resources. On the other hand, their unified memory architecture and narrower gap between CPU and GPU performance provide an opportunity to reduce inference latency by assigning tasks to both CPU and GPU. The main obstacles for such collaborative execution are the significant synchronization overhead required to combine partial results, and the difficulty of predicting execution times of tasks assigned to CPU and GPU (due to the dynamic selection of implementations and parallelism level). To overcome these obstacles, we propose both a lightweight synchronization mechanism based on OpenCL fine-grained shared virtual memory (SVM) and machine learning models to accurately predict execution times. Notably, these models capture the performance characteristics of GPU kernels and account for their dispatch times. A comprehensive evaluation on four mobile platforms shows that our approach can quickly select CPU-GPU co-execution strategies achieving up to 1.89x speedup for linear layers and 1.75x speedup for convolutional layers (close to the achievable maximum values of 2.01x and 1.87x, respectively, found by exhaustive grid search on a Pixel~5 smartphone).


#### cs.LG, cs.DC, cs.PF

### [GreenMalloc: Allocator Optimisation for Industrial Workloads](https://arxiv.org/abs/2510.21405)
**作者**：Aidan Dakhama, W. B. Langdon, Hector D. Menendez, Karine Even-Mendoza

We present GreenMalloc, a multi objective search-based framework for automatically configuring memory allocators. Our approach uses NSGA II and rand_malloc as a lightweight proxy benchmarking tool. We efficiently explore allocator parameters from execution traces and transfer the best configurations to gem5, a large system simulator, in a case study on two allocators: the GNU C/CPP compiler's glibc malloc and Google's TCMalloc. Across diverse workloads, our empirical results show up to 4.1 percantage reduction in average heap usage without loss of runtime efficiency; indeed, we get a 0.25 percantage reduction.


#### cs.SE, cs.AR, cs.PF

### [Selective Parallel Loading of Large-Scale Compressed Graphs with ParaGrapher](https://arxiv.org/abs/2404.19735)
**作者**：Mohsen Koohi Esfahani, Marco D'Antonio, Syed Ibtisam Tauhidi, Thai Son Mai, Hans Vandierendonck

Comprehensive evaluation is one of the basis of experimental science. In High-Performance Graph Processing, a thorough evaluation of contributions becomes more achievable by supporting common input formats over different frameworks. However, each framework creates its specific format, which may not support reading large-scale real-world graph datasets. This shows a demand for high-performance libraries capable of loading graphs to (i) accelerate designing new graph algorithms, (ii) to evaluate the contributions on a wide range of graph algorithms, and (iii) to facilitate easy and fast comparison over different graph frameworks.  To that end, we present ParaGrapher, a high-performance API and library for loading large-scale and compressed graphs. ParaGrapher supports different types of requests for accessing graphs in shared- and distributed-memory and out-of-core graph processing. We explain the design of ParaGrapher and present a performance model of graph decompression, which is used for evaluation of ParaGrapher over three storage types. Our evaluation shows that by decompressing compressed graphs in WebGraph format, ParaGrapher delivers up to 3.2 times speedup in loading and up to 5.2 times speedup in end-to-end execution in comparison to the binary and textual formats.  ParaGrapher is available online on https://blogs.qub.ac.uk/DIPSA/ParaGrapher/.


#### cs.AR, cs.PF, cs.SE

### [FIFOAdvisor: A DSE Framework for Automated FIFO Sizing of High-Level Synthesis Designs](https://arxiv.org/abs/2510.20981)
**作者**：Stefan Abi-Karam, Rishov Sarkar, Suhail Basalama, Jason Cong, Callie Hao

Dataflow hardware designs enable efficient FPGA implementations via high-level synthesis (HLS), but correctly sizing first-in-first-out (FIFO) channel buffers remains challenging. FIFO sizes are user-defined and balance latency and area-undersized FIFOs cause stalls and potential deadlocks, while oversized ones waste memory. Determining optimal sizes is non-trivial: existing methods rely on restrictive assumptions, conservative over-allocation, or slow RTL simulations. We emphasize that runtime-based analyses (i.e., simulation) are the only reliable way to ensure deadlock-free FIFO optimization for data-dependent designs.  We present FIFOAdvisor, a framework that automatically determines FIFO sizes in HLS designs. It leverages LightningSim, a 99.9\% cycle-accurate simulator supporting millisecond-scale incremental runs with new FIFO configurations. FIFO sizing is formulated as a dual-objective black-box optimization problem, and we explore heuristic and search-based methods to characterize the latency-resource trade-off. FIFOAdvisor also integrates with Stream-HLS, a framework for optimizing affine dataflow designs lowered from C++, MLIR, or PyTorch, enabling deeper optimization of FIFOs in these workloads.  We evaluate FIFOAdvisor on Stream-HLS design benchmarks spanning linear algebra and deep learning workloads. Our results reveal Pareto-optimal latency-memory frontiers across optimization strategies. Compared to baseline designs, FIFOAdvisor achieves much lower memory usage with minimal delay overhead. Additionally, it delivers significant runtime speedups over traditional HLS/RTL co-simulation, making it practical for rapid design space exploration. We further demonstrate its capability on a complex accelerator with data-dependent control flow.  Code and results: https://github.com/sharc-lab/fifo-advisor


#### cs.AR

### [Hardware-Efficient Accurate 4-bit Multiplier for Xilinx 7 Series FPGAs](https://arxiv.org/abs/2510.21533)
**作者**：Misaki Kida, Shimpei Sato

As IoT and edge inference proliferate,there is a growing need to simultaneously optimize area and delay in lookup-table (LUT)-based multipliers that implement large numbers of low-bitwidth operations in parallel. This paper proposes a hardwareefficientaccurate 4-bit multiplier design for AMD Xilinx 7-series FPGAs using only 11 LUTs and two CARRY4 blocks. By reorganizing the logic functions mapped to the LUTs, the proposed method reduces the LUT count by one compared with the prior 12-LUT design while also shortening the critical path. Evaluation confirms that the circuit attains minimal resource usage and a critical-path delay of 2.750 ns.


#### cs.AR

### [Accelerating Electrostatics-based Global Placement with Enhanced FFT Computation](https://arxiv.org/abs/2510.21547)
**作者**：Hangyu Zhang, Sachin S. Sapatnekar

Global placement is essential for high-quality and efficient circuit placement for complex modern VLSI designs. Recent advancements, such as electrostatics-based analytic placement, have improved scalability and solution quality. This work demonstrates that using an accelerated FFT technique, AccFFT, for electric field computation significantly reduces runtime. Experimental results on standard benchmarks show significant improvements when incorporated into the ePlace-MS and Pplace-MS algorithms, e.g., a 5.78x speedup in FFT computation and a 32% total runtime improvement against ePlace-MS, with 1.0% reduction of scaled half-perimeter wirelength after detailed placement.


#### cs.AR

### [Lincoln AI Computing Survey (LAICS) and Trends](https://arxiv.org/abs/2510.20931)
**作者**：Albert Reuther, Peter Michaleas, Michael Jones, Vijay Gadepally, Jeremy Kepner

In the past year, generative AI (GenAI) models have received a tremendous amount of attention, which in turn has increased attention to computing systems for training and inference for GenAI. Hence, an update to this survey is due. This paper is an update of the survey of AI accelerators and processors from past seven years, which is called the Lincoln AI Computing Survey -- LAICS (pronounced "lace"). This multi-year survey collects and summarizes the current commercial accelerators that have been publicly announced with peak performance and peak power consumption numbers. In the same tradition of past papers of this survey, the performance and power values are plotted on a scatter graph, and a number of dimensions and observations from the trends on this plot are again discussed and analyzed. Market segments are highlighted on the scatter plot, and zoomed plots of each segment are also included. A brief description of each of the new accelerators that have been added in the survey this year is included, and this update features a new categorization of computing architectures that implement each of the accelerators.


#### cs.DC, cs.AR

### [Towards Straggler-Resilient Split Federated Learning: An Unbalanced Update Approach](https://arxiv.org/abs/2510.21155)
**作者**：Dandan Liang, Jianing Zhang, Evan Chen, Zhe Li, Rui Li, Haibo Yang

Split Federated Learning (SFL) enables scalable training on edge devices by combining the parallelism of Federated Learning (FL) with the computational offloading of Split Learning (SL). Despite its great success, SFL suffers significantly from the well-known straggler issue in distributed learning systems. This problem is exacerbated by the dependency between Split Server and clients: the Split Server side model update relies on receiving activations from clients. Such synchronization requirement introduces significant time latency, making straggler a critical bottleneck to the scalability and efficiency of the system. To mitigate this problem, we propose MU-SplitFed, a straggler-resilient SFL algorithm in zeroth-order optimization that decouples training progress from straggler delays via a simple yet effective unbalanced update mechanism.  By enabling the server to perform $\tau$ local updates per client round, MU-SplitFed achieves a convergence rate of $O(\sqrt{d/(\tau T)})$ for non-convex objectives, demonstrating a linear speedup of $\tau$ in communication rounds. Experiments demonstrate that MU-SplitFed consistently outperforms baseline methods with the presence of stragglers and effectively mitigates their impact through adaptive tuning of $\tau$. The code for this project is available at https://github.com/Johnny-Zip/MU-SplitFed.


#### cs.DC, cs.AI, cs.LG

### [From SLA to vendor-neutral metrics: An intelligent knowledge-based approach for multi-cloud SLA-based broker](https://arxiv.org/abs/2510.21173)
**作者**：V\'ictor Ramp\'erez, Javier Soriano, David Lizcano, Shadi Aljawarneh, Juan A. Lara

Cloud computing has been consolidated as a support for the vast majority of current and emerging technologies. However, there are some barriers that prevent the exploitation of the full potential of this technology. First, the major cloud providers currently put the onus of implementing the mechanisms that ensure compliance with the desired service levels on cloud consumers. However, consumers do not have the required expertise. Since each cloud provider exports a different set of low-level metrics, the strategies defined to ensure compliance with the established service-level agreement (SLA) are bound to a particular cloud provider. This fosters provider lock-in and prevents consumers from benefiting from the advantages of multi-cloud environments. This paper presents a solution to the problem of automatically translating SLAs into objectives expressed as metrics that can be measured across multiple cloud providers. First, we propose an intelligent knowledge-based system capable of automatically translating high-level SLAs defined by cloud consumers into a set of conditions expressed as vendor-neutral metrics, providing feedback to cloud consumers (intelligent tutoring system). Secondly, we present the set of vendor-neutral metrics and explain how they can be measured for the different cloud providers. Finally, we report a validation based on two use cases (IaaS and PaaS) in a multi-cloud environment formed by leading cloud providers. This evaluation has demonstrated that, thanks to the complementarity of the two solutions, cloud consumers can automatically and transparently exploit the multi-cloud in many application domains, as endorsed by the cloud experts consulted in the course of this study.


#### cs.DC

### [Generative Federated Learning for Smart Prediction and Recommendation Applications](https://arxiv.org/abs/2510.21183)
**作者**：Anwesha Mukherjee, Rajkumar Buyya

This paper proposes a generative adversarial network and federated learning-based model to address various challenges of the smart prediction and recommendation applications, such as high response time, compromised data privacy, and data scarcity. The integration of the generative adversarial network and federated learning is referred to as Generative Federated Learning (GFL). As a case study of the proposed model, a heart health monitoring application is considered. The realistic synthetic datasets are generated using the generated adversarial network-based proposed algorithm for improving data diversity, data quality, and data augmentation, and remove the data scarcity and class imbalance issues. In this paper, we implement the centralized and decentralized federated learning approaches in an edge computing paradigm. In centralized federated learning, the edge nodes communicate with the central server to build the global and personalized local models in a collaborative manner. In the decentralized federated learning approach, the edge nodes communicate among themselves to exchange model updates for collaborative training. The comparative study shows that the proposed framework outperforms the existing heart health monitoring applications. The results show that using the proposed framework (i) the prediction accuracy is improved by 12% than the conventional framework, and (ii) the response time is reduced by 73% than the conventional cloud-only system.


#### cs.DC

### [Arbitration-Free Consistency is Available (and Vice Versa)](https://arxiv.org/abs/2510.21304)
**作者**：Hagit Attiya, Constantin Enea, Enrique Rom\'an-Calvo

The fundamental tension between \emph{availability} and \emph{consistency} shapes the design of distributed storage systems. Classical results capture extreme points of this trade-off: the CAP theorem shows that strong models like linearizability preclude availability under partitions, while weak models like causal consistency remain implementable without coordination. These theorems apply to simple read-write interfaces, leaving open a precise explanation of the combinations of object semantics and consistency models that admit available implementations.  This paper develops a general semantic framework in which storage specifications combine operation semantics and consistency models. The framework encompasses a broad range of objects (key-value stores, counters, sets, CRDTs, and transactional databases) and consistency models (from causal consistency and sequential consistency to snapshot isolation and transactional and non-transactional SQL).  Within this framework, we prove the \emph{Arbitration-Free Consistency} (AFC) theorem, showing that an object specification within a consistency model admits an available implementation if and only if it is \emph{arbitration-free}, that is, it does not require a total arbitration order to resolve visibility or read dependencies.  The AFC theorem unifies and generalizes previous results, revealing arbitration-freedom as the fundamental property that delineates coordination-free consistency from inherently synchronized behavior.


#### cs.DC

### [Parsley's Group Size Study](https://arxiv.org/abs/2510.21348)
**作者**：Jo\~ao A. Silva, Herv\'e Paulino, Jo\~ao M. Louren\c{c}o

Parsley is a resilient group-based Distributed Hash Table that incorporates a preemptive peer relocation technique and a dynamic data sharding mechanism to enhance robustness and balance. In addition to the hard limits on group size, defined by minimum and maximum thresholds, Parsley introduces two soft limits that define a target interval for maintaining stable group sizes. These soft boundaries allow the overlay to take proactive measures to prevent violations of the hard limits, improving system stability under churn. This work provides an in-depth analysis of the rationale behind the parameter values adopted for Parsley's evaluation. Unlike related systems, which specify group size limits without justification, we conduct a systematic overlay characterization study to understand the effects of these parameters on performance and scalability. The study examines topology operations, the behavior of large groups, and the overall trade-offs observed, offering a grounded explanation for the chosen configuration values.


#### cs.DC

### [Learning to Schedule: A Supervised Learning Framework for Network-Aware Scheduling of Data-Intensive Workloads](https://arxiv.org/abs/2510.21419)
**作者**：Sankalpa Timilsina, Susmit Shannigrahi

Distributed cloud environments hosting data-intensive applications often experience slowdowns due to network congestion, asymmetric bandwidth, and inter-node data shuffling. These factors are typically not captured by traditional host-level metrics like CPU or memory. Scheduling without accounting for these conditions can lead to poor placement decisions, longer data transfers, and suboptimal job performance. We present a network-aware job scheduler that uses supervised learning to predict the completion time of candidate jobs. Our system introduces a prediction-and-ranking mechanism that collects real-time telemetry from all nodes, uses a trained supervised model to estimate job duration per node, and ranks them to select the best placement. We evaluate the scheduler on a geo-distributed Kubernetes cluster deployed on the FABRIC testbed by running network-intensive Spark workloads. Compared to the default Kubernetes scheduler, which makes placement decisions based on current resource availability alone, our proposed supervised scheduler achieved 34-54% higher accuracy in selecting optimal nodes for job placement. The novelty of our work lies in the demonstration of supervised learning for real-time, network-aware job scheduling on a multi-site cluster.


#### cs.DC

### [On Reduction and Synthesis of Petri's Cycloids](https://arxiv.org/abs/2510.21493)
**作者**：R\"udiger Valk, Daniel Moldt

Cycloids are particular Petri nets for modelling processes of actions and events, belonging to the fundaments of Petri's general systems theory. Defined by four parameters they provide an algebraic formalism to describe strongly synchronized sequential processes. To further investigate their structure, reduction systems of cycloids are defined in the style of rewriting systems and properties of irreducible cycloids are proved. In particular the synthesis of cycloid parameters from their Petri net structure is derived, leading to an efficient method for a decision procedure for cycloid isomorphism.


#### cs.DC

### [ProFaaStinate: Delaying Serverless Function Calls to Optimize Platform Performance](https://arxiv.org/abs/2309.15471)
**作者**：Trever Schirmer, Natalie Carl, Tobias Pfandzelter, David Bermbach

Function-as-a-Service (FaaS) enables developers to run serverless applications without managing operational tasks. In current FaaS platforms, both synchronous and asynchronous calls are executed immediately. In this paper, we present ProFaaStinate, which extends serverless platforms to enable delayed execution of asynchronous function calls. This allows platforms to execute calls at convenient times with higher resource availability or lower load. ProFaaStinate is able to optimize performance without requiring deep integration into the rest of the platform, or a complex systems model. In our evaluation, our prototype built on top of Nuclio can reduce request response latency and workflow duration while also preventing the system from being overloaded during load peaks. Using a document preparation use case, we show a 54% reduction in average request response latency. This reduction in resource usage benefits both platforms and users as cost savings.


#### cs.DC

### [LIDC: A Location Independent Multi-Cluster Computing Framework for Data Intensive Science](https://arxiv.org/abs/2510.21373)
**作者**：Sankalpa Timilsina, Susmit Shannigrahi

Scientific communities are increasingly using geographically distributed computing platforms. The current methods of compute placement predominantly use logically centralized controllers such as Kubernetes (K8s) to match tasks to available resources. However, this centralized approach is unsuitable in multi-organizational collaborations. Furthermore, workflows often need to use manual configurations tailored for a single platform and cannot adapt to dynamic changes across infrastructure. Our work introduces a decentralized control plane for placing computations on geographically dispersed compute clusters using semantic names. We assign semantic names to computations to match requests with named Kubernetes (K8s) service endpoints. We show that this approach provides multiple benefits. First, it allows placement of computational jobs to be independent of location, enabling any cluster with sufficient resources to execute the computation. Second, it facilitates dynamic compute placement without requiring prior knowledge of cluster locations or predefined configurations.


#### cs.DC, cs.NI

### [A Survey on Heterogeneous Computing Using SmartNICs and Emerging Data Processing Units](https://arxiv.org/abs/2504.03653)
**作者**：Nathan Tibbetts, Sifat Ibtisum, Satish Puri

The emergence of new, off-path smart network cards (SmartNICs), known generally as Data Processing Units (DPU), has opened a wide range of research opportunities. Of particular interest is the use of these and related devices in tandem with their host's CPU, creating a heterogeneous computing system with new properties and strengths to be explored, capable of accelerating a wide variety of workloads. This survey begins by providing the motivation and relevant background information for this new field, including its origins, a few current hardware offerings, major programming languages and frameworks for using them, and associated challenges. We then review and categorize a number of recent works in the field, covering a wide variety of studies, benchmarks, and application areas, such as data center infrastructure, commercial uses, and AI and ML acceleration. We conclude with a few observations.


#### cs.DC, cs.NI

### [Distributed $(\Delta+1)$-Coloring in Graphs of Bounded Neighborhood Independence](https://arxiv.org/abs/2510.21549)
**作者**：Marc Fuchs, Fabian Kuhn

The distributed coloring problem is arguably one of the key problems studied in the area of distributed graph algorithms. The most standard variant of the problem asks for a proper vertex coloring of a graph with $\Delta+1$ colors, where $\Delta$ is the maximum degree of the graph. Despite an immense amount of work on distributed coloring problems in the distributed setting, determining the deterministic complexity of $(\Delta+1)$-coloring in the standard message passing model remains one of the most important open questions of the area. In this paper, we aim to improve our understanding of the deterministic complexity of $(\Delta+1)$-coloring as a function of $\Delta$ in a special family of graphs for which significantly faster algorithms are already known. The neighborhood independence $\theta$ of a graph is the maximum number of pairwise non-adjacent neighbors of some node of the graph. In general, in graphs of neighborhood independence $\theta=O(1)$ (e.g., line graphs), it is known that $(\Delta+1)$-coloring can be solved in $2^{O(\sqrt{\log\Delta})}+O(\log^* n)$ rounds. In the present paper, we significantly improve this result, and we show that in graphs of neighborhood independence $\theta$, a $(\Delta+1)$-coloring can be computed in $(\theta\cdot\log\Delta)^{O(\log\log\Delta / \log\log\log\Delta)}+O(\log^* n)$ rounds and thus in quasipolylogarithmic time in $\Delta$ as long as $\theta$ is at most polylogarithmic in $\Delta$. We also show that the known approach that leads to a polylogarithmic in $\Delta$ algorithm for $(2\Delta-1)$-edge coloring already fails for edge colorings of hypergraphs of rank at least $3$.


#### cs.DC, cs.CC

### [JSTprove: Pioneering Verifiable AI for a Trustless Future](https://arxiv.org/abs/2510.21024)
**作者**：Jonathan Gold, Tristan Freiberg, Haruna Isah, Shirin Shahabi

The integration of machine learning (ML) systems into critical industries such as healthcare, finance, and cybersecurity has transformed decision-making processes, but it also brings new challenges around trust, security, and accountability. As AI systems become more ubiquitous, ensuring the transparency and correctness of AI-driven decisions is crucial, especially when they have direct consequences on privacy, security, or fairness. Verifiable AI, powered by Zero-Knowledge Machine Learning (zkML), offers a robust solution to these challenges. zkML enables the verification of AI model inferences without exposing sensitive data, providing an essential layer of trust and privacy. However, traditional zkML systems typically require deep cryptographic expertise, placing them beyond the reach of most ML engineers. In this paper, we introduce JSTprove, a specialized zkML toolkit, built on Polyhedra Network's Expander backend, to enable AI developers and ML engineers to generate and verify proofs of AI inference. JSTprove provides an end-to-end verifiable AI inference pipeline that hides cryptographic complexity behind a simple command-line interface while exposing auditable artifacts for reproducibility. We present the design, innovations, and real-world use cases of JSTprove as well as our blueprints and tooling to encourage community review and extension. JSTprove therefore serves both as a usable zkML product for current engineering needs and as a reproducible foundation for future research and production deployments of verifiable AI.


#### cs.CR, cs.AI, cs.DC, cs.LG

### [Sensing and Storing Less: A MARL-based Solution for Energy Saving in Edge Internet of Things](https://arxiv.org/abs/2510.21103)
**作者**：Zongyang Yuan, Lailong Luo, Qianzhen Zhang, Bangbang Ren, Deke Guo, Richard T. B. Ma

As the number of Internet of Things (IoT) devices continuously grows and application scenarios constantly enrich, the volume of sensor data experiences an explosive increase. However, substantial data demands considerable energy during computation and transmission. Redundant deployment or mobile assistance is essential to cover the target area reliably with fault-prone sensors. Consequently, the ``butterfly effect" may appear during the IoT operation, since unreasonable data overlap could result in many duplicate data. To this end, we propose Senses, a novel online energy saving solution for edge IoT networks, with the insight of sensing and storing less at the network edge by adopting Muti-Agent Reinforcement Learning (MARL). Senses achieves data de-duplication by dynamically adjusting sensor coverage at the sensor level. For exceptional cases where sensor coverage cannot be altered, Senses conducts data partitioning and eliminates redundant data at the controller level. Furthermore, at the global level, considering the heterogeneity of IoT devices, Senses balances the operational duration among the devices to prolong the overall operational duration of edge IoT networks. We evaluate the performance of Senses through testbed experiments and simulations. The results show that Senses saves 11.37% of energy consumption on control devices and prolongs 20% overall operational duration of the IoT device network.


#### cs.NI, cs.DC

### [Benchmarking Catastrophic Forgetting Mitigation Methods in Federated Time Series Forecasting](https://arxiv.org/abs/2510.21491)
**作者**：Khaled Hallak, Oudom Kem

Catastrophic forgetting (CF) poses a persistent challenge in continual learning (CL), especially within federated learning (FL) environments characterized by non-i.i.d. time series data. While existing research has largely focused on classification tasks in vision domains, the regression-based forecasting setting prevalent in IoT and edge applications remains underexplored. In this paper, we present the first benchmarking framework tailored to investigate CF in federated continual time series forecasting. Using the Beijing Multi-site Air Quality dataset across 12 decentralized clients, we systematically evaluate several CF mitigation strategies, including Replay, Elastic Weight Consolidation, Learning without Forgetting, and Synaptic Intelligence. Key contributions include: (i) introducing a new benchmark for CF in time series FL, (ii) conducting a comprehensive comparative analysis of state-of-the-art methods, and (iii) releasing a reproducible open-source framework. This work provides essential tools and insights for advancing continual learning in federated time-series forecasting systems.


#### cs.LG, cs.DC, stat.ML

### [FlexLLM: Token-Level Co-Serving of LLM Inference and Finetuning with SLO Guarantees](https://arxiv.org/abs/2402.18789)
**作者**：Gabriele Oliaro, Xupeng Miao, Xinhao Cheng, Vineeth Kada, Mengdi Wu, Ruohan Gao, Yingyi Huang, Remi Delacourt, April Yang, Yingcheng Wang, Colin Unger, Zhihao Jia

Finetuning large language models (LLMs) is essential for task adaptation, yet today's serving stacks isolate inference and finetuning on separate GPU clusters -- wasting resources and under-utilizing hardware. We introduce FlexLLM, the first system to co-serve LLM inference and PEFT-based finetuning on shared GPUs by fusing computation at the token level. FlexLLM's static compilation optimizations -- dependent parallelization and graph pruning significantly shrink activation memory, leading to end-to-end GPU memory savings by up to 80%. At runtime, a novel token-level finetuning mechanism paired with a hybrid token scheduler dynamically interleaves inference and training tokens within each co-serving iteration, meeting strict latency SLOs while maximizing utilization. In end-to-end benchmarks on LLaMA-3.1-8B, Qwen-2.5-14B, and Qwen-2.5-32B, FlexLLM maintains inference SLO compliance at up to 20 req/s, and improves finetuning throughput by $1.9-4.8\times$ under heavy inference workloads and $2.5-6.8\times$ under light loads, preserving over 76% of peak finetuning progress even at peak demand. FlexLLM is publicly available at https://flexllm.github.io.


#### cs.DC, cs.CL, cs.LG

### [Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models](https://arxiv.org/abs/2407.04656)
**作者**：Yongji Wu, Wenjie Qu, Xueshen Liu, Tianyang Tao, Yifan Qiao, Zhuang Wang, Wei Bai, Yuan Tian, Jiaheng Zhang, Z. Morley Mao, Matthew Lentz, Danyang Zhuo, Ion Stoica

Sparsely-activated Mixture-of-Experts (MoE) architecture has increasingly been adopted to further scale large language models (LLMs). However, frequent failures still pose significant challenges as training scales. The cost of even a single failure is significant, as all GPUs need to idle wait until the failure is resolved, potentially losing considerable training progress as training has to restart from checkpoints. This problem is exacerbated by the growing use of spot instances on public clouds for model training, which despite offering substantial cost savings, introduce frequent preemptions-essentially failures that regularly occur throughout the training process. Existing solutions for efficient fault-tolerant training either lack elasticity or rely on building resiliency into pipeline parallelism, which cannot be applied to MoE models due to the expert parallelism strategy adopted by the MoE architecture.  We present Lazarus, a system for resilient and elastic training of MoE models. Lazarus adaptively allocates expert replicas to address the inherent imbalance in expert workload and speeds up training, while a provably optimal expert placement algorithm is developed to maximize the probability of recovery upon failures. Through adaptive expert placement and a flexible token dispatcher, Lazarus can also fully utilize all available nodes after failures, leaving no GPU idle. Our evaluation shows that Lazarus outperforms existing MoE training systems by up to 5.7x under frequent node failures and 3.4x on a real spot instance trace.


#### cs.DC, cs.LG

### [Domain Adaptation-based Edge Computing for Cross-Conditions Fault Diagnosis](https://arxiv.org/abs/2411.10340)
**作者**：Yanzhi Wang, Jinhong Wu, Chu Wang, Qi Zhou, Tingli Xie

Fault diagnosis of mechanical equipment provides robust support for industrial production. It is worth noting that, the operation of mechanical equipment is accompanied by changes in factors such as speed and load, leading to significant differences in data distribution, which pose challenges for fault diagnosis. Additionally, in terms of application deployment, commonly used cloud-based fault diagnosis methods often encounter issues such as time delays and data security concerns, while common fault diagnosis methods cannot be directly applied to edge computing devices. Therefore, conducting fault diagnosis under cross-operating conditions based on edge computing holds significant research value. This paper proposes a domain-adaptation-based lightweight fault diagnosis framework tailored for edge computing scenarios. Incorporating the local maximum mean discrepancy into knowledge transfer aligns the feature distributions of different domains in a high-dimensional feature space, to discover a common feature space across domains. The acquired fault diagnosis expertise from the cloud-based deep neural network model is transferred to the lightweight edge-based model (edge model) using adaptation knowledge transfer methods. It aims to achieve accurate fault diagnosis under cross-working conditions while ensuring real-time diagnosis capabilities. We utilized the NVIDIA Jetson Xavier NX kit as the edge computing platform and conducted validation experiments on two devices. In terms of diagnostic performance, the proposed method significantly improved diagnostic accuracy, with average increases of 34.44% and 17.33% compared to existing methods, respectively.


#### cs.DC, cs.AI, cs.SE
