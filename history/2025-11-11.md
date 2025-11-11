# ArXiv 新论文更新（2025-11-11)

### [Efficient Deployment of CNN Models on Multiple In-Memory Computing Units](https://arxiv.org/abs/2511.04682)
**作者**：Eleni Bougioukou, Theodore Antonakopoulos

In-Memory Computing (IMC) represents a paradigm shift in deep learning acceleration by mitigating data movement bottlenecks and leveraging the inherent parallelism of memory-based computations. The efficient deployment of Convolutional Neural Networks (CNNs) on IMC-based hardware necessitates the use of advanced task allocation strategies for achieving maximum computational efficiency. In this work, we exploit an IMC Emulator (IMCE) with multiple Processing Units (PUs) for investigating how the deployment of a CNN model in a multi-processing system affects its performance, in terms of processing rate and latency. For that purpose, we introduce the Load-Balance-Longest-Path (LBLP) algorithm, that dynamically assigns all CNN nodes to the available IMCE PUs, for maximizing the processing rate and minimizing latency due to efficient resources utilization. We are benchmarking LBLP against other alternative scheduling strategies for a number of CNN models and experimental results demonstrate the effectiveness of the proposed algorithm.


#### cs.AR, cs.AI

### [RAS: A Bit-Exact rANS Accelerator For High-Performance Neural Lossless Compression](https://arxiv.org/abs/2511.04684)
**作者**：Yuchao Qin, Anjunyi Fan, Bonan Yan

Data centers handle vast volumes of data that require efficient lossless compression, yet emerging probabilistic models based methods are often computationally slow. To address this, we introduce RAS, the Range Asymmetric Numeral System Acceleration System, a hardware architecture that integrates the rANS algorithm into a lossless compression pipeline and eliminates key bottlenecks. RAS couples an rANS core with a probabilistic generator, storing distributions in BF16 format and converting them once into a fixed-point domain shared by a unified division/modulo datapath. A two-stage rANS update with byte-level re-normalization reduces logic cost and memory traffic, while a prediction-guided decoding path speculatively narrows the cumulative distribution function (CDF) search window and safely falls back to maintain bit-exactness. A multi-lane organization scales throughput and enables fine-grained clock gating for efficient scheduling. On image workloads, our RTL-simulated prototype achieves 121.2x encode and 70.9x decode speedups over a Python rANS baseline, reducing average decoder binary-search steps from 7.00 to 3.15 (approximately 55% fewer). When paired with neural probability models, RAS sustains higher compression ratios than classical codecs and outperforms CPU/GPU rANS implementations, offering a practical approach to fast neural lossless compression.


#### cs.AR

### [Eliminating the Hidden Cost of Zone Management in ZNS SSDs](https://arxiv.org/abs/2511.04687)
**作者**：Teona Bagashvili, Tarikul Islam Papon, Subhadeep Sarkar, Manos Athanassoulis

Zoned Namespace (ZNS) SSDs offer a promising interface for stable throughput and low-latency storage by eliminating device-side garbage collection. They expose storage as append-only zones that give the host applications direct control over data placement. However, current ZNS implementations suffer from (a) device-level write amplification (DLWA), (b) increased wear, and (c) interference with host I/O due to zone mapping and management. We identify two primary design decisions as the main cause: (i) fixed physical zones and (ii) full-zone operations that lead to excessive physical writes. We propose SilentZNS, a new zone mapping and management approach that addresses the aforementioned limitations by on-the-fly allocating available resources to zones, while minimizing wear, maintaining parallelism, and avoiding unnecessary writes at the device-level. SilentZNS is a flexible zone allocation scheme that departs from the traditional logical-to-physical zone mapping and allows for arbitrary collections of blocks to be assigned to a zone. We add the necessary constraints to ensure wear-leveling and state-of-the-art read performance, and use only the required blocks to avoid dummy writes during zone reset. We implement SilentZNS using the state-of-the-art ConfZNS++ emulator and show that it eliminates the undue burden of dummy writes by up to 20x, leading to lower DLWA (86% less at 10% zone occupancy), less overall wear (up to 76.9%), and up to 3.7x faster workload execution.


#### cs.AR

### [MultiVic: A Time-Predictable RISC-V Multi-Core Processor Optimized for Neural Network Inference](https://arxiv.org/abs/2511.05321)
**作者**：Maximilian Kirschner, Konstantin Dudzik, Ben Krusekamp, J\"urgen Becker

Real-time systems, particularly those used in domains like automated driving, are increasingly adopting neural networks. From this trend arises the need for high-performance hardware exhibiting predictable timing behavior. While state-of-the-art real-time hardware often suffers from limited memory and compute resources, modern AI accelerators typically lack the crucial predictability due to memory interference.  We present a new hardware architecture to bridge this gap between performance and predictability. The architecture features a multi-core vector processor with predictable cores, each equipped with local scratchpad memories. A central management core orchestrates access to shared external memory following a statically determined schedule.  To evaluate the proposed hardware architecture, we analyze different variants of our parameterized design. We compare these variants to a baseline architecture consisting of a single-core vector processor with large vector registers. We find that configurations with a larger number of smaller cores achieve better performance due to increased effective memory bandwidth and higher clock frequencies. Crucially for real-time systems, execution time fluctuation remains very low, demonstrating the platform's time predictability.


#### cs.AR

### [MACO: A Multi-Agent LLM-Based Hardware/Software Co-Design Framework for CGRAs](https://arxiv.org/abs/2509.13557)
**作者**：Zesong Jiang, Yuqi Sun, Qing Zhong, Mahathi Krishna, Deepak Patil, Cheng Tan, Sriram Krishnamoorthy, Jeff Zhang

Coarse-grained Reconfigurable Arrays (CGRAs) are a promising computing architecture that can deliver high-performance, energy-efficient acceleration across diverse domains. By supporting reconfiguration at the functional unit level, CGRAs efficiently adapt to varying computational patterns and optimize resource utilization. However, designing CGRAs is highly challenging due to the vast design space, independent architectural parameters, and the time-consuming nature of manual design. Fortunately, the rapid advancement of large language models (LLMs) presents new opportunities to automate this process.  In this work, we propose MACO-- an open-source multi-agent LLM-based framework for Hardware/Software (HW/SW) co-design of CGRAs. The framework employs LLM reasoning to generate CGRAs across four stages: HW/SW co-design, Design error correction, Best design selection, and Evaluation & Feedback. Furthermore, MACO iteratively optimizes the generated CGRAs, leveraging agent reasoning and feedback to achieve higher PPA (that is, power, performance, and area) design points for a given domain. In addition, we introduce an LLM self-learning mechanism that employs LLM-driven decision making to select the optimal CGRA to accelerate the design process.  We evaluate the framework with state-of-the-art LLM-based methods and manual CGRA design, in terms of performance, power consumption, and area. Experimental results show that MACO efficiently generates high-quality CGRA architectures, significantly reducing manual design effort and demonstrating the potential of our framework for real-world CGRA design.


#### cs.AR

### [SMART-WRITE: Adaptive Learning-based Write Energy Optimization for Phase Change Memory](https://arxiv.org/abs/2511.04713)
**作者**：Mahek Desai, Rowena Quinn, Marjan Asadinia

As dynamic random access memory (DRAM) and other current transistor-based memories approach their scalability limits, the search for alternative storage methods becomes increasingly urgent. Phase-change memory (PCM) emerges as a promising candidate due to its scalability, fast access time, and zero leakage power compared to many existing memory technologies. However, PCM has significant drawbacks that currently hinder its viability as a replacement. PCM cells suffer from a limited lifespan because write operations degrade the physical material, and these operations consume a considerable amount of energy. For PCM to be a practical option for data storage-which involves frequent write operations-its cell endurance must be enhanced, and write energy must be reduced. In this paper, we propose SMART-WRITE, a method that integrates neural networks (NN) and reinforcement learning (RL) to dynamically optimize write energy and improve performance. The NN model monitors real-time operating conditions and device characteristics to determine optimal write parameters, while the RL model dynamically adjusts these parameters to further optimize PCM's energy consumption. By continuously adjusting PCM write parameters based on real-time system conditions, SMART-WRITE reduces write energy consumption by up to 63% and improves performance by up to 51% compared to the baseline and previous models.


#### cs.AR, cs.ET

### [MDM: Manhattan Distance Mapping of DNN Weights for Parasitic-Resistance-Resilient Memristive Crossbars](https://arxiv.org/abs/2511.04798)
**作者**：Matheus Farias, Wanghley Martins, H. T. Kung

Manhattan Distance Mapping (MDM) is a post-training deep neural network (DNN) weight mapping technique for memristive bit-sliced compute-in-memory (CIM) crossbars that reduces parasitic resistance (PR) nonidealities.  PR limits crossbar efficiency by mapping DNN matrices into small crossbar tiles, reducing CIM-based speedup. Each crossbar executes one tile, requiring digital synchronization before the next layer. At this granularity, designers either deploy many small crossbars in parallel or reuse a few sequentially-both increasing analog-to-digital conversions, latency, I/O pressure, and chip area.  MDM alleviates PR effects by optimizing active-memristor placement. Exploiting bit-level structured sparsity, it feeds activations from the denser low-order side and reorders rows according to the Manhattan distance, relocating active cells toward regions less affected by PR and thus lowering the nonideality factor (NF).  Applied to DNN models on ImageNet-1k, MDM reduces NF by up to 46% and improves accuracy under analog distortion by an average of 3.6% in ResNets. Overall, it provides a lightweight, spatially informed method for scaling CIM DNN accelerators.


#### cs.AR, cs.AI, cs.ET, cs.LG

### [FuseFlow: A Fusion-Centric Compilation Framework for Sparse Deep Learning on Streaming Dataflow](https://arxiv.org/abs/2511.04768)
**作者**：Rubens Lacouture, Nathan Zhang, Ritvik Sharma, Marco Siracusa, Fredrik Kjolstad, Kunle Olukotun, Olivia Hsu

As deep learning models scale, sparse computation and specialized dataflow hardware have emerged as powerful solutions to address efficiency. We propose FuseFlow, a compiler that converts sparse machine learning models written in PyTorch to fused sparse dataflow graphs for reconfigurable dataflow architectures (RDAs). FuseFlow is the first compiler to support general cross-expression fusion of sparse operations. In addition to fusion across kernels (expressions), FuseFlow also supports optimizations like parallelization, dataflow ordering, and sparsity blocking. It targets a cycle-accurate dataflow simulator for microarchitectural analysis of fusion strategies. We use FuseFlow for design-space exploration across four real-world machine learning applications with sparsity, showing that full fusion (entire cross-expression fusion across all computation in an end-to-end model) is not always optimal for sparse models-fusion granularity depends on the model itself. FuseFlow also provides a heuristic to identify and prune suboptimal configurations. Using Fuseflow, we achieve performance improvements, including a ~2.7x speedup over an unfused baseline for GPT-3 with BigBird block-sparse attention.


#### cs.LG, cs.AR, cs.PL

### [SLOFetch: Compressed-Hierarchical Instruction Prefetching for Cloud Microservices](https://arxiv.org/abs/2511.04774)
**作者**：Liu Jiang, Zerui Bao, Shiqi Sheng, Di Zhu

Large-scale networked services rely on deep soft-ware stacks and microservice orchestration, which increase instruction footprints and create frontend stalls that inflate tail latency and energy. We revisit instruction prefetching for these cloud workloads and present a design that aligns with SLO driven and self optimizing systems. Building on the Entangling Instruction Prefetcher (EIP), we introduce a Compressed Entry that captures up to eight destinations around a base using 36 bits by exploiting spatial clustering, and a Hierarchical Metadata Storage scheme that keeps only L1 resident and frequently queried entries on chip while virtualizing bulk metadata into lower levels. We further add a lightweight Online ML Controller that scores prefetch profitability using context features and a bandit adjusted threshold. On data center applications, our approach preserves EIP like speedups with smaller on chip state and improves efficiency for networked services in the ML era.


#### cs.LG, cs.AR

### [PhantomFetch: Obfuscating Loads against Prefetcher Side-Channel Attacks](https://arxiv.org/abs/2511.05110)
**作者**：Xingzhi Zhang, Buyi Lv, Yimin Lu, Kai Bu

The IP-stride prefetcher has recently been exploited to leak secrets through side-channel attacks. It, however, cannot be simply disabled for security with prefetching speedup as a sacrifice. The state-of-the-art defense tries to retain the prefetching effect by hardware modification. In this paper, we present PhantomFetch as the first prefetching-retentive and hardware-agnostic defense. It avoids potential remanufacturing cost and enriches applicability to off-the-shelf devices. The key idea is to directly break the exploitable coupling between trained prefetcher entries and the victim's secret-dependent loads by obfuscating the sensitive load effects of the victim. The experiment results show that PhantomFetch can secure the IP-stride prefetcher with only negligible overhead.


#### cs.CR, cs.AR

### [Improving Injection-Throttling Mechanisms for Congestion Control for Data-center and Supercomputer Interconnects](https://arxiv.org/abs/2511.05149)
**作者**：Cristina Olmedilla, Jesus Escudero-Sahuquillo, Pedro J. Garcia, Francisco J. Quiles, Jose Duato

Over the past decade, Supercomputers and Data centers have evolved dramatically to cope with the increasing performance requirements of applications and services, such as scientific computing, generative AI, social networks or cloud services. This evolution have led these systems to incorporate high-speed networks using faster links, end nodes using multiple and dedicated accelerators, or a advancements in memory technologies to bridge the memory bottleneck. The interconnection network is a key element in these systems and it must be thoroughly designed so it is not the bottleneck of the entire system, bearing in mind the countless communication operations that generate current applications and services. Congestion is serious threat that spoils the interconnection network performance, and its effects are even more dramatic when looking at the traffic dynamics and bottlenecks generated by the communication operations mentioned above. In this vein, numerous congestion control (CC) techniques have been developed to address congestion negative effects. One popular example is Data Center Quantized Congestion Notification (DCQCN), which allows congestion detection at network switch buffers, then marking congesting packets and notifying about congestion to the sources, which finally apply injection throttling of those packets contributing to congestion. While DCQCN has been widely studied and improved, its main principles for congestion detection, notification and reaction remain largely unchanged, which is an important shortcoming considering congestion dynamics in current high-performance interconnection networks. In this paper, we revisit the DCQCN closed-loop mechanism and refine its design to leverage a more accurate congestion detection, signaling, and injection throttling, reducing control traffic overhead and avoiding unnecessary throttling of non-congesting flows.


#### cs.NI, cs.AR

### [NeuroFlex: Column-Exact ANN-SNN Co-Execution Accelerator with Cost-Guided Scheduling](https://arxiv.org/abs/2511.05215)
**作者**：Varun Manjunath, Pranav Ramesh, Gopalakrishnan Srinivasan

NeuroFlex is a column-level accelerator that co-executes artificial and spiking neural networks to minimize energy-delay product on sparse edge workloads with competitive accuracy. The design extends integer-exact QCFS ANN-SNN conversion from layers to independent columns. It unifies INT8 storage with on-the-fly spike generation using an offline cost model to assign columns to ANN or SNN cores and pack work across processing elements with deterministic runtime. Our cost-guided scheduling algorithm improves throughput by 16-19% over random mapping and lowers EDP by 57-67% versus a strong ANN-only baseline across VGG-16, ResNet-34, GoogLeNet, and BERT models. NeuroFlex also delivers up to 2.5x speedup over LoAS and 2.51x energy reduction over SparTen. These results indicate that fine-grained and integer-exact hybridization outperforms single-mode designs on energy and latency without sacrificing accuracy.


#### cs.NE, cs.AR

### [Marionette: Data Structure Description and Management for Heterogeneous Computing](https://arxiv.org/abs/2511.04853)
**作者**：Nuno dos Santos Fernandes, Pedro Tom\'as, Nuno Roma, Frank Winklmeier, Patricia Conde-Mu\'i\~no

Adapting large, object-oriented C++ codebases for hardware acceleration might be extremely challenging, particularly when targeting heterogeneous platforms such as GPUs. Marionette is a C++17 library designed to address this by enabling flexible, efficient, and portable data structure definitions. It decouples data layout from the description of the interface, supports multiple memory management strategies, and provides efficient data transfers and conversions across devices, all of this with minimal runtime overhead due to the compile-time nature of its abstractions. By allowing interfaces to be augmented with arbitrary functions, Marionette maintains compatibility with existing code and offers a streamlined interface that supports both straightforward and advanced use cases. This paper outlines its design, usage, and performance, including a CUDA-based case study demonstrating its efficiency and flexibility.


#### cs.DC

### [GPU Under Pressure: Estimating Application's Stress via Telemetry and Performance Counters](https://arxiv.org/abs/2511.05067)
**作者**：Giuseppe Esposito, Juan-David Guerrero-Balaguera, Josie Esteban Rodriguez Condia, Matteo Sonza Reorda, Marco Barbiero, Rossella Fortuna

Graphics Processing Units (GPUs) are specialized accelerators in data centers and high-performance computing (HPC) systems, enabling the fast execution of compute-intensive applications, such as Convolutional Neural Networks (CNNs). However, sustained workloads can impose significant stress on GPU components, raising reliability concerns due to potential faults that corrupt the intermediate application computations, leading to incorrect results. Estimating the stress induced by an application is thus crucial to predict reliability (with\,special\,emphasis\,on\,aging\,effects). In this work, we combine online telemetry parameters and hardware performance counters to assess GPU stress induced by different applications. The experimental results indicate the stress induced by a parallel workload can be estimated by combining telemetry data and Performance Counters that reveal the efficiency in the resource usage of the target workload. For this purpose the selected performance counters focus on measuring the i) throughput, ii) amount of issued instructions and iii) stall events.


#### cs.DC

### [Almost Time-Optimal Loosely-Stabilizing Leader Election on Arbitrary Graphs Without Identifiers in Population Protocols](https://arxiv.org/abs/2411.03902)
**作者**：Haruki Kanaya, Ryota Eguchi, Taisho Sasada, Michiko Inoue

The population protocol model is a computational model for passive mobile agents. We address the leader election problem, which determines a unique leader on arbitrary communication graphs starting from any configuration. Unfortunately, self-stabilizing leader election is impossible to be solved without knowing the exact number of agents; thus, we consider loosely-stabilizing leader election, which converges to safe configurations in a relatively short time, and holds the specification (maintains a unique leader) for a relatively long time. When agents have unique identifiers, Sudo et al.(2019) proposed a protocol that, given an upper bound $N$ for the number of agents $n$, converges in $O(mN\log n)$ expected steps, where $m$ is the number of edges. When unique identifiers are not required, they also proposed a protocol that, using random numbers and given $N$, converges in $O(mN^2\log{N})$ expected steps. Both protocols have a holding time of $\Omega(e^{2N})$ expected steps and use $O(\log{N})$ bits of memory. They also showed that the lower bound of the convergence time is $\Omega(mN)$ expected steps for protocols with a holding time of $\Omega(e^N)$ expected steps given $N$.  In this paper, we propose protocols that do not require unique identifiers. These protocols achieve convergence times close to the lower bound with increasing memory usage. Specifically, given $N$ and an upper bound $\Delta$ for the maximum degree, we propose two protocols whose convergence times are $O(mN\log n)$ and $O(mN\log N)$ both in expectation and with high probability. The former protocol uses random numbers, while the latter does not require them. Both protocols utilize $O(\Delta \log N)$ bits of memory and hold the specification for $\Omega(e^{2N})$ expected steps.


#### cs.DC

### [OptiLog: Assigning Roles in Byzantine Consensus](https://arxiv.org/abs/2502.15428)
**作者**：Hanish Gogada, Christian Berger, Leander Jehl, Hans P. Reiser, Hein Meling

Byzantine Fault-Tolerant (BFT) protocols play an important role in blockchains. As the deployment of such systems extends to wide-area networks, the scalability of BFT protocols becomes a critical concern. Optimizations that assign specific roles to individual replicas can significantly improve the performance of BFT systems. However, such role assignment is highly sensitive to faults, potentially undermining the optimizations' effectiveness. To address these challenges, we present OptiLog, a logging framework for collecting and analyzing measurements that help to assign roles in globally distributed systems, despite the presence of faults. OptiLog presents local measurements in global data structures, to enable consistent decisions and hold replicas accountable if they do not perform according to their reported measurements. We demonstrate OptiLog's flexibility by applying it to two BFT protocols: (1) Aware, a highly optimized PBFT-like protocol, and (2) Kauri, a tree-based protocol designed for large-scale deployments. OptiLog detects and excludes replicas that misbehave during consensus and thus enables the system to operate in an optimized, low-latency configuration, even under adverse conditions. Experiments show that for tree overlays deployed across 73 worldwide cities, trees found by OptiLog display 39% lower latency than Kauri.


#### cs.DC

### [SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference](https://arxiv.org/abs/2505.24095)
**作者**：Tian Xia, Ziming Mao, Jamison Kerney, Ethan J. Jackson, Zhifei Li, Jiarong Xing, Scott Shenker, Ion Stoica

Serving Large Language Models (LLMs) efficiently in multi-region setups remains a challenge. Due to cost and GPU availability concerns, providers typically deploy LLMs in multiple regions using instance with long-term commitments, like reserved instances or on-premise clusters, which are often underutilized due to their region-local traffic handling and diurnal traffic variance. In this paper, we introduce SkyWalker, a multi-region load balancer for LLM inference that aggregates regional diurnal patterns through cross-region traffic handling. By doing so, SkyWalker enables providers to reserve instances based on expected global demand, rather than peak demand in each individual region. Meanwhile, SkyWalker preserves KV-Cache locality and load balancing, ensuring cost efficiency without sacrificing performance. SkyWalker achieves this with a cache-aware cross-region traffic handler and a selective pushing based load balancing mechanism. Our evaluation on real-world workloads shows that it achieves 1.12-2.06x higher throughput and 1.74-6.30x lower latency compared to existing load balancers, while reducing total serving cost by 25%.


#### cs.DC

### [Accelerating HDC-CNN Hybrid Models Using Custom Instructions on RISC-V GPUs](https://arxiv.org/abs/2511.05053)
**作者**：Wakuto Matsumi, Riaz-Ul-Haque Mian

Machine learning based on neural networks has advanced rapidly, but the high energy consumption required for training and inference remains a major challenge. Hyperdimensional Computing (HDC) offers a lightweight, brain-inspired alternative that enables high parallelism but often suffers from lower accuracy on complex visual tasks. To overcome this, hybrid accelerators combining HDC and Convolutional Neural Networks (CNNs) have been proposed, though their adoption is limited by poor generalizability and programmability. The rise of open-source RISC-V architectures has created new opportunities for domain-specific GPU design. Unlike traditional proprietary GPUs, emerging RISC-V-based GPUs provide flexible, programmable platforms suitable for custom computation models such as HDC. In this study, we design and implement custom GPU instructions optimized for HDC operations, enabling efficient processing for hybrid HDC-CNN workloads. Experimental results using four types of custom HDC instructions show a performance improvement of up to 56.2 times in microbenchmark tests, demonstrating the potential of RISC-V GPUs for energy-efficient, high-performance computing.


#### cs.DC, cs.AI, cs.GR

### [The Future of Fully Homomorphic Encryption System: from a Storage I/O Perspective](https://arxiv.org/abs/2511.04946)
**作者**：Lei Chen, Erci Xu, Yiming Sun, Shengyu Fan, Xianglong Deng, Guiming Shi, Guang Fan, Liang Kong, Yilan Zhu, Shoumeng Yan, Mingzhe Zhang

Fully Homomorphic Encryption (FHE) allows computations to be performed on encrypted data, significantly enhancing user privacy. However, the I/O challenges associated with deploying FHE applications remains understudied. We analyze the impact of storage I/O on the performance of FHE applications and summarize key lessons from the status quo. Key results include that storage I/O can degrade the performance of ASICs by as much as 357$\times$ and reduce GPUs performance by up to 22$\times$.


#### cs.CR, cs.DC

### [CUNQA: a Distributed Quantum Computing emulator for HPC](https://arxiv.org/abs/2511.05209)
**作者**：Jorge V\'azquez-P\'erez, Daniel Exp\'osito-Pati\~no, Marta Losada, \'Alvaro Carballido, Andr\'es G\'omez, Tom\'as F. Pena

The challenge of scaling quantum computers to gain computational power is expected to lead to architectures with multiple connected quantum processing units (QPUs), commonly referred to as Distributed Quantum Computing (DQC). In parallel, there is a growing momentum toward treating quantum computers as accelerators, integrating them into the heterogeneous architectures of high-performance computing (HPC) environments. This work combines these two foreseeable futures in CUNQA, an open-source DQC emulator designed for HPC environments that allows testing, evaluating and studying DQC in HPC before it even becomes real. It implements the three DQC models of no-communication, classical-communication and quantum-communication; which will be examined in this work. Addressing programming considerations, explaining emulation and simulation details, and delving into the specifics of the implementation will be part of the effort. The well-known Quantum Phase Estimation (QPE) algorithm is used to demonstrate and analyze the emulation of the models. To the best of our knowledge, CUNQA is the first tool designed to emulate the three DQC schemes in an HPC environment.


#### quant-ph, cs.DC

### [LLM4FaaS: No-Code Application Development using LLMs and FaaS](https://arxiv.org/abs/2502.14450)
**作者**：Minghe Wang, Tobias Pfandzelter, Trever Schirmer, David Bermbach

Large language models (LLMs) show great capabilities in generating code from natural language descriptions, bringing programming power closer to non-technical users. However, their lack of expertise in operating the generated code remains a key barrier to realizing customized applications. Function-as-a-Service (FaaS) platforms offer a high level of abstraction for code execution and deployment, allowing users to run LLM-generated code without requiring technical expertise or incurring operational overhead.  In this paper, we present LLM4FaaS, a no-code application development approach that integrates LLMs and FaaS platforms to enable non-technical users to build and run customized applications using only natural language. By deploying LLM-generated code through FaaS, LLM4FaaS abstracts away infrastructure management and boilerplate code generation. We implement a proof-of-concept prototype based on an open-source FaaS platform, and evaluate it using real prompts from non-technical users. Experiments with GPT-4o show that LLM4FaaS can automatically build and deploy code in 71.47% of cases, outperforming a non-FaaS baseline at 43.48% and an existing LLM-based platform at 14.55%, narrowing the gap to human performance at 88.99%. Further analysis of code quality, programming language diversity, latency, and consistency demonstrates a balanced performance in terms of efficiency, maintainability and availability.


#### cs.SE, cs.DC
