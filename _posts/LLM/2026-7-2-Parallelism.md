---
layout: post
title: 大模型训练并行策略总结
categories: [LLM, AI-Infra]
tags: [LLM, AI-Infra, Parallelism, 分布式训练]
description: 大模型训练中各种并行策略的原理与适用场景总结，涵盖数据并行、张量并行、流水线并行、序列并行、上下文并行与专家并行
---

> 参考：
> - [ColossalAI - Paradigms of Parallelism](https://colossalai.org/docs/concepts/paradigms_of_parallelism/)
> - [NVIDIA Megatron Bridge - Parallelisms Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)
> - [DeepSpeed - Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
> - [DP、TP、PP、EP 怎么算总卡数？通常模式与 Folding 模式一篇讲清](https://zhuanlan.zhihu.com/p/2053418550879168277)

## 1. 为什么需要并行？

随着模型规模增长（GPT-3: 175B, DeepSeek-V3: 671B），单卡显存和算力远远不够：

| 瓶颈 | 说明 |
|------|------|
| 显存不足 | 模型参数 + 优化器状态 + 激活值 >> 单卡显存 |
| 训练太慢 | 单卡算力有限，训练时间以年计 |
| 序列太长 | 长上下文场景的激活值内存爆炸 |

解决思路：将**数据、模型、序列**拆分到多张 GPU 上并行处理。

---

## 2. 数据并行（Data Parallelism, DP）

### 核心思想

每张 GPU 持有**完整的模型副本**，将数据集按 batch 维度切分，各自独立前向/反向，然后**同步梯度**。

![Data Parallel：每个 GPU 持有完整模型副本，数据集切分到各卡](/img/LLM/2026-7-2-Parallelism/data-parallel.png)

```
GPU 0: model copy + data shard 0  →  grad_0 ─┐
GPU 1: model copy + data shard 1  →  grad_1 ─┼─ All-Reduce → 更新参数
GPU 2: model copy + data shard 2  →  grad_2 ─┘
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 实现简单，线性加速 | 每卡需存储完整模型（冗余大） |
| 通信模式固定（All-Reduce） | 模型太大时单卡放不下 |

### ZeRO 优化（Zero Redundancy Optimizer）

ZeRO 通过分片消除数据并行中的冗余存储：

| 级别 | 分片内容 | 显存节省 |
|------|----------|----------|
| ZeRO-1 | 优化器状态（momentum, variance） | ~4x |
| ZeRO-2 | + 梯度 | ~8x |
| ZeRO-3 | + 模型参数 | ~N 倍（N = GPU 数） |

ZeRO-3 本质上已经在做模型并行，但保持了数据并行的编程接口。

### Distributed Optimizer（Megatron 实现）

Megatron 中的分布式优化器 = ZeRO-1 + reduce-scatter 替代 all-reduce：
- 每卡只存自己负责的参数分片的优化器状态和高精度参数
- 梯度用 reduce-scatter（而非 all-reduce）只聚合自己需要的部分
- 更新后用 all-gather 同步参数

---

## 3. 张量并行（Tensor Parallelism, TP）

### 核心思想

将单个层的**权重矩阵**切分到多张 GPU 上，每卡只计算一部分，再通过通信拼接结果。

![Tensor Parallel：将矩阵 B 按列切分到多卡，各卡计算部分结果后 All-Gather](/img/LLM/2026-7-2-Parallelism/tensor-parallel.png)

以矩阵乘法 \(C = AB\) 为例，将 B 按列切分：

```
        B = [B₀ | B₁]  (按列切分到 2 个 GPU)

GPU 0:  A × B₀ = C₀
GPU 1:  A × B₁ = C₁
                        → All-Gather → C = [C₀ | C₁]
```

### Transformer 中的 TP

对于 Attention 层和 FFN 层，Megatron-LM 设计了精巧的切分方式：

- **MLP**：第一个线性层按列切，第二个按行切 → 只需一次 All-Reduce
- **Attention**：各 head 天然独立，直接分配到不同 GPU

### 特点

| 优点 | 缺点 |
|------|------|
| 减少单卡显存（参数 + 激活） | 通信频繁（每层都要同步） |
| 对模型结构无侵入 | 最适合节点内 NVLink 高带宽场景 |

> **最佳实践**：TP 应限制在**单节点内**（NVLink 带宽高），跨节点用 PP。

---

## 4. 流水线并行（Pipeline Parallelism, PP）

### 核心思想

将模型按**层**切分为多个阶段（stage），每个阶段放在不同 GPU 上，像流水线一样依次处理。

![Pipeline Parallel：模型按层切分到不同 GPU，前向/反向传递激活值](/img/LLM/2026-7-2-Parallelism/pipeline-parallel.png)

```
GPU 0: Layer 0-7    (Stage 0)
GPU 1: Layer 8-15   (Stage 1)
GPU 2: Layer 16-23  (Stage 2)
GPU 3: Layer 24-31  (Stage 3)
```

### 流水线气泡（Pipeline Bubble）

简单的顺序执行会导致大量 GPU 空闲（气泡）。

![GPipe 调度示意图：微批次流水线执行，中间的 Bubble 为浪费的空闲时间](/img/LLM/2026-7-2-Parallelism/pipeline-schedule.png)

解决方案：

1. **Micro-batch 切分**：将一个 batch 切成多个 micro-batch，让流水线"流"起来
2. **GPipe 调度**：先做所有 micro-batch 的前向，再做所有反向
3. **1F1B 调度**：交替执行前向和反向，减少峰值显存
4. **Interleaved Schedule**：每卡持有多个不连续的层块（virtual pipeline），进一步减小气泡

### 特点

| 优点 | 缺点 |
|------|------|
| 通信量小（只传激活值） | 存在流水线气泡 |
| 适合跨节点场景 | 各阶段负载需均衡 |

> **注意：推理场景下 PP 的应用有限。** PP 在推理时仅涉及前向传播，通常只在单卡显存确实无法容纳模型权重时才采用。如果显存充足，TP 或 EP 通常是更优的选择（延迟更低）。

---

## 5. 序列并行（Sequence Parallelism, SP）

### 问题背景

TP 只切分了 Attention 和 FFN 的线性计算，但 **LayerNorm、Dropout** 等非线性操作仍然在每卡上处理完整序列，激活值没有被切分。

### Megatron SP 方案

在 TP 的基础上，对**非线性操作的激活值**沿序列维度切分：

```
TP 区域（Attention/FFN）:  各卡持有部分参数，完整序列
SP 区域（LayerNorm/Dropout）: 各卡持有完整参数，部分序列
```

切换时使用 All-Gather / Reduce-Scatter 通信。

### DeepSpeed-Ulysses 方案

- 将输入沿序列维度切分，每卡只持有 \(N/P\) 长度的序列
- 通过 **All-to-All** 通信在 Attention 计算前重新分配，使每卡获得完整序列但只负责部分 head
- 支持任意 Attention 模式（dense / sparse）

### Ring Attention

- 将 KV 按序列维度切分到各卡
- 计算 Attention 时，以**环形**方式传递 KV 块
- 每卡轮流获得其他卡的 KV 块，逐步完成完整的 Attention 计算
- 理论上支持**无限长**上下文

> **注意**：Megatron SP 必须和 TP 配合使用（`tensor_model_parallel_size > 1`）。

---

## 6. 上下文并行（Context Parallelism, CP）

### 与 SP 的区别

| | Sequence Parallelism | Context Parallelism |
|---|---|---|
| 切分范围 | 仅非线性层的激活 | **所有层**的激活 |
| 依赖条件 | 必须配合 TP | 可独立使用 |
| 适用场景 | 减少 TP 场景的激活内存 | **长上下文训练**（32K+ tokens） |

### 工作原理

- 将整个序列的激活值沿序列维度均分到多卡
- 前向时每卡只存自己分片的 KV pairs
- 反向时通过环形 All-Gather / Reduce-Scatter 重组
- 类似 Ring Attention 的思想，但集成在框架层面

### 适用策略

| 序列长度 | 建议 |
|----------|------|
| < 2K | 标准 TP + PP + DP |
| 2K-8K | 加 SP |
| 8K-32K | 加 CP=2 |
| 32K+ | CP=4-8 |

---

## 7. 专家并行（Expert Parallelism, EP）

### MoE 模型简介

MoE（Mixture of Experts）用多个"专家"替代单一 FFN，每个 token 只激活 Top-K 个专家：

```
Input → Router（门控网络）→ 选择 Top-K experts → 加权求和 → Output
```

- 参数量大（所有专家参数之和），但计算量小（每次只激活少数专家）
- Switch Transformer: 1.6T 参数，计算量 ≈ 10B dense 模型

### Expert Parallelism 原理

将不同的专家分配到不同的 GPU：

```
8 experts, 4 GPUs → 每卡 2 个 expert

Token routing:
  Input tokens → All-to-All → 各卡处理自己的 expert → All-to-All → 合并结果
```

### DeepSpeed MoE 的灵活并行组合

| 模式 | 组合 | 效果 |
|------|------|------|
| E | Expert only | 扩展模型规模 |
| E + D | Expert + Data | 加速训练吞吐 |
| E + Z | Expert + ZeRO | 支持更大的 base model |
| E + D + M | Expert + Data + Model | 支持更大 hidden size |
| E + Z-Off + M | Expert + ZeRO-Offload + Model | 在有限 GPU 上训练超大 MoE |

### 负载均衡

MoE 训练的核心挑战是**专家负载不均衡**（某些专家被频繁选中，其他闲置）：

- **Auxiliary Loss**：辅助损失函数，惩罚路由不均匀
- **Token Dropping**：限制每个专家的容量，超出的 token 被丢弃
- **Random Token Selection**：DeepSpeed 的随机选择技术，缓解偏置选择问题

### Wide-EP（宽专家并行）

Wide-EP 是 EP 的**极限扩展**，将专家分布到大量 GPU（8+，通常 64-72 卡），让每卡只持有极少专家，配合高带宽互联抵消通信开销。

```
普通 EP（EP=4）：256 experts / 4 GPU = 64 experts/GPU → 权重加载压力大
Wide-EP（EP=64）：256 experts / 64 GPU = 4 experts/GPU → 权重少，计算更高效
```

| 解决的问题 | Wide-EP 做法 |
|------|------|
| 单卡 expert 权重太多 | 分散到更多卡，减少 weight-loading 压力 |
| GroupGEMM 效率低 | 更少权重 → 更高算术强度（FLOPs/byte） |
| All-to-All 通信开销大 | 依赖 NVL72 的 130 TB/s NVLink 带宽 |
| 负载不均衡 | EPLB（Expert Parallel Load Balancer）动态重分配热/冷专家 |

> Wide-EP 不是新的并行维度，而是 EP 的规模化实践。前提是硬件互联带宽足够高（如 GB200 NVL72）。NVIDIA 实测：Wide-EP 部署 DeepSeek-R1，单卡吞吐提升 1.8x。

### TP vs EP：为什么 MoE 更适合 EP？

![EP 与 DP 结合的常见场景：Attention 层使用 DP，FFN/MoE 层使用 EP](/img/LLM/2026-7-2-Parallelism/ep-dp-combine.png)

| 对比维度 | TP（张量并行） | EP（专家并行） |
|------|------|------|
| 通信触发位置 | **每一层**都要通信 | 只在 **MoE 层**通信，Attention 层无跨卡通信 |
| 通信模式 | All-Reduce（每层 1-2 次） | All-to-All（只传被路由到的 token） |
| 通信量 | 高（所有 token、每层都参与） | 低（稀疏激活，只传必要数据） |
| 根本原因 | 切分权重矩阵，每步都要同步 | 稀疏激活 → 路由通信 → 只传必要数据 |

这是 MoE 架构天然适合 EP 的根本原因：**稀疏激活意味着通信只发生在必要的地方**。

### Wide-EP 如何解决 Decode 内存带宽瓶颈

Decode 阶段是典型的 **memory-bandwidth-bound**（内存带宽瓶颈）：每生成一个 token 都要把相关权重完整加载一遍，计算量却极小——典型的"搬运比算慢"。

Wide-EP 的解决思路：

```
DeepSeek-R1: 670B 参数，EP=32

单卡：需要加载 670B 权重，HBM 带宽是瓶颈
32 卡 Wide-EP：每卡只持有 670B/32 ≈ 21B 权重

等效带宽 = 单卡 HBM 带宽 × 32（模型只持有一份，但 32 卡并行加载）
```

简单说：**用更多 GPU 的 HBM 带宽之和来突破单卡的内存搞运瓶颈**。

---

## 8. 异构系统并行（Offloading）

利用 CPU 内存甚至 NVMe 磁盘扩展显存：

| 方法 | 思路 |
|------|------|
| ZeRO-Offload | 将优化器状态和梯度 offload 到 CPU |
| ZeRO-Infinity | 进一步 offload 到 NVMe 磁盘 |
| PatrickStar | 基于 chunk 的动态内存管理 |

![异构系统：GPU Memory ↔ CPU Memory ↔ Disk (SSD)](/img/LLM/2026-7-2-Parallelism/heterogeneous.png)

**适用场景**：GPU 数量有限，但模型很大（学术界/个人开发者）。

---

## 9. 各策略对比总结

| 并行策略 | 切分维度 | 通信模式 | 适用场景 |
|----------|----------|----------|----------|
| DP | Batch | All-Reduce | 通用加速 |
| ZeRO | 优化器/梯度/参数 | Reduce-Scatter + All-Gather | 减少冗余显存 |
| TP | 层内权重矩阵 | All-Reduce / All-Gather | 节点内并行（NVLink） |
| PP | 层间 | P2P（点对点） | 跨节点扩展 |
| SP | 序列（非线性层） | All-to-All / Reduce-Scatter | 配合 TP 减少激活 |
| CP | 序列（所有层） | Ring / All-Gather | 长上下文训练 |
| EP | 专家 | All-to-All | MoE 模型 |

---

## 10. 实际部署策略选择

### Dense 模型

| 模型规模 | GPU 数 | 推荐策略 |
|----------|--------|----------|
| < 1B | 1-8 | DP only |
| 1-10B | 8-16 | TP=2-4 + DP |
| 10-70B | 16-64 | TP=4-8 + PP=2-4 + DP |
| 70-175B | 64-256 | TP=8 + PP=4-8 + DP |
| 175-500B | 256-1024 | TP=8 + PP=8-16 + CP=2 + DP |

### MoE 模型

| 总参数/活跃参数 | 推荐策略 |
|----------------|----------|
| < 20B | EP only |
| 20-100B | TP=1-2 + PP=2-4 + EP=8-16 |
| 100-500B | TP=2-4 + PP=8-16 + EP=8-32 |
| 500B+ | TP=2 + PP=16 + EP=32-64 |

### 硬件拓扑考量

- **节点内 NVLink**：优先用 TP（带宽高）
- **节点间 InfiniBand**：TP 限制在节点内，PP 跨节点
- **以太网**：尽量减少 TP，PP 优先跨节点

### GPU 数量计算

$$
\text{DP size} = \frac{\text{World Size}}{\text{TP} \times \text{PP} \times \text{CP}}
$$

例如 32 GPUs，TP=2, PP=4, CP=2 → DP = 32 / (2×4×2) = 2

### ※ 注意 DP 口径的歧义（MoE 场景）

很多人算资源时把 `DP × TP × PP × EP` 直接相乘，这**有时对有时错**。根源在于 `DP` 这个词有两种含义：

```
DP_a：Attention/Dense 层看到的数据并行总副本数
DP_r：Residual-DP，扣掉 EP 之后剩下的 DP
关系：DP_a = EP × DP_r
```

因此总卡数有两种等价写法：

```
写法一：N_gpu = TP × PP × DP_a       （EP 是 DP_a 的子组，不另乘）
写法二：N_gpu = TP × PP × EP × DP_r （EP 拆出来单独乘）
```

**常见错误**：把 `DP_a` 当成 DP 又额外乘一次 EP，等于同一个维度算了两遍。

![通常模式下 EP 是 DP_a 的子组，不能重复相乘](/img/LLM/2026-7-2-Parallelism/folding-mode.jpg)

**例子**：TP=2, PP=4, DP\_a=8, EP=4

```
✔ 正确：2 × 4 × 8 = 64         （EP=4 已是 DP_a=8 的子组）
✖ 错误：2 × 4 × 8 × 4 = 256    （重复乘了 EP）
✔ 等价：2 × 4 × 4 × 2 = 64     （用 DP_r=8/4=2 口径）
```

### MoE Parallel Folding 模式

Folding 的核心思想：Attention/Dense 层和 MoE 层使用**两套不同的并行网格**，但复用同一批 GPU。

```
Attention/Dense 侧：TP_a × CP × DP_a × PP
MoE/Expert 侧：    ETP × EP × EDP × PP

约束：TP_a × CP × DP_a = ETP × EP × EDP（两侧覆盖同一批 GPU）
```

**Folding 的优势**：EP 可以从 `TP_a × CP × DP_a` 整块 rank 平面里折叠出来，而不仅限于从 DP\_a 里取。这意味着可以支持更大的 EP。

![Folding 模式：同一批 GPU 被两套网格解释（Attention 侧 vs MoE 侧）](/img/LLM/2026-7-2-Parallelism/folding-two-grids.jpg)

**例子**：TP\_a=4, DP\_a=8, PP=4（共 128 卡）

```
通常模式：EP 最大 = DP_a = 8
Folding 模式：EP 最大 = TP_a × DP_a = 32（ETP=1, EDP=1）
```

### 速查表

| 场景 | 总卡数公式 |
|------|----------|
| Dense 或 EP=1 | TP × PP × DP\_a |
| 通常 MoE（DP = DP\_a 口径） | TP × PP × DP\_a（EP 不另乘） |
| 通常 MoE（DP = DP\_r 口径） | TP × PP × EP × DP\_r |
| Folding | TP\_a × CP × DP\_a × PP = ETP × EP × EDP × PP |

---

## 11. 关键 Takeaways

1. **没有银弹**：不同策略解决不同瓶颈，实际训练需要**混合多种并行**
2. **通信是核心权衡**：TP 通信频繁但切分细、PP 通信少但有气泡、DP 最简单但冗余大
3. **TP in node, PP across nodes**：遵循硬件拓扑的黄金法则
4. **ZeRO 是 DP 的完美补充**：几乎零成本消除冗余
5. **长序列用 CP**：SP 减少激活内存，CP 才是解决长上下文的关键
6. **MoE 用 EP**：稀疏激活配合专家分布，实现参数量和计算量的解耦
