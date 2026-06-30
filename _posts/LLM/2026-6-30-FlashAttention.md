---
layout: post
title: FlashAttention 算法原理篇
categories: [LLM, AI-Infra]
tags: [LLM, AI-Infra, Attention, CUDA, 高性能计算]
description: FlashAttention 算法原理阅读笔记，涵盖 GPU 编程模型、Attention 运算回顾及 FlashAttention 的优化推导过程
---

> 原文：[不会 CUDA 也能轻松看懂的 FlashAttention 教程（算法原理篇）](https://zhouyifan.net/2025/08/24/20250511-flashattention-1/)

## 一、GPU 编程基础知识

### 1.1 存储模型

GPU 的存储层次（从快到慢、从小到大）：

```
GPU SRAM（片上缓存）→ GPU HBM（显存）→ CPU DRAM（内存）
```

- **GPU HBM**：即常说的"显存"，PyTorch 中的 Tensor 默认存放在这里
- **GPU SRAM**：片上高速缓存，容量小但读写极快
- 底层 GPU 计算的实际流程：先从 HBM **load** 数据到 SRAM，在 SRAM 上做运算，再将结果 **store** 回 HBM
- **访存开销不可忽略**，尤其是数据量大时

![GPU 存储层次模型（来自 FlashAttention 论文）](/img/LLM/2026-6-30-FlashAttention/7.jpg)

### 1.2 算子融合（Operator Fusion / Kernel Fusion）

核心思想：将多个连续的小算子合并成一个大算子，减少中间结果在 HBM 上的读写次数。

两个常见优化场景：
1. **多次用到同一批数据**：只从 HBM 读取一次，而不是每个算子各读一次
2. **中间变量只是过渡量**：不写回 HBM 再读回来，直接保留在 SRAM 上继续运算

### 1.3 并行编程

- GPU 拥有大量独立的计算单元，适合执行**可并行的运算**
- 并行编程的关键：写一段以"计算单元 ID"为参数的通用程序，每个单元根据自己的 ID 处理对应的数据块
- **可并行的前提**：各计算单元处理的任务之间必须相互独立
  - 例：向量逐元素加法 → 天然独立，可并行
  - 例：向量求和（每步依赖上一步的累加结果）→ 不能直接并行

![各计算单元独立处理数据块](/img/LLM/2026-6-30-FlashAttention/8.jpg)

![以计算单元 ID 为参数的通用并行程序](/img/LLM/2026-6-30-FlashAttention/9.jpg)

---

## 二、Attention 运算回顾

标准 Scaled Dot-Product Attention：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

![Attention 计算示意：从数据集合 B 中加权提取信息](/img/LLM/2026-6-30-FlashAttention/1.jpg)

**softmax 的计算拆解为三步：**
1. 对每个相似度 $s_i$ 求 $e^{s_i}$，得到分子（numerator）
2. 对所有分子求和，得到分母（denominator）
3. 每个分子除以分母，得到归一化权重 $p_i$

![标准 Softmax Attention 公式推导](/img/LLM/2026-6-30-FlashAttention/3.jpg)

![矩阵形式的 Attention 公式 Attention(Q,K,V)](/img/LLM/2026-6-30-FlashAttention/5.jpg)

**Attention 的关键性质：**
- 不同 query（$q_i$）之间的计算完全独立 → 天然适合并行
- 同一个 query 内，softmax 要求遍历所有 key 才能计算分母 → 是优化的瓶颈

---

## 三、FlashAttention 的优化推导

### 3.1 问题所在：巨大的中间变量

PyTorch 标准 Attention 实现中：

![PyTorch Attention 代码](/img/LLM/2026-6-30-FlashAttention/10.jpg)

加入访存操作后，可以看出中间变量的读写开销：

![加入 IO 操作后暴露的冗余 HBM 读写](/img/LLM/2026-6-30-FlashAttention/11.jpg)

```python
s = q @ k.T          # shape: [SL, SL]
p = softmax(s)       # shape: [SL, SL]
o = p @ v            # shape: [SL, D]
```

- 中间变量 `s`、`p` 的形状为 `[SL, SL]`
- 实际大模型中序列长度 SL 至少达到 $10^4$ 量级，而特征维度 D 通常只有 32/64
- `[SL, SL]` 远大于 `[SL, D]`，反复在 HBM 读写这两个中间变量，**严重拖慢速度**

### 3.2 优化步骤一：按 query 拆分，消除二维中间变量

由于各 query 之间独立，让每个 GPU 计算单元只负责**一个 query** 的计算：
- 中间变量 `s`、`p` 的形状从 `[SL, SL]` 降为 `[SL]`
- 对 K、V 用循环逐步读取，每次只读一个 `k_i`、`v_i`

![按 query 拆分后、带 K/V 循环的 GPU 程序](/img/LLM/2026-6-30-FlashAttention/13.jpg)

但 `[SL]` 形状的中间变量仍然超出 SRAM 限制，还需进一步优化。

### 3.3 优化步骤二：拆解 softmax，合并循环

将 softmax 的各步计算与 Q·K 点乘和 P·V 乘法合并，关键观察：

> **softmax 的除法（归一化）与之后和 V 的矩阵乘法是相互独立的。**

因此可以：
1. 在同一个循环里，边计算 Q·K 点乘，边对 `exp(s_i)` 与 `v_i` 做加权累加（忽略分母）
2. 同时维护 softmax 的分母（累加 `exp(s_i)`）
3. 循环结束后，将累加结果统一除以分母

```
# 伪代码示意
denominator = 0
O = 0
for i in range(SL):
    s_i = q · k[i]
    exp_s_i = exp(s_i)
    O += exp_s_i * v[i]      # 暂不除以分母
    denominator += exp_s_i
O = O / denominator          # 循环结束后统一归一化
```

![将 exp(s_i) 与 v_i 加权累加合并进同一循环，同时维护分母](/img/LLM/2026-6-30-FlashAttention/16.jpg)

这样，长度为 SL 的中间变量 `numerator` 被替换为一个**标量临时变量**，完全消除了超长中间变量。

![最终的简易 FlashAttention 实现：无超长中间变量的融合 GPU 程序](/img/LLM/2026-6-30-FlashAttention/20.jpg)

### 3.4 核心优化效果

| 实现方式 | 中间变量大小 | HBM 读写量 |
|---|---|---|
| 标准 PyTorch Attention | $O(SL^2)$ | 高 |
| FlashAttention（简易版） | $O(D)$（仅 SRAM 内） | 显著降低 |

---

## 四、总结

FlashAttention 的本质是**算子融合 + 访存优化**，核心思路：

1. **利用 Attention 各 query 独立的特性**，将计算拆分到各 GPU 计算单元
2. **利用 softmax 除法与 P·V 乘法的独立性**，将整个 Attention 融合为单一循环
3. 循环中只维护标量累积量，不产生长度为 SL 的中间变量
4. 避免了 `[SL, SL]` 中间矩阵在 HBM 上的读写，大幅减少访存开销

> **FlashAttention 并没有减少浮点运算量，而是通过减少 HBM 访存次数来提升速度。** 这是其在 IO-bound 场景（长序列）下加速效果显著的根本原因。