---
layout: post
title: Transformer
categories: [LLM, AI-Infra]
tags: [LLM, AI-Infra,]
description: Attention Is All You Need 论文阅读笔记
---

> 原文：[Attention Is All You Need (Transformer) 论文精读](https://zhouyifan.net/2022/11/12/20220925-Transformer/)

## 1. 背景与动机

- 传统序列建模主要依赖 RNN（LSTM、GRU），存在**无法并行计算**和**长距离依赖难以捕获**的问题
- CNN 可以并行，但捕获远距离依赖需要堆叠很多层（感受野受限）
- Transformer 完全基于 **Attention 机制**，抛弃了循环和卷积结构，同时实现了并行化和全局依赖建模

## 2. 整体架构

![Transformer 整体架构（论文 Figure 1）](/img/LLM/2026-7-1-Transformer/architecture.png)

Transformer 采用经典的 **Encoder-Decoder** 结构：

- **Encoder**：由 N=6 个相同的层堆叠，每层包含：
  1. Multi-Head Self-Attention
  2. Position-wise Feed-Forward Network（FFN）
  - 每个子层都有 **残差连接（Residual Connection）** + **层归一化（Layer Normalization）**

- **Decoder**：同样由 N=6 个相同的层堆叠，每层包含：
  1. Masked Multi-Head Self-Attention（防止看到未来信息）
  2. Multi-Head Cross-Attention（对 Encoder 输出做注意力）
  3. Position-wise FFN
  - 同样有残差连接 + LayerNorm

## 3. 核心机制：Scaled Dot-Product Attention

![Scaled Dot-Product Attention（论文 Figure 2 左）](/img/LLM/2026-7-1-Transformer/scaled-dot-product-attention.png)

### 计算公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 关键点

| 要素 | 说明 |
|------|------|
| Q (Query) | 查询向量，表示"我想找什么" |
| K (Key) | 键向量，表示"我有什么" |
| V (Value) | 值向量，表示"实际的内容" |
| \(\sqrt{d_k}\) 缩放 | 防止点积值过大导致 softmax 梯度消失 |

### 为什么要缩放？

当 \(d_k\) 较大时，\(QK^T\) 的方差为 \(d_k\)，值会很大，softmax 会趋向 one-hot 分布，梯度接近 0。除以 \(\sqrt{d_k}\) 将方差归一化为 1。

## 4. Multi-Head Attention

![Multi-Head Attention（论文 Figure 2 右）](/img/LLM/2026-7-1-Transformer/multi-head-attention.png)

### 核心思想

将 Q、K、V 分别通过 h 个不同的线性投影，在不同的子空间中并行计算注意力，最后拼接：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### 参数设置

- 论文使用 **h=8** 个头
- \(d_{model} = 512\)，每个头的维度 \(d_k = d_v = d_{model}/h = 64\)
- 多头的总计算量与单头全维度注意力相近，但表达能力更强

### 直觉理解

不同的 head 可以关注不同类型的关系（如语法关系、语义关系、位置关系等）。

## 5. 三种 Attention 的使用方式

| 类型 | Q 来源 | K/V 来源 | 用途 |
|------|--------|----------|------|
| Encoder Self-Attention | Encoder 自身 | Encoder 自身 | 编码上下文信息 |
| Decoder Masked Self-Attention | Decoder 自身 | Decoder 自身 | 自回归生成，mask 未来位置 |
| Encoder-Decoder Cross-Attention | Decoder | Encoder 输出 | Decoder 关注输入序列 |

### Cross-Attention 详解

Decoder 中的第二个注意力子层比较特殊——它是一个 **Cross-Attention** 层：
- **K、V 来自 Encoder 的输出**（即输入序列的编码表示）
- **Q 来自 Decoder 上一层的输出**（即当前已生成部分的表示）

这使得 Decoder 在生成每个 token 时，都能"回头看"整个输入序列，从中提取相关信息。直觉上：Q 表示"我现在需要什么信息"，而 K/V 则提供了"输入序列中有哪些信息可供参考"。

## 6. Position-wise FFN

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

- 两层全连接，中间用 **ReLU** 激活
- 内部维度 \(d_{ff} = 2048\)，输入输出维度 \(d_{model} = 512\)
- 对每个位置独立且相同地应用（position-wise）
- 可以看作 kernel size=1 的两层卷积

## 7. Positional Encoding（位置编码）

Transformer 没有循环结构，无法感知序列顺序，需要显式注入位置信息。

### 正弦/余弦位置编码

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

### 设计优势

- 每个位置有唯一的编码
- 不同位置间的相对关系可以通过线性变换表示：\(PE_{pos+k}\) 可由 \(PE_{pos}\) 线性表出
- 可以泛化到训练时未见过的序列长度
- 与可学习的位置编码效果相当，但无需额外参数

## 8. 残差连接 + Layer Normalization

每个子层的输出：

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

- **残差连接**：缓解深层网络的梯度消失问题，帮助梯度直接传播
- **Layer Norm**：对每个样本的特征维度归一化（不同于 Batch Norm 对 batch 维度归一化），更适合序列任务
- 为配合残差连接，模型所有子层输出维度统一为 \(d_{model} = 512\)

## 9. Decoder 的 Mask 机制

![Causal Mask 示意：下三角为有效注意力区域](/img/LLM/2026-7-1-Transformer/causal-mask.png)

- Decoder 在自注意力中使用 **causal mask**（上三角 mask）
- 将未来位置的注意力权重设为 \(-\infty\)，softmax 后为 0
- 确保位置 i 的预测只依赖于位置 < i 的已知输出，实现自回归特性

## 10. 训练细节

### 优化器

- Adam：\(\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}\)
- **Warmup + 衰减学习率调度**：

$$
lr = d_{model}^{-0.5} \cdot \min(step^{-0.5},\ step \cdot warmup\_steps^{-1.5})
$$

前 warmup_steps 步线性增长，之后按步数的平方根衰减。

![学习率调度曲线：先 warmup 后衰减](/img/LLM/2026-7-1-Transformer/lr-schedule.png)

### 正则化

| 方法 | 细节 |
|------|------|
| Residual Dropout | 每个子层输出 + embedding 加入 dropout，\(P_{drop}=0.1\) |
| Label Smoothing | \(\epsilon_{ls}=0.1\)，降低模型过于自信，提高 BLEU |

## 11. 为什么选择 Self-Attention？

论文对比了 Self-Attention、RNN、CNN 在三个维度的表现：

| 维度 | Self-Attention | RNN | CNN |
|------|---------------|-----|-----|
| 每层计算复杂度 | \(O(n^2 \cdot d)\) | \(O(n \cdot d^2)\) | \(O(k \cdot n \cdot d^2)\) |
| 顺序操作（并行性） | \(O(1)\) | \(O(n)\) | \(O(1)\) |
| 最大路径长度 | \(O(1)\) | \(O(n)\) | \(O(\log_k n)\) |

- Self-Attention 在**并行性**和**最大路径长度**上有明显优势
- 当序列长度 n < 模型维度 d 时（常见情况），Self-Attention 计算也更高效

## 12. 模型配置

| 参数 | Base 模型 | Big 模型 |
|------|-----------|----------|
| \(d_{model}\) | 512 | 1024 |
| \(d_{ff}\) | 2048 | 4096 |
| h (头数) | 8 | 16 |
| 层数 N | 6 | 6 |
| \(P_{drop}\) | 0.1 | 0.3 |
| 参数量 | 65M | 213M |

## 13. 关键 Takeaways

1. **Attention 是核心**：Self-Attention 让每个 token 能直接关注序列中任意其他 token，路径长度 O(1)
2. **并行化**：摆脱 RNN 的序列依赖，训练效率大幅提升
3. **可扩展性**：架构简洁统一，容易通过增加层数、维度来 scale up
4. **组件设计精妙**：缩放因子、多头、残差连接、位置编码——每个设计都有明确的动机
5. **奠基之作**：后续的 BERT、GPT、ViT 等模型都基于 Transformer 架构

---

## 14. 现代 LLM 的架构演进

> 参考：[Decoder-Only Transformers: The Workhorse of Generative LLMs](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)

今天的大语言模型（GPT、Qwen、DeepSeek 等）仍然基于 Transformer，但几乎每个组件都经过了迭代优化：

![Decoder-Only Transformer 架构（现代 LLM 的标准结构）](/img/LLM/2026-7-1-Transformer/decoder-only-architecture.png)

| 组件 | 原始 Transformer | 现代 LLM |
|------|------|------|
| 整体结构 | Encoder-Decoder | **Decoder-Only**（去掉 Encoder 和 Cross-Attention） |
| 归一化 | Post-LayerNorm | **Pre-RMSNorm**（训练更稳定，计算更快） |
| 位置编码 | 固定正弦/余弦 | **RoPE**（旋转位置编码，支持长度外推） |
| 注意力 | MHA | **GQA**（Qwen/LLaMA）/ **MLA**（DeepSeek，KV 压缩） |
| FFN 激活 | ReLU，2 个矩阵 | **SwiGLU**，3 个矩阵门控 |
| 稀疏化 | Dense | **MoE**（DeepSeek-V2/V3，参数大但计算稀疏） |
| 上下文长度 | 512 tokens | 128K+ tokens |

### 关键变动说明

1. **Decoder-Only**：纯生成任务不需要 Encoder，统一用 causal attention 做自回归预测即可
2. **Pre-RMSNorm**：归一化放在子层之前（而非之后），梯度流更顺畅；RMSNorm 去掉均值中心化，速度更快且效果相当
3. **RoPE**：将位置信息通过旋转矩阵作用在 Q/K 上，自然编码相对位置，配合 NTK/YaRN 可外推到训练时未见的序列长度
4. **GQA / MLA**：多个 Q head 共享同一组 K/V（GQA）或将 KV 压缩到低秩空间（MLA），大幅减少推理时的 KV Cache 显存占用
5. **SwiGLU**：\(\text{FFN}(x) = (\text{Swish}(xW_{gate}) \odot xW_{up}) W_{down}\)，门控机制提升表达能力
6. **MoE**：每层设置多个专家 FFN，每次只激活少数几个，参数量大但计算量不变

> **本质不变**：残差连接、注意力机制、自回归生成的核心范式没有变，变的是每个组件的具体实现。