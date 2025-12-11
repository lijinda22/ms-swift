# Visual Knowledge Distillation Implementation Report

# 视觉知识蒸馏技术报告

## 1. 代码层面的设计 (Code-Level Design)

本项目实现了一套高度灵活、可扩展的**多教师视觉知识蒸馏 (Multi-Teacher Visual Knowledge Distillation)** 框架，旨在将多个特定领域（如病理学）的视觉基础模型（teacher models）的知识迁移到多模态大语言模型（MLLM）的视觉编码器（student model）中。

### 1.1 核心架构与类设计

核心逻辑主要集中在 `swift/trainers/trainers.py` 中的 `SftKdTrainer` 类与 `KnowledgeDistillationModule` 模块。

#### A. 异构教师模型支持 (Heterogeneous Teacher Support)

为了兼容不同架构的教师模型（如 Conch/CoCa, UNI/ViT），我们设计了统一的接口适配层：

* **统一接入点**: `SftKdTrainer._get_teacher_info` 方法。
  * **CoCa架构 (Conch)**: 自动定位到 `model.visual.trunk`，并提取 `forward_features`。
  * **标准 ViT (UNI)**: 直接调用 `forward_features`。
  * **特殊 Token 处理**: 动态检测 `num_prefix_tokens`（如 CLS token 数量），兼容 `no_embed_class` 的模型（如 UNI2），确保特征对齐时的索引正确性。
* **适配器模块 (`KnowledgeDistillationModule`)**:
  * 为每个教师模型实例化一个独立的适配器。
  * 包含两个投影层（Projection Layers）：将 Student 和 Teacher 的特征投影到相同的维度（`projection_dim`）。
  * 支持 EMA（Exponential Moving Average）中心更新，用于 DINO Loss 的稳定性。

#### B. 多粒度蒸馏策略 (Multi-Granularity Distillation Strategies)

通过 `kd_token_strategy` 参数控制蒸馏的粒度：

1. **`cls_mean` (Global Alignment)**:
    * **Student**: 对 Patch tokens 进行 Global Average Pooling (GAP)。
    * **Teacher**: 提取 CLS token（如果存在）或使用 GAP。
    * **目的**: 对齐图像的全局语义表示。
2. **`patch_mse` (Local/Structural Alignment)**:
    * **Student**: 保留完整的 Patch tokens 序列。
    * **Teacher**: 提取 Patch tokens，将其 Reshape 还原为 2D 网格 $(H, W)$。
    * **空间插值 (Interpolation)**: 使用 `bicubic` 插值将 Teacher 的特征图调整为与 Student 的特征图尺寸一致（解决了不同模型 patch size 不一致的问题）。
    * **目的**: 强迫 Student 学习 Teacher 的细粒度空间特征和纹理信息。

#### C. 动态自适应加权 (Dynamic Adaptive Weighting)

为了解决“哪个教师对当前样本更重要”的问题，引入了 `kd_weight_strategy="similarity_weighted"`：

* **机制**: 计算 Student 全局特征与每个 Teacher 全局特征的 **余弦相似度 (Cosine Similarity)**。
* **归一化**: 对所有 Teacher 的相似度得分进行 **Softmax**，生成动态权重 $\alpha_i$。
* **效果**: 模型会自动“关注”与当前图像特征最契合的 Teacher，降低噪声干扰。

#### D. 损失函数设计

支持两种核心损失函数 (`kd_loss_type`)：

1. **DINO Loss**:基于 Cross-Entropy 的对比学习损失，强调分布对齐，适合全局语义。
2. **Normalized MSE**: 对特征进行 L2 归一化后计算 MSE，等价于优化余弦距离，数值稳定性优于原始 MSE。

---

## 2. 论文描述 (Academic Description - 简体中文)

以下内容可作为论文中 **Methodology** 章节的基础描述。

### 论文题目建议

**Adaptive Multi-Source Visual Alignment for Pathology Multimodal LLMs**
*(面向病理多模态大模型的自适应多源视觉对齐方法)*

### 方法论描述

为了提升多模态大语言模型（MLLM）在特定领域（如病理学）的视觉表征能力，我们提出了一种**自适应多源视觉知识蒸馏框架 (Adaptive Multi-Source Visual Distillation, AM-VD)**。该框架旨在利用现有的高性能病理视觉编码器（Teachers）作为专家网络，指导 MLLM 视觉投影层（Student）的学习。

#### 2.1 异构特征对齐 (Heterogeneous Feature Alignment)

给定输入图像 $x$，Student 网络提取特征 $S \in \mathbb{R}^{L_s \times D_s}$，以及 $K$ 个异构 Teacher 网络提取特征 $\{T_k\}_{k=1}^K, T_k \in \mathbb{R}^{L_{t_k} \times D_{t_k}}$。由于各网络架构（ViT, CoCa等）和特征维度不同，我们引入了一组可学习的投影模块 $\phi_s$ 和 $\{\phi_{t_k}\}$，将所有特征映射到共享的潜在空间 $\mathbb{R}^D$。

针对空间分辨率不匹配的问题，对于局部特征蒸馏，我们采用空间插值操作 $\mathcal{I}$：
$$ \hat{T}_k = \mathcal{I}(\text{Reshape}(T_{k}^{\text{patch}}), (H_s, W_s)) $$
其中 $(H_s, W_s)$ 为 Student 的特征图尺寸。

#### 2.2 自适应专家混合加权 (Adaptive Mixture-of-Experts Weighting)

考虑到不同 Teacher 模型在不同类型病理图像（如 H&E 染色、IHC 染色）上的表现差异，我们设计了一种基于能够感知的动态加权机制。对于每个样本，计算 Student 全局表征与其在每个 Teacher 空间中投影的余弦相似度作为门控信号：

$$ \text{sim}_k = \cos(\phi_s(\text{GAP}(S)), \phi_{t_k}(T_k^{\text{cls}})) $$

利用 Softmax 函数生成归一化的混合权重 $\alpha_k$：
$$ \alpha_k = \frac{\exp(\text{sim}_k / \tau)}{\sum_{j=1}^K \exp(\text{sim}_j / \tau)} $$

这一机制使得模型能够逐样本（Sample-wise）地动态调整对不同 Teacher 的依赖程度。

#### 2.3 蒸馏目标函数 (Distillation Objective)

总的蒸馏损失 $L_{\text{KD}}$ 定义为所有 Teacher 损失的加权和：

$$ L_{\text{KD}} = \lambda \sum_{k=1}^K \alpha_k \cdot \mathcal{L}_{\text{align}}(\phi_s(S), \phi_{t_k}(\hat{T}_k)) $$

其中 $\mathcal{L}_{\text{align}}$ 根据配置可选用 DINO 损失（用于分布匹配）或 归一化 MSE 损失（用于特征逼近）。这种方法有效地实现了将多个领域专家的知识 "软注入" 到 MLLM 的视觉对齐层中，显著增强了其细粒度特征提取能力。
