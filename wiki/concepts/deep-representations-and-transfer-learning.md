---
title: 深度表徵與遷移學習 (Deep Representations and Transfer Learning)
type: concept
tags: [representations, transfer-learning, contrastive-learning, infonce, clip, dag]
sources: [ch06-deep-neural-networks]
created: 2026-11-11
updated: 2026-11-11
---

# 深度表徵與遷移學習 (Deep Representations and Transfer Learning)

傳統機器學習倚賴繁複的人工特徵工程（Hand-crafted Feature Engineering）。深度神經網路（DNN）的核心變革在於**端到端表徵學習 (End-to-End Representation Learning)**：由淺入深自動學習分層特徵，並賦予模型強大的跨任務泛化與遷移能力。

---

## 1. 深度表徵的本質 (Nature of Deep Representations)

📌講義剖析了深度網路特徵表示的兩大關鍵維度：

### 1.1 分散式表徵 (Distributed Representations)
- **局部表徵 vs 分散式表徵**：
  - **局部表徵 (One-hot / Localist)**：單一概念由單一單元專屬負責（如 One-hot 編碼或 RBF 中心）。要表示 $N$ 個不同概念需要 $N$ 個單元。
  - **分散式表徵**：概念由大量神經元的**組合激活模式 (Pattern of Activation)** 共同編碼；同時，每個神經元亦參與多個不同概念的構成。
- **維度效益**：$M$ 個二元隱藏單元理論上可表示高達 $2^M$ 種相異概念，具備指數級別的資訊容納與緊湊度 (來源: [[ch06-deep-neural-networks]], p.12)。

### 1.2 階層式特徵組合 (Hierarchical Feature Composition)
神經網路透過縱向堆疊多個隱藏層，實現特徵的逐層抽象化：
- **淺層 (Early Layers)**：擷取局部、低階物理特徵（如影像中的邊緣 Edge、斑點 Gabor-like filters、材質 Texture）。
- **中層 (Intermediate Layers)**：組合低階特徵為具體的物件部件（Parts, 如眼睛、輪子、轉角）。
- **深層 (Deep Layers)**：形成語意層級的高階全域表徵（Semantic Concepts, 如特定種類的狗、汽車、人臉）(來源: [[ch06-deep-neural-networks]], p.13)。

---

## 2. 遷移學習與微調 (Transfer Learning and Fine-tuning)

📌講義指出，深度表徵的階層性使得在大型通用資料集（如 ImageNet）上預訓練的模型權重具有普適價值：

### 2.1 典範與策略
1. **特徵萃取器 (Feature Extractor / Linear Probe)**：
   凍結預訓練模型所有骨幹網路（Backbone）層的權重，僅將最後一或多層隱藏激活向量作為特徵輸入，重新訓練一個輕量級線性分類器。適用於目標任務樣本極少的情境。
2. **微調 (Fine-tuning)**：
   保留預訓練骨幹權重作為初始化參數，在下游目標任務資料上使用較小學習率對所有或部分層級進行反向傳播更新 (來源: [[ch06-deep-neural-networks]], p.14)。

### 2.2 多任務學習與元學習 (Multi-task & Meta-learning)
- **多任務學習 (Multi-task Learning)**：多個相關任務共享相同的特徵萃取骨幹網絡，各任務擁有獨立的輸出預測頭（Heads）。底層共享表徵必須兼顧多個任務的目標，發揮歸納偏置（Inductive Bias）效果，防止過度擬合。
- **元學習 (Meta-learning)**：強調「學習如何學習 (Learning to Learn)」，透過在眾多子任務中訓練，使模型能僅依據極少數幾筆樣本（Few-shot Learning）即快速調適至全新任務 (來源: [[ch06-deep-neural-networks]], p.14)。

---

## 3. 對比式學習與 CLIP (Contrastive Learning and Multimodal Alignment)

📌講義特別探討了現代無監督/自監督表徵學習的核心機制——**對比式學習 (Contrastive Learning)**：

### 3.1 對比式目標與 InfoNCE 損失函數
對比式學習旨在拉近正樣本對（Positive Pairs）在特徵空間中的距離，並推遠負樣本對（Negative Pairs）。
給定查詢向量 $\mathbf{q}$、一個匹配的正樣本 $\mathbf{k}_+$ 與 $K$ 個負樣本 $\{\mathbf{k}_-\}$，其 **InfoNCE 損失函數** 定義為：
$$\mathcal{L}_{\text{InfoNCE}} = -\ln \frac{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau)}{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau) + \sum_{i=1}^K \exp(\mathbf{q} \cdot \mathbf{k}_{-, i} / \tau)}$$
其中 $\tau > 0$ 為溫度係數（Temperature Parameter），用於控制特徵分布的集中程度與難負樣本的懲罰力度 (來源: [[ch06-deep-neural-networks]], p.15)。

### 3.2 CLIP (Contrastive Language-Image Pretraining)
- **架構**：包含一個影像編碼器 $f_{\text{img}}(\cdot)$ 與文字編碼器 $f_{\text{txt}}(\cdot)$。
- **對準機制**：在由 $N$ 個「影像-文字」配對構成的 Batch 內，計算 $N \times N$ 個跨模態餘弦相似度矩陣。
- **損失**：同時在影像對齊文字、文字對齊影像兩個方向上施加對稱的交叉熵損失：
  $$\mathcal{L}_{\text{CLIP}} = \frac{1}{2} (\mathcal{L}_{\text{img}\to\text{txt}} + \mathcal{L}_{\text{txt}\to\text{img}})$$
- 達成了強大的**零樣本遷移能力 (Zero-shot Transfer)** (來源: [[ch06-deep-neural-networks]], p.15)。

---

## 4. 計算圖、DAG 與張量維度 (Computational Graphs, DAG, and Tensors)

📌講義指出，現代深度神經網路已大幅跳脫傳統嚴格層層堆疊的鏈狀限制：
- **有向無環圖 (Directed Acyclic Graph, DAG)**：網路架構可任意分岔、合併（如 ResNet 的殘差跳躍連接 Residual Connections、Inception 模組、DenseNet 的多路串接）。
- **張量維度表示法 (Tensor Notation)**：
  - 影像資料張量維度常見為 $(\text{Batch Size}, \text{Height}, \text{Width}, \text{Channels})$ 或 $(\text{Batch}, C, H, W)$。
  - 隱藏狀態張量在節點間以前向與反向傳播的形式高效流動 (來源: [[ch06-deep-neural-networks]], p.16)。

---

## 自我測驗 Q&A

### Q1: 何謂「分散式表徵 (Distributed Representation)」？它與傳統的 One-hot 局部表徵有何本質差別？
> [!answer]- 答案
> One-hot 局部表徵將每個概念指派給單一特定維度/神經元，表示 $N$ 個概念需 $N$ 個維度，各概念間無法自然衡量相似性。而分散式表徵將概念分散編碼在多個神經元的連續激活模式中，同時單一神經元亦參與多種概念的表達。$M$ 個神經元理論上能表達 $2^M$ 種組合，具備指數級的資訊緊湊性與語意平滑性。

### Q2: 試解釋 InfoNCE 損失函數中溫度係數 $\tau$ 的作用機制。
> [!answer]- 答案
> InfoNCE 形式為負對數 Softmax 機率：$\mathcal{L} = -\ln \frac{\exp(\mathbf{q}\cdot \mathbf{k}_+ / \tau)}{\sum \exp(\mathbf{q}\cdot \mathbf{k} / \tau)}$。
> 溫度係數 $\tau$ 調節了內積相似度分佈的尖銳程度：
> - 若 $\tau$ 很小，Softmax 機率會極度極化，使模型強烈聚焦於最難區分的負樣本（Hard Negatives），但過小會導致訓練不穩定；
> - 若 $\tau$ 很大，Softmax 趨近均勻分布，削弱了對難樣本的判別激勵。

### Q3: 遷移學習中「線性探測 (Linear Probing)」與「全面微調 (Full Fine-tuning)」的適用情境有何差異？
> [!answer]- 答案
> - **線性探測**：凍結所有預訓練特徵層，僅訓練最頂層線性分類頭。適用於目標任務樣本極少（防止微調破壞泛化表徵並引起嚴重過度擬合）、或目標任務與預訓練資料語意極為相似的情境。
> - **全面微調**：微調整個網路所有權重。適用於目標任務擁有足夠樣本量、或下游任務領域（Domain）與原預訓練領域存在顯著差異，需對深層甚至淺層濾波器進行重新特徵適配的情境。

---

## 相關頁面
- [[ch06-deep-neural-networks|第六章 深度神經網路講義摘要]]
- [[multilayer-feedforward-networks|多層前饋神經網路]]
- [[error-functions-and-output-units|誤差函數與輸出單元]]
