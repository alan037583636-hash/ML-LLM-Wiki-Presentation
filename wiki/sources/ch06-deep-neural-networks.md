---
title: 第六章 深度神經網路 (Deep Neural Networks) 講義摘要
type: source
tags: [source-summary, deep-learning, neural-networks, representations, mdn]
sources: [ch06-deep-neural-networks]
created: 2026-11-11
updated: 2026-11-11
---

# 第六章 深度神經網路 (Deep Neural Networks) 講義摘要

本頁為原始講義 `raw/ch06-deep-neural-networks.pdf`（對應 Bishop & Bishop 2024 *Deep Learning: Foundations and Concepts* 第 6 章，共 26 頁）之核心知識綱要整理。

---

## 講義章節結構

講義內容聚焦於從單層網路邁向深度架構的核心理論轉折，涵蓋維度災難、多層感知機、表徵學習、各類輸出單元與誤差函數的典則配對，以及用於多峰條件分布的混合密度網路（MDN）：

1. **維度災難與資料流形 (Curse of Dimensionality and Data Manifold)**（p.1–5）
   - 維度災難在高維空間中的反直覺幾何特性（超球體積集中於表層、高斯分布質量集中於薄球殼）。
   - 傳統非參數模型（如高斯 RBF、SVM）受限於固定基底函數與局部泛化瓶頸。
   - 資料流形假設（Data Manifold Hypothesis）與深度網路適應性特徵萃取的成功根本。
2. **多層前饋神經網路 (Multilayer Feedforward Networks)**（p.6–11）
   - 網路矩陣代數形式與前向傳播機制。
   - 隱藏單元激活函數族譜：Logistic Sigmoid、$\tanh$、ReLU、Softplus、Leaky ReLU。
   - 通用近似定理（Universal Approximation Theorem）及其現實侷限性（寬度指數爆炸 vs 深度高效表徵）。
   - 權重空間對稱性（Weight-space Symmetries）：隱藏單元置換與正負號翻轉對稱（共 $M! 2^M$ 個等價極值點）。
3. **深度表徵與遷移學習 (Deep Representations and Transfer Learning)**（p.12–16）
   - 分散式表徵（Distributed Representation）與階層式特徵萃取（Hierarchical Feature Extraction）。
   - 遷移學習（Transfer Learning）、特徵微調（Fine-tuning）、多任務學習（Multi-task Learning）與元學習（Meta-learning）。
   - 對比式表徵學習（Contrastive Learning）、InfoNCE 損失函數與 CLIP 跨模態對齊。
   - 有向無環圖（DAG）架構與張量（Tensors）維度表示法。
4. **誤差函數與輸出單元 (Error Functions and Output Units)**（p.17–20）
   - 最大概似估計（MLE）視角下預測分布、輸出單元激活函數與損失函數的配對架構。
   - 迴歸、二元分類、多重二元分類與多類別分類的典則連結。
   - 前激活梯度（Pre-activation Gradient）極簡形式 $\frac{\partial E}{\partial a_k} = y_k - t_k$ 的數學統一性。
5. **混合密度網路 (Mixture Density Networks, MDN)**（p.21–26）
   - 逆問題（Inverse Problems）與多峰條件分布（Multimodal Conditional Distributions）。
   - 高斯混合模型（GMM）與前饋神經網路之結合架構。
   - 混合係數、均值向量與變異數之參數化與約束。
   - MDN 負對數概似損失函數推導與三組輸出前激活導數（$\delta_{\pi_k}, \delta_{\mu_{kc}}, \delta_{\sigma_k}$）。
   - 預測分布計算、條件期望值、條件變異數與條件眾數（Conditional Mode）預測策略。

---

## 本章產出之 Wiki 頁面

依據上述核心主題，本章產出 5 個概念專頁：

1. [[curse-of-dimensionality-data-manifold|維度災難與資料流形 (Curse of Dimensionality and Data Manifold)]]
2. [[multilayer-feedforward-networks|多層前饋神經網路 (Multilayer Feedforward Networks)]]
3. [[deep-representations-and-transfer-learning|深度表徵與遷移學習 (Deep Representations and Transfer Learning)]]
4. [[error-functions-and-output-units|誤差函數與輸出單元 (Error Functions and Output Units)]]
5. [[mixture-density-networks|混合密度網路 (Mixture Density Networks)]]

---

## 相關頁面
- [[index|知識庫目錄]]
- [[ch04-single-layer-regression|第四章 單層網路：迴歸]]
- [[ch05-single-layer-classification|第五章 單層網路：分類]]
