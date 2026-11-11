---
title: 第七章 梯度下降 (Gradient Descent) 講義摘要
type: source
tags: [source-summary, optimization, gradient-descent, adam, normalization]
sources: [ch07-gradient-descent]
created: 2026-11-11
updated: 2026-11-11
---

# 第七章 梯度下降 (Gradient Descent) 講義摘要

本頁為原始講義 `raw/ch07-gradient-descent.pdf`（對應 Bishop & Bishop 2024 *Deep Learning: Foundations and Concepts* 第 7 章，共 18 頁）之核心知識綱要整理。

---

## 講義章節結構

講義系統化地探討了深度學習中的最佳化機制：從高維權重空間的誤差曲面幾何出發，闡述梯度資訊相對於無導數搜尋的決定性優勢，深入分析隨機梯度下降與初始化方法，推導二次曲面下的收斂條件與病態條件數瓶頸，並依序推演動量、自適應學習率優化器（AdaGrad、RMSProp、Adam）與深度網路內部正規化技術：

1. **引言與現代深度學習最佳化觀**（p.1–2）
   - 目標從非追求「精確最小化（Exact Optimization）」，而在於選擇「具良好泛化行為之演算法（Well-behaved Algorithms）」。
   - 解析梯度的計算代價優勢：$\mathcal{O}(W)$ vs 數值微商 $\mathcal{O}(W^2)$ vs 隨機搜尋 $\mathcal{O}(W^3)$。
2. **誤差曲面與幾何特性 (Error Surfaces)**（p.2–4）
   - 駐點條件 $\nabla E(\mathbf{w}) = \mathbf{0}$，鞍點（Saddle Points）、局部極小與極大的本質區別。
   - 局部二次近似（Taylor 二階展開）與 Hessian 矩陣 $\mathbf{H}$ 的正定性判別。
   - 權重空間多重對稱性所衍生的等價局部極小值（$M! 2^M$）。
3. **梯度下降最佳化家族 (Gradient Descent Optimization)**（p.4–7）
   - 批次梯度下降 (BGD)、隨機梯度下降 (SGD) 與小批次梯度下降 (Mini-batch SGD)。
   - 小批次估計的標準差依 $\sigma / \sqrt{B}$ 縮放與 GPU 並行張量計算優勢。
   - 參數初始化（打破對稱性）與 He (Kaiming) 初始化在 ReLU 下方差守恆 $\text{Var}(w) = 2/D_{\text{in}}$ 的嚴謹推導。
4. **收斂性分析與進階優化器 (Convergence and Advanced Optimizers)**（p.7–11）
   - 二次曲面特徵座標下的收斂因子 $(1 - \eta \lambda_i)$ 與學習率穩定性上限 $\eta < 2/\lambda_{\max}$。
   - 條件數 $\kappa = \lambda_{\max} / \lambda_{\min}$ 引起的橫向劇烈震盪與縱向爬行慢速瓶頸。
   - 古典動量（Momentum）與 Nesterov 加速動量（NAG）。
   - 學習率排程（Step Decay, Exponential, Cosine Annealing, Warmup）。
   - 自適應優化器演進：AdaGrad 的累積平方梯度、RMSProp 的指數加權移動平均（EMA），以及 Adam 結合一階與二階動量並施加無偏修正（Bias Correction）。
5. **正規化技術 (Normalization Techniques)**（p.11–14）
   - 輸入資料標準化對降低 Hessian 條件數的幾何效果。
   - 批次正規化 (BatchNorm, Ioffe & Szegedy 2015)：跨 Batch 計算、可學習仿射參數 $\gamma, \beta$、推論期指數移動平均（EMA）。
   - 層正規化 (LayerNorm, Ba et al. 2016)：跨特徵維度標準化，適用於序列模型、Transformer 與非依賴批次大小的情境。

---

## 本章產出之 Wiki 頁面

依據上述核心主題，本章產出 4 個概念專頁與 1 個演算法綜合比較頁：

1. [[error-surfaces-and-geometry|誤差曲面與幾何特性 (Error Surfaces and Geometry)]]
2. [[gradient-descent-and-initialization|梯度下降家族與參數初始化 (Gradient Descent and Initialization)]]
3. [[momentum-and-adaptive-optimizers|動量法與自適應學習率優化器 (Momentum and Adaptive Optimizers)]]
4. [[normalization-techniques|網路正規化技術 (Normalization Techniques)]]
5. [[optimization-algorithms-comparison|最佳化演算法綜合比較 (Optimization Algorithms Comparison)]]

---

## 相關頁面
- [[index|知識庫目錄]]
- [[ch06-deep-neural-networks|第六章 深度神經網路]]
- [[error-functions-and-output-units|誤差函數與輸出單元]]
