---
title: 循序學習與 LMS 演算法 (Sequential Learning and LMS Algorithm)
type: algorithm
tags: [sequential-learning, online-learning, sgd, lms, adaptive-filter, rls, kalman]
sources: [ch04-single-layer-regression]
created: 2026-11-11
updated: 2026-11-11
---

# 循序學習與 LMS 演算法 (Sequential Learning and LMS Algorithm)

## 1. 批次 vs. 循序學習

📌講義：
- **批次方法 (Batch methods)**：如最大概似閉式解 $\mathbf{w}_{\text{ML}} = (\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{t}$，需一次載入並處理全體訓練集，計算成本與記憶體需求高。
- **循序方法 (Sequential methods / Online methods)**：資料點一次只考慮一個，呈現一個資料點就更新一次模型參數。非常適合超大型資料集或連續串流的即時應用（必須在看完所有資料前持續給出預測）。(來源: [[ch04-single-layer-regression]], p.7)

## 2. 隨機梯度下降 (SGD) 與 LMS

📌講義：若總誤差函數可分解為各單一樣本誤差之和 $E = \sum_n E_n$，隨機梯度下降在呈現第 $n$ 個資料點時更新參數：

$$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} - \eta \nabla E_n \tag{4.19}$$

其中 $\tau$ 為迭代次數，$\eta$ 為學習率 (learning rate)。
對於平方和誤差 $E_n = \frac{1}{2}\{t_n - \mathbf{w}^T\boldsymbol{\phi}_n\}^2$，梯度為 $\nabla E_n = -\{t_n - \mathbf{w}^T\boldsymbol{\phi}_n\}\boldsymbol{\phi}_n$。代入更新式得到：

$$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} + \eta \{t_n - \mathbf{w}^{(\tau)T}\boldsymbol{\phi}_n\} \boldsymbol{\phi}_n \tag{4.20}$$

其中 $\boldsymbol{\phi}_n = \boldsymbol{\phi}(\mathbf{x}_n)$。此式即為著名的**最小均方演算法 (Least-Mean-Squares, LMS algorithm)**，亦稱 Widrow–Hoff 學習規則。(來源: [[ch04-single-layer-regression]], p.7)

## 3. 訊號處理中的演化光譜（講義教學補充）

📌講義（講義教學補充：LMS、RLS 與 Kalman 濾波器）：
在訊號處理中，式 (4.20) 是 Widrow–Hoff 的適應性濾波器架構（$\boldsymbol{\phi}_n$ 為延遲線訊號，$\mathbf{w}$ 為 FIR 濾波器係數，用於通道等化與回音消除）：
1. **LMS**：僅依賴瞬時單步梯度，計算極快，收斂條件為 $0 < \eta < \frac{2}{\lambda_{\max}(\mathbf{R})}$，其中 $\mathbf{R} = \mathbb{E}[\boldsymbol{\phi}\boldsymbol{\phi}^T]$。
2. **遞迴最小平方 (RLS)**：每來一筆新資料，利用 Sherman–Morrison 公式遞迴更新協方差逆矩陣 $(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}$，達成精確的線上最小平方解。
3. **Kalman 濾波器**：在 RLS 的基礎上，進一步加入「狀態隨時間轉移的動態模型」。

三者層次關係為：

$$\text{LMS（單步瞬時梯度）} \subset \text{RLS（精確遞迴最小平方）} \subset \text{Kalman 濾波器（精確遞迴＋動態轉移）}$$

(來源: [[ch04-single-layer-regression]], p.7–8)

## 自我測驗 Q&A

> [!question] Q1. 為什麼在大資料集或即時應用中循序學習優於批次最小平方？
> [!answer]- 答案
> 批次方法需要儲存龐大的設計矩陣並計算 $O(M^3)$ 的反矩陣；循序學習每次只看一個樣本，記憶體常數級，且能配合串流資料即時自我調適。(來源: [[ch04-single-layer-regression]], p.7)

> [!question] Q2. LMS 演算法的權重更新量與什麼成正比？
> [!answer]- 答案
> 與預測誤差 $(t_n - \mathbf{w}^T\boldsymbol{\phi}_n)$ 以及該樣本的基底特徵向量 $\boldsymbol{\phi}_n$ 成正比。(來源: [[ch04-single-layer-regression]], p.7)

> [!question] Q3. LMS 與 Kalman 濾波器在最小平方架構上有什麼理論關聯？
> [!answer]- 答案
> LMS 是瞬時梯度的近似，RLS 是精確的遞迴更新，而 Kalman 濾波器則是加入系統動態狀態轉移方程式後的廣義 RLS。(來源: [[ch04-single-layer-regression]], p.7–8)

## 相關頁面

[[maximum-likelihood-regression]] · [[linear-basis-function-models]] · [[history-of-neural-networks]]
