---
title: 多重輸出迴歸 (Multiple Output Regression)
type: concept
tags: [multiple-outputs, matrix-normal-equations, decoupling, multi-task]
sources: [ch04-single-layer-regression]
created: 2026-11-11
updated: 2026-11-11
---

# 多重輸出迴歸 (Multiple Output Regression)

## 1. 模型設定

📌講義：當需要同時預測 $K > 1$ 個目標變數時，記目標向量為 $\mathbf{t} = (t_1, \dots, t_K)^T$。最常見的做法是用同一組基底函數為所有分量建模：

$$\mathbf{y}(\mathbf{x}, \mathbf{W}) = \mathbf{W}^T \boldsymbol{\phi}(\mathbf{x}) \tag{4.25}$$

其中：
- $\mathbf{y}$ 是 $K$ 維行向量。
- $\mathbf{W}$ 是 $M \times K$ 維的參數矩陣。
- $\boldsymbol{\phi}(\mathbf{x})$ 是包含偏置 $\phi_0(\mathbf{x}) \equiv 1$ 的 $M$ 維行向量。

此結構對應於一個具備 $M$ 個輸入基底節點與 $K$ 個線性輸出節點的單層神經網路。(來源: [[ch04-single-layer-regression]], p.8)

## 2. 最大概似解與解耦性質

📌講義：假設目標向量在給定 $\mathbf{x}$ 下的條件分布為等向高斯 (isotropic Gaussian)：

$$p(\mathbf{t} \mid \mathbf{x}, \mathbf{W}, \sigma^2) = \mathcal{N}(\mathbf{t} \mid \mathbf{W}^T \boldsymbol{\phi}(\mathbf{x}), \sigma^2 \mathbf{I}) \tag{4.26}$$

給定 $N$ 筆訓練觀測，將目標向量排成 $N \times K$ 的目標矩陣 $\mathbf{T}$（其第 $n$ 列為 $\mathbf{t}_n^T$）。對數概似函數為：

$$\ln p(\mathbf{T} \mid \mathbf{X}, \mathbf{W}, \sigma^2) = -\frac{NK}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^N \|\mathbf{t}_n - \mathbf{W}^T\boldsymbol{\phi}(\mathbf{x}_n)\|^2 \tag{4.27}$$

對矩陣 $\mathbf{W}$ 求導並令其為零，解得最大概似解：

$$\mathbf{W}_{\text{ML}} = (\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T \mathbf{T} \tag{4.28}$$

若個別檢視第 $k$ 個目標變數的權重向量 $\mathbf{w}_k$（即 $\mathbf{W}$ 的第 $k$ 行）：

$$\mathbf{w}_k = (\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T \mathbf{t}_k = \boldsymbol{\Phi}^\dagger \mathbf{t}_k \tag{4.29}$$

其中 $\mathbf{t}_k$ 為包含所有訓練樣本第 $k$ 個分量的 $N$ 維向量。

📌講義（解耦核心性質）：
**多重輸出迴歸在不同目標變數之間完全解耦 (decouple)**。解 $K$ 個輸出變數的多元迴歸問題，在計算上只需算一次偽反矩陣 $\boldsymbol{\Phi}^\dagger$，再分別乘上各自的目標向量 $\mathbf{t}_k$ 即可。即使推廣到具有任意共變異數矩陣的一般高斯雜訊分布，回歸問題依然解耦為 $K$ 個獨立問題（因為均值參數的最大概似估計與協方差矩陣無關）。(來源: [[ch04-single-layer-regression]], p.9)

## 自我測驗 Q&A

> [!question] Q1. 為什麼多重輸出線性迴歸不需要針對 $K$ 個輸出分別計算 $K$ 次矩陣反矩陣？
> [!answer]- 答案
> 因為所有輸出共用同一組基底函數 $\boldsymbol{\phi}(\mathbf{x})$，設計矩陣 $\boldsymbol{\Phi}$ 與其偽反矩陣 $\boldsymbol{\Phi}^\dagger = (\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T$ 完全相同，只需計算一次即可供所有目標向量共用。(來源: [[ch04-single-layer-regression]], p.9)

> [!question] Q2. 各輸出變數之間的解耦性質在機率模型上有何直觀原因？
> [!answer]- 答案
> 參數 $\mathbf{W}$ 僅定義高斯分布的條件均值；在多變量高斯分布中，均值的最大概似估計與協方差結構解耦獨立。(來源: [[ch04-single-layer-regression]], p.9)

## 相關頁面

[[linear-basis-function-models]] · [[maximum-likelihood-regression]]
