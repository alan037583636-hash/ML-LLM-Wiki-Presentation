---
title: 正則化最小平方（第四章） (Regularized Least Squares - Chapter 4)
type: concept
tags: [regularization, ridge-regression, weight-decay, shrinkage, map]
sources: [ch04-single-layer-regression]
created: 2026-11-11
updated: 2026-11-11
---

# 正則化最小平方（第四章） (Regularized Least Squares - Chapter 4)

## 1. 目的與目標函數

📌講義：為控制多參數模型的過度擬合，在資料誤差函數 $E_D(\mathbf{w})$ 中加入正則化項 $E_W(\mathbf{w})$，最小化總目標函數：

$$E(\mathbf{w}) = E_D(\mathbf{w}) + \lambda E_W(\mathbf{w}) \tag{4.21}$$

其中 $\lambda$ 為正則化係數。取最簡單的權重平方和（L2 正則化）：

$$E_W(\mathbf{w}) = \frac{1}{2}\mathbf{w}^T\mathbf{w} = \frac{1}{2}\sum_j w_j^2 \tag{4.22}$$

總誤差函數為：

$$E(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \{t_n - \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2 + \frac{\lambda}{2}\mathbf{w}^T\mathbf{w} \tag{4.23}$$

在統計學中稱為**參數收縮 (parameter shrinkage)** 或**脊迴歸 (ridge regression)**；在神經網路中稱為**權重衰減 (weight decay)**。(來源: [[ch04-single-layer-regression]], p.8)

## 2. 閉式解推導

📌講義：因目標函數仍為 $\mathbf{w}$ 的二次函數，取梯度並令其為零：

$$\nabla_{\mathbf{w}} E(\mathbf{w}) = \boldsymbol{\Phi}^T(\boldsymbol{\Phi}\mathbf{w} - \mathbf{t}) + \lambda \mathbf{w} = (\boldsymbol{\Phi}^T\boldsymbol{\Phi} + \lambda \mathbf{I})\mathbf{w} - \boldsymbol{\Phi}^T\mathbf{t} = 0$$

解得閉式解：

$$\mathbf{w} = (\lambda \mathbf{I} + \boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{t} \tag{4.24}$$

此為無正則化最小平方解的直接推廣。(來源: [[ch04-single-layer-regression]], p.8)

## 3. 正則化的三種讀法（📌講義）

📌講義總結了理解正則化最小平方的三個關鍵維度：

1. **數值面 (Numerical)**：
   $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ 為半正定（特徵值 $\ge 0$）；加上 $\lambda \mathbf{I}$（$\lambda > 0$）後，矩陣最小特徵值 $\ge \lambda > 0$，保證永遠嚴格正定、可逆。這彻底消除了基底共線退化帶來的數值奇異病態。
2. **幾何面 (Geometric)**：
   在 $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ 的特徵分解基底下，每個方向的分量被乘上收縮因子 $\frac{\mu_j}{\mu_j + \lambda}$。曲率極小、資料資訊貧乏的方向（$\mu_j \to 0$）被壓制得最劇烈，防止參數沿這些方向暴增。
3. **貝氏面 (Bayesian)**：
   等價於對權重 $\mathbf{w}$ 引入零均值高斯先驗 $p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})$ 後的最大後驗機率 (MAP) 估計，其中正則化係數 $\lambda = \alpha \sigma^2$。(來源: [[ch04-single-layer-regression]], p.8)

## 自我測驗 Q&A

> [!question] Q1. 為什麼 L2 正則化能保證反矩陣計算絕對不會遭遇除以零的奇異性？
> [!answer]- 答案
> 因為 $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ 特徵值 $\mu_j \ge 0$，而 $\lambda \mathbf{I} + \boldsymbol{\Phi}^T\boldsymbol{\Phi}$ 的特徵值為 $\mu_j + \lambda \ge \lambda > 0$，恆為嚴格正定可逆。(來源: [[ch04-single-layer-regression]], p.8)

> [!question] Q2. 收縮因子 $\frac{\mu_j}{\mu_j + \lambda}$ 對特徵值 $\mu_j$ 極大與極小的方向分別有何影響？
> [!answer]- 答案
> 當 $\mu_j \gg \lambda$ 時收縮因子接近 1（資訊充分方向保留）；當 $\mu_j \ll \lambda$ 時收縮因子趨近 0（資訊不足、容易擬合雜訊的方向被強力壓縮）。(來源: [[ch04-single-layer-regression]], p.8)

## 相關頁面

[[maximum-likelihood-regression]] · [[regularization-weight-decay]] · [[bias-variance-tradeoff]]
