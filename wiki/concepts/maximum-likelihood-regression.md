---
title: 迴歸的最大概似估計 (Maximum Likelihood Estimation in Regression)
type: concept
tags: [mle, regression, likelihood, gaussian-noise, normal-equations, pseudoinverse]
sources: [ch04-single-layer-regression]
created: 2026-11-11
updated: 2026-11-11
---

# 迴歸的最大概似估計 (Maximum Likelihood Estimation in Regression)

## 1. 高斯雜訊模型與概似函數

📌講義：假設目標變數 $t$ 由確定性函數 $y(\mathbf{x}, \mathbf{w})$ 加上高斯雜訊 $\epsilon$ 給出：

$$t = y(\mathbf{x}, \mathbf{w}) + \epsilon \tag{4.6}$$

其中 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 為零均值、變異數 $\sigma^2$ 的高斯隨機變數。因此條件分布為：

$$p(t \mid \mathbf{x}, \mathbf{w}, \sigma^2) = \mathcal{N}(t \mid y(\mathbf{x}, \mathbf{w}), \sigma^2) \tag{4.7}$$

給定訓練集 $\mathbf{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ 與目標值向量 $\mathbf{t} = (t_1, \dots, t_N)^T$，假設各樣本獨立同分布 (i.i.d.)，概似函數為：

$$p(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^2) = \prod_{n=1}^N \mathcal{N}(t_n \mid \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n), \sigma^2) \tag{4.8}$$

取對數得對數概似函數：

$$\ln p(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^2) = -\frac{N}{2} \ln \sigma^2 - \frac{N}{2} \ln(2\pi) - \frac{1}{\sigma^2} E_D(\mathbf{w}) \tag{4.9}$$

其中平方和誤差函數為：

$$E_D(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \{t_n - \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2 \tag{4.10}$$

(來源: [[ch04-single-layer-regression]], p.4–5)

📌講義（為什麼是平方和？）：
式 (4.9) 的前兩項與 $\mathbf{w}$ 無關。因此，**在高斯雜訊假設下，最大化概似函數等價於最小化平方和誤差**。平方項並非任意選擇，而是高斯機率密度指數項 $-\frac{(t-y)^2}{2\sigma^2}$ 的直接數學後果。(來源: [[ch04-single-layer-regression]], p.5)

## 2. 參數 $\mathbf{w}$ 的最大概似解

📌講義：對數概似對 $\mathbf{w}$ 取梯度：

$$\nabla_{\mathbf{w}} \ln p(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^2) = \frac{1}{\sigma^2}\sum_{n=1}^N \{t_n - \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\} \boldsymbol{\phi}(\mathbf{x}_n)^T \tag{4.11}$$

令梯度為零：

$$0 = \sum_{n=1}^N t_n \boldsymbol{\phi}(\mathbf{x}_n)^T - \mathbf{w}^T \sum_{n=1}^N \boldsymbol{\phi}(\mathbf{x}_n)\boldsymbol{\phi}(\mathbf{x}_n)^T \tag{4.12}$$

解出 $\mathbf{w}$ 得最大概似解：

$$\mathbf{w}_{\text{ML}} = (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{t} \tag{4.13}$$

此即最小平方問題的**正規方程 (normal equations)**。其中 $\boldsymbol{\Phi} \in \mathbb{R}^{N \times M}$ 為**設計矩陣 (design matrix)**，$\boldsymbol{\Phi}_{nj} = \phi_j(\mathbf{x}_n)$：

$$\boldsymbol{\Phi} = \begin{pmatrix} \phi_0(\mathbf{x}_1) & \phi_1(\mathbf{x}_1) & \cdots & \phi_{M-1}(\mathbf{x}_1) \\ \phi_0(\mathbf{x}_2) & \phi_1(\mathbf{x}_2) & \cdots & \phi_{M-1}(\mathbf{x}_2) \\ \vdots & \vdots & \ddots & \vdots \\ \phi_0(\mathbf{x}_N) & \phi_1(\mathbf{x}_N) & \cdots & \phi_{M-1}(\mathbf{x}_N) \end{pmatrix} \tag{4.14}$$

量

$$\boldsymbol{\Phi}^\dagger \equiv (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \tag{4.15}$$

稱為 $\boldsymbol{\Phi}$ 的 **Moore–Penrose 偽反矩陣 (pseudo-inverse)**。當 $\boldsymbol{\Phi}$ 為可逆方陣時，$\boldsymbol{\Phi}^\dagger \equiv \boldsymbol{\Phi}^{-1}$。(來源: [[ch04-single-layer-regression]], p.5)

📌講義（矩陣形式重推）：
把誤差函數寫成矩陣形式：$E_D(\mathbf{w}) = \frac{1}{2}(\mathbf{t} - \boldsymbol{\Phi}\mathbf{w})^T(\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}) = \frac{1}{2}\mathbf{t}^T\mathbf{t} - \mathbf{w}^T\boldsymbol{\Phi}^T\mathbf{t} + \frac{1}{2}\mathbf{w}^T\boldsymbol{\Phi}^T\boldsymbol{\Phi}\mathbf{w}$。
利用向量矩陣微分公式，梯度為 $\nabla_{\mathbf{w}}E_D = -\boldsymbol{\Phi}^T\mathbf{t} + \boldsymbol{\Phi}^T\boldsymbol{\Phi}\mathbf{w} = 0$，立得式 (4.13)。(來源: [[ch04-single-layer-regression]], p.6)

## 3. 偏置參數 $w_0$ 的解析角色

📌講義：若把偏置 $w_0$ 明確自總和中抽出：

$$E_D(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \left\{t_n - w_0 - \sum_{j=1}^{M-1} w_j \phi_j(\mathbf{x}_n)\right\}^2 \tag{4.16}$$

令 $\frac{\partial E_D}{\partial w_0} = 0$ 解得：

$$w_0 = \bar{t} - \sum_{j=1}^{M-1} w_j \bar{\phi}_j \tag{4.17}$$

其中 $\bar{t} = \frac{1}{N}\sum_{n=1}^N t_n$，$\bar{\phi}_j = \frac{1}{N}\sum_{n=1}^N \phi_j(\mathbf{x}_n)$。
**直觀意義**：偏置 $w_0$ 的作用在於補償目標變數平均值與基底函數加權平均值之間的差距。(來源: [[ch04-single-layer-regression]], p.6)

## 4. 雜訊變異數 $\sigma^2$ 的最大概似解

📌講義：對數概似 (4.9) 對精度 $\beta = 1/\sigma^2$ 求導並令其為零，解得：

$$\sigma_{\text{ML}}^2 = \frac{1}{N}\sum_{n=1}^N \{t_n - \mathbf{w}_{\text{ML}}^T \boldsymbol{\phi}(\mathbf{x}_n)\}^2 \tag{4.18}$$

即雜訊變異數的最大概似估計值等於**殘差變異數**。此估計為有偏估計（分母為 $N$ 而非自由度 $N - M$）；但在深度學習大規模資料 $N \gg M$ 下，此偏差可忽略不計。(來源: [[ch04-single-layer-regression]], p.6)

## 自我測驗 Q&A

> [!question] Q1. 為什麼說最大化高斯對數概似「自然導出」最小平方？
> [!answer]- 答案
> 單變量高斯的指數項為 $-\frac{(t - y)^2}{2\sigma^2}$。取對數後，與參數有關的項剛好就是負的平方和誤差；常數項與變異數項不影響 $\mathbf{w}$ 的極值位置。(來源: [[ch04-single-layer-regression]], p.4–5)

> [!question] Q2. 設計矩陣 $\boldsymbol{\Phi}$ 與其偽反矩陣 $\boldsymbol{\Phi}^\dagger$ 的維度各為何？
> [!answer]- 答案
> $\boldsymbol{\Phi} \in \mathbb{R}^{N \times M}$，每一列為一個樣本的基底特徵向量；$\boldsymbol{\Phi}^\dagger = (\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T \in \mathbb{R}^{M \times N}$。(來源: [[ch04-single-layer-regression]], p.5)

> [!question] Q3. 偏置參數 $w_0$ 在最大概似解中的幾何/物理意義是什麼？
> [!answer]- 答案
> 它精確吸收並補償了訓練目標值平均 $\bar{t}$ 與非偏置基底加權平均 $\sum_{j=1}^{M-1} w_j \bar{\phi}_j$ 之間的位移。(來源: [[ch04-single-layer-regression]], p.6)

## 相關頁面

[[linear-basis-function-models]] · [[geometry-of-least-squares]] · [[regularized-least-squares-ch04]] · [[least-squares-normal-equations]]
