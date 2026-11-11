---
title: Probit 迴歸與典則連結函數 (Probit Regression and Canonical Link Functions)
type: concept
tags: [probit-regression, canonical-link, generalized-linear-models, link-function, exponential-family]
sources: [ch05-single-layer-classification]
created: 2026-11-11
updated: 2026-11-11
---

# Probit 迴歸與典則連結函數 (Probit Regression and Canonical Link Functions)

## 1. 雜訊閾值模型與 Probit 迴歸

📌講義：並非所有問題都適合邏輯 S 形函數。考慮一個基於**隨機雜訊閾值**的二元分類機制：
對每個輸入計算線性值 $a_n = \mathbf{w}^T\boldsymbol{\phi}_n$；若 $a_n$ 超過隨機閾值 $\theta \sim p(\theta)$，則指派標籤 $t_n = 1$，否則為 $0$：

$$\begin{cases} t_n = 1, & a_n \ge \theta \\ t_n = 0, & a_n < \theta \end{cases} \tag{5.66}$$

此時對應的激活函數即為閾值分布的**累積分布函數 (CDF)**：

$$f(a) = \int_{-\infty}^a p(\theta) \, d\theta \tag{5.67}$$

### Probit 函數
若假設閾值雜訊服從標準常態分布 $\theta \sim \mathcal{N}(0, 1)$，其激活函數為：

$$\Phi(a) = \int_{-\infty}^a \mathcal{N}(\theta \mid 0, 1) \, d\theta \tag{5.68}$$

此模型稱為 **Probit 迴歸 (Probit Regression)**。它可透過數值套件中的誤差函數 $\text{erf}(a)$ 快速計算：

$$\Phi(a) = \frac{1}{2}\left\{1 + \frac{1}{\sqrt{2}}\text{erf}(a)\right\} \tag{5.69}$$

### 對離群值的敏感度比較（📌講義）
- **邏輯迴歸 S 形函數**：尾部衰減速率為 $\exp(-|x|)$（較長尾）。
- **Probit 激活函數**：高斯尾部衰減速率為 $\exp(-x^2)$（極速衰減）。
- **影響**：Probit 模型的尾部極度陡峭，若輸入端出現噪聲或標籤被誤標的離群值，Probit 模型對離群值的敏感度與被拉偏程度顯著高於邏輯迴歸。(來源: [[ch05-single-layer-classification]], p.20)

## 2. 典則連結函數理論 (Canonical Link Functions)

📌講義：回顧前幾節，我們發現無論是線性迴歸、邏輯迴歸或多類別 Softmax 迴歸，其對數概死函數的梯度都驚人地呈現出完全一致的簡潔形式：

$$\nabla_{\mathbf{w}} E = \sum_{n=1}^N (y_n - t_n) \boldsymbol{\phi}_n$$

這並非巧合，而是**廣義線性模型 (GLM)** 的一個一般性理論定理。

### 數學推導
📌講義：設目標變數 $t$ 來自受限形式的指數族分布：

$$p(t \mid \eta, s) = \frac{1}{s} h\left(\frac{t}{s}\right) g(\eta) \exp\left\{\frac{\eta t}{s}\right\} \tag{5.70}$$

其中 $\eta$ 為自然參數，$s$ 為尺度參數。目標變數的條件期望值為 $y \equiv \mathbb{E}[t \mid \eta] = -s \frac{d}{d\eta}\ln g(\eta)$。令反函數關係為 $\eta = \psi(y)$。
在廣義線性模型中，預測值 $y = f(a) = f(\mathbf{w}^T\boldsymbol{\phi})$，其中 $f$ 為激活函數，$f^{-1}$ 為統計學的**連結函數 (link function)**。
對數概似函數對權重 $\mathbf{w}$ 求導：

$$\nabla_{\mathbf{w}} \ln p = \sum_{n=1}^N \frac{1}{s}(t_n - y_n) \psi'(y_n) f'(a_n) \boldsymbol{\phi}_n$$

若我們將激活函數的反函數（連結函數）特意選取為自然參數對均值的映射：

$$f^{-1}(y) = \psi(y) \tag{5.72}$$

則滿足 $f'(a)\psi'(y) \equiv 1$。兩項非線性導數相互抵消，梯度完美化簡為：

$$\nabla E(\mathbf{w}) = \frac{1}{s}\sum_{n=1}^N (y_n - t_n) \boldsymbol{\phi}_n \tag{5.73}$$

滿足此條件的連結函數 $f^{-1}$ 即稱為**典則連結函數 (canonical link function)**。(來源: [[ch05-single-layer-classification]], p.21–22)

## 3. 三組「天生一對」（📌講義）

📌講義總結：指數族目標分布、激活函數與損失函數之間存在自然的數學配對：

| 目標變數分布（指數族） | 典則激活函數 $f(a)$ | 對應誤差函數（負對數概似） | 梯度形式 $\nabla E$ |
|---|---|---|---|
| **高斯分布 (Gaussian)** | 恆等函數 $f(a) = a$ | 平方和誤差 (Sum-of-squares) | $\sum_n (y_n - t_n)\boldsymbol{\phi}_n$ |
| **伯努利分布 (Bernoulli)** | 邏輯 S 形函數 $\sigma(a)$ | 二元交叉熵 (Binary Cross-Entropy) | $\sum_n (y_n - t_n)\boldsymbol{\phi}_n$ |
| **多項分布 (Multinomial)** | Softmax 函數 | 多類別交叉熵 (Multiclass Cross-Entropy) | $\sum_n (y_{nj} - t_{nj})\boldsymbol{\phi}_n$ |

**深度學習意義**：深度神經網路在輸出層的設計（分類任務最後一層接 Softmax + 交叉熵；迴歸任務接線性單元 + 平方誤差），其數理根基正是此處的典則連結函數理論。(來源: [[ch05-single-layer-classification]], p.22)

## 自我測驗 Q&A

> [!question] Q1. Probit 迴歸的激活函數是如何從隨機閾值機制推導出來的？
> [!answer]- 答案
> 假設判定門檻 $\theta$ 服從標準常態分布 $\mathcal{N}(0, 1)$，則輸入超過門檻的機率即為標準常態分布的累積分布函數 $\Phi(a)$。(來源: [[ch05-single-layer-classification]], p.20)

> [!question] Q2. 為什麼 Probit 模型比邏輯迴歸對離群值更加敏感？
> [!answer]- 答案
> Probit 函數的尾部以 $\exp(-x^2)$ 速度極度劇烈衰減，而 S 形函數以 $\exp(-x)$ 較平緩衰減，因此極端樣本在 Probit 下會受到更劇烈的扭曲拉偏。(來源: [[ch05-single-layer-classification]], p.20)

> [!question] Q3. 什麼是「典則連結函數」？它有何核心數學優勢？
> [!answer]- 答案
> 將連結函數 $f^{-1}$ 選取為指數族自然參數 $\eta$ 與條件均值 $y$ 的映射關係，使得激活函數導數與概似導數互為倒數相消，確保誤差函數對參數的梯度維持最簡潔的「誤差 $\times$ 特徵」形式。(來源: [[ch05-single-layer-classification]], p.21–22)

## 相關頁面

[[discriminative-classifiers-logistic-regression]] · [[linear-basis-function-models]] · [[maximum-likelihood-regression]] · [[ch05-single-layer-classification]]
