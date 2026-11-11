---
title: 誤差函數與輸出單元 (Error Functions and Output Units)
type: concept
tags: [error-functions, output-units, maximum-likelihood, cross-entropy, gradients]
sources: [ch06-deep-neural-networks]
created: 2026-11-11
updated: 2026-11-11
---

# 誤差函數與輸出單元 (Error Functions and Output Units)

在統計學習與深度神經網路中，**損失函數（Loss / Error Function）絕非隨意湊合的啟發式公式**，而是從機率分布與**最大概似估計（Maximum Likelihood Estimation, MLE）**中自然推導而出的必然結果。

---

## 1. 典則配對架構 (The Canonical Pairing Framework)

📌講義指出，神經網路的輸出單元與損失函數構成了一套高度統一的數學配對結構。輸出單元的**前激活值（Pre-activation）** $a_k$ 透過特定的激活函數 $\sigma(a_k)$ 映射為條件機率分布的參數，而誤差函數則由該分布的**負對數概似（Negative Log-Likelihood, NLL）**決定 (來源: [[ch06-deep-neural-networks]], p.17–19)：

| 學習任務類型 | 假設條件機率分布 $p(\mathbf{t} \mid \mathbf{x})$ | 輸出單元激活函數 $y_k = \sigma(a_k)$ | 負對數概似損失函數 $E(\mathbf{w})$ | 前激活導數 $\frac{\partial E_n}{\partial a_k}$ |
|---|---|---|---|---|
| **迴歸分析 (Regression)** | 單變量高斯分布 $\mathcal{N}(y(\mathbf{x}), \beta^{-1})$ | **線性 (Linear / Identity)**: $y_k = a_k$ | **平方和誤差 (SSE)**: $\frac{1}{2}\sum_n (y_n - t_n)^2$ | $y_{nk} - t_{nk}$ |
| **二元分類 (Binary Classification)** | 白努利分布 $\text{Bern}(t \mid y)$ | **Logistic Sigmoid**: $y = \sigma(a) = \frac{1}{1 + e^{-a}}$ | **二元交叉熵 (BCE)**: $-\sum_n [t_n \ln y_n + (1-t_n)\ln(1-y_n)]$ | $y_n - t_n$ |
| **多重獨立二元分類 (Multi-label)** | 獨立白努利乘積 $\prod_{k=1}^K \text{Bern}(t_k \mid y_k)$ | **獨立 Sigmoid**: $y_k = \sigma(a_k)$ | **多重二元交叉熵**: $-\sum_n \sum_k [t_{nk}\ln y_{nk} + (1-t_{nk})\ln(1-y_{nk})]$ | $y_{nk} - t_{nk}$ |
| **多類別分類 (Multiclass)** | 類別分布 $\text{Cat}(\mathbf{t} \mid \mathbf{y})$ (One-hot) | **Softmax**: $y_k = \frac{\exp(a_k)}{\sum_j \exp(a_j)}$ | **多類別交叉熵 (CCE)**: $-\sum_n \sum_k t_{nk} \ln y_{nk}$ | $y_{nk} - t_{nk}$ |

---

## 2. 負對數概似推導 (Derivations from Negative Log-Likelihood)

### 2.1 高斯迴歸與平方和誤差 (Regression)
給定獨立同分布樣本 $\{(\mathbf{x}_n, \mathbf{t}_n)\}$，假設目標值服從高斯雜訊分布：
$$p(\mathbf{t} \mid \mathbf{x}, \mathbf{w}, \beta) = \left(\frac{\beta}{2\pi}\right)^{D/2} \exp\left(-\frac{\beta}{2}\|\mathbf{t} - \mathbf{y}(\mathbf{x}, \mathbf{w})\|^2\right)$$
其負對數概似為：
$$E(\mathbf{w}) = -\ln p(\mathcal{D} \mid \mathbf{w}, \beta) = \frac{\beta}{2} \sum_{n=1}^N \|\mathbf{y}(\mathbf{x}_n, \mathbf{w}) - \mathbf{t}_n\|^2 - \frac{ND}{2}\ln \beta + \frac{ND}{2}\ln(2\pi)$$
固定精度 $\beta$ 時，最大化概似完全等價於最小化**平方和誤差函數 (Sum-of-Squares Error)** (來源: [[ch06-deep-neural-networks]], p.17)。

### 2.2 二元分類與二元交叉熵 (Binary Cross-Entropy)
設目標標籤 $t \in \{0, 1\}$，條件機率為 $p(t=1 \mid \mathbf{x}) = y = \sigma(a)$。單筆樣本的概似為：
$$p(t \mid \mathbf{x}) = y^t (1 - y)^{1 - t}$$
取負對數得到二元交叉熵：
$$E_n = - [t \ln y + (1 - t) \ln(1 - y)]$$
(來源: [[ch06-deep-neural-networks]], p.18)。

### 2.3 多類別分類與 Softmax 交叉熵 (Multiclass Cross-Entropy)
設類別目標為 1-of-K One-hot 向量 $\mathbf{t} = (t_1, \dots, t_K)^T$，$t_k \in \{0, 1\}, \sum_k t_k = 1$。
條件機率由 Softmax 輸出：
$$p(\mathbf{t} \mid \mathbf{x}) = \prod_{k=1}^K y_k^{t_k}, \quad y_k = \frac{\exp(a_k)}{\sum_{j=1}^K \exp(a_j)}$$
取負對數得到多類別交叉熵：
$$E_n = -\sum_{k=1}^K t_k \ln y_k = -\sum_{k=1}^K t_k \left[ a_k - \ln \sum_{j=1}^K \exp(a_j) \right]$$
(來源: [[ch06-deep-neural-networks]], p.19)。

---

## 3. 前激活梯度的極簡統一定理 (The Universal Pre-activation Gradient)

📌講義強調了一個貫穿整個深度學習最佳化的重大數學定理：
無論上述哪一種任務，**只要損失函數是該分布的負對數概似，且輸出單元採用對應的典則連結函數，則誤差對輸出層前激活值 $a_k$ 的偏導數恆具備完全相同的極簡形式**：
$$\frac{\partial E_n}{\partial a_k} = y_k - t_k$$
(來源: [[ch06-deep-neural-networks]], p.20)。

### 3.1 數學證明（以 Softmax 交叉熵為例）
Softmax 對前激活值的導數為：
$$\frac{\partial y_j}{\partial a_k} = y_j (\delta_{jk} - y_k)$$
利用連鎖律求 $E_n$ 對 $a_k$ 的導數：
$$\frac{\partial E_n}{\partial a_k} = \sum_{j=1}^K \frac{\partial E_n}{\partial y_j} \frac{\partial y_j}{\partial a_k} = \sum_{j=1}^K \left(-\frac{t_j}{y_j}\right) y_j (\delta_{jk} - y_k)$$
化簡展開：
$$\frac{\partial E_n}{\partial a_k} = -\sum_{j=1}^K t_j (\delta_{jk} - y_k) = -t_k + y_k \sum_{j=1}^K t_j$$
因為 $\mathbf{t}$ 為 One-hot 編碼，$\sum_{j=1}^K t_j = 1$，直接得到：
$$\frac{\partial E_n}{\partial a_k} = y_k - t_k \quad \blacksquare$$

### 3.2 工程重要性：抗梯度消失 (Preventing Gradient Saturation)
若在分類任務中錯誤地使用「平方和誤差」搭配 Sigmoid/Softmax 輸出：
$$E_n = \frac{1}{2}(y - t)^2 \implies \frac{\partial E_n}{\partial a} = (y - t) \cdot \sigma'(a) = (y - t) y (1 - y)$$
當模型輸出極度錯誤（例如 $t = 1$ 但 $y \to 0$，此時 $a \ll 0$），$\sigma'(a) \to 0$ 會使梯度直接消失，模型無法從錯誤中學習！
而採用交叉熵時，$\sigma'(a)$ 的項被對數概似的倒數精確抵消，梯度直接為 $(y - t)$：**預測誤差越大，梯度推動力越強，從根本上避免了輸出飽和導致的訓練停滯** (來源: [[ch06-deep-neural-networks]], p.20)。

---

## 自我測驗 Q&A

### Q1: 請寫出多類別分類中 Softmax 輸出搭配多類別交叉熵損失時，損失函數對輸出前激活值 $a_k$ 的偏導函數。
> [!answer]- 答案
> 偏導數為 $\frac{\partial E_n}{\partial a_k} = y_k - t_k$。其中 $y_k$ 為 Softmax 的預測機率輸出，$t_k$ 為 One-hot 目標標籤的第 $k$ 個分量。

### Q2: 為什麼在多標籤分類（Multi-label Classification）中不能使用 Softmax，而必須使用獨立的 Sigmoid 輸出？
> [!answer]- 答案
> Softmax 強制所有類別的輸出機率和為 1（$\sum_k y_k = 1$），隱含各類別間彼此互斥（Mutual Exclusion）的假設。在多標籤任務中，一張圖片可同時包含「貓」與「狗」，各類別標籤並非互斥而是彼此獨立的事件。因此必須將每個輸出單元視為獨立的白努利分佈，各自使用獨立的 Sigmoid 激活函數配對獨立二元交叉熵損失。

### Q3: 為什麼分類任務若誤用「平方和誤差（MSE）」會引發嚴重的梯度消失，而交叉熵不會？
> [!answer]- 答案
> 若使用平方誤差搭配 Sigmoid，梯度為 $\frac{\partial E}{\partial a} = (y - t) \sigma'(a) = (y - t) y (1 - y)$。當預測嚴重錯誤（如 $t=1, a \ll 0 \implies y \approx 0$）時，$\sigma'(a) \to 0$，使得梯度幾乎為 0，模型無法修正錯誤。反之，交叉熵損失的對數導數項剛好抵消了 $\sigma'(a)$，使得 $\frac{\partial E}{\partial a} = y - t$，誤差越大則推動力越強，不會產生飽和停滯。

---

## 相關頁面
- [[ch06-deep-neural-networks|第六章 深度神經網路講義摘要]]
- [[multilayer-feedforward-networks|多層前饋神經網路]]
- [[mixture-density-networks|混合密度網路]]
- [[cross-entropy-loss|交叉熵損失函數]]
