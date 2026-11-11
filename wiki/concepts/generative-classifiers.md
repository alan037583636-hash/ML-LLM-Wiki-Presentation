---
title: 生成式分類器 (Generative Classifiers)
type: concept
tags: [generative-models, lda, qda, gaussian-class-conditional, naive-bayes, exponential-family, softmax, sigmoid]
sources: [ch05-single-layer-classification]
created: 2026-11-11
updated: 2026-11-11
---

# 生成式分類器 (Generative Classifiers)

## 1. 後驗機率的廣義線性形式由來

📌講義：在生成式取徑中，我們對類別條件密度 $p(\mathbf{x} \mid \mathcal{C}_k)$ 與先驗 $p(\mathcal{C}_k)$ 建模，再透過貝氏定理推導後驗機率：

### 兩類別：邏輯 S 形函數 (Logistic Sigmoid)
$$p(\mathcal{C}_1 \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \mathcal{C}_1)p(\mathcal{C}_1)}{p(\mathbf{x} \mid \mathcal{C}_1)p(\mathcal{C}_1) + p(\mathbf{x} \mid \mathcal{C}_2)p(\mathcal{C}_2)} = \frac{1}{1 + \exp(-a)} = \sigma(a) \tag{5.31}$$

其中

$$a = \ln \frac{p(\mathbf{x} \mid \mathcal{C}_1)p(\mathcal{C}_1)}{p(\mathbf{x} \mid \mathcal{C}_2)p(\mathcal{C}_2)} \tag{5.32}$$

稱為 **對數勝算 (log odds)** 或 **logit 函數**（通訊中的對數概似比 LLR）。$\sigma(a) = \frac{1}{1 + e^{-a}}$ 滿足對稱性 $\sigma(-a) = 1 - \sigma(a)$。(來源: [[ch05-single-layer-classification]], p.12)

### 多類別：正規化指數 (Softmax)
對 $K > 2$ 類：

$$p(\mathcal{C}_k \mid \mathbf{x}) = \frac{\exp(a_k)}{\sum_j \exp(a_j)}, \quad a_k = \ln [p(\mathbf{x} \mid \mathcal{C}_k)p(\mathcal{C}_k)] \tag{5.36}$$

此即平滑版的最大值函數 **Softmax 函數**。(來源: [[ch05-single-layer-classification]], p.12–13)

## 2. 高斯類別條件密度推導（LDA vs. QDA）

📌講義：假設連續輸入 $\mathbf{x}$ 的類別條件密度為高斯分布：

$$p(\mathbf{x} \mid \mathcal{C}_k) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left\{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)\right\} \tag{5.38}$$

### 核心推導：共用共變異數矩陣 $\boldsymbol{\Sigma}$ 導出線性邊界 (LDA)
代入式 (5.32)：
$$a = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_1) + \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_2)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_2) + \ln \frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)}$$
將二次式展開：**二次項 $-\frac{1}{2}\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x}$ 完全抵銷**，僅剩下 $\mathbf{x}$ 的一次線性項！
整理得：

$$a = \mathbf{w}^T\mathbf{x} + w_0 \tag{5.39}$$

其中：

$$\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) \tag{5.40}$$

$$w_0 = -\frac{1}{2}\boldsymbol{\mu}_1^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 + \frac{1}{2}\boldsymbol{\mu}_2^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2 + \ln \frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)} \tag{5.41}$$

**結論**：因為二次項抵銷，後驗機率 $p(\mathcal{C}_1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + w_0)$ 是對輸入 $\mathbf{x}$ 的廣義線性模型，決策邊界為超平面（線性判別分析 Linear Discriminant Analysis, LDA）。先驗機率僅影響偏置 $w_0$。(來源: [[ch05-single-layer-classification]], p.13–14)

📌講義（二次判別 QDA）：
若各類別擁有各自的共變異數矩陣 $\boldsymbol{\Sigma}_k \ne \boldsymbol{\Sigma}_j$，二次項無法抵銷，所得之 $a(\mathbf{x})$ 包含 $\mathbf{x}^T\boldsymbol{\Sigma}_1^{-1}\mathbf{x}$ 等二次項，決策邊界為二次曲面（二次判別分析 Quadratic Discriminant Analysis, QDA）。(來源: [[ch05-single-layer-classification]], p.14)

## 3. 最大概似估計推導

📌講義：給定標籤 $t_n \in \{0, 1\}$，最大概似解為：
1. **先驗機率**：$\pi = p(\mathcal{C}_1) = \frac{N_1}{N}$（$\mathcal{C}_1$ 樣本所佔比例）。
2. **類別均值**：$\boldsymbol{\mu}_1 = \frac{1}{N_1}\sum_{n \in \mathcal{C}_1} \mathbf{x}_n$，$\boldsymbol{\mu}_2 = \frac{1}{N_2}\sum_{n \in \mathcal{C}_2} \mathbf{x}_n$。
3. **共用共變異數矩陣**：$\boldsymbol{\Sigma} = \frac{N_1}{N}\mathbf{S}_1 + \frac{N_2}{N}\mathbf{S}_2$（各類別樣本協方差矩陣的加權平均）。(來源: [[ch05-single-layer-classification]], p.14–15)

## 4. 離散特徵（樸素貝氏）與指數族一般性

📌講義：
- **樸素貝氏 (Naive Bayes)**：針對二元特徵 $x_i \in \{0, 1\}$，假設特徵在給定類別下條件獨立：$p(\mathbf{x} \mid \mathcal{C}_k) = \prod_{i=1}^D \mu_{ki}^{x_i}(1 - \mu_{ki})^{1 - x_i}$。代入式 (5.37) 後，$a_k(\mathbf{x})$ 依然精確化為輸入 $x_i$ 的線性函數。
- **指數族一般性 (Exponential Family)**：只要類別條件密度屬於共用尺度參數的指數族分布，其後驗機率**必然皆具有 S 形／Softmax 作用在線性函數上的數學形式**。(來源: [[ch05-single-layer-classification]], p.15–16)

## 自我測驗 Q&A

> [!question] Q1. 為什麼高斯條件分布在兩類共用共變異數矩陣時，決策面是線性的？
> [!answer]- 答案
> 在計算對數勝算比 $a(\mathbf{x})$ 時，高斯指數中的二次項 $-\frac{1}{2}\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x}$ 彼此完全抵銷，僅剩下一次項 $\mathbf{w}^T\mathbf{x}$。(來源: [[ch05-single-layer-classification]], p.13–14)

> [!question] Q2. LDA 中改變類別先驗機率 $p(\mathcal{C}_k)$ 會如何改變決策邊界？
> [!answer]- 答案
> 權重向量 $\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$ 不變（法向量不變），先驗項僅進入偏置 $w_0$，因此決策面只會平行移動。(來源: [[ch05-single-layer-classification]], p.14)

> [!question] Q3. 樸素貝氏分類器的核心假設是什麼？
> [!answer]- 答案
> 假設在給定類別的條件下，各個輸入特徵之間彼此條件獨立。(來源: [[ch05-single-layer-classification]], p.15)

## 相關頁面

[[classification-approaches]] · [[linear-discriminant-functions]] · [[discriminative-classifiers-logistic-regression]]
