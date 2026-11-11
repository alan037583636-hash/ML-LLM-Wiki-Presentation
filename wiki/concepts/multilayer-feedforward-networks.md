---
title: 多層前饋神經網路 (Multilayer Feedforward Networks)
type: concept
tags: [feedforward-networks, mlp, universal-approximation, activation-functions, symmetries]
sources: [ch06-deep-neural-networks]
created: 2026-11-11
updated: 2026-11-11
---

# 多層前饋神經網路 (Multilayer Feedforward Networks)

多層前饋神經網路（Multilayer Feedforward Networks），亦常稱為多層感知機（Multilayer Perceptron, MLP），是現代深度學習的基石架構。透過非線性激活函數的分層複合，網路擺脫了單層線性模型與固定基底函數的表現力限制。

---

## 1. 網路架構與矩陣代數形式 (Mathematical Formulation)

📌講義以兩層（單一隱藏層）神經網路為標準起點：設輸入向量為 $\mathbf{x} = (x_1, \dots, x_D)^T$，隱藏單元數為 $M$，輸出單元數為 $K$。

### 1.1 前向傳播傳遞式
1. **隱藏層前激活值 (Hidden Pre-activations)**：
   $$a_j = \sum_{i=1}^D w_{ji}^{(1)} x_i + w_{j0}^{(1)} \quad (j = 1, \dots, M)$$
   以矩陣形式表示：
   $$\mathbf{a} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$$
   其中 $\mathbf{W}^{(1)} \in \mathbb{R}^{M \times D}$ 為第一層權重矩陣，$\mathbf{b}^{(1)} \in \mathbb{R}^M$ 為偏置向量。

2. **隱藏層激活值 (Hidden Activations)**：
   透過逐元素非線性激活函數 $h(\cdot)$ 進行轉換：
   $$z_j = h(a_j) \implies \mathbf{z} = h(\mathbf{a})$$

3. **輸出層前激活值 (Output Pre-activations)**：
   $$a_k = \sum_{j=1}^M w_{kj}^{(2)} z_j + w_{k0}^{(2)} \quad (k = 1, \dots, K)$$
   以矩陣形式表示：
   $$\mathbf{a}_{\text{out}} = \mathbf{W}^{(2)} \mathbf{z} + \mathbf{b}^{(2)}$$

4. **最終網路輸出 (Output Activations)**：
   透過輸出層激活函數 $\sigma(\cdot)$：
   $$y_k = \sigma(a_k) \implies \mathbf{y} = \sigma(\mathbf{a}_{\text{out}})$$
   
整體兩層網路的複合函數為：
$$y_k(\mathbf{x}, \mathbf{w}) = \sigma\left( \sum_{j=1}^M w_{kj}^{(2)} h\left( \sum_{i=1}^D w_{ji}^{(1)} x_i + w_{j0}^{(1)} \right) + w_{k0}^{(2)} \right)$$
(來源: [[ch06-deep-neural-networks]], p.6–7)。

---

## 2. 隱藏單元激活函數族譜 (Hidden Activation Functions)

📌講義詳述了神經網路發展歷程中隱藏單元非線性轉換函數的演進 (來源: [[ch06-deep-neural-networks]], p.8)：

| 激活函數名稱 | 數學定義式 $h(a)$ | 導函數 $h'(a)$ | 輸出範圍 | 核心特徵與梯度表現 |
|---|---|---|---|---|
| **Logistic Sigmoid** | $\sigma(a) = \frac{1}{1 + e^{-a}}$ | $\sigma(a)(1 - \sigma(a))$ | $(0, 1)$ | 兩端飽和區導數趨近 0，極易引發**梯度消失** (Vanishing Gradient)；非零中心化 (Non-zero centered)。 |
| **雙曲正切 (Tanh)** | $\tanh(a) = \frac{e^a - e^{-a}}{e^a + e^{-a}}$ | $1 - \tanh^2(a)$ | $(-1, 1)$ | 零中心化（收斂通常快於 Sigmoid），但在 $|a|$ 較大時依然存在嚴重的梯度消失問題。 |
| **線性整流 (ReLU)** | $\max(0, a)$ | $\begin{cases} 1 & a > 0 \\ 0 & a < 0 \end{cases}$ | $[0, \infty)$ | 正半軸無飽和區（梯度恆為 1，緩解梯度消失）；計算極其高效；存在 **Dead ReLU** 缺陷。 |
| **平滑近似 (Softplus)** | $\ln(1 + e^a)$ | $\frac{1}{1 + e^{-a}} = \sigma(a)$ | $(0, \infty)$ | ReLU 的平滑可微近似，其導數即為 Logistic Sigmoid。 |
| **帶洩漏線性整流 (Leaky ReLU)** | $\max(\alpha a, a), \; 0 < \alpha \ll 1$ | $\begin{cases} 1 & a > 0 \\ \alpha & a < 0 \end{cases}$ | $(-\infty, \infty)$ | 在負半軸保留微小斜率 $\alpha$（如 $\alpha = 0.01$），避免神經元因長期負輸入而永久死亡。 |

---

## 3. 通用近似定理及其現實侷限 (Universal Approximation Theorem)

📌講義剖析了單層隱藏網路的理論極限與實踐鴻溝：

### 3.1 定理內容 (Cybenko, 1989; Hornik, 1991)
若激活函數 $h(a)$ 為連續、非平庸的非線性函數（如 Sigmoid），則具備單一隱藏層的前饋神經網路：
$$f(\mathbf{x}) = \sum_{j=1}^M w_j h(\mathbf{w}_j^T \mathbf{x} + b_j)$$
只要隱藏單元數量 $M$ 足夠大，就能以任意給定的精度 $\epsilon > 0$ 一致逼近緊湊集（Compact Set）$\mathcal{K} \subset \mathbb{R}^D$ 上的任何連續函數 (來源: [[ch06-deep-neural-networks]], p.9)。

### 3.2 理論與現實的落差 (Practical Limitations of Shallow Networks)
1. **無界寬度爆炸**：通用近似定理只證明了存在性，未對隱藏神經元數量 $M$ 給出實用上界。對於特定複雜函數（如奇偶性校驗 Parity Problem），淺層網路所需的單元數 $M$ 隨維度 $D$ 呈**指數級膨脹** ($M \sim \mathcal{O}(2^D)$)。
2. **缺乏學習保證**：定理並未保證基於有限資料與梯度下降最佳化演算法（如 SGD）能夠成功找到對應的最優權重。
3. **泛化性能低落**：淺層過寬網路傾向於對訓練集進行局部記憶，無法像深度結構那樣展現出自然的階層式組合泛化能力 (來源: [[ch06-deep-neural-networks]], p.10)。

---

## 4. 權重空間對稱性 (Weight-space Symmetries)

📌講義指出，神經網路的參數空間 $\mathbf{w}$ 具有強烈的多重對稱性，導致完全相同的輸入-輸出映射函數對應多個截然不同的參數點：

### 4.1 隱藏單元置換對稱性 (Permutation Symmetry)
對於擁有 $M$ 個隱藏單元的網路，任意置換這些隱藏單元的編號（同時置換連入該單元的權重向量 $\mathbf{w}_j^{(1)}$ 與連出該單元的權重向量 $\mathbf{w}_j^{(2)}$），網路的輸入-輸出對應函數 $y_k(\mathbf{x})$ 完全不變。共有 $M!$ 種不同的排列方式 (來源: [[ch06-deep-neural-networks]], p.11)。

### 4.2 正負號翻轉對稱性 (Sign-flip Symmetry)
若隱藏單元的激活函數為奇函數（Odd Function，如 $\tanh(-a) = -\tanh(a)$）：
同時將連入第 $j$ 個隱藏單元的所有權重與偏置變號（$\mathbf{w}_j^{(1)} \to -\mathbf{w}_j^{(1)}, b_j^{(1)} \to -b_j^{(1)}$），並將連出該單元的所有權重變號（$w_{kj}^{(2)} \to -w_{kj}^{(2)}$），兩次變號相乘抵消，網路輸出保持不變。由於 $M$ 個隱藏單元皆可獨立決定是否變號，共有 $2^M$ 種組合。

### 4.3 等價解空間大小
若一個兩層 $\tanh$ 網路找到了一組局部極值（Local Minima），則在權重空間中必然存在至少：
$$M! \cdot 2^M$$
個具有完全相同誤差函數值的等價局部極值點 (來源: [[ch06-deep-neural-networks]], p.11)。

---

## 自我測驗 Q&A

### Q1: 寫出標準兩層（單一隱藏層）神經網路的前向傳播矩陣代數表達式。
> [!answer]- 答案
> 前向傳播公式如下：
> 1. 隱藏層前激活值：$\mathbf{a} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$
> 2. 隱藏層激活值：$\mathbf{z} = h(\mathbf{a})$
> 3. 輸出層前激活值：$\mathbf{a}_{\text{out}} = \mathbf{W}^{(2)} \mathbf{z} + \mathbf{b}^{(2)}$
> 4. 網路最終輸出：$\mathbf{y} = \sigma(\mathbf{a}_{\text{out}})$

### Q2: 為什麼深度學習廣泛採用 ReLU 取代傳統的 Logistic Sigmoid 函數？
> [!answer]- 答案
> 1. **緩解梯度消失**：Sigmoid 函數在輸入絕對值較大時會進入飽和區，其導數趨近於 0，導致多層反向傳播時深層梯度呈指數衰減。而 ReLU 在正半軸 $a > 0$ 處導數恆為 1，有效解決梯度消失。
> 2. **計算極為高效**：ReLU 僅需進行簡單的閾值比較 $\max(0, a)$，避免了 Sigmoid 昂貴的指數運算 $\exp(-a)$。
> 3. **引發稀疏激活**：在負半軸輸出為 0，促成神經元的稀疏表徵（Sparse Representation）。

### Q3: 擁有 $M$ 個 $\tanh$ 隱藏單元的兩層前饋網路，在權重空間中存在多少個功能完全相同的等價解？其幾何成因為何？
> [!answer]- 答案
> 共有 $M! \cdot 2^M$ 個等價解。幾何成因為：
> 1. **置換對稱性**：隱藏單元的標號任意重排皆不改變輸出函數，共有 $M!$ 種排列。
> 2. **正負號翻轉對稱性**：因 $\tanh$ 為奇函數，任一隱藏單元的連入與連出權重同時乘以 $-1$ 時相互抵消，每個單元各有 2 種符號選擇，共 $2^M$ 種。兩者相乘得到 $M! 2^M$。

---

## 相關頁面
- [[ch06-deep-neural-networks|第六章 深度神經網路講義摘要]]
- [[curse-of-dimensionality-data-manifold|維度災難與資料流形]]
- [[deep-representations-and-transfer-learning|深度表徵與遷移學習]]
- [[error-functions-and-output-units|誤差函數與輸出單元]]
