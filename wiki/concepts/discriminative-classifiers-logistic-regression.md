---
title: 判別式分類器：邏輯迴歸與 Softmax 迴歸 (Discriminative Classifiers: Logistic and Softmax Regression)
type: concept
tags: [discriminative-models, logistic-regression, softmax-regression, cross-entropy, gradient-descent, irls, overfitting]
sources: [ch05-single-layer-classification]
created: 2026-11-11
updated: 2026-11-11
---

# 判別式分類器：邏輯迴歸與 Softmax 迴歸 (Discriminative Classifiers: Logistic and Softmax Regression)

## 1. 廣義線性模型與參數優勢

📌講義：生成式模型的理論指出，對於廣泛的指數族條件密度，後驗機率皆為 S 形（兩類）或 Softmax（多類）作用在線性函數上的形式。
**判別式取徑**直接假設此函數形式，跳過密度估計，直接以最大概似法決定參數 $\mathbf{w}$。

### 參數量級的對比（$M$ 維特徵空間）
- **高斯生成式模型**：需估計均值、共變異數矩陣與先驗，參數量為 $M(M+5)/2 + 1$，隨特徵維度 $M$ 呈**平方級增長**。
- **邏輯迴歸 (Logistic Regression)**：僅需估計單一權重向量 $\mathbf{w}$，參數量為 $M$，隨特徵維度**線性增長**。
當特徵維度很高時，判別式模型在參數效率與避免過度擬合上具有壓倒性優勢。(來源: [[ch05-single-layer-classification]], p.16–18)

## 2. 邏輯迴歸與二元交叉熵

📌講義：對於兩類分類問題，模型後驗表示為：

$$p(\mathcal{C}_1 \mid \boldsymbol{\phi}) = y(\boldsymbol{\phi}) = \sigma(\mathbf{w}^T\boldsymbol{\phi}) \tag{5.54}$$

其中 $\sigma(a) = \frac{1}{1 + e^{-a}}$，其導數具有優美的自身表示形式：

$$\frac{d\sigma}{da} = \sigma(1 - \sigma) \tag{5.55}$$

給定二元標籤 $t_n \in \{0, 1\}$，概似函數為：

$$p(\mathbf{t} \mid \mathbf{w}) = \prod_{n=1}^N y_n^{t_n} (1 - y_n)^{1 - t_n} \tag{5.56}$$

取負對數概似，導出**交叉熵誤差函數 (Cross-Entropy Error Function)**：

$$E(\mathbf{w}) = -\sum_{n=1}^N \{t_n \ln y_n + (1 - t_n)\ln(1 - y_n)\} \tag{5.57}$$

### 梯度的極簡形式推導
對單一資料點 $E_n = -t_n \ln y_n - (1 - t_n)\ln(1 - y_n)$，由鏈鎖律：
$$\frac{\partial E_n}{\partial a_n} = \left(-\frac{t_n}{y_n} + \frac{1 - t_n}{1 - y_n}\right) \frac{dy_n}{da_n} = \frac{y_n - t_n}{y_n(1 - y_n)} \cdot y_n(1 - y_n) = y_n - t_n$$
S 形函數的導數項 $y_n(1 - y_n)$ 恰好與交叉熵的倒數分母完全相消！
因此全體梯度化為極度簡潔的表達式：

$$\nabla_{\mathbf{w}} E(\mathbf{w}) = \sum_{n=1}^N (y_n - t_n) \boldsymbol{\phi}_n \tag{5.58}$$

**核心洞見**：每個樣本對梯度的貢獻就是「預測誤差 $(y_n - t_n)$ 乘上輸入基底特徵向量 $\boldsymbol{\phi}_n$」。這與線性迴歸平方誤差的梯度形式完全一致！(來源: [[ch05-single-layer-classification]], p.18)

📌講義（最佳化與可分資料的過度擬合）：
1. 由於 $y_n$ 的非線性，令 $\nabla E(\mathbf{w}) = 0$ 無法取得閉式解，但 $E(\mathbf{w})$ 是嚴格凸函數，可用隨機梯度下降 (SGD) 或迭代重加權最小平方 (IRLS) 求解。
2. **線性可分上的發散過度擬合**：若訓練資料完全線性可分，最大概似解會驅使權重範數 $\|\mathbf{w}\| \to \infty$。此時 $\sigma(a)$ 變成垂直階梯函數，所有訓練樣本預測機率都達到 1。為了解決權重無限發散問題，必須加入正則化項（權重衰減 $\frac{\lambda}{2}\|\mathbf{w}\|^2$）。(來源: [[ch05-single-layer-classification]], p.18–19)

## 3. 多類別邏輯迴歸 (Softmax 迴歸)

📌講義：對於 $K > 2$ 類，後驗機率由 Softmax 函數給出：

$$y_k(\boldsymbol{\phi}) = \frac{\exp(a_k)}{\sum_j \exp(a_j)}, \quad a_k = \mathbf{w}_k^T \boldsymbol{\phi} \tag{5.59}$$

Softmax 對前激活的導數為：$\frac{\partial y_k}{\partial a_j} = y_k(I_{kj} - y_j)$。
在 1-of-K 標籤矩陣 $\mathbf{T}$ 下，負對數概似為**多類交叉熵誤差函數**：

$$E(\mathbf{w}_1, \dots, \mathbf{w}_K) = -\sum_{n=1}^N \sum_{k=1}^K t_{nk} \ln y_{nk} \tag{5.62}$$

對權重向量 $\mathbf{w}_j$ 取梯度，同樣消去所有非線性交叉項，再次導出極簡梯度形式：

$$\nabla_{\mathbf{w}_j} E(\mathbf{w}_1, \dots, \mathbf{w}_K) = \sum_{n=1}^N (y_{nj} - t_{nj}) \boldsymbol{\phi}_n \tag{5.63}$$

單個權重連線 $w_{ik}$（連接基底 $\phi_i$ 與輸出 $k$）的偏導數為：

$$\frac{\partial E}{\partial w_{ik}} = \sum_{n=1}^N (y_{nk} - t_{nk}) \phi_i(\mathbf{x}_n) \tag{5.64}$$

即「輸入端基底激活值 $\times$ 輸出端誤差」——這正是深度神經網路反向傳播演算法的原型。(來源: [[ch05-single-layer-classification]], p.19)

## 自我測驗 Q&A

> [!question] Q1. 交叉熵誤差配上 S 形激活函數時，其梯度公式為何特別簡潔？
> [!answer]- 答案
> 在對前激活求導時，交叉熵倒數項產生的分母 $y(1-y)$ 剛好被 S 形函數自身導數 $\frac{d\sigma}{da} = y(1-y)$ 完全抵銷，直接得到誤差 $(y - t)$。(來源: [[ch05-single-layer-classification]], p.18)

> [!question] Q2. 為什麼邏輯迴歸在完全線性可分的資料集上使用最大概似法會出現數值發散？
> [!answer]- 答案
> 為了讓對數概死最大（使預測機率逼近 1），模型會不斷將 $\|\mathbf{w}\|$ 放大至無窮大以讓 S 形曲線變成階梯狀，故必須引入正則化以約束權重範數。(來源: [[ch05-single-layer-classification]], p.18–19)

> [!question] Q3. 多類別 Softmax 迴歸對權重 $\mathbf{w}_j$ 的梯度形式為何？
> [!answer]- 答案
> $\sum_{n=1}^N (y_{nj} - t_{nj})\boldsymbol{\phi}_n$，形式為第 $j$ 類別的預測誤差乘以特徵向量。(來源: [[ch05-single-layer-classification]], p.19)

## 相關頁面

[[classification-approaches]] · [[generative-classifiers]] · [[probit-regression-canonical-link]] · [[regularized-least-squares-ch04]]
