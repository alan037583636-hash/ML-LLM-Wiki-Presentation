---
title: 誤差曲面與幾何特性 (Error Surfaces and Geometry)
type: concept
tags: [error-surfaces, geometry, hessian, saddle-points, condition-number, taylor-expansion]
sources: [ch07-gradient-descent]
created: 2026-11-11
updated: 2026-11-11
---

# 誤差曲面與幾何特性 (Error Surfaces and Geometry)

在深度神經網路中，訓練過程可視為在高維權重空間（Weight Space）上探尋誤差曲面（Error Surface）$E(\mathbf{w})$ 極小值的幾何歷程。理解誤差曲面的局部曲率、駐點性質以及對稱結構，是掌握現代最佳化演算法行為的理論核心。

---

## 1. 局部一階展開與駐點分類 (Stationary Points)

在權重空間中，從當前參數 $\mathbf{w}$ 沿微小位移向量 $\delta\mathbf{w}$ 移動，誤差函數的一階變化量為：
$$\delta E \approx \delta\mathbf{w}^T \nabla E(\mathbf{w})$$
其中梯度向量 $\nabla E(\mathbf{w})$ 指向誤差曲面上升最陡峭的方向。

### 1.1 駐點定義 (Stationary Point)
當梯度向量為零向量時，該點稱為駐點（Stationary Point）或臨界點（Critical Point）：
$$\nabla E(\mathbf{w}^*) = \mathbf{0}$$
此時一階導數無法提供搜尋方向，必須透過二階導數（曲率資訊）進一步判定其幾何性質 (來源: [[ch07-gradient-descent]], p.2)。

### 1.2 高維空間中的駐點類型
1. **局部極小值 (Local Minimum)**：所有方向的曲率皆為正（所有微小位移皆使誤差增加）。
2. **局部極大值 (Local Maximum)**：所有方向的曲率皆為負（所有微小位移皆使誤差減小）。
3. **鞍點 (Saddle Points)**：部分方向曲率為正（向上彎曲），部分方向曲率為負（向下彎曲）。
📌講義特別強調：在超高維權重空間中，**絕大多數駐點並非局部極小值，而是鞍點**。若有 $W$ 個獨立維度，隨機駐點在所有 $W$ 個方向上曲率皆為正的機率僅為 $(1/2)^W$，呈指數級稀少；因此，神經網路最佳化的主要瓶頸在於如何逃離大量鞍點與平坦高原（Plateaus），而非陷入不良局部極小值 (來源: [[ch07-gradient-descent]], p.2, p.17)。

---

## 2. 局部二次近似與 Hessian 矩陣 (Local Quadratic Approximation)

在駐點 $\mathbf{w}^*$ 附近，將誤差函數進行二階泰勒級數（Taylor Series）展開：
$$E(\mathbf{w}) \approx E(\mathbf{w}^*) + (\mathbf{w} - \mathbf{w}^*)^T \underbrace{\nabla E(\mathbf{w}^*)}_{=\mathbf{0}} + \frac{1}{2} (\mathbf{w} - \mathbf{w}^*)^T \mathbf{H} (\mathbf{w} - \mathbf{w}^*)$$
其中 $\mathbf{H} \in \mathbb{R}^{W \times W}$ 為 **Hessian 矩陣**（二階偏導矩陣）：
$$H_{ij} = \frac{\partial^2 E}{\partial w_i \partial w_j}$$
由於多數誤差函數二階偏導平滑連續，根據克萊羅定理（Schwarz's Theorem），Hessian 矩陣為**實對稱矩陣（Real Symmetric Matrix）**，即 $\mathbf{H} = \mathbf{H}^T$ (來源: [[ch07-gradient-descent]], p.3)。

### 2.1 特徵分解與正交座標變換 (Eigendecomposition)
實對稱矩陣 $\mathbf{H}$ 具備一組標準正交特徵向量基底 $\{\mathbf{u}_i\}_{i=1}^W$，滿足：
$$\mathbf{H} \mathbf{u}_i = \lambda_i \mathbf{u}_i, \quad \mathbf{u}_i^T \mathbf{u}_j = \delta_{ij}$$
其特徵分解為：
$$\mathbf{H} = \sum_{i=1}^W \lambda_i \mathbf{u}_i \mathbf{u}_i^T = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T$$
將相對於駐點的位移向量分解在特徵向量基底上：
$$\mathbf{w} - \mathbf{w}^* = \sum_{i=1}^W \alpha_i \mathbf{u}_i, \quad \alpha_i = \mathbf{u}_i^T (\mathbf{w} - \mathbf{w}^*)$$
代入二階展開式中，二階交叉項完全正交解耦：
$$E(\mathbf{w}) \approx E(\mathbf{w}^*) + \frac{1}{2} \sum_{i=1}^W \lambda_i \alpha_i^2$$
(來源: [[ch07-gradient-descent]], p.3)。

### 2.2 Hessian 正定性與極值判別
由上式可直接得出幾何判別準則：
- **正定 (Positive Definite, $\mathbf{H} \succ 0$)**：所有特徵值 $\lambda_i > 0$ $\implies$ 嚴格**局部極小值**。
- **負定 (Negative Definite, $\mathbf{H} \prec 0$)**：所有特徵值 $\lambda_i < 0$ $\implies$ 嚴格**局部極大值**。
- **不定 (Indefinite)**：特徵值正負兼具 $\implies$ **鞍點**。
- **半正定/奇異 (Singular, 存在 $\lambda_i = 0$)**：沿零特徵值方向為平坦山谷（Valley），形成連續流形極值點。

---

## 3. 條件數與病態曲率 (Condition Number and Ill-conditioning)

在局部極小值處（所有 $\lambda_i > 0$），誤差曲面的等高線為 $W$ 維超橢球。
- **長軸與短軸**：特徵值最大的方向（$\lambda_{\max}$）對應曲率最大、坡度最陡峭的短軸；特徵值最小的方向（$\lambda_{\min}$）對應曲率最小、坡度最平緩的長軸。
- **條件數定義**：
  $$\kappa(\mathbf{H}) = \frac{\lambda_{\max}}{\lambda_{\min}}$$
📌講義指出，當 $\kappa \gg 1$ 時稱為**病態條件（Ill-conditioned）**。此時誤差曲面呈現極其狹長的「峽谷（Ravine）」形狀。梯度下降演算法在此類幾何曲面下，極易在陡峭峭壁間發生劇烈橫向震盪，而在平緩谷底推進速度極端緩慢 (來源: [[ch07-gradient-descent]], p.7–8)。

---

## 4. 權重空間對稱性與等價極小值 (Weight-space Symmetries)

📌講義提醒，神經網路的非凸誤差曲面絕非隨機分佈，而是蘊含嚴格的離散對稱性：
- 對於包含 $M$ 個隱藏單元的兩層前饋網路，因隱藏單元置換（Permutation Symmetry, $M!$ 種）與激活函數變號（Sign-flip Symmetry, 如 $\tanh$ 的 $2^M$ 種），權重空間中至少存在：
  $$M! \cdot 2^M$$
  個完全等價的局部極小值 (來源: [[ch07-gradient-descent]], p.4)。
- 因此，神經網路最佳化並不需要（也不可能）尋找「唯一」全域最優解，而是尋找任一具備良好泛化能力的對稱解池。

---

## 自我測驗 Q&A

### Q1: 在高維權重空間中，為什麼最佳化過程遇到的駐點大多數是鞍點而非局部極小值？
> [!answer]- 答案
> 在 $W$ 維空間中，駐點處的 Hessian 矩陣擁有 $W$ 個特徵值。要成為局部極小值，必須滿足所有 $W$ 個特徵值皆為正（$\lambda_i > 0$）。若假設特徵值為正或負的機率各為 $1/2$，則所有特徵值全為正的機率為 $(1/2)^W$。當參數數量 $W$ 達到數萬至數十億時，該機率幾乎為零。相反地，只要特徵值正負混雜即為鞍點，因此高維空間中駐點絕大多數為鞍點。

### Q2: 試寫出誤差函數在駐點 $\mathbf{w}^*$ 處的二階泰勒級數解耦展開式，並說明特徵值 $\lambda_i$ 的幾何意義。
> [!answer]- 答案
> 展開式為：
> $$E(\mathbf{w}) \approx E(\mathbf{w}^*) + \frac{1}{2} \sum_{i=1}^W \lambda_i \alpha_i^2$$
> 其中 $\alpha_i = \mathbf{u}_i^T (\mathbf{w} - \mathbf{w}^*)$ 為位移在 Hessian 正交特徵向量 $\mathbf{u}_i$ 上的投影分量。
> 特徵值 $\lambda_i$ 代表沿著特徵向量 $\mathbf{u}_i$ 方向的**固有曲率（Curvature）**：$\lambda_i > 0$ 表示向上彎曲，$\lambda_i < 0$ 表示向下彎曲，$\lambda_i = 0$ 表示平坦無曲率。

### Q3: 何謂 Hessian 矩陣的「條件數（Condition Number）」？它如何影響梯度下降的移動軌跡？
> [!answer]- 答案
> 條件數定義為 Hessian 最大特徵值與最小特徵值的比值：$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$。
> 當 $\kappa \gg 1$ 時，誤差曲面形成極狹長的山谷。沿 $\lambda_{\max}$（陡壁）方向梯度極大，容易造成沿垂直山谷方向的劇烈橫向震盪；沿 $\lambda_{\min}$（谷底）方向梯度極微弱，導致沿山谷前進極其緩慢。

---

## 相關頁面
- [[ch07-gradient-descent|第七章 梯度下降講義摘要]]
- [[gradient-descent-and-initialization|梯度下降家族與參數初始化]]
- [[momentum-and-adaptive-optimizers|動量法與自適應學習率優化器]]
- [[normalization-techniques|網路正規化技術]]
