---
title: 最小平方法的幾何詮釋 (Geometry of Least Squares)
type: concept
tags: [geometry, least-squares, orthogonal-projection, hat-matrix, subspace, linear-algebra]
sources: [ch04-single-layer-regression]
created: 2026-11-11
updated: 2026-11-11
---

# 最小平方法的幾何詮釋 (Geometry of Least Squares)

## 1. $N$ 維空間中的幾何視角

📌講義：考慮一個以各資料點目標值 $t_1, \dots, t_N$ 為座標軸的 **$N$ 維空間**：
- 目標向量 $\mathbf{t} = (t_1, \dots, t_N)^T \in \mathbb{R}^N$ 是該空間中的一個點。
- 每個基底函數 $\phi_j(\mathbf{x})$ 在 $N$ 個訓練點上的取值形成長度為 $N$ 的行向量，記為 $\boldsymbol{\varphi}_j = (\phi_j(\mathbf{x}_1), \dots, \phi_j(\mathbf{x}_N))^T$（對應設計矩陣 $\boldsymbol{\Phi}$ 的第 $j$ 個行 column）。
- 當基底數 $M < N$ 時，$M$ 個向量 $\{\boldsymbol{\varphi}_j\}_{j=0}^{M-1}$ 張出一個 **$M$ 維線性子空間 $\mathcal{S}$**。
- 模型預測向量 $\mathbf{y} = \boldsymbol{\Phi}\mathbf{w} = \sum_{j=0}^{M-1} w_j \boldsymbol{\varphi}_j$ 是 $\boldsymbol{\varphi}_j$ 的任意線性組合，因此 $\mathbf{y}$ 落在 $M$ 維子空間 $\mathcal{S}$ 中。(來源: [[ch04-single-layer-regression]], p.6)

## 2. 正交投影與帽子矩陣

📌講義：
平方和誤差 $E_D(\mathbf{w}) = \frac{1}{2}\|\mathbf{y} - \mathbf{t}\|^2$ 等價於向量 $\mathbf{y}$ 與 $\mathbf{t}$ 之間的歐氏距離平方。
要讓誤差最小，$\mathbf{y}$ 必須是子空間 $\mathcal{S}$ 中離 $\mathbf{t}$ 最近的點——這就是 $\mathbf{t}$ 在子空間 $\mathcal{S}$ 上的**正交投影 (orthogonal projection)**：

$$\mathbf{y} = \boldsymbol{\Phi}\mathbf{w}_{\text{ML}} = \boldsymbol{\Phi}(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T \mathbf{t} = \mathbf{P}\mathbf{t}$$

其中投影矩陣

$$\mathbf{P} \equiv \boldsymbol{\Phi}(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T$$

在統計學中稱為**帽子矩陣 (hat matrix)**（因為它把 $\mathbf{t}$ 戴上帽子變成預測值 $\hat{\mathbf{t}} = \mathbf{y}$）。(來源: [[ch04-single-layer-regression]], p.6)

📌講義（投影矩陣的性質與殘差正交性）：
1. **對稱性**：$\mathbf{P}^T = \mathbf{P}$。
2. **冪等性 (Idempotent)**：$\mathbf{P}^2 = \mathbf{P}$（再次投影不變）。
3. **保持基底不變**：$\mathbf{P}\boldsymbol{\Phi} = \boldsymbol{\Phi}$。
4. **殘差正交**：殘差向量 $\mathbf{t} - \mathbf{y} = (\mathbf{I} - \mathbf{P})\mathbf{t}$ 滿足：
   $$\boldsymbol{\Phi}^T(\mathbf{t} - \mathbf{y}) = \boldsymbol{\Phi}^T\mathbf{t} - \boldsymbol{\Phi}^T\boldsymbol{\Phi}\mathbf{w}_{\text{ML}} = \mathbf{0}$$
   這代表**殘差向量正交於子空間 $\mathcal{S}$ 中的每一個基底向量**（$\mathbf{t} - \mathbf{y} \perp \mathcal{S}$）。這正是「正規方程 (normal equations)」中「正規 (normal / 法線)」一詞的幾何真諦。(來源: [[ch04-single-layer-regression]], p.6)

## 3. 共線性、退化與 SVD

📌講義：當兩個或多個基底向量 $\boldsymbol{\varphi}_j$ 接近共線時，子空間 $\mathcal{S}$ 的定義變得扁平病態，$\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ 接近奇異，求逆會導致數值不穩定，解出的權重 $\mathbf{w}$ 量級暴增。此數值困難可用**奇異值分解 (SVD)** 處理；而加入正則化項 $\lambda \mathbf{I}$ 則保證矩陣恆正定可逆。(來源: [[ch04-single-layer-regression]], p.6–7)

📌講義（講義教學補充：GNSS 定位幾何精度因子 DOP）：
在 GNSS 單點定位的觀測方程 $\delta\boldsymbol{\rho} = \mathbf{H} \delta\mathbf{x} + \boldsymbol{\epsilon}$ 中，最小平方解為 $\delta\mathbf{x} = (\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T \delta\boldsymbol{\rho}$。幾何精度因子 $\text{DOP} = \sqrt{\text{tr}(\mathbf{H}^T\mathbf{H})^{-1}}$ 正是基底向量共線性程度的量化：當可見衛星在天空中擠在同一方向時，矩陣 $\mathbf{H}$ 的行向量接近共線，導致估計變異數劇烈放大。(來源: [[ch04-single-layer-regression]], p.7)

## 自我測驗 Q&A

> [!question] Q1. 最小平方法的幾何觀點是在幾維空間中討論？各軸代表什麼？
> [!answer]- 答案
> 在 $N$ 維空間（$N$ 為訓練樣本數），各座標軸代表各資料點的目標值 $t_n$。(來源: [[ch04-single-layer-regression]], p.6)

> [!question] Q2. 帽子矩陣 $\mathbf{P} = \boldsymbol{\Phi}(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T$ 的兩個核心代數性質為何？
> [!answer]- 答案
> 對稱性 $\mathbf{P}^T = \mathbf{P}$ 與冪等性 $\mathbf{P}^2 = \mathbf{P}$。(來源: [[ch04-single-layer-regression]], p.6)

> [!question] Q3. 為什麼正規方程稱為「正規 (Normal)」？
> [!answer]- 答案
> 因為最小平方解要求殘差向量 $\mathbf{t} - \mathbf{y}$ 與基底張成的預測子空間 $\mathcal{S}$ 正交（法線方向垂直），即 $\boldsymbol{\Phi}^T(\mathbf{t} - \mathbf{y}) = \mathbf{0}$。(來源: [[ch04-single-layer-regression]], p.6)

## 相關頁面

[[maximum-likelihood-regression]] · [[regularized-least-squares-ch04]] · [[least-squares-normal-equations]]
