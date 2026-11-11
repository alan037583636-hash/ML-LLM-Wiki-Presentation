---
title: 線性判別函數與幾何 (Linear Discriminant Functions and Geometry)
type: concept
tags: [discriminant-functions, geometry, decision-boundary, least-squares-classification, convex-regions]
sources: [ch05-single-layer-classification]
created: 2026-11-11
updated: 2026-11-11
---

# 線性判別函數與幾何 (Linear Discriminant Functions and Geometry)

## 1. 兩類別線性判別函數的幾何

📌講義：線性判別函數最簡單的表示為輸入向量的線性函數：

$$y(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + w_0 \tag{5.2}$$

- 若 $y(\mathbf{x}) \ge 0$，將 $\mathbf{x}$ 指派到 $\mathcal{C}_1$；否則指派到 $\mathcal{C}_2$。
- 對應的決策邊界為超平面 $y(\mathbf{x}) = 0$。

### 核心幾何性質推導
1. **法向量方向**：
   取決策面上的任意兩點 $\mathbf{x}_A, \mathbf{x}_B$。因 $y(\mathbf{x}_A) = y(\mathbf{x}_B) = 0$，得 $\mathbf{w}^T(\mathbf{x}_A - \mathbf{x}_B) = 0$。因此**權重向量 $\mathbf{w}$ 正交於決策面內的每個向量**，決定了決策面的空間方向。
2. **原點法向距離**：
   若 $\mathbf{x}$ 在決策面上，原點到該超平面的正交投影長度為：
   $$\frac{\mathbf{w}^T\mathbf{x}}{\|\mathbf{w}\|} = -\frac{w_0}{\|\mathbf{w}\|} \tag{5.3}$$
   因此**偏置 $w_0$ 決定了決策面的位置截距**。
3. **任意點的正交距離**：
   設 $\mathbf{x}_\perp$ 為點 $\mathbf{x}$ 在決策面上的正交投影，則 $\mathbf{x} = \mathbf{x}_\perp + r \frac{\mathbf{w}}{\|\mathbf{w}\|}$。兩邊同乘 $\mathbf{w}^T$ 並加 $w_0$，利用 $y(\mathbf{x}_\perp) = 0$ 解得：
   $$r = \frac{y(\mathbf{x})}{\|\mathbf{w}\|} \tag{5.5}$$
   即 $y(\mathbf{x})$ 的值給出點 $\mathbf{x}$ 到超平面之帶號垂直正交距離。(來源: [[ch05-single-layer-classification]], p.3)

## 2. 多類別判別函數與凸決策區域

📌講義：將兩類判別器拼湊到 $K > 2$ 類會遭遇結構性困難：
- **一對其餘 (One-versus-the-rest)**：用 $K - 1$ 個分類器，會在空間中產生未被任何類別挑選、或同時被多個類別挑選的**歧義區域 (ambiguity regions)**。
- **一對一 (One-versus-one)**：用 $K(K-1)/2$ 個二元判別器以多數決投票，同樣會引發投票平手或不一致的歧義區域。

### 單一 $K$ 類判別器
📌講義：避免歧義的正確方式是定義包含 $K$ 個線性函數的判別器：

$$y_k(\mathbf{x}) = \mathbf{w}_k^T\mathbf{x} + w_{k0} \tag{5.7}$$

並將 $\mathbf{x}$ 指派給滿足 $y_k(\mathbf{x}) > y_j(\mathbf{x})$（對所有 $j \ne k$）的類別 $\mathcal{C}_k$。類別 $\mathcal{C}_k$ 與 $\mathcal{C}_j$ 之間的決策邊界為超平面：

$$(\mathbf{w}_k - \mathbf{w}_j)^T\mathbf{x} + (w_{k0} - w_{j0}) = 0 \tag{5.8}$$

📌講義（決策區域是單連通凸集）：
若 $\mathbf{x}_A, \mathbf{x}_B$ 都在決策區域 $\mathcal{R}_k$ 內，考慮連線上任意點 $\hat{\mathbf{x}} = \lambda \mathbf{x}_A + (1 - \lambda)\mathbf{x}_B$（$0 \le \lambda \le 1$）。由線性性：
$$y_k(\hat{\mathbf{x}}) = \lambda y_k(\mathbf{x}_A) + (1 - \lambda)y_k(\mathbf{x}_B) > \lambda y_j(\mathbf{x}_A) + (1 - \lambda)y_j(\mathbf{x}_B) = y_j(\hat{\mathbf{x}})$$
因此 $\hat{\mathbf{x}}$ 亦屬於 $\mathcal{R}_k$。這證明了**線性分類器下的每個決策區域 $\mathcal{R}_k$ 必然是凸集 (convex set)，且為單連通空間**。(來源: [[ch05-single-layer-classification]], p.4)

## 3. 最小平方分類法及其致命弱點

📌講義：考慮對目標標籤使用 1-of-K 獨熱編碼 $\mathbf{T}$，並最小化平方和誤差：

$$E_D(\widetilde{\mathbf{W}}) = \frac{1}{2} \text{Tr}\{(\widetilde{\mathbf{X}}\widetilde{\mathbf{W}} - \mathbf{T})^T(\widetilde{\mathbf{X}}\widetilde{\mathbf{W}} - \mathbf{T})\}$$

解得閉式解 $\widetilde{\mathbf{W}} = (\widetilde{\mathbf{X}}^T\widetilde{\mathbf{X}})^{-1}\widetilde{\mathbf{X}}^T\mathbf{T} = \widetilde{\mathbf{X}}^\dagger \mathbf{T}$。

### 致命弱點：對離群值極度不穩健
📌講義：
1. **機率超出界限**：最小平方預測的輸出沒有被約束在 $(0, 1)$ 範圍內，無法合理解釋為機率。
2. **缺乏穩健性 (Lack of Robustness)**：平方和誤差隱含「高斯雜訊」假設，但二元/獨熱標籤絕非高斯分布。若資料中存在遠離決策邊界、但**已經被正確分類**的極端離群樣本，因其預測值遠大於 1，平方和誤差 $\{y(\mathbf{x}_n) - 1\}^2$ 會產生極為巨大的代價。為了壓低這些離群點的平方誤差，最小平方會強行把決策邊界「拉向」離群點，導致原本可分的核心資料點反而被嚴重誤判（「分對還被嚴懲」）。(來源: [[ch05-single-layer-classification]], p.5–6)

## 自我測驗 Q&A

> [!question] Q1. 權重向量 $\mathbf{w}$ 與偏置 $w_0$ 在兩類線性判別邊界上各自控制什麼？
> [!answer]- 答案
> $\mathbf{w}$ 垂直於超平面，決定決策面的法向角度；$w_0$ 決定超平面相對於座標原點的正交位移距離 $-\frac{w_0}{\|\mathbf{w}\|}$。(來源: [[ch05-single-layer-classification]], p.3)

> [!question] Q2. 為什麼多類別線性分類器中的各個決策區域 $\mathcal{R}_k$ 保證是凸集？
> [!answer]- 答案
> 因為判別函數 $y_k(\mathbf{x})$ 是輸入的線性函數，區域內兩點的凸組合依然滿足 $y_k(\hat{\mathbf{x}}) > y_j(\hat{\mathbf{x}})$。(來源: [[ch05-single-layer-classification]], p.4)

> [!question] Q3. 為什麼將線性迴歸的最小平方法直接套用到分類問題是不合適的？
> [!answer]- 答案
> 平方誤差對離決策面極遠的正確樣本施以不成比例的巨大懲罰（缺乏穩健性），且其輸出無法約束在 $[0, 1]$ 區間內。(來源: [[ch05-single-layer-classification]], p.5–6)

## 相關頁面

[[classification-approaches]] · [[decision-theory-classification]] · [[discriminative-classifiers-logistic-regression]] · [[geometry-of-least-squares]]
