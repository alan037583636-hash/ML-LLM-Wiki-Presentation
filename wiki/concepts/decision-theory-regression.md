---
title: 迴歸的決策論與損失函數 (Decision Theory for Regression)
type: concept
tags: [decision-theory, loss-functions, expected-loss, conditional-expectation, minkowski-loss, calculus-of-variations]
sources: [ch04-single-layer-regression]
created: 2026-11-11
updated: 2026-11-11
---

# 迴歸的決策論與損失函數 (Decision Theory for Regression)

## 1. 推論階段 vs. 決策階段

📌講義：機器學習在解決實際問題時，任務明確區分為兩個階段：
- **推論階段 (Inference stage)**：用訓練資料建立條件機率分布 $p(t \mid \mathbf{x})$，完整表達對未觀測目標值的不確定性。
- **決策階段 (Decision stage)**：在實際行動中（例如決定腫瘤的放療劑量），往往需要給出一個確切的數值預測 $f(\mathbf{x})$。做法是定義一個損失函數 $L(t, f(\mathbf{x}))$，並在預測分布下最小化其**期望損失 (expected loss)**。(來源: [[ch04-single-layer-regression]], p.9–10)

> [!important] 📌講義：不要混淆「誤差函數」與「損失函數」
> - **誤差函數 (Error function，如 $E_D(\mathbf{w})$)**：屬於**推論階段**，在訓練時用來調整參數 $\mathbf{w}$，以擬合出條件機率分布 $p(t \mid \mathbf{x})$。
> - **損失函數 (Loss function，如 $L(t, f(\mathbf{x}))$)**：屬於**決策階段**，在已知機率分布後，用來決定如何輸出最佳單點決策值 $f(\mathbf{x})$。
> 兩者數學形式常相同（例如都是平方），但角色完全不同。(來源: [[ch04-single-layer-regression]], p.10)

## 2. 平方損失下的最佳決策：條件期望值

📌講義：期望損失定義為損失對聯合分布 $p(\mathbf{x}, t)$ 的期望：

$$\mathbb{E}[L] = \iint L(t, f(\mathbf{x})) p(\mathbf{x}, t) \, d\mathbf{x} dt \tag{4.31}$$

在迴歸問題中，最常用的損失函數是**平方損失 (squared loss)** $L(t, f(\mathbf{x})) = \{f(\mathbf{x}) - t\}^2$：

$$\mathbb{E}[L] = \iint \{f(\mathbf{x}) - t\}^2 p(\mathbf{x}, t) \, d\mathbf{x} dt \tag{4.32}$$

### 變分法推導
📌講義：若假設 $f(\mathbf{x})$ 是完全自由的非參數化函數，利用變分法對 $f(\mathbf{x})$ 取泛函導數並令其為零：

$$\frac{\delta \mathbb{E}[L]}{\delta f(\mathbf{x})} = 2 \int \{f(\mathbf{x}) - t\} p(\mathbf{x}, t) \, dt = 0 \tag{4.33}$$

解出 $f(\mathbf{x})$：

$$f^\star(\mathbf{x}) = \frac{1}{p(\mathbf{x})}\int t \, p(\mathbf{x}, t) \, dt = \int t \, p(t \mid \mathbf{x}) \, dt = \mathbb{E}_t[t \mid \mathbf{x}] \tag{4.34}$$

**結論**：在平方損失下，使期望損失最小的最佳預測器是**給定 $\mathbf{x}$ 下目標變數的條件平均值 (conditional expectation)**，稱為**迴歸函數 (regression function)**。(來源: [[ch04-single-layer-regression]], p.10)

### 期望損失的正交分解推導
📌講義：將 $\{f(\mathbf{x}) - t\}^2$ 展開：

$$\{f(\mathbf{x}) - t\}^2 = \{f(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}] + \mathbb{E}[t \mid \mathbf{x}] - t\}^2$$

代入期望損失積分中，交叉乘積項對 $t$ 積分為零（因 $\int (\mathbb{E}[t \mid \mathbf{x}] - t) p(t \mid \mathbf{x}) dt = 0$），得到：

$$\mathbb{E}[L] = \int \{f(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]\}^2 p(\mathbf{x}) \, d\mathbf{x} + \int \text{var}[t \mid \mathbf{x}] p(\mathbf{x}) \, d\mathbf{x} \tag{4.36}$$

- **第一項**：依賴於我們的預測函數 $f(\mathbf{x})$；當 $f(\mathbf{x}) = \mathbb{E}[t \mid \mathbf{x}]$ 時達到絕對最小值 0。
- **第二項**：目標值本身的條件變異數 $\text{var}[t \mid \mathbf{x}]$，代表資料固有的本質雜訊，與模型 $f(\mathbf{x})$ 完全無關，為任何模型皆無法消除的**不可縮減誤差下限**。(來源: [[ch04-single-layer-regression]], p.11)

## 3. Minkowski 損失函數族與穩健性

📌講義：平方損失是 **Minkowski 損失** 的特例：

$$\mathbb{E}[L_q] = \iint |f(\mathbf{x}) - t|^q p(\mathbf{x}, t) \, d\mathbf{x} dt \tag{4.37}$$

其最佳預測值隨次方 $q$ 的變化而不同：
- **$q = 2$**：最佳解為**條件均值** $\mathbb{E}[t \mid \mathbf{x}]$。
- **$q = 1$**：最佳解為**條件中位數 (conditional median)**。
- **$q \to 0$**：最佳解逼近**條件眾數 (conditional mode)**。(來源: [[ch04-single-layer-regression]], p.11)

📌講義（講義教學補充：損失函數反映對離群值的態度）：
- $q = 2$（平方損失）：對大誤差施以平方級懲罰，極少數嚴重離群值就會把預測結果拉偏。
- $q = 1$（絕對值損失 / 條件中位數）：對極端離群值具備很強的穩健性 (robustness)。在 GNSS 遭遇多路徑效應或訊號遮蔽引發的粗差 (blunder) 時，L1 型損失或 RANSAC 式估計遠比傳統最小平方法可靠。(來源: [[ch04-single-layer-regression]], p.11)

## 自我測驗 Q&A

> [!question] Q1. 為什麼說平方損失下的最佳決策必然是條件期望值 $\mathbb{E}[t \mid \mathbf{x}]$？
> [!answer]- 答案
> 透過變分法最小化期望損失 $\mathbb{E}[L]$，或將損失分解為 $\{f(\mathbf{x}) - \mathbb{E}[t\mid\mathbf{x}]\}^2$ 與條件變異數之和，皆可嚴格證明只有當 $f(\mathbf{x}) = \mathbb{E}[t\mid\mathbf{x}]$ 時，第一項非負誤差項恰好歸零。(來源: [[ch04-single-layer-regression]], p.10–11)

> [!question] Q2. 式 (4.36) 中不可縮減的誤差下限項是什麼？它有何物理意涵？
> [!answer]- 答案
> $\int \text{var}[t \mid \mathbf{x}] p(\mathbf{x}) d\mathbf{x}$；代表目標變數圍繞回歸函數的本質隨機性（固有雜訊），無論模型多完美都無法降低。(來源: [[ch04-single-layer-regression]], p.11)

> [!question] Q3. 當資料中充斥嚴重的隨機粗差（離群值）時，將 Minkowski 損失的 $q$ 由 2 換成 1 有什麼好處？
> [!answer]- 答案
> $q=1$ 對應的最佳預測是條件中位數，其懲罰隨殘差線性增長而非平方暴增，大幅削弱離群值對估計結果的支配影響。(來源: [[ch04-single-layer-regression]], p.11)

## 相關頁面

[[maximum-likelihood-regression]] · [[bias-variance-tradeoff]] · [[ch04-single-layer-regression]]
