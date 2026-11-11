---
title: 動量法與自適應學習率優化器 (Momentum and Adaptive Optimizers)
type: concept
tags: [optimizers, momentum, nesterov, learning-rate-schedules, adagrad, rmsprop, adam, bias-correction]
sources: [ch07-gradient-descent]
created: 2026-11-11
updated: 2026-11-11
---

# 動量法與自適應學習率優化器 (Momentum and Adaptive Optimizers)

在病態條件（Ill-conditioned）曲面上，標準隨機梯度下降（SGD）面臨橫向劇烈震盪與縱向推進極慢的困境。為加速深度神經網路訓練，優化演算法從「引入慣性動量」進一步演進至「逐參數自適應縮放學習率」。

---

## 1. 二次曲面下的收斂性與學習率上限 (Convergence Analysis)

考慮純二次誤差函數（其極小值位於原點，Hessian 矩陣為常數 $\mathbf{H}$）：
$$E(\mathbf{w}) = \frac{1}{2} \mathbf{w}^T \mathbf{H} \mathbf{w}$$
梯度為 $\nabla E(\mathbf{w}) = \mathbf{H} \mathbf{w}$。標準梯度下降更新式為：
$$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} - \eta \mathbf{H} \mathbf{w}^{(\tau)} = (\mathbf{I} - \eta \mathbf{H}) \mathbf{w}^{(\tau)}$$
將參數在 $\mathbf{H}$ 的正交特徵向量基底 $\{\mathbf{u}_i\}$ 上展開：$\mathbf{w}^{(\tau)} = \sum_{i=1}^W \alpha_i^{(\tau)} \mathbf{u}_i$。由於各特徵軸解耦：
$$\alpha_i^{(\tau+1)} = (1 - \eta \lambda_i) \alpha_i^{(\tau)} \implies \alpha_i^{(\tau)} = (1 - \eta \lambda_i)^\tau \alpha_i^{(0)}$$
(來源: [[ch07-gradient-descent]], p.7)。

### 1.1 學習率穩定性上界
為了確保所有特徵分量收斂至 0，必須對所有特徵值滿足收斂條件 $|1 - \eta \lambda_i| < 1$：
$$-1 < 1 - \eta \lambda_i < 1 \iff 0 < \eta < \frac{2}{\lambda_i}$$
因此，保證梯度下降不發散的最大容許學習率為：
$$\eta < \frac{2}{\lambda_{\max}}$$

### 1.2 條件數拖慢收斂
若為了維持穩定性而將學習率設為上限 $\eta \approx 1/\lambda_{\max}$，則在最小特徵值 $\lambda_{\min}$ 軸上的收斂因子為：
$$1 - \eta \lambda_{\min} = 1 - \frac{\lambda_{\min}}{\lambda_{\max}} = 1 - \frac{1}{\kappa}$$
當條件數 $\kappa \gg 1$ 時，$1 - 1/\kappa \approx 1$，沿最小曲率方向的誤差幾何衰減極度緩慢，導致網路陷入漫長的「爬行」狀態 (來源: [[ch07-gradient-descent]], p.8)。

---

## 2. 動量法與 Nesterov 加速動量 (Momentum Techniques)

### 2.1 古典動量法 (Classical Momentum, Polyak 1964)
為解決峽谷震盪問題，動量法將物理學中的「質量慣性」引入更新過程，維護一個速度向量 $\mathbf{v}$：
$$\mathbf{v}^{(\tau+1)} = \mu \mathbf{v}^{(\tau)} - \eta \nabla E(\mathbf{w}^{(\tau)})$$
$$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} + \mathbf{v}^{(\tau+1)}$$
其中 $\mu \in [0, 1)$ 為動量參數（常設為 0.9）。
- **幾何直覺**：在震盪嚴重的維度，連續兩步的梯度方向相反，相加後相互抵消；而在平緩推進的維度，梯度方向恆定，速度不斷累積疊加，最終等效步長放大為 $\frac{1}{1 - \mu}$ 倍 (來源: [[ch07-gradient-descent]], p.8)。

### 2.2 Nesterov 加速動量 (Nesterov Accelerated Gradient, NAG)
古典動量使用當前位置 $\mathbf{w}$ 的梯度來更新速度；Nesterov 則主張「先按照當前慣性走一步，再在未來預估位置計算梯度」：
$$\mathbf{v}^{(\tau+1)} = \mu \mathbf{v}^{(\tau)} - \eta \nabla E(\mathbf{w}^{(\tau)} + \mu \mathbf{v}^{(\tau)})$$
$$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} + \mathbf{v}^{(\tau+1)}$$
具備「前瞻預判」與「主動煞車」能力，顯著減少在極值點附近的衝過頭（Overshooting）現象 (來源: [[ch07-gradient-descent]], p.9)。

---

## 3. 學習率排程 (Learning Rate Scheduling)

📌講義指出，固定學習率難以兼顧訓練初期的探索速度與後期的收斂精度，因此廣泛搭配排程策略 (來源: [[ch07-gradient-descent]], p.9)：
1. **步階衰減 (Step Decay)**：每隔固定 Epoch 衰減為原來的因子（如乘以 0.1）。
2. **指數衰減 (Exponential Decay)**：$\eta_t = \eta_0 e^{-kt}$。
3. **餘弦退火 (Cosine Annealing)**：$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t\pi}{T}\right)$。
4. **學習率預熱 (Warmup)**：在訓練前數個 Epoch 從極小學習率線性遞增至目標基準值，防止未適應隨機權重時過大梯度破壞初始化結構。

---

## 4. 自適應學習率優化器族譜 (Adaptive Optimizers)

📌講義詳細梳理了從 AdaGrad 到 Adam 的演化脈絡 (來源: [[ch07-gradient-descent]], p.9–11)：

### 4.1 AdaGrad (Duchi et al., 2011)
為每個參數分配獨立的學習率，以歷史平方梯度總和進行歸一化：
$$s_i^{(\tau)} = s_i^{(\tau-1)} + (g_i^{(\tau)})^2$$
$$w_i^{(\tau+1)} = w_i^{(\tau)} - \frac{\eta}{\sqrt{s_i^{(\tau)} + \epsilon}} g_i^{(\tau)}$$
- **致命缺陷**：$s_i$ 是單調遞增的，導致有效學習率持續縮小，訓練後期過早「凍結（Freeze）」，無法繼續探索新特徵。

### 4.2 RMSProp (Hinton, 2012)
將累加和改為**指數加權移動平均 (Exponential Moving Average, EMA)**，遺忘遠古歷史梯度：
$$r_i^{(\tau)} = \beta r_i^{(\tau-1)} + (1 - \beta) (g_i^{(\tau)})^2$$
$$w_i^{(\tau+1)} = w_i^{(\tau)} - \frac{\eta}{\sqrt{r_i^{(\tau)} + \epsilon}} g_i^{(\tau)}$$
（通常設定 $\beta = 0.9$ 或 $0.99$）。有效克服了 AdaGrad 學習率早衰問題。

### 4.3 Adam (Adaptive Moment Estimation, Kingma & Ba 2014)
現代深度學習的預設優化器。Adam 同時維護梯度的**一階動量（均值 $s$）**與**二階動量（未中心化變異數 $r$）**：
$$s_i^{(\tau)} = \beta_1 s_i^{(\tau-1)} + (1 - \beta_1) g_i^{(\tau)}$$
$$r_i^{(\tau)} = \beta_2 r_i^{(\tau-1)} + (1 - \beta_2) (g_i^{(\tau)})^2$$
（預設超參數為 $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$）。

#### 偏差修正 (Bias Correction) 的推導與必要性
當初始狀態設為 $s^{(0)} = 0$ 與 $r^{(0)} = 0$ 時，展開遞迴式：
$$s^{(\tau)} = (1 - \beta_1) \sum_{k=1}^\tau \beta_1^{\tau - k} g^{(k)}$$
取期望值（假設當前局部梯度期望值為常數 $\mathbb{E}[g]$）：
$$\mathbb{E}[s^{(\tau)}] = \mathbb{E}[g] (1 - \beta_1) \sum_{k=1}^\tau \beta_1^{\tau - k} = \mathbb{E}[g] (1 - \beta_1) \frac{1 - \beta_1^\tau}{1 - \beta_1} = \mathbb{E}[g] (1 - \beta_1^\tau)$$
在訓練初期（$\tau$ 較小，特別是 $\beta_2 = 0.999$ 時，$1 - \beta_2^\tau \approx 0$），估計值嚴重偏向於 0。因此必須透過無偏修正消除冷啟動偏差：
$$\hat{s}_i^{(\tau)} = \frac{s_i^{(\tau)}}{1 - \beta_1^\tau}, \quad \hat{r}_i^{(\tau)} = \frac{r_i^{(\tau)}}{1 - \beta_2^\tau}$$
最終參數更新公式為：
$$w_i^{(\tau+1)} = w_i^{(\tau)} - \frac{\eta}{\sqrt{\hat{r}_i^{(\tau)}} + \epsilon} \hat{s}_i^{(\tau)}$$
(來源: [[ch07-gradient-descent]], p.10–11)。

---

## 自我測驗 Q&A

### Q1: 在二次誤差曲面中，為什麼學習率必須滿足 $\eta < \frac{2}{\lambda_{\max}}$？若超過此上限會發生什麼現象？
> [!answer]- 答案
> 在 Hessian 矩陣特徵基底解耦後，第 $i$ 個特徵軸的分量更新式為 $\alpha_i^{(\tau+1)} = (1 - \eta \lambda_i) \alpha_i^{(\tau)}$。要使序列收斂，其乘數絕對值必須滿足 $|1 - \eta \lambda_i| < 1$，解出 $\eta < \frac{2}{\lambda_i}$。為了使所有維度皆穩定，學習率必須小於 $\frac{2}{\lambda_{\max}}$。若 $\eta \ge \frac{2}{\lambda_{\max}}$，沿最大曲率方向（$\lambda_{\max}$）的誤差分量將呈指數級擴散放大，導致訓練發散爆炸。

### Q2: 試解釋動量法（Momentum）如何改善病態峽谷曲面中的「之字形震盪」與「緩慢前進」？
> [!answer]- 答案
> 動量法透過維護速度向量 $\mathbf{v}^{(\tau+1)} = \mu \mathbf{v}^{(\tau)} - \eta \nabla E$ 累積歷史方向：
> 1. 在垂直峽谷的陡峭短軸上，梯度符號交替反轉，經過動量加權累加後正負相互抵消，大幅抑制了橫向劇烈震盪。
> 2. 在沿著峽谷平緩延伸的長軸上，梯度方向始終一致，動量累積使速度疊加為穩定的常態前進，有效步長被放大為原來的 $\frac{1}{1-\mu}$ 倍，顯著加速收斂。

### Q3: Adam 優化器在初始訓練步時為什麼必須對動量項進行「偏差修正（Bias Correction）」？若不修正會有什麼後果？
> [!answer]- 答案
> 因為一階與二階動量初始化為 0（$s^{(0)}=0, r^{(0)}=0$），其指數移動平均在前期嚴重偏向原點 0，期望值滿足 $\mathbb{E}[s^{(\tau)}] = \mathbb{E}[g](1 - \beta_1^\tau)$ 與 $\mathbb{E}[r^{(\tau)}] = \mathbb{E}[g^2](1 - \beta_2^\tau)$。特別是 $\beta_2 = 0.999$ 時，在第 1 步時 $r^{(1)} = 0.001 g^2$，二階動量極小；若不除以 $(1 - \beta_2^\tau)$ 修正，分母的極小值會導致更新步長在初始階段發生病態巨大幅度跳躍，破壞預訓練或初始化權重。

---

## 相關頁面
- [[ch07-gradient-descent|第七章 梯度下降講義摘要]]
- [[error-surfaces-and-geometry|誤差曲面與幾何特性]]
- [[gradient-descent-and-initialization|梯度下降家族與參數初始化]]
- [[normalization-techniques|網路正規化技術]]
- [[optimization-algorithms-comparison|最佳化演算法綜合比較]]
