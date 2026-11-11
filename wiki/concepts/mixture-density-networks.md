---
title: 混合密度網路 (Mixture Density Networks)
type: concept
tags: [mdn, mixture-density-networks, inverse-problems, gmm, multimodal]
sources: [ch06-deep-neural-networks]
created: 2026-11-11
updated: 2026-11-11
---

# 混合密度網路 (Mixture Density Networks, MDN)

標準前饋神經網路在處理迴歸問題時，隱含了目標變數服從單峰高斯分布的假設，預測輸出對應於條件期望值 $\mathbb{E}[\mathbf{t} \mid \mathbf{x}]$。然而，在許多物理系統、機器人運動學與**逆問題（Inverse Problems）**中，給定輸入 $\mathbf{x}$ 時，目標 $\mathbf{t}$ 可能存在多個完全有效且合理的解，呈現**多峰條件分布（Multimodal Conditional Distribution）**。若強行套用平方誤差，網路將輸出無意義的多峰平均值。**混合密度網路 (MDN)** 正是為解決此類問題而設計。

---

## 1. 逆問題與單峰假設的崩潰 (Inverse Problems and Failure of Mean)

📌講義以機器人逆向運動學（Inverse Kinematics）為經典範例：
- **前向映射 (Forward Mapping)**：給定機器人各關節角度 $\mathbf{x}$，機械手臂末端夾爪的空間座標 $\mathbf{t}$ 是唯一確定的單值函數。
- **逆向問題 (Inverse Problem)**：給定末端夾爪欲抵達的目標位置 $\mathbf{x}$，可能存在多組截然不同的關節角度構型（如肘部向上 vs 肘部向下）皆能達到該位置。
- **均方誤差（MSE）的致命缺陷**：若使用標準神經網路搭配平方誤差訓練，模型最小化損失時會收斂至條件均值：
  $$y(\mathbf{x}) = \mathbb{E}[\mathbf{t} \mid \mathbf{x}]$$
  而在多峰分布中，兩個有效解的中間平均值往往落在物理上根本無法到達的無效區域（例如撞到障礙物或物理關節極限）(來源: [[ch06-deep-neural-networks]], p.21)。

---

## 2. 混合密度網路架構 (MDN Architecture)

📌講義指出，MDN 結合了前饋神經網路與**高斯混合模型 (Gaussian Mixture Model, GMM)**。網路不再直接預測 $\mathbf{t}$，而是以輸入 $\mathbf{x}$ 為條件，預測 GMM 的所有混合參數 (來源: [[ch06-deep-neural-networks]], p.22–23)：

$$p(\mathbf{t} \mid \mathbf{x}) = \sum_{k=1}^K \pi_k(\mathbf{x}) \, \mathcal{N}\left(\mathbf{t} \;\middle|\; \boldsymbol{\mu}_k(\mathbf{x}), \, \sigma_k^2(\mathbf{x})\mathbf{I}\right)$$

其中共有 $K$ 個混合成分（Components）：

### 2.1 網路輸出與參數約束 (Parameter Constraints)
神經網路輸出三組不同的前激活值，並分別施加合適的轉換函數以滿足機率公理：

1. **混合係數 (Mixing Coefficients) $\pi_k(\mathbf{x})$**：
   必須滿足機率約束 $0 \le \pi_k(\mathbf{x}) \le 1$ 且 $\sum_{k=1}^K \pi_k(\mathbf{x}) = 1$。
   使用 **Softmax** 函數映射：
   $$\pi_k(\mathbf{x}) = \frac{\exp(a_k^\pi)}{\sum_{j=1}^K \exp(a_j^\pi)}$$
2. **成分變異數 (Component Variances) $\sigma_k^2(\mathbf{x})$**：
   變異數（或標準差）必須恆為正數 $\sigma_k(\mathbf{x}) > 0$。
   使用 **指數函數 (Exponential)** 或 Softplus 映射：
   $$\sigma_k(\mathbf{x}) = \exp(a_k^\sigma)$$
3. **成分均值 (Component Means) $\boldsymbol{\mu}_k(\mathbf{x})$**：
   均值向量 $\boldsymbol{\mu}_k \in \mathbb{R}^C$ 無數值邊界限制，直接由**線性單元**輸出：
   $$\mu_{kc}(\mathbf{x}) = a_{kc}^\mu \quad (c = 1, \dots, C)$$
   (來源: [[ch06-deep-neural-networks]], p.23)。

---

## 3. 損失函數與前激活梯度推導 (Loss Function and Error Gradients)

### 3.1 負對數概似損失函數
給定訓練樣本 $(\mathbf{x}_n, \mathbf{t}_n)$，單筆樣本的誤差函數定義為條件機率的負對數：
$$E_n(\mathbf{w}) = -\ln p(\mathbf{t}_n \mid \mathbf{x}_n) = -\ln \left\{ \sum_{k=1}^K \pi_k(\mathbf{x}_n) \, \mathcal{N}\left(\mathbf{t}_n \;\middle|\; \boldsymbol{\mu}_k(\mathbf{x}_n), \, \sigma_k^2(\mathbf{x}_n)\mathbf{I}\right) \right\}$$
(來源: [[ch06-deep-neural-networks]], p.24)。

### 3.2 後驗責任度 (Posterior Responsibility)
定義第 $k$ 個高斯成分生成目標值 $\mathbf{t}_n$ 的後驗機率（責任度）：
$$\gamma_{nk} = p(k \mid \mathbf{x}_n, \mathbf{t}_n) = \frac{\pi_k \mathcal{N}(\mathbf{t}_n \mid \boldsymbol{\mu}_k, \sigma_k^2 \mathbf{I})}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{t}_n \mid \boldsymbol{\mu}_j, \sigma_j^2 \mathbf{I})}$$

### 3.3 三組前激活梯度公式
利用鏈鎖律與 Softmax/Exponential 導數，誤差函數對三組前激活值的偏導數具有極其優美的解析形式 (來源: [[ch06-deep-neural-networks]], p.25)：

1. **混合係數前激活梯度**：
   $$\delta_{nk}^\pi = \frac{\partial E_n}{\partial a_k^\pi} = \pi_k - \gamma_{nk}$$
   （直覺：預測的先驗機率 $\pi_k$ 與觀測後驗責任度 $\gamma_{nk}$ 之間的差值！）
2. **均值前激活梯度**：
   $$\delta_{nkc}^\mu = \frac{\partial E_n}{\partial a_{kc}^\mu} = \gamma_{nk} \left( \frac{\mu_{kc} - t_{nc}}{\sigma_k^2} \right)$$
   （直覺：由後驗責任度 $\gamma_{nk}$ 加權的標準高斯誤差！）
3. **變異數前激活梯度**：
   $$\delta_{nk}^\sigma = \frac{\partial E_n}{\partial a_k^\sigma} = -\gamma_{nk} \left\{ \frac{\|\mathbf{t}_n - \boldsymbol{\mu}_k\|^2}{\sigma_k^2} - C \right\}$$
   （其中 $C$ 為目標向量 $\mathbf{t}$ 的維度）。

---

## 4. 預測與決策 (Prediction and Decision)

訓練完成後，給定測試輸入 $\mathbf{x}$，MDN 輸出一整個連續條件機率密度函數 $p(\mathbf{t} \mid \mathbf{x})$，可依據任務需求採取不同預測策略 (來源: [[ch06-deep-neural-networks]], p.26)：

1. **條件均值 (Conditional Mean)**：
   $$\mathbb{E}[\mathbf{t} \mid \mathbf{x}] = \sum_{k=1}^K \pi_k(\mathbf{x}) \boldsymbol{\mu}_k(\mathbf{x})$$
   （注意：在強多峰問題中，通常應避免直接採用此值作為點預測）。
2. **條件變異數 (Conditional Variance)**：
   $$\text{Var}(\mathbf{t} \mid \mathbf{x}) = \sum_{k=1}^K \pi_k(\mathbf{x}) \left[ \sigma_k^2(\mathbf{x})\mathbf{I} + \|\boldsymbol{\mu}_k(\mathbf{x}) - \mathbb{E}[\mathbf{t} \mid \mathbf{x}]\|^2 \right]$$
   分解為**內部各成分雜訊（Aleatoric Uncertainty）**與**各峰均值分散度**的總和。
3. **最高混合係數均值 (Most Probable Component)**：
   選擇權重最大的成分均值：$\mathbf{t}^* = \boldsymbol{\mu}_{k^*}(\mathbf{x})$，其中 $k^* = \arg\max_k \pi_k(\mathbf{x})$。
4. **全域條件眾數 (Conditional Mode)**：
   數值搜尋使總密度最大化的點：$\mathbf{t}^* = \arg\max_\mathbf{t} p(\mathbf{t} \mid \mathbf{x})$，最適合逆問題求解。

---

## 自我測驗 Q&A

### Q1: 在逆問題（Inverse Problems）中，傳統回歸神經網路搭配均方誤差（MSE）會遇到什麼致命問題？
> [!answer]- 答案
> 在逆問題中，單一輸入 $\mathbf{x}$ 可能對應多個有效解 $\mathbf{t}$（條件分布為多峰 Multimodal）。MSE 的最佳解是條件期望值 $\mathbb{E}[\mathbf{t} \mid \mathbf{x}]$，其輸出為所有可能解的加權平均。然而，多個有效解的算術平均值往往落在無效的狀態空間中（例如機器人關節的不可行構型），導致預測結果完全失敗。

### Q2: 混合密度網路（MDN）如何保證混合係數 $\pi_k$ 與標準差 $\sigma_k$ 滿足機率分佈的數學公理？
> [!answer]- 答案
> 1. 混合係數 $\pi_k$ 必須非負且總和為 1，因此 MDN 將對應的前激活值通過 **Softmax** 函數進行映射。
> 2. 標準差 $\sigma_k$ 必須嚴格大於 0，因此 MDN 將對應的前激活值通過 **指數函數（Exponential）**（或 Softplus 函數）進行映射。

### Q3: 寫出 MDN 負對數概似損失對混合係數前激活值 $a_k^\pi$ 的偏導函數，並說明其物理意義。
> [!answer]- 答案
> 偏導數公式為：
> $$\delta_{nk}^\pi = \frac{\partial E_n}{\partial a_k^\pi} = \pi_k - \gamma_{nk}$$
> 其中 $\pi_k$ 為網路前向預測的先驗混合權重，$\gamma_{nk} = p(k \mid \mathbf{x}_n, \mathbf{t}_n)$ 為第 $k$ 個成分生成觀測目標 $\mathbf{t}_n$ 的後驗責任度。其物理意義是「先驗預測機率與觀測後驗機率的誤差」，當第 $k$ 個成分對實際數據的責任度高於先驗機率時，梯度將驅動增大該成分的權重。

---

## 相關頁面
- [[ch06-deep-neural-networks|第六章 深度神經網路講義摘要]]
- [[error-functions-and-output-units|誤差函數與輸出單元]]
- [[multilayer-feedforward-networks|多層前饋神經網路]]
