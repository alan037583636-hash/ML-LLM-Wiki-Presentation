---
title: 網路正規化技術 (Normalization Techniques)
type: concept
tags: [normalization, batch-normalization, layer-normalization, internal-covariate-shift, optimization]
sources: [ch07-gradient-descent]
created: 2026-11-11
updated: 2026-11-11
---

# 網路正規化技術 (Normalization Techniques)

隨著神經網路層數加深，前向傳播的激活值分布與反向傳播的梯度極易發生數值漂移，導致訓練極度困難。正規化技術（Normalization）透過重塑輸入與隱藏層的統計特徵，大幅平滑了誤差曲面並改善條件數。

---

## 1. 輸入資料標準化 (Input Normalization)

📌講義指出，在訓練任何機器學習模型之前，對原始特徵進行標準化（Standardization / Z-score Normalization）是基礎且關鍵的一步：
$$\hat{x}_{ni} = \frac{x_{ni} - \mu_i}{\sigma_i}$$
其中 $\mu_i = \frac{1}{N}\sum_{n=1}^N x_{ni}$，$\sigma_i^2 = \frac{1}{N}\sum_{n=1}^N (x_{ni} - \mu_i)^2$。

### 幾何與收斂性意義
- 若未正規化，特徵間的量綱尺度差異懸殊（例如特徵 1 範圍 $[0, 1]$，特徵 2 範圍 $[1000, 10000]$），對應的 Hessian 矩陣特徵值將極度極化，條件數 $\kappa \gg 1$，誤差曲面呈現極度狹長扁平的峽谷。
- 經過標準化後，各特徵維度的方差對齊，誤差曲面的等高線被「重塑為接近對稱超球體」，Hessian 條件數 $\kappa \approx 1$，使得標準梯度下降能以近乎筆直的軌跡直奔最優解 (來源: [[ch07-gradient-descent]], p.11–12)。

---

## 2. 批次正規化 (Batch Normalization, BatchNorm)

輸入正規化僅能保證第一層網路的輸入穩定。在深層網路中，前面所有層的權重微小變動，會在後續深層引起激活值分布的劇烈累積漂移（稱為**內部共變量偏移 Internal Covariate Shift**）。Ioffe & Szegedy (2015) 提出了**批次正規化 (Batch Normalization)**，將標準化機制直接內嵌於深層神經網路的各個隱藏層中。

### 2.1 訓練期演算法 (Training Step)
考慮包含 $B$ 個樣本的小批次 $\mathcal{B} = \{x_1, \dots, x_B\}$，對隱藏層中特定的某一個神經元標量激活值：
1. **計算小批次均值**：
   $$\mu_{\mathcal{B}} = \frac{1}{B} \sum_{i=1}^B x_i$$
2. **計算小批次變異數**：
   $$\sigma_{\mathcal{B}}^2 = \frac{1}{B} \sum_{i=1}^B (x_i - \mu_{\mathcal{B}})^2$$
3. **標準化**：
   $$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$
   （$\epsilon > 0$ 為極小數值穩定常數）。
4. **可學習縮放與平移 (Scale and Shift)**：
   若僅進行標準化，神經元輸出被限制在均值 0 與方差 1，可能破壞非線性激活函數的表達能力（例如強制將 Sigmoid 侷限在線性區）。因此引入一對可學習參數 $\gamma$（縮放）與 $\beta$（平移）：
   $$y_i = \text{BN}_{\gamma, \beta}(x_i) = \gamma \hat{x}_i + \beta$$
   若模型自適應學習出 $\gamma = \sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}$ 與 $\beta = \mu_{\mathcal{B}}$，可完全恢復原始未正規化信號，確保表現力下界不衰減 (來源: [[ch07-gradient-descent]], p.12–13)。

### 2.2 推論期處理 (Inference / Evaluation Phase)
在模型部署推論時，通常以單一資料點（Batch Size = 1）即時輸入，無法計算小批次的均值與變異數。
- **解法**：在訓練階段，各層 BatchNorm 同步以**指數加權移動平均 (EMA)** 維護全資料集的總體均值 $\mu_{\text{run}}$ 與總體變異數 $\sigma^2_{\text{run}}$：
  $$\mu_{\text{run}} \gets (1 - \alpha) \mu_{\text{run}} + \alpha \mu_{\mathcal{B}}$$
  $$\sigma_{\text{run}}^2 \gets (1 - \alpha) \sigma_{\text{run}}^2 + \alpha \frac{B}{B-1}\sigma_{\mathcal{B}}^2$$
- 推論時直接代入該靜態全域參數進行確定性線性轉換：
  $$y = \gamma \left( \frac{x - \mu_{\text{run}}}{\sqrt{\sigma_{\text{run}}^2 + \epsilon}} \right) + \beta$$
  (來源: [[ch07-gradient-descent]], p.13)。

---

## 3. 層正規化 (Layer Normalization, LayerNorm)

批次正規化對小批次大小 $B$ 具有強烈依賴性（若 $B$ 極小，批次統計量估計極不穩定），且在循環神經網路（RNN）與變長序列任務中難以適用。Ba, Kiros & Hinton (2016) 提出了**層正規化 (Layer Normalization)**。

### 3.1 核心機制
LayerNorm 的統計量是在**單一資料樣本內部，跨該層的所有隱藏單元（特徵維度 $D$）**計算：
$$\mu_L = \frac{1}{D} \sum_{j=1}^D x_{ij}, \quad \sigma_L^2 = \frac{1}{D} \sum_{j=1}^D (x_{ij} - \mu_L)^2$$
$$\hat{x}_{ij} = \frac{x_{ij} - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}, \quad y_{ij} = g_j \hat{x}_{ij} + b_j$$
其中 $g_j, b_j$ 為可學習的特徵縮放與偏置參數 (來源: [[ch07-gradient-descent]], p.13–14)。

---

## 4. BatchNorm 與 LayerNorm 的切片幾何對比 (Comparison of Tensor Slicing)

📌講義以張量切片概念生動對比了兩者的本質差異 (來源: [[ch07-gradient-descent]], p.14)：
設網路激活張量形狀為 $(B, D)$（其中 $B$ 為小批次大小，$D$ 為特徵單元維度）：

| 正規化類型 | 統計聚合方向 (Aggregation Axis) | 對批次大小 $B$ 的依賴 | 訓練與推論行為差異 | 典型應用架構 |
|---|---|---|---|---|
| **Batch Normalization (BatchNorm)** | **垂直切片**：固定某個特徵通道 $d$，沿著批次維度 $B$ 取平均與變異數。 | 強烈依賴 $B$（$B \ge 32$ 效果較好，極小 Batch 表現崩潰）。 | 訓練期計算動態小批次統計量；推論期切換為靜態全域移動平均。 | 卷積神經網路 (CNN)、電腦視覺 (CV)。 |
| **Layer Normalization (LayerNorm)** | **水平切片**：固定單一資料點 $b$，沿著特徵維度 $D$ 取平均與變異數。 | 完全獨立於 $B$（即使 $B=1$ 亦完全等價運作）。 | 訓練期與推論期的計算公式與行為完全一致。 | Transformer (BERT/GPT)、大型語言模型 (LLM)、RNN。 |

---

## 自我測驗 Q&A

### Q1: 為什麼輸入特徵標準化（Zero Mean, Unit Variance）能顯著加速梯度下降的收斂過程？
> [!answer]- 答案
> 若特徵維度尺度差異過大，誤差曲面會在不同方向呈現極不均勻的曲率，導致 Hessian 矩陣的條件數 $\kappa = \lambda_{\max}/\lambda_{\min} \gg 1$。梯度下降會被最大曲率方向限制步長，在垂直山谷方向劇烈震盪，而在平緩軸方向推進極慢。輸入標準化將特徵尺度對齊，使等高線重塑為接近對稱的超球體（$\kappa \approx 1$），消除病態震盪，使梯度幾乎直指極小點。

### Q2: 批次正規化（BatchNorm）在訓練階段與推論階段的計算方式有何根本不同？
> [!answer]- 答案
> - **訓練階段**：對當前小批次的樣本即時計算均值 $\mu_{\mathcal{B}}$ 與變異數 $\sigma_{\mathcal{B}}^2$ 進行標準化，並同時在背景以指數移動平均（EMA）累積記錄全資料集的統計量 $\mu_{\text{run}}, \sigma_{\text{run}}^2$。
> - **推論階段**：不再使用當前測試批次的統計量（避免依賴測試批次大小或在單一樣本 $B=1$ 時無法計算變異數），而是固定使用訓練期累積的全局移動平均常數 $\mu_{\text{run}}, \sigma_{\text{run}}^2$ 進行確定性仿射變換。

### Q3: 為什麼在自然語言處理（NLP）與 Transformer 模型中，普遍採用 LayerNorm 而非 BatchNorm？
> [!answer]- 答案
> 1. **變長序列限制**：NLP 文本序列長度不一，BatchNorm 在 Batch 維度統計時需要複雜的 Padding 遮罩，且難以捕捉時間步間的特徵分佈。
> 2. **批次大小無關性**：大型語言模型訓練與推論的 Batch Size 常隨記憶體限制而動態變化（推論時往往是 $B=1$ 即時生成）。LayerNorm 的統計量僅在單一樣本內部所有特徵維度計算，計算完全不依賴 Batch 內的其餘樣本，訓練與推論行為完全一致。

---

## 相關頁面
- [[ch07-gradient-descent|第七章 梯度下降講義摘要]]
- [[error-surfaces-and-geometry|誤差曲面與幾何特性]]
- [[gradient-descent-and-initialization|梯度下降家族與參數初始化]]
- [[momentum-and-adaptive-optimizers|動量法與自適應學習率優化器]]
- [[optimization-algorithms-comparison|最佳化演算法綜合比較]]
