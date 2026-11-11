---
title: 偏差–變異權衡（第四章） (Bias-Variance Trade-off - Chapter 4)
type: concept
tags: [bias-variance, generalization, overfitting, underfitting, model-complexity, regularization]
sources: [ch04-single-layer-regression]
created: 2026-11-11
updated: 2026-11-11
---

# 偏差–變異權衡（第四章） (Bias-Variance Trade-off - Chapter 4)

## 1. 頻率學派的觀點與思想實驗

📌講義：從頻率學派的觀點來看，模型複雜度問題可透過「偏差–變異權衡 (bias–variance trade-off)」獲得深入洞見。
設想一個思想實驗：
假設有大量互相獨立的資料集 $\mathcal{D}$，每個資料集大小均為 $N$，且皆從真實聯合分布 $p(\mathbf{x}, t)$ 獨立抽取。
對每個資料集 $\mathcal{D}$ 執行學習演算法，將得到一個特定的預測函數 $f(\mathbf{x}; \mathcal{D})$。評估該學習演算法的整體表現，即評估其在整個資料集系集 (ensemble) 上的平均期望損失。(來源: [[ch04-single-layer-regression]], p.12–13)

## 2. 偏差–變異分解的嚴格推導

📌講義：由決策論已知，平方損失下的真實最佳預測是條件期望值 $h(\mathbf{x}) \equiv \mathbb{E}[t \mid \mathbf{x}]$。期望平方損失為：

$$\mathbb{E}[L] = \int \{f(\mathbf{x}) - h(\mathbf{x})\}^2 p(\mathbf{x}) \, d\mathbf{x} + \iint \{h(\mathbf{x}) - t\}^2 p(\mathbf{x}, t) \, d\mathbf{x} dt \tag{4.39}$$

第二項是資料的固有雜訊。聚焦於第一項中的被積函數 $\{f(\mathbf{x}; \mathcal{D}) - h(\mathbf{x})\}^2$，在括號內加減系集平均預測 $\mathbb{E}_{\mathcal{D}}[f(\mathbf{x}; \mathcal{D})]$：

$$\{f(\mathbf{x}; \mathcal{D}) - h(\mathbf{x})\}^2 = \Big(\{f(\mathbf{x}; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[f(\mathbf{x}; \mathcal{D})]\} + \{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x}; \mathcal{D})] - h(\mathbf{x})\}\Big)^2$$

展開後對所有資料集 $\mathcal{D}$ 取期望值 $\mathbb{E}_{\mathcal{D}}[\cdot]$。由於交叉項中 $\mathbb{E}_{\mathcal{D}}[f(\mathbf{x}; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[f(\mathbf{x}; \mathcal{D})]] \equiv 0$，交叉項整項消失，得到：

$$\mathbb{E}_{\mathcal{D}}\big[\{f(\mathbf{x}; \mathcal{D}) - h(\mathbf{x})\}^2\big] = \underbrace{\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x}; \mathcal{D})] - h(\mathbf{x})\}^2}_{(\text{偏差})^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\big[\{f(\mathbf{x}; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[f(\mathbf{x}; \mathcal{D})]\}^2\big]}_{\text{變異}} \tag{4.42}$$

代回全域期望損失，得到完整的**偏差–變異分解 (bias–variance decomposition)**：

$$\mathbb{E}[L] = (\text{偏差})^2 + \text{變異} + \text{雜訊} \tag{4.43}$$

其中各積分分量為：
- **平方偏差 (Squared bias)**：
  $$(\text{偏差})^2 = \int \{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x}; \mathcal{D})] - h(\mathbf{x})\}^2 p(\mathbf{x}) \, d\mathbf{x} \tag{4.44}$$
- **變異 (Variance)**：
  $$\text{變異} = \int \mathbb{E}_{\mathcal{D}}\big[\{f(\mathbf{x}; \mathcal{D}) - \mathbb{E}_{\mathcal{D}}[f(\mathbf{x}; \mathcal{D})]\}^2\big] p(\mathbf{x}) \, d\mathbf{x} \tag{4.45}$$
- **雜訊 (Noise)**：
  $$\text{雜訊} = \iint \{h(\mathbf{x}) - t\}^2 p(\mathbf{x}, t) \, d\mathbf{x} dt \tag{4.46}$$

(來源: [[ch04-single-layer-regression]], p.13)

## 3. 三個分量的物理意義與責任歸屬

📌講義（三個分量各自「怪誰」）：

| 分量 | 物理本質與來源 | 如何降低 |
|---|---|---|
| **$(\text{偏差})^2$** | 模型表達力太僵硬（欠擬合），即便平均所有資料集的解也逼近不了真實 $h(\mathbf{x})$ | 增加模型彈性（更多基底、神經網路層數、縮小正則化 $\lambda$） |
| **變異** | 模型過度靈活，對「特定這一份」訓練資料的隨機雜訊過度敏感（過度擬合） | 擴增訓練資料、增大正則化係數 $\lambda$、採用系集平均 (Ensemble) |
| **雜訊** | 資料固有的條件變異數 $\text{var}[t \mid \mathbf{x}]$ | **不可縮減**（除非獲取新的輸入特徵以減少不確定性） |

(來源: [[ch04-single-layer-regression]], p.14)

## 4. 正則化係數 $\lambda$ 對權衡的調控

📌講義：
- **正則化係數 $\lambda$ 很大時**：權重被強烈壓向零，模型高度僵硬。此時在不同資料集訓練出的曲線彼此極為相似（**低變異**），但整體平均完全偏離真實正弦曲線（**高偏差**）。
- **正則化係數 $\lambda$ 很小時**：模型高度靈活，平均解雖然精準貼合真實曲線（**低偏差**），但在不同資料集擬合出的個別曲線劇烈震盪（**高變異**）。
- 泛化誤差（測試誤差）的最低點出現在 $(\text{偏差})^2 + \text{變異}$ 總和最小的平衡處。
- **測試誤差曲線的平移現象**：測試誤差曲線剛好浮在 $[(\text{偏差})^2 + \text{變異}]$ 曲線之上，二者高度差即為常數固有雜訊 $\sigma^2$（式 4.43 的數值驗證）。(來源: [[ch04-single-layer-regression]], p.14–16)

📌講義備註：偏差–變異分解在實務上無法精確計算（因我們只有單一訓練集），但它為模型複雜度與正則化取捨提供了深刻的理論指導。此外，過度擬合是頻率學派最大概似估計的特徵，在貝氏完全邊際化參數的架構下並不會出現。(來源: [[ch04-single-layer-regression]], p.12, p.16)

## 自我測驗 Q&A

> [!question] Q1. 偏差–變異分解中的「變異 (Variance)」衡量的是什麼？
> [!answer]- 答案
> 衡量個別資料集所訓練出的預測函數 $f(\mathbf{x}; \mathcal{D})$，圍繞系集平均預測 $\mathbb{E}_{\mathcal{D}}[f]$ 波動分散的程度，反映模型對訓練資料隨機抽樣變異的敏感度。(來源: [[ch04-single-layer-regression]], p.13)

> [!question] Q2. 為什麼無論怎樣增加模型複雜度或收集無窮多資料，測試平方誤差都無法降到 0？
> [!answer]- 答案
> 因為存在固有的不可縮減雜訊項 $\iint \{h(\mathbf{x}) - t\}^2 p(\mathbf{x}, t) d\mathbf{x} dt = \int \text{var}[t \mid \mathbf{x}] p(\mathbf{x}) d\mathbf{x}$。(來源: [[ch04-single-layer-regression]], p.13, p.16)

> [!question] Q3. 增大 L2 正則化係數 $\lambda$ 會如何改變偏差與變異？
> [!answer]- 答案
> 增大 $\lambda$ 會約束權重自由度，降低模型對個別資料雜訊的敏感度（**降低變異**），但代價是使模型更加僵硬、偏離真實函數（**提高偏差**）。(來源: [[ch04-single-layer-regression]], p.14)

## 相關頁面

[[decision-theory-regression]] · [[regularized-least-squares-ch04]] · [[generalization-overfitting]] · [[ch04-single-layer-regression]]
