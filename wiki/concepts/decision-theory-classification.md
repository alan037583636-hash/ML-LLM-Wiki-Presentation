---
title: 分類的決策論與評估指標 (Decision Theory for Classification and Evaluation Metrics)
type: concept
tags: [decision-theory, loss-matrix, reject-option, confusion-matrix, roc-curve, auc, f-score]
sources: [ch05-single-layer-classification]
created: 2026-11-11
updated: 2026-11-11
---

# 分類的決策論與評估指標 (Decision Theory for Classification and Evaluation Metrics)

## 1. 決策準則與期望損失

📌講義：在推論階段求得後驗機率 $p(\mathcal{C}_k \mid \mathbf{x})$ 後，決策論指導我們如何制定最佳指派規則。

### 1.1 最小錯誤分類率準則 (Minimum Misclassification Rate)
若目標是盡可能減少分類錯誤的次數：
兩類問題中犯錯的機率為：

$$p(\text{mistake}) = \int_{\mathcal{R}_1} p(\mathbf{x}, \mathcal{C}_2) \, d\mathbf{x} + \int_{\mathcal{R}_2} p(\mathbf{x}, \mathcal{C}_1) \, d\mathbf{x} \tag{5.16}$$

消去共同因子 $p(\mathbf{x})$ 後可證明：**最小化錯誤率的規則，是將每個 $\mathbf{x}$ 指派給後驗機率 $p(\mathcal{C}_k \mid \mathbf{x})$ 最大的類別**。(來源: [[ch05-single-layer-classification]], p.7)

### 1.2 損失矩陣與最小期望損失 (Loss Matrix and Expected Loss)
不同錯誤的後果往往天差地別（如將癌症誤診為正常，代價遠高於正常誤診為癌症）。
引入損失矩陣 $\mathbf{L}$，其元素 $L_{kj}$ 表示真實類別為 $\mathcal{C}_k$ 卻誤判為 $\mathcal{C}_j$ 時所招致的損失。平均期望損失為：

$$\mathbb{E}[L] = \sum_k \sum_j \int_{\mathcal{R}_j} L_{kj} p(\mathbf{x}, \mathcal{C}_k) \, d\mathbf{x} \tag{5.19}$$

**最佳決策規則**：對每個新輸入 $\mathbf{x}$，選擇使條件風險最小的類別 $j$：

$$\arg\min_j \sum_k L_{kj} p(\mathcal{C}_k \mid \mathbf{x}) \tag{5.20}$$

(來源: [[ch05-single-layer-classification]], p.7–8)

### 1.3 拒絕選項 (Reject Option)
在決策邊界附近，最大後驗機率明顯偏低（分類信心不足）。引入閾值 $\theta \in [1/K, 1]$：
- 若 $\max_k p(\mathcal{C}_k \mid \mathbf{x}) \ge \theta$，正常做出分類決策。
- 若 $\max_k p(\mathcal{C}_k \mid \mathbf{x}) < \theta$，啟動**拒絕選項 (reject option)**，不作判斷，交由人工專家或進一步檢驗處理。(來源: [[ch05-single-layer-classification]], p.8)

## 2. 分類器效能評估指標

### 2.1 混淆矩陣 (Confusion Matrix)
以二元分類為例，樣本總數 $N = N_{TP} + N_{FP} + N_{TN} + N_{FN}$：

| | 預測為陰性 (Negative) | 預測為陽性 (Positive) |
|---|---|---|
| **真實為陰性** | 真陰性 ($N_{TN}$) | 偽陽性 ($N_{FP}$，型一錯誤) |
| **真實為陽性** | 偽陰性 ($N_{FN}$，型二錯誤) | 真陽性 ($N_{TP}$) |

(來源: [[ch05-single-layer-classification]], p.10)

### 2.2 核心指標定義
📌講義：
- **準確率 (Accuracy)**：
  $$\text{Accuracy} = \frac{N_{TP} + N_{TN}}{N} \tag{5.25}$$
  （類別極度不平衡時，準確率會產生嚴重誤導）。
- **精確率 (Precision)**：預測為陽性中實際為陽性的比例：
  $$\text{Precision} = \frac{N_{TP}}{N_{TP} + N_{FP}} \tag{5.26}$$
- **召回率 (Recall) / 靈敏度 (Sensitivity)**：真實陽性被正確檢出的比例：
  $$\text{Recall} = \frac{N_{TP}}{N_{TP} + N_{FN}} \tag{5.27}$$
- **偽陽性率 (False Positive Rate, FPR)**：正常個體被誤判為陽性的比例：
  $$\text{FPR} = \frac{N_{FP}}{N_{FP} + N_{TN}} \tag{5.28}$$
- **F 分數 (F-score)**：精確率與召回率的調和平均數：
  $$F = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 N_{TP}}{2 N_{TP} + N_{FP} + N_{FN}} \tag{5.30}$$

(來源: [[ch05-single-layer-classification]], p.10–11)

### 2.3 ROC 曲線與 AUC
📌講義：
- **接收者操作特性曲線 (ROC Curve)**：將機率輸出轉化為決策的分類閾值由 1 掃至 0，繪製**真陽性率 (TPR, 召回率)** 對 **偽陽性率 (FPR)** 的取捨軌跡。
- **隨機猜測線**：對角線（$\text{AUC} = 0.5$）。
- **曲線下面積 (AUC)**：量化 ROC 曲線性能的單一數值指標，$\text{AUC} \in [0.5, 1.0]$，1.0 代表完美分類器。(來源: [[ch05-single-layer-classification]], p.11)

📌講義（講義教學補充：ROC 來自雷達偵測理論）：
ROC 誕生於二戰雷達目標偵測：在雜訊中判定「有無目標」，對應偵測機率 $P_d$（真陽性率）對虛警機率 $P_{fa}$（偽陽性率）。Neyman–Pearson 準則（固定虛警率最大化偵測率）等價於在 ROC 曲線上尋找工作點；GNSS 欺騙偵測與訊號捕獲門檻也是同一理論的應用。(來源: [[ch05-single-layer-classification]], p.12)

## 自我測驗 Q&A

> [!question] Q1. 為什麼在極度不平衡的資料中，準確率 (Accuracy) 不是一個好的指標？
> [!answer]- 答案
> 若負樣本佔 99.9%，分類器即便無腦全猜陰性，準確率也高達 99.9%，但無法找出任何真正需要偵測的陽性樣本。(來源: [[ch05-single-layer-classification]], p.10)

> [!question] Q2. 什麼是拒絕選項？它的使用時機是什麼？
> [!answer]- 答案
> 當模型預測的最大後驗機率低於設定的信心閾值時放棄自動指派；適用於醫療或高風險場景，將高不確定性樣本交由人工複檢以降低嚴重誤診。(來源: [[ch05-single-layer-classification]], p.8)

> [!question] Q3. ROC 曲線的橫軸與縱軸各是什麼？
> [!answer]- 答案
> 橫軸為偽陽性率 (FPR)，縱軸為真陽性率 (TPR / 召回率)。(來源: [[ch05-single-layer-classification]], p.11)

## 相關頁面

[[classification-approaches]] · [[decision-theory-regression]] · [[discriminative-classifiers-logistic-regression]]
