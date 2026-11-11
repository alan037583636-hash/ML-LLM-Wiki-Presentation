---
title: 第 1 章 深度學習革命 (Chapter 1: The Deep Learning Revolution)
type: source
tags: [source, ch01, deep-learning, polynomial-fitting, history]
sources: [ch01-deep-learning-revolution]
created: 2026-11-11
updated: 2026-11-11
---

# 第 1 章 深度學習革命 (Chapter 1: The Deep Learning Revolution)

## 基本資料

| 項目 | 內容 |
|---|---|
| 檔案 | `raw/ch01-deep-learning-revolution.pdf`（21 頁，文字型 PDF，圖表為向量繪圖） |
| 講義 | 《深度學習講義》第一章，王和盛（國立臺灣海洋大學通訊與導航工程學系），版本 2026 年 8 月 26 日 |
| 依據 | C. M. Bishop & H. Bishop, *Deep Learning: Foundations and Concepts* (Springer, 2024) 第一章整理 |
| 頁碼 | 講義頁碼與 PDF 頁碼一致（「第 N 頁，共 21 頁」） |

(來源: [[ch01-deep-learning-revolution]], p.1)

> [!note] 關於講義內「Ch. N」的章號
> 講義內文提到的 Ch. 2（機率論）、Ch. 4（線性模型）、Ch. 5（決策論）、Ch. 7（梯度下降）、Ch. 8（反向傳播）、Ch. 9（正則化）、Ch. 10（CNN）、Ch. 12（Transformer／LLM）等，皆為**講義引用原書章號**，未必對應本講義 Ch1、4、5、6、7 的章號（本講義缺第 2、3 章）。各頁已標註「（講義引用原書 Ch. X，未必對應本講義 ChX）」，不補造缺漏章節。(來源: [[ch01-deep-learning-revolution]], p.2)

## 學習目標（📌講義）

1. 透過四個跨領域案例，理解深度學習作為通用學習框架的廣度，掌握監督式、非監督式、自監督式學習與遷移學習等基本詞彙。
2. 以多項式曲線擬合為教學範例，建立線性模型、誤差函數、最小平方閉式解、過度擬合、正則化（權重衰減）、超參數與交叉驗證等核心觀念，並完成相關數學推導。
3. 沿著單層網路→反向傳播→深度網路三階段，理解類神經網路的歷史脈絡，以及規模化（資料、參數、算力）在當代深度學習中的角色。
4. 能將本章觀念連結到後續章節（機率論、決策論、梯度優化、反向傳播、CNN、Transformer 與大型語言模型）。

(來源: [[ch01-deep-learning-revolution]], p.1)

## 章節大綱與對應 wiki 頁面

| 講義章節 | PDF 頁 | 內容 | 對應頁面 |
|---|---|---|---|
| 引言 | p.2 | 從手工演算法到從資料中學習；ML／DL／AI／AGI／LLM | [[learning-paradigms]] |
| 1.1.1 醫學診斷 | p.2–3 | 皮膚癌影像分類、監督式、分類／迴歸、遷移學習 | [[learning-paradigms]]、[[transfer-learning]] |
| 1.1.2 蛋白質結構預測 | p.3 | AlphaFold，序列→3D 結構 | [[learning-paradigms]] |
| 1.1.3 影像合成 | p.3 | 非監督式、生成模型、生成式 AI | [[learning-paradigms]] |
| 1.1.4 大型語言模型 | p.4 | 自迴歸、自監督、三種典範對照表 | [[learning-paradigms]]、[[supervised-vs-unsupervised-vs-self-supervised]] |
| 1.2.1 合成資料 | p.5 | $\sin(2\pi x)$＋雜訊、泛化 | [[polynomial-curve-fitting]] |
| 1.2.2 線性模型 | p.5 | 對參數線性、基底函數 | [[linear-models-basis-functions]] |
| 1.2.3 誤差函數 | p.6 | 平方和誤差、正規方程推導 | [[least-squares-normal-equations]] |
| 1.2.4 模型複雜度 | p.7–8 | 過度擬合、RMS、係數量級、資料量 | [[generalization-overfitting]] |
| 1.2.5 正則化 | p.8–10 | 權重衰減、脊迴歸、MAP 詮釋 | [[regularization-weight-decay]] |
| 1.2.6 模型選擇 | p.10–11 | 超參數、驗證集、交叉驗證 | [[hyperparameters-model-selection]]、[[cross-validation]] |
| 1.3.1 神經元 | p.12 | $a=\sum w_ix_i$，$y=f(a)$ | [[artificial-neuron-perceptron]] |
| 1.3.2 第一階段：單層網路 | p.13 | 感知器、Minsky–Papert、XOR | [[artificial-neuron-perceptron]] |
| 1.3.3 第二階段：反向傳播 | p.13–14 | 可微激活、SGD、歸納偏差 | [[history-of-neural-networks]] |
| 1.3.4 第三階段：深度網路 | p.14–16 | 規模、GPU、表徵學習、基礎模型、殘差、自動微分 | [[scaling-and-representation-learning]]、[[three-eras-of-neural-networks]] |
| 課後習題 | p.16–18 | 12 題 | 見下表 |
| 附錄 A、B | p.19–21 | 中英名詞對照表、延伸閱讀 | 名詞已併入各頁 |

## 關鍵要點（📌講義）

1. **同一框架解多領域問題**：皮膚癌、蛋白質、影像合成、LLM 皆為深度學習的變體。(來源: [[ch01-deep-learning-revolution]], p.2)
2. **三種學習典範**：監督式（輸入＋標註）、非監督式（無標籤）、自監督式（目標由輸入自動導出）；遷移學習跨越典範。(來源: [[ch01-deep-learning-revolution]], p.4)
3. **線性模型**：對參數線性，$y=\mathbf w^T\boldsymbol\phi(x)$；平方和誤差有閉式解 $\mathbf w^\star=(\Phi^T\Phi)^{-1}\Phi^T\mathbf t$。(來源: [[ch01-deep-learning-revolution]], p.5–6)
4. **過度擬合**：模型越靈活越擬合雜訊；訓練誤差為 0 不代表泛化好；資料越多越能負擔複雜模型。(來源: [[ch01-deep-learning-revolution]], p.7–8)
5. **正則化**：$\widetilde E=E+\frac\lambda2\|\mathbf w\|^2$，閉式解 $(\Phi^T\Phi+\lambda I)^{-1}\Phi^T\mathbf t$；等價於高斯先驗下的 MAP，$\lambda=\alpha\sigma^2$。(來源: [[ch01-deep-learning-revolution]], p.9)
6. **超參數需另行決定**：訓練／驗證／測試三份資料，資料少用交叉驗證。(來源: [[ch01-deep-learning-revolution]], p.10–11)
7. **多項式是神經元特例**：線性模型與神經網路的分野是基底是否可學習、$f$ 是否非線性。(來源: [[ch01-deep-learning-revolution]], p.12)
8. **三階段歷史**：單層網路→反向傳播→深度網路；規模化、表徵學習、殘差連結、自動微分、基礎模型。(來源: [[ch01-deep-learning-revolution]], p.12–16)

## 習題清單與對應頁面

講義的 12 題皆已在對應頁面提供解答／推導與（適用時）NumPy 實作，**解答皆標為 💡補充**。(來源: [[ch01-deep-learning-revolution]], p.16–18)

| 習題 | 類型 | 題目重點 | PDF 頁 | 解答所在頁面 |
|---|---|---|---|---|
| 習題 1 | ◦觀念 | 學習典範辨識（AIS、衛星影像分群、GNSS 欺騙、VTEC） | p.16 | [[learning-paradigms]] |
| 習題 2 | ⋆推導 | 正規方程、Vandermonde、$N=M+1$ 時 $E=0$ | p.16–17 | [[least-squares-normal-equations]] |
| 習題 3 | ⋆推導 | 正則化閉式解與收縮因子 $\mu_j/(\mu_j+\lambda)$ | p.17 | [[regularization-weight-decay]] |
| 習題 4 | ⋆推導 | RMS 誤差與尺度 | p.17 | [[generalization-overfitting]] |
| 習題 5 | ⋆推導 | 權重衰減的 MAP 詮釋、LASSO | p.17 | [[regularization-weight-decay]] |
| 習題 6 | ◦觀念 | 超參數不能用訓練誤差決定 | p.17 | [[hyperparameters-model-selection]] |
| 習題 7 | ⋆推導 | 交叉驗證成本、留一法捷徑（Sherman–Morrison） | p.17 | [[cross-validation]] |
| 習題 8 | ⋆推導 | 多項式是神經元的特例（tanh 變體） | p.17 | [[artificial-neuron-perceptron]] |
| 習題 9 | ⋆推導 | 感知器與 XOR（含兩層網路設計） | p.17–18 | [[artificial-neuron-perceptron]] |
| 習題 10 | ⋆推導 | 算力成長的算術 | p.18 | [[scaling-and-representation-learning]] |
| 習題 11 | ⋆⋆延伸 | NumPy 重現圖 1–7（含 Fourier 基底） | p.18 | [[polynomial-curve-fitting]] |
| 習題 12 | ⋆⋆延伸 | 多參數也能泛化？double descent | p.18 | [[generalization-overfitting]] |

## 產出／更新的頁面

- Concepts（10）：[[learning-paradigms]]、[[transfer-learning]]、[[polynomial-curve-fitting]]、[[linear-models-basis-functions]]、[[generalization-overfitting]]、[[regularization-weight-decay]]、[[hyperparameters-model-selection]]、[[artificial-neuron-perceptron]]、[[history-of-neural-networks]]、[[scaling-and-representation-learning]]
- Algorithms（2）：[[least-squares-normal-equations]]、[[cross-validation]]
- Comparisons（2）：[[supervised-vs-unsupervised-vs-self-supervised]]、[[three-eras-of-neural-networks]]
- 圖表：`assets/ch01/`（SVG 重繪圖 1、3–11；另增補學習典範總覽、XOR、double descent 圖；產生程式 `make_figures.py`）

## 發現的差異與備註

> [!warning] 差異：圖 11 縱軸刻度與正文數值不一致（講義內部）
> 圖 11 縱軸標示 $10^{-14}\sim10^{2}$ petaflop/s-days，正文卻說圖上最高處為 $10^{24}$ 次浮點運算（$\approx1.2\times10^4$ petaflop/s-days）。不自行裁決，詳見 [[scaling-and-representation-learning]]。(來源: [[ch01-deep-learning-revolution]], p.15, p.18)

> [!warning] 差異：古典參數／資料法則 vs 深度學習現象
> 講義並陳「參數數不應少於資料點的 5～10 倍」的古典經驗與「參數遠多於資料仍可極佳」的深度學習觀察，並以習題 12 延伸。詳見 [[generalization-overfitting]]。(來源: [[ch01-deep-learning-revolution]], p.8, p.18)

> [!note] 數值差異說明
> 講義圖與表（如表 1、表 2）由作者以 pgfplots 重新產生，數值與原書略有差異（p.5）。本 wiki 的重繪圖與 NumPy 結果則以 `seed=0` 另行產生，數值與講義表 1、表 2 不同（趨勢一致），各頁皆已註明。

## 附錄 A 名詞對照（📌講義，節錄）

machine learning 機器學習；deep learning 深度學習；supervised／unsupervised／self-supervised 監督式／非監督式／自監督式；generalization 泛化；over-fitting 過度擬合；regularization 正則化；weight decay 權重衰減；hyperparameter 超參數；cross-validation 交叉驗證；perceptron 感知器；feed-forward network 前饋網路；backpropagation 反向傳播；foundation model 基礎模型；residual connection 殘差連結。(來源: [[ch01-deep-learning-revolution]], p.19–20)

## 相關頁面

[[index]] · [[overview]] · [[log]]
