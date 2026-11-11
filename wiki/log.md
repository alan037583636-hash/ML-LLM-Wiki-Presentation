# Log

## [2026-11-11] schema | 初始化專案
建立 AGENTS.md、raw/、wiki/、assets/ 結構；5 份講義 PDF 重新命名並移入 raw/。


## [2026-11-11] schema | 講義改為連號命名
PDF 改名為 lecture-01～05；AGENTS.md 更新命名與原章號對照（習題仍以原章號稱呼）。

## [2026-11-11] schema | 還原為原章號命名
依使用者要求，維持講義原章號（Ch1、4、5、6、7）：PDF 改為 ch01、ch04、ch05、ch06、ch07，AGENTS.md 同步還原，以便對應講義內文與習題編號。

## [2026-11-11] ingest | ch04 單層網路：迴歸 (核心觀念與推導)
- 來源: raw/ch04-single-layer-regression.pdf (Bishop 2024 Ch4, 21頁)
- 建立 source 頁: [[ch04-single-layer-regression]]
- 建立 concepts 頁面:
  - [[linear-basis-function-models]]: 線性基底函數模型、四種基底與單層網路結構
  - [[maximum-likelihood-regression]]: 高斯雜訊概似函數、正規方程推導、偏置與殘差變異數解
  - [[geometry-of-least-squares]]: N 維空間正交投影幾何、帽子矩陣性質、退化與 SVD
  - [[regularized-least-squares-ch04]]: 脊迴歸閉式解、數值/幾何/貝氏三種詮釋
  - [[multiple-output-regression]]: 多重輸出迴歸與矩陣偽反解耦性質
  - [[decision-theory-regression]]: 迴歸決策論、變分法推導條件均值、Minkowski 損失族
  - [[bias-variance-tradeoff]]: 頻率學派系集思想實驗、偏差^2–變異–雜訊分解推導、正則化係數調控
- 建立 algorithms 頁面:
  - [[sequential-learning-lms]]: 隨機梯度下降、LMS 演算法與適應性濾波/Kalman 關係
- 更新 [[index]]: 納入 Ch4 新建之 8 篇 Markdown 筆記。
- 遵循指示：完全排除課後習題與程式碼範例。

## [2026-11-11] schema | 調整匯入規範（簡化習題與實作）
- 依使用者指示更新 AGENTS.md：因運算資源有限，後續所有講義匯入完全忽略且跳過「課後習題」、「解答推導」與「程式碼實作」，專注擷取各章的核心觀念、名詞解釋與數學公式。

## [2026-11-11] ingest | ch05 單層網路：分類 (核心觀念與推導)
- 來源: raw/ch05-single-layer-classification.pdf (Bishop 2024 Ch5, 28頁)
- 建立 source 頁: [[ch05-single-layer-classification]]
- 建立 concepts 頁面:
  - [[classification-approaches]]: 分類問題的三種取徑比較、地圖與保留後驗機率的好處
  - [[linear-discriminant-functions]]: 兩類與多類線性判別函數幾何、凸決策區域、最小平方分類的不穩健性
  - [[decision-theory-classification]]: 最小錯誤率、損失矩陣、拒絕選項、混淆矩陣、精確率/召回率、ROC 與 AUC
  - [[generative-classifiers]]: 邏輯 S 形與 Softmax 由來、高斯類別條件密度推導、LDA/QDA、樸素貝氏與指數族
  - [[discriminative-classifiers-logistic-regression]]: 邏輯迴歸與二元交叉熵、多類別 Softmax 迴歸、極簡梯度推導、線性可分過度擬合
  - [[probit-regression-canonical-link]]: 隨機雜訊閾值模型、Probit 迴歸、典則連結函數理論與三組天生一對
- 更新 [[index]]: 加入 Ch5 的 7 篇新 Markdown 頁面。
- 遵循策略規範：完全跳過課後習題與程式碼實作。


## [2026-11-11] ingest | ch06 深度神經網路 (核心觀念與推導)
- 來源: raw/ch06-deep-neural-networks.pdf (Bishop 2024 Ch6, 26頁)
- 建立 source 頁: [[ch06-deep-neural-networks]]
- 建立 concepts 頁面:
  - [[curse-of-dimensionality-data-manifold]]: 維度災難、超球體積表面集中、高斯機率質量薄殼、RBF/SVM 局部泛化困境與資料流形假設
  - [[multilayer-feedforward-networks]]: 多層前饋網路矩陣形式、激活函數族譜（Sigmoid/Tanh/ReLU/Softplus/Leaky ReLU）、通用近似定理與局限、權重空間對稱性 ($M! 2^M$)
  - [[deep-representations-and-transfer-learning]]: 分散式表徵、階層式特徵抽象、遷移學習/線性探測/微調、多任務與元學習、對比式學習 InfoNCE 與 CLIP、DAG/張量
  - [[error-functions-and-output-units]]: 最大概似視角下預測分布、輸出激活與損失函數的典則配對、極簡前激活梯度 $\frac{\partial E_n}{\partial a_k} = y_k - t_k$ 證明與抗梯度消失機制
  - [[mixture-density-networks]]: 逆問題與多峰條件分布、混合密度網路 (MDN) 架構、參數約束、負對數概似三組前激活梯度公式、預測分布與條件眾數
- 更新 [[index]]: 加入 Ch6 的 6 篇新 Markdown 頁面。
- 遵循策略規範：完全跳過課後習題與程式碼實作。
