---
title: 神經網路三階段比較 (Three Eras of Neural Networks)
type: comparison
tags: [history, perceptron, backpropagation, deep-learning]
sources: [ch01-deep-learning-revolution]
created: 2026-11-11
updated: 2026-11-11
---

# 神經網路三階段比較 (Three Eras of Neural Networks)

## 1. 講義的三階段總覽

📌講義：

| 階段 | 年代 | 代表性技術 | 瓶頸／突破 |
|---|---|---|---|
| 單層網路 | 1940s–1960s | McCulloch–Pitts 神經元、感知器 | 只能學線性可分問題；Minsky–Papert（1969）後陷入低潮 |
| 反向傳播 | 1980s–2000s | 可微激活、平方和誤差、反向傳播、SGD、CNN | 深層難訓練，仰賴手工特徵；2000 年代讓位給核方法／SVM |
| 深度網路 | 2010s– | GPU、大資料、殘差連結、自動微分、Transformer、基礎模型 | 規模化→表徵學習、遷移學習、LLM |

(來源: [[ch01-deep-learning-revolution]], p.16)

## 2. 延伸比較（💡補充）

下列「可學習層數」「訓練方式」等欄位由講義內文（p.13–16）整理而成，欄位組織為 agent 補充。

| 面向 | 單層網路 | 反向傳播時代 | 深度網路 |
|---|---|---|---|
| 可學習的權重層 | 1 層 | 多層，但實際有效的僅最後兩層 (p.14) | 多層權重，可有效訓練 (p.14) |
| 激活函數 | 階梯函數 (7) | 連續可微、梯度非零 | 同左，並配合殘差連結等 |
| 訓練演算法 | Rosenblatt 感知器演算法（只適用單層） | 反向傳播 + SGD | 反向傳播 + SGD + 自動微分 |
| 特徵 | 固定輸入 | 手工特徵萃取 | 表徵由資料學得 |
| 規模（參數） | 極小 | 數百～數千（1980s）(p.15) | 百萬、十億，約 $10^{12}$ (p.15) |
| 硬體 | 類比（Mark 1 感知器，p.13） | 一般 CPU | GPU，數千顆組成陣列 (p.15) |
| 資料 | 小 | 中 | 巨大 |
| 對應頁面 | [[artificial-neuron-perceptron]] | [[history-of-neural-networks]] | [[scaling-and-representation-learning]] |

> [!note] 各欄頁碼
> 表中括號內頁碼皆為講義 PDF 頁碼，來源: [[ch01-deep-learning-revolution]]。「CPU」「資料量中等」為 💡補充的概括，講義未明述。

## 3. 一句話總結（📌講義）

單層網路的限制由 Minsky–Papert 證明、反向傳播解決了多層訓練、規模化（資料、參數、算力）加上表徵學習讓深度網路成為通用框架。(綜合來源: [[ch01-deep-learning-revolution]], p.13–16)

## 相關頁面

[[history-of-neural-networks]] · [[artificial-neuron-perceptron]] · [[scaling-and-representation-learning]] · [[learning-paradigms]]
