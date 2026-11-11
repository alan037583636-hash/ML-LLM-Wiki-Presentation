---
title: 第 5 章 單層網路：分類 (Chapter 5: Single-Layer Networks: Classification)
type: source
tags: [source, ch05, classification, logistic-regression, softmax, discriminant-functions, decision-theory, roc-auc, canonical-link]
sources: [ch05-single-layer-classification]
created: 2026-11-11
updated: 2026-11-11
---

# 第 5 章 單層網路：分類 (Chapter 5: Single-Layer Networks: Classification)

## 基本資料

| 項目 | 內容 |
|---|---|
| 檔案 | `raw/ch05-single-layer-classification.pdf`（28 頁，文字型 PDF） |
| 講義 | 《深度學習講義》第五章，王和盛（國立臺灣海洋大學通訊與導航工程學系），版本 2026 年 8 月 31 日 |
| 依據 | C. M. Bishop & H. Bishop, *Deep Learning: Foundations and Concepts* (Springer, 2024) 第五章整理 |
| 頁碼 | 講義頁碼與 PDF 頁碼一致（「第 N 頁，共 28 頁」） |

(來源: [[ch05-single-layer-classification]], p.1)

## 學習目標（📌講義）

1. 理解分類問題的三種取徑——判別函數、生成式機率模型、判別式機率模型——及其優缺點與適用時機。
2. 掌握線性判別函數的幾何（權重向量決定方向、偏置決定位置、$y(\mathbf{x})/\|\mathbf{w}\|$ 為帶號距離）、多類別判別的歧義區問題，以及最小平方分類法對離群值不穩健的原因。
3. 以決策論分析分類：最小錯誤分類率準則、損失矩陣與期望損失、拒絕選項、混淆矩陣與精確率／召回率／ROC 曲線／AUC／F 分數。
4. 推導生成式分類器：共用共變異數的高斯類別條件密度導出 S 形／softmax 作用在 $\mathbf{x}$ 之線性函數上的後驗機率，並以最大概似求出 $\pi, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}$；理解指數族的一般結果與樸素貝氏模型。
5. 推導判別式分類器：邏輯迴歸與交叉熵誤差、梯度 $\sum_n(y_n - t_n)\boldsymbol{\phi}_n$ 的簡潔形式、多類別 softmax 迴歸、probit 迴歸，以及典則連結函數的一般理論。

(來源: [[ch05-single-layer-classification]], p.1)

## 章節大綱與對應 wiki 頁面

| 講義章節 | PDF 頁 | 核心內容 | 對應頁面 |
|---|---|---|---|
| 引言 | p.2 | 分類定義、決策面、分類問題的三種取徑比較與地圖 | [[classification-approaches]] |
| 5.1 判別函數 | p.3–6 | 線性判別幾何、兩類與多類、凸決策區域、最小平方分類的不穩健性 | [[linear-discriminant-functions]] |
| 5.2 決策論 | p.6–12 | 最小錯誤率、損失矩陣、拒絕選項、保留後驗的好處、混淆矩陣、ROC/AUC | [[decision-theory-classification]] |
| 5.3 生成式分類器 | p.12–16 | 邏輯 S 形與 softmax 由來、高斯類別條件密度推導、最大概似解、指數族一般性 | [[generative-classifiers]] |
| 5.4.1–5.4.4 判別式分類器 | p.16–20 | 廣義線性模型、固定基底、邏輯迴歸與二元交叉熵、多類 softmax 迴歸、線性可分過度擬合 | [[discriminative-classifiers-logistic-regression]] |
| 5.4.5–5.4.6 Probit 與典則連結 | p.20–22 | Probit 迴歸、隨機閾值模型、典則連結函數推導、三組天生一對 | [[probit-regression-canonical-link]] |

## 產出／更新的頁面

- Concepts（6 頁）：
  - [[classification-approaches]] — 分類問題的三種取徑（判別函數、生成式、判別式）
  - [[linear-discriminant-functions]] — 線性判別函數的幾何與最小平方分類的限制
  - [[decision-theory-classification]] — 分類的決策論與效能評估（混淆矩陣、ROC/AUC）
  - [[generative-classifiers]] — 生成式分類器（高斯條件密度、LDA/QDA、樸素貝氏）
  - [[discriminative-classifiers-logistic-regression]] — 判別式分類器：邏輯迴歸與 Softmax 迴歸
  - [[probit-regression-canonical-link]] — Probit 迴歸與典則連結函數理論

## 相關頁面

[[index]] · [[overview]] · [[log]] · [[ch04-single-layer-regression]]
