---
title: 第 4 章 單層網路：迴歸 (Chapter 4: Single-Layer Networks: Regression)
type: source
tags: [source, ch04, regression, linear-models, mle, decision-theory, bias-variance]
sources: [ch04-single-layer-regression]
created: 2026-11-11
updated: 2026-11-11
---

# 第 4 章 單層網路：迴歸 (Chapter 4: Single-Layer Networks: Regression)

## 基本資料

| 項目 | 內容 |
|---|---|
| 檔案 | `raw/ch04-single-layer-regression.pdf`（21 頁，文字型 PDF） |
| 講義 | 《深度學習講義》第四章，王和盛（國立臺灣海洋大學通訊與導航工程學系），版本 2026 年 8 月 26 日 |
| 依據 | C. M. Bishop & H. Bishop, *Deep Learning: Foundations and Concepts* (Springer, 2024) 第四章整理 |
| 頁碼 | 講義頁碼與 PDF 頁碼一致（「第 N 頁，共 21 頁」） |

(來源: [[ch04-single-layer-regression]], p.1)

> [!note] 關於講義內引用章號
> 講義內文提及的第三章單變量高斯（p.6）、第五章分類（p.2, p.12）、第六章深度神經網路與高斯混合（p.2, p.3, p.12）、第七章隨機梯度下降（p.7）、第九章正則化（p.8）等，除本講義 Ch5、Ch6、Ch7 之外，其餘若為原書章節皆標註「（講義引用原書章節，未必對應本講義章號）」。(來源: [[ch04-single-layer-regression]], p.2)

## 學習目標（📌講義）

1. 把第一章的多項式擬合推廣為線性基底函數模型，理解它就是「只有單層可學習參數」的神經網路，並認識多項式、高斯、S 形與 Fourier／小波等基底函數。
2. 從高斯雜訊的概似函數出發，推導最大概似解（正規方程、Moore–Penrose 偽反矩陣）、偏置參數與雜訊變異數的解，並掌握最小平方的幾何投影詮釋。
3. 掌握循序學習（隨機梯度下降、LMS 演算法）、正則化最小平方與多重輸出的處理方式。
4. 區分推論階段與決策階段：以變分法證明平方損失下的最佳預測是條件期望值，並理解 Minkowski 損失與條件中位數／眾數的關係。
5. 推導偏差–變異分解（期望損失 = 偏差$^2$ + 變異 + 雜訊），並理解正則化係數 $\lambda$ 如何在兩者之間取捨。

(來源: [[ch04-single-layer-regression]], p.1)

## 章節大綱與對應 wiki 頁面

| 講義章節 | PDF 頁 | 核心內容 | 對應頁面 |
|---|---|---|---|
| 4.1.1 基底函數 | p.2–4 | 線性基底函數模型、多項式/高斯/S形/小波基底、單層網路結構 | [[linear-basis-function-models]] |
| 4.1.2 概似函數 | p.4–5 | 高斯雜訊假設、對數概似函數、最大概似與平方和誤差等價性 | [[maximum-likelihood-regression]] |
| 4.1.3 最大概似 | p.5–6 | 梯度推導、正規方程、偽反矩陣、偏置參數 $w_0$、雜訊變異數 $\sigma_{\text{ML}}^2$ | [[maximum-likelihood-regression]] |
| 4.1.4 最小平方的幾何 | p.6–7 | $N$ 維空間正交投影、帽子矩陣 $\mathbf{P}$、殘差正交性、退化與 SVD | [[geometry-of-least-squares]] |
| 4.1.5 循序學習 | p.7–8 | 線上學習、隨機梯度下降 (SGD)、LMS 演算法、與 RLS/Kalman 的關係 | [[sequential-learning-lms]] |
| 4.1.6 正則化最小平方 | p.8 | 參數收縮、閉式解 $(\lambda \mathbf{I} + \boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{t}$、數值/幾何/貝氏三種詮釋 | [[regularized-least-squares-ch04]] |
| 4.1.7 多重輸出 | p.8–9 | 矩陣參數 $\mathbf{W}$、等向高斯雜訊、多目標解耦性質 | [[multiple-output-regression]] |
| 4.2 決策論 | p.9–12 | 推論階段 vs. 決策階段、期望平方損失、變分法推導條件均值、Minkowski 損失族 | [[decision-theory-regression]] |
| 4.3 偏差–變異權衡 | p.12–16 | 頻率學派資料集系集、偏差$^2$–變異–雜訊分解、正則化 $\lambda$ 調控 | [[bias-variance-tradeoff]] |

## 產出／更新的頁面

- Concepts / Algorithms（8 頁）：
  - [[linear-basis-function-models]]
  - [[maximum-likelihood-regression]]
  - [[geometry-of-least-squares]]
  - [[sequential-learning-lms]]
  - [[regularized-least-squares-ch04]]
  - [[multiple-output-regression]]
  - [[decision-theory-regression]]
  - [[bias-variance-tradeoff]]

## 相關頁面

[[index]] · [[overview]] · [[log]] · [[ch01-deep-learning-revolution]]
