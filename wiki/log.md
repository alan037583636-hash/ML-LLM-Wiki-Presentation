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
