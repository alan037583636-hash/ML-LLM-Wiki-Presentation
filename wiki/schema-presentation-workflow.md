---
title: 簡報製作工作流程 Graph 規範
type: schema
tags: [workflow, graph, presentation, guidelines]
created: 2026-11-11
updated: 2026-11-11
---

# 簡報製作工作流程 Graph 規範

本文件記錄「機器學習基礎與模型演進」教學型 HTML 簡報的製作工作流程 Graph 與自動檢查標準。

## 1. 核心設計原則
- **主題**：機器學習基礎與模型演進 (Machine Learning Foundations and Model Evolution)
- **視覺風格**：專業科技感（深藍黑底色 `#0b0f19` / `#111827`、高對比白字、電路青藍 `#00f2fe` 與霓虹藍 `#3b82f6` 點綴）。
- **互動架構**：16:9 卡片式幻燈片（Slide-deck），支援鍵盤左右鍵翻頁、全螢幕與進度指示。
- **數學呈現**：分層揭露（主視覺保留核心直覺結論，複雜推導收納於右下角「深入探討 Modal」）。
- **圖表元件**：原生 HTML/SVG 幾何動態示意圖、LaTeX KaTeX 渲染。

## 2. Graph 流程與門禁 (Gates)

```
[Node 1: 檢索 Wiki] ──► [Node 2: 規劃大綱] ──► 【Gate 1: 人機審查】
                                                      │ (退回 Loop 1)
                                                      ▼
[Node 4: 整合 index.html] ◄── 【Gate 2: 防爆版檢查】 ◄── [Node 3: 逐頁草稿]
           │                         │ (退回 Loop 2)
           ▼
【Gate 3: 終端技術驗證】 ──► [發布/完成]
```

- **Gate 1**：大綱節奏是否符合初學者心智模型（痛點 ➔ 直覺 ➔ 解法 ➔ 演進）、頁數（15-20 頁）、每頁單一主題。由人類審查核可後方得推進。
- **Gate 2**：每頁字數（中文 120-180 字）、無未折行超長公式、圖文比強制 4:6 或 5:5、Modal 內容完整。
- **Gate 3**：獨立單檔運行、KaTeX 正常渲染、無水平/垂直捲軸（防爆版）、鍵盤監聽事件正常。
