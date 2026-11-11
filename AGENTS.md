# AGENTS.md — 機器學習 LLM Wiki 運作規範

本專案依 Karpathy「LLM Wiki」模式建立：agent 負責撰寫並維護 `wiki/`，使用者負責選材、提問與判斷。
主題：機器學習／深度學習（來源為 5 份中文講義）。最終成果需發布到 **GitHub Pages**（教學型 HTML 簡報）。

## 1. 三層架構

| 層 | 路徑 | 規則 |
|---|---|---|
| Raw sources | `raw/` | 不可變。只讀，絕不修改、刪除。 |
| Wiki | `wiki/` | agent 全權維護的 Markdown。使用者只讀。 |
| Schema | `AGENTS.md` | 本檔。與使用者共同演進，修改前需告知使用者。 |
| 圖資 | `assets/` | 圖表、SVG、HTML 簡報素材。 |

```
llm-wiki-project/
├─ AGENTS.md
├─ raw/                      # 原始講義 PDF
├─ assets/                   # 圖表 (PNG 擷取 / SVG / HTML)，依章節分子資料夾
├─ wiki/
│  ├─ index.md               # 內容目錄
│  ├─ log.md                 # 時間序紀錄 (append-only)
│  ├─ overview.md            # 整體綜述與演進中的論點
│  ├─ sources/               # 每份講義一頁摘要
│  ├─ concepts/              # 概念頁
│  ├─ algorithms/            # 演算法頁
│  ├─ comparisons/           # 比較頁
│  └─ queries/               # 回存的問答／分析
```

## 2. 初始來源（raw/）

原檔名已統一重新命名，章號沿用講義原章號（習題以章號稱呼，如「第一章習題 11」）。

| 檔案 | 章 | 主題 | source 頁 |
|---|---|---|---|
| `ch01-deep-learning-revolution.pdf` | 1 | 深度學習革命 (Deep Learning Revolution) | `sources/ch01-deep-learning-revolution.md` |
| `ch04-single-layer-regression.pdf` | 4 | 單層網路：迴歸 (Single-Layer Regression) | `sources/ch04-single-layer-regression.md` |
| `ch05-single-layer-classification.pdf` | 5 | 單層網路：分類 (Single-Layer Classification) | `sources/ch05-single-layer-classification.md` |
| `ch06-deep-neural-networks.pdf` | 6 | 深度神經網路 (Deep Neural Networks) | `sources/ch06-deep-neural-networks.md` |
| `ch07-gradient-descent.pdf` | 7 | 梯度下降 (Gradient Descent) | `sources/ch07-gradient-descent.md` |

注意：講義章號不連續（缺 2、3），不要自行假設或補造缺漏章節；若內容引用到未提供的章節，標註「（講義引用，未提供）」。

## 3. 語言與命名

- 內文：**繁體中文與英文並列**，首次出現寫成「梯度下降 (Gradient Descent)」，之後可單用其一，但同頁保持一致。
- 頁面標題（H1 與 frontmatter `title`）：中英並列，如 `# 梯度下降 (Gradient Descent)`。
- 檔名：英文 kebab-case，如 `gradient-descent.md`。
- 連結：Obsidian `[[wikilink]]`，使用檔名（不含副檔名），必要時 `[[gradient-descent|梯度下降]]`。

## 4. 頁面格式

每頁必須有 YAML frontmatter：

```yaml
---
title: 梯度下降 (Gradient Descent)
type: concept | algorithm | comparison | source | query | overview
tags: [optimization, training]
sources: [ch07-gradient-descent]     # 用到的 source 頁檔名
created: 2026-11-11
updated: 2026-11-11
---
```

內文慣例：
- 公式一律用 LaTeX：行內 `$...$`，區塊 `$$...$$`。
- **標記來源性質**（每段／每條要點都要能分辨）：
  - `📌講義`：直接來自講義內容。
  - `💡補充`：agent 補充（外部知識、推導、網路資料、額外程式碼），須明確可辨識。
- **引用格式（必填）**：`📌講義` 內容結尾標註 `(來源: [[ch07-gradient-descent]], p.15)`，頁碼為 PDF 頁碼；若講義頁碼與 PDF 頁碼不同，兩者並註。無法確認頁碼時寫 `p.?` 並回報使用者，不得捏造。
- **矛盾處理**：講義之間或與標準教材不一致時，在該頁加 `> [!warning] 差異` 區塊，並列各說法與出處，**不自行裁決**，並記入 log。
- **Python 範例**：可使用講義內容，也可補充常見 NumPy／scikit-learn 範例（補充者標 `💡補充`）。講義習題的 NumPy 實作要求（如第一章習題 11、第六章習題 15）在對應概念頁提供範例，註明對應習題。程式碼須可執行、附簡短說明。
- **每個 concept／algorithm 頁尾**必須有 `## 自我測驗 Q&A`，3–5 題，答案用 `> [!answer]-` 折疊區塊。
- 頁尾 `## 相關頁面` 列出交叉連結。

### 各頁類型重點
- **source**：講義摘要，含章節大綱、關鍵要點、產出／更新了哪些頁面、習題清單與對應頁面。
- **concept**：定義、直覺、公式、例子、與其他概念的關係、程式碼（如適用）、Q&A。
- **algorithm**：問題設定、步驟（偽碼）、公式推導、複雜度／優缺點、NumPy 實作、Q&A。
- **comparison**：表格比較（如迴歸 vs 分類、各種 loss、各種激活函數）。
- **query**：使用者提問的答案存檔，註明提問與日期。

## 5. 圖表與 assets

- 重要圖表為日後 HTML 簡報準備：優先以 **SVG／HTML 重繪**；無法重繪時才從 PDF 擷取 PNG。
- 路徑：`assets/<chapter>/<descriptive-name>.svg|png`，如 `assets/ch07/gradient-descent-contour.svg`。
- Wiki 頁中以相對路徑引用並附圖說與來源頁碼（擷取或重繪皆標註原講義位置）。

## 6. 工作流程

### Ingest（一次一份講義，互動進行）
1. 檢視 PDF：文字型或掃描、公式多寡、圖表數量；回報並決定讀取方式（必要時 OCR／逐頁看圖）。
2. 讀取全文，**先與使用者討論重點**並確認強調方向，再寫入。
3. 建立 `sources/` 摘要頁。
4. 建立／更新相關 concept、algorithm、comparison 頁（單份來源預計動 10–15 頁），維護交叉連結。
5. 製作所需圖表放入 `assets/`。
6. 檢查矛盾，標出差異。
7. 更新 `wiki/index.md`、`wiki/overview.md`，append 到 `wiki/log.md`。
8. **Git commit**（見第 8 節）。

### Query
1. 先讀 `index.md`，再讀相關頁面，回答並附連結／引用。
2. 有價值的答案詢問使用者是否存成 `queries/` 頁（或比較頁）；存檔後更新 index、log，並 commit。

### Lint（使用者要求時執行）
檢查：頁面矛盾、被取代的過時說法、孤兒頁、缺少獨立頁面的重要概念、缺少交叉連結、缺少 frontmatter／Q&A／引用／頁碼、斷鏈、資料缺口。輸出報告、修正，並建議新問題與新資料。記入 log 並 commit。

## 7. index.md 與 log.md

- `index.md`：依類別（Sources／Concepts／Algorithms／Comparisons／Queries）列出每頁 `[[連結]]` + 一行摘要 + 來源章。每次 ingest／query 存檔都要更新。
- `log.md`：只能附加。格式固定：`## [YYYY-MM-DD] ingest | 標題`（類型：`ingest`、`query`、`lint`、`schema`），下方簡述異動頁面與發現。

## 8. Git 與發布

- 專案已 `git init`。每次 ingest、query 存檔、lint、schema 變更後**自動 commit**，不需再詢問。
- Commit 訊息：`ingest: ch07 梯度下降`、`query: ...`、`lint: ...`、`schema: ...`。
- 不 push、不改遠端，除非使用者明確要求。
- 後續目標：將 wiki 內容與 `assets/` 轉為教學型 HTML 簡報，發布至 GitHub Pages（發布設定屆時再與使用者討論）。

## 9. 搜尋

目前僅用 `index.md` 導覽。頁數超過約百頁、index 不敷使用時，再與使用者討論導入 qmd 等搜尋工具。

## 10. 一般守則

- 不捏造講義內容、頁碼或習題；不確定就問。
- 不修改 `raw/`。
- 使用者提出的修正或偏好，必要時更新本檔（並記入 log）。
