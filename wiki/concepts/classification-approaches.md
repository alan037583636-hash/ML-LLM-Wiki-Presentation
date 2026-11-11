---
title: 分類問題的三種取徑 (Three Approaches to Classification)
type: concept
tags: [classification, generative-models, discriminative-models, discriminant-functions, novelty-detection]
sources: [ch05-single-layer-classification]
created: 2026-11-11
updated: 2026-11-11
---

# 分類問題的三種取徑 (Three Approaches to Classification)

## 1. 問題定義

📌講義：分類的目標是將輸入向量 $\mathbf{x} \in \mathbb{R}^D$ 指派到 $K$ 個離散類別 $\mathcal{C}_k$（$k = 1, \dots, K$）之一。最常見的情境中，各類別互斥，每個輸入恰好被指派到一個類別；輸入空間因此被劃分為**決策區域 (decision regions)**，其邊界稱為**決策邊界／決策面 (decision boundaries / decision surfaces)**。
線性分類模型中，決策面是輸入向量 $\mathbf{x}$ 的線性函數（在 $D$ 維空間中為 $(D-1)$ 維超平面）。若資料集能被線性決策面完美分開，稱為**線性可分 (linearly separable)**。(來源: [[ch05-single-layer-classification]], p.2)

## 2. 三種取徑的核心對照

📌講義總結了解決分類問題的三種取徑（依建模複雜度遞減）：

```
[生成式機率模型]           [判別式機率模型]           [判別函數]
p(x | C_k) 與 p(C_k)         p(C_k | x)              直接 f(x)
       ↓ (貝氏定理)                ↓                      ↓
   p(C_k | x)                p(C_k | x)              類別標籤
       ↓ (決策論)                  ↓ (決策論)               
    類別決策                   類別決策
```

| 面向 | (a) 判別函數 (Discriminant Function) | (b) 生成式機率模型 (Generative Model) | (c) 判別式機率模型 (Discriminative Model) |
|---|---|---|---|
| **建模對象** | 直接建構映射 $f(\mathbf{x}) \to \mathcal{C}_k$ | 類別條件密度 $p(\mathbf{x} \mid \mathcal{C}_k)$ 與先驗 $p(\mathcal{C}_k)$ | 條件後驗機率 $p(\mathcal{C}_k \mid \mathbf{x})$ |
| **機率角色** | 機率完全不出現 | 透過貝氏定理計算後驗機率 | 直接以參數化模型表示後驗機率 |
| **代表模型** | 最小平方分類、感知器、SVM | 高斯判別分析 (LDA/QDA)、樸素貝氏 | 邏輯迴歸 (Logistic)、Softmax 迴歸 |
| **參數量級** | 較少，隨維度線性增長 | 較大，估計完整密度協方差為 $O(D^2)$ | 較少，隨特徵維度線性增長 $O(KD)$ |
| **附帶收穫** | 無 | 獲得邊際分布 $p(\mathbf{x})$，可用於離群／新穎性偵測 | 專注於分類邊界，參數效率高 |
| **主要弱點** | 缺乏機率詮釋、不穩健 | 密度假設錯誤時表現劣化 | 線性可分資料易過度擬合 |

(來源: [[ch05-single-layer-classification]], p.2, p.8–9, p.22)

## 3. 為什麼依然需要後驗機率？（📌講義）

📌講義指出，即使最終目標只是做出類別指派，保留機率推論（相較於純判別函數）具有不可替代的四大優勢：

1. **風險最小化 (Minimizing Risk)**：
   若決策損失矩陣（成本）隨時間動態改變（如金融交易或醫療政策），有了後驗機率 $p(\mathcal{C}_k \mid \mathbf{x})$，只需重新計算期望損失矩陣的加權即可瞬間調整決策邊界；純判別函數則必須回到原始資料重新訓練。
2. **拒絕選項 (Reject Option)**：
   在後驗機率相對平坦、高度不確定的模糊區域（例如 $\max_k p(\mathcal{C}_k \mid \mathbf{x}) \le \theta$），模型可主動放棄自動決策，轉交人類專家進行病理切片或二次複驗。
3. **補償類別先驗 (Compensating for Class Priors)**：
   在極度不平衡的資料中（如癌症篩檢 1000 人中僅 1 人確診），若直接訓練模型會陷入全部分為正常類的平凡解。正確做法是以各類樣本數平衡的資料進行抽樣訓練，之後再依貝氏定理乘上真實母體的先驗比例並重新正規化。
4. **模組化模型結合 (Combining Models)**：
   若對同一對象具備不同模態的異質輸入（例如皮膚影像 $\mathbf{x}_I$ 與血液生化數據 $\mathbf{x}_B$），各自建立模型輸出後驗機率後，即可透過機率規則（如條件獨立下的樸素貝氏）直接融合為 $p(\mathcal{C}_k \mid \mathbf{x}_I, \mathbf{x}_B) \propto \frac{p(\mathcal{C}_k \mid \mathbf{x}_I)p(\mathcal{C}_k \mid \mathbf{x}_B)}{p(\mathcal{C}_k)}$。(來源: [[ch05-single-layer-classification]], p.9–10)

## 自我測驗 Q&A

> [!question] Q1. 生成式分類器與判別式分類器在建模對象上有何本質差別？
> [!answer]- 答案
> 生成式模型對類別條件機率 $p(\mathbf{x}\mid\mathcal{C}_k)$ 與先驗 $p(\mathcal{C}_k)$ 建模，再利用貝氏定理間接求解後驗；判別式模型則跳過輸入分布的估計，直接對後驗機率 $p(\mathcal{C}_k\mid\mathbf{x})$ 進行參數化建模。(來源: [[ch05-single-layer-classification]], p.2, p.8–9)

> [!question] Q2. 生成式模型估計出 $p(\mathbf{x})$ 有何額外實用價值？
> [!answer]- 答案
> 可用於離群偵測／新穎性偵測 (novelty detection)：若新輸入點在模型估計的 $p(\mathbf{x})$ 下機率極低，代表其落於訓練資料稀疏區，模型可發出警訊提示該預測高度不可靠。(來源: [[ch05-single-layer-classification]], p.9)

> [!question] Q3. 為什麼純判別函數在損失矩陣改變時必須重新訓練，而機率式模型不需要？
> [!answer]- 答案
> 純判別函數將推論與決策混在一起；而機率式模型先完整求出後驗機率分布，決策只是在已知分布下計算最小期望損失，損失矩陣變動時僅需變更代價係數加權即可，無須更動模型參數。(來源: [[ch05-single-layer-classification]], p.9)

## 相關頁面

[[linear-discriminant-functions]] · [[decision-theory-classification]] · [[generative-classifiers]] · [[discriminative-classifiers-logistic-regression]] · [[ch05-single-layer-classification]]
