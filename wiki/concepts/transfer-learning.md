---
title: 遷移學習與微調 (Transfer Learning and Fine-tuning)
type: concept
tags: [transfer-learning, pre-training, fine-tuning, foundation-model, numpy]
sources: [ch01-deep-learning-revolution]
created: 2026-11-11
updated: 2026-11-11
---

# 遷移學習與微調 (Transfer Learning and Fine-tuning)

## 1. 定義

📌講義：**遷移學習 (transfer learning)**：網路先從大量資料學到一般性質，再專門化到特定問題。**微調 (fine-tuning)**：以目標任務資料進一步調整預訓練模型。它是跨越學習典範的訓練策略：先在大資料上**預訓練 (pre-train)**，再於目標任務上微調。(來源: [[ch01-deep-learning-revolution]], p.3, p.4, p.19)

## 2. 皮膚癌案例

📌講義：可用的標註皮膚病灶影像約 129,000 張，在深度學習尺度下算相對少量。因此網路先在一個更大的資料集——128 萬張日常物件（狗、建築、蘑菇等）影像——上訓練，再於病灶影像資料集上微調。網路先從大量日常影像學到自然影像的一般性質，再專門化到病灶分類。透過深度學習，皮膚病灶影像分類的準確率已超越專業皮膚科醫師（Brinker et al., 2019）。(來源: [[ch01-deep-learning-revolution]], p.3)

## 3. 為什麼可行：表徵可重用

📌講義：深度網路的隱藏層學到語意上有意義的內部**表徵**，這些內部表徵可重新利用於相關問題，即遷移學習；可針對一系列下游任務調適或微調的大型網路稱**基礎模型**。(來源: [[ch01-deep-learning-revolution]], p.15)　見 [[scaling-and-representation-learning]]。

## 4. NumPy 玩具範例（💡補充）

💡補充：資料很少（8 筆）而特徵很多（20 維）時，從零訓練的線性模型是欠定的；若有來自「相近任務」的大量資料預訓練，則可把預訓練權重當成起點，在小資料上只做小幅調整。這裡用「朝預訓練權重收縮」的閉式解 $\mathbf w=\mathbf w_s+(\Phi^T\Phi+\lambda I)^{-1}\Phi^T(\mathbf t-\Phi\mathbf w_s)$ 模擬微調（對應 [[regularization-weight-decay]] 中以先驗為中心的正則化）。

```python
import numpy as np

rng = np.random.default_rng(0)
d = 20
w_source = rng.normal(0, 1, d)                        # 來源任務（大資料）的真實權重
w_target = w_source + rng.normal(0, 0.2, d)           # 目標任務：相近但不同


def make(w, n, noise=0.1):
    X = rng.normal(0, 1, (n, d))
    return X, X @ w + rng.normal(0, noise, n)


# 預訓練：來源任務有 5000 筆資料
Xs, ts = make(w_source, 5000)
w_pre = np.linalg.lstsq(Xs, ts, rcond=None)[0]

# 目標任務：只有 8 筆資料（小於特徵數 20）；另有 1000 筆測試資料
Xt, tt = make(w_target, 8)
Xtest, ttest = make(w_target, 1000)


def rmse(w):
    return float(np.sqrt(np.mean((Xtest @ w - ttest) ** 2)))


w_scratch = np.linalg.lstsq(Xt, tt, rcond=None)[0]    # 從零訓練（最小範數解）
lam = 1.0
w_ft = w_pre + np.linalg.solve(Xt.T @ Xt + lam * np.eye(d), Xt.T @ (tt - Xt @ w_pre))  # 微調

print(f"從零訓練（8 筆資料）  測試 RMSE = {rmse(w_scratch):.3f}")
print(f"只用預訓練、不微調     測試 RMSE = {rmse(w_pre):.3f}")
print(f"預訓練 + 微調          測試 RMSE = {rmse(w_ft):.3f}")
```

## 自我測驗 Q&A

> [!question] Q1. 皮膚癌案例為什麼要先在 128 萬張日常物件影像上預訓練？
> [!answer]- 答案
> 標註病灶影像只有約 129,000 張，相對較少；先用更大的日常影像資料學自然影像的一般性質，再微調到病灶分類。(來源: [[ch01-deep-learning-revolution]], p.3)

> [!question] Q2. 預訓練與微調各自做什麼？
> [!answer]- 答案
> 預訓練：在大資料上學通用表徵；微調：用目標任務資料進一步調整預訓練模型。(來源: [[ch01-deep-learning-revolution]], p.4, p.19)

> [!question] Q3. 遷移學習為什麼和「表徵學習」有關？
> [!answer]- 答案
> 隱藏層學到的語意表徵可以重用於相關問題，這就是遷移學習的基礎。(來源: [[ch01-deep-learning-revolution]], p.15)

> [!question] Q4. 上方玩具範例中，為什麼預訓練＋微調比從零訓練好？
> [!answer]- 答案
> 💡補充：8 筆資料不足以決定 20 維權重，從零訓練欠定；預訓練權重提供了接近目標的起點（先驗），微調只需用少量資料修正差異。

## 相關頁面

[[learning-paradigms]] · [[scaling-and-representation-learning]] · [[regularization-weight-decay]] · [[supervised-vs-unsupervised-vs-self-supervised]]
