---
title: 超參數與模型選擇 (Hyperparameters and Model Selection)
type: concept
tags: [hyperparameter, model-selection, validation, test-set, numpy]
sources: [ch01-deep-learning-revolution]
created: 2026-11-11
updated: 2026-11-11
---

# 超參數與模型選擇 (Hyperparameters and Model Selection)

## 1. 定義

📌講義：**超參數 (hyperparameter)** 是在最小化誤差函數以決定模型參數 $\mathbf w$ 的過程中**固定不變**的量，例如正則化係數 $\lambda$ 與多項式階數 $M$。**模型選擇／比較 (model selection / comparison)** 是在不同複雜度的模型間做選擇，決定 $M$ 就是其一例。(來源: [[ch01-deep-learning-revolution]], p.7, p.10–11, p.20)

📌講義：不能把 $\lambda$ 與 $\mathbf w$ 一起對誤差函數做聯合最小化——那會導致 $\lambda\to0$，得到訓練誤差極小甚至為零的過度擬合模型；同理，直接對 $M$ 最小化訓練誤差會導致 $M$ 過大而過度擬合。因此需要另闢蹊徑決定超參數。(來源: [[ch01-deep-learning-revolution]], p.10–11)

## 2. 資料分割：訓練／驗證／測試

📌講義：簡單方法是把可用資料分成**訓練集**（用來決定 $\mathbf w$）與獨立的**驗證集** (validation set，又稱 hold-out set 或 development set)，選擇在驗證集上誤差最低的模型。若以有限資料反覆迭代模型設計多次，可能對驗證資料也產生某種程度的過度擬合，此時需再保留第三組**測試集**，對最終選定的模型做最後的效能評估。(來源: [[ch01-deep-learning-revolution]], p.11)

| 資料集 | 用途 | 決定的對象 |
|---|---|---|
| 訓練集 (training) | 最小化誤差函數 | 參數 $\mathbf w$ |
| 驗證集 (validation) | 比較不同超參數設定 | 超參數 $M$、$\lambda$、架構等 |
| 測試集 (test) | **只用一次**，報告最終效能（不用來做任何選擇） | — |

(來源: [[ch01-deep-learning-revolution]], p.11)

📌講義：當訓練與測試資料供應有限時，驗證集若太小，對預測效能的估計會相當雜亂；解決方法見 [[cross-validation]]。現代最先進模型極其龐大、訓練資料也極其龐大，探索超參數設定的空間有限，多半仰賴小模型上累積的經驗與各種啟發式法則。(來源: [[ch01-deep-learning-revolution]], p.11)

## 3. 習題 6 解答（對應：第一章習題 6）

📌講義：習題 6（◦超參數不能用訓練誤差決定）：說明為何「對 $\mathbf w$ 與 $\lambda$ 聯合最小化 $\widetilde E$」必然得到 $\lambda=0$，以及「對 $M$ 最小化訓練誤差」必然選到最大的 $M$；接著說明驗證集與測試集的角色為何不能合併。(來源: [[ch01-deep-learning-revolution]], p.17)

**以下解答皆為 💡補充：**

1. **聯合最小化得 $\lambda=0$**：$\widetilde E(\mathbf w,\lambda)=E(\mathbf w)+\frac\lambda2\|\mathbf w\|^2$，對任意固定 $\mathbf w$，由於 $\|\mathbf w\|^2\ge0$，$\widetilde E$ 對 $\lambda\ge0$ 為非減函數，故最小值出現在 $\lambda=0$（偏導 $\partial\widetilde E/\partial\lambda=\frac12\|\mathbf w\|^2\ge0$）。此時 $\mathbf w$ 回到無正則化的最小平方解，即過度擬合。
2. **對 $M$ 最小化訓練誤差選到最大 $M$**：階數 $M$ 的模型族包含於 $M+1$ 的模型族（令 $w_{M+1}=0$），所以 $\min_{\mathbf w}E_{M+1}\le\min_{\mathbf w}E_M$：訓練誤差隨 $M$ 單調不增，必選到允許的最大 $M$；在 $N$ 個相異點上，$M=N-1$ 時訓練誤差為 0。
3. **驗證集與測試集不能合併**：驗證集被用來做選擇（選 $M$、$\lambda$、架構）；選擇次數越多，選出的模型越可能「恰好」擬合到這份驗證資料的雜訊，所以驗證誤差是**偏樂觀**的估計。測試集必須完全不參與任何選擇，才能對最終模型的泛化表現做**不偏**的評估。若合併，就沒有任何資料能誠實回答「這個模型對新資料表現如何」。

NumPy 驗證（已實際執行）：

```python
import numpy as np


def make_data(n, rng, sigma=0.3):
    x = rng.uniform(0.0, 1.0, n)
    return x, np.sin(2 * np.pi * x) + rng.normal(0.0, sigma, n)


def poly(x, M):
    return np.vander(x, M + 1, increasing=True)


def ridge(Phi, t, lam):
    return np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ t)


def erms(Phi, t, w):
    return float(np.sqrt(np.mean((Phi @ w - t) ** 2)))


# 習題 6(1)：訓練誤差對 M 單調不增（巢狀模型）→ 以訓練誤差選 M 必選最大者
rng = np.random.default_rng(0)
x, t = make_data(10, rng)
train = [erms(poly(x, M), t, np.linalg.lstsq(poly(x, M), t, rcond=None)[0]) for M in range(10)]
print("訓練 E_RMS 隨 M:", np.round(train, 3))
print("單調不增:", all(a >= b - 1e-9 for a, b in zip(train, train[1:])))

# 習題 6(2)：對 lambda 最小化訓練目標 -> lambda = 0
P = poly(x, 9)
for ll in (-30, -15, -5, 0):
    w = ridge(P, t, np.exp(ll))
    obj = 0.5 * np.sum((P @ w - t) ** 2) + 0.5 * np.exp(ll) * (w @ w)
    print(f"ln lambda={ll:>4}: 訓練目標 = {obj:.4e}")

# 習題 6(3)：驗證集被重複用來挑選後會「過於樂觀」，所以需要獨立測試集
gaps = []
for s in range(200):
    r = np.random.default_rng(1000 + s)
    xa, ta = make_data(15, r)                         # 訓練
    xv, tv = make_data(15, r)                         # 驗證
    xs, ts = make_data(2000, r)                       # 測試（大，近似真實泛化）
    best_val, best_test = np.inf, None
    for M in range(0, 10):
        for ll in range(-30, 1, 2):                   # 10 x 16 = 160 組超參數
            w = ridge(poly(xa, M), ta, np.exp(ll))
            e_val = erms(poly(xv, M), tv, w)
            if e_val < best_val:
                best_val, best_test = e_val, erms(poly(xs, M), ts, w)
    gaps.append((best_val, best_test))
gaps = np.array(gaps)
print(f"\n挑出的最佳模型：驗證 E_RMS 平均 = {gaps[:, 0].mean():.3f}，獨立測試 E_RMS 平均 = {gaps[:, 1].mean():.3f}")
```

```text
訓練 E_RMS 隨 M: [0.644 0.555 0.555 0.226 0.191 0.154 0.084 0.023 0.002 0.   ]
單調不增: True
ln lambda= -30: 訓練目標 = 6.9097e-05
ln lambda= -15: 訓練目標 = 1.0105e-01
ln lambda=  -5: 訓練目標 = 5.9448e-01
ln lambda=   0: 訓練目標 = 1.6725e+00

挑出的最佳模型：驗證 E_RMS 平均 = 0.324，獨立測試 E_RMS 平均 = 0.405
```

最後一行顯示：從 160 組超參數中挑出驗證誤差最低者，其驗證誤差（0.324）明顯低於它在獨立測試集上的真實表現（0.405）——驗證誤差過於樂觀。

## 與其他概念的關係

- 超參數 $\lambda$ 來自 [[regularization-weight-decay]]；$M$ 來自 [[polynomial-curve-fitting]]；後續章節還有學習率 $\eta$（[[history-of-neural-networks]] 的訓練迴圈）。
- 資料不足時的做法：[[cross-validation]]。

## 自我測驗 Q&A

> [!question] Q1. 什麼是超參數？舉出本章的兩個例子。
> [!answer]- 答案
> 在最小化誤差函數決定 $\mathbf w$ 時固定不變、需另行決定的量：多項式階數 $M$、正則化係數 $\lambda$（另有學習率 $\eta$）。(來源: [[ch01-deep-learning-revolution]], p.10–11, p.14)

> [!question] Q2. 訓練集、驗證集、測試集各自用來決定什麼？
> [!answer]- 答案
> 訓練集決定參數 $\mathbf w$；驗證集比較/決定超參數與架構；測試集只用一次，報告最終效能，不參與任何選擇。(來源: [[ch01-deep-learning-revolution]], p.11)

> [!question] Q3. 為什麼不能用訓練誤差來選 $M$ 或 $\lambda$？
> [!answer]- 答案
> 訓練誤差對 $M$ 單調不增、對 $\lambda$ 在 $\lambda=0$ 最小，會一律選到最複雜、最容易過度擬合的設定。(來源: [[ch01-deep-learning-revolution]], p.10–11)

> [!question] Q4. 為什麼驗證集反覆使用後還需要第三份測試集？
> [!answer]- 答案
> 反覆用驗證集做選擇會使模型對驗證資料也產生某種程度的過度擬合，驗證誤差變得樂觀；需要從未參與選擇的測試集才能得到不偏的最終評估。(來源: [[ch01-deep-learning-revolution]], p.11)

## 相關頁面

[[cross-validation]] · [[regularization-weight-decay]] · [[generalization-overfitting]] · [[polynomial-curve-fitting]]
