---
title: 線性模型與基底函數 (Linear Models and Basis Functions)
type: concept
tags: [linear-model, basis-function, design-matrix, inductive-bias, numpy]
sources: [ch01-deep-learning-revolution]
created: 2026-11-11
updated: 2026-11-11
---

# 線性模型與基底函數 (Linear Models and Basis Functions)

## 1. 定義

📌講義：**線性模型 (linear models)** 指對**未知參數為線性**的函數。多項式 $y(x,\mathbf w)=\sum_{j=0}^Mw_jx^j$ 雖是 $x$ 的非線性函數，卻是係數 $\mathbf w$ 的線性函數，具有重要性質，也有顯著限制。(來源: [[ch01-deep-learning-revolution]], p.5)

📌講義：以矩陣語言，令 $\boldsymbol\phi(x)=(1,x,x^2,\dots,x^M)^T$，則 $y(x,\mathbf w)=\mathbf w^T\boldsymbol\phi(x)$：模型是「**固定基底函數 (basis function) 的線性組合**」，可學習的只有組合係數。各列為各樣本之基底函數值的矩陣稱為設計矩陣 $\Phi$。(來源: [[ch01-deep-learning-revolution]], p.5, p.19)

> [!note] 講義引用
> 講義說線性模型將在「Ch. 4」詳述（講義引用原書 Ch. 4，未必對應本講義 Ch4；本講義 Ch4 為單層網路迴歸，內容見後續 ingest）。

## 2. 直覺：改變基底，不改變演算法

💡補充：無論基底是多項式 $x^j$、Fourier $\{1,\sin2\pi kx,\cos2\pi kx\}$ 或其他函數，只要固定基底，學習問題都是同一個最小平方問題 $\mathbf w^\star=(\Phi^T\Phi)^{-1}\Phi^T\mathbf t$（見 [[least-squares-normal-equations]]），差別只在設計矩陣 $\Phi$ 的欄。(來源: [[ch01-deep-learning-revolution]], p.6，講義教學補充：差別只在基底函數的選取)

📌講義：「線性模型」與「神經網路」的分野，僅在於基底（特徵）是**固定**還是**可學習**，以及 $f$ 是否非線性。深度學習的關鍵一步，是讓基底函數本身也從資料中學得（表徵學習）。(來源: [[ch01-deep-learning-revolution]], p.6, p.12)

| 比較 | 線性模型 | 神經網路 |
|---|---|---|
| 基底函數 | 固定 $\phi_j(x)$（手工設計） | 隱藏層學得 |
| 對參數 | 線性 | 非線性 |
| 平方和誤差 | 二次函數，有閉式解 | 高度非線性，需迭代（SGD） |
| 歸納偏差來源 | 基底的選擇 | 架構與訓練方式 |

（整理自 p.5–6, p.11, p.14 之內容，表格組織為 💡補充）

## 3. NumPy 範例：同一套求解器，不同基底

💡補充：

```python
import numpy as np

rng = np.random.default_rng(0)
N = 10
x = np.linspace(0, 1, N)
t = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, N)
xt = np.linspace(0, 1, 100)
tt = np.sin(2 * np.pi * xt) + rng.normal(0, 0.3, 100)


def polynomial_basis(x, M):
    return np.vander(x, M + 1, increasing=True)


def fourier_basis(x, K):
    cols = [np.ones_like(x)]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * x), np.cos(2 * np.pi * k * x)]
    return np.stack(cols, axis=1)


def fit_and_score(basis, order):
    Phi, Phit = basis(x, order), basis(xt, order)
    w = np.linalg.lstsq(Phi, t, rcond=None)[0]       # 同一個最小平方求解器
    rms = lambda P, y: np.sqrt(np.mean((P @ w - y) ** 2))
    return Phi.shape[1], rms(Phi, t), rms(Phit, tt)


for name, basis, order in [("多項式 M=3", polynomial_basis, 3),
                           ("多項式 M=9", polynomial_basis, 9),
                           ("Fourier K=1", fourier_basis, 1),
                           ("Fourier K=2", fourier_basis, 2)]:
    p, tr, te = fit_and_score(basis, order)
    print(f"{name:<12} 參數數={p:<2}  訓練 E_RMS={tr:.3f}  測試 E_RMS={te:.3f}")
```

> [!tip] 觀察
> Fourier $K=1$（3 個參數）就貼近雜訊底線 0.3，因為 $\sin2\pi x$ 恰在基底張成的空間內——這是「基底選擇＝歸納偏差」的最直接例子（習題 11(e)，詳見 [[polynomial-curve-fitting]]）。

## 自我測驗 Q&A

> [!question] Q1. 「線性模型」的「線性」是對什麼而言？
> [!answer]- 答案
> 對未知參數 $\mathbf w$ 線性，而非對輸入 $x$。多項式對 $x$ 非線性，對 $\mathbf w$ 線性，所以仍是線性模型。(來源: [[ch01-deep-learning-revolution]], p.5)

> [!question] Q2. 寫出多項式的基底函數向量與設計矩陣的元素。
> [!answer]- 答案
> $\boldsymbol\phi(x)=(1,x,\dots,x^M)^T$；$\Phi_{nj}=x_n^j$。(來源: [[ch01-deep-learning-revolution]], p.5–6)

> [!question] Q3. 線性模型與神經網路的分野是什麼？
> [!answer]- 答案
> 基底是固定還是可從資料學得，以及激活函數 $f$ 是否非線性。(來源: [[ch01-deep-learning-revolution]], p.12)

> [!question] Q4. 為什麼換成 Fourier 基底不需要換求解器？
> [!answer]- 答案
> 💡補充：只要基底固定，模型對 $\mathbf w$ 仍是線性、平方和誤差仍是二次函數，只需改設計矩陣 $\Phi$ 的欄，正規方程不變。

## 相關頁面

[[polynomial-curve-fitting]] · [[least-squares-normal-equations]] · [[artificial-neuron-perceptron]] · [[scaling-and-representation-learning]]
