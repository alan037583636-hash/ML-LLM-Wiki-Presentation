---
title: 最小平方法與正規方程 (Least Squares and Normal Equations)
type: algorithm
tags: [regression, closed-form, linear-algebra, numpy]
sources: [ch01-deep-learning-revolution]
created: 2026-11-11
updated: 2026-11-11
---

# 最小平方法與正規方程 (Least Squares and Normal Equations)

## 1. 問題設定

📌講義：給定訓練資料 $(x_n,t_n)_{n=1}^N$ 與固定基底（如多項式），令設計矩陣 (design matrix) $\Phi\in\mathbb R^{N\times(M+1)}$，$\Phi_{nj}=x_n^j$，目標向量 $\mathbf t=(t_1,\dots,t_N)^T$。要找使平方和誤差

$$E(\mathbf w)=\tfrac12\sum_{n=1}^N\{y(x_n,\mathbf w)-t_n\}^2$$

最小的 $\mathbf w^\star$。$E$ 是 $\mathbf w$ 的二次函數，其導數是 $\mathbf w$ 的線性函數，故最小值唯一且有閉式解 (closed form)。(來源: [[ch01-deep-learning-revolution]], p.6)

## 2. 推導

📌講義：
$$E(\mathbf w)=\tfrac12\|\Phi\mathbf w-\mathbf t\|^2=\tfrac12\left(\mathbf w^T\Phi^T\Phi\mathbf w-2\mathbf w^T\Phi^T\mathbf t+\mathbf t^T\mathbf t\right)$$
$$\nabla_{\mathbf w}E=\Phi^T\Phi\mathbf w-\Phi^T\mathbf t=0\;\Longrightarrow\;\mathbf w^\star=(\Phi^T\Phi)^{-1}\Phi^T\mathbf t$$
此即**正規方程 (normal equations)** $\Phi^T\Phi\mathbf w=\Phi^T\mathbf t$。Hessian 為 $\Phi^T\Phi\succeq0$，當 $\Phi$ 行滿秩（$N\ge M+1$ 且 $x_n$ 互異）時為正定，故 $\mathbf w^\star$ 為唯一全域極小值。$(\Phi^T\Phi)^{-1}\Phi^T$ 稱為 $\Phi$ 的 Moore–Penrose 偽反矩陣。(來源: [[ch01-deep-learning-revolution]], p.6)

📌講義（講義教學補充）：此結構在通訊與導航領域反覆出現：GNSS 定位的最小平方解算（線性化觀測方程 $H\delta\mathbf x=\delta\boldsymbol\rho$）、Wiener 濾波器最佳係數、陣列訊號處理中的線性估計，都是同一個 $\mathbf w^\star=(\Phi^T\Phi)^{-1}\Phi^T\mathbf t$，差別只在基底函數 $\boldsymbol\phi$ 的選取；深度學習的關鍵一步是讓基底函數也由資料學得。(來源: [[ch01-deep-learning-revolution]], p.6)

> [!note] 講義引用
> 講義說明此閉式解可見「習題 4.1」、$E$ 的機率推導見「§2.3.4」，皆屬（講義引用原書 Ch. 4 / Ch. 2，未必對應本講義 Ch4 / Ch2；原書章節未提供）。

## 3. 偽碼

```text
輸入: x[1..N], t[1..N], 階數 M
1. 建立 Phi[n, j] = x[n] ** j            # N x (M+1)
2. 解 (Phi^T Phi) w = Phi^T t            # 或 w = pinv(Phi) @ t
3. 輸出 w
```

## 4. 複雜度與優缺點

💡補充：
- 時間：形成 $\Phi^T\Phi$ 為 $O(N(M+1)^2)$，求解為 $O((M+1)^3)$；沒有迭代、沒有學習率。
- 優點：一步到位、全域最佳、可以證明唯一性。
- 缺點：(1) $\Phi^T\Phi$ 條件數是 $\Phi$ 條件數的平方，$M$ 大時數值不穩（見下方程式）；(2) 只對「參數線性＋平方誤差」成立。講義指出真實的神經網路誤差函數是參數的高度非線性函數，無法閉式最小化，必須迭代式最佳化（見 [[history-of-neural-networks]]）。(來源: [[ch01-deep-learning-revolution]], p.11)

## 5. NumPy 實作

💡補充：以下程式比較三種求解方式並驗證推導的性質（已實際執行）。

```python
import numpy as np

rng = np.random.default_rng(0)
N = 10
x = np.linspace(0, 1, N)
t = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, N)


def design(x, M):
    return np.vander(x, M + 1, increasing=True)      # Phi[n, j] = x_n ** j


M = 3
Phi = design(x, M)
w_normal = np.linalg.solve(Phi.T @ Phi, Phi.T @ t)   # 正規方程
w_lstsq = np.linalg.lstsq(Phi, t, rcond=None)[0]     # 數值較穩定
w_pinv = np.linalg.pinv(Phi) @ t                     # 偽反矩陣
print("M=3 三種解一致:", np.allclose(w_normal, w_lstsq), np.allclose(w_normal, w_pinv))
print("梯度 Phi^T(Phi w - t) 的範數:", np.linalg.norm(Phi.T @ (Phi @ w_normal - t)))

# 習題 2(c)：x 互異且 N >= M+1 時 Phi^T Phi 正定（最小特徵值 > 0）
print("M=3 最小特徵值:", np.linalg.eigvalsh(Phi.T @ Phi).min())

# 習題 2(d)：N = M+1 時 Phi 為方陣（Vandermonde），E(w*) = 0
M = N - 1
Phi = design(x, M)
w = np.linalg.lstsq(Phi, t, rcond=None)[0]
print("M=9, N=10: E(w*) =", 0.5 * np.sum((Phi @ w - t) ** 2))
print("Vandermonde 行列式 =", np.linalg.det(Phi), "；理論值 =", np.prod([x[j] - x[i] for i in range(N) for j in range(i + 1, N)]))
print("cond(Phi) =", f"{np.linalg.cond(Phi):.2e}", "；cond(Phi^T Phi) =", f"{np.linalg.cond(Phi.T @ Phi):.2e}")
```

## 6. 習題 2 解答與推導（對應：第一章習題 2）

📌講義：習題 2（⋆正規方程）：設 $\Phi\in\mathbb R^{N\times(M+1)}$，$\Phi_{nj}=x_n^j$。(a) 證明 $E(\mathbf w)=\frac12\|\Phi\mathbf w-\mathbf t\|^2$；(b) 微分導出 $\Phi^T\Phi\mathbf w^\star=\Phi^T\mathbf t$；(c) 證明 $x_n$ 互異且 $N\ge M+1$ 時 $\Phi^T\Phi$ 為正定，故 $\mathbf w^\star$ 唯一（提示：前 $M+1$ 列構成 Vandermonde 矩陣）；(d) 說明為何 $N=M+1$ 時必有 $E(\mathbf w^\star)=0$，並與圖 3 的 $M=9$ 對照。(來源: [[ch01-deep-learning-revolution]], p.16–17)

**以下解答皆為 💡補充：**

**(a)** $(\Phi\mathbf w)_n=\sum_j\Phi_{nj}w_j=\sum_j w_jx_n^j=y(x_n,\mathbf w)$，所以 $\Phi\mathbf w-\mathbf t$ 的第 $n$ 個分量為 $y(x_n,\mathbf w)-t_n$。而向量範數平方 $\|\mathbf v\|^2=\sum_n v_n^2$，故 $\frac12\|\Phi\mathbf w-\mathbf t\|^2=\frac12\sum_n\{y(x_n,\mathbf w)-t_n\}^2=E(\mathbf w)$。

**(b)** 展開 $E=\frac12(\mathbf w^T\Phi^T\Phi\mathbf w-2\mathbf w^T\Phi^T\mathbf t+\mathbf t^T\mathbf t)$。使用 $\nabla_{\mathbf w}(\mathbf w^TA\mathbf w)=(A+A^T)\mathbf w=2A\mathbf w$（$A=\Phi^T\Phi$ 對稱）與 $\nabla_{\mathbf w}(\mathbf w^T\mathbf b)=\mathbf b$，得
$\nabla E=\Phi^T\Phi\mathbf w-\Phi^T\mathbf t$。令其為零即 $\Phi^T\Phi\mathbf w^\star=\Phi^T\mathbf t$。

**(c)** 任取 $\mathbf v\ne0$，$\mathbf v^T\Phi^T\Phi\mathbf v=\|\Phi\mathbf v\|^2\ge0$，故半正定。若 $\|\Phi\mathbf v\|=0$，則 $\Phi\mathbf v=0$，表示多項式 $p(x)=\sum_jv_jx^j$ 在 $x_1,\dots,x_N$ 全部為零；而 $p$ 是次數 $\le M$、非零的多項式，至多有 $M$ 個相異根；但 $x_n$ 互異且 $N\ge M+1$，有至少 $M+1$ 個根，矛盾。所以 $\mathbf v\ne0\Rightarrow\mathbf v^T\Phi^T\Phi\mathbf v>0$，$\Phi^T\Phi$ 正定。等價地：取任意 $M+1$ 列構成方陣 Vandermonde 矩陣，其行列式 $\prod_{i<j}(x_j-x_i)\neq0$，故 $\Phi$ 行滿秩。正定的 Hessian 表示 $E$ 嚴格凸，臨界點即唯一全域最小值。

**(d)** $N=M+1$ 時 $\Phi$ 是 $(M+1)\times(M+1)$ 的 Vandermonde 方陣且可逆（$x_n$ 互異），所以可取 $\mathbf w=\Phi^{-1}\mathbf t$ 使 $\Phi\mathbf w=\mathbf t$，即 $E=0$，而 $E\ge0$，故最小值為 0。對照圖 3：$M=9,N=10$ 時 9 階多項式恰好內插 10 個點，訓練誤差為零卻嚴重過度擬合。

## 自我測驗 Q&A

> [!question] Q1. 正規方程為什麼保證得到「全域」最小值而不只是局部最小值？
> [!answer]- 答案
> 因為 $E(\mathbf w)$ 是 $\mathbf w$ 的二次函數，Hessian $\Phi^T\Phi\succeq0$（行滿秩時正定），所以 $E$ 為（嚴格）凸函數，唯一的臨界點就是全域最小值。(來源: [[ch01-deep-learning-revolution]], p.6)

> [!question] Q2. $\Phi^T\Phi$ 可逆需要什麼條件？
> [!answer]- 答案
> $\Phi$ 行滿秩：$N\ge M+1$ 且 $x_n$ 互異（Vandermonde 矩陣行列式不為零）。若 $M+1>N$，$\Phi^T\Phi$ 奇異，需加正則化 $\lambda I$（見 [[regularization-weight-decay]]）或使用偽反矩陣取最小範數解。(來源: [[ch01-deep-learning-revolution]], p.6, p.9)

> [!question] Q3. 為什麼實務上不直接用 `solve(Phi.T @ Phi, Phi.T @ t)` 處理高階多項式？
> [!answer]- 答案
> 💡補充：$\mathrm{cond}(\Phi^T\Phi)=\mathrm{cond}(\Phi)^2$，$M=9$ 時已達約 $10^{14}$，浮點誤差明顯；`lstsq`／QR／SVD 直接作用在 $\Phi$ 上，數值更穩。

> [!question] Q4. 為何深度網路不能像這樣一步解出權重？
> [!answer]- 答案
> 網路輸出對權重非線性，平方和誤差不再是二次函數，沒有閉式解，必須以梯度為基礎的迭代最佳化（SGD）。(來源: [[ch01-deep-learning-revolution]], p.11, p.14)

## 相關頁面

[[polynomial-curve-fitting]] · [[linear-models-basis-functions]] · [[regularization-weight-decay]] · [[cross-validation]] · [[artificial-neuron-perceptron]]
