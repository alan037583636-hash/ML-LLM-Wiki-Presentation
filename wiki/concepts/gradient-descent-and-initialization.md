---
title: 梯度下降家族與參數初始化 (Gradient Descent and Initialization)
type: concept
tags: [gradient-descent, sgd, mini-batch, initialization, he-initialization, xavier]
sources: [ch07-gradient-descent]
created: 2026-11-11
updated: 2026-11-11
---

# 梯度下降家族與參數初始化 (Gradient Descent and Initialization)

機器學習模型的最佳化核心在於如何有效地計算並沿著負梯度方向前進。本頁探討梯度資訊的計算代價、梯度下降三大變體（批次、隨機與小批次）的權衡，以及神經網路參數初始化的對稱性打破與方差守恆推導。

---

## 1. 為什麼要使用梯度資訊？(Computational Complexity of Gradients)

📌講義指出，在選擇最佳化策略時，計算代價（Computational Cost）具有決定性影響：
設網路參數總數為 $W$。
1. **無導數搜尋法 (Derivative-free Optimization)**：
   若不使用導數，僅透過反覆代入參數直接計算誤差函數值 $E(\mathbf{w})$（例如網格搜尋、隨機探索、遺傳演算法），在高維空間中需嘗試的點數隨維度呈指數或 $\mathcal{O}(W^3)$ 次方成長，完全無法應用於實際網路。
2. **數值有限差分法 (Numerical Finite Differences)**：
   對每個參數擾動微小值 $\epsilon$ 以數值估計偏導：
   $$\frac{\partial E}{\partial w_i} \approx \frac{E(\mathbf{w} + \epsilon \mathbf{e}_i) - E(\mathbf{w})}{\epsilon}$$
   計算完整梯度向量 $\nabla E(\mathbf{w})$ 需要進行 $W + 1$ 次完整網路前向傳播。每次前向傳播的計算量為 $\mathcal{O}(W)$，故總時間複雜度為：
   $$\mathcal{O}(W^2)$$
3. **解析反向傳播法 (Analytic Backpropagation)**：
   透過連鎖律解析求導，單次反向傳播的計算量與一次前向傳播相當（約為前向傳播的 2 到 3 倍）。取得所有 $W$ 個參數的精確梯度總共僅需：
   $$\mathcal{O}(W)$$
   這項計算複雜度上的質變，是現代數億至兆級參數深度模型得以成功訓練的根本前提 (來源: [[ch07-gradient-descent]], p.4)。

---

## 2. 梯度下降三大演算法變體 (Gradient Descent Variants)

📌講義對比了三種不同的批次處理模式 (來源: [[ch07-gradient-descent]], p.4–6)：

### 2.1 批次梯度下降 (Batch Gradient Descent, BGD)
每次更新計算整個訓練資料集（$N$ 筆樣本）的完整總誤差梯度：
$$\nabla E(\mathbf{w}) = \sum_{n=1}^N \nabla E_n(\mathbf{w}), \quad \mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} - \eta \nabla E(\mathbf{w}^{(\tau)})$$
- **優點**：梯度方向完全精確，保證沿最陡下降前進，收斂平穩。
- **缺點**：每走一步需遍歷全部 $N$ 筆資料，當 $N$ 極大時計算成本過高；且純凸優化外極易被困在淺鞍點或平坦高原，無法進行線上即時更新 (Online Learning)。

### 2.2 隨機梯度下降 (Stochastic Gradient Descent, SGD)
每次更新僅隨機挑選單一筆資料點 $(\mathbf{x}_n, \mathbf{t}_n)$，以其單點梯度更新參數：
$$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} - \eta \nabla E_n(\mathbf{w}^{(\tau)})$$
- **優點**：單步更新極為迅速；單點梯度是整體全域梯度的無偏估計（$\mathbb{E}_n[\nabla E_n] = \nabla E$）；其帶來的「梯度雜訊（Gradient Noise）」有助於網路逃離狹小鞍點與不良局部極小值。
- **缺點**：軌跡隨機擾動劇烈，在極值點附近持續震盪徘徊；無法充分利用 GPU 巨量並行矩陣運算。

### 2.3 小批次梯度下降 (Mini-batch SGD)
在實務中最廣泛採用的折衷方案：每次隨機抽取大小為 $B$ 的小批次（Mini-batch，例如 $B=32, 64, 128$）：
$$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} - \eta \frac{1}{B} \sum_{n \in \mathcal{B}} \nabla E_n(\mathbf{w}^{(\tau)})$$
- **$\sigma / \sqrt{B}$ 折衷定理**：設單一資料點梯度的方差為 $\sigma^2$，則大小為 $B$ 的獨立隨機樣本平均梯度的標準差為：
  $$\text{SE} = \frac{\sigma}{\sqrt{B}}$$
  當 $B$ 從 1 增加至 100 時，估計精度提升 10 倍；但若再增加到 10000，精度僅再提升 10 倍，計算量卻暴增百倍。因此存在明顯的邊際報酬遞減 (來源: [[ch07-gradient-descent]], p.5–6)。
- **GPU 平行效率**：現代硬體以 SIMD 矩陣相乘為核心架構，計算大小為 $B$ 的矩陣乘法所耗時間與單一向量乘法相差無幾，大幅最大化硬體吞吐量。

---

## 3. 參數初始化與對稱性打破 (Parameter Initialization)

神經網路訓練始於權重初始值 $\mathbf{w}^{(0)}$。初始化的良窳直接決定了前向傳播信號與反向傳播梯度是否會消失或爆炸。

### 3.1 為什麼不能初始化為全零？(Breaking Symmetry)
若將隱藏層所有權重與偏置設為 0：
- 隱藏層所有神經元的前激活值相同（$a_j = 0$）。
- 反向傳播時，所有神經元接收到的誤差反傳梯度完全相等（$\delta_j$ 相同）。
- 梯度更新後，所有權重仍完全一致，網路永遠無法學習多樣化特徵，退化為僅有一個神經元的單層模型。因此必須透過**隨機初始化**打破對稱性 (來源: [[ch07-gradient-descent]], p.6)。

### 3.2 偏置與權重的分工
- **偏置 (Biases)**：通常可直接初始化為全零（$b_j = 0$），因為對稱性已由隨機權重打破。

### 3.3 He (Kaiming) 初始化推導 (He et al., 2015)
📌講義推導了針對 **ReLU** 激活函數的方差守恆初始化準則 (來源: [[ch07-gradient-descent]], p.6–7)：
考慮第 $l$ 層單一神經元的前激活值：
$$a_j = \sum_{i=1}^{D_{\text{in}}} w_{ji} z_i$$
假設權重 $w_{ji}$ 獨立同分布（i.i.d.），均值為 0，變異數為 $\sigma_w^2$；輸入激活值 $z_i$ 亦為同分布，均值與變異數分別為 $\mathbb{E}[z], \text{Var}(z)$。
由於 $w$ 與 $z$ 相互獨立：
$$\mathbb{E}[a_j] = 0$$
$$\text{Var}(a_j) = D_{\text{in}} \cdot \mathbb{E}[w^2 z^2] = D_{\text{in}} \sigma_w^2 \mathbb{E}[z^2]$$
若激活函數為 ReLU：$z = \max(0, a_{\text{prev}})$。由於 $a_{\text{prev}}$ 關於 0 對稱分布，有一半機率為正、一半機率被截斷為 0：
$$\mathbb{E}[z^2] = \frac{1}{2} \mathbb{E}[a_{\text{prev}}^2] = \frac{1}{2} \text{Var}(a_{\text{prev}})$$
將其代入方差表達式：
$$\text{Var}(a_j) = D_{\text{in}} \sigma_w^2 \cdot \frac{1}{2} \text{Var}(a_{\text{prev}}) = \left( \frac{1}{2} D_{\text{in}} \sigma_w^2 \right) \text{Var}(a_{\text{prev}})$$
為了使訊號在深層傳播時信號變異數不衰減（$\text{Var}(a_j) = \text{Var}(a_{\text{prev}})$），必須滿足條件：
$$\frac{1}{2} D_{\text{in}} \sigma_w^2 = 1 \implies \sigma_w^2 = \frac{2}{D_{\text{in}}}$$
此即 **He Normal 初始化**：從高斯分布 $\mathcal{N}\left(0, \frac{2}{D_{\text{in}}}\right)$ 中隨機抽樣初始權重 (來源: [[ch07-gradient-descent]], p.6)。

---

## 自我測驗 Q&A

### Q1: 為什麼利用反向傳播計算梯度比數值有限差分法（Numerical Differentiation）更適用於深層神經網路？
> [!answer]- 答案
> 若使用數值有限差分，每個參數都需要單獨擾動並重新執行一次前向傳播，計算 $W$ 個參數的梯度總共需要執行 $W+1$ 次前向計算，複雜度為 $\mathcal{O}(W^2)$。而反向傳播利用微積分連鎖律，單次反向傳播即可同時取得所有 $W$ 個參數的精確梯度，複雜度僅為 $\mathcal{O}(W)$。在現代參數動輒數百萬至數十億的模型中，$\mathcal{O}(W)$ 是唯一具備計算可行性的途徑。

### Q2: 隨機梯度下降（SGD）相較於批次梯度下降（BGD）為何具有跳出鞍點或較差極小值的潛力？
> [!answer]- 答案
> BGD 每次更新使用全資料集平均，梯度方向是平滑且確定的，一旦接近平坦高原或鞍點（梯度趨近於 0），參數更新極易陷入停滯。而 SGD 每次僅使用單筆樣本計算梯度，單點梯度圍繞總體真實梯度產生波動，引入了隨機噪聲（Gradient Noise）。這種隨機擾動使參數能像布朗運動般震盪，具有較高機率沿負曲率方向彈出鞍點。

### Q3: 試推導並說明為什麼針對 ReLU 激活函數的神經網路，權重初始化變異數應設為 $\sigma_w^2 = \frac{2}{D_{\text{in}}}$ 而非 Xavier 初始化的 $\frac{1}{D_{\text{in}}}$？
> [!answer]- 答案
> 由於前激活值 $a$ 是均值為 0 的對稱隨機變數，經過 ReLU 激活函數 $z = \max(0, a)$ 後，負半軸的一半信號被置為 0，導致二階矩減半：$\mathbb{E}[z^2] = \frac{1}{2} \text{Var}(a)$。若按照全線性假設的 Xavier 初始化（$\sigma_w^2 = \frac{1}{D_{\text{in}}}$），經過每一層後變異數都會減半（$\text{Var}(a_l) = 0.5 \text{Var}(a_{l-1})$），在多層神經網路中前向信號將呈指數級衰減至 0。為了補償 ReLU 造成的 $1/2$ 信號損失，權重變異數必須加倍設定為 $\frac{2}{D_{\text{in}}}$。

---

## 相關頁面
- [[ch07-gradient-descent|第七章 梯度下降講義摘要]]
- [[error-surfaces-and-geometry|誤差曲面與幾何特性]]
- [[momentum-and-adaptive-optimizers|動量法與自適應學習率優化器]]
- [[normalization-techniques|網路正規化技術]]
