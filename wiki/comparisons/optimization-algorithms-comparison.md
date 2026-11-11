---
title: 最佳化演算法綜合比較 (Optimization Algorithms Comparison)
type: comparison
tags: [optimizers, comparison, sgd, momentum, adagrad, rmsprop, adam]
sources: [ch07-gradient-descent]
created: 2026-11-11
updated: 2026-11-11
---

# 最佳化演算法綜合比較 (Optimization Algorithms Comparison)

在深度神經網路中，最佳化演算法決定了參數在複雜高維非凸誤差曲面上的移動軌跡與收斂效率。本頁對梯度下降家族與主流一階自適應優化器進行全方位的橫向比較。

---

## 1. 核心更新公式與機制對照表

| 演算法 | 狀態記憶維護項 | 核心更新公式 | 關鍵特性與超參數 |
|---|---|---|---|
| **標準 SGD** | 無 | $\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} - \eta \mathbf{g}^{(\tau)}$ | 最基礎，但易受條件數影響在病態峽谷震盪。超參數：$\eta$。 |
| **動量法 (Momentum)** | 速度向量 $\mathbf{v}$ (一階動量) | $\mathbf{v}^{(\tau+1)} = \mu \mathbf{v}^{(\tau)} - \eta \mathbf{g}^{(\tau)}$<br>$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} + \mathbf{v}^{(\tau+1)}$ | 引入慣性抑制高頻震盪，平緩方向步長放大為 $\frac{1}{1-\mu}$。超參數：$\eta, \mu \approx 0.9$。 |
| **Nesterov (NAG)** | 速度向量 $\mathbf{v}$ | $\mathbf{v}^{(\tau+1)} = \mu \mathbf{v}^{(\tau)} - \eta \nabla E(\mathbf{w}^{(\tau)} + \mu \mathbf{v}^{(\tau)})$<br>$\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} + \mathbf{v}^{(\tau+1)}$ | 先依慣性預走一步再求梯度，具備主動煞車減速功能。超參數：$\eta, \mu \approx 0.9$。 |
| **AdaGrad** | 累積平方梯度 $\mathbf{s}$ | $s_i^{(\tau)} = s_i^{(\tau-1)} + (g_i^{(\tau)})^2$<br>$w_i^{(\tau+1)} = w_i^{(\tau)} - \frac{\eta}{\sqrt{s_i^{(\tau)} + \epsilon}} g_i^{(\tau)}$ | 頻繁特徵步長小、稀疏特徵步長大。但 $s$ 單調遞增導致學習率後期過早歸零。 |
| **RMSProp** | 指數移動平均 $\mathbf{r}$ (二階矩) | $r_i^{(\tau)} = \beta r_i^{(\tau-1)} + (1-\beta)(g_i^{(\tau)})^2$<br>$w_i^{(\tau+1)} = w_i^{(\tau)} - \frac{\eta}{\sqrt{r_i^{(\tau)} + \epsilon}} g_i^{(\tau)}$ | 引入指數衰減遺忘遠古歷史，解決學習率過早早衰問題。超參數：$\eta, \beta \approx 0.9$。 |
| **Adam** | 一階動量 $\mathbf{s}$ 與二階動量 $\mathbf{r}$ | $s_i^{(\tau)} = \beta_1 s_i^{(\tau-1)} + (1-\beta_1)g_i^{(\tau)}$<br>$r_i^{(\tau)} = \beta_2 r_i^{(\tau-1)} + (1-\beta_2)(g_i^{(\tau)})^2$<br>$\hat{s}_i = \frac{s_i}{1-\beta_1^\tau}, \; \hat{r}_i = \frac{r_i}{1-\beta_2^\tau}$<br>$w_i^{(\tau+1)} = w_i^{(\tau)} - \frac{\eta}{\sqrt{\hat{r}_i} + \epsilon}\hat{s}_i$ | 同時結合動量平滑與自適應尺度調整，並透過偏差修正消除冷啟動偏置。超參數：$\eta, \beta_1=0.9, \beta_2=0.999$。 |

(來源: [[ch07-gradient-descent]], p.8–11)。

---

## 2. 多維度特性評比

| 評比維度 | SGD / Mini-batch | SGD with Momentum | RMSProp | Adam |
|---|---|---|---|---|
| **額外記憶體開銷** | **$0$**（無額外狀態） | **$1 \times W$**（速度向量 $\mathbf{v}$） | **$1 \times W$**（二階動量 $\mathbf{r}$） | **$2 \times W$**（一階動量 $\mathbf{s}$ + 二階動量 $\mathbf{r}$） |
| **對學習率敏感度** | 極高（需精細調諧） | 高（需搭配排程） | 中等（自適應縮放） | **低**（預設超參數通常表現良好） |
| **逃離鞍點能力** | 依賴小批次隨機噪聲 | 良好（藉由慣性衝過平坦區） | 普通 | **優秀**（動量推進與自適應放大微弱梯度） |
| **泛化能力 (邊界性質)** | **極佳**（收斂至平坦極小 Flat Minima） | **極佳** | 偶爾過擬合 | 良好（部分任務需搭配 AdamW 權重衰減） |
| **主流適用場景** | 傳統統計模型、極度注重最終泛化的特定 CV 任務 | 電腦視覺 (ResNet / ConvNet) | 強化學習 (RL)、序列預測 | **大型語言模型 (LLM)、Transformer、多模態通用預設** |

---

## 3. 演進邏輯與總結

```
[標準 SGD] (容易受困於狹長峽谷與病態條件數)
    │
    ├─► 引入物理「質量慣性」 ────────► [Momentum / Nesterov] (消除高頻震盪，加速谷底推進)
    │
    └─► 引入逐參數「自適應縮放」
            │
            ├─► 歷史平方累加 ──────► [AdaGrad] (頻繁特徵自適應減速，但後期早衰凍結)
            │
            └─► 指數加權移動平均 ──► [RMSProp] (解決早衰，適應非平穩環境)
                    │
                    └─► 結合「一階動量」+「二階動量」+「偏差修正」
                            │
                            ▼
                        [Adam] (現代深度學習王者，快速收斂與超參數魯棒性)
```

---

## 相關頁面
- [[ch07-gradient-descent|第七章 梯度下降講義摘要]]
- [[error-surfaces-and-geometry|誤差曲面與幾何特性]]
- [[gradient-descent-and-initialization|梯度下降家族與參數初始化]]
- [[momentum-and-adaptive-optimizers|動量法與自適應學習率優化器]]
- [[normalization-techniques|網路正規化技術]]
