---
title: 線性基底函數模型 (Linear Basis Function Models)
type: concept
tags: [linear-models, basis-functions, regression, neural-networks, rbf, wavelets]
sources: [ch04-single-layer-regression]
created: 2026-11-11
updated: 2026-11-11
---

# 線性基底函數模型 (Linear Basis Function Models)

## 1. 定義與架構

📌講義：迴歸的目標是給定 $D$ 維輸入向量 $\mathbf{x}$，預測一個或多個連續目標變數 $t$。最簡單的線性迴歸模型是輸入變數的線性組合：

$$y(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + \cdots + w_D x_D \tag{4.1}$$

此模型雖然對參數 $\mathbf{w}$ 是線性的，但同時對輸入變數 $x_i$ 也是線性的，對模型表達力造成顯著限制。(來源: [[ch04-single-layer-regression]], p.2)

📌講義：為了擴展模型類別，考慮輸入變數之固定非線性函數的線性組合：

$$y(\mathbf{x}, \mathbf{w}) = w_0 + \sum_{j=1}^{M-1} w_j \phi_j(\mathbf{x}) \tag{4.2}$$

其中 $\phi_j(\mathbf{x})$ 稱為**基底函數 (basis functions)**，模型參數總數為 $M$。參數 $w_0$ 稱為**偏置參數 (bias parameter)**，容許資料中任何固定的偏移量（勿與統計上的偏差 bias 混淆）。(來源: [[ch04-single-layer-regression]], p.2)

📌講義：定義虛擬基底函數 $\phi_0(\mathbf{x}) \equiv 1$，則模型可簡記為向量內積形式：

$$y(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M-1} w_j \phi_j(\mathbf{x}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}) \tag{4.3}$$

其中 $\mathbf{w} = (w_0, \dots, w_{M-1})^T$，$\boldsymbol{\phi} = (\phi_0, \dots, \phi_{M-1})^T$。(來源: [[ch04-single-layer-regression]], p.2)

## 2. 單層神經網路的觀點

📌講義：式 (4.3) 可表示為**只有單層可學習參數的簡單類神經網路**：
- 每個基底函數 $\phi_j(\mathbf{x})$ 以一個輸入節點表示（偏置基底 $\phi_0$ 為實心節點）。
- 每個參數 $w_j$ 代表連接基底函數節點與輸出節點的權重連線。
- 輸出節點計算線性組合 $y(\mathbf{x}, \mathbf{w}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})$。

使用非線性基底函數可讓 $y(\mathbf{x}, \mathbf{w})$ 成為輸入 $\mathbf{x}$ 的非線性函數；然而只要模型對參數 $\mathbf{w}$ 保持線性，就統稱為**線性模型 (linear models)**。(來源: [[ch04-single-layer-regression]], p.2–3)

📌講義（深度學習之前 vs. 之後）：
- **傳統做法**：依賴領域專家手工設計固定的前處理（特徵萃取），選定一組強大的基底函數 $\{\phi_j(\mathbf{x})\}$，再由單層網路學習權重。除少數簡單問題外，手工設計極其困難。
- **深度學習**：直接從資料本身學習多層特徵轉換與非線性表徵，迴避了手工打造特徵的瓶頸。(來源: [[ch04-single-layer-regression]], p.3)

## 3. 常見基底函數形式

📌講義介紹了四種主要的基底函數形式：

### 3.1 多項式基底 (Polynomial)
$$\phi_j(x) = x^j$$
第一章的多項式擬合即其特例。(來源: [[ch04-single-layer-regression]], p.3)

### 3.2 高斯基底 (Gaussian / Radial Basis Function, RBF)
$$\phi_j(x) = \exp\left(-\frac{(x - \mu_j)^2}{2s^2}\right) \tag{4.4}$$
其中 $\mu_j$ 決定基底在輸入空間的位置，$s$ 決定空間尺度。此處不需要機率詮釋，正規化係數並不重要（因為會被可學習參數 $w_j$ 吸收）。(來源: [[ch04-single-layer-regression]], p.3)

### 3.3 S 形基底 (Sigmoidal) 與 tanh
$$\phi_j(x) = \sigma\left(\frac{x - \mu_j}{s}\right), \quad \sigma(a) = \frac{1}{1 + \exp(-a)} \tag{4.5}$$
其中 $\sigma(a)$ 為邏輯 S 形函數 (logistic sigmoid)。因 $\tanh(a) = 2\sigma(2a) - 1$，S 形函數的一般線性組合與 $\tanh$ 的一般線性組合能表示同一類輸入–輸出函數。(來源: [[ch04-single-layer-regression]], p.3)

### 3.4 Fourier 基底與小波 (Wavelets)
- **Fourier 基底**：以正弦/餘弦函數展開，每個基底具特定頻率，具有無限空間範圍（頻率完全局部、空間完全全域）。
- **小波 (Wavelets)**：在空間與頻率上皆具備局部化特性，彼此正交，適合處理非平穩訊號或格點資料（如時間序列、影像像素）。(來源: [[ch04-single-layer-regression]], p.3)

📌講義（講義教學補充：訊號處理觀點的基底取捨）：
- 多項式：全域非局部，一處變動影響全曲線，高階時條件數極大（數值病態）。
- 高斯：空間局部，但高維度時中心點數量隨維度呈指數暴增（維度災難）。
- Fourier：頻率局部，為 GNSS/電離層 VTEC 建模中球諧函數與三角級數的基礎。
- 小波：時頻折衷，符合測不準原理。(來源: [[ch04-single-layer-regression]], p.4)

## 自我測驗 Q&A

> [!question] Q1. 線性迴歸模型在神經網路架構中對應什麼角色？
> [!answer]- 答案
> 對應到只有「單層可學習參數」的網路：輸入層為計算固定基底函數 $\phi_j(\mathbf{x})$ 的節點，輸出層為線性激活單元。(來源: [[ch04-single-layer-regression]], p.2–3)

> [!question] Q2. 為何高斯基底函數不需要乘上高斯機率密度的正規化常數 $1/\sqrt{2\pi s^2}$？
> [!answer]- 答案
> 因為基底函數會再乘上可學習的參數 $w_j$，常數尺度可直接被參數吸收，且高斯基底本身無須機率詮釋。(來源: [[ch04-single-layer-regression]], p.3)

> [!question] Q3. 傳統手工特徵工程與深度學習在基底函數觀點上有何根本區別？
> [!answer]- 答案
> 傳統做法由人工設計固定基底函數 $\boldsymbol{\phi}(\mathbf{x})$，僅學習末端組合權重；深度學習則讓基底函數（多層隱藏層表徵）直接從資料中學習。(來源: [[ch04-single-layer-regression]], p.3)

## 相關頁面

[[maximum-likelihood-regression]] · [[geometry-of-least-squares]] · [[linear-models-basis-functions]] · [[ch04-single-layer-regression]]
