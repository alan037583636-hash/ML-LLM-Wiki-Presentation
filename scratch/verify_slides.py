import json

slides_data = [
    {
        "id": 1,
        "title": "機器學習基礎與模型演進",
        "subtitle": "Machine Learning Foundations & Model Evolution",
        "badge": "課程總覽 · Introduction",
        "points": [
            "核心宗旨：從傳統「手工規則」邁向「讓資料自己學會表徵與規律」。",
            "演進軸線：單層線性模型 ➔ 幾何分類 ➔ 深度表徵 ➔ 高維梯度最佳化。",
            "幾何與統計視角：把模型當作空間座標變換，把訓練當作高維曲面導航。",
            "互動提示：按鍵盤左右方向鍵 [←] [→] 翻頁，點擊右下角按鈕展開數學推導。"
        ],
        "modal_title": "簡報地圖與核心講義脈絡對照",
        "modal_content": """<p>本教學簡報精煉自 Bishop & Bishop (2024) <em>Deep Learning: Foundations and Concepts</em>：</p>
<ul>
  <li><strong>Ch 1 深度學習革命</strong>：三種學習典範、多項式曲線擬合、過度擬合與泛化。</li>
  <li><strong>Ch 4 單層網路：迴歸</strong>：線性基底函數、最大概似估計、正則化最小平方。</li>
  <li><strong>Ch 5 單層網路：分類</strong>：線性判別函數、邏輯迴歸、Softmax、交叉熵損失。</li>
  <li><strong>Ch 6 深度神經網路</strong>：維度災難、多層感知機、表徵學習、MDN 混合密度網路。</li>
  <li><strong>Ch 7 梯度下降</strong>：誤差曲面幾何、SGD、動量、Adam、批次與層正規化。</li>
</ul>"""
    },
    {
        "id": 2,
        "title": "典範轉移：為什麼我們需要機器學習？",
        "subtitle": "Paradigm Shift: From Rules to Data-Driven Learning",
        "badge": "第一單元 · 核心思維",
        "points": [
            "傳統程式設計：人類手寫規則 + 輸入資料 ➔ 產出答案（遇複雜場景崩潰）。",
            "機器學習典範：輸入資料 + 正確答案 ➔ 演算法自動萃取潛在規則與參數。",
            "監督式學習：擁有成對標籤 $(x, t)$，透過最小化預測誤差修正模型。",
            "非監督與自監督：從未標註資料發掘集群結構，或利用資料自身構造預測任務。"
        ],
        "modal_title": "三大學習典範的數學目標對比",
        "modal_content": """<p>三種學習模式對應不同的損失函數目標：</p>
<ol>
  <li><strong>監督式學習 (Supervised)</strong>：最小化負對數概似：<br>$$\\mathcal{L}_{\\text{sup}} = -\\sum_{n=1}^N \\ln p(t_n \\mid x_n, \\mathbf{w})$$</li>
  <li><strong>非監督式學習 (Unsupervised)</strong>：學習資料生成密度 $p(x)$ 或低維潛在結構，如高斯混合模型 GMM：<br>$$p(x) = \\sum_{k=1}^K \\pi_k \\mathcal{N}(x \\mid \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)$$</li>
  <li><strong>自監督式學習 (Self-supervised)</strong>：利用資料自身結構建立對比預測（如 InfoNCE）：<br>$$\\mathcal{L}_{\\text{InfoNCE}} = -\\ln \\frac{\\exp(\\mathbf{q} \\cdot \\mathbf{k}_+ / \\tau)}{\\sum_i \\exp(\\mathbf{q} \\cdot \\mathbf{k}_i / \\tau)}$$</li>
</ol>"""
    },
    {
        "id": 3,
        "title": "單層模型起點：線性迴歸與多項式擬合",
        "subtitle": "Polynomial Curve Fitting: The Toy Problem of ML",
        "badge": "第一單元 · 核心思維",
        "points": [
            "擬合目標：給定含雜訊的離散樣本，找出一條平滑曲線逼近真實規律。",
            "多項式模型：$y(x, \\mathbf{w}) = w_0 + w_1 x + w_2 x^2 + \\dots + w_M x^M$。",
            "本質是線性模型：對輸入 $x$ 是非線性的，但對權重 $\\mathbf{w}$ 嚴格保持線性。",
            "最小平方法 (OLS)：藉由最小化預測殘差平方和，求得閉式解析解。"
        ],
        "modal_title": "最小平方法正規方程 (Normal Equations) 推導",
        "modal_content": """<p>定義平方誤差函數：</p>
$$E(\\mathbf{w}) = \\frac{1}{2} \\sum_{n=1}^N (y(x_n, \\mathbf{w}) - t_n)^2 = \\frac{1}{2} \\|\\boldsymbol{\\Phi} \\mathbf{w} - \\mathbf{t}\\|^2$$
<p>其中 $\\boldsymbol{\\Phi}_{ni} = \\phi_i(x_n) = x_n^i$ 為設計矩陣（Design Matrix）。對 $\\mathbf{w}$ 求梯度並令其為零：</p>
$$\\nabla_\\mathbf{w} E(\\mathbf{w}) = \\boldsymbol{\\Phi}^T (\\boldsymbol{\\Phi} \\mathbf{w} - \\mathbf{t}) = \\mathbf{0} \\implies \\boldsymbol{\\Phi}^T \\boldsymbol{\\Phi} \\mathbf{w} = \\boldsymbol{\\Phi}^T \\mathbf{t}$$
<p>當 $\\boldsymbol{\\Phi}^T \\boldsymbol{\\Phi}$ 可逆時，得到唯一的正規方程閉式解：</p>
$$\\mathbf{w}^* = (\\boldsymbol{\\Phi}^T \\boldsymbol{\\Phi})^{-1} \\boldsymbol{\\Phi}^T \\mathbf{t} = \\boldsymbol{\\Phi}^+ \\mathbf{t}$$
<p>在幾何上，預測向量 $\\mathbf{y}^* = \\boldsymbol{\\Phi} \\mathbf{w}^*$ 恰好是目標向量 $\\mathbf{t}$ 在設計矩陣行空間（Column Space）上的正交投影。</p>"""
    },
    {
        "id": 4,
        "title": "機器學習的頭號大敵：過度擬合 (Overfitting)",
        "subtitle": "When Good Training Means Bad Generalization",
        "badge": "第一單元 · 核心思維",
        "points": [
            "欠擬合 (M=1)：模型表達力不足，無法捕捉資料的非線性規律。",
            "適度擬合 (M=3)：平滑穿過數據趨勢，成功學到母體 $\\sin(2\\pi x)$。",
            "過度擬合 (M=9)：完美穿過所有訓練點，但在點之間產生瘋狂震盪！",
            "泛化鴻溝：訓練誤差趨近 0，未看過的測試集誤差卻指數級垂直飆升。"
        ],
        "modal_title": "過度擬合的數值本質：權重係數爆炸",
        "modal_content": """<p>當多項式階數 $M=9$ 擬合僅 10 個訓練點時，模型為了強制經過每一個含雜訊的離散點，權重數值會暴增到不可思議的量級（例如數十萬至數百萬）：</p>
<table style=\"width:100%; border-collapse:collapse; margin:10px 0;\">
  <tr style=\"border-bottom:1px solid #4b5563;\"><th style=\"text-align:left;\">階數 M</th><th style=\"text-align:right;\">$w_0$</th><th style=\"text-align:right;\">$w_1$</th><th style=\"text-align:right;\">$w_6$</th><th style=\"text-align:right;\">$w_9$</th></tr>
  <tr><td>M=3</td><td style=\"text-align:right;\">0.82</td><td style=\"text-align:right;\">-12.7</td><td style=\"text-align:right;\">-</td><td style=\"text-align:right;\">-</td></tr>
  <tr style=\"color:#f87171;\"><td>M=9</td><td style=\"text-align:right;\">0.35</td><td style=\"text-align:right;\">232.3</td><td style=\"text-align:right;\">-1.4e5</td><td style=\"text-align:right;\">3.8e5</td></tr>
</table>
<p>相鄰高階多項式項以極大的正負係數相互精密抵消，只要輸入 $x$ 產生微小擾動，輸出就會巨幅震盪。評估指標均方根誤差為：</p>
$$E_{\\text{RMS}} = \\sqrt{\\frac{2 E(\\mathbf{w}^*)}{N}}$$"""
    },
    {
        "id": 5,
        "title": "馴服過擬合：正則化與偏差-變異權衡",
        "subtitle": "Regularization & The Bias-Variance Tradeoff",
        "badge": "第一單元 · 核心思維",
        "points": [
            "權重衰減 ($L_2$ 正則化)：在損失中加入罰項 $\\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2$，約束權重膨脹。",
            "偏差 (Bias)：模型本身的假設局限；高偏差導致「學不會」（欠擬合）。",
            "變異 (Variance)：模型對樣本的敏感度；高變異導致「過度敏感」（過擬合）。",
            "黃金平衡：調節正則化係數 $\\lambda$，在死板與過敏之間取得泛化最優點。"
        ],
        "modal_title": "脊迴歸閉式解與偏差-變異分解推導",
        "modal_content": """<p>加入 $L_2$ 正則化（脊迴歸 Ridge Regression）之損失函數：</p>
$$\\tilde{E}(\\mathbf{w}) = \\frac{1}{2} \\|\\boldsymbol{\\Phi}\\mathbf{w} - \\mathbf{t}\\|^2 + \\frac{\\lambda}{2} \\mathbf{w}^T \\mathbf{w}$$
<p>對 $\\mathbf{w}$ 求導令為零，得解析解：</p>
$$\\mathbf{w}^* = (\\boldsymbol{\\Phi}^T \\boldsymbol{\\Phi} + \\lambda \\mathbf{I})^{-1} \\boldsymbol{\\Phi}^T \\mathbf{t}$$
<p><strong>期望平方損失的偏差-變異-雜訊三重分解：</strong></p>
$$\\mathbb{E}_\\mathcal{D}[(y(x;\\mathcal{D}) - t)^2] = \\underbrace{(\\mathbb{E}_\\mathcal{D}[y(x;\\mathcal{D})] - h(x))^2}_{(\\text{Bias})^2} + \\underbrace{\\mathbb{E}_\\mathcal{D}[(y(x;\\mathcal{D}) - \\mathbb{E}_\\mathcal{D}[y(x;\\mathcal{D})])^2}_{\\text{Variance}} + \\underbrace{\\sigma^2}_{\\text{Noise}}$$"""
    },
    {
        "id": 6,
        "title": "決策的幾何學：線性判別函數",
        "subtitle": "Linear Discriminant Functions & Decision Boundaries",
        "badge": "第二單元 · 分類與邊界",
        "points": [
            "空間切分：分類本質是在特徵空間中劃出決策邊界（Decision Surface）。",
            "超平面門檻：由線性方程 $y(\\mathbf{x}) = \\mathbf{w}^T \\mathbf{x} + w_0 = 0$ 決定類別分界。",
            "幾何法向量：權重向量 $\\mathbf{w}$ 正交於超平面，偏置 $w_0$ 決定其與原點截距。",
            "正交投影距離：任意樣本點到邊界的垂直距離為 $r = \\frac{y(\\mathbf{x})}{\\|\\mathbf{w}\\|}$。"
        ],
        "modal_title": "幾何投影證明與最小平方分類的致命缺陷",
        "modal_content": """<p>在決策面 $y(\\mathbf{x})=0$ 上任取兩點 $\\mathbf{x}_A, \\mathbf{x}_B$：</p>
$$\\mathbf{w}^T \\mathbf{x}_A + w_0 = \\mathbf{w}^T \\mathbf{x}_B + w_0 = 0 \\implies \\mathbf{w}^T (\\mathbf{x}_A - \\mathbf{x}_B) = 0$$
<p>證明了 $\\mathbf{w}$ 必然正交於超平面上的任意向量。</p>
<p><strong>為什麼不能直接用最小平方法做分類？</strong><br>
最小平方會嚴格懲罰「預測值遠大於 1 的正樣本」（過度正確點），將其誤認為巨大殘差，導致決策邊界被極端值嚴重拖曳偏轉，分類準確率急遽下降。</p>"""
    },
    {
        "id": 7,
        "title": "機率的優雅：邏輯迴歸與 Logistic Sigmoid",
        "subtitle": "Probabilistic Classification & The Sigmoid Function",
        "badge": "第二單元 · 分類與邊界",
        "points": [
            "告別生硬階躍：階躍函數處處不可導；Sigmoid 實現全域平滑映射。",
            "輸出壓入區間：$\\sigma(a) = \\frac{1}{1 + e^{-a}}$ 將實數激活值嚴格壓縮至 $(0, 1)$。",
            "後驗機率詮釋：輸出值可自然視為條件機率 $p(C_1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^T\\mathbf{x} + w_0)$。",
            "飽和區隱憂：當 $|a|$ 較大時導數 $\\sigma'(a) \\to 0$，潛藏梯度消失風險。"
        ],
        "modal_title": "生成式視角：Sigmoid 函數的嚴格貝氏推導",
        "modal_content": """<p>考慮兩類分類問題，依貝氏定理，類別 $C_1$ 的後驗機率為：</p>
$$p(C_1 \\mid \\mathbf{x}) = \\frac{p(\\mathbf{x} \\mid C_1) p(C_1)}{p(\\mathbf{x} \\mid C_1) p(C_1) + p(\\mathbf{x} \\mid C_2) p(C_2)} = \\frac{1}{1 + \\exp(-a)} = \\sigma(a)$$
<p>其中對數勝算比（Log-odds / Logit）為：</p>
$$a = \\ln \\frac{p(\\mathbf{x} \\mid C_1) p(C_1)}{p(\\mathbf{x} \\mid C_2) p(C_2)}$$
<p>若兩類條件密度皆為同協方差高斯分布 $\\mathcal{N}(\\mathbf{x} \\mid \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma})$，展開後二次項精確抵消，$a$ 嚴格退化為 $\\mathbf{x}$ 的線性函數 $\\mathbf{w}^T \\mathbf{x} + w_0$！</p>"""
    },
    {
        "id": 8,
        "title": "損失函數的天生一對：交叉熵與抗梯度消失",
        "subtitle": "Cross-Entropy Loss & The Canonical Pairing",
        "badge": "第二單元 · 分類與邊界",
        "points": [
            "最大概似推導：假設標籤服從白努利分布，匯出二元交叉熵 (BCE)。",
            "負對數損失：$E = -[t \\ln y + (1 - t) \\ln(1 - y)]$，誤差越大處罰越強烈。",
            "導數精確抵消：對數倒數剛好消解 Sigmoid 飽和微商 $y(1-y)$。",
            "極簡前激活梯度：偏導直接化簡為直觀的預測差值 $\\frac{\\partial E}{\\partial a} = y - t$！"
        ],
        "modal_title": "前激活梯度 y - t 的微積分連鎖律抵消證明",
        "modal_content": """<p>單一資料點交叉熵：$E = -[t \\ln y + (1-t) \\ln(1-y)]$，其中 $y = \\sigma(a) = \\frac{1}{1+e^{-a}}$。</p>
<p>利用連鎖律求 $E$ 對前激活值 $a$ 的偏導：</p>
$$\\frac{\\partial E}{\\partial a} = \\frac{\\partial E}{\\partial y} \\cdot \\frac{\\partial y}{\\partial a}$$
<p>計算兩項導數：</p>
$$\\frac{\\partial E}{\\partial y} = -\\frac{t}{y} + \\frac{1-t}{1-y} = \\frac{y - t}{y(1-y)}$$
$$\\frac{\\partial y}{\\partial a} = \\sigma(a)(1 - \\sigma(a)) = y(1-y)$$
<p>兩者相乘，分母 $y(1-y)$ 完美對消：</p>
$$\\frac{\\partial E}{\\partial a} = \\frac{y - t}{y(1-y)} \\cdot y(1-y) = y - t \\quad \\blacksquare$$
<p>若誤用 MSE 損失，$\\frac{\\partial E}{\\partial a} = (y-t)y(1-y)$，在嚴重預測錯誤時梯度會直接消失！</p>"""
    },
    {
        "id": 9,
        "title": "擴展至多元宇宙：Softmax 迴歸與多類別決策",
        "subtitle": "Multiclass Classification & The Softmax Operator",
        "badge": "第二單元 · 分類與邊界",
        "points": [
            "多類別輸出：設有 $K$ 個互斥類別，標籤採 1-of-K One-hot 編碼。",
            "Softmax 映射：$y_k = \\frac{\\exp(a_k)}{\\sum_j \\exp(a_j)}$，約束所有輸出非負且總和為 1。",
            "勝者放大效應：指數運算自然拉大首名差距，形成平滑的機率競爭態勢。",
            "統一步調：多類別交叉熵對前激活值的導數依然優雅保持為 $y_k - t_k$。"
        ],
        "modal_title": "Softmax 交叉熵梯度推導與不變性性質",
        "modal_content": """<p>Softmax 對各前激活值的偏導為：</p>
$$\\frac{\\partial y_j}{\\partial a_k} = y_j (\\delta_{jk} - y_k)$$
<p>代入多類別交叉熵 $E = -\\sum_{j} t_j \\ln y_j$：</p>
$$\\frac{\\partial E}{\\partial a_k} = -\\sum_j \\frac{t_j}{y_j} \\frac{\\partial y_j}{\\partial a_k} = -\\sum_j t_j (\\delta_{jk} - y_k) = -t_k + y_k \\sum_j t_j$$
<p>因 One-hot 標籤滿足 $\\sum_j t_j = 1$，直接得到極簡形式：</p>
$$\\frac{\\partial E}{\\partial a_k} = y_k - t_k$$
<p><strong>數值平移不變性</strong>：若對所有 $a_k$ 加上任意常數 $c$，Softmax 機率恆不變，實務上常令 $c = -\\max_j a_j$ 防止浮點數溢位（Overflow）。</p>"""
    },
    {
        "id": 10,
        "title": "淺層模型的嘆息：XOR 困境與維度災難",
        "subtitle": "The Limits of Shallow Models & Curse of Dimensionality",
        "badge": "第三單元 · 深度與表徵",
        "points": [
            "XOR 線性不可分：單層感知機只能切出超平面，對最基礎的「異或」邏輯束手無策。",
            "AI 寒冬之因：Minsky & Papert (1969) 指出單層限制，導致神經網路研究陷入長期停滯。",
            "維度災難：高維空間中欲靠固定基底網格覆蓋，參數數量呈 $10^D$ 指數級毀滅性暴增。",
            "幾何反直覺：高維超球體積幾乎 100% 集中於極薄表面，傳統局部平滑估計徹底失效。"
        ],
        "modal_title": "維度災難的高維超球體積集中性數學證明",
        "modal_content": """<p>考慮 $D$ 維半徑為 $r$ 的超球，其體積為 $V_D(r) = S_D r^D$。考慮外半徑 $r$、內半徑 $r(1-\\epsilon)$ 的表面薄殼，其佔整體體積的比例為：</p>
$$\\frac{V_D(r) - V_D(r(1-\\epsilon))}{V_D(r)} = 1 - (1 - \\epsilon)^D$$
<p>當空間維度 $D \\to \\infty$ 時，對於任何極小之 $\\epsilon > 0$：</p>
$$\\lim_{D \\to \\infty} [1 - (1 - \\epsilon)^D] = 1$$
<p>高維球體內部幾乎完全是「空心」的！同樣地，高斯分布的機率質量亦完全集中在半徑 $r \\approx \\sqrt{D}$ 的極薄球面薄殼上，傳統 RBF 基底函數在高維空間退化為無效的局部查表。</p>"""
    },
    {
        "id": 11,
        "title": "救贖之道：多層前饋網路與空間折疊",
        "subtitle": "Multilayer Feedforward Networks & Space Folding",
        "badge": "第三單元 · 深度與表徵",
        "points": [
            "空間折疊術：隱藏層神經元透過非線性激活，將原始輸入空間彎曲扭折。",
            "化不可分為可分：在隱藏層的表徵空間中，原本糾纏的 XOR 資料點被拉平為線性可分！",
            "通用近似定理：單一隱藏層只要節點數充足，理論上可逼近任意連續非線性函數。",
            "深度的勝利：淺層網路逼近特定函數需 $2^D$ 個節點，而深層網路能以指數級緊湊參數達成。"
        ],
        "modal_title": "兩層前饋網路矩陣形式與權重空間對稱性",
        "modal_content": """<p>標準兩層前饋網路之矩陣前向傳播：</p>
$$\\mathbf{a} = \\mathbf{W}^{(1)} \\mathbf{x} + \\mathbf{b}^{(1)}, \\quad \\mathbf{z} = h(\\mathbf{a})$$
$$\\mathbf{y} = \\sigma(\\mathbf{W}^{(2)} \\mathbf{z} + \\mathbf{b}^{(2)})$$
<p><strong>權重空間對稱性（Symmetries）</strong>：</p>
<ol>
  <li><strong>置換對稱性</strong>：$M$ 個隱藏單元任意調換順序，網路輸出不變，存在 $M!$ 種對稱點。</li>
  <li><strong>正負號翻轉對稱性</strong>：若激活函數為奇函數（如 $\\tanh$），同時反轉進出權重符號不改變函數值，存在 $2^M$ 種組合。</li>
</ol>
<p>兩層網路在權重空間中至少存在 $M! \\cdot 2^M$ 個完全等價的局部極小值點！</p>"""
    },
    {
        "id": 12,
        "title": "激活函數的演進：從 Sigmoid 走向 ReLU",
        "subtitle": "Activation Functions: The Breakthrough of ReLU",
        "badge": "第三單元 · 深度與表徵",
        "points": [
            "Sigmoid 的詛咒：深層反向傳播時，導數 $\\sigma'(a) \\le 0.25$ 連乘導致梯度指數級衰減至零。",
            "雙曲正切 Tanh：具備零中心化（Zero-centered）特性，但依然無法擺脫兩端飽和區。",
            "ReLU 的革命：$h(a) = \\max(0, a)$ 在正半軸導數恆為 1，徹底打破梯度消失枷鎖！",
            "計算超輕量：僅需簡單閾值比對；負半軸歸零自然誘發稀疏表徵（Sparse Representation）。"
        ],
        "modal_title": "現代激活函數族譜導函數與 Dead ReLU 解析",
        "modal_content": """<table style=\"width:100%; border-collapse:collapse; margin:10px 0;\">
  <tr style=\"border-bottom:1px solid #4b5563;\"><th style=\"text-align:left;\">函數名稱</th><th style=\"text-align:left;\">定義式 $h(a)$</th><th style=\"text-align:left;\">導函數 $h'(a)$</th><th style=\"text-align:left;\">特性</th></tr>
  <tr><td>Logistic Sigmoid</td><td>$\\frac{1}{1+e^{-a}}$</td><td>$h(1-h)$</td><td>嚴重梯度消失</td></tr>
  <tr><td>Tanh</td><td>$\\frac{e^a-e^{-a}}{e^a+e^{-a}}$</td><td>$1-h^2$</td><td>零中心化，仍飽和</td></tr>
  <tr style=\"color:#38bdf8;\"><td>ReLU</td><td>$\\max(0, a)$</td><td>$1 \\; (a>0) \\mid 0 \\; (a<0)$</td><td>無正飽和區，計算快</td></tr>
  <tr><td>Leaky ReLU</td><td>$\\max(\\alpha a, a)$</td><td>$1 \\; (a>0) \\mid \\alpha \\; (a<0)$</td><td>避免神經元永久壞死</td></tr>
  <tr><td>Softplus</td><td>$\\ln(1+e^a)$</td><td>$\\sigma(a)$</td><td>ReLU 的平滑近似</td></tr>
</table>
<p><strong>Dead ReLU 現象</strong>：若神經元權重被過大梯度更新至負區，該單元在後續所有樣本上前激活均小於 0，梯度恆為 0 且永不復甦。</p>"""
    },
    {
        "id": 13,
        "title": "深度學習的真正秘密：階層式表徵學習",
        "subtitle": "Hierarchical Representations & Transfer Learning",
        "badge": "第三單元 · 深度與表徵",
        "points": [
            "表徵學習：深度學習不再靠人工設計特徵，而是端到端自動學習特徵轉換。",
            "特徵金字塔：淺層提取邊緣線條 ➔ 中層組合形狀部件 ➔ 深層形成完整概念與語意。",
            "分散式表徵：概念由多神經元組合編碼，具備指數級資訊容量與平滑泛化力。",
            "遷移學習 (Transfer)：底層通用特徵可直接借用，微調頂層即可征服新任務。"
        ],
        "modal_title": "對比式學習 InfoNCE 損失函數與 CLIP 跨模態對齊",
        "modal_content": """<p>現代自監督表徵學習依賴<strong>對比式學習 (Contrastive Learning)</strong>，其核心為 <strong>InfoNCE 損失函數</strong>：</p>
$$\\mathcal{L}_{\\text{InfoNCE}} = -\\ln \\frac{\\exp(\\mathbf{q} \\cdot \\mathbf{k}_+ / \\tau)}{\\exp(\\mathbf{q} \\cdot \\mathbf{k}_+ / \\tau) + \\sum_{i=1}^K \\exp(\\mathbf{q} \\cdot \\mathbf{k}_{-, i} / \\tau)}$$
<p>其中 $\\tau > 0$ 為溫度係數，調控難負樣本（Hard Negatives）的處罰敏感度。</p>
<p><strong>OpenAI CLIP 架構</strong>：同時訓練影像編碼器 $f_I$ 與文字編碼器 $f_T$，在一個 Batch 內最大化正配對餘弦相似度並最小化負配對相似度，達成前所未有的強大零樣本遷移（Zero-shot Transfer）能力。</p>"""
    },
    {
        "id": 14,
        "title": "誤差曲面的高維迷宮：駐點、鞍點與條件數",
        "subtitle": "Geometry of Error Surfaces: The Ravine Problem",
        "badge": "第四單元 · 梯度與最佳化",
        "points": [
            "高維幾何真相：參數維度動輒百萬，所有方向曲率皆為正的機率僅 $(1/2)^W$。",
            "鞍點遍布：神經網路訓練最大的阻礙不是局部極小，而是大量鞍點與平坦高原！",
            "Hessian 矩陣：二階偏導矩陣 $\\mathbf{H}$ 描述局部曲率，其特徵值決定地形起伏。",
            "病態條件數：最大曲率與最小曲率比值 $\\kappa = \\frac{\\lambda_{\\max}}{\\lambda_{\\min}} \\gg 1$，形成狹長峽谷地形。"
        ],
        "modal_title": "局部二次泰勒展開與收斂極限推導",
        "modal_content": """<p>在駐點 $\\mathbf{w}^*$ 附近進行二次泰勒級數展開：</p>
$$E(\\mathbf{w}) \\approx E(\\mathbf{w}^*) + \\frac{1}{2} (\\mathbf{w} - \\mathbf{w}^*)^T \\mathbf{H} (\\mathbf{w} - \\mathbf{w}^*)$$
<p>利用 Hessian 的正交特徵分解 $\\mathbf{H} = \\sum_i \\lambda_i \\mathbf{u}_i \\mathbf{u}_i^T$，座標完全正交解耦：</p>
$$E(\\mathbf{w}) \\approx E(\\mathbf{w}^*) + \\frac{1}{2} \\sum_{i=1}^W \\lambda_i \\alpha_i^2, \\quad \\alpha_i = \\mathbf{u}_i^T (\\mathbf{w} - \\mathbf{w}^*)$$
<p>梯度下降每步在特徵軸上的遞迴為 $\\alpha_i^{(\\tau+1)} = (1 - \\eta \\lambda_i) \\alpha_i^{(\\tau)}$。收斂穩定條件為 $|1 - \\eta \\lambda_i| < 1$，故學習率受限於：</p>
$$\\eta < \\frac{2}{\\lambda_{\\max}}$$"""
    },
    {
        "id": 15,
        "title": "訓練引擎的心跳：批次、隨機與小批次 SGD",
        "subtitle": "Batch vs Stochastic vs Mini-batch Gradient Descent",
        "badge": "第四單元 · 梯度與最佳化",
        "points": [
            "批次梯度下降 (BGD)：使用全量資料算精確梯度；走法穩定，但計算沉重且易困在鞍點。",
            "隨機梯度下降 (SGD)：單點梯度更新極快；隨機噪聲如同高溫熱擾動，能彈出鞍點。",
            "小批次 (Mini-batch)：折衷黃金解！梯度估計標準差依 $\\frac{\\sigma}{\\sqrt{B}}$ 縮減。",
            "GPU 並行暴力：計算大小為 $B$ 的張量矩陣乘法極為高效，最大化硬體吞吐量。"
        ],
        "modal_title": "梯度計算複雜度跨越與 He (Kaiming) 初始化推導",
        "modal_content": """<p><strong>計算代價對比</strong>：數值有限差分求梯度需 $\\mathcal{O}(W^2)$；解析反向傳播僅需 $\\mathcal{O}(W)$，是深度學習得以規模化的核心工程奇蹟。</p>
<p><strong>He 初始化推導</strong>：對 ReLU 網路，輸入方差有一半被負截斷清零：$\\mathbb{E}[z^2] = \\frac{1}{2} \\text{Var}(a_{\\text{prev}})$。第 $l$ 層前激活方差為：</p>
$$\\text{Var}(a) = D_{\\text{in}} \\sigma_w^2 \\cdot \\frac{1}{2} \\text{Var}(a_{\\text{prev}})$$
<p>為維持前向傳播方差守恆，權重初始方差必須翻倍：</p>
$$\\sigma_w^2 = \\frac{2}{D_{\\text{in}}} \\implies w \\sim \\mathcal{N}\\left(0, \\frac{2}{D_{\\text{in}}}\\right)$$"""
    },
    {
        "id": 16,
        "title": "突破峽谷震盪：動量法與 Nesterov 慣性",
        "subtitle": "Accelerating Optimization: Momentum & Nesterov",
        "badge": "第四單元 · 梯度與最佳化",
        "points": [
            "之字形悲劇：在病態峽谷中，普通 SGD 在陡峭岩壁間瘋狂橫跳，谷底前進如龜速。",
            "動量法 (Momentum)：引入物理慣性質量，將速度向量累積加權：$\\mathbf{v} \\gets \\mu \\mathbf{v} - \\eta \\mathbf{g}$。",
            "橫消縱加：反向震盪的橫向分量互相抵消，同向推動的縱向分量加速疊加為 $\\frac{1}{1-\\mu}$ 倍。",
            "Nesterov (NAG)：先按當前慣性滑行預估下一步位置，再計算梯度，具備主動煞車能力。"
        ],
        "modal_title": "動量法等效步長放大與 Nesterov 預測煞車機制",
        "modal_content": """<p><strong>古典動量更新式</strong>：</p>
$$\\mathbf{v}^{(\\tau+1)} = \\mu \\mathbf{v}^{(\\tau)} - \\eta \\nabla E(\\mathbf{w}^{(\\tau)}), \\quad \\mathbf{w}^{(\\tau+1)} = \\mathbf{w}^{(\\tau)} + \\mathbf{v}^{(\\tau+1)}$$
<p>在平緩谷底方向，設每步梯度恆為常數 $g$。穩定狀態速度滿足：</p>
$$v^* = \\mu v^* - \\eta g \\implies v^* = -\\frac{\\eta}{1 - \\mu} g$$
<p>若設 $\\mu = 0.9$，有效推動步長整整放大了 10 倍！</p>
<p><strong>Nesterov 加速動量 (NAG)</strong>：</p>
$$\\mathbf{v}^{(\\tau+1)} = \\mu \\mathbf{v}^{(\\tau)} - \\eta \\nabla E(\\mathbf{w}^{(\\tau)} + \\mu \\mathbf{v}^{(\\tau)})$$
<p>在谷底衝向對面上坡前，未來預測點已提前出現逆向梯度，產生「主動減速煞車」效果，顯著減少 Overshooting。</p>"""
    },
    {
        "id": 17,
        "title": "現代優化之王：從 RMSProp 到 Adam",
        "subtitle": "Adaptive Learning Rates: AdaGrad to Adam",
        "badge": "第四單元 · 梯度與最佳化",
        "points": [
            "AdaGrad：累積歷史平方梯度為每個參數自適應縮放，但分母單調累加導致後期過早凍結。",
            "RMSProp：改採指數加權移動平均（EMA），只記憶近期梯度方差，解決學習率早衰問題。",
            "Adam：同時維護一階動量（速度均值）與二階動量（未中心化變異數），集大成者。",
            "偏差修正：巧妙消除初始化為 0 帶來的冷啟動偏置，成為現代深度學習的首選優化器。"
        ],
        "modal_title": "Adam 優化器偏差修正 (Bias Correction) 數學證明",
        "modal_content": """<p>Adam 維護一階動量 $s$ 與二階動量 $r$：</p>
$$s^{(\\tau)} = \\beta_1 s^{(\\tau-1)} + (1-\\beta_1) g^{(\\tau)}, \\quad r^{(\\tau)} = \\beta_2 r^{(\\tau-1)} + (1-\\beta_2) (g^{(\\tau)})^2$$
<p>因初始狀態 $s^{(0)} = 0$，展開遞迴式取期望值：</p>
$$\\mathbb{E}[s^{(\\tau)}] = \\mathbb{E}\\left[(1-\\beta_1)\\sum_{k=1}^\\tau \\beta_1^{\\tau-k} g^{(k)}\\right] = \\mathbb{E}[g] (1-\\beta_1) \\frac{1-\\beta_1^\\tau}{1-\\beta_1} = \\mathbb{E}[g](1 - \\beta_1^\\tau)$$
<p>前期期望值嚴重縮水！因此必須透過無偏修正補償：</p>
$$\\hat{s}^{(\\tau)} = \\frac{s^{(\\tau)}}{1 - \\beta_1^\\tau}, \\quad \\hat{r}^{(\\tau)} = \\frac{r^{(\\tau)}}{1 - \\beta_2^\\tau}$$
<p>最終更新式：$w_i \\gets w_i - \\frac{\\eta}{\\sqrt{\\hat{r}_i} + \\epsilon} \\hat{s}_i$。</p>"""
    },
    {
        "id": 18,
        "title": "隱形守護者：網路正規化 (BatchNorm vs LayerNorm)",
        "subtitle": "Internal Stabilization: Normalization Techniques",
        "badge": "第五單元 · 穩定與總結",
        "points": [
            "輸入標準化：消除各維度尺度懸殊，重塑 Hessian 等高線為對稱球形，收斂十倍速。",
            "內部共變量偏移：前層微小權重變動會在深層引起分布雪崩；正規化為深層提供穩定地基。",
            "BatchNorm：跨 Batch 維度計算均值與變異數，搭配可學習參數 $\\gamma, \\beta$，廣泛用於 CNN。",
            "LayerNorm：跨 Feature 維度計算，完全脫鉤批次大小 $B$，是 Transformer 與 LLM 的心臟。"
        ],
        "modal_title": "張量切片幾何 (B, D) 與 BatchNorm 推論期移動平均",
        "modal_content": """<p>設層激活張量形狀為 $(B, D)$（批次大小 $B$，特徵維度 $D$）：</p>
<ul>
  <li><strong>Batch Normalization (BatchNorm)</strong>：沿縱向 Batch 軸聚合。<br>對特徵 $j$：$\\mu_j = \\frac{1}{B}\\sum_i x_{ij}$，$\\sigma_j^2 = \\frac{1}{B}\\sum_i (x_{ij}-\\mu_j)^2$。</li>
  <li><strong>Layer Normalization (LayerNorm)</strong>：沿橫向 Feature 軸聚合。<br>對樣本 $i$：$\\mu_i = \\frac{1}{D}\\sum_j x_{ij}$，$\\sigma_i^2 = \\frac{1}{D}\\sum_j (x_{ij}-\\mu_i)^2$。</li>
</ul>
<p><strong>BatchNorm 推論期無縫切換</strong>：<br>
推論時可能只有單一樣本（$B=1$），此時直接代入訓練期間以 EMA 記錄的全局統計量：</p>
$$y = \\gamma \\left(\\frac{x - \\mu_{\\text{run}}}{\\sqrt{\\sigma_{\\text{run}}^2 + \\epsilon}}\\right) + \\beta$$"""
    },
    {
        "id": 19,
        "title": "結語與全景演進地圖：邁向未來的機器學習",
        "subtitle": "The Big Picture: From Hand-Crafted to Learned Representations",
        "badge": "第五單元 · 穩定與總結",
        "points": [
            "宏觀演進：固定基底模型 ➔ 幾何邊界劃分 ➔ 空間多層折疊 ➔ 高維自適應導航。",
            "核心智慧：深度學習的成功在於「以可微架構取代人工假設，讓資料自適應流形」。",
            "泛化哲學：目標從來不是精確最小化訓練誤差，而是找到具備平坦曲率的優良解池。",
            "展望未來：大型基礎模型、跨模態對齊與大規模最佳化，持續譜寫智慧革命新篇章。"
        ],
        "modal_title": "知識庫完整連結與進階延伸主題",
        "modal_content": """<p>本簡報相關概念於專案 Wiki 知識庫中均有獨立專頁深入探討：</p>
<ul>
  <li><strong>混合密度網路 (MDN)</strong>：解決逆問題中的多峰條件機率分布。參閱 <code>wiki/concepts/mixture-density-networks.md</code></li>
  <li><strong>Probit 迴歸與典則連結</strong>：隨機噪聲閾值模型。參閱 <code>wiki/concepts/probit-regression-canonical-link.md</code></li>
  <li><strong>決策論與損失矩陣</strong>：推論與決策解耦思維。參閱 <code>wiki/concepts/decision-theory-regression.md</code></li>
</ul>
<p>恭喜您完成《機器學習基礎與模型演進》全景探索！</p>"""
    }
]

if __name__ == "__main__":
    print(f"Total slides loaded: {len(slides_data)}")
    all_ok = True
    for s in slides_data:
        char_count = sum(len(p) for p in s['points'])
        point_count = len(s['points'])
        is_ok = (100 <= char_count <= 230) and (point_count <= 4)
        print(f"Slide {s['id']:02d}: {point_count} pts, {char_count} chars -> Gate 2 Check: {is_ok}")
        if not is_ok:
            all_ok = False
    print("Gate 2 Overall Verification Passed:", all_ok)
