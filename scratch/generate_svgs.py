import sys
sys.path.append('C:/Users/User/Desktop/llm-wiki-project/scratch')
from verify_slides import slides_data

svg_dict = {
    1: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <defs>
    <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00f2fe" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.2"/>
    </linearGradient>
    <radialGradient id="r1" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#0b0f19" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <circle cx="250" cy="160" r="140" fill="url(#r1)"/>
  <circle cx="250" cy="160" r="110" fill="none" stroke="#00f2fe" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.6"/>
  <circle cx="250" cy="160" r="70" fill="none" stroke="#60a5fa" stroke-width="1.5" opacity="0.8"/>
  <circle cx="250" cy="160" r="8" fill="#00f2fe"/>
  <!-- Orbiting nodes -->
  <g>
    <circle cx="140" cy="160" r="16" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
    <text x="140" y="164" fill="#e2e8f0" font-size="9" text-anchor="middle" font-family="sans-serif">Data</text>
    
    <circle cx="360" cy="160" r="16" fill="#1e293b" stroke="#818cf8" stroke-width="2"/>
    <text x="360" y="164" fill="#e2e8f0" font-size="9" text-anchor="middle" font-family="sans-serif">Model</text>
    
    <circle cx="250" cy="50" r="16" fill="#1e293b" stroke="#34d399" stroke-width="2"/>
    <text x="250" y="54" fill="#e2e8f0" font-size="9" text-anchor="middle" font-family="sans-serif">Loss</text>

    <circle cx="250" cy="270" r="16" fill="#1e293b" stroke="#f472b6" stroke-width="2"/>
    <text x="250" y="274" fill="#e2e8f0" font-size="9" text-anchor="middle" font-family="sans-serif">Opt</text>
  </g>
  <path d="M156 160 Q 200 90 234 56" fill="none" stroke="#38bdf8" stroke-width="2" stroke-dasharray="3 3"/>
  <path d="M266 56 Q 300 90 344 160" fill="none" stroke="#818cf8" stroke-width="2" stroke-dasharray="3 3"/>
  <path d="M344 160 Q 300 230 266 264" fill="none" stroke="#f472b6" stroke-width="2" stroke-dasharray="3 3"/>
  <path d="M234 264 Q 200 230 156 160" fill="none" stroke="#34d399" stroke-width="2" stroke-dasharray="3 3"/>
  <text x="250" y="195" fill="#94a3b8" font-size="11" text-anchor="middle" font-family="sans-serif">幾何表徵與流形導航</text>
</svg>''',

    2: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Traditional vs ML comparison boxes -->
  <g transform="translate(30, 30)">
    <rect width="200" height="260" rx="10" fill="#111827" stroke="#374151" stroke-width="1.5"/>
    <text x="100" y="32" fill="#9ca3af" font-size="14" font-weight="bold" text-anchor="middle">傳統程式設計</text>
    <rect x="25" y="55" width="150" height="40" rx="6" fill="#1f2937" stroke="#4b5563"/>
    <text x="100" y="80" fill="#e5e7eb" font-size="12" text-anchor="middle">人類規則 (Rules)</text>
    <text x="100" y="115" fill="#6b7280" font-size="16" text-anchor="middle">+</text>
    <rect x="25" y="130" width="150" height="40" rx="6" fill="#1f2937" stroke="#4b5563"/>
    <text x="100" y="155" fill="#e5e7eb" font-size="12" text-anchor="middle">輸入資料 (Data)</text>
    <path d="M100 180 L100 205" stroke="#9ca3af" stroke-width="2" marker-end="url(#arr)"/>
    <rect x="25" y="215" width="150" height="36" rx="6" fill="#374151"/>
    <text x="100" y="238" fill="#f87171" font-size="12" font-weight="bold" text-anchor="middle">答案 (Answers)</text>
  </g>
  <g transform="translate(270, 30)">
    <rect width="200" height="260" rx="10" fill="#0f172a" stroke="#00f2fe" stroke-width="1.5"/>
    <text x="100" y="32" fill="#38bdf8" font-size="14" font-weight="bold" text-anchor="middle">機器學習 (ML)</text>
    <rect x="25" y="55" width="150" height="40" rx="6" fill="#1e293b" stroke="#0284c7"/>
    <text x="100" y="80" fill="#e0f2fe" font-size="12" text-anchor="middle">輸入資料 (Data)</text>
    <text x="100" y="115" fill="#38bdf8" font-size="16" text-anchor="middle">+</text>
    <rect x="25" y="130" width="150" height="40" rx="6" fill="#1e293b" stroke="#0284c7"/>
    <text x="100" y="155" fill="#e0f2fe" font-size="12" text-anchor="middle">標準答案 (Answers)</text>
    <path d="M100 180 L100 205" stroke="#38bdf8" stroke-width="2"/>
    <rect x="25" y="215" width="150" height="36" rx="6" fill="#0369a1"/>
    <text x="100" y="238" fill="#34d399" font-size="12" font-weight="bold" text-anchor="middle">學得規則 (Rules/Model)</text>
  </g>
</svg>''',

    3: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Polynomial curve fitting illustration -->
  <line x1="40" y1="280" x2="460" y2="280" stroke="#4b5563" stroke-width="1.5"/>
  <line x1="40" y1="40" x2="40" y2="280" stroke="#4b5563" stroke-width="1.5"/>
  <text x="450" y="270" fill="#9ca3af" font-size="11">x</text>
  <text x="50" y="55" fill="#9ca3af" font-size="11">t</text>
  <!-- True sine wave (green dashed) -->
  <path d="M 50 160 Q 150 40 250 160 T 450 160" fill="none" stroke="#10b981" stroke-width="2" stroke-dasharray="4 4"/>
  <!-- M=1 straight line (red) -->
  <line x1="50" y1="240" x2="450" y2="80" stroke="#f87171" stroke-width="2"/>
  <!-- M=3 fitted curve (blue solid) -->
  <path d="M 50 165 C 130 50 180 90 250 160 C 320 230 370 270 450 155" fill="none" stroke="#38bdf8" stroke-width="2.5"/>
  <!-- Sample noisy dots -->
  <circle cx="70" cy="180" r="4" fill="#fbbf24"/>
  <circle cx="120" cy="100" r="4" fill="#fbbf24"/>
  <circle cx="170" cy="85" r="4" fill="#fbbf24"/>
  <circle cx="230" cy="140" r="4" fill="#fbbf24"/>
  <circle cx="280" cy="200" r="4" fill="#fbbf24"/>
  <circle cx="340" cy="235" r="4" fill="#fbbf24"/>
  <circle cx="400" cy="210" r="4" fill="#fbbf24"/>
  <circle cx="435" cy="150" r="4" fill="#fbbf24"/>
  <!-- Legend -->
  <g transform="translate(60, 45)">
    <line x1="0" y1="0" x2="25" y2="0" stroke="#10b981" stroke-width="2" stroke-dasharray="3 3"/>
    <text x="32" y="4" fill="#9ca3af" font-size="10">真實母體 sin(2πx)</text>
    <line x1="160" y1="0" x2="185" y2="0" stroke="#f87171" stroke-width="2"/>
    <text x="192" y="4" fill="#9ca3af" font-size="10">欠擬合 M=1</text>
    <line x1="280" y1="0" x2="305" y2="0" stroke="#38bdf8" stroke-width="2.5"/>
    <text x="312" y="4" fill="#9ca3af" font-size="10">適度擬合 M=3</text>
  </g>
</svg>''',

    4: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Overfitting M=9 vs Generalization Gap -->
  <!-- Left: Wild Oscillations -->
  <g transform="translate(10, 20)">
    <text x="110" y="20" fill="#f87171" font-size="12" font-weight="bold" text-anchor="middle">M=9: 狂暴震盪的過度擬合</text>
    <rect x="10" y="35" width="210" height="230" fill="#111827" rx="6" stroke="#374151"/>
    <!-- Wild M=9 curve -->
    <path d="M 20 150 Q 35 250 50 170 T 80 50 T 110 250 T 140 45 T 170 240 T 205 160" fill="none" stroke="#ef4444" stroke-width="2.5"/>
    <!-- Samples -->
    <circle cx="35" cy="180" r="4" fill="#fbbf24"/>
    <circle cx="65" cy="100" r="4" fill="#fbbf24"/>
    <circle cx="95" cy="80" r="4" fill="#fbbf24"/>
    <circle cx="125" cy="150" r="4" fill="#fbbf24"/>
    <circle cx="155" cy="220" r="4" fill="#fbbf24"/>
    <circle cx="185" cy="190" r="4" fill="#fbbf24"/>
    <text x="115" y="250" fill="#9ca3af" font-size="10" text-anchor="middle">訓練點全過，相鄰處暴衝</text>
  </g>
  <!-- Right: Generalization Gap (U-curve) -->
  <g transform="translate(250, 20)">
    <text x="115" y="20" fill="#38bdf8" font-size="12" font-weight="bold" text-anchor="middle">均方根誤差 ERMS 泛化鴻溝</text>
    <rect x="10" y="35" width="225" height="230" fill="#111827" rx="6" stroke="#374151"/>
    <!-- Axes -->
    <line x1="30" y1="230" x2="215" y2="230" stroke="#6b7280" stroke-width="1.2"/>
    <line x1="30" y1="55" x2="30" y2="230" stroke="#6b7280" stroke-width="1.2"/>
    <text x="210" y="245" fill="#9ca3af" font-size="10">M</text>
    <text x="20" y="50" fill="#9ca3af" font-size="10">E</text>
    <!-- Train error (blue, descending to 0) -->
    <path d="M 40 180 Q 80 130 110 110 T 160 80 T 200 45" fill="none" stroke="#38bdf8" stroke-width="2"/>
    <text x="160" y="105" fill="#38bdf8" font-size="9">訓練誤差 (0)</text>
    <!-- Test error (red, drops then explodes) -->
    <path d="M 40 190 Q 75 140 100 120 T 130 115 Q 160 140 195 55" fill="none" stroke="#f87171" stroke-width="2.5"/>
    <text x="160" y="65" fill="#f87171" font-size="10" font-weight="bold">測試誤差飆升！</text>
    <!-- Optimal sweet spot mark -->
    <line x1="110" y1="55" x2="110" y2="230" stroke="#34d399" stroke-width="1.2" stroke-dasharray="3 3"/>
    <text x="110" y="250" fill="#34d399" font-size="10" text-anchor="middle">最佳 M=3</text>
  </g>
</svg>''',

    5: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Bias-Variance Tradeoff Balance -->
  <g transform="translate(50, 40)">
    <rect width="400" height="240" rx="10" fill="#111827" stroke="#374151"/>
    <!-- U-shaped Curves -->
    <path d="M 40 70 Q 140 190 360 210" fill="none" stroke="#60a5fa" stroke-width="2"/>
    <text x="320" y="225" fill="#60a5fa" font-size="10">偏差平方 (Bias²)</text>
    <path d="M 40 210 Q 240 190 360 70" fill="none" stroke="#f87171" stroke-width="2"/>
    <text x="320" y="65" fill="#f87171" font-size="10">變異 (Variance)</text>
    <!-- Total test error (green U) -->
    <path d="M 40 190 Q 180 80 360 190" fill="none" stroke="#34d399" stroke-width="2.5"/>
    <text x="180" y="70" fill="#34d399" font-size="11" font-weight="bold" text-anchor="middle">總泛化誤差 (Total Error)</text>
    <!-- Optimal line -->
    <line x1="185" y1="75" x2="185" y2="220" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="4 3"/>
    <text x="185" y="235" fill="#fbbf24" font-size="11" text-anchor="middle">正則化最佳平衡點</text>
  </g>
</svg>''',

    6: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Linear Decision Boundary & Normal Vector -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <!-- Decision line w^T x + w0 = 0 -->
  <line x1="80" y1="260" x2="380" y2="50" stroke="#38bdf8" stroke-width="2.5"/>
  <text x="385" y="45" fill="#38bdf8" font-size="12" font-weight="bold">wᵀx + w₀ = 0</text>
  <!-- Normal vector w -->
  <line x1="230" y1="155" x2="290" y2="70" stroke="#f43f5e" stroke-width="2.5"/>
  <polygon points="295,65 282,72 290,83" fill="#f43f5e"/>
  <text x="305" y="75" fill="#f43f5e" font-size="12" font-weight="bold">w (法向量)</text>
  <!-- Data dots: Class 1 (blue circles) -->
  <circle cx="340" cy="130" r="5" fill="#38bdf8"/>
  <circle cx="380" cy="160" r="5" fill="#38bdf8"/>
  <circle cx="310" cy="90" r="5" fill="#38bdf8"/>
  <circle cx="420" cy="110" r="5" fill="#38bdf8"/>
  <text x="380" y="195" fill="#38bdf8" font-size="11">類別 C₁ (y > 0)</text>
  <!-- Data dots: Class 2 (orange squares) -->
  <rect x="120" y="160" width="10" height="10" fill="#fb923c"/>
  <rect x="160" y="200" width="10" height="10" fill="#fb923c"/>
  <rect x="100" y="220" width="10" height="10" fill="#fb923c"/>
  <rect x="200" y="240" width="10" height="10" fill="#fb923c"/>
  <text x="120" y="145" fill="#fb923c" font-size="11">類別 C₂ (y < 0)</text>
  <!-- Orthogonal distance illustration -->
  <line x1="340" y1="130" x2="300" y2="105" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="3 3"/>
  <text x="330" y="110" fill="#94a3b8" font-size="10">距離 r</text>
</svg>''',

    7: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Sigmoid curve with saturation zones -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <line x1="60" y1="160" x2="430" y2="160" stroke="#4b5563" stroke-width="1.2"/>
  <line x1="245" y1="50" x2="245" y2="260" stroke="#4b5563" stroke-width="1.2"/>
  <text x="435" y="164" fill="#9ca3af" font-size="11">a</text>
  <text x="250" y="45" fill="#9ca3af" font-size="11">σ(a)</text>
  <!-- 0 and 1 limits -->
  <line x1="60" y1="60" x2="430" y2="60" stroke="#374151" stroke-dasharray="2 2"/>
  <text x="70" y="55" fill="#9ca3af" font-size="10">y = 1.0</text>
  <text x="70" y="255" fill="#9ca3af" font-size="10">y = 0.0</text>
  <!-- Sigmoid curve -->
  <path d="M 60 255 C 160 255 190 230 245 160 C 300 90 330 65 430 65" fill="none" stroke="#00f2fe" stroke-width="3"/>
  <!-- Saturation annotations -->
  <rect x="70" y="190" width="100" height="40" rx="4" fill="#1e293b" opacity="0.8"/>
  <text x="120" y="215" fill="#f87171" font-size="10" text-anchor="middle">飽和區 σ'≈0</text>
  <rect x="320" y="90" width="100" height="40" rx="4" fill="#1e293b" opacity="0.8"/>
  <text x="370" y="115" fill="#f87171" font-size="10" text-anchor="middle">飽和區 σ'≈0</text>
  <!-- Center point -->
  <circle cx="245" cy="160" r="5" fill="#fbbf24"/>
  <text x="260" y="175" fill="#fbbf24" font-size="10">(0, 0.5)</text>
</svg>''',

    8: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Cross-Entropy vs MSE Gradient Saturation -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <!-- Target t=1. X axis is y from 0 to 1 -->
  <line x1="70" y1="240" x2="420" y2="240" stroke="#6b7280" stroke-width="1.2"/>
  <line x1="70" y1="50" x2="70" y2="240" stroke="#6b7280" stroke-width="1.2"/>
  <text x="425" y="244" fill="#9ca3af" font-size="11">預測輸出 y (目標 t=1)</text>
  <text x="75" y="45" fill="#9ca3af" font-size="11">梯度推動力 |∂E/∂a|</text>
  <!-- Cross-entropy gradient: y - t. When y->0, error is 1. (blue linear) -->
  <line x1="70" y1="60" x2="400" y2="240" stroke="#00f2fe" stroke-width="3"/>
  <text x="210" y="110" fill="#00f2fe" font-size="12" font-weight="bold">交叉熵梯度 |y - 1|</text>
  <text x="210" y="128" fill="#94a3b8" font-size="10">預測越錯，推動力越強！</text>
  <!-- MSE gradient: (y-1)*y*(1-y). When y->0, gradient is 0! (red hump) -->
  <path d="M 70 240 Q 220 80 400 240" fill="none" stroke="#f87171" stroke-width="2.5" stroke-dasharray="4 3"/>
  <text x="290" y="190" fill="#f87171" font-size="11" font-weight="bold">MSE 梯度 (y-1)y(1-y)</text>
  <text x="80" y="225" fill="#ef4444" font-size="10">嚴重錯誤時梯度歸零！(假死)</text>
</svg>''',

    9: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Softmax Probability Bars -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <text x="140" y="60" fill="#9ca3af" font-size="12" text-anchor="middle">未歸一化得分 (Logits a)</text>
  <text x="350" y="60" fill="#38bdf8" font-size="12" text-anchor="middle" font-weight="bold">Softmax 機率 (y = eᵃ/Σeᵃ)</text>
  <!-- Bars Class 1 -->
  <text x="60" y="105" fill="#e2e8f0" font-size="11">貓 (Cat)</text>
  <rect x="110" y="90" width="60" height="22" fill="#475569" rx="4"/>
  <text x="140" y="105" fill="#cbd5e1" font-size="10" text-anchor="middle">1.2</text>
  <rect x="250" y="90" width="35" height="22" fill="#0284c7" rx="4"/>
  <text x="300" y="105" fill="#38bdf8" font-size="11">11%</text>
  <!-- Bars Class 2 (Winner) -->
  <text x="60" y="160" fill="#e2e8f0" font-size="11">狗 (Dog)</text>
  <rect x="110" y="145" width="110" height="22" fill="#475569" rx="4"/>
  <text x="165" y="160" fill="#cbd5e1" font-size="10" text-anchor="middle">3.4</text>
  <rect x="250" y="145" width="165" height="22" fill="#00f2fe" rx="4"/>
  <text x="430" y="160" fill="#00f2fe" font-size="11" font-weight="bold">82% (勝出)</text>
  <!-- Bars Class 3 -->
  <text x="60" y="215" fill="#e2e8f0" font-size="11">鳥 (Bird)</text>
  <rect x="110" y="200" width="40" height="22" fill="#475569" rx="4"/>
  <text x="130" y="215" fill="#cbd5e1" font-size="10" text-anchor="middle">0.5</text>
  <rect x="250" y="200" width="22" height="22" fill="#0284c7" rx="4"/>
  <text x="285" y="215" fill="#38bdf8" font-size="11">7%</text>
  <!-- Bottom note -->
  <line x1="250" y1="245" x2="430" y2="245" stroke="#38bdf8" stroke-width="1.2"/>
  <text x="340" y="265" fill="#9ca3af" font-size="10" text-anchor="middle">總機率和嚴格等於 100% (互斥決策)</text>
</svg>''',

    10: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- XOR Problem & Curse of Dimensionality -->
  <g transform="translate(30, 30)">
    <rect width="200" height="260" rx="8" fill="#111827" stroke="#374151"/>
    <text x="100" y="25" fill="#f87171" font-size="12" font-weight="bold" text-anchor="middle">XOR 線性不可分困局</text>
    <line x1="30" y1="220" x2="170" y2="220" stroke="#6b7280" stroke-width="1.2"/>
    <line x1="30" y1="80" x2="30" y2="220" stroke="#6b7280" stroke-width="1.2"/>
    <!-- 4 points -->
    <circle cx="50" cy="200" r="7" fill="#38bdf8"/><text x="40" y="225" fill="#9ca3af" font-size="9">(0,0):0</text>
    <rect x="143" y="193" width="14" height="14" fill="#fb923c"/><text x="140" y="225" fill="#9ca3af" font-size="9">(1,0):1</text>
    <rect x="43" y="93" width="14" height="14" fill="#fb923c"/><text x="25" y="95" fill="#9ca3af" font-size="9">(0,1):1</text>
    <circle cx="150" cy="100" r="7" fill="#38bdf8"/><text x="155" y="95" fill="#9ca3af" font-size="9">(1,1):0</text>
    <!-- Failed split lines -->
    <line x1="20" y1="130" x2="180" y2="170" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3 3"/>
    <text x="100" y="248" fill="#f87171" font-size="10" text-anchor="middle">任何單一線都無法正確切分</text>
  </g>
  <g transform="translate(260, 30)">
    <rect width="210" height="260" rx="8" fill="#111827" stroke="#374151"/>
    <text x="105" y="25" fill="#38bdf8" font-size="12" font-weight="bold" text-anchor="middle">維度災難：表面薄殼集中</text>
    <circle cx="105" cy="135" r="75" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
    <circle cx="105" cy="135" r="65" fill="#0f172a"/>
    <path d="M 105 60 A 75 75 0 0 1 180 135 L 170 135 A 65 65 0 0 0 105 70 Z" fill="#00f2fe" opacity="0.6"/>
    <text x="105" y="138" fill="#9ca3af" font-size="10" text-anchor="middle">內部核心幾無體積</text>
    <text x="105" y="235" fill="#38bdf8" font-size="10" text-anchor="middle">高維空間中體積 100% 集中於薄殼</text>
    <text x="105" y="250" fill="#9ca3af" font-size="9" text-anchor="middle">常規網格需 10ᴰ 個節點 (爆炸)</text>
  </g>
</svg>''',

    11: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Multilayer Space Folding Mechanism -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <!-- Input nodes -->
  <g transform="translate(70, 80)">
    <circle cx="0" cy="40" r="14" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
    <text x="0" y="44" fill="#e2e8f0" font-size="10" text-anchor="middle">x₁</text>
    <circle cx="0" cy="120" r="14" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
    <text x="0" y="124" fill="#e2e8f0" font-size="10" text-anchor="middle">x₂</text>
    <text x="0" y="170" fill="#94a3b8" font-size="11" text-anchor="middle">輸入空間</text>
  </g>
  <!-- Hidden nodes with activation curves -->
  <g transform="translate(220, 50)">
    <circle cx="0" cy="30" r="16" fill="#1e293b" stroke="#34d399" stroke-width="2"/>
    <text x="0" y="34" fill="#e2e8f0" font-size="10" text-anchor="middle">z₁</text>
    <circle cx="0" cy="90" r="16" fill="#1e293b" stroke="#34d399" stroke-width="2"/>
    <text x="0" y="94" fill="#e2e8f0" font-size="10" text-anchor="middle">z₂</text>
    <circle cx="0" cy="150" r="16" fill="#1e293b" stroke="#34d399" stroke-width="2"/>
    <text x="0" y="154" fill="#e2e8f0" font-size="10" text-anchor="middle">z₃</text>
    <text x="0" y="200" fill="#34d399" font-size="11" text-anchor="middle">非線性隱藏表徵 (空間折疊)</text>
  </g>
  <!-- Output node -->
  <g transform="translate(370, 110)">
    <circle cx="0" cy="50" r="16" fill="#1e293b" stroke="#f472b6" stroke-width="2"/>
    <text x="0" y="54" fill="#e2e8f0" font-size="10" text-anchor="middle">y</text>
    <text x="0" y="90" fill="#f472b6" font-size="11" text-anchor="middle">線性可分輸出</text>
  </g>
  <!-- Connecting lines -->
  <line x1="84" y1="120" x2="204" y2="80" stroke="#475569" stroke-width="1.2"/>
  <line x1="84" y1="120" x2="204" y2="140" stroke="#475569" stroke-width="1.2"/>
  <line x1="84" y1="200" x2="204" y2="140" stroke="#475569" stroke-width="1.2"/>
  <line x1="84" y1="200" x2="204" y2="200" stroke="#475569" stroke-width="1.2"/>
  <line x1="236" y1="80" x2="354" y2="160" stroke="#34d399" stroke-width="1.5"/>
  <line x1="236" y1="140" x2="354" y2="160" stroke="#34d399" stroke-width="1.5"/>
  <line x1="236" y1="200" x2="354" y2="160" stroke="#34d399" stroke-width="1.5"/>
</svg>''',

    12: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Activation Functions comparison: Sigmoid vs ReLU -->
  <g transform="translate(30, 30)">
    <rect width="200" height="260" rx="8" fill="#111827" stroke="#374151"/>
    <text x="100" y="25" fill="#f87171" font-size="12" font-weight="bold" text-anchor="middle">Sigmoid: 梯度消失</text>
    <line x1="20" y1="160" x2="180" y2="160" stroke="#4b5563" stroke-width="1"/>
    <line x1="100" y1="60" x2="100" y2="240" stroke="#4b5563" stroke-width="1"/>
    <!-- Sigmoid curve -->
    <path d="M 25 220 C 65 220 80 200 100 160 C 120 120 135 100 175 100" fill="none" stroke="#f87171" stroke-width="2.5"/>
    <text x="100" y="250" fill="#9ca3af" font-size="10" text-anchor="middle">最大導函數僅 0.25 (連乘速歸零)</text>
  </g>
  <g transform="translate(260, 30)">
    <rect width="210" height="260" rx="8" fill="#111827" stroke="#00f2fe" stroke-width="1.5"/>
    <text x="105" y="25" fill="#00f2fe" font-size="12" font-weight="bold" text-anchor="middle">ReLU: 深度學習的解藥</text>
    <line x1="20" y1="180" x2="190" y2="180" stroke="#4b5563" stroke-width="1"/>
    <line x1="105" y1="60" x2="105" y2="240" stroke="#4b5563" stroke-width="1"/>
    <!-- ReLU curve: 0 for x<0, linear for x>0 -->
    <line x1="20" y1="180" x2="105" y2="180" stroke="#00f2fe" stroke-width="3"/>
    <line x1="105" y1="180" x2="185" y2="90" stroke="#00f2fe" stroke-width="3"/>
    <circle cx="105" cy="180" r="4" fill="#fbbf24"/>
    <text x="105" y="250" fill="#38bdf8" font-size="10" text-anchor="middle">正半軸導數恆為 1 (百層不衰減)</text>
  </g>
</svg>''',

    13: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Feature Pyramid: Pixels -> Edges -> Parts -> Objects -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <!-- Level 1 (Bottom): Pixels -->
  <rect x="70" y="220" width="360" height="40" rx="6" fill="#1e293b" stroke="#475569"/>
  <text x="85" y="245" fill="#94a3b8" font-size="11" font-weight="bold">淺層 (Layer 1-2)</text>
  <text x="260" y="245" fill="#e2e8f0" font-size="11">低階物理特徵：邊緣 (Edges)、斑點、紋理方向</text>
  <!-- Level 2: Parts -->
  <rect x="100" y="150" width="300" height="40" rx="6" fill="#1e293b" stroke="#0284c7"/>
  <text x="115" y="175" fill="#38bdf8" font-size="11" font-weight="bold">中層 (Layer 3-5)</text>
  <text x="260" y="175" fill="#e0f2fe" font-size="11">局部幾何部件：眼睛輪廓、輪胎花紋、拐角</text>
  <!-- Level 3: Semantic Concepts -->
  <rect x="140" y="80" width="220" height="40" rx="6" fill="#0f172a" stroke="#00f2fe" stroke-width="2"/>
  <text x="155" y="105" fill="#00f2fe" font-size="11" font-weight="bold">深層 (Deep)</text>
  <text x="260" y="105" fill="#00f2fe" font-size="11">高階語意：特定狗品種、人臉、汽車概念</text>
  <!-- Upward Arrows -->
  <path d="M 250 215 L 250 195" stroke="#38bdf8" stroke-width="2"/>
  <path d="M 250 145 L 250 125" stroke="#00f2fe" stroke-width="2"/>
  <text x="250" y="55" fill="#fbbf24" font-size="11" text-anchor="middle" font-weight="bold">逐層抽象化 · 表徵可直接遷移微調</text>
</svg>''',

    14: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Saddle point & Ravine contour -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <!-- Narrow Elliptical Contours of a Ravine -->
  <ellipse cx="250" cy="160" rx="170" ry="40" fill="none" stroke="#334155" stroke-width="1.5"/>
  <ellipse cx="250" cy="160" rx="120" ry="25" fill="none" stroke="#475569" stroke-width="1.5"/>
  <ellipse cx="250" cy="160" rx="60" ry="12" fill="none" stroke="#0284c7" stroke-width="1.5"/>
  <circle cx="250" cy="160" r="5" fill="#34d399"/>
  <text x="250" y="185" fill="#34d399" font-size="10" text-anchor="middle">極值點 w*</text>
  <!-- Curvature directions -->
  <line x1="250" y1="80" x2="250" y2="240" stroke="#f43f5e" stroke-width="2"/>
  <text x="260" y="95" fill="#f43f5e" font-size="10">λₘₐₓ (陡峭壁：易橫向震盪)</text>
  <line x1="60" y1="160" x2="440" y2="160" stroke="#38bdf8" stroke-width="2"/>
  <text x="390" y="150" fill="#38bdf8" font-size="10">λₘᵢₙ (平緩谷底)</text>
  <text x="250" y="55" fill="#fbbf24" font-size="12" font-weight="bold" text-anchor="middle">病態條件數 κ = λₘₐₓ / λₘᵢₙ ≫ 1</text>
</svg>''',

    15: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- BGD vs SGD vs Mini-batch paths -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <ellipse cx="360" cy="160" rx="70" ry="50" fill="none" stroke="#334155"/>
  <ellipse cx="360" cy="160" rx="35" ry="25" fill="none" stroke="#475569"/>
  <circle cx="360" cy="160" r="5" fill="#fbbf24"/>
  <!-- BGD Path (Smooth blue line) -->
  <path d="M 70 80 Q 200 90 355 158" fill="none" stroke="#38bdf8" stroke-width="2.5"/>
  <text x="130" y="75" fill="#38bdf8" font-size="10">BGD: 步調平滑但計算巨重</text>
  <!-- SGD Path (Wild jagged orange line) -->
  <path d="M 70 230 L 110 180 L 90 140 L 160 200 L 190 110 L 250 190 L 300 130 L 330 180 L 358 162" fill="none" stroke="#f87171" stroke-width="1.8"/>
  <text x="150" y="240" fill="#f87171" font-size="10">SGD: 單點更新劇烈隨機擾動</text>
  <!-- Mini-batch Path (Balanced green line) -->
  <path d="M 70 150 L 130 135 L 180 160 L 240 145 L 300 155 L 357 160" fill="none" stroke="#34d399" stroke-width="2.5"/>
  <text x="180" y="175" fill="#34d399" font-size="11" font-weight="bold">Mini-batch: 穩健前進 + 硬體吞吐王</text>
</svg>''',

    16: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Momentum smoothing zig-zag -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <!-- Contours -->
  <path d="M 60 70 Q 250 110 440 70" fill="none" stroke="#334155"/>
  <path d="M 60 160 Q 250 160 440 160" fill="none" stroke="#0284c7" stroke-width="1.5"/>
  <path d="M 60 250 Q 250 210 440 250" fill="none" stroke="#334155"/>
  <!-- Standard SGD zig-zag (red) -->
  <path d="M 80 80 L 130 240 L 170 80 L 210 240 L 250 85 L 290 235" fill="none" stroke="#f87171" stroke-width="2" stroke-dasharray="3 3"/>
  <text x="170" y="60" fill="#f87171" font-size="11">無動量：之字形瘋狂橫跳</text>
  <!-- Momentum smooth curve (cyan) -->
  <path d="M 80 120 C 130 180 180 140 250 160 T 400 160" fill="none" stroke="#00f2fe" stroke-width="3"/>
  <text x="320" y="145" fill="#00f2fe" font-size="12" font-weight="bold">動量法：慣性滑行加速！</text>
  <text x="250" y="280" fill="#9ca3af" font-size="10" text-anchor="middle">橫向震盪正負抵消，縱向步長放大為 1/(1-μ) 倍 (如 10x)</text>
</svg>''',

    17: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Evolution Tree: SGD -> Momentum/AdaGrad -> RMSProp -> Adam -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <!-- SGD root -->
  <rect x="60" y="130" width="70" height="35" rx="6" fill="#1e293b" stroke="#64748b"/>
  <text x="95" y="152" fill="#e2e8f0" font-size="11" text-anchor="middle">SGD</text>
  <!-- Momentum branch -->
  <rect x="180" y="70" width="90" height="35" rx="6" fill="#1e293b" stroke="#38bdf8"/>
  <text x="225" y="92" fill="#38bdf8" font-size="11" text-anchor="middle">動量法 (1階矩)</text>
  <!-- AdaGrad branch -->
  <rect x="180" y="190" width="90" height="35" rx="6" fill="#1e293b" stroke="#f59e0b"/>
  <text x="225" y="212" fill="#f59e0b" font-size="11" text-anchor="middle">AdaGrad (2階)</text>
  <!-- RMSProp -->
  <rect x="300" y="190" width="80" height="35" rx="6" fill="#1e293b" stroke="#ec4899"/>
  <text x="340" y="212" fill="#ec4899" font-size="11" text-anchor="middle">RMSProp</text>
  <!-- Adam Crown -->
  <rect x="370" y="125" width="80" height="45" rx="8" fill="#0f172a" stroke="#00f2fe" stroke-width="2"/>
  <text x="410" y="152" fill="#00f2fe" font-size="13" font-weight="bold" text-anchor="middle">Adam</text>
  <!-- Connecting lines -->
  <line x1="130" y1="140" x2="180" y2="90" stroke="#64748b" stroke-width="1.5"/>
  <line x1="130" y1="155" x2="180" y2="205" stroke="#64748b" stroke-width="1.5"/>
  <line x1="270" y1="205" x2="300" y2="205" stroke="#f59e0b" stroke-width="1.5"/>
  <line x1="270" y1="90" x2="370" y2="140" stroke="#38bdf8" stroke-width="2"/>
  <line x1="380" y1="200" x2="400" y2="170" stroke="#ec4899" stroke-width="2"/>
  <text x="250" y="280" fill="#9ca3af" font-size="10" text-anchor="middle">結合「一階慣性」與「二階自適應縮放」+ 偏差修正</text>
</svg>''',

    18: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- BatchNorm vs LayerNorm Tensor Slicing -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <!-- BatchNorm (Vertical Slice) -->
  <g transform="translate(60, 50)">
    <rect width="150" height="180" rx="6" fill="#1e293b" stroke="#38bdf8"/>
    <text x="75" y="25" fill="#38bdf8" font-size="12" font-weight="bold" text-anchor="middle">Batch Normalization</text>
    <rect x="50" y="45" width="25" height="110" fill="#0284c7" opacity="0.8"/>
    <text x="62" y="170" fill="#93c5fd" font-size="9" text-anchor="middle">跨 Batch 聚合</text>
    <text x="75" y="200" fill="#9ca3af" font-size="10" text-anchor="middle">依賴批次大小 B (CNN 首選)</text>
  </g>
  <!-- LayerNorm (Horizontal Slice) -->
  <g transform="translate(250, 50)">
    <rect width="180" height="180" rx="6" fill="#1e293b" stroke="#00f2fe" stroke-width="2"/>
    <text x="90" y="25" fill="#00f2fe" font-size="12" font-weight="bold" text-anchor="middle">Layer Normalization</text>
    <rect x="25" y="75" width="130" height="25" fill="#00f2fe" opacity="0.6"/>
    <text x="90" y="170" fill="#67e8f9" font-size="9" text-anchor="middle">跨 Feature 聚合</text>
    <text x="90" y="200" fill="#9ca3af" font-size="10" text-anchor="middle">脫鉤 Batch (LLM / Transformer)</text>
  </g>
</svg>''',

    19: '''<svg viewBox="0 0 500 320" class="w-full h-full">
  <!-- Big Picture Roadmap -->
  <rect x="40" y="30" width="420" height="260" rx="8" fill="#111827" stroke="#374151"/>
  <!-- Milestone Cards -->
  <g transform="translate(60, 60)">
    <rect width="80" height="50" rx="6" fill="#1e293b" stroke="#64748b"/>
    <text x="40" y="25" fill="#94a3b8" font-size="10" text-anchor="middle">單層線性模型</text>
    <text x="40" y="40" fill="#cbd5e1" font-size="9" text-anchor="middle">正規方程 / OLS</text>
  </g>
  <line x1="140" y1="85" x2="170" y2="85" stroke="#64748b" stroke-width="2"/>
  <g transform="translate(170, 60)">
    <rect width="80" height="50" rx="6" fill="#1e293b" stroke="#38bdf8"/>
    <text x="40" y="25" fill="#38bdf8" font-size="10" text-anchor="middle">機率與邊界</text>
    <text x="40" y="40" fill="#cbd5e1" font-size="9" text-anchor="middle">Sigmoid / Softmax</text>
  </g>
  <line x1="250" y1="85" x2="280" y2="85" stroke="#38bdf8" stroke-width="2"/>
  <g transform="translate(280, 60)">
    <rect width="80" height="50" rx="6" fill="#1e293b" stroke="#a855f7"/>
    <text x="40" y="25" fill="#a855f7" font-size="10" text-anchor="middle">多層折疊表徵</text>
    <text x="40" y="40" fill="#cbd5e1" font-size="9" text-anchor="middle">MLP / ReLU / 階層</text>
  </g>
  <path d="M 320 110 L 320 160 L 220 160" stroke="#a855f7" stroke-width="2" fill="none"/>
  <g transform="translate(140, 140)">
    <rect width="80" height="50" rx="6" fill="#1e293b" stroke="#34d399"/>
    <text x="40" y="25" fill="#34d399" font-size="10" text-anchor="middle">現代優化引擎</text>
    <text x="40" y="40" fill="#cbd5e1" font-size="9" text-anchor="middle">SGD / 動量 / Adam</text>
  </g>
  <line x1="220" y1="190" x2="250" y2="190" stroke="#34d399" stroke-width="2"/>
  <g transform="translate(250, 165)">
    <rect width="140" height="55" rx="8" fill="#0f172a" stroke="#00f2fe" stroke-width="2"/>
    <text x="70" y="28" fill="#00f2fe" font-size="11" font-weight="bold" text-anchor="middle">大型語言模型與基礎模型</text>
    <text x="70" y="44" fill="#38bdf8" font-size="9" text-anchor="middle">Transformer · 自監督 · 泛化</text>
  </g>
  <text x="250" y="260" fill="#9ca3af" font-size="11" text-anchor="middle">一場由資料驅動、數學約束與高維最佳化共同譜寫的智慧革命</text>
</svg>'''
}

print("SVG count:", len(svg_dict))
