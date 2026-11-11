"""Regenerate every Ch1 figure as SVG (text kept as <text> so HTML slides can restyle it).

Run:  python assets/ch01/make_figures.py      (needs numpy + matplotlib)
Output: assets/ch01/*.svg
Conventions follow the lecture (lecture-notes Ch1, Figures 1-11): N=10 evenly spaced x,
t = sin(2*pi*x) + N(0, 0.3^2), seed 0.  Numerical values differ from the printed book/lecture.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))
from polyfit_core import design_poly, erms, fit_ls, fit_ridge, make_data  # noqa: E402

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
BLUE, RED, GREEN, GRAY, ORANGE = "#1f6feb", "#d1242f", "#1a7f37", "#6e7781", "#d4a72c"


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


rng = np.random.default_rng(0)
x, t = make_data(10, rng)
xt, tt = make_data(100, rng)
xx = np.linspace(0, 1, 400)
sin = np.sin(2 * np.pi * xx)


def base_axes(ax, title=None):
    ax.plot(xx, sin, color=GREEN, lw=1.5, label="sin(2πx)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    if title:
        ax.set_title(title)


# Fig 1 -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.4))
base_axes(ax, "圖 1：N = 10 筆合成訓練資料")
ax.scatter(x, t, facecolor="none", edgecolor=BLUE, s=45, lw=1.6, label="訓練資料")
ax.legend(frameon=False, loc="upper right")
save(fig, "fig01-synthetic-data.svg")

# Fig 3 -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
for ax, M in zip(axes.ravel(), (0, 1, 3, 9)):
    w = fit_ls(design_poly(x, M), t)
    base_axes(ax, f"M = {M}")
    ax.scatter(x, t, facecolor="none", edgecolor=BLUE, s=35, lw=1.4)
    ax.plot(xx, design_poly(xx, M) @ w, color=RED, lw=1.8)
    ax.set_ylim(-1.8, 1.8)
fig.suptitle("圖 3：不同階數 M 的多項式擬合（紅）、真實函數（綠）", y=1.0)
fig.tight_layout()
save(fig, "fig03-polyfit-m0-1-3-9.svg")

# Fig 4 -------------------------------------------------------------
Ms = np.arange(10)
tr = [erms(design_poly(x, M), t, fit_ls(design_poly(x, M), t)) for M in Ms]
te = [erms(design_poly(xt, M), tt, fit_ls(design_poly(x, M), t)) for M in Ms]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(Ms, tr, "o-", color=BLUE, label="訓練集 Training")
ax.plot(Ms, te, "o-", color=RED, label="測試集 Test")
ax.axhline(0.3, color=GRAY, ls=":", lw=1)
ax.text(9.0, 0.32, "雜訊 σ = 0.3", ha="right", color=GRAY, fontsize=9)
ax.set_xlabel("M")
ax.set_ylabel("E_RMS")
ax.set_title("圖 4：RMS 誤差隨 M 變化")
ax.legend(frameon=False)
save(fig, "fig04-rms-vs-m.svg")

# Fig 5 -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, N in zip(axes, (15, 100)):
    xn, tn = make_data(N, np.random.default_rng(1))
    w = fit_ls(design_poly(xn, 9), tn)
    base_axes(ax, f"N = {N}, M = 9")
    ax.scatter(xn, tn, facecolor="none", edgecolor=BLUE, s=22, lw=1.2)
    ax.plot(xx, design_poly(xx, 9) @ w, color=RED, lw=1.8)
    ax.set_ylim(-1.8, 1.8)
fig.suptitle("圖 5：資料量增加可減輕過度擬合", y=1.02)
fig.tight_layout()
save(fig, "fig05-overfit-vs-n.svg")

# Fig 6 -------------------------------------------------------------
P, Pt = design_poly(x, 9), design_poly(xt, 9)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, ll in zip(axes, (-18, 0)):
    w = fit_ridge(P, t, np.exp(ll))
    base_axes(ax, f"ln λ = {ll}")
    ax.scatter(x, t, facecolor="none", edgecolor=BLUE, s=30, lw=1.3)
    ax.plot(xx, design_poly(xx, 9) @ w, color=RED, lw=1.8)
    ax.set_ylim(-1.8, 1.8)
fig.suptitle("圖 6：M = 9 的正則化擬合", y=1.02)
fig.tight_layout()
save(fig, "fig06-regularized-fits.svg")

# Fig 7 -------------------------------------------------------------
lls = np.linspace(-35, 0, 200)
tr = [erms(P, t, fit_ridge(P, t, np.exp(l))) for l in lls]
te = [erms(Pt, tt, fit_ridge(P, t, np.exp(l))) for l in lls]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(lls, tr, color=BLUE, label="訓練集 Training")
ax.plot(lls, te, color=RED, label="測試集 Test")
ax.set_xlabel("ln λ")
ax.set_ylabel("E_RMS")
ax.set_title("圖 7：M = 9，RMS 誤差隨 ln λ 變化")
ax.legend(frameon=False)
save(fig, "fig07-rms-vs-ln-lambda.svg")

# Fig 8 -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 2.6))
S = 4
for run in range(S):
    for g in range(S):
        ax.add_patch(Rectangle((g * 1.5, -run * 1.0), 1.4, 0.8,
                               facecolor=RED if g == run else "#d0d7de", edgecolor="white"))
    ax.text(-0.15, -run * 1.0 + 0.4, f"run {run + 1}", ha="right", va="center", fontsize=10)
ax.set_xlim(-1.4, S * 1.5)
ax.set_ylim(-S + 0.5, 1.2)
ax.axis("off")
ax.set_title("圖 8：S = 4 折交叉驗證（紅：驗證；灰：訓練）", fontsize=11)
save(fig, "fig08-cross-validation.svg")

# Fig 9 -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.2))
ys = [1.0, 0.33, -0.33, -1.0]
labels = ["x1", "x2", "...", "xM"]
for yv, lab in zip(ys, labels):
    ax.text(-0.15, yv, lab, ha="right", va="center", fontsize=12)
    if lab != "...":
        ax.add_patch(Circle((0.2, yv), 0.1, facecolor="white", edgecolor=BLUE, lw=1.6))
        ax.annotate("", xy=(2.2, 0), xytext=(0.3, yv),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
ax.text(1.0, 0.95, "w1", color=GRAY)
ax.text(1.0, 0.25, "w2", color=GRAY)
ax.text(1.0, -0.85, "wM", color=GRAY)
ax.add_patch(Circle((2.4, 0), 0.22, facecolor="#ddf4ff", edgecolor=BLUE, lw=1.8))
ax.text(2.4, 0, "f", ha="center", va="center", fontsize=13, style="italic")
ax.annotate("", xy=(3.4, 0), xytext=(2.65, 0), arrowprops=dict(arrowstyle="->", lw=1.4))
ax.text(3.5, 0, "y", va="center", fontsize=13)
ax.text(2.4, 0.55, "a = Σ wi·xi", ha="center", fontsize=10)
ax.text(2.4, -0.6, "y = f(a)", ha="center", fontsize=10)
ax.set_xlim(-0.6, 3.9)
ax.set_ylim(-1.3, 1.4)
ax.axis("off")
ax.set_title("圖 9：單一神經元 a = Σ wi·xi，y = f(a)")
save(fig, "fig09-artificial-neuron.svg")

# Fig 11 (schematic, as in the lecture) -----------------------------
yrs1 = np.array([1958, 2012])
yrs2 = np.array([2012, 2020])
y1 = np.array([-14.0, -14.0 + 54 / 2 * np.log10(2)])
y2 = np.array([y1[1], y1[1] + (2020 - 2012) * 12 / 3.4 * np.log10(2)])
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(yrs1, y1, color=BLUE, lw=2, label="第一階段：約每 2 年翻倍")
ax.plot(yrs2, y2, color=RED, lw=2, label="第二階段：約每 3.4 個月翻倍")
for name, yr in [("Perceptron", 1958), ("LeNet-5", 1998), ("AlexNet", 2012),
                 ("AlphaGo Zero", 2018), ("GPT-3", 2020)]:
    yv = np.interp(yr, np.r_[yrs1, yrs2[1:]], np.r_[y1, y2[1:]])
    ax.scatter([yr], [yv], color="black", s=18, zorder=3)
    ax.text(yr - 1, yv + 0.8, name, fontsize=8, ha="right")
ax.set_xlabel("年份")
ax.set_ylabel("log10(訓練算力)  [示意]")
ax.set_title("圖 11：訓練算力成長（示意重繪，里程碑位置為概略）", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
save(fig, "fig11-compute-growth.svg")

# Learning paradigms -------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.4))
cols = [("監督式 Supervised", "輸入 x  +  人工標註 t", "皮膚病灶分類\n蛋白質結構預測", BLUE),
        ("非監督式 Unsupervised", "只有輸入 x，沒有標籤", "人臉影像合成\n（生成模型）", GREEN),
        ("自監督式 Self-supervised", "目標由輸入自動導出", "LLM：預測下一個詞", ORANGE)]
for i, (a, b, c, col) in enumerate(cols):
    x0 = i * 3.1
    ax.add_patch(FancyBboxPatch((x0, 0.3), 2.8, 2.6, boxstyle="round,pad=0.05",
                                facecolor="white", edgecolor=col, lw=2))
    ax.text(x0 + 1.4, 2.5, a, ha="center", fontsize=12, color=col, weight="bold")
    ax.text(x0 + 1.4, 1.8, "訓練資料：" + b, ha="center", fontsize=9.5)
    ax.text(x0 + 1.4, 0.95, "本章案例：\n" + c, ha="center", va="center", fontsize=9.5)
ax.text(4.65, -0.25, "遷移學習：先在大資料上預訓練 (pre-train)，再於目標任務微調 (fine-tune) — 跨越三種典範的訓練策略",
        ha="center", fontsize=9, color=GRAY)
ax.set_xlim(-0.2, 9.2)
ax.set_ylim(-0.6, 3.2)
ax.axis("off")
save(fig, "learning-paradigms.svg")

# XOR ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
pts = {(0, 0): 0, (1, 1): 0, (0, 1): 1, (1, 0): 1}
for ax in axes:
    for (a, b), lab in pts.items():
        ax.scatter([a], [b], s=140, color=RED if lab else BLUE, zorder=3)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.4, 1.4)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
axes[0].set_title("單層感知器：任何一條直線都分不開")
axes[0].plot([-0.4, 1.4], [1.15, -0.65], color=GRAY, ls="--", lw=1)
axes[1].set_title("兩個隱藏單元：OR 與 AND 兩條線")
xs_ = np.linspace(-0.4, 1.4, 10)
axes[1].plot(xs_, 0.5 - xs_, color=GREEN, lw=1.5, label="OR: x1+x2=0.5")
axes[1].plot(xs_, 1.5 - xs_, color=ORANGE, lw=1.5, label="AND: x1+x2=1.5")
axes[1].legend(frameon=False, fontsize=8, loc="upper right")
fig.text(0.5, -0.03, "紅 = 1，藍 = 0", ha="center", fontsize=9, color=GRAY)
fig.tight_layout()
save(fig, "xor-perceptron.svg")

# Double descent (Legendre basis, min-norm) -----------------------------
def leg(xv, M):
    return np.polynomial.legendre.legvander(2 * xv - 1, M)


Ms = np.arange(0, 41)
med = []
for M in Ms:
    errs = []
    for s in range(200):
        r = np.random.default_rng(900 + s)
        xs, ts = make_data(10, r)
        xv, tv = make_data(100, r)
        w = np.linalg.pinv(leg(xs, M)) @ ts
        errs.append(erms(leg(xv, M), tv, w))
    med.append(np.median(errs))
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(Ms, med, "o-", color=RED, ms=3)
ax.axvline(9, color=GRAY, ls="--", lw=1)
ax.text(9.5, max(med) * 0.95, "M+1 = N = 10\n（插值門檻）", fontsize=8, color=GRAY, va="top")
ax.set_ylim(0, 1.2)
ax.set_xlabel("M（參數個數 = M+1）")
ax.set_ylabel("測試 E_RMS（200 次中位數）")
ax.set_title("Double descent 示意（Legendre 基底、最小範數解、N = 10）", fontsize=9.5)
save(fig, "double-descent-legendre.svg")


# Fig 10 (two-layer feed-forward network) --------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.8))
layers = [(0.0, [1.5, 0.5, -0.5, -1.5], "輸入 inputs"),
          (2.2, [1.8, 0.9, 0.0, -0.9, -1.8], "隱藏單元 hidden units"),
          (4.4, [0.7, -0.7], "輸出 outputs")]
for (x0, ys, _), (x1, ys1, _) in zip(layers[:-1], layers[1:]):
    for ya in ys:
        for yb in ys1:
            ax.plot([x0, x1], [ya, yb], color="#c9d1d9", lw=0.7, zorder=1)
for x0, ys, name in layers:
    for yv in ys:
        ax.add_patch(Circle((x0, yv), 0.17, facecolor="white", edgecolor=BLUE, lw=1.6, zorder=2))
    ax.text(x0, -2.45, name, ha="center", fontsize=10)
ax.annotate("", xy=(4.9, 2.2), xytext=(0.0, 2.2), arrowprops=dict(arrowstyle="->", color=GRAY))
ax.text(2.45, 2.32, "資訊向前流動 (feed-forward)", ha="center", fontsize=9, color=GRAY)
ax.set_xlim(-0.5, 5.0)
ax.set_ylim(-2.8, 2.7)
ax.axis("off")
ax.set_title("圖 10：具有兩層參數的前饋神經網路")
save(fig, "fig10-two-layer-network.svg")
