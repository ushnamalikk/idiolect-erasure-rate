"""Figure 1: heavy-rewrite IER, surface vs deep (LUAR), across three corpora.

Grouped bars with author-clustered 95% CIs and direct value labels. Colors and
chrome follow the validated reference data-viz palette (light surface).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- validated reference palette (light surface) ---
INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
SURFACE_C = "#2a78d6"   # blue -> surface (stylometric)
DEEP_C = "#e0559a"      # pink -> deep (LUAR); direct labels satisfy the relief rule

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "svg.fonttype": "none",
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": SECOND,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

cis = json.load(open("results/cluster_cis.json"))
order = [("blogs", "Blogs\n(personal)"), ("enron", "Enron\n(email)"),
         ("c50", "C50\n(news)")]


def series(kind):
    v, lo, hi, ns = [], [], [], []
    for k, _ in order:
        r = cis[k][kind]
        v.append(r["IER"]); lo.append(r["IER"] - r["author_ci95"][0])
        hi.append(r["author_ci95"][1] - r["IER"]); ns.append(r["author_ci95"][0] <= 0)
    return np.array(v), [lo, hi], ns


sv, se, sns = series("surface")
dv, de, dns = series("luar")

x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.axhline(0, color=BASE, lw=0.8, zorder=2)

ebar = dict(ecolor=SECOND, elinewidth=1.0, capsize=3, capthick=1.0, alpha=0.75)
ax.bar(x - w/2, sv, w, color=SURFACE_C, label="Surface (stylometric)",
       yerr=se, error_kw=ebar, zorder=3)
ax.bar(x + w/2, dv, w, color=DEEP_C, label="Deep (LUAR)",
       yerr=de, error_kw=ebar, zorder=3)

for xi, (v, hi, ns) in enumerate(zip(sv, se[1], sns)):
    ax.text(xi - w/2, v + hi + 2.5, "n.s." if ns else f"+{v:.1f}",
            ha="center", va="bottom", fontsize=8.5, color=INK)
for xi, (v, hi, ns) in enumerate(zip(dv, de[1], dns)):
    ax.text(xi + w/2, v + hi + 2.5, "n.s." if ns else f"+{v:.1f}",
            ha="center", va="bottom", fontsize=8.5, color=INK)

ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in order],
                                     color=SECOND, fontsize=9.5)
ax.set_ylabel("Idiolect Erasure Rate (pp)", fontsize=10.5)
ax.set_ylim(-13, 84)
ax.tick_params(length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9.5, loc="upper right",
          handlelength=1.0, borderaxespad=0.2)
fig.tight_layout()
fig.savefig("results/compare_figure.png", dpi=200, bbox_inches="tight")
print("saved -> results/compare_figure.png",
      "| surface", [round(v) for v in sv], "| deep", [round(v, 1) for v in dv])
