"""Figure 2: population accumulation (LUAR). Author re-identification vs number of
pooled messages, for original vs AI-assisted messages. Direct-labeled lines,
validated reference palette (light surface)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
ORIG_C = "#2a78d6"      # blue -> original (intact)
ASSIST_C = "#e0559a"    # pink -> AI-assisted (erased)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": SECOND,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

r = json.load(open("results/population_luar.json"))
ks = r["KS"]; o = [v * 100 for v in r["orig"]]; a = [v * 100 for v in r["assist"]]

fig, ax = plt.subplots(figsize=(4.3, 3.1))
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, lw=0.7)

ax.plot(ks, o, "-o", color=ORIG_C, lw=2.0, ms=6, mec="white", mew=0.8, zorder=3)
ax.plot(ks, a, "-o", color=ASSIST_C, lw=2.0, ms=6, mec="white", mew=0.8, zorder=3)

# direct labels at the right end (no legend box)
ax.text(ks[-1] + 0.4, o[-1], "original", color=ORIG_C, fontsize=9.5,
        va="center", ha="left", fontweight="bold")
ax.text(ks[-1] + 0.4, a[-1], "AI-assisted", color=ASSIST_C, fontsize=9.5,
        va="center", ha="left", fontweight="bold")

ax.set_xlabel("messages aggregated (k)", fontsize=10.5)
ax.set_ylabel("author re-identification (%)", fontsize=10.5)
ax.set_ylim(0, 105); ax.set_xlim(0.5, 19.5)
ax.set_xticks([1, 5, 10, 15]); ax.set_yticks([0, 25, 50, 75, 100])
ax.tick_params(length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("results/population_figure.png", dpi=200, bbox_inches="tight")
print("saved -> results/population_figure.png | k=15 orig", o[-1], "assist", a[-1])
