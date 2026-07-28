"""Bar chart of IER by condition (both attributers) -> results/ier_figure.png."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = json.load(open("results/ier.json"))
conds = ["light", "heavy", "preserve"]
sty = [res["conditions"][c]["IER_stylometric"] * 100 for c in conds]
neu = [res["conditions"][c]["IER_neural"] * 100 for c in conds]

x = np.arange(len(conds)); w = 0.38
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.bar(x - w / 2, sty, w, label="stylometric attributer")
ax.bar(x + w / 2, neu, w, label="neural attributer")
ax.set_xticks(x); ax.set_xticklabels([c + "\nrewrite" for c in conds])
ax.set_ylabel("Idiolect Erasure Rate\n(pp drop in attribution accuracy)")
ax.set_title("Everyday assistants erase the author's fingerprint")
ax.axhline(0, color="#888", lw=0.8)
ax.legend(frameon=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig("results/ier_figure.png", dpi=160)
print("saved -> results/ier_figure.png")
