"""Capability curve figure: heavy-rewrite IER vs. assistant capability."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("results/capability.json"))["points"]
order = sorted(d, key=lambda k: d[k]["rank"])
xs = np.arange(len(order))
sur = [d[k]["IER_surface"] * 100 for k in order]
neu = [d[k]["IER_neural"] * 100 for k in order]

fig, ax = plt.subplots(figsize=(6.2, 3.7))
ax.plot(xs, sur, "-o", color="#555", label="surface fingerprint")
ax.plot(xs, neu, "-o", color="#1f77b4", label="deep fingerprint")
ax.set_xticks(xs); ax.set_xticklabels(order, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Idiolect Erasure Rate (pp), heavy rewrite")
ax.set_xlabel("assistant  (less capable  →  more capable)")
ax.set_title("Erasure is stable across model scale (Qwen family)")
ax.axhline(0, color="#bbb", lw=0.8)
ax.legend(frameon=False, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("results/capability_figure.png", dpi=160)
print("saved -> results/capability_figure.png")
print("surface:", dict(zip(order, [round(x) for x in sur])))
print("neural: ", dict(zip(order, [round(x) for x in neu])))
