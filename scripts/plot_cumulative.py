"""Plot cumulative IER vs number of repeated LIGHT edits -> the 'ripple' figure."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cur = json.load(open("results/cumulative.json"))["curve"]
ks = sorted(int(k.split("_")[1]) for k in cur)
ier = [cur[f"pass_{k}"]["IER_sty"] * 100 for k in ks]

fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(ks, ier, "-o", color="#1f77b4", lw=2)
ax.set_xlabel("number of repeated 'grammar-only' edits on the same message")
ax.set_ylabel("Idiolect Erasure Rate\n(pp drop, stylometric)")
ax.set_title("Re-editing a message does not compound erasure")
ax.set_xticks(ks)
ax.axhline(0, color="#bbb", lw=0.8)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig("results/cumulative_figure.png", dpi=160)
print("saved -> results/cumulative_figure.png  | IER by pass:", dict(zip(ks, [round(x,1) for x in ier])))
