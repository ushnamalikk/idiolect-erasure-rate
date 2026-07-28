"""IER with a properly sized training set (same authors, same test messages).

The pilot trained each attributer on only 16 posts per author, which depressed
attribution baselines and invited the objection that IER measures the
perturbation of a weak classifier rather than the erasure of identity. Here we
keep the author set and the held-out messages identical (so cached rewrites
remain valid) and train on every remaining post, then recompute IER.

Writes results/blogs_rich_ier.json.
"""
import json
import numpy as np
from scipy import stats as sps

from src.data import load_blogs_rich
from src.attributers import StylometricAttributer

CONDS = ["light", "heavy", "preserve"]


def mcnemar(cb, ca):
    b = int(np.sum((cb == 1) & (ca == 0))); c = int(np.sum((cb == 0) & (ca == 1)))
    n = b + c
    return b, c, (1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue))


def boot(cb, ca, nb=5000):
    d = cb - ca; rng = np.random.default_rng(0); idx = np.arange(len(d))
    m = [d[rng.choice(idx, len(idx), True)].mean() for _ in range(nb)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


train, test = load_blogs_rich(max_authors=50, load_posts=200)
n_train = sum(len(v) for v in train.values())
print(f"authors={len(train)} train_posts={n_train} "
      f"(mean {n_train/len(train):.0f}/author) test={sum(len(v) for v in test.values())}")

items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
gold = [a for a, _, _ in items]
orig = [t for _, _, t in items]

cache = {}
for line in open("results/blogs_rewrites.jsonl"):
    r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
missing = [(c, a, i) for c in CONDS for a, i, _ in items if (c, a, i) not in cache]
print(f"cached rewrites reusable: {len(items)*3 - len(missing)}/{len(items)*3}")

sty = StylometricAttributer().fit(train)
c_o = np.array([p == g for p, g in zip(sty.predict(orig), gold)], dtype=int)
res = {"n_authors": len(train), "n_train_posts": n_train, "chance": 1/len(train),
       "baseline_acc": float(c_o.mean()), "conditions": {}}
print(f"\nSURFACE baseline = {c_o.mean():.3f}  (chance {1/len(train):.3f}, "
      f"{c_o.mean()*len(train):.0f}x chance)")
for cond in CONDS:
    rew = [cache[(cond, a, i)] for a, i, _ in items]
    c_r = np.array([p == g for p, g in zip(sty.predict(rew), gold)], dtype=int)
    b, c, p = mcnemar(c_o, c_r); lo, hi = boot(c_o, c_r)
    ier = float(c_o.mean() - c_r.mean())
    res["conditions"][cond] = {"acc_after": float(c_r.mean()), "IER": ier,
                               "ci95": [lo, hi], "p_value": p, "b": b, "c": c,
                               "frac_signal_destroyed": ier / float(c_o.mean())}
    print(f"  {cond:8} after={c_r.mean():.3f} IER={ier*100:+.1f} "
          f"CI=[{lo*100:+.0f},{hi*100:+.0f}] p={p:.3g} "
          f"({ier/float(c_o.mean())*100:.0f}% of signal)")

json.dump(res, open("results/blogs_rich_ier.json", "w"), indent=2)
print("saved -> results/blogs_rich_ier.json")
