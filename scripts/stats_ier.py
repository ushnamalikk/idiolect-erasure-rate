"""Significance + CIs for the pilot IER, computed per-item (paired).

Refits the attributers, gets per-message correctness on originals and on each
rewrite condition, then runs McNemar's exact test (paired) and a bootstrap 95%
CI on the accuracy drop. Saves results/stats.json.
"""
import json
import numpy as np
from scipy import stats as sps

from src.data import load_c50
from src.attributers import StylometricAttributer, NeuralAttributer

CONDS = ["light", "heavy", "preserve"]
cache = {}
for line in open("results/rewrites.jsonl"):
    r = json.loads(line)
    cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]

train = load_c50("data", "train", max_words=90)
test = load_c50("data", "test", max_authors=25, max_docs_per_author=4, max_words=90)
items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
gold = [a for a, _, _ in items]
orig = [t for _, _, t in items]

attributers = {"stylometric": StylometricAttributer().fit(train),
               "neural": NeuralAttributer().fit(train)}


def correct(model, texts):
    return np.array([p == g for p, g in zip(model.predict(texts), gold)], dtype=int)


def mcnemar_exact(c_before, c_after):
    b = int(np.sum((c_before == 1) & (c_after == 0)))   # was right, now wrong
    c = int(np.sum((c_before == 0) & (c_after == 1)))   # was wrong, now right
    n = b + c
    p = 1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue)
    return b, c, p


def boot_ci(c_before, c_after, n_boot=5000):
    d = c_before - c_after                # 1 = erased this item
    idx = np.arange(len(d))
    rng = np.random.default_rng(0)
    means = [d[rng.choice(idx, len(idx), replace=True)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


out = {"n_items": len(items), "n_authors": len(test)}
for name, model in attributers.items():
    c_orig = correct(model, orig)
    out[name] = {"baseline_acc": float(c_orig.mean()), "conditions": {}}
    for cond in CONDS:
        rew = [cache[(cond, a, i)] for a, i, _ in items]
        c_rew = correct(model, rew)
        b, c, p = mcnemar_exact(c_orig, c_rew)
        lo, hi = boot_ci(c_orig, c_rew)
        out[name]["conditions"][cond] = {
            "acc_after": float(c_rew.mean()),
            "IER": float(c_orig.mean() - c_rew.mean()),
            "IER_ci95": [lo, hi],
            "mcnemar_b_was_right_now_wrong": b,
            "mcnemar_c_was_wrong_now_right": c,
            "p_value": p,
        }
        print(f"[{name:11}/{cond:8}] IER={c_orig.mean()-c_rew.mean():+.3f} "
              f"CI=[{lo:+.3f},{hi:+.3f}]  b={b} c={c}  p={p:.4f}")

json.dump(out, open("results/stats.json", "w"), indent=2)
print("saved -> results/stats.json")
