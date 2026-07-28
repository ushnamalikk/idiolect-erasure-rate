"""Blogs IER on full-length messages (no mid-sentence truncation).

The pilot truncated every message to 90 words, which (a) depressed attribution
baselines by roughly 12 points and (b) introduced an artifact: most truncated
originals ended mid-sentence, so part of what the assistant "fixed" was our own
truncation. Here messages are used at their natural length (capped at 400 words)
and rewrites are regenerated to match.

Resumable: results/blogs400_rewrites.jsonl.
"""
import json, os, time
import numpy as np
from scipy import stats as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import load_blogs_rich, to_xy
from src.rewriters import get_rewriter

CONDS = ["light", "heavy", "preserve"]
CACHE = "results/blogs400_rewrites.jsonl"
MAXW = 400


def mcnemar(cb, ca):
    b = int(np.sum((cb == 1) & (ca == 0))); c = int(np.sum((cb == 0) & (ca == 1)))
    n = b + c
    return b, c, (1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue))


def boot(cb, ca, nb=5000):
    d = cb - ca; rng = np.random.default_rng(0); idx = np.arange(len(d))
    return tuple(float(x) for x in np.percentile(
        [d[rng.choice(idx, len(idx), True)].mean() for _ in range(nb)], [2.5, 97.5]))


train, test = load_blogs_rich(max_authors=50, load_posts=200, max_words=MAXW)
items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
gold = [a for a, _, _ in items]; orig = [t for _, _, t in items]
print(f"authors={len(train)} train={sum(len(v) for v in train.values())} "
      f"test={len(items)} mean_test_len={np.mean([len(t.split()) for t in orig]):.0f}w",
      flush=True)

cache = {}
if os.path.exists(CACHE):
    for line in open(CACHE):
        r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
todo = [(c, a, i, t) for c in CONDS for (a, i, t) in items if (c, a, i) not in cache]
print(f"cached={len(cache)} todo={len(todo)}", flush=True)
if todo:
    rw = get_rewriter("local", model_name="Qwen/Qwen2.5-1.5B-Instruct",
                      max_new_tokens=600)
    t0 = time.time()
    for n, (c, a, i, t) in enumerate(todo, 1):
        out = rw.rewrite([t], c)[0]
        with open(CACHE, "a") as f:
            f.write(json.dumps({"cond": c, "author": a, "idx": i, "rewrite": out}) + "\n")
        cache[(c, a, i)] = out
        if n % 20 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)} ({el/n:.1f}s/gen, ~{el/n*(len(todo)-n)/60:.0f} min left)",
                  flush=True)

def build():
    return Pipeline([("f", FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))])),
        ("c", LinearSVC(C=1.0, class_weight="balanced"))])

Xtr, ytr = to_xy(train)
m = build(); m.fit(Xtr, ytr)
c_o = np.array([p == g for p, g in zip(m.predict(orig), gold)], dtype=int)
res = {"max_words": MAXW, "n_authors": len(train), "chance": 1/len(train),
       "baseline_acc": float(c_o.mean()), "conditions": {}}
print(f"\nSURFACE baseline = {c_o.mean():.3f}", flush=True)
for cond in CONDS:
    rew = [cache[(cond, a, i)] for a, i, _ in items]
    c_r = np.array([p == g for p, g in zip(m.predict(rew), gold)], dtype=int)
    b, c, p = mcnemar(c_o, c_r); lo, hi = boot(c_o, c_r)
    ier = float(c_o.mean() - c_r.mean())
    res["conditions"][cond] = {"acc_after": float(c_r.mean()), "IER": ier,
                               "ci95": [lo, hi], "p_value": p, "b": b, "c": c,
                               "frac_signal_destroyed": ier / float(c_o.mean())}
    print(f"  {cond:8} after={c_r.mean():.3f} IER={ier*100:+.1f} "
          f"CI=[{lo*100:+.0f},{hi*100:+.0f}] p={p:.3g} "
          f"({ier/float(c_o.mean())*100:.0f}% of signal)", flush=True)
json.dump(res, open("results/blogs400_ier.json", "w"), indent=2)
print("saved -> results/blogs400_ier.json")
