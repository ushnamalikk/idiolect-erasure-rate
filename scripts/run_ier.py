"""Compute the Idiolect Erasure Rate (IER) end-to-end.

IER(condition, attributer) = acc(subset originals) - acc(subset rewrites).

Pipeline:
  1. fit both attributers on the FULL C50 train set (real fingerprints).
  2. take a pilot subset of test docs, truncated to message length.
  3. rewrite the subset with a local model at 3 conditions (dose ladder).
  4. re-attribute the rewrites; IER = the drop in accuracy.
  5. idiolect drift: how far rewrites move toward the global 'average voice'.

Fully resumable: every rewrite is cached to results/rewrites.jsonl, so a
restart regenerates only what is missing, then computes metrics.

Run:  python3 -m scripts.run_ier --authors 25 --docs 4 --max_words 90
"""
import argparse, json, os, time
import numpy as np

from src.data import load_c50
from src.attributers import StylometricAttributer, NeuralAttributer
from src.rewriters import get_rewriter

CONDS = ["light", "heavy", "preserve"]
CACHE = "results/rewrites.jsonl"


def load_cache():
    seen = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            r = json.loads(line)
            seen[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
    return seen


def append_cache(rec):
    with open(CACHE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--authors", type=int, default=25)
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--max_words", type=int, default=90)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--metrics_only", action="store_true")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    # 1. fit attributers on FULL train
    train = load_c50(args.root, "train", max_words=args.max_words)
    print("fitting attributers on full train ...")
    sty = StylometricAttributer().fit(train)
    neu = NeuralAttributer().fit(train)
    centroid = neu.centroid()

    # 2. pilot subset of test docs
    test = load_c50(args.root, "test", max_authors=args.authors,
                    max_docs_per_author=args.docs, max_words=args.max_words)
    items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
    authors_gold = [a for a, _, _ in items]
    originals = [t for _, _, t in items]
    print(f"subset: {len(items)} docs from {len(test)} authors")

    # 3. rewrite (resumable)
    cache = load_cache()
    todo = [(c, a, i, t) for c in CONDS for (a, i, t) in items
            if (c, a, i) not in cache]
    print(f"cached rewrites: {len(cache)}  |  to generate: {len(todo)}")
    if todo and not args.metrics_only:
        rw = get_rewriter("local", model_name=args.model)
        t0 = time.time()
        for n, (c, a, i, t) in enumerate(todo, 1):
            out = rw.rewrite([t], c)[0]
            append_cache({"cond": c, "author": a, "idx": i, "rewrite": out})
            cache[(c, a, i)] = out
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  ({el/n:.1f}s/gen, ~{el/n*(len(todo)-n)/60:.1f} min left)",
                      flush=True)

    still = [(c, a, i) for c in CONDS for (a, i, t) in items
             if (c, a, i) not in cache]
    if still:
        print(f"still missing {len(still)} rewrites; rerun to finish.")
        return

    # 4-5. metrics
    def acc(pred, gold):
        return float(np.mean([p == g for p, g in zip(pred, gold)]))

    base = {"stylometric": acc(sty.predict(originals), authors_gold),
            "neural": acc(neu.predict(originals), authors_gold)}
    orig_emb = neu._embed(originals)
    orig_drift = float(np.mean(orig_emb @ centroid))

    res = {"n_docs": len(items), "n_authors": len(test),
           "chance": 1.0 / len(train), "max_words": args.max_words,
           "model": args.model, "baseline_subset_acc": base,
           "orig_centroid_sim": orig_drift, "conditions": {}}

    for c in CONDS:
        rew = [cache[(c, a, i)] for (a, i, t) in items]
        sty_acc = acc(sty.predict(rew), authors_gold)
        neu_acc = acc(neu.predict(rew), authors_gold)
        rew_emb = neu._embed(rew)
        drift = float(np.mean(rew_emb @ centroid))
        res["conditions"][c] = {
            "stylometric_acc_after": sty_acc,
            "neural_acc_after": neu_acc,
            "IER_stylometric": base["stylometric"] - sty_acc,
            "IER_neural": base["neural"] - neu_acc,
            "rewrite_centroid_sim": drift,
            "drift_toward_mean": drift - orig_drift,
        }
        print(f"[{c:8}] IER_sty={res['conditions'][c]['IER_stylometric']:+.3f}  "
              f"IER_neu={res['conditions'][c]['IER_neural']:+.3f}  "
              f"drift={res['conditions'][c]['drift_toward_mean']:+.3f}")

    with open("results/ier.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved -> results/ier.json")


if __name__ == "__main__":
    main()
