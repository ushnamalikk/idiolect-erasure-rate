"""Cumulative-dose IER: does erasure accumulate across repeated LIGHT edits?

The workshop's thesis is that individually-harmless edits accumulate. We test it
directly: apply the *light* (grammar-only) rewrite iteratively --
pass_k = light(pass_{k-1}) -- and measure IER after each pass. If a single light
edit erases little but k of them erase a lot, that is the ripple.

Pass 1 is seeded from the existing light rewrites (results/rewrites.jsonl).
Resumable: checkpoints to results/cumulative.jsonl keyed (author, idx, pass).

Run:  python3 -m scripts.run_cumulative --passes 4
"""
import argparse, json, os, time
import numpy as np

from src.data import load_c50
from src.attributers import StylometricAttributer, NeuralAttributer
from src.rewriters import get_rewriter

CACHE = "results/cumulative.jsonl"


def load_cache():
    seen = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            r = json.loads(line)
            seen[(r["author"], r["idx"], r["pass"])] = r["text"]
    return seen


def append(rec):
    with open(CACHE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--authors", type=int, default=25)
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--max_words", type=int, default=90)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    train = load_c50("data", "train", max_words=args.max_words)
    sty = StylometricAttributer().fit(train)
    neu = NeuralAttributer().fit(train)

    test = load_c50("data", "test", max_authors=args.authors,
                    max_docs_per_author=args.docs, max_words=args.max_words)
    items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
    gold = [a for a, _, _ in items]

    cache = load_cache()
    # seed pass 1 from the earlier light rewrites
    if os.path.exists("results/rewrites.jsonl"):
        for line in open("results/rewrites.jsonl"):
            r = json.loads(line)
            if r["cond"] == "light" and (r["author"], r["idx"], 1) not in cache:
                cache[(r["author"], r["idx"], 1)] = r["rewrite"]
                append({"author": r["author"], "idx": r["idx"], "pass": 1,
                        "text": r["rewrite"]})

    rw = None
    for k in range(1, args.passes + 1):
        todo = [(a, i) for a, i, _ in items if (a, i, k) not in cache]
        if not todo:
            continue
        if rw is None:
            rw = get_rewriter("local", model_name=args.model)
        print(f"pass {k}: generating {len(todo)} ...", flush=True)
        t0 = time.time()
        for n, (a, i) in enumerate(todo, 1):
            prev = cache[(a, i, k - 1)]           # rewrite the previous pass
            out = rw.rewrite([prev], "light")[0]
            cache[(a, i, k)] = out
            append({"author": a, "idx": i, "pass": k, "text": out})
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} ({el/n:.1f}s/gen)", flush=True)

    # score every pass
    def acc(texts):
        return float(np.mean([p == g for p, g in zip(sty.predict(texts), gold)]))

    def acc_n(texts):
        return float(np.mean([p == g for p, g in zip(neu.predict(texts), gold)]))

    originals = [t for _, _, t in items]
    base_s, base_n = acc(originals), acc_n(originals)
    curve = {"pass_0": {"acc_sty": base_s, "acc_neu": base_n,
                        "IER_sty": 0.0, "IER_neu": 0.0}}
    print(f"pass 0 (original): acc_sty={base_s:.3f}")
    for k in range(1, args.passes + 1):
        txt = [cache[(a, i, k)] for a, i, _ in items]
        a_s, a_n = acc(txt), acc_n(txt)
        curve[f"pass_{k}"] = {"acc_sty": a_s, "acc_neu": a_n,
                              "IER_sty": base_s - a_s, "IER_neu": base_n - a_n}
        print(f"pass {k}: acc_sty={a_s:.3f}  IER_sty={base_s-a_s:+.3f}", flush=True)

    json.dump({"model": args.model, "n_docs": len(items), "curve": curve},
              open("results/cumulative.json", "w"), indent=2)
    print("saved -> results/cumulative.json")


if __name__ == "__main__":
    main()
