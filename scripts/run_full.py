"""Full-study IER on an interpersonal corpus (blogs or Enron email).

The decisive test the C50 pilot could not run: on topic-varied interpersonal
text, topic is NOT tied to author, so a content-based attributer cannot cheat.
If NEURAL IER is positive and significant, deep idiolect erasure is real.

Resumable: rewrites cached to results/<corpus>_rewrites.jsonl.
Run:  python3 -m scripts.run_full --corpus enron
      python3 -m scripts.run_full --corpus blogs --authors 50
"""
import argparse, json, os, time
import numpy as np
from scipy import stats as sps

from src.data import load_blogs, load_enron, split_corpus
from src.attributers import StylometricAttributer, NeuralAttributer
from src.rewriters import get_rewriter

CONDS = ["light", "heavy", "preserve"]


def load_cache(path):
    seen = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            seen[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
    return seen


def mcnemar(c_b, c_a):
    b = int(np.sum((c_b == 1) & (c_a == 0)))
    c = int(np.sum((c_b == 0) & (c_a == 1)))
    n = b + c
    p = 1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue)
    return b, c, p


def boot(c_b, c_a, nb=5000):
    d = c_b - c_a
    rng = np.random.default_rng(0)
    idx = np.arange(len(d))
    m = [d[rng.choice(idx, len(idx), True)].mean() for _ in range(nb)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["blogs", "enron"], default="blogs")
    ap.add_argument("--authors", type=int, default=25)
    ap.add_argument("--posts", type=int, default=20)
    ap.add_argument("--n_test", type=int, default=4)
    ap.add_argument("--max_words", type=int, default=90)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--rewriter", choices=["local", "openai", "gemini", "mock"], default="local")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    tag = "" if args.rewriter == "local" else \
        f"_{args.rewriter}_{args.model.replace('/', '-').replace('.', '-')}"
    cache_path = f"results/{args.corpus}{tag}_rewrites.jsonl"
    out_path = f"results/{args.corpus}{tag}_ier.json"

    if args.corpus == "blogs":
        corpus = load_blogs(max_authors=args.authors, posts_per_author=args.posts,
                            max_words=args.max_words)
    else:
        corpus = load_enron(max_authors=args.authors, posts_per_author=args.posts,
                            max_words=args.max_words)
    train, test = split_corpus(corpus, n_test=args.n_test)
    sty = StylometricAttributer().fit(train)
    neu = NeuralAttributer().fit(train)
    centroid = neu.centroid()

    items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
    gold = [a for a, _, _ in items]
    originals = [t for _, _, t in items]
    print(f"{args.corpus}: {len(corpus)} authors, {len(items)} test msgs, chance={1/len(corpus):.3f}")

    cache = load_cache(cache_path)
    todo = [(c, a, i, t) for c in CONDS for (a, i, t) in items if (c, a, i) not in cache]
    print(f"cached={len(cache)} todo={len(todo)}")
    if todo:
        if args.rewriter in ("openai", "gemini"):
            rw = get_rewriter(args.rewriter, model=args.model)
        else:
            rw = get_rewriter("local", model_name=args.model)
        t0 = time.time()
        for n, (c, a, i, t) in enumerate(todo, 1):
            out = rw.rewrite([t], c)[0]
            with open(cache_path, "a") as f:
                f.write(json.dumps({"cond": c, "author": a, "idx": i, "rewrite": out}) + "\n")
            cache[(c, a, i)] = out
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} ({el/n:.1f}s/gen, ~{el/n*(len(todo)-n)/60:.1f} min left)", flush=True)

    def corr(model, texts):
        return np.array([p == g for p, g in zip(model.predict(texts), gold)], dtype=int)

    res = {"corpus": args.corpus, "n_authors": len(corpus), "n_msgs": len(items),
           "chance": 1 / len(corpus), "model": args.model, "attributers": {}}
    orig_sim = float(np.mean(neu._embed(originals) @ centroid))
    for name, model in [("stylometric", sty), ("neural", neu)]:
        c_o = corr(model, originals)
        d = {"baseline_acc": float(c_o.mean()), "conditions": {}}
        for cond in CONDS:
            rew = [cache[(cond, a, i)] for a, i, _ in items]
            c_r = corr(model, rew)
            b, c, p = mcnemar(c_o, c_r)
            lo, hi = boot(c_o, c_r)
            entry = {"acc_after": float(c_r.mean()), "IER": float(c_o.mean() - c_r.mean()),
                     "IER_ci95": [lo, hi], "p_value": p, "b": b, "c": c}
            d["conditions"][cond] = entry
            print(f"[{name:11}/{cond:8}] IER={entry['IER']:+.3f} CI=[{lo:+.3f},{hi:+.3f}] p={p:.4f}", flush=True)
        res["attributers"][name] = d

    json.dump(res, open(out_path, "w"), indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
