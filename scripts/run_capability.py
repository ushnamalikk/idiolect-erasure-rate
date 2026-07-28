"""Capability curve: does a more capable assistant erase more or less identity?

Measures heavy-rewrite IER on the blog test set for assistants of increasing
capability: Qwen2.5 {0.5B, 1.5B, 3B} (local) plus commercial gpt-4o-mini.
Reuses existing rewrites where available; generates the rest (resumable).

Run:  python3 -m scripts.run_capability
"""
import json, os, time
import numpy as np

from src.data import load_blogs, split_corpus
from src.attributers import StylometricAttributer, NeuralAttributer
from src.rewriters import get_rewriter

# label, capability-rank, kind, model_name, cache_file
POINTS = [
    ("Qwen-0.5B", 1, "local", "Qwen/Qwen2.5-0.5B-Instruct", "results/cap_qwen05_rewrites.jsonl"),
    ("Qwen-1.5B", 2, "reuse", None, "results/blogs_rewrites.jsonl"),
    ("Qwen-3B", 3, "local", "Qwen/Qwen2.5-3B-Instruct", "results/cap_qwen3_rewrites.jsonl"),
    ("gpt-4o-mini", 4, "reuse", None, "results/blogs_openai_gpt-4o-mini_rewrites.jsonl"),
]

corpus = load_blogs(max_authors=50, posts_per_author=20, max_words=90)
train, test = split_corpus(corpus, n_test=4)
sty = StylometricAttributer().fit(train)
neu = NeuralAttributer().fit(train)
items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
gold = [a for a, _, _ in items]
originals = [t for _, _, t in items]
base_s = float(np.mean([p == g for p, g in zip(sty.predict(originals), gold)]))
base_n = float(np.mean([p == g for p, g in zip(neu.predict(originals), gold)]))
print(f"blogs baseline surface={base_s:.2f} neural={base_n:.2f}")


def heavy_rewrites_from_cache(path):
    m = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r["cond"] == "heavy":
                m[(r["author"], r["idx"])] = r["rewrite"]
    return m


def generate_heavy(model_name, path):
    cache = heavy_rewrites_from_cache(path)
    todo = [(a, i, t) for (a, i, t) in items if (a, i) not in cache]
    if todo:
        rw = get_rewriter("local", model_name=model_name)
        t0 = time.time()
        for n, (a, i, t) in enumerate(todo, 1):
            out = rw.rewrite([t], "heavy")[0]
            with open(path, "a") as f:
                f.write(json.dumps({"cond": "heavy", "author": a, "idx": i, "rewrite": out}) + "\n")
            cache[(a, i)] = out
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  [{model_name}] {n}/{len(todo)} ({el/n:.1f}s/gen)", flush=True)
    return cache


results = {}
for label, rank, kind, model_name, path in POINTS:
    if kind == "local":
        cache = generate_heavy(model_name, path)
    else:
        cache = heavy_rewrites_from_cache(path)
    if len(cache) < len(items):
        print(f"  {label}: only {len(cache)}/{len(items)} rewrites; skipping")
        continue
    rew = [cache[(a, i)] for a, i, _ in items]
    ier_s = base_s - float(np.mean([p == g for p, g in zip(sty.predict(rew), gold)]))
    ier_n = base_n - float(np.mean([p == g for p, g in zip(neu.predict(rew), gold)]))
    results[label] = {"rank": rank, "IER_surface": ier_s, "IER_neural": ier_n}
    print(f"[{label:12}] surface IER={ier_s*100:+.0f}  neural IER={ier_n*100:+.0f}", flush=True)

json.dump({"baseline_surface": base_s, "baseline_neural": base_n, "points": results},
          open("results/capability.json", "w"), indent=2)
print("saved -> results/capability.json")
