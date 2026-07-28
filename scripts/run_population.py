"""Population-accumulation experiment: identity erosion vs. assistance prevalence.

The paper argues the 'ripple' lives at the population level: no single message
compounds, but a PERSON's identifiability erodes as more of their messages are
assisted. We measure it directly. For each author we form a query set of k of
their messages, of which a fraction p are AI-rewritten, and try to re-identify
the author from the SET (neural: mean embedding -> nearest author; stylometric:
summed log-probabilities). We sweep k and p.

If original-message sets re-identify the author well as k grows, but assisted
sets do not, then assistance erases identity at the corpus level -- and p traces
the accumulation.

Resumable heavy rewrites cached to results/pop_rewrites.jsonl.
Run:  python3 -m scripts.run_population
"""
import argparse, json, os, time
import numpy as np

from src.data import load_blogs, split_corpus
from src.attributers import StylometricAttributer, NeuralAttributer
from src.rewriters import get_rewriter

CACHE = "results/pop_rewrites.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", type=int, default=20)
    ap.add_argument("--posts", type=int, default=30)
    ap.add_argument("--n_test", type=int, default=15)
    ap.add_argument("--max_words", type=int, default=90)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    corpus = load_blogs(max_authors=args.authors, posts_per_author=args.posts,
                        max_words=args.max_words)
    train, test = split_corpus(corpus, n_test=args.n_test)
    sty = StylometricAttributer().fit(train)
    neu = NeuralAttributer().fit(train)

    items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
    print(f"blogs: {len(corpus)} authors, {len(items)} test msgs")

    # --- generate heavy rewrites (resumable) ---
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            r = json.loads(line); cache[(r["author"], r["idx"])] = r["rewrite"]
    todo = [(a, i, t) for (a, i, t) in items if (a, i) not in cache]
    print(f"cached={len(cache)} todo={len(todo)}")
    if todo:
        rw = get_rewriter("local", model_name=args.model)
        t0 = time.time()
        for n, (a, i, t) in enumerate(todo, 1):
            out = rw.rewrite([t], "heavy")[0]
            with open(CACHE, "a") as f:
                f.write(json.dumps({"author": a, "idx": i, "rewrite": out}) + "\n")
            cache[(a, i)] = out
            if n % 20 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} ({el/n:.1f}s/gen, ~{el/n*(len(todo)-n)/60:.1f} min left)", flush=True)

    # --- precompute per-message signals ---
    authors = neu.authors
    aidx = {a: k for k, a in enumerate(authors)}
    by_author = {}
    for a, i, t in items:
        by_author.setdefault(a, []).append((t, cache[(a, i)]))

    # neural embeddings; stylometric log-probs over classes
    sty_classes = list(sty.model.named_steps["clf"].classes_)
    sty_map = {c: j for j, c in enumerate(sty_classes)}

    def set_scores(msgs, use_rewrite_mask):
        """msgs: list of (orig, rewrite); mask: bool per msg -> use rewrite."""
        texts = [rw if m else o for (o, rw), m in zip(msgs, use_rewrite_mask)]
        # neural: mean embedding -> nearest author profile
        emb = neu._embed(texts).mean(0)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        neu_pred = authors[int((neu.profiles @ emb).argmax())]
        # stylometric: sum log-proba over messages
        lp = np.log(sty.model.predict_proba(texts) + 1e-12).sum(0)
        sty_pred = sty_classes[int(lp.argmax())]
        return sty_pred, neu_pred

    rng = np.random.default_rng(0)
    KS = [1, 2, 3, 5, 8, 12, 15]
    PS = [0.0, 0.25, 0.5, 0.75, 1.0]
    curve_k = {"orig": {"sty": [], "neu": []}, "assist": {"sty": [], "neu": []}}
    for k in KS:
        acc = {"orig": {"sty": 0, "neu": 0}, "assist": {"sty": 0, "neu": 0}}
        tot = 0
        for a, msgs in by_author.items():
            if len(msgs) < k:
                continue
            for _ in range(args.reps):
                sel = [msgs[j] for j in rng.choice(len(msgs), k, replace=False)]
                for cond, mask in [("orig", [False]*k), ("assist", [True]*k)]:
                    sp, npd = set_scores(sel, mask)
                    acc[cond]["sty"] += (sp == a)
                    acc[cond]["neu"] += (npd == a)
                tot += 1
        for cond in ("orig", "assist"):
            curve_k[cond]["sty"].append(acc[cond]["sty"]/tot)
            curve_k[cond]["neu"].append(acc[cond]["neu"]/tot)
        print(f"k={k:2d} | orig neu={acc['orig']['neu']/tot:.2f} sty={acc['orig']['sty']/tot:.2f}"
              f" | assist neu={acc['assist']['neu']/tot:.2f} sty={acc['assist']['sty']/tot:.2f}", flush=True)

    # prevalence sweep at fixed k
    KP = 10
    prev = {"sty": [], "neu": []}
    for p in PS:
        a_s = a_n = tot = 0
        for a, msgs in by_author.items():
            if len(msgs) < KP:
                continue
            for _ in range(args.reps):
                sel = [msgs[j] for j in rng.choice(len(msgs), KP, replace=False)]
                mask = [False]*KP
                for j in rng.choice(KP, int(round(p*KP)), replace=False):
                    mask[j] = True
                sp, npd = set_scores(sel, mask)
                a_s += (sp == a); a_n += (npd == a); tot += 1
        prev["sty"].append(a_s/tot); prev["neu"].append(a_n/tot)
        print(f"p={p:.2f} (k={KP}) | neu={a_n/tot:.2f} sty={a_s/tot:.2f}", flush=True)

    out = {"n_authors": len(corpus), "chance": 1/len(corpus), "KS": KS, "PS": PS,
           "KP": KP, "curve_k": curve_k, "prevalence": prev}
    json.dump(out, open("results/population.json", "w"), indent=2)
    print("saved -> results/population.json")


if __name__ == "__main__":
    main()
