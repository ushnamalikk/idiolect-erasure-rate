"""Erasure vs. transformation, and author-clustered inference (full-length v2).

The main experiment trains on original text and tests on rewritten text, so a
drop could mean the author signal was destroyed OR merely moved. This script
adds the missing control the reviewer called most important: are authors still
distinguishable AMONG rewritten texts?

Because we do not have rewritten versions of the large training sets, we answer
this with a leave-one-out (LOO) evaluation restricted to the held-out messages,
using LUAR (which needs no fitting, only author profiles):

  - orig->orig LOO:      profile each author from 3 of their 4 ORIGINAL held-out
                         messages, attribute the 4th; rotate. Author separability
                         among originals at this small sample size.
  - rewr->rewr LOO:      same, but using each author's 4 HEAVY-REWRITTEN messages.
                         Author separability among rewrites.

If rewr->rewr stays near orig->orig, the assistant transformed each author's
style but preserved author *structure* (a continuity break, not destruction).
If rewr->rewr collapses toward chance, rewriting homogenizes authors (erasure).

It also reports the headline original->rewritten IER with an AUTHOR-CLUSTERED
bootstrap CI (resampling authors, not messages), addressing pseudoreplication.

Writes results/transform.json.
"""
import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.data import load_blogs_rich, load_enron_rich

DEV = "cpu"; L = 256
tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True).to(DEV).eval()


def embed(texts, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i+bs], max_length=L, padding="max_length",
                truncation=True, return_tensors="pt")
        with torch.no_grad():
            v = model(input_ids=e["input_ids"].unsqueeze(1),
                      attention_mask=e["attention_mask"].unsqueeze(1))
        v = v[0] if isinstance(v, (tuple, list)) else v
        v = torch.nan_to_num(v, 0., 0., 0.)
        out.append(np.nan_to_num(torch.nn.functional.normalize(v, dim=1).numpy()))
    return np.vstack(out)


def loo_separability(by_author_texts):
    """LOO nearest-profile accuracy within a set of texts per author."""
    authors = list(by_author_texts.keys())
    # embed everything once
    emb = {a: embed(by_author_texts[a]) for a in authors}
    hits = tot = 0
    for a in authors:
        n = len(emb[a])
        for i in range(n):
            # held-out message i of author a
            q = emb[a][i]
            best, best_a = -1e9, None
            for b in authors:
                if b == a:
                    idx = [j for j in range(len(emb[b])) if j != i]
                else:
                    idx = list(range(len(emb[b])))
                prof = emb[b][idx].mean(0)
                prof = prof / (np.linalg.norm(prof) + 1e-9)
                s = float(q @ prof)
                if s > best:
                    best, best_a = s, b
            hits += (best_a == a); tot += 1
    return hits / tot, tot


def profile_attr(train, test_items):
    """Standard profile attribution: profile from up to 60 train posts."""
    authors, profs = [], []
    for a, tx in train.items():
        authors.append(a); m = embed(tx[:60]).mean(0)
        profs.append(m / (np.linalg.norm(m) + 1e-9))
    P = np.vstack(profs)
    v = embed([t for _, _, t in test_items])
    pred = [authors[i] for i in (v @ P.T).argmax(1)]
    return np.array([p == g for (g, _, _), p in zip(test_items, pred)], dtype=int)


def author_boot(pairs_by_author, nb=5000):
    """Cluster bootstrap over authors of paired (c_orig, c_rewr) message vectors."""
    authors = list(pairs_by_author.keys())
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(nb):
        pick = rng.choice(len(authors), len(authors), replace=True)
        co = np.concatenate([pairs_by_author[authors[k]][0] for k in pick])
        cr = np.concatenate([pairs_by_author[authors[k]][1] for k in pick])
        diffs.append(co.mean() - cr.mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run(name, train, test, cache_path):
    cache = {}
    for line in open(cache_path):
        r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
    orig_by = {a: list(tx) for a, tx in test.items()}
    rewr_by = {a: [cache[("heavy", a, i)] for i in range(len(tx))]
               for a, tx in test.items()}

    acc_oo, n = loo_separability(orig_by)
    acc_rr, _ = loo_separability(rewr_by)
    chance = 1 / len(test)
    print(f"[{name}] LOO orig->orig={acc_oo:.3f}  rewr->rewr={acc_rr:.3f}  "
          f"chance={chance:.3f}  (retained {acc_rr/acc_oo*100:.0f}% of separability)",
          flush=True)

    # headline IER with author-clustered CI
    items = [(a, i, t) for a, tx in test.items() for i, t in enumerate(tx)]
    c_o = profile_attr(train, items)
    rew_items = [(a, i, cache[("heavy", a, i)]) for a, i, _ in items]
    c_r = profile_attr(train, rew_items)
    pairs = {}
    for (a, _, _), co, cr in zip(items, c_o, c_r):
        pairs.setdefault(a, [[], []])
        pairs[a][0].append(co); pairs[a][1].append(cr)
    pairs = {a: (np.array(v[0]), np.array(v[1])) for a, v in pairs.items()}
    lo, hi = author_boot(pairs)
    ier = float(c_o.mean() - c_r.mean())
    print(f"[{name}] heavy IER={ier*100:+.1f}  author-clustered CI=[{lo*100:+.0f},{hi*100:+.0f}]",
          flush=True)
    return {"loo_orig": acc_oo, "loo_rewr": acc_rr, "chance": chance,
            "separability_retained": acc_rr / acc_oo,
            "heavy_IER": ier, "author_cluster_ci95": [lo, hi], "n_authors": len(test)}


res = {}
tb, te = load_blogs_rich(max_authors=50, load_posts=200, max_words=400)
res["blogs"] = run("blogs", tb, te, "results/blogs400_rewrites.jsonl")
eb, ee = load_enron_rich(max_words=400)
res["enron"] = run("enron", eb, ee, "results/enron400_rewrites.jsonl")
json.dump(res, open("results/transform.json", "w"), indent=2)
print("saved -> results/transform.json")
