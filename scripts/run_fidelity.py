"""Content-preservation / rewrite-fidelity per condition and corpus.

The point: if a heavy rewrite keeps high semantic content similarity to the
original yet collapses authorship attribution, the lost signal is stylistic, not
a content artifact (hallucination or regeneration). Reports, per corpus x
condition:
  - semantic cosine (MiniLM, original vs rewrite)
  - length ratio (rewrite words / original words)
  - token-change fraction (1 - difflib token similarity)

Writes results/fidelity.json.
"""
import json
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data import load_blogs_rich, load_enron_rich, load_c50

CONDS = ["light", "heavy", "preserve"]
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def load(corpus):
    if corpus == "blogs":
        _, test = load_blogs_rich(max_authors=50, load_posts=200, max_words=400)
    elif corpus == "enron":
        _, test = load_enron_rich(max_words=400)
    else:
        test = load_c50("data", "test", max_authors=25, max_docs_per_author=4,
                        max_words=400)
    cache = {}
    for line in open(f"results/{corpus}400_rewrites.jsonl"):
        r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
    items = [(a, i, t) for a, tx in test.items() for i, t in enumerate(tx)]
    return items, cache


def cosines(a_texts, b_texts):
    ea = enc.encode(a_texts, convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False)
    eb = enc.encode(b_texts, convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False)
    return (ea * eb).sum(1)


def token_change(a, b):
    ta, tb = a.split(), b.split()
    return 1.0 - SequenceMatcher(None, ta, tb).ratio()


res = {}
for corpus in ["blogs", "enron", "c50"]:
    items, cache = load(corpus)
    orig = [t for _, _, t in items]
    res[corpus] = {}
    for cond in CONDS:
        rew = [cache[(cond, a, i)] for a, i, _ in items]
        cos = cosines(orig, rew)
        lr = np.array([len(r.split()) / max(1, len(o.split()))
                       for o, r in zip(orig, rew)])
        tc = np.array([token_change(o, r) for o, r in zip(orig, rew)])
        res[corpus][cond] = {"cosine": float(cos.mean()),
                             "cosine_sd": float(cos.std()),
                             "length_ratio": float(lr.mean()),
                             "token_change": float(tc.mean())}
        print(f"{corpus:6}/{cond:8} cosine={cos.mean():.3f} "
              f"len_ratio={lr.mean():.2f} token_change={tc.mean():.2f}", flush=True)

json.dump(res, open("results/fidelity.json", "w"), indent=2)
print("saved -> results/fidelity.json")
