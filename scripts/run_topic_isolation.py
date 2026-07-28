"""C50 topic-isolation (#3): decompose attribution into topic vs. style.

Mirror of the function-word (pure-style) attributer: a CONTENT-word-only attributer
(TF-IDF word n-grams with function words removed) captures topic, not style. If on
C50 the content-word attributer has a high baseline AND survives rewriting (low
IER), while the function-word attributer is erased, then topic (preserved by
rewriting) carries C50 attribution and style is erased, testing the "topic
substitutes for identity" hypothesis directly rather than by inference.

Writes results/topic_isolation.json.
"""
import json
import numpy as np
from scipy import stats as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from src.data import load_blogs_rich, load_enron_rich, load_c50, to_xy
from scripts.run_funcword import FUNCTION_WORDS  # reuse the same list

CONDS = ["heavy"]
FUNC = set(FUNCTION_WORDS)


def mcnemar(cb, ca):
    b = int(np.sum((cb == 1) & (ca == 0))); c = int(np.sum((cb == 0) & (ca == 1)))
    n = b + c
    return (1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue))


def content_model():
    # content words only: drop function words via stop_words, keep word 1-2 grams
    return Pipeline([
        ("tf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                               stop_words=list(FUNC), min_df=2, sublinear_tf=True)),
        ("c", LinearSVC(C=1.0, class_weight="balanced"))])


def load(corpus):
    if corpus == "blogs":
        train, test = load_blogs_rich(max_authors=50, load_posts=200, max_words=400)
    elif corpus == "enron":
        train, test = load_enron_rich(max_words=400)
    else:
        train = load_c50("data", "train", max_authors=25, max_words=400)
        test = load_c50("data", "test", max_authors=25, max_docs_per_author=4, max_words=400)
    cache = {}
    for line in open(f"results/{corpus}400_rewrites.jsonl"):
        r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
    return train, test, cache


res = {}
for corpus in ["blogs", "enron", "c50"]:
    train, test, cache = load(corpus)
    m = content_model(); m.fit(*to_xy(train))
    items = [(a, i, t) for a, tx in test.items() for i, t in enumerate(tx)]
    gold = [a for a, _, _ in items]; orig = [t for _, _, t in items]
    c_o = np.array([p == g for p, g in zip(m.predict(orig), gold)], dtype=int)
    rew = [cache[("heavy", a, i)] for a, i, _ in items]
    c_r = np.array([p == g for p, g in zip(m.predict(rew), gold)], dtype=int)
    ier = float(c_o.mean() - c_r.mean())
    res[corpus] = {"content_baseline": float(c_o.mean()), "content_heavy_IER": ier,
                   "p_value": mcnemar(c_o, c_r)}
    print(f"[{corpus}] CONTENT-word baseline={c_o.mean():.3f} heavy IER={ier*100:+.1f} "
          f"p={mcnemar(c_o, c_r):.3g}", flush=True)

json.dump(res, open("results/topic_isolation.json", "w"), indent=2)
print("saved -> results/topic_isolation.json")
print("(compare to function-word/style IER: blogs +23, enron +8.8 n.s., c50 +27)")
