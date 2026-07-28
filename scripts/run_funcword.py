"""Function-word-only (pure-style) attributer: a third column for IER.

Function words (articles, pronouns, prepositions, auxiliaries, conjunctions) plus
punctuation carry authorship but almost no topic. If IER persists on these
features alone, erasure is of involuntary style, not content, which is the exact
ambiguity the topic-confound discussion raises. Restricting the vocabulary to a
fixed function-word list makes the attributer content-blind by construction.

Writes results/funcword.json.
"""
import json
import numpy as np
from scipy import stats as sps
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

from src.data import load_blogs_rich, load_enron_rich, load_c50, to_xy

FUNCTION_WORDS = ("a an the this that these those my your his her its our their "
    "i you he she it we they me him us them mine yours hers ours theirs myself "
    "yourself himself herself itself ourselves themselves who whom whose which what "
    "and or but nor so yet for as if then than because although though while whereas "
    "unless until whether since when where why how of in on at by to from with about "
    "against between into through during before after above below up down out off over "
    "under again further once here there all any both each few more most other some "
    "such no not only own same too very can will just should now do does did doing done "
    "have has had having be been being am is are was were would could may might must shall "
    "into onto upon within without along across behind beyond plus versus per").split()
PUNCT = list(".,;:!?'\"()-")
CONDS = ["light", "heavy", "preserve"]


def mcnemar(cb, ca):
    b = int(np.sum((cb == 1) & (ca == 0))); c = int(np.sum((cb == 0) & (ca == 1)))
    n = b + c
    return (1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue))


def build():
    # relative frequency of a fixed function-word vocabulary + punctuation marks,
    # L1-normalized so it is content-blind and length-invariant.
    vocab = list(dict.fromkeys(FUNCTION_WORDS + PUNCT))  # dedupe, keep order
    return Pipeline([
        ("cv", CountVectorizer(vocabulary=vocab, lowercase=True,
                               token_pattern=r"(?u)\b\w+\b|[.,;:!?'\"()\-]")),
        ("nz", Normalizer(norm="l1")),
        ("c", LinearSVC(C=1.0, class_weight="balanced"))])


def load(corpus):
    if corpus == "blogs":
        train, test = load_blogs_rich(max_authors=50, load_posts=200, max_words=400)
    elif corpus == "enron":
        train, test = load_enron_rich(max_words=400)
    else:
        train = load_c50("data", "train", max_authors=25, max_words=400)
        test = load_c50("data", "test", max_authors=25, max_docs_per_author=4,
                        max_words=400)
    cache = {}
    for line in open(f"results/{corpus}400_rewrites.jsonl"):
        r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
    return train, test, cache


res = {}
for corpus in ["blogs", "enron", "c50"]:
    train, test, cache = load(corpus)
    Xtr, ytr = to_xy(train)
    m = build(); m.fit(Xtr, ytr)
    items = [(a, i, t) for a, tx in test.items() for i, t in enumerate(tx)]
    gold = [a for a, _, _ in items]; orig = [t for _, _, t in items]
    c_o = np.array([p == g for p, g in zip(m.predict(orig), gold)], dtype=int)
    res[corpus] = {"baseline_acc": float(c_o.mean()), "chance": 1/len(train),
                   "conditions": {}}
    print(f"[{corpus}] func-word baseline={c_o.mean():.3f} (chance {1/len(train):.3f})",
          flush=True)
    for cond in CONDS:
        rew = [cache[(cond, a, i)] for a, i, _ in items]
        c_r = np.array([p == g for p, g in zip(m.predict(rew), gold)], dtype=int)
        ier = float(c_o.mean() - c_r.mean()); p = mcnemar(c_o, c_r)
        res[corpus]["conditions"][cond] = {"acc_after": float(c_r.mean()),
                                           "IER": ier, "p_value": p}
        print(f"  {cond:8} after={c_r.mean():.3f} IER={ier*100:+.1f} p={p:.3g}",
              flush=True)

json.dump(res, open("results/funcword.json", "w"), indent=2)
print("saved -> results/funcword.json")
