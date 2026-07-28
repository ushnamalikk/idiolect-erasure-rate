"""Two LUAR checks needed for the revision.

(1) Word-shuffle sensitivity on blogs: does LUAR depend on word order (style) or
    only on content words? This justifies promoting LUAR over the MiniLM
    semantic baseline as the deep attributer.
(2) LUAR on Enron with significance: the missing third point that makes the
    claim "deep erasure tracks the idiolect the text carried" testable rather
    than tautological.

Writes results/luar_checks.json.
"""
import json, random
import numpy as np
import torch
from scipy import stats as sps
from transformers import AutoModel, AutoTokenizer

from src.data import load_blogs, load_enron, split_corpus

DEV = "cpu"
MODEL = "rrivera1849/LUAR-MUD"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(DEV).eval()


def embed(texts, bs=16, L=128):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], max_length=L, padding="max_length",
                  truncation=True, return_tensors="pt")
        ids = enc["input_ids"].unsqueeze(1).to(DEV)
        am = enc["attention_mask"].unsqueeze(1).to(DEV)
        with torch.no_grad():
            e = model(input_ids=ids, attention_mask=am)
        e = e[0] if isinstance(e, (tuple, list)) else e
        e = torch.nan_to_num(e, 0.0, 0.0, 0.0)
        e = torch.nn.functional.normalize(e, dim=1)
        out.append(np.nan_to_num(e.cpu().numpy()))
    return np.vstack(out)


def build(train):
    authors, profs = [], []
    for a, texts in train.items():
        authors.append(a)
        m = embed(texts[:25]).mean(0)
        profs.append(m / (np.linalg.norm(m) + 1e-9))
    return authors, np.vstack(profs)


def correct(texts, gold, authors, P):
    v = embed(texts)
    pred = [authors[i] for i in (v @ P.T).argmax(1)]
    return np.array([p == g for p, g in zip(pred, gold)], dtype=int)


def mcnemar(cb, ca):
    b = int(np.sum((cb == 1) & (ca == 0))); c = int(np.sum((cb == 0) & (ca == 1)))
    n = b + c
    p = 1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue)
    return b, c, p


def boot(cb, ca, nb=5000):
    d = cb - ca; rng = np.random.default_rng(0); idx = np.arange(len(d))
    m = [d[rng.choice(idx, len(idx), True)].mean() for _ in range(nb)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


res = {}

# ---------- (1) shuffle sensitivity on blogs ----------
corpus = load_blogs(max_authors=50, posts_per_author=20, max_words=90)
train, test = split_corpus(corpus, n_test=4)
authors, P = build(train)
items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
gold = [a for a, _, _ in items]; orig = [t for _, _, t in items]
rng = random.Random(0)
def shuf(t):
    w = t.split(); rng.shuffle(w); return " ".join(w)
c_o = correct(orig, gold, authors, P)
c_s = correct([shuf(t) for t in orig], gold, authors, P)
res["blogs_shuffle"] = {"original": float(c_o.mean()), "shuffled": float(c_s.mean())}
print(f"[LUAR blogs] original={c_o.mean():.3f} shuffled={c_s.mean():.3f}", flush=True)

# ---------- LUAR blogs stats for all conditions ----------
cache = {}
for line in open("results/blogs_rewrites.jsonl"):
    r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
res["blogs_conditions"] = {}
for cond in ["light", "heavy", "preserve"]:
    rew = [cache[(cond, a, i)] for a, i, _ in items]
    c_r = correct(rew, gold, authors, P)
    b, c, p = mcnemar(c_o, c_r); lo, hi = boot(c_o, c_r)
    res["blogs_conditions"][cond] = {"baseline": float(c_o.mean()),
        "acc_after": float(c_r.mean()), "IER": float(c_o.mean() - c_r.mean()),
        "ci95": [lo, hi], "p_value": p, "b": b, "c": c}
    print(f"[LUAR blogs/{cond:8}] IER={c_o.mean()-c_r.mean():+.3f} "
          f"CI=[{lo:+.3f},{hi:+.3f}] p={p:.4g}", flush=True)

# ---------- (2) LUAR on Enron ----------
ecorpus = load_enron(max_authors=25, posts_per_author=20, max_words=90)
etrain, etest = split_corpus(ecorpus, n_test=4)
eauthors, EP = build(etrain)
eitems = [(a, i, t) for a, texts in etest.items() for i, t in enumerate(texts)]
egold = [a for a, _, _ in eitems]; eorig = [t for _, _, t in eitems]
ecache = {}
for line in open("results/enron_rewrites.jsonl"):
    r = json.loads(line); ecache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
ec_o = correct(eorig, egold, eauthors, EP)
res["enron_conditions"] = {"baseline": float(ec_o.mean())}
print(f"[LUAR enron] baseline={ec_o.mean():.3f}", flush=True)
for cond in ["light", "heavy", "preserve"]:
    rew = [ecache[(cond, a, i)] for a, i, _ in eitems]
    ec_r = correct(rew, egold, eauthors, EP)
    b, c, p = mcnemar(ec_o, ec_r); lo, hi = boot(ec_o, ec_r)
    res["enron_conditions"][cond] = {"acc_after": float(ec_r.mean()),
        "IER": float(ec_o.mean() - ec_r.mean()), "ci95": [lo, hi],
        "p_value": p, "b": b, "c": c}
    print(f"[LUAR enron/{cond:8}] IER={ec_o.mean()-ec_r.mean():+.3f} "
          f"CI=[{lo:+.3f},{hi:+.3f}] p={p:.4g}", flush=True)

json.dump(res, open("results/luar_checks.json", "w"), indent=2)
print("saved -> results/luar_checks.json")
