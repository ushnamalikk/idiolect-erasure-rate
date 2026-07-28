"""(#5) Author-clustered CIs for every main IER cell, and a blog-metadata ablation.

Part 1: for each corpus, surface (LinearSVC) and LUAR heavy-rewrite IER with a
cluster bootstrap that resamples AUTHORS (not messages), for all six cells.

Part 2: blog leakage ablation, mirroring the Enron one: strip modal sign-off names
and date/number tokens, recompute the blog baseline and heavy IER.

Writes results/cluster_cis.json and results/blog_ablation.json.
"""
import json, re
from collections import Counter
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import AutoModel, AutoTokenizer

from src.data import load_blogs_rich, load_enron_rich, load_c50, to_xy

ltok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
lmodel = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True).eval()


def lembed(texts, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        e = ltok(texts[i:i+bs], max_length=256, padding="max_length", truncation=True,
                 return_tensors="pt")
        with torch.no_grad():
            v = lmodel(input_ids=e["input_ids"].unsqueeze(1),
                       attention_mask=e["attention_mask"].unsqueeze(1))
        v = v[0] if isinstance(v, (tuple, list)) else v
        out.append(np.nan_to_num(torch.nn.functional.normalize(torch.nan_to_num(v), dim=1).numpy()))
    return np.vstack(out)


def surface_model():
    return Pipeline([("f", FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))])),
        ("c", LinearSVC(C=1.0, class_weight="balanced"))])


def author_boot(pairs_by_author, nb=5000):
    authors = list(pairs_by_author); rng = np.random.default_rng(0); d = []
    for _ in range(nb):
        pick = rng.choice(len(authors), len(authors), True)
        co = np.concatenate([pairs_by_author[authors[k]][0] for k in pick])
        cr = np.concatenate([pairs_by_author[authors[k]][1] for k in pick])
        d.append(co.mean() - cr.mean())
    return float(np.percentile(d, 2.5)) * 100, float(np.percentile(d, 97.5)) * 100


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


def luar_correct(train, items, texts, gold):
    authors = list(train.keys())
    P = np.vstack([(lambda m: m/(np.linalg.norm(m)+1e-9))(lembed(train[a][:60]).mean(0))
                   for a in authors])
    v = lembed(texts); pred = [authors[i] for i in (v @ P.T).argmax(1)]
    return np.array([p == g for p, g in zip(pred, gold)], dtype=int)


# ---------- Part 1: cluster CIs ----------
cis = {}
for corpus in ["blogs", "enron", "c50"]:
    train, test, cache = load(corpus)
    items = [(a, i, t) for a, tx in test.items() for i, t in enumerate(tx)]
    gold = [a for a, _, _ in items]; orig = [t for _, _, t in items]
    heavy = [cache[("heavy", a, i)] for a, i, _ in items]
    cis[corpus] = {}
    for name, co, cr in [
        ("surface", *(lambda m: (np.array([p == g for p, g in zip(m.predict(orig), gold)], int),
                                 np.array([p == g for p, g in zip(m.predict(heavy), gold)], int)))(
            surface_model().fit(*to_xy(train)))),
        ("luar", luar_correct(train, items, orig, gold), luar_correct(train, items, heavy, gold)),
    ]:
        pairs = {}
        for (a, _, _), x, y in zip(items, co, cr):
            pairs.setdefault(a, [[], []])
            pairs[a][0].append(x); pairs[a][1].append(y)
        pairs = {a: (np.array(v[0]), np.array(v[1])) for a, v in pairs.items()}
        lo, hi = author_boot(pairs)
        ier = float(co.mean() - cr.mean()) * 100
        cis[corpus][name] = {"IER": ier, "author_ci95": [lo, hi]}
        print(f"[{corpus}/{name}] heavy IER={ier:+.1f} author-CI=[{lo:+.0f},{hi:+.0f}]", flush=True)
json.dump(cis, open("results/cluster_cis.json", "w"), indent=2)

# ---------- Part 2: blog ablation ----------
CLOSINGS = r"(thanks|thank you|regards|best|sincerely|cheers|love|xoxo|peace)"
def signoff(msgs):
    last = [re.findall(r"[A-Za-z]+", m)[-1].lower() for m in msgs if re.findall(r"[A-Za-z]+", m)]
    if not last: return None
    nm, n = Counter(last).most_common(1)[0]
    return nm if n >= 3 and len(nm) >= 3 and nm not in {"the","you","and","was","have","that"} else None
def strip(t, name):
    t = re.sub(r"\b\d{1,4}([:/.-]\d{1,4})+\b", " ", t)             # dates/times
    t = re.sub(r"[\s,.;-]*" + CLOSINGS + r"[.,!\s-]*[A-Z][a-z]+\.?\s*$", ".", t, flags=re.I)
    if name: t = re.sub(r"\b" + re.escape(name) + r"\b", "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()

train, test, cache = load("blogs")
names = {a: signoff(train[a] + test[a]) for a in train}
tr_s = {a: [strip(m, names[a]) for m in v] for a, v in train.items()}
items = [(a, i) for a, tx in test.items() for i in range(len(tx))]
gold = [a for a, _ in items]
orig_s = [strip(test[a][i], names[a]) for a, i in items]
heavy_s = [strip(cache[("heavy", a, i)], names[a]) for a, i in items]
m = surface_model().fit(*to_xy(tr_s))
so = np.mean([p == g for p, g in zip(m.predict(orig_s), gold)])
sh = np.mean([p == g for p, g in zip(m.predict(heavy_s), gold)])
tr_items = [(a, i, None) for a, i in items]
lo = luar_correct(tr_s, tr_items, orig_s, gold).mean()
lh = luar_correct(tr_s, tr_items, heavy_s, gold).mean()
blog_abl = {"n_names": sum(v is not None for v in names.values()),
            "surface": {"baseline_stripped": float(so), "baseline_orig": 0.810,
                        "heavy_IER_stripped": float(so - sh)},
            "luar": {"baseline_stripped": float(lo), "baseline_orig": 0.815,
                     "heavy_IER_stripped": float(lo - lh)}}
print(f"[blog ablation] surface base 0.810->{so:.3f} IER {(so-sh)*100:+.1f} | "
      f"LUAR base 0.815->{lo:.3f} IER {(lo-lh)*100:+.1f}", flush=True)
json.dump(blog_abl, open("results/blog_ablation.json", "w"), indent=2)
print("saved -> results/cluster_cis.json, results/blog_ablation.json")
