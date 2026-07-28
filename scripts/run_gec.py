"""Grammar-correction (GEC) baseline: is the erasure specific to generative
rewriting, or would any correction do it?

We pass originals through a task-specific grammar-correction model (targeted
minimal edits, not an instructed chat rewrite) and compute IER. If GEC erases far
less than the LLM light/heavy conditions, the effect is tied to generative
rewriting rather than mere correction. (A rule-based tool such as LanguageTool
would be cleaner but needs a Java runtime unavailable here; we label this a
task-specific neural GEC comparator.)

Resumable via results/gec_rewrites.jsonl. Writes results/gec.json.
"""
import json, os
import numpy as np
import torch
from scipy import stats as sps
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import load_blogs_rich, to_xy

GEC_MODEL = "vennify/t5-base-grammar-correction"
CACHE = "results/gec_rewrites.jsonl"


def mcnemar(cb, ca):
    b = int(np.sum((cb == 1) & (ca == 0))); c = int(np.sum((cb == 0) & (ca == 1)))
    n = b + c
    return (1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue))


train, test = load_blogs_rich(max_authors=50, load_posts=200, max_words=400)
items = [(a, i, t) for a, tx in test.items() for i, t in enumerate(tx)]
gold = [a for a, _, _ in items]; orig = [t for _, _, t in items]

# ---- GEC correction (cached) ----
cache = {}
if os.path.exists(CACHE):
    for line in open(CACHE):
        r = json.loads(line); cache[(r["author"], r["idx"])] = r["gec"]
todo = [(a, i, t) for a, i, t in items if (a, i) not in cache]
print(f"GEC cached={len(cache)} todo={len(todo)}", flush=True)
if todo:
    tok = AutoTokenizer.from_pretrained(GEC_MODEL)
    m = AutoModelForSeq2SeqLM.from_pretrained(GEC_MODEL, torch_dtype=torch.float32)
    for n, (a, i, t) in enumerate(todo, 1):
        ins = tok("grammar: " + t, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = m.generate(**ins, max_length=512, num_beams=1)
        g = tok.decode(out[0], skip_special_tokens=True).strip()
        with open(CACHE, "a") as f:
            f.write(json.dumps({"author": a, "idx": i, "gec": g}) + "\n")
        cache[(a, i)] = g
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}", flush=True)

gec = [cache[(a, i)] for a, i, _ in items]

# ---- surface IER under GEC ----
surf = Pipeline([("f", FeatureUnion([
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
    ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))])),
    ("c", LinearSVC(C=1.0, class_weight="balanced"))])
surf.fit(*to_xy(train))
c_o = np.array([p == g for p, g in zip(surf.predict(orig), gold)], dtype=int)
c_g = np.array([p == g for p, g in zip(surf.predict(gec), gold)], dtype=int)
surf_ier = float(c_o.mean() - c_g.mean()); surf_p = mcnemar(c_o, c_g)

# ---- LUAR IER under GEC ----
from transformers import AutoModel
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


authors = list(train.keys())
P = np.vstack([(lambda m: m/(np.linalg.norm(m)+1e-9))(lembed(train[a][:60]).mean(0))
               for a in authors])


def lacc(texts):
    v = lembed(texts); pred = [authors[i] for i in (v @ P.T).argmax(1)]
    return np.array([p == g for p, g in zip(pred, gold)], dtype=int)


lo = lacc(orig); lg = lacc(gec)
luar_ier = float(lo.mean() - lg.mean()); luar_p = mcnemar(lo, lg)

res = {"model": GEC_MODEL,
       "surface": {"baseline": float(c_o.mean()), "acc_after": float(c_g.mean()),
                   "IER": surf_ier, "p_value": surf_p},
       "luar": {"baseline": float(lo.mean()), "acc_after": float(lg.mean()),
                "IER": luar_ier, "p_value": luar_p}}
print(f"GEC surface IER={surf_ier*100:+.1f} (p={surf_p:.3g}) vs LLM light +5.0 heavy +38.5",
      flush=True)
print(f"GEC LUAR    IER={luar_ier*100:+.1f} (p={luar_p:.3g}) vs LLM light +30.5 heavy +66.5",
      flush=True)
json.dump(res, open("results/gec.json", "w"), indent=2)
print("saved -> results/gec.json")
