"""Full-length LUAR (and surface) IER for a commercial assistant (Gemini).

Moves the commercial replication from surface-only/90-word to the full-length
protocol with the headline authorship model. Also tracks safety-block fallbacks
(the API returning the original text unchanged), which bias IER downward, and
reports IER with them excluded.

Resumable via results/{corpus}400_gemini_rewrites.jsonl.
Run: python3 -m scripts.run_commercial400 --corpus blogs|enron
"""
import argparse, json, os, time
import numpy as np
import torch
from scipy import stats as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import AutoModel, AutoTokenizer

from src.data import load_blogs_rich, load_enron_rich, to_xy
from src.rewriters import GeminiRewriter, OpenAIRewriter

CONDS = ["heavy", "light", "preserve"]
PROVIDERS = {
    "gemini": (GeminiRewriter, "gemini-flash-latest"),
    "openai": (OpenAIRewriter, "gpt-4o-mini"),
}


def mcnemar(cb, ca):
    b = int(np.sum((cb == 1) & (ca == 0))); c = int(np.sum((cb == 0) & (ca == 1)))
    n = b + c
    return (1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue))


ap = argparse.ArgumentParser()
ap.add_argument("--corpus", choices=["blogs", "enron"], required=True)
ap.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
a = ap.parse_args()
RW_CLS, MODEL = PROVIDERS[a.provider]
CACHE = f"results/{a.corpus}400_{a.provider}_rewrites.jsonl"
OUT = f"results/{a.corpus}400_{a.provider}_ier.json"

if a.corpus == "blogs":
    train, test = load_blogs_rich(max_authors=50, load_posts=200, max_words=400)
else:
    train, test = load_enron_rich(max_words=400)
items = [(au, i, t) for au, tx in test.items() for i, t in enumerate(tx)]
gold = [x[0] for x in items]; orig = [x[2] for x in items]
orig_by_key = {(au, i): t for au, i, t in items}

cache = {}
if os.path.exists(CACHE):
    for line in open(CACHE):
        r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
todo = [(c, au, i, t) for c in CONDS for (au, i, t) in items if (c, au, i) not in cache]
print(f"[{a.corpus}] test={len(items)} cached={len(cache)} todo={len(todo)}", flush=True)
if todo:
    rw = RW_CLS(model=MODEL)
    t0 = time.time()
    for n, (c, au, i, t) in enumerate(todo, 1):
        out = rw.rewrite([t], c)[0]
        with open(CACHE, "a") as f:
            f.write(json.dumps({"cond": c, "author": au, "idx": i, "rewrite": out}) + "\n")
        cache[(c, au, i)] = out
        if n % 25 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)} ({el/n:.1f}s/call, ~{el/n*(len(todo)-n)/60:.0f} min left)",
                  flush=True)

# surface attributer
surf = Pipeline([("f", FeatureUnion([
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
    ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))])),
    ("c", LinearSVC(C=1.0, class_weight="balanced"))])
surf.fit(*to_xy(train))

# LUAR
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
P = np.vstack([(lambda m: m/(np.linalg.norm(m)+1e-9))(lembed(train[au][:60]).mean(0))
               for au in authors])


def lpred(texts):
    v = lembed(texts); return [authors[i] for i in (v @ P.T).argmax(1)]


surf_o = np.array([p == g for p, g in zip(surf.predict(orig), gold)], dtype=int)
luar_o = np.array([p == g for p, g in zip(lpred(orig), gold)], dtype=int)

res = {"corpus": a.corpus, "model": MODEL, "n": len(items),
       "surface_baseline": float(surf_o.mean()), "luar_baseline": float(luar_o.mean()),
       "conditions": {}}
print(f"baselines: surface={surf_o.mean():.3f} luar={luar_o.mean():.3f}", flush=True)
for cond in CONDS:
    rew = [cache[(cond, au, i)] for au, i, _ in items]
    # safety-block fallbacks: rewrite identical to original
    fb = [j for j, (au, i, _) in enumerate(items)
          if cache[(cond, au, i)].strip() == orig_by_key[(au, i)].strip()]
    keep = [j for j in range(len(items)) if j not in fb]
    surf_r = np.array([p == g for p, g in zip(surf.predict(rew), gold)], dtype=int)
    luar_r = np.array([p == g for p, g in zip(lpred(rew), gold)], dtype=int)
    def block(o, r, idx):
        oo, rr = o[idx], r[idx]
        return {"IER": float(oo.mean() - rr.mean()), "p_value": mcnemar(oo, rr),
                "acc_after": float(rr.mean())}
    res["conditions"][cond] = {
        "n_fallback": len(fb),
        "surface": block(surf_o, surf_r, np.arange(len(items))),
        "luar": block(luar_o, luar_r, np.arange(len(items))),
        "surface_excl_fallback": block(surf_o, surf_r, np.array(keep)),
        "luar_excl_fallback": block(luar_o, luar_r, np.array(keep)),
    }
    c = res["conditions"][cond]
    print(f"  {cond:8} fallbacks={len(fb)} | surface IER={c['surface']['IER']*100:+.1f} "
          f"| LUAR IER={c['luar']['IER']*100:+.1f} "
          f"(excl fb {c['luar_excl_fallback']['IER']*100:+.1f})", flush=True)

json.dump(res, open(OUT, "w"), indent=2)
print("saved ->", OUT)
