"""LUAR with richer author profiles (same authors, same test messages)."""
import json
import numpy as np
import torch
from scipy import stats as sps
from transformers import AutoModel, AutoTokenizer

from src.data import load_blogs_rich

DEV = "cpu"; MODEL = "rrivera1849/LUAR-MUD"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(DEV).eval()
PROFILE_POSTS = 60


def embed(texts, bs=8, L=256):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], max_length=256, padding="max_length",
                  truncation=True, return_tensors="pt")
        ids = enc["input_ids"].unsqueeze(1).to(DEV)
        am = enc["attention_mask"].unsqueeze(1).to(DEV)
        with torch.no_grad():
            e = model(input_ids=ids, attention_mask=am)
        e = e[0] if isinstance(e, (tuple, list)) else e
        e = torch.nan_to_num(e, 0.0, 0.0, 0.0)
        out.append(np.nan_to_num(torch.nn.functional.normalize(e, dim=1).cpu().numpy()))
    return np.vstack(out)


def mcnemar(cb, ca):
    b = int(np.sum((cb == 1) & (ca == 0))); c = int(np.sum((cb == 0) & (ca == 1)))
    n = b + c
    return b, c, (1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue))


def boot(cb, ca, nb=5000):
    d = cb - ca; rng = np.random.default_rng(0); idx = np.arange(len(d))
    return tuple(float(x) for x in np.percentile(
        [d[rng.choice(idx, len(idx), True)].mean() for _ in range(nb)], [2.5, 97.5]))


train, test = load_blogs_rich(max_authors=50, load_posts=200, max_words=400)
authors, profs = [], []
for a, texts in train.items():
    authors.append(a)
    m = embed(texts[:PROFILE_POSTS]).mean(0)
    profs.append(m / (np.linalg.norm(m) + 1e-9))
P = np.vstack(profs)
print(f"profiles built from up to {PROFILE_POSTS} posts/author", flush=True)

items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
gold = [a for a, _, _ in items]; orig = [t for _, _, t in items]
cache = {}
for line in open("results/blogs400_rewrites.jsonl"):
    r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]


def correct(texts):
    v = embed(texts)
    pred = [authors[i] for i in (v @ P.T).argmax(1)]
    return np.array([p == g for p, g in zip(pred, gold)], dtype=int)


c_o = correct(orig)
res = {"profile_posts": PROFILE_POSTS, "baseline_acc": float(c_o.mean()), "conditions": {}}
print(f"LUAR baseline = {c_o.mean():.3f}", flush=True)
for cond in ["light", "heavy", "preserve"]:
    c_r = correct([cache[(cond, a, i)] for a, i, _ in items])
    b, c, p = mcnemar(c_o, c_r); lo, hi = boot(c_o, c_r)
    ier = float(c_o.mean() - c_r.mean())
    res["conditions"][cond] = {"acc_after": float(c_r.mean()), "IER": ier,
                               "ci95": [lo, hi], "p_value": p, "b": b, "c": c,
                               "frac_signal_destroyed": ier / float(c_o.mean())}
    print(f"  {cond:8} after={c_r.mean():.3f} IER={ier*100:+.1f} "
          f"CI=[{lo*100:+.0f},{hi*100:+.0f}] p={p:.3g} "
          f"({ier/float(c_o.mean())*100:.0f}% of signal)", flush=True)
json.dump(res, open("results/blogs400_luar.json", "w"), indent=2)
print("saved -> results/blogs400_luar.json")
