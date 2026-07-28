"""LUAR (deep attributer) on the full-length v2 protocol, for enron or c50."""
import argparse, json
import numpy as np, torch
from scipy import stats as sps
from transformers import AutoModel, AutoTokenizer
from src.data import load_enron_rich, load_c50

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", choices=["enron", "c50"], required=True)
a = ap.parse_args()
MAXW, L, PROF = 400, 256, 60

tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True).eval()


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


def mcnemar(cb, ca):
    b = int(np.sum((cb == 1) & (ca == 0))); c = int(np.sum((cb == 0) & (ca == 1)))
    n = b + c
    return (1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue))


def boot(cb, ca, nb=5000):
    d = cb - ca; rng = np.random.default_rng(0); idx = np.arange(len(d))
    return tuple(float(x) for x in np.percentile(
        [d[rng.choice(idx, len(idx), True)].mean() for _ in range(nb)], [2.5, 97.5]))


if a.corpus == "enron":
    train, test = load_enron_rich(max_words=MAXW)
else:
    train = load_c50("data", "train", max_authors=25, max_words=MAXW)
    test = load_c50("data", "test", max_authors=25, max_docs_per_author=4, max_words=MAXW)

authors, profs = [], []
for au, tx in train.items():
    authors.append(au); m = embed(tx[:PROF]).mean(0)
    profs.append(m / (np.linalg.norm(m) + 1e-9))
P = np.vstack(profs)
items = [(au, i, t) for au, tx in test.items() for i, t in enumerate(tx)]
gold = [x[0] for x in items]; orig = [x[2] for x in items]
cache = {}
for line in open(f"results/{a.corpus}400_rewrites.jsonl"):
    r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]


def corr(texts):
    v = embed(texts); pred = [authors[i] for i in (v @ P.T).argmax(1)]
    return np.array([p == g for p, g in zip(pred, gold)], dtype=int)


c_o = corr(orig)
res = {"corpus": a.corpus, "n_authors": len(authors), "baseline_acc": float(c_o.mean()),
       "conditions": {}}
print(f"[LUAR {a.corpus}] authors={len(authors)} baseline={c_o.mean():.3f}", flush=True)
for cond in ["light", "heavy", "preserve"]:
    c_r = corr([cache[(cond, au, i)] for au, i, _ in items])
    p = mcnemar(c_o, c_r); lo, hi = boot(c_o, c_r)
    ier = float(c_o.mean() - c_r.mean())
    res["conditions"][cond] = {"acc_after": float(c_r.mean()), "IER": ier,
                               "ci95": [lo, hi], "p_value": p,
                               "frac_signal_destroyed": ier / float(c_o.mean())}
    print(f"  {cond:8} after={c_r.mean():.3f} IER={ier*100:+.1f} p={p:.3g} "
          f"({ier/float(c_o.mean())*100:.0f}% of signal)", flush=True)
json.dump(res, open(f"results/{a.corpus}400_luar.json", "w"), indent=2)
print(f"saved -> results/{a.corpus}400_luar.json")
