"""Enron leakage ablation: is the 0.938 stylometric baseline real idiolect, or
signature/name leakage?

Enron messages end with the author's own sign-off name (e.g. "Thanks. Lynn" for
author blair-l). An attributer could key on that rather than on style. We strip,
per author, (a) their modal sign-off name wherever it appears, (b) common closings,
and (c) quoted-reply / forwarded blocks and headers, then recompute the baseline
and heavy-rewrite IER with the same surface and LUAR attributers. If the baseline
holds, the signal is idiolect; if it drops, part of it was leakage.

Writes results/enron_ablation.json.
"""
import json, re
from collections import Counter
import numpy as np
import torch
from scipy import stats as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from transformers import AutoModel, AutoTokenizer

from src.data import load_enron_rich, to_xy

CLOSINGS = (r"(thanks|thank you|thankyou|regards|best regards|best|sincerely|"
            r"cheers|talk to you soon|talk to you|take care|later)")


def signoff_name(msgs):
    """Modal trailing alphabetic token across an author's messages = sign-off."""
    last = []
    for m in msgs:
        toks = re.findall(r"[A-Za-z]+", m)
        if toks:
            last.append(toks[-1].lower())
    if not last:
        return None
    name, n = Counter(last).most_common(1)[0]
    # only treat as a name if it recurs and is not a common word
    common = {"you", "me", "it", "well", "now", "today", "please", "know",
              "soon", "time", "day", "this", "that", "am", "pm"}
    return name if (n >= 2 and name not in common and len(name) >= 3) else None


def strip_artifacts(text, name):
    t = text
    # quoted / forwarded / header blocks
    t = re.split(r"-{2,}\s*original message|-{2,}\s*forwarded|^\s*from:\s",
                 t, flags=re.IGNORECASE | re.MULTILINE)[0]
    t = re.sub(r"^\s*>.*$", " ", t, flags=re.MULTILINE)
    # trailing closing (+ optional following name)
    t = re.sub(r"[\s,.;-]*" + CLOSINGS + r"[.,!\s-]*[A-Z][a-z]+\.?\s*$", ".",
               t, flags=re.IGNORECASE)
    t = re.sub(r"[\s,.;-]*" + CLOSINGS + r"[.,!]*\s*$", ".", t, flags=re.IGNORECASE)
    # the author's own sign-off name, anywhere
    if name:
        t = re.sub(r"\b" + re.escape(name) + r"\b", "", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def mcnemar(cb, ca):
    b = int(np.sum((cb == 1) & (ca == 0))); c = int(np.sum((cb == 0) & (ca == 1)))
    n = b + c
    return (1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5).pvalue))


train, test = load_enron_rich(max_words=400)
cache = {}
for line in open("results/enron400_rewrites.jsonl"):
    r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]

names = {a: signoff_name(train[a] + test[a]) for a in train}
print("authors with detected sign-off name:",
      sum(v is not None for v in names.values()), "/", len(names), flush=True)

def strip_map(d):
    return {a: [strip_artifacts(m, names[a]) for m in msgs] for a, msgs in d.items()}

train_s = strip_map(train)
test_s = strip_map(test)
items = [(a, i) for a, tx in test.items() for i in range(len(tx))]
gold = [a for a, _ in items]
orig_s = [test_s[a][i] for a, i in items]
heavy_s = [strip_artifacts(cache[("heavy", a, i)], names[a]) for a, i in items]

# surface
surf = Pipeline([("f", FeatureUnion([
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)),
    ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))])),
    ("c", LinearSVC(C=1.0, class_weight="balanced"))])
surf.fit(*to_xy(train_s))
so = np.array([p == g for p, g in zip(surf.predict(orig_s), gold)], dtype=int)
sh = np.array([p == g for p, g in zip(surf.predict(heavy_s), gold)], dtype=int)

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
authors = list(train_s.keys())
P = np.vstack([(lambda m: m/(np.linalg.norm(m)+1e-9))(lembed(train_s[a][:60]).mean(0))
               for a in authors])
def lpred(texts):
    v = lembed(texts); return [authors[i] for i in (v @ P.T).argmax(1)]
lo = np.array([p == g for p, g in zip(lpred(orig_s), gold)], dtype=int)
lh = np.array([p == g for p, g in zip(lpred(heavy_s), gold)], dtype=int)

res = {
    "surface": {"baseline_stripped": float(so.mean()), "baseline_original": 0.938,
                "heavy_IER_stripped": float(so.mean() - sh.mean()),
                "heavy_IER_original": 0.287, "p_value": mcnemar(so, sh)},
    "luar": {"baseline_stripped": float(lo.mean()), "baseline_original": 0.713,
             "heavy_IER_stripped": float(lo.mean() - lh.mean()),
             "heavy_IER_original": 0.525, "p_value": mcnemar(lo, lh)},
}
print(f"SURFACE baseline {0.938:.3f} -> {so.mean():.3f} (stripped); "
      f"heavy IER {28.7:.1f} -> {(so.mean()-sh.mean())*100:+.1f}", flush=True)
print(f"LUAR    baseline {0.713:.3f} -> {lo.mean():.3f} (stripped); "
      f"heavy IER {52.5:.1f} -> {(lo.mean()-lh.mean())*100:+.1f}", flush=True)
json.dump(res, open("results/enron_ablation.json", "w"), indent=2)
print("saved -> results/enron_ablation.json")
