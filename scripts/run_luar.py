"""Re-score IER with LUAR (a purpose-built authorship-representation model).

On C50 the deep null was a topic confound and LUAR was out-of-domain (news).
On BLOGS -- informal personal writing, closer to LUAR's social-media training --
LUAR is a fair SOTA 'deep fingerprint' test. If LUAR IER is positive on blogs,
the deep-erasure result is corroborated by a specialised author model.

Run:  python3 -m scripts.run_luar --corpus blogs
"""
import argparse, json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.data import load_c50, load_blogs, split_corpus

DEV = "cpu"   # LUAR's einops pooling is not MPS-friendly
MODEL = "rrivera1849/LUAR-MUD"

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", choices=["c50", "blogs"], default="blogs")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(DEV).eval()


def embed(texts, bs=16, L=128):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], max_length=L, padding="max_length",
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


if args.corpus == "c50":
    train = load_c50("data", "train", max_docs_per_author=25, max_words=90)
    test = load_c50("data", "test", max_authors=25, max_docs_per_author=4, max_words=90)
    items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
    cache_file = "results/rewrites.jsonl"
else:
    corpus = load_blogs(max_authors=50, posts_per_author=20, max_words=90)
    train, test = split_corpus(corpus, n_test=4)
    items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
    cache_file = "results/blogs_rewrites.jsonl"

authors, profs = [], []
for a, texts in train.items():
    authors.append(a)
    m = embed(texts[:25]).mean(0)
    profs.append(m / (np.linalg.norm(m) + 1e-9))
P = np.vstack(profs)


def predict(texts):
    v = embed(texts)
    return [authors[i] for i in (v @ P.T).argmax(1)]


gold = [a for a, _, _ in items]
cache = {}
for line in open(cache_file):
    r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]

acc = lambda txt: float(np.mean([p == g for p, g in zip(predict(txt), gold)]))
base = acc([t for _, _, t in items])
res = {"corpus": args.corpus, "model": MODEL, "baseline_acc": base, "conditions": {}}
print(f"[LUAR/{args.corpus}] baseline acc = {base:.3f}")
for c in ["light", "heavy", "preserve"]:
    a = acc([cache[(c, au, i)] for au, i, _ in items])
    res["conditions"][c] = {"acc_after": a, "IER": base - a}
    print(f"[LUAR/{args.corpus}] {c:8} acc_after={a:.3f}  IER={base-a:+.3f}")
json.dump(res, open(f"results/luar_{args.corpus}.json", "w"), indent=2)
print(f"saved -> results/luar_{args.corpus}.json")
