"""Population accumulation with LUAR (the headline instrument), full length.

Re-identify an author from a pooled set of k of their messages (mean LUAR
embedding -> nearest author profile), for originals vs heavy-rewritten messages.
Fixes the transparency issue that the surface population curve used a different
attributer. Embeddings are precomputed once, so sampling is instant.

Writes results/population_luar.json.
"""
import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.data import load_blogs_rich

L, PROF = 256, 60
KS = [1, 2, 3, 5, 8, 12, 15]
REPS = 60
tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True).eval()


def embed(texts, bs=16):
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


train, test = load_blogs_rich(max_authors=20, load_posts=200, test_slice=(16, 31),
                              max_words=400)
authors = list(train.keys())
# author profiles from training posts
P = np.vstack([(lambda m: m/(np.linalg.norm(m)+1e-9))(embed(train[a][:PROF]).mean(0))
               for a in authors])

cache = {}
for line in open("results/pop400_rewrites.jsonl"):
    r = json.loads(line); cache[(r["author"], r["idx"])] = r["rewrite"]

# precompute per-author message embeddings (orig and assisted)
emb_o, emb_a = {}, {}
for a in authors:
    msgs = test[a]
    emb_o[a] = embed(msgs)
    emb_a[a] = embed([cache[(a, i)] for i in range(len(msgs))])

rng = np.random.default_rng(0)


def curve(emb_by_author):
    accs = []
    for k in KS:
        hit = tot = 0
        for ai, a in enumerate(authors):
            E = emb_by_author[a]; n = len(E)
            if n < k:
                continue
            for _ in range(REPS):
                sel = rng.choice(n, k, replace=False)
                pooled = E[sel].mean(0)
                pooled = pooled / (np.linalg.norm(pooled) + 1e-9)
                pred = int((pooled @ P.T).argmax())
                hit += (pred == ai); tot += 1
        accs.append(hit / tot)
    return accs


res = {"KS": KS, "orig": curve(emb_o), "assist": curve(emb_a),
       "n_authors": len(authors)}
for k, o, a in zip(KS, res["orig"], res["assist"]):
    print(f"k={k:2d} orig={o:.3f} assist={a:.3f}", flush=True)
json.dump(res, open("results/population_luar.json", "w"), indent=2)
print("saved -> results/population_luar.json")
