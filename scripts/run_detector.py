"""Detector side of 'double erasure': are assisted messages flagged as AI?

Double erasure claims assisted text is BOTH unattributable to its human author
(shown via IER) AND hard to flag as AI. Here we run an off-the-shelf AI-text
detector on the blog originals and their light/heavy rewrites and report the
AI-flag rate. Caveat (stated in the paper): consumer AI detectors are unreliable
-- we report the pattern, not a precise rate.
"""
import json
import numpy as np
from transformers import pipeline

from src.data import load_blogs, split_corpus

clf = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta",
               device=-1, truncation=True, max_length=256)

corpus = load_blogs(max_authors=50, posts_per_author=20, max_words=90)
_, test = split_corpus(corpus, n_test=4)
items = [(a, i, t) for a, texts in test.items() for i, t in enumerate(texts)]
cache = {}
for line in open("results/blogs_rewrites.jsonl"):
    r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]


def ai_rate(texts):
    """Fraction scored as AI/ChatGPT (label contains 'chatgpt'/'fake'/'ai' or LABEL_1)."""
    flags = []
    for r in clf([t[:2000] for t in texts], batch_size=16):
        lab = str(r["label"]).lower()
        is_ai = ("chatgpt" in lab or "fake" in lab or lab.endswith("1"))
        flags.append(1 if is_ai else 0)
    return float(np.mean(flags))


orig = [t for _, _, t in items]
res = {"detector": "Hello-SimpleAI/chatgpt-detector-roberta", "n": len(items),
       "ai_flag_rate": {"original": ai_rate(orig)}}
for cond in ["light", "heavy", "preserve"]:
    rew = [cache[(cond, a, i)] for a, i, _ in items]
    res["ai_flag_rate"][cond] = ai_rate(rew)

for k, v in res["ai_flag_rate"].items():
    print(f"  {k:9} AI-flag rate = {v:.2f}")
json.dump(res, open("results/detector.json", "w"), indent=2)
print("saved -> results/detector.json")
