"""Detector robustness for the 'double erasure' claim: two detectors + a positive
control, on the full-length blogs protocol.

Addresses the 'single classifier' and 'does the detector detect anything at this
length' objections:
  - two off-the-shelf AI-text detectors, flag rates on original vs assisted;
  - a positive control: fully machine-generated (from-scratch) short posts, to
    show each detector's true-positive rate on known-AI text of the same register.

Writes results/detector2.json.
"""
import json
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from src.data import load_blogs_rich

DETS = {
    "chatgpt-detector-roberta": "Hello-SimpleAI/chatgpt-detector-roberta",
    "roberta-base-openai-detector": "openai-community/roberta-base-openai-detector",
}


def make_rate_fn(model_id):
    clf = pipeline("text-classification", model=model_id, truncation=True, max_length=512)
    id2 = clf.model.config.id2label
    # figure out which label index means "AI/fake/generated"
    ai_ids = {k for k, v in id2.items() if any(t in str(v).lower()
              for t in ("fake", "chatgpt", "ai", "generated", "machine"))}

    def rate(texts):
        flags = []
        for r in clf(texts, batch_size=16):
            lab = str(r["label"]).lower()
            is_ai = ("fake" in lab or "chatgpt" in lab or "generated" in lab
                     or lab.endswith("1"))
            flags.append(1 if is_ai else 0)
        return float(np.mean(flags))
    return rate


# ---- data: full-length blogs ----
_, test = load_blogs_rich(max_authors=50, load_posts=200, max_words=400)
items = [(a, i, t) for a, tx in test.items() for i, t in enumerate(tx)]
orig = [t for _, _, t in items]
cache = {}
for line in open("results/blogs400_rewrites.jsonl"):
    r = json.loads(line); cache[(r["cond"], r["author"], r["idx"])] = r["rewrite"]
conds = {c: [cache[(c, a, i)] for a, i, _ in items] for c in ["light", "heavy", "preserve"]}

# ---- positive control: fully machine-generated posts ----
TOPICS = ["my weekend", "a movie I saw", "my new job", "the weather lately",
          "a book I'm reading", "my morning routine", "a trip I took",
          "my favorite meal", "something that annoyed me", "a goal for this year",
          "my pet", "a concert", "learning to cook", "my commute", "a bad day",
          "my hometown", "a gift I received", "my hobby", "a conversation with a friend",
          "the news today", "my apartment", "a new song", "getting older",
          "a decision I made", "my family", "a walk in the park", "my phone",
          "a mistake I made", "the future", "my neighborhood"]
print("generating positive-control texts...", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
gen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                                 torch_dtype=torch.float32)
gen = []
for topic in TOPICS:
    msgs = [{"role": "user",
             "content": f"Write a short personal blog post (about 100 words) about {topic}."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ins = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = gen_model.generate(**ins, max_new_tokens=180, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    gen.append(tok.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True).strip())

res = {"n": len(items), "n_positive_control": len(gen), "detectors": {}}
for name, mid in DETS.items():
    rate = make_rate_fn(mid)
    res["detectors"][name] = {
        "original": rate(orig),
        "light": rate(conds["light"]), "heavy": rate(conds["heavy"]),
        "preserve": rate(conds["preserve"]),
        "positive_control_fully_AI": rate(gen),
    }
    d = res["detectors"][name]
    print(f"[{name}] orig={d['original']:.2f} light={d['light']:.2f} "
          f"heavy={d['heavy']:.2f} preserve={d['preserve']:.2f} "
          f"| fully-AI TPR={d['positive_control_fully_AI']:.2f}", flush=True)

json.dump(res, open("results/detector2.json", "w"), indent=2)
print("saved -> results/detector2.json")
