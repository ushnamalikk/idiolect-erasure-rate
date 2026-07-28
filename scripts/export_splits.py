"""Export the exact author lists and train/test splits used in the paper.

Writes one JSON per corpus to splits/. Each file lists the author IDs (in load
order), the held-out test indices, and the loader call that reproduces the split.
This lets others reproduce the identical evaluation without redistributing the
corpora themselves.
"""
import json, os
from src.data import load_blogs_rich, load_enron_rich, load_c50

os.makedirs("splits", exist_ok=True)


def dump(name, train, test, test_desc, loader):
    authors = list(train.keys())
    rec = {
        "corpus": name,
        "n_authors": len(authors),
        "authors": authors,
        "n_train_per_author": {a: len(train[a]) for a in authors},
        "n_test_per_author": {a: len(test[a]) for a in authors},
        "held_out": test_desc,
        "max_words": 400,
        "loader": loader,
        "seed": 0,
    }
    with open(f"splits/{name}_split.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"splits/{name}_split.json  ({len(authors)} authors, "
          f"{sum(len(v) for v in test.values())} test messages)")


tr, te = load_blogs_rich(max_authors=50, load_posts=200, max_words=400)
dump("blogs", tr, te, "posts[16:20] of each author's first 200 qualifying posts",
     "load_blogs_rich(max_authors=50, load_posts=200, max_words=400)")

tr, te = load_enron_rich(max_words=400)
dump("enron", tr, te, "posts[16:20] of each user's sent messages",
     "load_enron_rich(max_words=400)")

tr = load_c50("data", "train", max_authors=25, max_words=400)
te = load_c50("data", "test", max_authors=25, max_docs_per_author=4, max_words=400)
dump("c50", tr, te, "first 4 documents per author in the C50 test split",
     "load_c50(split='train'/'test', max_authors=25, max_docs_per_author=4, max_words=400)")
print("done")
