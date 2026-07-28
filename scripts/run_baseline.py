"""Baseline authorship attribution on ORIGINAL (human) text.

This is the 'before' number in IER = accuracy(before) - accuracy(after).
Trains each attributer on C50 train, evaluates on C50 test originals.

Run:  python3 -m scripts.run_baseline --neural
"""
import argparse
import json
import os
import time

from src.data import load_c50, to_xy
from src.attributers import StylometricAttributer, NeuralAttributer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--max_authors", type=int, default=50)
    ap.add_argument("--max_docs", type=int, default=50)
    ap.add_argument("--max_words", type=int, default=None,
                    help="truncate docs to N words (simulate message length)")
    ap.add_argument("--neural", action="store_true")
    args = ap.parse_args()

    train = load_c50(args.root, "train", args.max_authors, args.max_docs, args.max_words)
    test = load_c50(args.root, "test", args.max_authors, args.max_docs, args.max_words)
    Xte, yte = to_xy(test)
    n_authors = len(train)
    print(f"authors={n_authors}  train_docs={sum(len(v) for v in train.values())}  "
          f"test_docs={len(Xte)}  chance={1.0/n_authors:.3f}")

    results = {"n_authors": n_authors, "n_test": len(Xte),
               "chance": 1.0 / n_authors, "max_words": args.max_words}

    t = time.time()
    sty = StylometricAttributer().fit(train)
    acc = sty.score(Xte, yte)
    results["stylometric_acc"] = acc
    print(f"[stylometric] accuracy on originals = {acc:.3f}  ({time.time()-t:.1f}s)")

    if args.neural:
        t = time.time()
        neu = NeuralAttributer().fit(train)
        acc = neu.score(Xte, yte)
        results["neural_acc"] = acc
        print(f"[neural]      accuracy on originals = {acc:.3f}  ({time.time()-t:.1f}s)")

    os.makedirs("results", exist_ok=True)
    with open("results/baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved -> results/baseline.json")


if __name__ == "__main__":
    main()
