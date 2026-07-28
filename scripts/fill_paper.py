"""Inject real numbers from results/*.json into the paper draft.

Reads paper/ier_paper.md, replaces every '<...>'-style placeholder with the
measured value, writes paper/ier_paper_filled.md. Idempotent.
"""
import json

base = json.load(open("results/baseline.json"))
ier = json.load(open("results/ier.json"))
C = ier["conditions"]


def pp(x):      # percentage points, signed
    return f"{x*100:+.1f} pts"


def pct(x):
    return f"{x*100:.1f}%"


def drift(c):
    return f"{C[c]['drift_toward_mean']:+.3f}"


ier_vals = [C[c][k] for c in C for k in ("IER_stylometric", "IER_neural")]
repl = {
    "base_sty": pct(base["stylometric_acc"]),
    "base_neu": pct(base.get("neural_acc", ier["baseline_subset_acc"]["neural"])),
    "n_docs": str(ier["n_docs"]),
    "n_authors": str(ier["n_authors"]),
    "ier_max": pp(max(ier_vals)).replace("+", ""),
    "ier_light_sty": pp(C["light"]["IER_stylometric"]),
    "ier_light_neu": pp(C["light"]["IER_neural"]),
    "ier_heavy_sty": pp(C["heavy"]["IER_stylometric"]),
    "ier_heavy_neu": pp(C["heavy"]["IER_neural"]),
    "ier_preserve_sty": pp(C["preserve"]["IER_stylometric"]),
    "ier_preserve_neu": pp(C["preserve"]["IER_neural"]),
    "drift_light": drift("light"),
    "drift_heavy": drift("heavy"),
    "drift_preserve": drift("preserve"),
}

text = open("paper/ier_paper.md").read()
for k, v in repl.items():
    text = text.replace(f"⟨{k}⟩", v)   # <k>
out = "paper/ier_paper_filled.md"
open(out, "w").write(text)
print("filled placeholders:")
for k, v in repl.items():
    print(f"  {k:20} -> {v}")
print("saved ->", out)
