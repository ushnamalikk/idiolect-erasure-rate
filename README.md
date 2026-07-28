# The Idiolect Erasure Rate (IER)

Open protocol and code for the paper **"The Assistant Erased You: Measuring Loss of
Authorship Signals in AI-Mediated Communication"** (Ushna Malik and Moiz Sadiq Awan).

IER is the reduction in authorship-attribution accuracy caused by passing a message
through an AI writing assistant. This repository is the reproducible protocol behind
every result in the paper: the attributers, the rewriting conditions, the analysis
scripts, and the released prompts, seeds, model checkpoint, and author splits.

---

## What IER measures

For an attributer `f`, an assistant `g_c` under rewriting condition `c`, and held-out
human messages `{x_i}` with authors `{a_i}`:

```
IER(g_c, f) = acc( f({x_i}), {a_i} )  -  acc( f({g_c(x_i)}), {a_i} )
```

the percentage-point drop in attribution accuracy after AI-assisted rewriting. It is
**instrument-dependent**: a property of the assistant, the condition, the attributer,
and the corpus together — not of the assistant alone.

We also report **double erasure**: a heavily rewritten message can be hard both to
attribute to its human author (low attribution) and to flag as AI-assisted (evades
AI-text detectors).

---

## What this repository releases

The paper states that we release the IER protocol along with the prompts, seeds,
LUAR checkpoint, and author splits. All of that is here:

| Claim in paper | Where |
|---|---|
| Open IER protocol (surface + deep attributers, dose ladder) | `src/`, `scripts/` |
| Exact rewriting prompts | [`PROMPTS.md`](PROMPTS.md), `src/rewriters.py` |
| Random seed | `0` everywhere (`numpy` `default_rng(0)`, `random.Random(0)`) |
| Deep-attributer checkpoint | `rrivera1849/LUAR-MUD` (HuggingFace) |
| Author splits (per corpus) | `splits/*_split.json` (via `scripts/export_splits.py`) |
| All numbers behind tables/figures | `results/*.json` |
| Figures | `results/compare_figure.png`, `results/population_figure.png` |

The generated **rewrites are not redistributed** — they contain real author text and
names (Enron, blogs) and occasional model fabrications. Regenerate them from the code,
prompts, and seed (see "Reproduce"). See **Data & ethics** below.

---

## Layout

```
src/
  data.py          corpus loaders + train/test splitting
  attributers.py   surface (stylometric) and deep (neural) attributers
  rewriters.py     the assistants: local (Qwen), OpenAI, Gemini; the PROMPTS dict
scripts/
  run_*.py         experiments (one per result; see table below)
  plot_*.py        figures
  export_splits.py dumps the author splits to splits/
splits/            author lists + held-out indices per corpus
results/           *.json numbers and *.png figures (rewrite caches are gitignored)
paper/             ier_paper.tex + ier_paper.pdf
```

---

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+. The LUAR model runs on CPU (its pooling is not MPS-compatible); the
scripts force CPU for it automatically.

---

## Data

Three public corpora, each predating generative writing assistants. **They are not
redistributed here** (licensing + personal content); place or fetch them under
`data/`:

- **Blog Authorship Corpus** (Schler et al. 2006) — informal personal writing.
- **Enron sent email** — real interpersonal workplace messages.
- **Reuters C50 (Reuter_50_50)** — news, one fixed beat per journalist (a
  topic-confound control).

The loaders in `src/data.py` read these from `data/`. All messages are used at their
natural length (mean ~190 words, capped at 400); we do not truncate to a fixed length.

---

## The protocol

**Attributers** (`src/attributers.py`):
- **Surface** — TF-IDF character (2–4)-grams + word (1–2)-grams with a linear SVM
  (`C=1.0`, balanced).
- **Deep** — LUAR (`rrivera1849/LUAR-MUD`), max-length 256; an author profile is the
  mean embedding of up to 60 training messages; attribution is nearest-profile.
- A general sentence encoder (MiniLM) is included only as a **topic-sensitive
  baseline** — it is nearly unaffected by word shuffling (`run_luar_checks.py`).

**Assistants** (`src/rewriters.py`): local `Qwen2.5-1.5B-Instruct` (primary),
`gpt-4o-mini`, and `gemini-flash-latest`. Decoding is greedy (temperature 0).

**Conditions** (dose ladder): `light`, `heavy`, `preserve` — see `PROMPTS.md`.

---

## Reproduce

Each result maps to one script. Run from the repository root. Surface/deep runs cache
their rewrites under `results/*_rewrites.jsonl` (gitignored) and write numbers to
`results/*.json`.

```bash
# --- author splits (Table/Fig setup) ---
python3 -m scripts.export_splits

# --- Table 1 + Figure 1: heavy-rewrite IER, surface + deep, 3 corpora ---
python3 -m scripts.run_blogs400            # blogs surface  -> blogs400_ier.json
python3 -m scripts.run_corpus400 --corpus enron
python3 -m scripts.run_corpus400 --corpus c50
python3 -m scripts.run_blogs400_luar       # blogs deep (LUAR)
python3 -m scripts.run_luar400 --corpus enron
python3 -m scripts.run_luar400 --corpus c50
python3 -m scripts.plot_compare            # -> results/compare_figure.png

# --- Figure 2: population accumulation (LUAR) ---
python3 -m scripts.run_pop400              # generate heavy rewrites for the k-set
python3 -m scripts.run_pop_luar            # -> population_luar.json
python3 -m scripts.plot_population         # -> results/population_figure.png

# --- Robustness checks ---
python3 -m scripts.run_cluster_and_blog    # author-clustered CIs (all cells) + blog ablation
python3 -m scripts.run_enron_ablation      # sign-off / quote leakage ablation (Enron)
python3 -m scripts.run_gec                 # non-generative grammar-correction baseline
python3 -m scripts.run_funcword            # content-blind (function-word) attributer
python3 -m scripts.run_topic_isolation     # content-word (topic) attributer
python3 -m scripts.run_fidelity            # semantic cosine / length / edit distance
python3 -m scripts.run_transform           # erasure-vs-transformation (LOO separability)
python3 -m scripts.run_luar_checks         # MiniLM/LUAR word-shuffle sensitivity
python3 -m scripts.run_detector2           # two AI-text detectors + positive control
```

**Commercial replication** (optional; you supply your own keys — they are never stored
in the repo):

```bash
export OPENAI_KEY_FILE=/path/to/openai.key      # or OPENAI_API_KEY=...
export GEMINI_KEY_FILE=/path/to/gemini.key      # or GEMINI_API_KEY=...
python3 -m scripts.run_commercial400 --corpus blogs --provider openai
python3 -m scripts.run_commercial400 --corpus blogs --provider gemini
python3 -m scripts.run_commercial400 --corpus enron --provider openai
python3 -m scripts.run_commercial400 --corpus enron --provider gemini
```

The `results/*.json` shipped in the repo are the exact numbers used in the paper, so
figures and tables can be regenerated without re-running the models.

---

## Data & ethics

- We release **code, prompts, seeds, the checkpoint id, and author splits** — enough
  to reproduce every number — but not the corpora or the generated rewrites.
- The corpora contain real people's writing; redistribute them only under their own
  terms. The rewrite caches additionally contain occasional model fabrications about
  named individuals, so they are gitignored and should be regenerated, not shared.
- IER can be dual-use (it can inform de-anonymization as well as identity-preserving
  design). We measure computational attributability, not human recognition; see the
  paper's Limitations.

---

## Citation

```bibtex
@inproceedings{malik2026ier,
  title     = {The Assistant Erased You: Measuring Loss of Authorship Signals
               in AI-Mediated Communication},
  author    = {Malik, Ushna and Awan, Moiz Sadiq},
  year      = {2026},
  note      = {Idiolect Erasure Rate (IER)}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
