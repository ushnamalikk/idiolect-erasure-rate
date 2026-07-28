"""Authorship attributers.

Two independent "fingerprint" models, reported together so that the Idiolect
Erasure Rate is not an artifact of one method:

  * StylometricAttributer  -- character n-grams + word n-grams -> linear model.
    Captures the *surface* idiolect (spelling, function words, punctuation).
  * NeuralAttributer        -- sentence-embedding profiles + nearest-author.
    Captures the *deep* idiolect. A lightweight stand-in for LUAR; the full
    study swaps in rrivera1849/LUAR-MUD via the same interface.

Both expose fit(corpus) and predict(texts) -> list[author], plus
score(texts, gold) -> accuracy.
"""
from typing import Dict, List
import numpy as np


class StylometricAttributer:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                               min_df=2, sublinear_tf=True)
        word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                               min_df=2, sublinear_tf=True)
        self.model = Pipeline([
            ("feats", FeatureUnion([("char", char), ("word", word)])),
            ("clf", LogisticRegression(max_iter=2000, C=10.0,
                                       class_weight="balanced")),
        ])

    def fit(self, corpus: Dict[str, List[str]]):
        from .data import to_xy
        X, y = to_xy(corpus)
        self.model.fit(X, y)
        return self

    def predict(self, texts: List[str]) -> List[str]:
        return list(self.model.predict(texts))

    def score(self, texts: List[str], gold: List[str]) -> float:
        pred = self.predict(texts)
        return float(np.mean([p == g for p, g in zip(pred, gold)]))


class NeuralAttributer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.enc = SentenceTransformer(model_name)
        self.authors: List[str] = []
        self.profiles: np.ndarray = None  # (n_authors, dim), L2-normalised

    def _embed(self, texts: List[str]) -> np.ndarray:
        v = self.enc.encode(texts, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        return v

    def fit(self, corpus: Dict[str, List[str]]):
        self.authors, profs = [], []
        for a, texts in corpus.items():
            if not texts:
                continue
            self.authors.append(a)
            m = self._embed(texts).mean(axis=0)
            m = m / (np.linalg.norm(m) + 1e-9)
            profs.append(m)
        self.profiles = np.vstack(profs)
        return self

    def predict(self, texts: List[str]) -> List[str]:
        v = self._embed(texts)                       # (n, d)
        sims = v @ self.profiles.T                    # cosine (normalised)
        idx = sims.argmax(axis=1)
        return [self.authors[i] for i in idx]

    def score(self, texts: List[str], gold: List[str]) -> float:
        pred = self.predict(texts)
        return float(np.mean([p == g for p, g in zip(pred, gold)]))

    def centroid(self) -> np.ndarray:
        """Global 'average voice' -- used for the idiolect-drift metric."""
        c = self.profiles.mean(axis=0)
        return c / (np.linalg.norm(c) + 1e-9)
