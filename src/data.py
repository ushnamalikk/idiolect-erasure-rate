"""Data loading for the Idiolect Erasure Rate (IER) experiments.

Primary pilot corpus: Reuters C50 (50 authors x 50 train / 50 test docs).
The full study additionally targets the Blog Authorship Corpus (informal
personal writing) and the Enron email corpus (real interpersonal messages);
both plug into the same loader interface (a dict: author -> list[str]).
"""
import os
import glob
from typing import Dict, List, Optional


def _truncate_words(text: str, max_words: Optional[int]) -> str:
    if max_words is None:
        return text.strip()
    return " ".join(text.split()[:max_words]).strip()


def load_c50(root: str,
             split: str = "train",
             max_authors: Optional[int] = None,
             max_docs_per_author: Optional[int] = None,
             max_words: Optional[int] = None) -> Dict[str, List[str]]:
    """Load one split of C50 as {author: [texts]}.

    max_words truncates each document to simulate message-length text
    (real writing assistants operate on short messages, not full articles).
    """
    folder = os.path.join(root, "C50train" if split == "train" else "C50test")
    authors = sorted(os.listdir(folder))
    authors = [a for a in authors if os.path.isdir(os.path.join(folder, a))]
    if max_authors is not None:
        authors = authors[:max_authors]
    out: Dict[str, List[str]] = {}
    for a in authors:
        files = sorted(glob.glob(os.path.join(folder, a, "*.txt")))
        if max_docs_per_author is not None:
            files = files[:max_docs_per_author]
        texts = []
        for f in files:
            with open(f, "r", encoding="latin-1") as fh:
                texts.append(_truncate_words(fh.read(), max_words))
        out[a] = texts
    return out


def load_blogs(root: str = "data/blogs",
               max_authors: int = 25,
               posts_per_author: int = 20,
               min_words: int = 25,
               max_words: Optional[int] = 90,
               scan_limit: int = 6000) -> Dict[str, List[str]]:
    """Blog Authorship Corpus -> {author_id: [posts]}.

    Informal, topic-varied personal writing: unlike C50, topic is NOT tied to
    author, so a content-based attributer cannot cheat via topic. This is the
    corpus that can adjudicate *deep* (not just surface) idiolect erasure.
    """
    import re
    files = sorted(glob.glob(os.path.join(root, "*.xml")))[:scan_limit]
    post_re = re.compile(r"<post>(.*?)</post>", re.DOTALL | re.IGNORECASE)
    out: Dict[str, List[str]] = {}
    for f in files:
        if len(out) >= max_authors:
            break
        author = os.path.basename(f).split(".")[0]
        try:
            raw = open(f, "r", encoding="latin-1").read()
        except Exception:
            continue
        posts = []
        for p in post_re.findall(raw):
            s = p.replace("urlLink", " ")
            s = re.sub(r"\s+", " ", s).strip()
            if len(s.split()) >= min_words:
                posts.append(_truncate_words(s, max_words))
            if len(posts) >= posts_per_author:
                break
        if len(posts) >= posts_per_author:
            out[author] = posts
    return out


def load_blogs_rich(root: str = "data/blogs",
                    max_authors: int = 50,
                    min_posts: int = 20,
                    load_posts: int = 200,
                    test_slice=(16, 20),
                    min_words: int = 25,
                    max_words: Optional[int] = 90,
                    scan_limit: int = 6000):
    """Blogs with a large training set but the SAME authors and test messages.

    Author qualification uses min_posts (20), so the author set is identical to
    load_blogs(posts_per_author=20). Test messages are posts[16:20], also
    identical, which keeps previously generated rewrites valid. Every remaining
    post (up to load_posts) becomes training data, raising attribution baselines
    without changing what is being attributed.
    """
    import re
    files = sorted(glob.glob(os.path.join(root, "*.xml")))[:scan_limit]
    post_re = re.compile(r"<post>(.*?)</post>", re.DOTALL | re.IGNORECASE)
    train, test = {}, {}
    for f in files:
        if len(train) >= max_authors:
            break
        author = os.path.basename(f).split(".")[0]
        try:
            raw = open(f, "r", encoding="latin-1").read()
        except Exception:
            continue
        posts = []
        for p in post_re.findall(raw):
            s = re.sub(r"\s+", " ", p.replace("urlLink", " ")).strip()
            if len(s.split()) >= min_words:
                posts.append(_truncate_words(s, max_words))
            if len(posts) >= load_posts:
                break
        if len(posts) < min_posts:
            continue
        lo, hi = test_slice
        test[author] = posts[lo:hi]
        train[author] = posts[:lo] + posts[hi:]
    return train, test


def _clean_email_body(body: str, max_words: Optional[int]) -> str:
    """Strip quoted replies, forwarded headers, and signatures from an email."""
    import re
    lines = []
    for ln in body.splitlines():
        s = ln.strip()
        if re.match(r"-+\s*Original Message", s, re.I):
            break
        if re.match(r"^(From|To|Sent|Subject|Cc):\s", s):   # forwarded header block
            break
        if s.startswith(">") or s.startswith("|"):
            continue
        lines.append(ln)
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return _truncate_words(text, max_words)


def load_enron(tar_path: str = "data/enron.tar.gz",
               max_authors: int = 25,
               posts_per_author: int = 20,
               min_words: int = 25,
               max_words: Optional[int] = 90) -> Dict[str, List[str]]:
    """Enron email -> {user: [sent-email bodies]}.

    Reads each user's 'sent' folders directly from the tar (no full extract).
    Real interpersonal messages -- the most AIMC-native register -- and a second,
    independent interpersonal corpus to test whether deep erasure replicates.
    """
    import tarfile, email
    out: Dict[str, List[str]] = {}
    scanned = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            scanned += 1
            if scanned > 120000:                 # safety cap on scan depth
                break
            qualified = sum(1 for v in out.values() if len(v) >= posts_per_author)
            if qualified >= max_authors:
                break
            if not m.isfile():
                continue
            parts = m.name.split("/")            # maildir/<user>/<folder>/...
            if len(parts) < 4 or parts[0] != "maildir":
                continue
            folder = parts[2].lower()
            if "sent" not in folder:             # sent_items, _sent_mail, sent
                continue
            user = parts[1]
            out.setdefault(user, [])
            if len(out[user]) >= posts_per_author:
                continue
            try:
                raw = tar.extractfile(m).read()
                msg = email.message_from_bytes(raw)
                body = msg.get_payload()
                if not isinstance(body, str):
                    continue
                cleaned = _clean_email_body(body, max_words)
            except Exception:
                continue
            if len(cleaned.split()) >= min_words:
                out[user].append(cleaned)
    return {u: v for u, v in out.items() if len(v) >= posts_per_author}


def load_enron_rich(tar_path: str = "data/enron.tar.gz",
                    max_authors: int = 25,
                    min_posts: int = 20,
                    load_posts: int = 120,
                    test_slice=(16, 20),
                    min_words: int = 25,
                    max_words: Optional[int] = 400):
    """Enron with a large training set but the same authors and test messages.

    Mirrors load_blogs_rich: qualification at min_posts keeps the author set
    identical to the pilot, and posts[16:20] remain the held-out messages.
    """
    corpus = load_enron(tar_path, max_authors=max_authors,
                        posts_per_author=load_posts, min_words=min_words,
                        max_words=max_words)
    train, test = {}, {}
    lo, hi = test_slice
    for a, texts in corpus.items():
        if len(texts) < min_posts:
            continue
        test[a] = texts[lo:hi]
        train[a] = texts[:lo] + texts[hi:]
    return train, test


def split_corpus(corpus: Dict[str, List[str]], n_test: int = 4):
    """Split each author's texts into (train, test); test = last n_test."""
    train, test = {}, {}
    for a, texts in corpus.items():
        test[a] = texts[-n_test:]
        train[a] = texts[:-n_test]
    return train, test


def to_xy(corpus: Dict[str, List[str]]):
    """Flatten {author:[texts]} to parallel lists (texts, labels)."""
    X, y = [], []
    for author, texts in corpus.items():
        for t in texts:
            X.append(t)
            y.append(author)
    return X, y
