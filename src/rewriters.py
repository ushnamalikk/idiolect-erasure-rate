"""Rewriter backends = the 'AI writing assistant' being audited.

Conditions (the dose ladder):
  light    -- grammar/spelling only, keep wording        (minimal touch)
  heavy    -- rewrite to be clear and professional         (heavy touch)
  preserve -- improve but keep the author's voice          (design lever /
              style-preserving control: IER should shrink here)

Backends share one interface:  rewrite(texts, condition) -> list[str].

  * MockRewriter      -- deterministic, no model. Validates the pipeline and
                         serves as a null 'light-normalisation' floor.
  * LocalHFRewriter   -- an open instruction model (default Qwen2.5-1.5B-
                         Instruct) run locally. This is the honest, no-API-key
                         proxy for a lightweight commercial writing assistant.
  * (APIRewriter)     -- stub for GPT-4o-mini / Claude Haiku / Gemini Flash /
                         Apple Intelligence; same interface, needs a key.
"""
from typing import List
import re

PROMPTS = {
    "light": ("You are a writing assistant. Correct only grammar, spelling, and "
              "punctuation in the user's message. Keep the wording, phrasing, and "
              "style unchanged. Reply with only the corrected message."),
    "heavy": ("You are a writing assistant. Rewrite the user's message to be clear, "
              "polished, and professional. Reply with only the rewritten message."),
    "preserve": ("You are a writing assistant. Lightly improve the clarity of the "
                 "user's message but preserve the author's personal voice, tone, and "
                 "characteristic word choices. Reply with only the improved message."),
}


class MockRewriter:
    """Deterministic light normalisation -- no ML. Pipeline test / null floor."""
    name = "mock"

    def rewrite(self, texts: List[str], condition: str) -> List[str]:
        out = []
        for t in texts:
            s = re.sub(r"\s+", " ", t).strip()
            s = s.replace(" ,", ",").replace(" .", ".")
            if condition in ("heavy", "preserve"):
                # crude 'professionalising' normalisation
                s = re.sub(r"\bgonna\b", "going to", s)
                s = re.sub(r"\bwanna\b", "want to", s)
                s = re.sub(r"\bkids\b", "children", s)
            out.append(s)
        return out


class LocalHFRewriter:
    name = "local-hf"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str = None, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device is None:
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device).eval()

    def _rewrite_one(self, text: str, condition: str) -> str:
        import torch
        msgs = [{"role": "system", "content": PROMPTS[condition]},
                {"role": "user", "content": text}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()

    def rewrite(self, texts: List[str], condition: str) -> List[str]:
        return [self._rewrite_one(t, condition) for t in texts]


class OpenAIRewriter:
    """Commercial-tier assistant proxy via the OpenAI Chat Completions API.

    Reads the key from OPENAI_API_KEY, else from a file at OPENAI_KEY_FILE
    (path only in the environment -- the secret never appears in a command).
    """
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", **_):
        import os
        self.model = model
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            kf = os.environ.get("OPENAI_KEY_FILE")
            if kf and os.path.exists(kf):
                key = open(kf).read().strip()
        if not key:
            raise RuntimeError("no OpenAI key (set OPENAI_API_KEY or OPENAI_KEY_FILE)")
        self._key = key

    def _rewrite_one(self, text: str, condition: str) -> str:
        import json, time, urllib.request, urllib.error
        body = json.dumps({
            "model": self.model, "temperature": 0,
            "messages": [{"role": "system", "content": PROMPTS[condition]},
                         {"role": "user", "content": text}],
        }).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions", data=body,
                    headers={"Authorization": f"Bearer {self._key}",
                             "Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=90))
                return r["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < 4:
                    time.sleep(2 ** attempt); continue
                raise
            except Exception:
                if attempt < 4:
                    time.sleep(2 ** attempt); continue
                raise

    def rewrite(self, texts, condition: str):
        return [self._rewrite_one(t, condition) for t in texts]


class GeminiRewriter:
    """Commercial-tier assistant proxy via the Google Gemini API.

    Reads the key from GEMINI_API_KEY / GOOGLE_API_KEY, else from a file at
    GEMINI_KEY_FILE (path only in the environment).
    """
    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash", **_):
        import os
        self.model = model
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            kf = os.environ.get("GEMINI_KEY_FILE")
            if kf and os.path.exists(kf):
                key = open(kf).read().strip()
        if not key:
            raise RuntimeError("no Gemini key (set GEMINI_API_KEY or GEMINI_KEY_FILE)")
        self._key = key

    def _rewrite_one(self, text: str, condition: str) -> str:
        import json, time, urllib.request, urllib.error
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self._key}")
        body = json.dumps({
            "system_instruction": {"parts": [{"text": PROMPTS[condition]}]},
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0},
        }).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    url, data=body, headers={"Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=90))
                cand = r.get("candidates", [])
                if not cand or "content" not in cand[0]:
                    return text  # safety-blocked/empty -> fall back to original
                return "".join(p.get("text", "") for p in
                               cand[0]["content"]["parts"]).strip()
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < 4:
                    time.sleep(2 ** attempt); continue
                raise
            except Exception:
                if attempt < 4:
                    time.sleep(2 ** attempt); continue
                raise

    def rewrite(self, texts, condition: str):
        return [self._rewrite_one(t, condition) for t in texts]


def get_rewriter(kind: str = "local", **kw):
    if kind == "mock":
        return MockRewriter()
    if kind == "local":
        return LocalHFRewriter(**kw)
    if kind == "openai":
        return OpenAIRewriter(**kw)
    if kind == "gemini":
        return GeminiRewriter(**kw)
    raise ValueError(f"unknown rewriter kind: {kind}")
