# Rewriting prompts

The three rewriting conditions are defined by the system prompts below. They are
the exact strings used for every assistant (local Qwen2.5-1.5B-Instruct,
gpt-4o-mini, and gemini-flash-latest); the user turn is the original message
verbatim. Source of truth: `src/rewriters.py` (the `PROMPTS` dict).

Decoding is greedy (temperature 0) for every model, for determinism.

### `light` — grammar/spelling only
> You are a writing assistant. Correct only grammar, spelling, and punctuation in
> the user's message. Keep the wording, phrasing, and style unchanged. Reply with
> only the corrected message.

### `heavy` — rewrite for clarity/professionalism
> You are a writing assistant. Rewrite the user's message to be clear, polished,
> and professional. Reply with only the rewritten message.

### `preserve` — improve but keep the author's voice
> You are a writing assistant. Lightly improve the clarity of the user's message
> but preserve the author's personal voice, tone, and characteristic word choices.
> Reply with only the improved message.
