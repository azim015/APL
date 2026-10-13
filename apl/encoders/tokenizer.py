"""Text tokenisation for APL.

The paper tokenises text with CLIP's byte-pair-encoding tokenizer (vocabulary
49,408, context length 77).  When the OpenAI ``clip`` or ``open_clip`` package
is installed we use exactly that tokenizer.  Otherwise we fall back to a
deterministic word-level hashing tokenizer with the same vocabulary size and
context length, so that the whole pipeline stays runnable offline.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Sequence

import torch

_WORD_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")

SOT_ID = 49406
EOT_ID = 49407


class Tokenizer:
    """Thin wrapper around a CLIP BPE tokenizer with an offline fallback."""

    def __init__(self, vocab_size: int = 49408, context_length: int = 77):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self._impl = None
        self.name = "hash"
        for module_name in ("clip", "open_clip"):
            try:
                module = __import__(module_name)
            except Exception:  # pragma: no cover - depends on environment
                continue
            tokenize = getattr(module, "tokenize", None)
            if callable(tokenize):
                self._impl = tokenize
                self.name = module_name
                break

    # ------------------------------------------------------------------ #
    def __call__(self, texts: str | Sequence[str]) -> torch.Tensor:
        return self.encode(texts)

    def encode(self, texts: str | Sequence[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if self._impl is not None:  # pragma: no cover - requires CLIP install
            return self._impl(texts, context_length=self.context_length,
                              truncate=True)
        return self._hash_encode(texts)

    # ------------------------------------------------------------------ #
    def _hash_encode(self, texts: Sequence[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), self.context_length, dtype=torch.long)
        for i, text in enumerate(texts):
            ids = [SOT_ID] + self._token_ids(text)
            ids = ids[: self.context_length - 1] + [EOT_ID]
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def _token_ids(self, text: str) -> List[int]:
        words = _WORD_RE.findall(text.lower())
        limit = self.vocab_size - 2  # reserve SOT / EOT
        ids = []
        for word in words:
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            ids.append(int.from_bytes(digest, "big") % limit)
        return ids

    # ------------------------------------------------------------------ #
    @staticmethod
    def eot_positions(tokens: torch.Tensor) -> torch.Tensor:
        """Index of the end-of-text token used to pool the text transformer."""
        return tokens.argmax(dim=-1)
