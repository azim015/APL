"""Agreement of generated descriptions with human references (Table VII).

Pure-python BLEU-4, ROUGE-L and CIDEr-D so that no NLP toolkit has to be
installed.  All three take a list of candidate strings and a list of reference
lists (one or more references per candidate).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


# --------------------------------------------------------------------------- #
def bleu(candidates: Sequence[str], references: Sequence[Sequence[str]],
         max_n: int = 4) -> float:
    """Corpus BLEU-``n`` with the standard brevity penalty."""
    clipped = [0] * max_n
    totals = [0] * max_n
    cand_len = ref_len = 0

    for candidate, refs in zip(candidates, references):
        cand_tokens = tokenize(candidate)
        ref_token_lists = [tokenize(r) for r in refs]
        if not ref_token_lists:
            continue
        cand_len += len(cand_tokens)
        ref_len += min(
            (abs(len(r) - len(cand_tokens)), len(r)) for r in ref_token_lists
        )[1]
        for n in range(1, max_n + 1):
            cand_ngrams = _ngrams(cand_tokens, n)
            if not cand_ngrams:
                continue
            max_ref = Counter()
            for ref_tokens in ref_token_lists:
                for gram, count in _ngrams(ref_tokens, n).items():
                    max_ref[gram] = max(max_ref[gram], count)
            clipped[n - 1] += sum(
                min(count, max_ref[gram]) for gram, count in cand_ngrams.items()
            )
            totals[n - 1] += sum(cand_ngrams.values())

    if min(totals) == 0:
        return 0.0
    precisions = [
        (clipped[i] / totals[i]) if totals[i] > 0 else 0.0 for i in range(max_n)
    ]
    if min(precisions) <= 0:
        return 0.0
    log_mean = sum(math.log(p) for p in precisions) / max_n
    brevity = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / max(1, cand_len))
    return brevity * math.exp(log_mean)


# --------------------------------------------------------------------------- #
def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def rouge_l(candidates: Sequence[str], references: Sequence[Sequence[str]],
            beta: float = 1.2) -> float:
    """Average sentence-level ROUGE-L F-measure."""
    scores = []
    for candidate, refs in zip(candidates, references):
        cand_tokens = tokenize(candidate)
        best = 0.0
        for ref in refs:
            ref_tokens = tokenize(ref)
            if not cand_tokens or not ref_tokens:
                continue
            lcs = _lcs(cand_tokens, ref_tokens)
            if lcs == 0:
                continue
            precision = lcs / len(cand_tokens)
            recall = lcs / len(ref_tokens)
            f = ((1 + beta ** 2) * precision * recall) / (
                recall + beta ** 2 * precision
            )
            best = max(best, f)
        scores.append(best)
    return sum(scores) / max(1, len(scores))


# --------------------------------------------------------------------------- #
def cider(candidates: Sequence[str], references: Sequence[Sequence[str]],
          max_n: int = 4, sigma: float = 6.0) -> float:
    """CIDEr-D with corpus-level TF-IDF weighting."""
    num_docs = len(candidates)
    if num_docs == 0:
        return 0.0

    doc_freq: List[Dict[Tuple[str, ...], int]] = [defaultdict(int) for _ in range(max_n)]
    ref_ngrams_all = []
    for refs in references:
        per_ref = []
        seen = [set() for _ in range(max_n)]
        for ref in refs:
            tokens = tokenize(ref)
            grams = [_ngrams(tokens, n + 1) for n in range(max_n)]
            per_ref.append((tokens, grams))
            for n in range(max_n):
                seen[n].update(grams[n].keys())
        for n in range(max_n):
            for gram in seen[n]:
                doc_freq[n][gram] += 1
        ref_ngrams_all.append(per_ref)

    log_docs = math.log(max(1, num_docs))

    def vector(grams: Counter, n: int):
        vec, norm = {}, 0.0
        length = max(1, sum(grams.values()))
        for gram, count in grams.items():
            idf = log_docs - math.log(max(1, doc_freq[n][gram]))
            value = (count / length) * idf
            vec[gram] = value
            norm += value ** 2
        return vec, math.sqrt(norm)

    scores = []
    for candidate, per_ref in zip(candidates, ref_ngrams_all):
        cand_tokens = tokenize(candidate)
        total = 0.0
        for n in range(max_n):
            cand_vec, cand_norm = vector(_ngrams(cand_tokens, n + 1), n)
            per_n = []
            for ref_tokens, ref_grams in per_ref:
                ref_vec, ref_norm = vector(ref_grams[n], n)
                prod = sum(
                    value * ref_vec.get(gram, 0.0) for gram, value in cand_vec.items()
                )
                if cand_norm > 0 and ref_norm > 0:
                    sim = prod / (cand_norm * ref_norm)
                else:
                    sim = 0.0
                delta = len(cand_tokens) - len(ref_tokens)
                sim *= math.exp(-(delta ** 2) / (2 * sigma ** 2))
                per_n.append(sim)
            if per_n:
                total += sum(per_n) / len(per_n)
        scores.append(10.0 * total / max_n)
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------- #
@dataclass
class TextQualityReport:
    bleu: float
    rouge: float
    cider: float

    def as_row(self) -> Dict[str, float]:
        return {
            "BLEU": round(self.bleu, 3),
            "ROUGE": round(self.rouge, 3),
            "CIDEr": round(self.cider, 3),
        }


def text_quality(candidates: Sequence[str],
                 references: Sequence) -> TextQualityReport:
    refs: List[Sequence[str]] = [
        [r] if isinstance(r, str) else list(r) for r in references
    ]
    return TextQualityReport(
        bleu=bleu(candidates, refs),
        rouge=rouge_l(candidates, refs),
        cider=cider(candidates, refs),
    )
