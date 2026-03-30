"""
Textual Quality Metrics — Section VI.C

Evaluates the semantic quality of APL-generated scene descriptions
against ground-truth human annotations using:

    - BLEU  (Bilingual Evaluation Understudy)
    - ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
    - CIDEr (Consensus-based Image Description Evaluation)

These metrics assess whether the textual abstraction accurately
captures the critical evidentiary details of the traffic scene.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def _ngram_counts(tokens: List[str], n: int) -> Counter:
    """Extract n-gram counts from a token list."""
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    max_n: int = 4,
) -> float:
    """
    Corpus-level BLEU score (BLEU-4 by default).

    Args:
        hypotheses: List of generated descriptions (one per sample)
        references: List of ground-truth descriptions (one per sample)
        max_n:      Maximum n-gram order

    Returns:
        BLEU score ∈ [0, 1]
    """
    assert len(hypotheses) == len(references), "Must have equal number of hyp/ref pairs."

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.lower().split()
        ref_tokens = ref.lower().split()
        hyp_len += len(hyp_tokens)
        ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            hyp_ngrams = _ngram_counts(hyp_tokens, n)
            ref_ngrams = _ngram_counts(ref_tokens, n)
            clipped = sum(
                min(count, ref_ngrams[ngram])
                for ngram, count in hyp_ngrams.items()
            )
            clipped_counts[n - 1] += clipped
            total_counts[n - 1] += max(len(hyp_tokens) - n + 1, 0)

    # Brevity penalty
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))

    # Log-average of n-gram precisions
    log_avg = 0.0
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_avg += math.log(clipped_counts[n] / total_counts[n]) / max_n

    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Longest Common Subsequence length via dynamic programming."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(
    hypotheses: List[str],
    references: List[str],
    beta: float = 1.2,
) -> float:
    """
    Corpus-level ROUGE-L score based on Longest Common Subsequence.

    Args:
        hypotheses: List of generated descriptions
        references: List of ground-truth descriptions
        beta:       F-measure weighting (default 1.2 emphasises recall)

    Returns:
        ROUGE-L F-score ∈ [0, 1]
    """
    total_f = 0.0
    for hyp, ref in zip(hypotheses, references):
        h_tokens = hyp.lower().split()
        r_tokens = ref.lower().split()
        lcs = _lcs_length(h_tokens, r_tokens)
        precision = lcs / max(len(h_tokens), 1)
        recall = lcs / max(len(r_tokens), 1)
        if precision + recall < 1e-10:
            f = 0.0
        else:
            f = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        total_f += f
    return total_f / max(len(hypotheses), 1)


# ---------------------------------------------------------------------------
# CIDEr (simplified sentence-level TF-IDF version)
# ---------------------------------------------------------------------------

def compute_cider(
    hypotheses: List[str],
    references: List[str],
    n_range: int = 4,
    sigma: float = 6.0,
    corpus_override: Optional[List[str]] = None,
) -> float:
    """
    Simplified CIDEr-D score (corpus-level).

    Computes TF-IDF weighted n-gram cosine similarity between
    hypothesis and reference.

    Args:
        hypotheses:       List of generated descriptions
        references:       List of ground-truth descriptions
        n_range:          Maximum n-gram order (default 4)
        sigma:            Gaussian length penalty std dev
        corpus_override:  If provided, use this corpus for IDF computation
                          instead of references (useful when references is tiny)

    Returns:
        CIDEr score (typically ∈ [0, ~2.0] for good descriptions)
    """
    corpus = corpus_override if corpus_override is not None else references

    # Build IDF (document frequency over reference corpus)
    df: Dict[Tuple, int] = Counter()
    for doc in corpus:
        tokens = doc.lower().split()
        seen_ngrams: set = set()
        for n in range(1, n_range + 1):
            for i in range(len(tokens) - n + 1):
                ng = tuple(tokens[i : i + n])
                if ng not in seen_ngrams:
                    df[ng] += 1
                    seen_ngrams.add(ng)
    num_docs = max(len(corpus), 1)

    def _tfidf_vec(text: str) -> Dict[Tuple, float]:
        tokens = text.lower().split()
        vec: Dict[Tuple, float] = {}
        for n in range(1, n_range + 1):
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            tf = Counter(ngrams)
            for ng, count in tf.items():
                idf = math.log((num_docs + 1) / (df.get(ng, 0) + 1))
                vec[ng] = (count / max(len(ngrams), 1)) * idf
        return vec

    def _cos_sim(v1: Dict, v2: Dict) -> float:
        shared = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[k] * v2[k] for k in shared)
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        return dot / max(norm1 * norm2, 1e-10)

    scores = []
    for hyp, ref in zip(hypotheses, references):
        # Gaussian length penalty
        h_len = len(hyp.split())
        r_len = len(ref.split())
        length_penalty = math.exp(-((h_len - r_len) ** 2) / (2 * sigma ** 2))

        v_hyp = _tfidf_vec(hyp)
        v_ref = _tfidf_vec(ref)
        score = length_penalty * _cos_sim(v_hyp, v_ref)
        scores.append(score)

    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# Convenience: compute all text quality metrics
# ---------------------------------------------------------------------------

def compute_all_text_metrics(
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute BLEU, ROUGE-L, and CIDEr for a set of generated descriptions.

    Args:
        hypotheses: APL-generated scene descriptions
        references: Ground-truth human annotations

    Returns:
        Dict with metric names → scalar values
    """
    return {
        "BLEU": compute_bleu(hypotheses, references),
        "ROUGE": compute_rouge_l(hypotheses, references),
        "CIDEr": compute_cider(hypotheses, references),
    }
