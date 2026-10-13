"""Task-utility metrics (Sec. VI-E, Sec. VI-F, Table VIII, Table IX).

* ``Semantic Utility = 100 * N_preserved / N_relevant``  (Eq. 7)
* scene-level violation accuracy over the V/U/N vocabulary
* downstream ITS metrics: traffic-flow RMSE, sign-recognition accuracy and
  pedestrian-count MAE, evaluated on the textual representation rather than on
  the raw frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
def attribute_preserved(attribute: str, text: str) -> bool:
    """A task-relevant attribute counts as preserved when the description
    still carries its content words."""
    lowered = text.lower()
    phrase = attribute.replace("_", " ").lower()
    if phrase in lowered:
        return True
    tokens = [t for t in phrase.split() if len(t) > 3]
    if not tokens:
        return phrase in lowered
    hits = sum(1 for t in tokens if t in lowered)
    return hits / len(tokens) >= 0.6


def semantic_utility(
    descriptions: Sequence[str],
    relevant_attributes: Sequence[Sequence[str]],
) -> float:
    """Eq. (7): percentage of task-relevant evidence retained."""
    total, preserved = 0, 0
    for text, attributes in zip(descriptions, relevant_attributes):
        for attribute in attributes:
            total += 1
            if attribute_preserved(attribute, text):
                preserved += 1
    if total == 0:
        return 100.0
    return 100.0 * preserved / total


# --------------------------------------------------------------------------- #
def violation_accuracy(predicted: Sequence[str], target: Sequence[str]) -> float:
    """Scene-level accuracy over the V / U / N vocabulary.

    As the paper states, this is a *scene-level* figure: it does not certify
    that every co-occurring violation of a multi-violation scene was recovered.
    """
    if not target:
        return 0.0
    hits = sum(1 for p, t in zip(predicted, target) if p == t)
    return 100.0 * hits / len(target)


def scenario_prf(
    predicted: Sequence[Sequence[str]],
    target: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """Multi-label precision / recall / F1 and exact-set accuracy.

    The paper flags the missing multi-label breakdown as a limitation; this
    helper supplies it whenever scenario-level annotations are available.
    """
    tp = fp = fn = 0
    exact = 0
    for pred, tgt in zip(predicted, target):
        p, t = set(pred), set(tgt)
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
        exact += int(p == t)
    precision = 100.0 * tp / max(1, tp + fp)
    recall = 100.0 * tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_set_accuracy": 100.0 * exact / max(1, len(target)),
    }


# --------------------------------------------------------------------------- #
def rmse(predicted: Sequence[float], target: Sequence[float]) -> float:
    if not target:
        return 0.0
    return math.sqrt(
        sum((p - t) ** 2 for p, t in zip(predicted, target)) / len(target)
    )


def mae(predicted: Sequence[float], target: Sequence[float]) -> float:
    if not target:
        return 0.0
    return sum(abs(p - t) for p, t in zip(predicted, target)) / len(target)


def accuracy(predicted: Sequence, target: Sequence) -> float:
    if not target:
        return 0.0
    return 100.0 * sum(1 for p, t in zip(predicted, target) if p == t) / len(target)


@dataclass
class DownstreamReport:
    """Table VIII: one row per input representation."""

    name: str
    flow_rmse: Optional[float] = None
    sign_accuracy: Optional[float] = None
    pedestrian_mae: Optional[float] = None

    def as_row(self) -> Dict[str, object]:
        return {
            "Representation": self.name,
            "Flow Pred. (RMSE)": None if self.flow_rmse is None else round(self.flow_rmse, 2),
            "Sign Recog. (Acc %)": None if self.sign_accuracy is None else round(self.sign_accuracy, 1),
            "Ped. Count (MAE)": None if self.pedestrian_mae is None else round(self.pedestrian_mae, 2),
        }


def downstream_report(
    name: str,
    flow_pred: Optional[Sequence[float]] = None,
    flow_true: Optional[Sequence[float]] = None,
    sign_pred: Optional[Sequence] = None,
    sign_true: Optional[Sequence] = None,
    ped_pred: Optional[Sequence[float]] = None,
    ped_true: Optional[Sequence[float]] = None,
) -> DownstreamReport:
    return DownstreamReport(
        name=name,
        flow_rmse=rmse(flow_pred, flow_true) if flow_true is not None else None,
        sign_accuracy=accuracy(sign_pred, sign_true) if sign_true is not None else None,
        pedestrian_mae=mae(ped_pred, ped_true) if ped_true is not None else None,
    )


# --------------------------------------------------------------------------- #
@dataclass
class AblationRow:
    """Table IX: one ablation variant."""

    variant: str
    violation_accuracy: float
    privacy_score: float
    semantic_utility: float

    def as_row(self) -> Dict[str, object]:
        return {
            "Variant": self.variant,
            "Violation Acc. (%)": round(self.violation_accuracy, 1),
            "Privacy Score (%)": round(self.privacy_score, 1),
            "Semantic Utility (%)": round(self.semantic_utility, 1),
        }


def format_table(rows: Sequence[Dict[str, object]]) -> str:
    """Render a list of dict rows as a fixed-width table."""
    if not rows:
        return ""
    headers = list(rows[0].keys())
    widths = {
        h: max(len(str(h)), max(len(str(r.get(h, ""))) for r in rows))
        for h in headers
    }
    line = "  ".join(str(h).ljust(widths[h]) for h in headers)
    sep = "  ".join("-" * widths[h] for h in headers)
    body = [
        "  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers) for r in rows
    ]
    return "\n".join([line, sep] + body)
