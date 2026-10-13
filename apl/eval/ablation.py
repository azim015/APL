"""Component ablation study (Table IX).

Variants reproduced here:

* full APL (GI + ID + FG + refinement + rule engine)
* w/o Global-Invariant branch
* w/o Instance-Dependent branch
* w/o Fine-Grained branch
* w/o iterative prompt refinement
* w/o rule engine (a direct classifier decides instead)
* single-branch variants (GI only, ID only, FG only)

Each variant is trained with the same budget and scored on scene-level
violation accuracy, Privacy Score and Semantic Utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch

from ..logic.rule_engine import INSUFFICIENT, NO_VIOLATION, VIOLATION
from ..model import APLModel
from ..train.trainer import Trainer, set_seed
from .privacy import privacy_score
from .utility import AblationRow, semantic_utility, violation_accuracy

VARIANTS: Dict[str, Dict[str, object]] = {
    "Full APL (GI + ID + FG + Refinement + Rule Engine)": {},
    "w/o Global-Invariant (GI) branch": {"branches.use_global_invariant": False},
    "w/o Instance-Dependent (ID) branch": {"branches.use_instance_dependent": False},
    "w/o Fine-Grained (FG) branch": {"branches.use_fine_grained": False},
    "w/o Iterative Prompt Refinement": {"refinement.enabled": False},
    "w/o Rule Engine (direct classifier instead)": {"_direct_classifier": True},
    "Single-branch only (GI only)": {
        "branches.use_instance_dependent": False,
        "branches.use_fine_grained": False,
    },
    "Single-branch only (ID only)": {
        "branches.use_global_invariant": False,
        "branches.use_fine_grained": False,
    },
    "Single-branch only (FG only)": {
        "branches.use_global_invariant": False,
        "branches.use_instance_dependent": False,
    },
}


def direct_classifier_label(evidence, index: int = 0,
                            threshold: float = 0.5) -> str:
    """Rule-engine replacement: a plain threshold on the strongest evidence.

    This is the "direct classifier" baseline - it drops the scenario logic and
    the insufficient-evidence outcome is no longer derived from a rule.
    """
    best = float(evidence.satisfaction[index].max())
    return VIOLATION if best >= threshold else NO_VIOLATION


@dataclass
class VariantResult:
    name: str
    row: AblationRow
    scene_predictions: List[str]
    descriptions: List[str]


def evaluate_variant(
    model: APLModel,
    samples: Sequence[Dict],
    direct_classifier: bool = False,
) -> VariantResult:
    predictions: List[str] = []
    descriptions: List[str] = []
    positive: List[str] = []
    targets: List[str] = []
    sensitive: List[Sequence[str]] = []
    relevant: List[Sequence[str]] = []

    for sample in samples:
        image = sample["image"].unsqueeze(0)
        out = model.infer(image)
        if direct_classifier:
            predictions.append(direct_classifier_label(out.evidence))
        else:
            predictions.append(out.engine.scene_label)
        descriptions.append(out.description.text)
        positive.append(out.description.positive_text)
        targets.append(sample.get("scene_label_symbol", NO_VIOLATION))
        sensitive.append(sample.get("sensitive_attributes", []))
        relevant.append(sample.get("relevant_attributes", []))

    row = AblationRow(
        variant="",
        violation_accuracy=violation_accuracy(predictions, targets),
        privacy_score=privacy_score(descriptions, sensitive),
        semantic_utility=semantic_utility(positive, relevant),
    )
    return VariantResult(
        name="", row=row, scene_predictions=predictions, descriptions=descriptions
    )


def run_ablation(
    cfg,
    train_set,
    samples: Sequence[Dict],
    bank,
    collate_fn: Optional[Callable] = None,
    epochs: Optional[int] = None,
    variants: Optional[Sequence[str]] = None,
    log_fn: Optional[Callable[[str], None]] = print,
) -> List[AblationRow]:
    names = list(variants) if variants else list(VARIANTS)
    rows: List[AblationRow] = []

    for name in names:
        overrides = dict(VARIANTS[name])
        direct = bool(overrides.pop("_direct_classifier", False))
        variant_cfg = cfg.clone()
        if overrides:
            variant_cfg.update(overrides)

        set_seed(variant_cfg.optim.seed)
        model = APLModel(variant_cfg, bank=bank)
        trainer = Trainer(model, variant_cfg, collate_fn=collate_fn,
                          log_fn=lambda _m: None)
        trainer.fit(train_set, epochs=epochs or variant_cfg.optim.epochs)

        result = evaluate_variant(model, samples, direct_classifier=direct)
        result.row.variant = name
        rows.append(result.row)
        if log_fn:
            log_fn(
                f"{name:52s} acc {result.row.violation_accuracy:5.1f}  "
                f"privacy {result.row.privacy_score:5.1f}  "
                f"utility {result.row.semantic_utility:5.1f}"
            )
    return rows
