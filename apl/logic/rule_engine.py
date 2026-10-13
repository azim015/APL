"""Rule-based violation decision (Sec. V-G).

At termination the engine turns the weighted compatibility score of every
scenario into one of three symbolic decisions:

.. math::

   y_j = \\begin{cases}
     \\mathrm{V}, & \\gamma_j^{(t_\\star)} \\ge \\Gamma_{strict}\\\\
     \\mathrm{U}, & \\Gamma_{loose} \\le \\gamma_j^{(t_\\star)} < \\Gamma_{strict}\\\\
     \\mathrm{N}, & \\gamma_j^{(t_\\star)} < \\Gamma_{loose}
   \\end{cases}

with ``Gamma_loose = 0.40`` and ``Gamma_strict = 0.80``.  ``U`` ("insufficient
evidence") is a first-class outcome: a low compatibility score is *not* proof
that an unobserved predicate is false.

The engine returns one decision **per scenario** and does not stop at the first
positive violation, so a scene may carry several violation types,
``V(I) = {nu_j : y_j = V}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

VIOLATION = "V"
INSUFFICIENT = "U"
NO_VIOLATION = "N"


@dataclass
class ScenarioDecision:
    scenario_id: str
    violation_type: str
    decision: str
    gamma: float
    predicate_satisfaction: Dict[str, float] = field(default_factory=dict)
    missing_predicates: List[str] = field(default_factory=list)
    rule_satisfied: bool = False
    legal_reference: str = ""

    @property
    def is_violation(self) -> bool:
        return self.decision == VIOLATION

    def explain(self) -> str:
        label = {
            VIOLATION: "violation",
            INSUFFICIENT: "insufficient evidence",
            NO_VIOLATION: "no violation",
        }[self.decision]
        parts = [
            f"[{self.scenario_id}] {self.violation_type}: {label} "
            f"(gamma={self.gamma:.3f})"
        ]
        if self.predicate_satisfaction:
            evidence = ", ".join(
                f"{k}={v:.2f}" for k, v in self.predicate_satisfaction.items()
            )
            parts.append(f"  evidence: {evidence}")
        if self.missing_predicates:
            parts.append("  missing: " + ", ".join(self.missing_predicates))
        if self.legal_reference:
            parts.append(f"  reference: {self.legal_reference}")
        return "\n".join(parts)


@dataclass
class EngineOutput:
    decisions: List[ScenarioDecision]
    gamma: torch.Tensor
    iterations: int = 0
    stop_reason: str = ""

    @property
    def violations(self) -> List[str]:
        """``V(I)`` - the set of violation types classified as positive."""
        return [d.violation_type for d in self.decisions if d.is_violation]

    @property
    def scene_label(self) -> str:
        if any(d.decision == VIOLATION for d in self.decisions):
            return VIOLATION
        if any(d.decision == INSUFFICIENT for d in self.decisions):
            return INSUFFICIENT
        return NO_VIOLATION

    def to_dict(self) -> Dict:
        return {
            "scene_label": self.scene_label,
            "violations": self.violations,
            "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "scenarios": [
                {
                    "id": d.scenario_id,
                    "violation_type": d.violation_type,
                    "decision": d.decision,
                    "gamma": d.gamma,
                    "rule_satisfied": d.rule_satisfied,
                    "missing_predicates": d.missing_predicates,
                    "legal_reference": d.legal_reference,
                }
                for d in self.decisions
            ],
        }

    def report(self) -> str:
        head = (
            f"scene decision: {self.scene_label}; "
            f"violations: {self.violations or 'none'}; "
            f"iterations: {self.iterations} ({self.stop_reason})"
        )
        return "\n".join([head] + [d.explain() for d in self.decisions])


class RuleEngine:
    """Deterministic evaluation of the scenario bank."""

    def __init__(self, bank, cfg):
        self.bank = bank
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    def decide_score(self, gamma: float) -> str:
        if gamma >= self.cfg.gamma_strict:
            return VIOLATION
        if gamma >= self.cfg.gamma_loose:
            return INSUFFICIENT
        return NO_VIOLATION

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        description,
        gamma: torch.Tensor,
        index: int = 0,
        iterations: int = 0,
        stop_reason: str = "",
    ) -> EngineOutput:
        """Evaluate every scenario for one image of the batch."""
        gamma_row = (gamma[index] if gamma.dim() > 1 else gamma).detach()
        satisfaction = description.satisfaction[index].detach()
        decisions: List[ScenarioDecision] = []

        for j, scenario in enumerate(self.bank.scenarios):
            per_predicate = {}
            assignment = {}
            missing = []
            for predicate in scenario.predicates:
                p = self.bank.predicate_index[predicate.id]
                value = float(satisfaction[p])
                per_predicate[predicate.id] = value
                resolved = value >= self.cfg.gap_threshold
                assignment[predicate.id] = resolved
                if not resolved:
                    missing.append(predicate.id)
            decisions.append(
                ScenarioDecision(
                    scenario_id=scenario.id,
                    violation_type=scenario.violation_type,
                    decision=self.decide_score(float(gamma_row[j])),
                    gamma=float(gamma_row[j]),
                    predicate_satisfaction=per_predicate,
                    missing_predicates=missing,
                    rule_satisfied=scenario.rule.evaluate(assignment),
                    legal_reference=scenario.legal_reference,
                )
            )

        return EngineOutput(
            decisions=decisions,
            gamma=gamma_row.detach(),
            iterations=iterations,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------ #
    def evaluate_batch(self, description, gamma: torch.Tensor,
                       iterations: int = 0,
                       stop_reason: str = "") -> List[EngineOutput]:
        return [
            self.evaluate(description, gamma, i, iterations, stop_reason)
            for i in range(description.batch_size)
        ]

    def scene_labels(self, gamma: torch.Tensor) -> List[str]:
        """Fast path: scene-level V/U/N labels without the per-scenario report."""
        out = []
        for row in gamma:
            values = [self.decide_score(float(v)) for v in row]
            if VIOLATION in values:
                out.append(VIOLATION)
            elif INSUFFICIENT in values:
                out.append(INSUFFICIENT)
            else:
                out.append(NO_VIOLATION)
        return out
