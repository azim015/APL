"""
Rule-Based Violation Decision Engine — Section V.G

After the iterative refinement loop converges (Δγ^(t) < ε or t ≥ T_max),
the rule engine evaluates the final structured evidence and issues a
legally interpretable decision:

    y = { violation,            if γ^(T_max) ≥ Γ_strict
        { insufficient_evidence, if Γ_loose ≤ γ^(T_max) < Γ_strict
        { no_violation,          otherwise

Uses differentiable probabilistic logic for robustness against visual noise
while maintaining deterministic, interpretable output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------

class ViolationDecision(str, Enum):
    VIOLATION = "violation"
    NO_VIOLATION = "no_violation"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ---------------------------------------------------------------------------
# Decision result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    """
    Output of the RuleEngine for a single image (or batch item).

    Attributes:
        decision:        One of ViolationDecision enum values
        confidence:      Compatibility score γ (float)
        scenario_type:   The matched violation type (e.g. "red_light_violation")
        scenario_idx:    Index of the matched scenario in the bank
        num_iterations:  Number of refinement iterations performed
        gamma_history:   Sequence of γ values across iterations
    """
    decision: ViolationDecision
    confidence: float
    scenario_type: str
    scenario_idx: int
    num_iterations: int
    gamma_history: List[float]


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

class RuleEngine(nn.Module):
    """
    Deterministic Rule-Based Violation Decision Engine (Section V.G).

    Converts a final compatibility score γ into a three-way decision
    using legally-defined confidence bounds Γ_strict and Γ_loose.

    Args:
        gamma_strict:  Γ_strict — violation confidence threshold (default 0.75)
        gamma_loose:   Γ_loose  — insufficient evidence threshold (default 0.40)
    """

    def __init__(
        self,
        gamma_strict: float = 0.75,
        gamma_loose: float = 0.40,
    ):
        super().__init__()
        assert 0.0 < gamma_loose < gamma_strict <= 1.0, (
            f"Expected 0 < gamma_loose ({gamma_loose}) < gamma_strict ({gamma_strict}) ≤ 1."
        )
        self.gamma_strict = gamma_strict
        self.gamma_loose = gamma_loose

    # ------------------------------------------------------------------

    def decide(
        self,
        gamma: float,
        scenario_type: str = "unknown",
        scenario_idx: int = 0,
        num_iterations: int = 1,
        gamma_history: Optional[List[float]] = None,
    ) -> DecisionResult:
        """
        Apply the three-way decision rule for a single confidence score.

        Args:
            gamma:          Final compatibility score γ ∈ [0, 1]
            scenario_type:  String label of the matched scenario
            scenario_idx:   Index of the matched scenario
            num_iterations: Number of refinement iterations run
            gamma_history:  Per-iteration γ values (for diagnostics)

        Returns:
            DecisionResult with the final decision and metadata.
        """
        if gamma >= self.gamma_strict:
            decision = ViolationDecision.VIOLATION
        elif gamma >= self.gamma_loose:
            decision = ViolationDecision.INSUFFICIENT_EVIDENCE
        else:
            decision = ViolationDecision.NO_VIOLATION

        return DecisionResult(
            decision=decision,
            confidence=float(gamma),
            scenario_type=scenario_type,
            scenario_idx=scenario_idx,
            num_iterations=num_iterations,
            gamma_history=gamma_history or [gamma],
        )

    # ------------------------------------------------------------------

    def decide_batch(
        self,
        gamma: torch.Tensor,     # B — compatibility scores
        scenario_type: str = "unknown",
        scenario_idx: int = 0,
        num_iterations: int = 1,
        gamma_history: Optional[List[float]] = None,
    ) -> List[DecisionResult]:
        """
        Apply the three-way decision rule for a batch of confidence scores.

        Returns a list of DecisionResult objects (one per image in batch).
        """
        results = []
        for i in range(gamma.size(0)):
            g = gamma[i].item()
            results.append(
                self.decide(
                    g,
                    scenario_type=scenario_type,
                    scenario_idx=scenario_idx,
                    num_iterations=num_iterations,
                    gamma_history=gamma_history,
                )
            )
        return results

    # ------------------------------------------------------------------

    def forward(
        self,
        gamma: torch.Tensor,
        scenario_type: str = "unknown",
        scenario_idx: int = 0,
        num_iterations: int = 1,
        gamma_history: Optional[List[float]] = None,
    ) -> List[DecisionResult]:
        """Alias for decide_batch — called as a module."""
        return self.decide_batch(
            gamma, scenario_type, scenario_idx, num_iterations, gamma_history
        )

    # ------------------------------------------------------------------

    def convergence_check(
        self,
        gamma_prev: float,
        gamma_curr: float,
        epsilon: float = 0.01,
    ) -> bool:
        """
        Check convergence criterion: Δγ^(t) < ε.

        Returns True if the loop should terminate.
        """
        return abs(gamma_curr - gamma_prev) < epsilon

    # ------------------------------------------------------------------

    def format_results(self, results: List[DecisionResult]) -> str:
        """Human-readable summary of batch decisions."""
        lines = []
        for i, r in enumerate(results):
            lines.append(
                f"  [Image {i}] Decision: {r.decision.value:28s} | "
                f"Confidence: {r.confidence:.4f} | "
                f"Scenario: {r.scenario_type} | "
                f"Iterations: {r.num_iterations}"
            )
        return "\n".join(lines)
