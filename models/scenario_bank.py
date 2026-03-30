"""
Scenario Bank Matcher — Section V.E

Implements:
    - DMVScenario: data class for a single violation scenario  s_j = (ν_j, E_j, r_j)
    - ScenarioBankMatcher: computes compatibility scores γ_j^(t) via
      differentiable soft-logic gates ψ(z_jq)

Differentiable soft-logic gate (scaled sigmoid):
    ψ(z_jq) = 1 / (1 + exp(-κ(z_jq − τ_q)))

Weighted compatibility score:
    γ_j^(t) = Σ_q ω_jq · ψ(z_jq)

The function bridges the continuous probability space of the VLM
and the discrete Boolean logic of the rule engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data class for a single DMV violation scenario
# ---------------------------------------------------------------------------

@dataclass
class DMVScenario:
    """
    A single DMV-aligned violation scenario  s_j = (ν_j, E_j, r_j).

    Attributes:
        violation_type:   ν_j — string label, e.g. "red_light_violation"
        predicate_names:  Human-readable names of evidentiary predicates
        predicate_indices: Indices into the M-dim semantic class vector
        predicate_weights: ω_jq — regulatory weight for each predicate
        description:      Human-readable description of the scenario
    """
    violation_type: str
    predicate_names: List[str]
    predicate_indices: List[int]
    predicate_weights: List[float]
    description: str = ""

    def __post_init__(self):
        assert len(self.predicate_names) == len(self.predicate_indices) == len(
            self.predicate_weights
        ), "predicate_names, predicate_indices, and predicate_weights must have equal length."


# ---------------------------------------------------------------------------
# Built-in scenario bank (DMV-aligned, ITS-relevant)
# ---------------------------------------------------------------------------

def build_default_scenario_bank() -> List[DMVScenario]:
    """
    Returns a default set of DMV-aligned traffic violation scenarios
    covering common ITS use cases.

    Predicate indices map to the 11-class semantic space:
        0: vehicle       1: pedestrian    2: traffic_light
        3: stop_sign     4: lane_marking  5: road_sign
        6: cyclist       7: intersection  8: crosswalk
        9: parking_zone  10: construction_zone
    """
    return [
        DMVScenario(
            violation_type="red_light_violation",
            predicate_names=["traffic_light", "vehicle", "intersection"],
            predicate_indices=[2, 0, 7],
            predicate_weights=[0.5, 0.3, 0.2],
            description="Vehicle crosses intersection while traffic light is red.",
        ),
        DMVScenario(
            violation_type="stop_sign_violation",
            predicate_names=["stop_sign", "vehicle", "intersection"],
            predicate_indices=[3, 0, 7],
            predicate_weights=[0.5, 0.3, 0.2],
            description="Vehicle fails to stop at a stop sign.",
        ),
        DMVScenario(
            violation_type="illegal_lane_change",
            predicate_names=["vehicle", "lane_marking"],
            predicate_indices=[0, 4],
            predicate_weights=[0.4, 0.6],
            description="Vehicle crosses solid lane markings illegally.",
        ),
        DMVScenario(
            violation_type="pedestrian_right_of_way",
            predicate_names=["pedestrian", "crosswalk", "vehicle"],
            predicate_indices=[1, 8, 0],
            predicate_weights=[0.4, 0.4, 0.2],
            description="Vehicle fails to yield to pedestrian in crosswalk.",
        ),
        DMVScenario(
            violation_type="illegal_parking",
            predicate_names=["vehicle", "parking_zone", "road_sign"],
            predicate_indices=[0, 9, 5],
            predicate_weights=[0.3, 0.5, 0.2],
            description="Vehicle parked in a prohibited zone.",
        ),
        DMVScenario(
            violation_type="construction_zone_speeding",
            predicate_names=["vehicle", "construction_zone", "road_sign"],
            predicate_indices=[0, 10, 5],
            predicate_weights=[0.4, 0.3, 0.3],
            description="Vehicle exceeds speed limit in a construction zone.",
        ),
    ]


# ---------------------------------------------------------------------------
# Soft-logic gate ψ
# ---------------------------------------------------------------------------

def soft_logic_gate(
    z: torch.Tensor,   # cosine similarity scores — arbitrary shape
    tau_q: float,      # predicate-specific threshold
    kappa: float = 10.0,  # steepness of Boolean transition
) -> torch.Tensor:
    """
    Differentiable soft-logic gate (Eq. in Section V.E):
        ψ(z) = σ(κ(z − τ_q))   where σ is the sigmoid function

    Maps continuous similarity scores to [0, 1] satisfaction probabilities.
    """
    return torch.sigmoid(kappa * (z - tau_q))


# ---------------------------------------------------------------------------
# Scenario Bank Matcher
# ---------------------------------------------------------------------------

class ScenarioBankMatcher(nn.Module):
    """
    Scenario Bank Matcher (Section V.E).

    For each DMV scenario in the bank, computes a compatibility score γ_j^(t)
    that measures how well the current evidence description matches the
    required evidentiary predicates.

    The predicate-specific thresholds τ_q are learnable parameters
    (one per unique predicate index in the bank).

    Args:
        scenarios:    List of DMVScenario objects
        kappa:        Temperature scaling κ for sigmoid sharpness
        num_classes:  M — number of semantic classes
    """

    def __init__(
        self,
        scenarios: Optional[List[DMVScenario]] = None,
        kappa: float = 10.0,
        num_classes: int = 11,
    ):
        super().__init__()
        self.kappa = kappa
        self.num_classes = num_classes

        if scenarios is None:
            scenarios = build_default_scenario_bank()
        self.scenarios = scenarios

        # Collect all unique predicate indices across all scenarios
        all_pred_indices = sorted(
            {idx for s in scenarios for idx in s.predicate_indices}
        )
        self.pred_index_to_param_idx = {
            pred_idx: param_idx
            for param_idx, pred_idx in enumerate(all_pred_indices)
        }

        # Learnable predicate-specific thresholds τ_q ∈ R (one per unique predicate)
        num_unique = len(all_pred_indices)
        self.tau_q = nn.Parameter(torch.full((num_unique,), 0.3))

    # ------------------------------------------------------------------

    def _get_tau(self, predicate_index: int) -> torch.Tensor:
        """Return the learnable threshold for a given predicate index."""
        param_idx = self.pred_index_to_param_idx.get(predicate_index, 0)
        return self.tau_q[param_idx]

    # ------------------------------------------------------------------

    def compute_compatibility(
        self,
        probs: torch.Tensor,   # B x M — ensemble probability scores
        scenario_idx: int,
    ) -> torch.Tensor:
        """
        Compute the compatibility score γ_j for a single scenario (Section V.E):
            γ_j = Σ_q ω_jq · ψ(z_jq)

        Args:
            probs:        B x M — probability scores from evidence extractor
            scenario_idx: Index into self.scenarios

        Returns:
            gamma: B — compatibility scores (one per image)
        """
        scenario = self.scenarios[scenario_idx]
        B = probs.size(0)
        gamma = torch.zeros(B, device=probs.device)

        for pred_name, pred_idx, weight in zip(
            scenario.predicate_names,
            scenario.predicate_indices,
            scenario.predicate_weights,
        ):
            if pred_idx >= self.num_classes:
                continue
            z_jq = probs[:, pred_idx]           # B
            tau_q = self._get_tau(pred_idx)
            psi = soft_logic_gate(z_jq, tau_q.item(), self.kappa)  # B
            gamma = gamma + weight * psi

        return gamma   # B

    # ------------------------------------------------------------------

    def forward(
        self,
        probs: torch.Tensor,   # B x M
    ) -> Tuple[torch.Tensor, int, List[int]]:
        """
        Match evidence description against all scenarios.

        Returns:
            gamma_best:   B — compatibility scores for the best-matching scenario
            best_idx:     Index of the best-matching scenario
            gap_predicates: List of predicate indices with low satisfaction
                            for the best-matching scenario (first image in batch)
        """
        all_gammas = []
        for j in range(len(self.scenarios)):
            gamma_j = self.compute_compatibility(probs, j)   # B
            all_gammas.append(gamma_j)

        gamma_stack = torch.stack(all_gammas, dim=1)         # B x num_scenarios
        best_idx = int(gamma_stack[0].argmax().item())
        gamma_best = gamma_stack[:, best_idx]                # B

        # Identify evidence gaps for the best scenario
        scenario = self.scenarios[best_idx]
        gap_predicates: List[int] = []
        probs_0 = probs[0]   # first image
        for pred_idx, pred_name in zip(
            scenario.predicate_indices, scenario.predicate_names
        ):
            if pred_idx < self.num_classes:
                tau_q = self._get_tau(pred_idx).item()
                if probs_0[pred_idx].item() < tau_q:
                    gap_predicates.append(pred_idx)

        return gamma_best, best_idx, gap_predicates

    # ------------------------------------------------------------------

    def get_required_predicates(self, scenario_idx: int) -> List[int]:
        """Return the predicate indices required by scenario scenario_idx."""
        return self.scenarios[scenario_idx].predicate_indices
