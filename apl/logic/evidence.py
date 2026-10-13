"""Structured evidence extraction ``Phi_extract`` (Sec. V-D).

The generated text features are collected into

``T = {w^GI, h^ID, h^FG_1, ..., h^FG_Z}``

and every ``d``-dimensional row ``t_i in rows(T)`` is scored against the visual
embedding, ``z_i = cos(f_hat, t_i)``.  Because the rows of ``w`` are the
evidentiary predicates of the scenario bank, these scores are exactly the
continuous evidence that the discrete rule engine consumes.

Fine-grained rows are optionally scored against the *local* features with the
top-``k_z`` pooling of Eq. (4) (``logic.fg_local_evidence``), which is what the
fine-grained branch is built for; set the flag to ``False`` to score every row
against the global embedding as written in Sec. V-D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..modules.branches import cosine
from .soft_logic import SoftLogicGate, compatibility_scores, gap_mask


@dataclass
class TextFeatureSet:
    """``T_t`` - the generated text features at refinement step ``t``."""

    w_gi: Optional[torch.Tensor] = None      # [C, d]
    h_id: Optional[torch.Tensor] = None      # [B, C, d]
    h_fg: Optional[torch.Tensor] = None      # [Z, C, d]
    iteration: int = 0

    def rows(self) -> List[torch.Tensor]:
        out = []
        if self.w_gi is not None:
            out.append(self.w_gi)
        if self.h_id is not None:
            out.append(self.h_id)
        if self.h_fg is not None:
            out.append(self.h_fg)
        return out

    def stacked_rows(self) -> torch.Tensor:
        """All rows flattened to ``[R, d]`` (batch dimensions collapsed)."""
        flat = [r.reshape(-1, r.shape[-1]) for r in self.rows()]
        return torch.cat(flat, dim=0)

    def detach(self) -> "TextFeatureSet":
        return TextFeatureSet(
            w_gi=None if self.w_gi is None else self.w_gi.detach(),
            h_id=None if self.h_id is None else self.h_id.detach(),
            h_fg=None if self.h_fg is None else self.h_fg.detach(),
            iteration=self.iteration,
        )


@dataclass
class StructuredDescription:
    """``D_t`` - continuous evidence addressed by predicate."""

    predicate_ids: List[str]
    scores: torch.Tensor                      # [B, P] cosine evidence z_jq
    satisfaction: torch.Tensor                # [B, P] psi(z_jq)
    iteration: int = 0
    branch_scores: Dict[str, torch.Tensor] = field(default_factory=dict)
    text: Optional[List[str]] = None

    @property
    def batch_size(self) -> int:
        return self.scores.shape[0]

    def predicate_dict(self, index: int = 0) -> Dict[str, float]:
        return {
            pid: float(self.satisfaction[index, i])
            for i, pid in enumerate(self.predicate_ids)
        }

    def resolved(self, threshold: float = 0.5, index: int = 0) -> Dict[str, bool]:
        return {
            pid: bool(self.satisfaction[index, i] >= threshold)
            for i, pid in enumerate(self.predicate_ids)
        }


class EvidenceExtractor(nn.Module):
    """``Phi_extract(f_hat, f, T_t) -> D_t`` plus scenario matching helpers."""

    def __init__(self, bank, cfg, fine_grained_branch=None):
        super().__init__()
        self.bank = bank
        self.cfg = cfg
        self.fg_branch = fine_grained_branch
        self.gate = SoftLogicGate(
            bank.num_predicates, cfg.tau_init, cfg.kappa, cfg.learnable_tau
        )
        weights = bank.scenario_predicate_matrix()
        self.register_buffer("weights", weights, persistent=True)
        self.register_buffer("membership", (weights > 0).float(), persistent=True)

    # ------------------------------------------------------------------ #
    def forward(self, features: TextFeatureSet, f_hat: torch.Tensor,
                local: Optional[torch.Tensor] = None) -> StructuredDescription:
        return self.extract(features, f_hat, local)

    def extract(self, features: TextFeatureSet, f_hat: torch.Tensor,
                local: Optional[torch.Tensor] = None) -> StructuredDescription:
        branch_scores: Dict[str, torch.Tensor] = {}

        if features.w_gi is not None:
            branch_scores["gi"] = cosine(
                f_hat.unsqueeze(1), features.w_gi.unsqueeze(0)
            )                                                   # [B, C]
        if features.h_id is not None:
            branch_scores["id"] = cosine(f_hat.unsqueeze(1), features.h_id)
        if features.h_fg is not None:
            if (self.cfg.fg_local_evidence and local is not None
                    and self.fg_branch is not None):
                fg = self.fg_branch.similarity(features.h_fg, local)   # [B, Z, C]
            else:
                fg = cosine(
                    f_hat.unsqueeze(1).unsqueeze(1), features.h_fg.unsqueeze(0)
                )                                                      # [B, Z, C]
            branch_scores["fg"] = fg.mean(dim=1)

        if not branch_scores:
            raise ValueError("no text features available for evidence extraction")

        scores = torch.stack(list(branch_scores.values()), dim=0).mean(dim=0)
        satisfaction = self.gate(scores)
        return StructuredDescription(
            predicate_ids=[p.id for p in self.bank.predicates],
            scores=scores,
            satisfaction=satisfaction,
            iteration=features.iteration,
            branch_scores=branch_scores,
        )

    # ------------------------------------------------------------------ #
    def compatibility(self, description: StructuredDescription) -> torch.Tensor:
        """``gamma^{(t)}`` of shape ``[B, M]``."""
        return compatibility_scores(description.satisfaction, self.weights)

    def gaps(self, description: StructuredDescription) -> torch.Tensor:
        """``[B, M, P]`` boolean mask of missing / unresolved required predicates."""
        return gap_mask(
            description.satisfaction, self.membership, self.cfg.gap_threshold
        )

    def gap_set(self, description: StructuredDescription,
                index: int = 0) -> List[tuple]:
        """``G_t = union_j {(j, e) : e in G_{j,t}}`` for one image."""
        mask = self.gaps(description)[index]
        pairs = []
        for j, scenario in enumerate(self.bank.scenarios):
            for predicate in scenario.predicates:
                p = self.bank.predicate_index[predicate.id]
                if bool(mask[j, p]):
                    pairs.append((scenario.id, predicate.id))
        return pairs
