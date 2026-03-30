"""
Evidence Extractor — Section V.D

Implements:
    D_t = Φ_extract(f_hat, f, T_t)

Fuses global + local visual features with the multi-perspective text features
(w_GI, h_ID, h_FG) from the APL module to produce a structured scene
representation as a dictionary of cosine similarity scores over semantic classes.

The output is used by the ScenarioBankMatcher for violation detection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceExtractor(nn.Module):
    """
    Structured Evidence Extractor.

    Computes cosine similarity scores between visual features and
    all generated text features:

        T = {w_GI, h_ID, h_FG_1, ..., h_FG_Z}
        z_i = cos(f_hat, t_i)   for t_i ∈ T

    These scores form the "structured description" D_t that is
    passed to the scenario bank matcher.

    Args:
        feature_dim:          d — embedding dimension
        num_semantic_classes: M — number of scene categories
        num_fine_grained:     Z — number of FG features
        temperature:          τ — softmax temperature
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_semantic_classes: int = 11,
        num_fine_grained: int = 4,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.d = feature_dim
        self.M = num_semantic_classes
        self.Z = num_fine_grained
        self.tau = temperature

    # ------------------------------------------------------------------

    def _cosine_sim_global(
        self,
        f_hat: torch.Tensor,    # B x d
        text_feat: torch.Tensor,  # M x d  (or B x M x d for ID)
    ) -> torch.Tensor:
        """
        Compute cosine similarity between global image feature and text features.
        Handles both (M, d) and (B, M, d) text feature shapes.

        Returns: B x M
        """
        f_norm = F.normalize(f_hat, dim=-1)                      # B x d

        if text_feat.dim() == 2:
            # (M, d) → shared across batch
            t_norm = F.normalize(text_feat, dim=-1)              # M x d
            return f_norm @ t_norm.T                             # B x M
        else:
            # (B, M, d) → per-sample
            t_norm = F.normalize(text_feat, dim=-1)              # B x M x d
            f_exp = f_norm.unsqueeze(1)                          # B x 1 x d
            return (f_exp * t_norm).sum(dim=-1)                  # B x M

    # ------------------------------------------------------------------

    def forward(
        self,
        f_hat: torch.Tensor,         # B x d  — global visual embedding
        f: torch.Tensor,             # B x N x d — local visual features
        w_gi: torch.Tensor,          # M x d  — GI text feature
        h_id: torch.Tensor,          # B x M x d — ID text features
        h_fg: torch.Tensor,          # Z x M x d — FG text features
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the structured evidence description D_t.

        Returns a dictionary with keys:
            "scores_gi":    B x M   — GI branch similarity scores
            "scores_id":    B x M   — ID branch similarity scores
            "scores_fg":    B x Z x M — FG branch similarity scores (per z)
            "scores_mean":  B x M   — ensemble average similarity scores
            "probs_mean":   B x M   — softmax probabilities from ensemble
        """
        B = f_hat.size(0)

        # --- GI scores ---
        scores_gi = self._cosine_sim_global(f_hat, w_gi)              # B x M

        # --- ID scores ---
        scores_id = self._cosine_sim_global(f_hat, h_id)              # B x M

        # --- FG scores (multi-scale spatial selection) ---
        scores_fg_list = []
        for z in range(self.Z):
            h_fg_z = h_fg[z]                                           # M x d
            # Top-k spatial selection (delegate to similarity fn)
            f_norm = F.normalize(f, dim=-1)                            # B x N x d
            h_norm = F.normalize(h_fg_z, dim=-1)                      # M x d
            cos_sim = torch.einsum("bnd,md->bnm", f_norm, h_norm)     # B x N x M
            # For the evidence extractor, use mean over all local features
            # (spatial selection with η is handled in APL training)
            scores_fg_z = cos_sim.mean(dim=1)                         # B x M
            scores_fg_list.append(scores_fg_z)
        scores_fg = torch.stack(scores_fg_list, dim=1)                 # B x Z x M

        # --- Ensemble average ---
        scores_fg_mean = scores_fg.mean(dim=1)                         # B x M
        scores_mean = (scores_gi + scores_id + scores_fg_mean) / 3.0  # B x M
        probs_mean = F.softmax(scores_mean / self.tau, dim=-1)         # B x M

        return {
            "scores_gi": scores_gi,
            "scores_id": scores_id,
            "scores_fg": scores_fg,
            "scores_mean": scores_mean,
            "probs_mean": probs_mean,
        }

    # ------------------------------------------------------------------

    def identify_evidence_gaps(
        self,
        description: Dict[str, torch.Tensor],
        required_predicates: List[int],
        threshold: float = 0.3,
    ) -> List[int]:
        """
        Identify evidentiary gaps — predicate indices where the confidence
        score falls below the threshold.

        Args:
            description:         Output from forward()
            required_predicates: List of semantic class indices required
                                 by the matched DMV scenario
            threshold:           Minimum confidence to consider a predicate satisfied

        Returns:
            List of predicate indices that are below threshold (gaps)
        """
        probs = description["probs_mean"]   # B x M
        # Use first image in batch for gap detection (could be extended)
        probs_0 = probs[0]                  # M
        gaps = [
            p for p in required_predicates
            if probs_0[p].item() < threshold
        ]
        return gaps

    def build_gap_signal(
        self,
        gaps: List[int],
        w: torch.Tensor,       # M x d — base text feature
        device: torch.device,
    ) -> torch.Tensor:
        """
        Construct the gap-conditioned signal α_gap for prompt refinement.
        The signal is the average of base features for the gap predicates.

        Returns: L x d  where L = len(gaps) (or 1 if empty)
        """
        if not gaps:
            # No gaps — return a zero signal
            return torch.zeros(1, self.d, device=device)
        gap_indices = torch.tensor(gaps, device=device)
        gap_feats = w[gap_indices]   # len(gaps) x d
        return gap_feats             # L x d
