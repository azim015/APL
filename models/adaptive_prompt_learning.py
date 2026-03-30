"""
Adaptive Prompt Learning (APL) Module — Section V.B

Implements all three feature branches described in the paper:

    1. Global-Invariant (GI) features  — w_GI
       Stable, rule-aligned features regularised toward zero-shot representations.
       Loss: L_GI_CE = L_CE + λ1‖w_GI − w_zs‖ + λ2‖f_hat − f_hat_zs‖

    2. Instance-Dependent (ID) features — h_ID
       Scene-adaptive features conditioned on the global visual embedding.
       α_ID = f_hat  (the global visual embedding of the current image)
       Loss: L_ID_CE = standard cross-entropy over cosine similarities

    3. Fine-Grained Evidence (FG) features — h_FG
       Localised evidentiary features using top-zη spatial selection.
       α_FG ∈ R^{Z x d}  learnable parameters (optimised jointly)
       Loss: L_FG_CE = average cross-entropy over Z fine-grained features

Joint training objective:
    L_APL = λ_GI · L_GI_CE + λ_ID · L_ID_CE + λ_FG · L_FG_CE

Inference: average over all three branches
    p(m|I) = (1/3)[p_GI(m|I) + p_ID(m|I) + (1/Z)Σ_z p_FG_z(m|I)]
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_generator import FeatureGenerator


class AdaptivePromptLearning(nn.Module):
    """
    Adaptive Prompt Learning (APL) module.

    Args:
        feature_dim:          d — embedding dimension
        num_semantic_classes: M — number of scene categories
        num_fine_grained:     Z — number of fine-grained prompt features
        eta:                  η ∈ (0,1] — multi-scale spatial selection factor
        temperature:          τ — softmax temperature
        num_local_features:   N — number of local visual patches
        lambda_text_reg:      λ1 — GI text regularisation coefficient
        lambda_visual_reg:    λ2 — GI visual regularisation coefficient
        lambda_gi:            Loss weight for GI branch
        lambda_id:            Loss weight for ID branch
        lambda_fg:            Loss weight for FG branch
        num_heads:            Attention heads in FeatureGenerator
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_semantic_classes: int = 11,
        num_fine_grained: int = 4,
        eta: float = 0.5,
        temperature: float = 0.07,
        num_local_features: int = 196,
        lambda_text_reg: float = 0.25,
        lambda_visual_reg: float = 0.25,
        lambda_gi: float = 1.0,
        lambda_id: float = 1.0,
        lambda_fg: float = 1.0,
        num_heads: int = 8,
    ):
        super().__init__()
        self.d = feature_dim
        self.M = num_semantic_classes
        self.Z = num_fine_grained
        self.eta = eta
        self.tau = temperature
        self.N = num_local_features
        self.lambda1 = lambda_text_reg
        self.lambda2 = lambda_visual_reg
        self.lambda_gi = lambda_gi
        self.lambda_id = lambda_id
        self.lambda_fg = lambda_fg

        # Shared cross-attention Feature Generator (SFG)
        self.sfg = FeatureGenerator(feature_dim=feature_dim, num_heads=num_heads)

        # Learnable text prompt tokens  s ∈ R^{M x d}
        self.text_prompt_tokens = nn.Parameter(
            torch.randn(num_semantic_classes, feature_dim) * 0.02
        )
        # Learnable visual prompt tokens  u ∈ R^{num_heads x d}  (simplified)
        self.visual_prompt_tokens = nn.Parameter(
            torch.randn(num_heads, feature_dim) * 0.02
        )

        # GI branch: learnable global-invariant feature (initialised from text prompt)
        self.w_gi = nn.Parameter(
            torch.randn(num_semantic_classes, feature_dim) * 0.02
        )

        # FG branch: learnable conditional signals  α_FG ∈ R^{Z x d}
        self.alpha_fg = nn.Parameter(
            torch.randn(num_fine_grained, feature_dim) * 0.02
        )

        # Local feature projection  θ_Proj  (maps local patch features → d)
        self.theta_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.eye_(self.theta_proj.weight)  # start as identity

    # ==================================================================
    # Branch 1: Global-Invariant (GI) Features
    # ==================================================================

    def compute_gi_features(self) -> torch.Tensor:
        """
        Return the global-invariant feature w_GI.
        Normalised along the feature dimension.
        Shape: (M, d)
        """
        return F.normalize(self.w_gi, dim=-1)

    def gi_loss(
        self,
        f_hat: torch.Tensor,      # B x d  — global visual embedding
        labels: torch.Tensor,     # B      — class labels
        w_zs: torch.Tensor,       # M x d  — zero-shot text features
        f_hat_zs: torch.Tensor,   # B x d  — zero-shot visual features
    ) -> torch.Tensor:
        """
        GI branch loss (Eq. in Section V.B.5):
            L_GI_CE = L_CE(w_GI, f_hat) + λ1‖w_GI − w_zs‖ + λ2‖f_hat − f_hat_zs‖
        """
        w_gi = self.compute_gi_features()  # M x d

        # --- Cross-entropy over cosine similarity ---
        logits = f_hat @ w_gi.T / self.tau    # B x M
        ce_loss = F.cross_entropy(logits, labels)

        # --- Regularisation terms ---
        reg_text = torch.norm(w_gi - w_zs.detach(), p=2)
        reg_vis = torch.norm(f_hat - f_hat_zs.detach(), p=2).mean()

        return ce_loss + self.lambda1 * reg_text + self.lambda2 * reg_vis

    # ==================================================================
    # Branch 2: Instance-Dependent (ID) Features
    # ==================================================================

    def compute_id_features(
        self,
        w: torch.Tensor,       # M x d  — base text feature
        f_hat: torch.Tensor,   # B x d  — global visual embedding
    ) -> torch.Tensor:
        """
        Instance-dependent text features h_ID (Section V.B.3).
        α_ID = f_hat (global visual embedding of the current image).

        For a batch of B images, runs the SFG B times, one per image.
        Returns:  B x M x d
        """
        B = f_hat.size(0)
        # Expand w for each sample in the batch: B x M x d
        w_expanded = w.unsqueeze(0).expand(B, -1, -1)
        # α_ID for each image: B x 1 x d  (sequence length L=1)
        alpha_id = f_hat.unsqueeze(1)   # B x 1 x d
        # SFG: forward handles batched input
        h_id = self.sfg(w_expanded, alpha_id)   # B x M x d
        return F.normalize(h_id, dim=-1)

    def id_loss(
        self,
        w: torch.Tensor,        # M x d
        f_hat: torch.Tensor,    # B x d
        labels: torch.Tensor,   # B
    ) -> torch.Tensor:
        """
        ID branch cross-entropy loss (Eq. in Section V.B.3):
            p_b(m|I) = exp(cos(f_hat_b, h_ID_{b,m}) / τ) / Σ_j exp(...)
        """
        h_id = self.compute_id_features(w, f_hat)   # B x M x d
        # Cosine similarity between each image global feat and each class text feat
        # f_hat: B x d  →  B x 1 x d
        # h_id:  B x M x d
        f_hat_norm = F.normalize(f_hat, dim=-1).unsqueeze(1)   # B x 1 x d
        logits = (f_hat_norm * h_id).sum(dim=-1) / self.tau    # B x M
        return F.cross_entropy(logits, labels)

    # ==================================================================
    # Branch 3: Fine-Grained Evidence (FG) Features
    # ==================================================================

    def compute_fg_features(
        self, w: torch.Tensor
    ) -> torch.Tensor:
        """
        Fine-grained text features h_FG (Section V.B.4).
        α_FG ∈ R^{Z x d} are learnable parameters.
        The SFG is run once per fine-grained feature z.

        Returns:  Z x M x d
        """
        fg_features = []
        for z in range(self.Z):
            alpha_z = self.alpha_fg[z].unsqueeze(0)   # 1 x d  → treated as (L=1, d)
            h_z = self.sfg(w, alpha_z)                # M x d
            fg_features.append(F.normalize(h_z, dim=-1))
        return torch.stack(fg_features, dim=0)         # Z x M x d

    def _top_zeta_similarity(
        self,
        h_fg_z: torch.Tensor,   # M x d  — z-th fine-grained text feature
        f: torch.Tensor,        # B x N x d — local visual features
    ) -> torch.Tensor:
        """
        Multi-scale top-zη similarity aggregation (Section V.B.4).

            sim(h_FG_{z,m}, f) = (1/zη) Σ_n 1_{top-zη}(n) · cos(h_FG_{z,m}, f^n)

        where zη = max(1, round(η · N)).

        Returns:  B x M
        """
        B, N, d = f.shape
        M = h_fg_z.size(0)
        z_eta = max(1, round(self.eta * N))

        # Cosine similarity: B x N x M
        f_norm = F.normalize(f, dim=-1)                            # B x N x d
        h_norm = F.normalize(h_fg_z, dim=-1)                       # M x d
        cos_sim = torch.einsum("bnd,md->bnm", f_norm, h_norm)      # B x N x M

        # Select top-zη local features per class
        top_vals, _ = cos_sim.topk(z_eta, dim=1)                   # B x zη x M
        sim = top_vals.mean(dim=1)                                  # B x M
        return sim

    def fg_loss(
        self,
        w: torch.Tensor,        # M x d
        f: torch.Tensor,        # B x N x d  — local features
        labels: torch.Tensor,   # B
    ) -> torch.Tensor:
        """
        FG branch loss (Section V.B.4):
            L_FG_CE = (1/Z) Σ_z L_CE(p_FG_z)
        """
        h_fg = self.compute_fg_features(w)   # Z x M x d
        total_loss = torch.tensor(0.0, device=w.device)
        for z in range(self.Z):
            sim = self._top_zeta_similarity(h_fg[z], f)   # B x M
            logits = sim / self.tau
            total_loss = total_loss + F.cross_entropy(logits, labels)
        return total_loss / self.Z

    # ==================================================================
    # Joint inference prediction
    # ==================================================================

    def predict(
        self,
        w: torch.Tensor,        # M x d
        f_hat: torch.Tensor,    # B x d
        f: torch.Tensor,        # B x N x d
    ) -> torch.Tensor:
        """
        Ensemble prediction from all three branches (Section V.B.6):
            p(m|I) = (1/3)[p_GI + p_ID + (1/Z)Σ_z p_FG_z]

        Returns:  B x M  (log probabilities)
        """
        B, M = f_hat.size(0), self.M

        # GI branch
        w_gi = self.compute_gi_features()                            # M x d
        f_hat_norm = F.normalize(f_hat, dim=-1)                      # B x d
        logits_gi = f_hat_norm @ w_gi.T / self.tau                   # B x M
        p_gi = F.softmax(logits_gi, dim=-1)                          # B x M

        # ID branch
        h_id = self.compute_id_features(w, f_hat)                   # B x M x d
        f_hat_exp = f_hat_norm.unsqueeze(1)                          # B x 1 x d
        logits_id = (f_hat_exp * h_id).sum(-1) / self.tau           # B x M
        p_id = F.softmax(logits_id, dim=-1)                          # B x M

        # FG branch
        h_fg = self.compute_fg_features(w)                           # Z x M x d
        p_fg_sum = torch.zeros(B, M, device=w.device)
        for z in range(self.Z):
            sim = self._top_zeta_similarity(h_fg[z], f)             # B x M
            p_fg_sum = p_fg_sum + F.softmax(sim / self.tau, dim=-1)
        p_fg = p_fg_sum / self.Z                                     # B x M

        # Ensemble average
        p = (p_gi + p_id + p_fg) / 3.0   # B x M
        return torch.log(p + 1e-8)       # log probabilities

    # ==================================================================
    # Joint training loss
    # ==================================================================

    def compute_loss(
        self,
        w: torch.Tensor,          # M x d   — base text feature
        f_hat: torch.Tensor,      # B x d   — global visual embedding
        f: torch.Tensor,          # B x N x d — local visual features
        labels: torch.Tensor,     # B       — ground-truth class labels
        w_zs: torch.Tensor,       # M x d   — zero-shot text features
        f_hat_zs: torch.Tensor,   # B x d   — zero-shot visual features
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the joint APL training objective:
            L_APL = λ_GI · L_GI_CE + λ_ID · L_ID_CE + λ_FG · L_FG_CE

        Returns a dict with individual and total losses.
        """
        l_gi = self.gi_loss(f_hat, labels, w_zs, f_hat_zs)
        l_id = self.id_loss(w, f_hat, labels)
        l_fg = self.fg_loss(w, f, labels)

        total = (
            self.lambda_gi * l_gi
            + self.lambda_id * l_id
            + self.lambda_fg * l_fg
        )

        return {
            "loss_gi": l_gi,
            "loss_id": l_id,
            "loss_fg": l_fg,
            "loss_total": total,
        }

    # ==================================================================
    # Gap-conditioned prompt refinement (Eq. 3 / Section V.F)
    # ==================================================================

    def refine_with_gaps(
        self,
        w: torch.Tensor,           # M x d  — current base feature
        gap_signal: torch.Tensor,  # L x d  — α_gap: gap-conditioned signal
    ) -> torch.Tensor:
        """
        Generate updated text feature conditioned on evidence gaps.
        Used in the iterative refinement loop (Eq. 3):
            T_{t+1} = θ_SFG(T_t, α_gap(G_t))

        Returns: M x d
        """
        h_refined = self.sfg(w, gap_signal)
        return F.normalize(h_refined, dim=-1)

    # ==================================================================
    # Forward (convenience wrapper)
    # ==================================================================

    def forward(
        self,
        w: torch.Tensor,
        f_hat: torch.Tensor,
        f: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        w_zs: Optional[torch.Tensor] = None,
        f_hat_zs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward: computes features + losses.
        Inference forward: computes predictions (if labels is None).

        Returns dict with features, predictions, and optionally losses.
        """
        output: Dict[str, torch.Tensor] = {}

        # Always compute features
        output["w_gi"] = self.compute_gi_features()               # M x d
        output["h_id"] = self.compute_id_features(w, f_hat)       # B x M x d
        output["h_fg"] = self.compute_fg_features(w)              # Z x M x d

        if labels is not None:
            assert w_zs is not None and f_hat_zs is not None, (
                "w_zs and f_hat_zs are required for training loss computation."
            )
            losses = self.compute_loss(w, f_hat, f, labels, w_zs, f_hat_zs)
            output.update(losses)

        # Always produce predictions
        output["log_probs"] = self.predict(w, f_hat, f)           # B x M

        return output
