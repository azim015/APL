"""
Feature Generator / Surrogate Feature Generator (SFG) — Section V.B.2

Implements the cross-attention module θ_SFG described in the paper:

    h = θ_SFG(w, α)

where:
    w  — base prompted text feature   shape: (M, d)   [query]
    α  — conditional signal           shape: (L, d)   [key / value]
    h  — synthesised text feature     shape: (M, d)

Mathematical formulation (from the paper):
    Q = LN(w) · θ_in1
    K = LN(α) · θ_in2
    V = LN(α) · θ_in3

    Attn = w + softmax(QK^T / √d_k) · V · θ_out

    h = Attn + ReLU(LN(Attn) · θ_fc1) · θ_fc2

with a two-layer MLP bottleneck  d_mid = d / 2.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGenerator(nn.Module):
    """
    Cross-attention Surrogate Feature Generator (SFG).

    Given a query substrate w and a conditional signal α, synthesises
    a new feature h that encodes the perspective captured by α.

    Args:
        feature_dim: Embedding dimension d
        num_heads:   Number of attention heads (default 8)

    Inputs:
        w: (M, d)  or  (B, M, d) — base text feature (query)
        α: (L, d)  or  (B, L, d) — conditional signal (key / value)

    Output:
        h: same shape as w  — synthesised text feature
    """

    def __init__(self, feature_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dk = feature_dim // num_heads  # per-head key dimension

        # Input projection layers  θ_in1, θ_in2, θ_in3 ∈ R^{d x d}
        self.theta_in1 = nn.Linear(feature_dim, feature_dim, bias=False)  # query proj
        self.theta_in2 = nn.Linear(feature_dim, feature_dim, bias=False)  # key   proj
        self.theta_in3 = nn.Linear(feature_dim, feature_dim, bias=False)  # value proj

        # Output projection  θ_out ∈ R^{d x d}
        self.theta_out = nn.Linear(feature_dim, feature_dim, bias=False)

        # Layer norms for query and key/value
        self.ln_query = nn.LayerNorm(feature_dim)
        self.ln_kv = nn.LayerNorm(feature_dim)
        self.ln_attn = nn.LayerNorm(feature_dim)

        # MLP bottleneck  θ_fc1 ∈ R^{d x d_mid},  θ_fc2 ∈ R^{d_mid x d}
        d_mid = feature_dim // 2
        self.theta_fc1 = nn.Linear(feature_dim, d_mid, bias=True)
        self.theta_fc2 = nn.Linear(d_mid, feature_dim, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for module in [self.theta_in1, self.theta_in2, self.theta_in3, self.theta_out]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.theta_fc1.weight)
        nn.init.zeros_(self.theta_fc1.bias)
        nn.init.xavier_uniform_(self.theta_fc2.weight)
        nn.init.zeros_(self.theta_fc2.bias)

    # ------------------------------------------------------------------

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention.
        Supports both batched (B, seq, d) and unbatched (seq, d) inputs.
        """
        scale = math.sqrt(self.feature_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (..., M, L)
        weights = F.softmax(scores, dim=-1)                      # (..., M, L)
        return torch.matmul(weights, V)                          # (..., M, d)

    # ------------------------------------------------------------------

    def forward(self, w: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SFG cross-attention module.

        Args:
            w:     Base text feature  — (M, d) or (B, M, d)
            alpha: Conditional signal — (L, d) or (B, L, d)

        Returns:
            h: Synthesised feature — same shape as w
        """
        # Handle unbatched input by adding a batch dim
        unbatched = w.dim() == 2
        if unbatched:
            w = w.unsqueeze(0)          # 1 x M x d
            alpha = alpha.unsqueeze(0)  # 1 x L x d

        # --- Input projections with layer norm ---
        Q = self.theta_in1(self.ln_query(w))       # B x M x d
        K = self.theta_in2(self.ln_kv(alpha))       # B x L x d
        V = self.theta_in3(self.ln_kv(alpha))       # B x L x d

        # --- Scaled dot-product attention + residual + output projection ---
        attn_out = self._scaled_dot_product_attention(Q, K, V)  # B x M x d
        attn_out = self.theta_out(attn_out)
        attn = w + attn_out                                      # B x M x d  (residual)

        # --- MLP bottleneck + residual ---
        mlp_out = self.theta_fc2(F.relu(self.theta_fc1(self.ln_attn(attn))))
        h = attn + mlp_out                                       # B x M x d

        if unbatched:
            h = h.squeeze(0)   # M x d

        return h
