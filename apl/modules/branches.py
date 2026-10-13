"""The three complementary feature branches of APL (Sec. V-B3..V-C).

* :class:`GlobalInvariantBranch`  - rule-conditioned, scene-agnostic ``w^GI``.
* :class:`InstanceDependentBranch` - image-conditioned ``h^ID = theta_FG(w, f_hat)``.
* :class:`FineGrainedBranch`       - evidence-gap conditioned ``h^FG_z`` scored
  against the top-``z*eta`` most similar local visual features.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=dim, eps=1e-8)


# --------------------------------------------------------------------------- #
class GlobalInvariantBranch(nn.Module):
    """``w^GI`` is derived from the base feature ``w`` and kept close to ``w^zs``.

    The conditioning signal is a small learnable, image-independent bank, so the
    branch stays anchored to rule-aligned semantics instead of adapting to every
    visual artifact.
    """

    def __init__(self, feature_generator, dim: int, num_tokens: int = 1,
                 init_std: float = 0.02):
        super().__init__()
        self.fg = feature_generator
        self.alpha = nn.Parameter(torch.empty(num_tokens, dim).normal_(std=init_std))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Returns ``w^GI`` of shape ``[C, d]``."""
        return self.fg(w, self.alpha)

    @staticmethod
    def logits(w_gi: torch.Tensor, f_hat: torch.Tensor,
               temperature: float) -> torch.Tensor:
        sim = cosine(f_hat.unsqueeze(1), w_gi.unsqueeze(0))   # [B, C]
        return sim / temperature


# --------------------------------------------------------------------------- #
class InstanceDependentBranch(nn.Module):
    """``h^ID_b = theta_FG(w, alpha^ID_b)`` with ``alpha^ID_b = f_hat_b``."""

    def __init__(self, feature_generator):
        super().__init__()
        self.fg = feature_generator

    def forward(self, w: torch.Tensor, f_hat: torch.Tensor) -> torch.Tensor:
        """Returns ``h^ID`` of shape ``[B, C, d]``."""
        alpha = f_hat.unsqueeze(1)                            # [B, 1, d]
        w_b = w.unsqueeze(0).expand(f_hat.shape[0], -1, -1)
        return self.fg(w_b, alpha)

    @staticmethod
    def logits(h_id: torch.Tensor, f_hat: torch.Tensor,
               temperature: float) -> torch.Tensor:
        sim = cosine(f_hat.unsqueeze(1), h_id)                # [B, C]
        return sim / temperature


# --------------------------------------------------------------------------- #
class FineGrainedBranch(nn.Module):
    """Learnable conditioning bank ``alpha^FG in R^{Z x d}`` plus local matching."""

    def __init__(self, feature_generator, dim: int, num_tokens: int = 4,
                 eta: float = 1.0, topk_mode: str = "proportion",
                 init_std: float = 0.02):
        super().__init__()
        self.fg = feature_generator
        self.num_tokens = num_tokens
        self.eta = eta
        self.topk_mode = topk_mode
        self.alpha = nn.Parameter(torch.empty(num_tokens, dim).normal_(std=init_std))

    # ------------------------------------------------------------------ #
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Returns ``h^FG`` of shape ``[Z, C, d]``."""
        feats = [self.fg(w, self.alpha[z: z + 1]) for z in range(self.num_tokens)]
        return torch.stack(feats, dim=0)

    # ------------------------------------------------------------------ #
    def top_k(self, z_index: int, num_local: int) -> int:
        """Number of local features pooled by the ``z``-th fine-grained feature.

        ``proportion`` (default) reads ``eta`` as the fraction of the ``N``
        local features used by the widest scale, so scale ``z`` pools
        ``ceil(z/Z * eta * N)`` patches - a genuine coarse-to-fine ladder.
        ``count`` follows the literal ``top-z*eta`` of Eq. (4).
        """
        z = z_index + 1
        if self.topk_mode == "count":
            k = int(math.ceil(z * self.eta))
        else:
            k = int(math.ceil(z / self.num_tokens * self.eta * num_local))
        return max(1, min(num_local, k))

    # ------------------------------------------------------------------ #
    def similarity(self, h_fg: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """``sim(h^FG_{z,m}, f)`` of shape ``[B, Z, C]``.

        For every scale ``z`` the cosine similarities against the ``N`` local
        features are ranked and averaged over the ``top-k_z`` of them.
        """
        # [B, Z, C, N]
        sim = torch.einsum(
            "zcd,bnd->bzcn",
            F.normalize(h_fg, dim=-1),
            F.normalize(local, dim=-1),
        )
        num_local = local.shape[1]
        pooled = []
        for z in range(self.num_tokens):
            k = self.top_k(z, num_local)
            top = sim[:, z].topk(k, dim=-1).values            # [B, C, k]
            pooled.append(top.mean(dim=-1))
        return torch.stack(pooled, dim=1)                     # [B, Z, C]

    @staticmethod
    def logits(sim: torch.Tensor, temperature: float) -> torch.Tensor:
        return sim / temperature
