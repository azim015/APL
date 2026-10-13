"""APL training objective (Sec. V-B3..V-B6, Sec. V-H).

.. math::

    \\mathcal{L}^{APL}_{\\phi}
      = \\lambda_{GI}\\mathcal{L}^{GI}_{CE}
      + \\lambda_{ID}\\mathcal{L}^{ID}_{CE}
      + \\lambda_{FG}\\mathcal{L}^{FG}_{CE}

with
``L^GI_CE = L_CE + lambda1 ||w^GI - w^zs|| + lambda2 ||f_hat - f_hat^zs||``,
``L^ID_CE`` the cross entropy over ``cos(f_hat, h^ID)/tau`` and ``L^FG_CE`` the
mean cross entropy over the ``Z`` fine-grained scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    terms: Dict[str, torch.Tensor] = field(default_factory=dict)

    def item_dict(self) -> Dict[str, float]:
        out = {"total": float(self.total.detach())}
        out.update({k: float(v.detach()) for k, v in self.terms.items()})
        return out


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def global_invariant_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_gi: torch.Tensor,
    w_zs: Optional[torch.Tensor],
    f_hat: torch.Tensor,
    f_hat_zs: Optional[torch.Tensor],
    lambda1: float,
    lambda2: float,
) -> torch.Tensor:
    """``L^GI_CE`` with the two invariance regularisers."""
    loss = cross_entropy(logits, targets)
    if w_zs is not None and lambda1 > 0:
        loss = loss + lambda1 * torch.norm(w_gi - w_zs.to(w_gi), p=2, dim=-1).mean()
    if f_hat_zs is not None and lambda2 > 0:
        loss = loss + lambda2 * torch.norm(
            f_hat - f_hat_zs.to(f_hat), p=2, dim=-1
        ).mean()
    return loss


def fine_grained_loss(logits_z: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross entropy over the ``Z`` fine-grained scales; ``logits_z`` is ``[B, Z, C]``."""
    z = logits_z.shape[1]
    total = logits_z.new_zeros(())
    for i in range(z):
        total = total + cross_entropy(logits_z[:, i], targets)
    return total / z


def apl_loss(
    logits_gi: Optional[torch.Tensor],
    logits_id: Optional[torch.Tensor],
    logits_fg: Optional[torch.Tensor],
    targets: torch.Tensor,
    cfg,
    w_gi: Optional[torch.Tensor] = None,
    w_zs: Optional[torch.Tensor] = None,
    f_hat: Optional[torch.Tensor] = None,
    f_hat_zs: Optional[torch.Tensor] = None,
) -> LossOutput:
    terms: Dict[str, torch.Tensor] = {}
    device = targets.device
    total = torch.zeros((), device=device)

    if logits_gi is not None:
        gi = global_invariant_loss(
            logits_gi, targets, w_gi, w_zs, f_hat, f_hat_zs,
            cfg.lambda1, cfg.lambda2,
        )
        terms["gi"] = gi
        total = total + cfg.lambda_gi * gi
    if logits_id is not None:
        idl = cross_entropy(logits_id, targets)
        terms["id"] = idl
        total = total + cfg.lambda_id * idl
    if logits_fg is not None:
        fgl = fine_grained_loss(logits_fg, targets)
        terms["fg"] = fgl
        total = total + cfg.lambda_fg * fgl

    if not terms:
        raise ValueError("at least one APL branch must be enabled")
    return LossOutput(total=total, terms=terms)


def ensemble_probability(
    logits_gi: Optional[torch.Tensor],
    logits_id: Optional[torch.Tensor],
    logits_fg: Optional[torch.Tensor],
) -> torch.Tensor:
    """``p(m|I) = 1/3 [ p^GI + p^ID + 1/Z sum_z p^FG_z ]`` over the active branches."""
    parts = []
    if logits_gi is not None:
        parts.append(logits_gi.softmax(dim=-1))
    if logits_id is not None:
        parts.append(logits_id.softmax(dim=-1))
    if logits_fg is not None:
        parts.append(logits_fg.softmax(dim=-1).mean(dim=1))
    if not parts:
        raise ValueError("no branch produced logits")
    return torch.stack(parts, dim=0).mean(dim=0)
