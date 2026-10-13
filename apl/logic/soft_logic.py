"""Differentiable logic interface between the VLM and the rule engine (Sec. V-E).

The calibrated soft gate

.. math::  \\psi(z_{jq}) = \\frac{1}{1 + \\exp(-\\kappa (z_{jq} - \\tau_q))}

turns a cosine similarity into a predicate satisfaction probability.  ``tau_q``
is a *predicate-specific learnable threshold* (initialised at 0.25) that
calibrates how hard a given piece of evidence is to observe - "red light"
versus "seatbelt" - and ``kappa`` (10) controls the steepness of the Boolean
transition.

The weighted compatibility score of scenario ``s_j`` is

.. math::  \\gamma_j^{(t)} = \\sum_q \\omega_{jq}\\,\\psi(z_{jq}) \\in [0, 1],
           \\qquad \\omega_{jq} = 1/Q_j .
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SoftLogicGate(nn.Module):
    """Calibrated sigmoid gate ``psi`` with per-predicate thresholds."""

    def __init__(self, num_predicates: int, tau_init: float = 0.25,
                 kappa: float = 10.0, learnable: bool = True):
        super().__init__()
        tau = torch.full((num_predicates,), float(tau_init))
        if learnable:
            self.tau = nn.Parameter(tau)
        else:
            self.register_buffer("tau", tau)
        self.kappa = float(kappa)
        self.num_predicates = num_predicates

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: ``[..., P]`` cosine scores -> ``[..., P]`` satisfactions."""
        return torch.sigmoid(self.kappa * (z - self.tau.to(z.device)))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def calibrate(self, scores: torch.Tensor, labels: torch.Tensor,
                  steps: int = 200, lr: float = 0.05) -> torch.Tensor:
        """Fit ``tau_q`` with predicate-level binary cross entropy (Table I).

        ``scores`` ``[Nc, P]`` cosine scores on held-out calibration data and
        ``labels`` ``[Nc, P]`` the corresponding binary predicate annotations.
        """
        tau = self.tau.detach().clone().requires_grad_(True)
        optimiser = torch.optim.Adam([tau], lr=lr)
        target = labels.float().to(scores.device)
        with torch.enable_grad():
            for _ in range(steps):
                optimiser.zero_grad()
                probs = torch.sigmoid(self.kappa * (scores - tau))
                loss = nn.functional.binary_cross_entropy(
                    probs.clamp(1e-6, 1 - 1e-6), target
                )
                loss.backward()
                optimiser.step()
        self.tau.data.copy_(tau.detach())
        return self.tau.data

    def extra_repr(self) -> str:
        return f"P={self.num_predicates}, kappa={self.kappa}"


def compatibility_scores(satisfaction: torch.Tensor,
                         weights: torch.Tensor) -> torch.Tensor:
    """``gamma_j = sum_q omega_jq psi(z_jq)``.

    Parameters
    ----------
    satisfaction: ``[..., P]`` predicate satisfaction probabilities.
    weights:      ``[M, P]`` normalised scenario/predicate weights.
    """
    return satisfaction @ weights.to(satisfaction.device).t()


def gap_mask(satisfaction: torch.Tensor, membership: torch.Tensor,
             threshold: float = 0.5) -> torch.Tensor:
    """Boolean ``[..., M, P]`` mask of required-but-unresolved predicates.

    ``membership`` is the ``[M, P]`` binary scenario/predicate incidence matrix.
    """
    unresolved = (satisfaction < threshold).unsqueeze(-2)     # [..., 1, P]
    return unresolved & membership.bool().to(satisfaction.device)
