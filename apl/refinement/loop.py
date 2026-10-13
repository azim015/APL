"""Algorithm 1 - iterative violation evidence extraction (Sec. V-I).

::

    v <- E_v(I); initialize prompts T_0
    D_0 <- ExtractDescription(v, T_0)
    for t = 0 .. T_max:
        G_t <- {}
        for j = 1 .. M:
            gamma_j^(t) <- ScenarioMatch(D_t, s_j)
            G_{j,t}     <- MissingEvidence(D_t, s_j)
            G_t         <- G_t  u  {(j, e) : e in G_{j,t}}
        if G_t = {} or t = T_max: break
        if t >= 1 and max_j |gamma_j^(t) - gamma_j^(t-1)| < eps: break
        prompts <- GeneratePrompts(G_t)
        T_{t+1} <- PromptUpdate(T_t, prompts)
        D_{t+1} <- UpdateDescription(v, T_{t+1})
    t_star <- t
    y_j <- RuleEngine(D_t, s_j) for every j

Every iteration evaluates **all** ``M`` scenarios: gap detection does not
require a positive violation decision in the initial description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class RefinementTrace:
    """Per-iteration record of the feedback loop."""

    iteration: int
    gamma: torch.Tensor
    num_gaps: int
    delta_gamma: Optional[float] = None
    prompts: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "iteration": self.iteration,
            "gamma": [round(float(v), 4) for v in self.gamma.flatten().tolist()],
            "num_gaps": self.num_gaps,
            "delta_gamma": self.delta_gamma,
            "prompts": self.prompts,
        }


@dataclass
class RefinementResult:
    description: "object"
    gamma: torch.Tensor
    features: "object"
    iterations: int
    stop_reason: str
    trace: List[RefinementTrace] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)


def run_refinement_loop(
    extractor,
    refiner,
    features,
    f_hat: torch.Tensor,
    local: Optional[torch.Tensor],
    cfg,
    index: int = 0,
) -> RefinementResult:
    """Run Algorithm 1 for the image at position ``index`` of the batch."""
    description = extractor.extract(features, f_hat, local)
    gamma = extractor.compatibility(description)
    trace: List[RefinementTrace] = []
    prev_gamma: Optional[torch.Tensor] = None
    all_prompts: List[str] = []
    t = 0
    stop_reason = "max_iterations"
    t_max = cfg.t_max if cfg.enabled else 0

    for t in range(t_max + 1):
        gaps = extractor.gaps(description)
        num_gaps = int(gaps[index].sum())
        delta = None
        if prev_gamma is not None:
            delta = float((gamma[index] - prev_gamma).abs().max())

        trace.append(
            RefinementTrace(iteration=t, gamma=gamma[index].detach(),
                            num_gaps=num_gaps, delta_gamma=delta)
        )

        if num_gaps == 0:
            stop_reason = "no_gaps"
            break
        if t == t_max:
            stop_reason = "max_iterations"
            break
        if delta is not None and delta < cfg.epsilon:
            stop_reason = "converged"
            break

        prompts = refiner.generate_prompts(description, gaps, index)
        trace[-1].prompts = prompts
        all_prompts.extend(prompts)

        alpha_gap = refiner.gap_conditioning(prompts)
        features = refiner.refine_features(features, alpha_gap)

        prev_gamma = gamma[index].detach().clone()
        description = extractor.extract(features, f_hat, local)
        gamma = extractor.compatibility(description)

    return RefinementResult(
        description=description,
        gamma=gamma,
        features=features,
        iterations=t,
        stop_reason=stop_reason,
        trace=trace,
        prompts=all_prompts,
    )
