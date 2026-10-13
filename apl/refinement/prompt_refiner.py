"""Gap-conditioned prompt refinement ``R(D_t, G_t, S)`` (Sec. V-F).

When the evidence check leaves required predicates unresolved, the gap set

``G_t = union_j { (j, e) : e in G_{j,t} }``

drives two things:

1. a set of **targeted textual prompts** ``p_{t+1} = R(D_t, G_t, S)`` that ask
   specifically about the missing evidence, and
2. a **conditioning signal** ``alpha^gap(G_t)`` that re-conditions the feature
   generator, ``T_{t+1} = theta_FG(T_t, alpha^gap(G_t))``.

Prompts are ranked by how often a missing predicate blocks a scenario and by
how far its satisfaction sits below the gap threshold, so the refinement
budget is spent on the evidence that blocks the most decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

GAP_TEMPLATES = (
    "Look again at the scene and report whether {text}.",
    "Focus on the region that would show whether {text}.",
    "Describe the visual evidence for or against the statement that {text}.",
)


@dataclass
class GapPrompt:
    predicate_id: str
    scenario_ids: List[str]
    text: str
    deficit: float

    def render(self, variant: int = 0) -> str:
        return GAP_TEMPLATES[variant % len(GAP_TEMPLATES)].format(text=self.text)


class PromptRefiner(nn.Module):
    """Builds refinement prompts and the gap conditioning bank ``alpha^gap``."""

    def __init__(self, bank, prompt_learner, feature_generator, cfg,
                 gap_threshold: float = 0.5):
        super().__init__()
        self.bank = bank
        self.prompt_learner = prompt_learner
        self.fg = feature_generator
        self.cfg = cfg
        self.gap_threshold = float(gap_threshold)
        dim = feature_generator.dim
        # learnable gap tokens; the encoded gap prompts are added on top of them
        self.gap_tokens = nn.Parameter(
            torch.empty(cfg.gap_token_length, dim).normal_(std=0.02)
        )

    # ------------------------------------------------------------------ #
    def rank_gaps(self, description, gaps: torch.Tensor,
                  index: int = 0) -> List[GapPrompt]:
        """Turn the ``[B, M, P]`` gap mask into ranked, de-duplicated prompts."""
        mask = gaps[index]                                  # [M, P]
        satisfaction = description.satisfaction[index].detach()
        blocked: Dict[int, List[str]] = {}
        for j, scenario in enumerate(self.bank.scenarios):
            for predicate in scenario.predicates:
                p = self.bank.predicate_index[predicate.id]
                if bool(mask[j, p]):
                    blocked.setdefault(p, []).append(scenario.id)

        prompts = []
        for p, scenario_ids in blocked.items():
            predicate = self.bank.predicates[p]
            deficit = float(self.gap_threshold - satisfaction[p])
            prompts.append(
                GapPrompt(
                    predicate_id=predicate.id,
                    scenario_ids=scenario_ids,
                    text=predicate.text,
                    deficit=deficit,
                )
            )
        prompts.sort(key=lambda g: (-len(g.scenario_ids), -g.deficit))
        return prompts[: self.cfg.max_gap_prompts]

    # ------------------------------------------------------------------ #
    def generate_prompts(self, description, gaps: torch.Tensor,
                         index: int = 0) -> List[str]:
        """``p_{t+1} = R(D_t, G_t, S)``."""
        return [g.render(i) for i, g in enumerate(self.rank_gaps(description, gaps, index))]

    # ------------------------------------------------------------------ #
    def gap_conditioning(self, prompts: Sequence[str]) -> torch.Tensor:
        """``alpha^gap(G_t) in R^{L x d}``.

        The targeted prompts are encoded by the (frozen) text encoder with the
        learned prompt tokens, pooled into ``L`` rows and added to the
        learnable gap tokens.
        """
        alpha = self.gap_tokens
        if not prompts:
            return alpha
        encoded = self.prompt_learner.encode_texts(list(prompts))   # [G, d]
        length = alpha.shape[0]
        if encoded.shape[0] >= length:
            pooled = encoded[:length]
        else:
            pad = encoded.mean(dim=0, keepdim=True).expand(
                length - encoded.shape[0], -1
            )
            pooled = torch.cat([encoded, pad], dim=0)
        return alpha + pooled.to(alpha.device)

    # ------------------------------------------------------------------ #
    def refine_features(self, features, alpha_gap: torch.Tensor):
        """``T_{t+1} = theta_FG(T_t, alpha^gap(G_t))`` applied row-block wise."""
        from ..logic.evidence import TextFeatureSet

        w_gi = None if features.w_gi is None else self.fg(features.w_gi, alpha_gap)
        h_id = None
        if features.h_id is not None:
            batch = features.h_id.shape[0]
            alpha_b = alpha_gap.unsqueeze(0).expand(batch, -1, -1)
            h_id = self.fg(features.h_id, alpha_b)
        h_fg = None
        if features.h_fg is not None:
            h_fg = torch.stack(
                [self.fg(features.h_fg[z], alpha_gap)
                 for z in range(features.h_fg.shape[0])],
                dim=0,
            )
        return TextFeatureSet(
            w_gi=w_gi, h_id=h_id, h_fg=h_fg, iteration=features.iteration + 1
        )
