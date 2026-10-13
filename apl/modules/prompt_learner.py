"""Base prompt, learnable prompt tokens and the base text feature ``w``.

Sec. V-B1: the initial prompt is

    p0 = "Describe the traffic scene including vehicles, signals, road markings"

and the frozen text encoder produces ``w = E_t(p0) in R^{C x d}``, one row per
semantic category.  Table I adds four learnable text tokens and four learnable
visual tokens, inserted at the encoder inputs and initialised from
``N(0, 0.02^2)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class PromptLearner(nn.Module):
    def __init__(self, backbone, categories: Sequence[str], cfg):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.categories: List[str] = list(categories)
        self.num_categories = len(self.categories)

        token_width = backbone.token_width
        visual_width = backbone.visual_width

        self.text_ctx = nn.Parameter(
            torch.empty(cfg.n_text_tokens, token_width).normal_(std=cfg.init_std)
        ) if cfg.n_text_tokens > 0 else None
        self.visual_ctx = nn.Parameter(
            torch.empty(cfg.n_visual_tokens, visual_width).normal_(std=cfg.init_std)
        ) if cfg.n_visual_tokens > 0 else None

        prompts = [self.render(category) for category in self.categories]
        self.register_buffer("tokens", backbone.tokenize(prompts), persistent=False)
        self._zero_shot: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def render(self, category: str) -> str:
        return self.cfg.class_template.format(
            base=self.cfg.base_prompt, category=category
        )

    @property
    def device(self) -> torch.device:
        if self.text_ctx is not None:
            return self.text_ctx.device
        return self.tokens.device

    # ------------------------------------------------------------------ #
    def forward(self) -> torch.Tensor:
        """``w in R^{C x d}`` - the base prompted text feature."""
        return self.encode_tokens(self.tokens)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(self.device)
        return self.backbone.encode_text(tokens, self.text_ctx)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode arbitrary text (predicate phrasings, refinement prompts)."""
        tokens = self.backbone.tokenize(list(texts)).to(self.device)
        return self.encode_tokens(tokens)

    def set_categories(self, categories: Sequence[str]) -> None:
        """Swap the semantic categories while keeping the learned prompts.

        This is the base-to-new protocol: the prompt tokens and the feature
        generator transfer, only the class names change.
        """
        self.categories = list(categories)
        self.num_categories = len(self.categories)
        prompts = [self.render(category) for category in self.categories]
        tokens = self.backbone.tokenize(prompts).to(self.tokens.device)
        self.register_buffer("tokens", tokens, persistent=False)
        self.reset_zero_shot_cache()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def zero_shot_features(self) -> torch.Tensor:
        """``w^zs`` from the unmodified encoder; cached after the first call."""
        if self._zero_shot is None or self._zero_shot.device != self.device:
            self._zero_shot = self.backbone.encode_text(
                self.tokens.to(self.device), None
            ).detach()
        return self._zero_shot

    def reset_zero_shot_cache(self) -> None:
        self._zero_shot = None

    # ------------------------------------------------------------------ #
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"categories={self.num_categories}, "
            f"text_tokens={0 if self.text_ctx is None else self.text_ctx.shape[0]}, "
            f"visual_tokens="
            f"{0 if self.visual_ctx is None else self.visual_ctx.shape[0]}"
        )
