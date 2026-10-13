"""Frozen vision-language backbone (Sec. V-A of the paper).

``CLIPBackbone`` exposes one interface regardless of where the weights come
from:

``encode_image(images, visual_prompts)``
    returns ``(f_hat, f)`` with ``f_hat`` of shape ``[B, d]`` (global scene
    embedding) and ``f`` of shape ``[B, N, d]`` (local region-level features).

``encode_text(tokens, prompt_embeddings)``
    returns ``[*, d]`` text embeddings, with the learnable prompt tokens
    inserted at the encoder input.

Three providers are supported and selected automatically:

``clip``        the OpenAI CLIP package (``pip install git+https://github.com/openai/CLIP.git``)
``open_clip``   the open_clip_torch package
``fallback``    the bundled CLIP-geometry re-implementation (random weights)

The encoders are frozen (``requires_grad_(False)``) exactly as in Table I:
only the prompts, the feature generator and the local projection are learned.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .fallback_clip import FallbackCLIP
from .tokenizer import Tokenizer


class CLIPBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.context_length = cfg.context_length
        self.num_local_features = cfg.num_local_features
        self.tokenizer = Tokenizer(cfg.vocab_size, cfg.context_length)

        provider, model = _build_model(cfg)
        self.provider = provider
        self.model = model

        if cfg.freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

    # ------------------------------------------------------------------ #
    # properties used by the prompt learner
    # ------------------------------------------------------------------ #
    @property
    def token_embedding(self) -> nn.Embedding:
        if hasattr(self.model, "token_embedding"):
            return self.model.token_embedding
        return self.model.text.token_embedding  # pragma: no cover

    @property
    def token_width(self) -> int:
        return self.token_embedding.embedding_dim

    @property
    def visual_width(self) -> int:
        if self.provider == "fallback":
            return self.model.visual.conv1.out_channels
        return self.model.visual.conv1.out_channels  # pragma: no cover

    def tokenize(self, texts) -> torch.Tensor:
        return self.tokenizer(texts)

    # ------------------------------------------------------------------ #
    def encode_image(
        self, images: torch.Tensor, visual_prompts: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.provider == "fallback":
            return self.model.encode_image(images, visual_prompts)
        return _openai_encode_image(self.model, images, visual_prompts)

    def encode_text(
        self,
        tokens: torch.Tensor,
        prompt_embeddings: Optional[torch.Tensor] = None,
        eot_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.provider == "fallback":
            return self.model.encode_text(tokens, prompt_embeddings, eot_index)
        return _openai_encode_text(  # pragma: no cover
            self.model, tokens, prompt_embeddings, eot_index
        )

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def zero_shot_text(self, texts) -> torch.Tensor:
        """``w^zs``: text features from the *unmodified* encoder."""
        tokens = self.tokenize(texts).to(_device_of(self.model))
        return self.encode_text(tokens)

    @torch.no_grad()
    def zero_shot_image(self, images: torch.Tensor) -> torch.Tensor:
        """``f_hat^zs``: visual features from the *unmodified* encoder."""
        return self.encode_image(images)[0]


# --------------------------------------------------------------------------- #
# provider construction
# --------------------------------------------------------------------------- #
def _build_model(cfg):
    provider = (cfg.provider or "auto").lower()
    order = [provider] if provider != "auto" else ["clip", "open_clip", "fallback"]
    errors = []
    for name in order:
        try:
            if name == "clip":
                return "clip", _load_openai_clip(cfg)
            if name == "open_clip":
                return "open_clip", _load_open_clip(cfg)
            if name == "fallback":
                return "fallback", _load_fallback(cfg)
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{name}: {exc}")
    warnings.warn(
        "Falling back to the bundled CLIP-geometry backbone with random "
        "weights (" + "; ".join(errors) + ").",
        RuntimeWarning,
        stacklevel=2,
    )
    return "fallback", _load_fallback(cfg)


def _load_openai_clip(cfg):  # pragma: no cover - requires the CLIP package
    import clip

    model, _ = clip.load(cfg.backbone, device=cfg.device, jit=False)
    return model.float()


def _load_open_clip(cfg):  # pragma: no cover - requires open_clip_torch
    import open_clip

    name = cfg.backbone.replace("/", "-")
    model, _, _ = open_clip.create_model_and_transforms(
        name, pretrained=cfg.pretrained, device=cfg.device
    )
    return model.float()


def _load_fallback(cfg):
    model = FallbackCLIP(
        embed_dim=cfg.embed_dim,
        image_resolution=cfg.image_resolution,
        vision_layers=cfg.vision_layers,
        vision_width=cfg.vision_width,
        vision_heads=cfg.vision_heads,
        patch_size=cfg.patch_size,
        context_length=cfg.context_length,
        vocab_size=cfg.vocab_size,
        text_layers=cfg.text_layers,
        text_width=cfg.text_width,
        text_heads=cfg.text_heads,
    )
    if cfg.checkpoint:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    return model.to(cfg.device)


# --------------------------------------------------------------------------- #
# OpenAI/open_clip adapters (dense features + prompt insertion)
# --------------------------------------------------------------------------- #
def _openai_encode_image(model, images, visual_prompts=None):  # pragma: no cover
    visual = model.visual
    x = visual.conv1(images.type(visual.conv1.weight.dtype))
    b, c = x.shape[0], x.shape[1]
    x = x.reshape(b, c, -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        b, 1, c, dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)[: x.shape[1]]
    n_tokens = x.shape[1]
    if visual_prompts is not None:
        p = visual_prompts.to(x.dtype)
        if p.dim() == 2:
            p = p.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([x, p], dim=1)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)
    if getattr(visual, "proj", None) is not None:
        x = x @ visual.proj
    return x[:, 0], x[:, 1:n_tokens]


def _openai_encode_text(model, tokens, prompt_embeddings=None,
                        eot_index=None):  # pragma: no cover
    x = model.token_embedding(tokens).type(model.dtype)
    if eot_index is None:
        eot_index = tokens.argmax(dim=-1)
    if prompt_embeddings is not None:
        p = prompt_embeddings.to(x.dtype)
        if p.dim() == 2:
            p = p.unsqueeze(0).expand(x.shape[0], -1, -1)
        n_prompt = p.shape[1]
        ctx = model.positional_embedding.shape[0]
        x = torch.cat([x[:, :1], p, x[:, 1:]], dim=1)[:, :ctx]
        eot_index = torch.clamp(eot_index + n_prompt, max=ctx - 1)
    x = x + model.positional_embedding.type(model.dtype)[: x.shape[1]]
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    pooled = x[torch.arange(x.shape[0], device=x.device), eot_index]
    return pooled @ model.text_projection


def _device_of(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    return torch.device("cpu")
