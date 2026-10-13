"""A self-contained CLIP-geometry backbone used when no CLIP weights are present.

The module re-implements the parts of CLIP that APL actually needs:

* a ViT image encoder that exposes **both** the global embedding ``f_hat`` and
  the ``N`` local (patch-level) features ``f = {f^n}``;
* a causal text transformer with the CLIP pooling convention (the embedding is
  read out at the end-of-text token);
* hooks for inserting learnable prompt tokens at the encoder inputs.

Weights are randomly initialised: this backbone exists so that the APL
pipeline, its tests and its demos run end-to-end without network access.  When
the real CLIP checkpoint is available the same interface is served by
:class:`apl.encoders.clip_backbone.CLIPBackbone`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, attn_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4), QuickGELU(), nn.Linear(width * 4, width)
        )
        self.ln_2 = nn.LayerNorm(width)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        mask = None
        if self.attn_mask is not None:
            mask = self.attn_mask[: x.shape[1], : x.shape[1]]
        attn, _ = self.attn(h, h, h, need_weights=False, attn_mask=mask)
        x = x + attn
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,
                 attn_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class VisionTransformer(nn.Module):
    """ViT image encoder returning the global embedding and local features."""

    def __init__(self, resolution: int, patch_size: int, width: int, layers: int,
                 heads: int, output_dim: int):
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        self.num_patches = (resolution // patch_size) ** 2
        self.conv1 = nn.Conv2d(3, width, patch_size, patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_patches + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(
        self, images: torch.Tensor, visual_prompts: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(images)                      # [B, width, H/p, W/p]
        b, c = x.shape[0], x.shape[1]
        x = x.reshape(b, c, -1).permute(0, 2, 1)    # [B, N, width]
        cls = self.class_embedding.to(x.dtype).expand(b, 1, c)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)[: x.shape[1]]
        n_tokens = x.shape[1]
        if visual_prompts is not None:
            prompts = visual_prompts.to(x.dtype)
            if prompts.dim() == 2:
                prompts = prompts.unsqueeze(0).expand(b, -1, -1)
            x = torch.cat([x, prompts], dim=1)      # inserted at the encoder input
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)
        x = x @ self.proj
        global_feature = x[:, 0]                    # f_hat
        local_features = x[:, 1:n_tokens]           # f = {f^n}_{n=1..N}
        return global_feature, local_features


class TextTransformer(nn.Module):
    """Causal text transformer with CLIP's end-of-text pooling."""

    def __init__(self, vocab_size: int, context_length: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, width).normal_(std=0.01)
        )
        mask = torch.full((context_length, context_length), float("-inf"))
        mask.triu_(1)
        self.transformer = Transformer(width, layers, heads, attn_mask=mask)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(
            torch.empty(width, output_dim).normal_(std=width ** -0.5)
        )
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    # ------------------------------------------------------------------ #
    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        prompt_embeddings: Optional[torch.Tensor] = None,
        eot_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_embedding(tokens)
        if eot_index is None:
            eot_index = tokens.argmax(dim=-1)
        if prompt_embeddings is not None:
            p = prompt_embeddings.to(x.dtype)
            if p.dim() == 2:
                p = p.unsqueeze(0).expand(x.shape[0], -1, -1)
            n_prompt = p.shape[1]
            # insert the learnable prompt right after the SOT token
            x = torch.cat([x[:, :1], p, x[:, 1:]], dim=1)[:, : self.context_length]
            eot_index = torch.clamp(
                eot_index + n_prompt, max=self.context_length - 1
            )
        x = x + self.positional_embedding[: x.shape[1]].to(x.dtype)
        x = self.transformer(x)
        x = self.ln_final(x)
        pooled = x[torch.arange(x.shape[0], device=x.device), eot_index]
        return pooled @ self.text_projection


class FallbackCLIP(nn.Module):
    """CLIP-shaped vision-language model with randomly initialised weights."""

    def __init__(
        self,
        embed_dim: int = 512,
        image_resolution: int = 224,
        vision_layers: int = 12,
        vision_width: int = 768,
        vision_heads: int = 12,
        patch_size: int = 16,
        context_length: int = 77,
        vocab_size: int = 49408,
        text_layers: int = 12,
        text_width: int = 512,
        text_heads: int = 8,
    ):
        super().__init__()
        self.visual = VisionTransformer(
            image_resolution, patch_size, vision_width, vision_layers,
            vision_heads, embed_dim,
        )
        self.text = TextTransformer(
            vocab_size, context_length, text_width, text_layers, text_heads,
            embed_dim,
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.embed_dim = embed_dim
        self.context_length = context_length

    # ------------------------------------------------------------------ #
    @property
    def token_embedding(self) -> nn.Embedding:
        return self.text.token_embedding

    def encode_image(self, images, visual_prompts=None):
        return self.visual(images, visual_prompts)

    def encode_text(self, tokens, prompt_embeddings=None, eot_index=None):
        return self.text(tokens, prompt_embeddings, eot_index)
