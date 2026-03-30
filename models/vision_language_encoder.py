"""
Vision-Language Encoder (Section V.A)

Wraps a pretrained CLIP model to produce:
    - Global visual embedding:  f_hat ∈ R^d
    - Local region features:    f = {f^n}_{n=1}^N ∈ R^{N x d}
    - Text base feature:        w = E_t(p_0) ∈ R^{M x d}

The local features are extracted by forwarding through the last
visual encoder block with self-attention removed and a linear
projection θ_Proj appended (as described in Section V.B.4).

To keep the implementation fully runnable on CPU without requiring
a real CLIP checkpoint, we provide a lightweight MockCLIP fallback
that is used automatically when open_clip is not available or
when `use_mock=True` is passed.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight mock encoder (CPU-friendly, no pretrained weights needed)
# ---------------------------------------------------------------------------

class _MockVisualEncoder(nn.Module):
    """
    Deterministic mock of a ViT visual encoder.
    Produces global + local features of the correct shape.
    num_patches controls N — uses adaptive pooling so any image size works.
    """

    def __init__(self, feature_dim: int = 512, num_patches: int = 196):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_patches = num_patches
        # Conv stem: B x 3 x H x W → B x 64 x H' x W'
        self.stem_conv = nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=0)
        # Adaptive pool to exactly num_patches spatial positions
        self.adapt_pool = nn.AdaptiveAvgPool2d((num_patches, 1))
        self.proj = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: image tensor  B x 3 x H x W
        Returns:
            f_hat: global features  B x d
            f:     local features   B x N x d
        """
        B = x.size(0)
        h = self.stem_conv(x)              # B x 64 x H' x W'
        # Flatten spatial dims → B x 64 x (H'*W')
        h = h.flatten(2)                   # B x 64 x S
        # Adaptive pool to exactly num_patches
        h = h.unsqueeze(-1)                # B x 64 x S x 1
        h = self.adapt_pool(h)             # B x 64 x num_patches x 1
        h = h.squeeze(-1).permute(0, 2, 1)  # B x N x 64
        f = self.proj(h)                   # B x N x d
        f_hat = f.mean(dim=1)              # B x d
        f_hat = F.normalize(f_hat, dim=-1)
        f = F.normalize(f, dim=-1)
        return f_hat, f


class _MockTextEncoder(nn.Module):
    """
    Deterministic mock of a transformer text encoder.
    Maps tokenized prompts to feature vectors.
    """

    def __init__(self, vocab_size: int = 49408, feature_dim: int = 512, max_len: int = 77):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, feature_dim)
        self.pos = nn.Embedding(max_len, feature_dim)
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: B x L  (token ids)
        Returns:
            features: B x d  (mean-pooled text features)
        """
        B, L = tokens.shape
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0)
        h = self.embed(tokens) + self.pos(pos_ids)
        h = self.proj(h.mean(dim=1))
        return F.normalize(h, dim=-1)


# ---------------------------------------------------------------------------
# Real CLIP wrapper (optional, requires open_clip_torch)
# ---------------------------------------------------------------------------

class _CLIPVisualWrapper(nn.Module):
    """Wraps an open_clip visual encoder to return global + local features."""

    def __init__(self, visual_encoder: nn.Module, feature_dim: int):
        super().__init__()
        self.encoder = visual_encoder
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # open_clip encode_image returns global features
        f_hat = self.encoder(x)
        f_hat = F.normalize(f_hat, dim=-1)
        # For local features we approximate via spatial average (simplified)
        # A full implementation would hook intermediate transformer blocks
        B, d = f_hat.shape
        # Replicate global as placeholder local features (14x14 grid)
        f = f_hat.unsqueeze(1).expand(B, 196, d)
        return f_hat, f


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class VisionLanguageEncoder(nn.Module):
    """
    Vision-Language Encoder (Section V.A).

    Produces:
        f_hat  — global visual embedding   shape: (B, d)
        f      — local region features     shape: (B, N, d)
        w      — base text features        shape: (M, d)
                 where M = num_semantic_classes

    Args:
        feature_dim:          Embedding dimension d (default 512)
        num_local_features:   Number of local patches N (default 196)
        num_semantic_classes: M — number of scene categories
        use_mock:             If True, use lightweight mock encoders
        clip_model_name:      CLIP checkpoint name (ignored when use_mock=True)
        base_prompt:          The initial textual prompt p_0
    """

    BASE_PROMPT = (
        "Describe the traffic scene including vehicles, signals, road markings"
    )

    # Default semantic class labels for traffic scenes
    DEFAULT_CLASSES: List[str] = [
        "vehicle", "pedestrian", "traffic_light", "stop_sign",
        "lane_marking", "road_sign", "cyclist", "intersection",
        "crosswalk", "parking_zone", "construction_zone",
    ]

    def __init__(
        self,
        feature_dim: int = 512,
        num_local_features: int = 196,
        num_semantic_classes: int = 11,
        use_mock: bool = True,
        clip_model_name: str = "ViT-B-16",
        base_prompt: str = BASE_PROMPT,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_local_features = num_local_features
        self.num_semantic_classes = num_semantic_classes
        self.base_prompt = base_prompt

        if use_mock:
            self._init_mock(feature_dim, num_local_features)
        else:
            self._init_clip(clip_model_name, feature_dim)

        # Learnable base text feature w ∈ R^{M x d}
        # In the full model this is E_t(p_0); here we parameterise it directly
        self.base_text_feature = nn.Parameter(
            torch.randn(num_semantic_classes, feature_dim) * 0.02
        )

    # ------------------------------------------------------------------

    def _init_mock(self, feature_dim: int, num_local_features: int):
        self.visual_encoder = _MockVisualEncoder(feature_dim, num_local_features)
        self.text_encoder = _MockTextEncoder(feature_dim=feature_dim)
        self._use_mock = True

    def _init_clip(self, model_name: str, feature_dim: int):
        try:
            import open_clip  # type: ignore
            model, _, _ = open_clip.create_model_and_transforms(model_name)
            self.visual_encoder = _CLIPVisualWrapper(model.visual, feature_dim)
            self.text_encoder = model.encode_text
            self._use_mock = False
        except ImportError:
            import warnings
            warnings.warn(
                "open_clip_torch not found — falling back to mock encoders.",
                RuntimeWarning,
            )
            self._init_mock(feature_dim, 196)

    # ------------------------------------------------------------------

    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode an image into global and local features.

        Args:
            image: B x 3 x H x W
        Returns:
            f_hat: B x d    global visual embedding
            f:     B x N x d  local region features
        """
        return self.visual_encoder(image)

    def get_base_text_feature(self) -> torch.Tensor:
        """
        Return the base text feature w ∈ R^{M x d}.
        Normalized along the feature dimension.
        """
        return F.normalize(self.base_text_feature, dim=-1)

    def get_zero_shot_features(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return zero-shot text (w_zs) and visual (f_hat_zs) features
        from the *unmodified* (frozen) encoders.  Used in the GI loss.

        In a real system these are cached at initialisation time.
        Here we recompute them with detached gradients.
        """
        with torch.no_grad():
            f_hat_zs, _ = self.encode_image(image)
            w_zs = self.get_base_text_feature().detach()
        return w_zs, f_hat_zs

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            f_hat: B x d
            f:     B x N x d
            w:     M x d   (base text feature, same for all images in batch)
        """
        f_hat, f = self.encode_image(image)
        w = self.get_base_text_feature()
        return f_hat, f, w
