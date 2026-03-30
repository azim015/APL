"""
Privacy Preservation Metrics — Section VI.B

Implements the four metrics used to evaluate privacy-preservation quality:

    1. SSIM  — Structural Similarity Index (lower = stronger privacy)
    2. PSNR  — Peak Signal-to-Noise Ratio (lower = stronger privacy)
    3. MSE   — Mean Squared Error (higher = stronger privacy)
    4. SRRA  — Success Rate of Replay Attacks (lower = stronger privacy)

Also includes Membership Inference Attack (MIA) evaluation metrics:
    - Attack Accuracy
    - Precision
    - Average Privacy Risk Score
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level metric functions
# ---------------------------------------------------------------------------

def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Mean Squared Error between original and reconstructed images.
    Higher MSE → stronger privacy (harder reconstruction).

    Args:
        original:      B x C x H x W  or  C x H x W
        reconstructed: same shape as original
    Returns:
        scalar MSE value
    """
    return F.mse_loss(reconstructed.float(), original.float()).item()


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor,
                 max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Lower PSNR → stronger privacy.

    Args:
        original:      image tensor (values in [0, max_val])
        reconstructed: image tensor
        max_val:       maximum pixel value (1.0 for float, 255.0 for uint8)
    Returns:
        PSNR in dB
    """
    mse = compute_mse(original, reconstructed)
    if mse < 1e-10:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)


def compute_ssim(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    max_val: float = 1.0,
) -> float:
    """
    Structural Similarity Index (SSIM).
    Lower SSIM → stronger privacy (less structural preservation).

    Implements the standard SSIM formula using Gaussian-weighted patches.
    Works on CPU; handles both single images (C x H x W) and batches (B x C x H x W).

    Returns:
        Mean SSIM across batch (scalar)
    """
    # Ensure batch dimension
    if original.dim() == 3:
        original = original.unsqueeze(0)
        reconstructed = reconstructed.unsqueeze(0)

    B, C, H, W = original.shape

    # Build Gaussian kernel
    kernel = _gaussian_kernel(window_size, sigma, C)
    kernel = kernel.to(original.device)

    def _mu(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=window_size // 2, groups=C)

    mu1 = _mu(original)
    mu2 = _mu(reconstructed)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = _mu(original ** 2) - mu1_sq
    sigma2_sq = _mu(reconstructed ** 2) - mu2_sq
    sigma12 = _mu(original * reconstructed) - mu1_mu2

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    ssim_map = (
        (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    ) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create a Gaussian kernel for SSIM computation."""
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss /= gauss.sum()
    kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)   # window_size x window_size
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)           # 1 x 1 x W x W
    return kernel.repeat(channels, 1, 1, 1)                # C x 1 x W x W


# ---------------------------------------------------------------------------
# SRRA — Success Rate of Replay Attacks
# ---------------------------------------------------------------------------

def compute_srra(
    original_features: torch.Tensor,    # B x d  — features of original images
    reconstructed_features: torch.Tensor,  # B x d — features of reconstructed images
    threshold: float = 0.9,
) -> float:
    """
    Success Rate of Replay Attacks (SRRA).

    A replay attack is considered successful if the cosine similarity
    between the original and reconstructed feature vectors exceeds a threshold,
    indicating that the reconstruction retains enough identity to be replayed.

    Lower SRRA → stronger privacy.

    Args:
        original_features:      B x d — feature embeddings of original images
        reconstructed_features: B x d — feature embeddings of reconstructed images
        threshold:              cosine similarity threshold for "success"
    Returns:
        SRRA as a percentage [0, 100]
    """
    orig_norm = F.normalize(original_features, dim=-1)
    recon_norm = F.normalize(reconstructed_features, dim=-1)
    cos_sim = (orig_norm * recon_norm).sum(dim=-1)   # B
    successes = (cos_sim >= threshold).float().sum().item()
    return 100.0 * successes / original_features.size(0)


# ---------------------------------------------------------------------------
# MIA — Membership Inference Attack metrics
# ---------------------------------------------------------------------------

def compute_mia_metrics(
    member_scores: torch.Tensor,     # N — confidence scores for training members
    non_member_scores: torch.Tensor, # N — confidence scores for non-members
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Membership Inference Attack (MIA) evaluation metrics.

    Args:
        member_scores:     Model confidence scores for training set members
        non_member_scores: Model confidence scores for non-members
        threshold:         Decision boundary for member / non-member

    Returns:
        Dict with:
            "attack_accuracy":  fraction of correct membership predictions
            "precision":        precision of predicting "member"
            "avg_risk":         average privacy risk (mean of member_scores)
    """
    all_scores = torch.cat([member_scores, non_member_scores])
    true_labels = torch.cat([
        torch.ones(len(member_scores)),
        torch.zeros(len(non_member_scores)),
    ])
    predictions = (all_scores >= threshold).float()

    correct = (predictions == true_labels).float().sum().item()
    attack_accuracy = correct / len(all_scores)

    true_positives = ((predictions == 1) & (true_labels == 1)).float().sum().item()
    predicted_positives = (predictions == 1).float().sum().item()
    precision = true_positives / max(predicted_positives, 1.0)

    avg_risk = member_scores.mean().item()

    return {
        "attack_accuracy": attack_accuracy * 100,
        "precision": precision * 100,
        "avg_risk": avg_risk,
    }


# ---------------------------------------------------------------------------
# Convenience: compute all privacy metrics at once
# ---------------------------------------------------------------------------

def compute_all_privacy_metrics(
    original: torch.Tensor,              # B x C x H x W
    reconstructed: torch.Tensor,         # B x C x H x W
    original_features: torch.Tensor,     # B x d  (optional for SRRA)
    reconstructed_features: torch.Tensor, # B x d (optional for SRRA)
    srra_threshold: float = 0.9,
) -> Dict[str, float]:
    """
    Compute SSIM, PSNR, MSE, and SRRA for a batch of image pairs.

    Returns:
        Dict with metric names → scalar values
    """
    return {
        "SSIM": compute_ssim(original, reconstructed),
        "PSNR": compute_psnr(original, reconstructed),
        "MSE": compute_mse(original, reconstructed),
        "SRRA": compute_srra(original_features, reconstructed_features, srra_threshold),
    }
