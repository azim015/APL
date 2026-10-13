"""Privacy-preservation metrics (Sec. VI-C, Sec. VI-F).

Implemented here:

* ``Privacy Score = 100 (1 - N_leaked / N_sensitive)``  (Eq. 6)
* reconstruction / replay-attack measures: MSE, PSNR, SSIM, SRRA
  (Tables IV and V; lower SSIM/PSNR/SRRA and higher MSE are better for the
  defender)
* membership-inference attacks in the black-box and white-box settings with
  attack accuracy, precision and the average privacy risk
  ``Avg. Risk = 1/N_M sum_{i in M} P_atk(m_i = 1 | x_i)``  (Eq. 5)

Images are compared in ``[0, 255]`` intensities, averaged over all pixels and
colour channels, then averaged over the evaluated images - the protocol stated
in the paper.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from ..describe import PII_PATTERNS, detect_pii

# --------------------------------------------------------------------------- #
# attribute-level leakage
# --------------------------------------------------------------------------- #
#: surface forms that betray each sensitive attribute in a textual description
ATTRIBUTE_CUES: Dict[str, Sequence[str]] = {
    "driver_face": ("driver's face", "face of the driver", "facial", "beard",
                    "glasses", "hairstyle"),
    "passenger_face": ("passenger's face", "passenger looks", "facial"),
    "pedestrian_face": ("pedestrian's face", "facial", "smiling", "skin tone"),
    "license_plate": ("license plate", "licence plate", "plate number",
                      "plate reads", "tag number"),
    "vehicle_vin": ("vin", "vehicle identification number"),
    "street_address": ("street address", "house number", "住所"),
    "phone_number": ("phone", "telephone", "cell number"),
    "person_name": ("named", "name is", "mr.", "mrs.", "ms."),
}


def attribute_leaked(attribute: str, text: str) -> bool:
    """True when the description still identifies the sensitive attribute."""
    lowered = text.lower()
    for cue in ATTRIBUTE_CUES.get(attribute, (attribute.replace("_", " "),)):
        if cue and cue.lower() in lowered:
            return True
    # a residual raw PII string of the matching family also counts as a leak
    family = attribute.split("_")[-1]
    for name, found in detect_pii(text).items():
        if family in name and found:
            return True
    return False


def privacy_score(
    descriptions: Sequence[str],
    sensitive_attributes: Sequence[Sequence[str]],
) -> float:
    """Eq. (6): percentage of annotated sensitive attributes suppressed."""
    total, leaked = 0, 0
    for text, attributes in zip(descriptions, sensitive_attributes):
        for attribute in attributes:
            total += 1
            if attribute_leaked(attribute, text):
                leaked += 1
    if total == 0:
        return 100.0
    return 100.0 * (1.0 - leaked / total)


def leakage_report(
    descriptions: Sequence[str],
    sensitive_attributes: Sequence[Sequence[str]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text, attributes in zip(descriptions, sensitive_attributes):
        for attribute in attributes:
            if attribute_leaked(attribute, text):
                counts[attribute] = counts.get(attribute, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# reconstruction-attack metrics
# --------------------------------------------------------------------------- #
def _to_255(images: torch.Tensor) -> torch.Tensor:
    images = images.float()
    if images.max() <= 1.5:
        images = images * 255.0
    return images.clamp(0, 255)


def mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    a, b = _to_255(original), _to_255(reconstructed)
    per_image = ((a - b) ** 2).flatten(1).mean(dim=1)
    return float(per_image.mean())


def psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    a, b = _to_255(original), _to_255(reconstructed)
    per_image = ((a - b) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
    values = 10.0 * torch.log10((255.0 ** 2) / per_image)
    return float(values.mean())


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


def ssim(original: torch.Tensor, reconstructed: torch.Tensor,
         window_size: int = 11) -> float:
    """Mean structural similarity over the batch (channel-averaged)."""
    import torch.nn.functional as F

    a, b = _to_255(original), _to_255(reconstructed)
    if a.dim() == 3:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    channels = a.shape[1]
    window = _gaussian_window(window_size).to(a.device)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(a, window, padding=pad, groups=channels)
    mu2 = F.conv2d(b, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1 = F.conv2d(a * a, window, padding=pad, groups=channels) - mu1_sq
    sigma2 = F.conv2d(b * b, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(a * b, window, padding=pad, groups=channels) - mu12

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    smap = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    return float(smap.flatten(1).mean(dim=1).mean())


def replay_attack_success_rate(
    probe: torch.Tensor,
    gallery: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """SRRA (%): fraction of reconstructed probes that match their identity.

    ``probe`` and ``gallery`` are embeddings ``[N, d]`` produced by the face /
    re-identification model used by the attacker; a probe counts as a success
    when its cosine similarity with the matching gallery entry exceeds
    ``threshold`` **and** is the nearest neighbour in the gallery.
    """
    import torch.nn.functional as F

    p = F.normalize(probe.float(), dim=-1)
    g = F.normalize(gallery.float(), dim=-1)
    sim = p @ g.t()
    nearest = sim.argmax(dim=-1)
    matched = nearest == torch.arange(p.shape[0], device=p.device)
    above = sim.diagonal() >= threshold
    return 100.0 * float((matched & above).float().mean())


@dataclass
class ReconstructionMetrics:
    ssim: float
    psnr: float
    mse: float
    srra: Optional[float] = None

    def as_row(self) -> Dict[str, float]:
        row = {
            "SSIM": round(self.ssim, 4),
            "PSNR": round(self.psnr, 2),
            "MSE": round(self.mse, 1),
        }
        if self.srra is not None:
            row["SRRA (%)"] = round(self.srra, 2)
        return row


def reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    probe: Optional[torch.Tensor] = None,
    gallery: Optional[torch.Tensor] = None,
    srra_threshold: float = 0.5,
) -> ReconstructionMetrics:
    return ReconstructionMetrics(
        ssim=ssim(original, reconstructed),
        psnr=psnr(original, reconstructed),
        mse=mse(original, reconstructed),
        srra=(
            replay_attack_success_rate(probe, gallery, srra_threshold)
            if probe is not None and gallery is not None else None
        ),
    )


# --------------------------------------------------------------------------- #
# membership inference
# --------------------------------------------------------------------------- #
@dataclass
class MIAResult:
    setting: str
    accuracy: float
    precision: float
    avg_risk: float

    def as_row(self) -> Dict[str, float]:
        return {
            "Setting": self.setting,
            "Atk. Acc. (%)": round(self.accuracy, 1),
            "Precision (%)": round(self.precision, 1),
            "Avg. Risk": round(self.avg_risk, 3),
        }


def membership_inference_attack(
    member_signal: torch.Tensor,
    nonmember_signal: torch.Tensor,
    setting: str = "black-box",
    threshold: Optional[float] = None,
) -> MIAResult:
    """Threshold attack over a per-sample signal.

    ``black-box``  the signal is the model's confidence on the true class;
    ``white-box``  the signal is the negative per-sample loss (higher = member).

    ``P_atk(m_i = 1 | x_i)`` is the calibrated sigmoid of the signal around the
    attack threshold; the average over true members is the privacy risk of
    Eq. (5).
    """
    member = member_signal.detach().float().flatten()
    nonmember = nonmember_signal.detach().float().flatten()
    if threshold is None:
        threshold = float(torch.cat([member, nonmember]).median())

    scale = float(torch.cat([member, nonmember]).std()) + 1e-8
    p_member = torch.sigmoid((member - threshold) / scale)
    p_nonmember = torch.sigmoid((nonmember - threshold) / scale)

    tp = float((member > threshold).float().sum())
    fn = float((member <= threshold).float().sum())
    fp = float((nonmember > threshold).float().sum())
    tn = float((nonmember <= threshold).float().sum())

    accuracy = 100.0 * (tp + tn) / max(1.0, tp + tn + fp + fn)
    precision = 100.0 * tp / max(1.0, tp + fp)
    return MIAResult(
        setting=setting,
        accuracy=accuracy,
        precision=precision,
        avg_risk=float(p_member.mean()),
    )


@torch.no_grad()
def model_confidence_signal(model, dataset, collate_fn=None,
                            batch_size: int = 32) -> torch.Tensor:
    """Black-box signal: the probability the model assigns to the true class."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    out = []
    model.eval()
    for batch in loader:
        probs = model(batch["image"].to(model.device), compute_loss=False).probs
        labels = batch["label"].to(probs.device)
        out.append(probs.gather(1, labels.view(-1, 1)).squeeze(1).cpu())
    return torch.cat(out) if out else torch.zeros(0)


@torch.no_grad()
def model_loss_signal(model, dataset, collate_fn=None,
                      batch_size: int = 32) -> torch.Tensor:
    """White-box signal: the negative per-sample cross-entropy."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    out = []
    model.eval()
    for batch in loader:
        probs = model(batch["image"].to(model.device), compute_loss=False).probs
        labels = batch["label"].to(probs.device)
        nll = -torch.log(
            probs.gather(1, labels.view(-1, 1)).squeeze(1).clamp_min(1e-9)
        )
        out.append((-nll).cpu())
    return torch.cat(out) if out else torch.zeros(0)
