"""Shared fixtures for the test-suite: a tiny, fast APL configuration."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from apl.config import APLConfig  # noqa: E402
from apl.logic import ScenarioBank  # noqa: E402
from apl.model import APLModel  # noqa: E402

IMAGE_SIZE = 32


def tiny_config(**overrides) -> APLConfig:
    cfg = APLConfig()
    cfg.update(
        {
            "encoder.provider": "fallback",
            "encoder.vision_layers": 1,
            "encoder.vision_width": 64,
            "encoder.vision_heads": 2,
            "encoder.text_layers": 1,
            "encoder.text_width": 64,
            "encoder.text_heads": 2,
            "encoder.embed_dim": 64,
            "encoder.image_resolution": IMAGE_SIZE,
            "encoder.patch_size": 16,
            "encoder.num_local_features": 4,
            "feature_generator.dim": 64,
            "feature_generator.key_dim": 64,
            "feature_generator.mid_dim": 32,
            "optim.epochs": 2,
            "optim.batch_size": 4,
            "optim.warmup_epochs": 1,
            "data.image_size": IMAGE_SIZE,
            "output_dir": "runs/tests",
        }
    )
    if overrides:
        cfg.update(overrides)
    return cfg


def tiny_model(cfg=None, **overrides) -> APLModel:
    cfg = cfg or tiny_config(**overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return APLModel(cfg)


def bank() -> ScenarioBank:
    return ScenarioBank.load()


def images(batch: int = 2, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
