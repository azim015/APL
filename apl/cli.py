"""Shared command-line helpers for the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import APLConfig, parse_overrides
from .data import FewShotImageFolder, SyntheticTrafficDataset, TrafficSceneDataset
from .data.synthetic import collate
from .logic import ScenarioBank


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[],
        help="dotted overrides, e.g. optim.lr=0.01 branches.eta=0.5",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="output directory")
    return parser


def load_config(args) -> APLConfig:
    cfg = APLConfig.load(args.config)
    cfg.update(parse_overrides(args.overrides))
    if args.device:
        cfg.encoder.device = args.device
    if args.seed is not None:
        cfg.optim.seed = args.seed
    if args.output:
        cfg.output_dir = args.output
    if cfg.encoder.device == "cuda" and not torch.cuda.is_available():
        cfg.encoder.device = "cpu"
    return cfg


def load_bank(cfg: APLConfig) -> ScenarioBank:
    return ScenarioBank.load(cfg.logic.scenario_bank)


def build_datasets(cfg: APLConfig, bank: Optional[ScenarioBank] = None,
                   num_train: int = 96, num_test: int = 48):
    """Return ``(train_set, test_set, categories, collate_fn)``."""
    dataset = cfg.data.dataset.lower()

    if dataset == "synthetic":
        bank = bank or load_bank(cfg)
        categories = bank.semantic_categories()
        train = SyntheticTrafficDataset(
            categories, num_train, cfg.data.image_size, seed=cfg.optim.seed,
            bank=bank,
        )
        test = SyntheticTrafficDataset(
            categories, num_test, cfg.data.image_size,
            seed=cfg.optim.seed + 977, bank=bank,
        )
        return train, test, categories, collate

    if dataset in {"imagefolder", "fewshot"}:
        root = Path(cfg.data.root)
        train = FewShotImageFolder(
            root / "train", cfg.data.num_shots, cfg.data.image_size,
            cfg.data.subsample, cfg.optim.seed,
        )
        test_root = root / "test" if (root / "test").exists() else root / "train"
        test = FewShotImageFolder(
            test_root, 0, cfg.data.image_size, cfg.data.subsample, cfg.optim.seed,
        )
        return train, test, train.category_names(), None

    if dataset == "traffic":
        bank = bank or load_bank(cfg)
        root = Path(cfg.data.root)
        train = TrafficSceneDataset(root / "train.json", cfg.data.image_size, bank)
        test_manifest = root / "test.json"
        test = TrafficSceneDataset(
            test_manifest if test_manifest.exists() else root / "train.json",
            cfg.data.image_size, bank,
        )
        return train, test, train.category_names(), None

    raise ValueError(f"unknown dataset: {cfg.data.dataset}")


def save_json(path: str | Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def banner(title: str) -> str:
    line = "=" * max(8, len(title))
    return f"\n{line}\n{title}\n{line}"
