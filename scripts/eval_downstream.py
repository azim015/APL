#!/usr/bin/env python3
"""End-to-end ITS task validation (Table VIII).

Compares four input representations on three downstream tasks:

================  =========================================================
original image    the raw frame embedding - the utility reference
blurred image     the classic obfuscation baseline (Gaussian blur)
caption baseline  a single generic caption per scene (RL-MoE-style text)
APL description   the structured, privacy-preserving description
================  =========================================================

Tasks: traffic-flow prediction (RMSE), road-sign recognition (accuracy) and
pedestrian counting (MAE).  The downstream reasoning module ``g_phi`` is a
ridge regressor / nearest-centroid classifier fitted on each representation,
so the comparison measures how much task signal each representation carries
rather than the capacity of the head.

    python scripts/eval_downstream.py --config configs/smoke.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from apl.cli import banner, base_parser, build_datasets, load_bank, load_config, save_json
from apl.eval import accuracy, downstream_report, format_table, mae, rmse
from apl.model import APLModel
from apl.train import Trainer, set_seed


# --------------------------------------------------------------------------- #
def gaussian_blur(images: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    size = int(2 * round(3 * sigma) + 1)
    coords = torch.arange(size).float() - size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels = images.shape[1]
    k2 = kernel.outer(kernel).expand(channels, 1, size, size).contiguous()
    return F.conv2d(images, k2, padding=size // 2, groups=channels)


def bag_of_predicates(texts, vocabulary) -> np.ndarray:
    out = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
    for i, text in enumerate(texts):
        lowered = text.lower()
        for j, phrase in enumerate(vocabulary):
            out[i, j] = float(phrase.lower() in lowered)
    return out


def ridge_fit_predict(train_x, train_y, test_x, alpha: float = 1.0):
    x = np.concatenate([train_x, np.ones((train_x.shape[0], 1), np.float32)], axis=1)
    xt = np.concatenate([test_x, np.ones((test_x.shape[0], 1), np.float32)], axis=1)
    gram = x.T @ x + alpha * np.eye(x.shape[1], dtype=np.float32)
    weights = np.linalg.solve(gram, x.T @ train_y)
    return xt @ weights


def nearest_centroid(train_x, train_y, test_x):
    classes = sorted(set(train_y.tolist()))
    centroids = np.stack([train_x[train_y == c].mean(axis=0) for c in classes])
    distances = ((test_x[:, None, :] - centroids[None]) ** 2).sum(axis=-1)
    return np.array([classes[i] for i in distances.argmin(axis=1)])


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = base_parser("Downstream ITS task validation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-train", type=int, default=48)
    parser.add_argument("--num-test", type=int, default=24)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args)
    set_seed(cfg.optim.seed)
    bank = load_bank(cfg)
    train_set, test_set, _, collate_fn = build_datasets(cfg, bank)

    model = APLModel(cfg, bank=bank)
    if args.checkpoint:
        model.load(args.checkpoint)
    else:
        Trainer(model, cfg, collate_fn=collate_fn).fit(train_set, epochs=args.epochs)

    vocabulary = bank.semantic_categories()
    rng = np.random.default_rng(cfg.optim.seed)

    def build(dataset, count):
        images, labels, descriptions, captions = [], [], [], []
        for i in range(min(count, len(dataset))):
            sample = dataset[i]
            images.append(sample["image"])
            labels.append(int(sample["label"]))
            out = model.infer(sample["image"].unsqueeze(0))
            descriptions.append(out.description.text)
            captions.append("A traffic scene with vehicles on the road.")
        images = torch.stack(images)
        labels = np.array(labels)
        with torch.no_grad():
            original = model.backbone.zero_shot_image(images).cpu().numpy()
            blurred = model.backbone.zero_shot_image(
                gaussian_blur(images)
            ).cpu().numpy()
        features = {
            "Orig. Img.": original,
            "Blurred": blurred,
            "Caption baseline": bag_of_predicates(captions, vocabulary),
            "APL (Ours)": bag_of_predicates(descriptions, vocabulary),
        }
        # synthetic downstream targets, deterministic in the class label
        flow = labels * 1.7 + rng.normal(0, 0.5, size=labels.shape)
        pedestrians = (labels % 5).astype(np.float32)
        return features, labels, flow.astype(np.float32), pedestrians

    train_feats, train_labels, train_flow, train_ped = build(train_set, args.num_train)
    test_feats, test_labels, test_flow, test_ped = build(test_set, args.num_test)

    rows = []
    for name in train_feats:
        flow_pred = ridge_fit_predict(
            train_feats[name], train_flow[:, None], test_feats[name]
        ).ravel()
        sign_pred = nearest_centroid(train_feats[name], train_labels, test_feats[name])
        ped_pred = ridge_fit_predict(
            train_feats[name], train_ped[:, None], test_feats[name]
        ).ravel()
        rows.append(
            downstream_report(
                name,
                flow_pred=flow_pred.tolist(), flow_true=test_flow.tolist(),
                sign_pred=sign_pred.tolist(), sign_true=test_labels.tolist(),
                ped_pred=ped_pred.tolist(), ped_true=test_ped.tolist(),
            ).as_row()
        )

    print(banner("Table VIII - downstream ITS performance"))
    print(format_table(rows))
    print("\nlower RMSE / MAE and higher accuracy are better; the original "
          "image is the utility reference, not a theoretical upper bound.")
    print("saved:", save_json(Path(cfg.output_dir) / "downstream.json", rows))


if __name__ == "__main__":
    main()
