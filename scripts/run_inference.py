#!/usr/bin/env python3
"""Run the APL traffic-scene pipeline (Algorithm 1) on real images.

Examples::

    # one or more image files
    python scripts/run_inference.py --images scene1.jpg scene2.jpg \\
        --checkpoint runs/its/apl.pt

    # a whole annotated manifest (see apl/data/traffic.py for the schema)
    python scripts/run_inference.py --manifest data/traffic/test.json

    # no arguments at all: synthetic scenes, so the pipeline can be inspected
    python scripts/run_inference.py --config configs/smoke.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from apl.cli import banner, base_parser, load_bank, load_config, save_json
from apl.data import SyntheticTrafficDataset, TrafficSceneDataset, load_image
from apl.model import APLModel


def main() -> None:
    parser = base_parser("Privacy-preserving traffic-scene inference with APL")
    parser.add_argument("--images", nargs="*", default=None, help="image paths")
    parser.add_argument("--manifest", default=None, help="annotated manifest")
    parser.add_argument("--checkpoint", default=None, help="trained APL weights")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--verbose-trace", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args)
    bank = load_bank(cfg)
    model = APLModel(cfg, bank=bank)
    if args.checkpoint:
        model.load(args.checkpoint)
        print(f"loaded checkpoint: {args.checkpoint}")

    # ---- assemble the batch of scenes ---------------------------------- #
    images, names = [], []
    if args.images:
        for path in args.images[: args.limit]:
            images.append(load_image(path, cfg.data.image_size))
            names.append(Path(path).name)
    elif args.manifest:
        dataset = TrafficSceneDataset(args.manifest, cfg.data.image_size, bank)
        for i in range(min(args.limit, len(dataset))):
            sample = dataset[i]
            images.append(sample["image"])
            names.append(Path(sample["path"]).name)
    else:
        dataset = SyntheticTrafficDataset(
            bank.semantic_categories(), args.limit, cfg.data.image_size,
            seed=cfg.optim.seed, bank=bank,
        )
        for i in range(len(dataset)):
            images.append(dataset[i]["image"])
            names.append(f"synthetic_{i:03d}")
        print("no --images/--manifest given: running on synthetic scenes")

    batch = torch.stack(images)

    # ---- inference ------------------------------------------------------ #
    payload = []
    for i, name in enumerate(names):
        out = model.infer(batch, index=i)
        print(banner(name))
        print(out.description.text)
        print()
        print(out.engine.report())
        if args.verbose_trace:
            print("\ntrace:")
            for step in out.trace:
                print(f"  t={step.iteration} gaps={step.num_gaps} "
                      f"delta={step.delta_gamma} prompts={len(step.prompts)}")
        record = out.to_dict()
        record["image"] = name
        payload.append(record)

    path = save_json(Path(cfg.output_dir) / "inference.json", payload)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
