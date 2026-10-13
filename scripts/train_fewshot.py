#!/usr/bin/env python3
"""Few-shot prompt-learning training (Table II, Fig. 2).

Examples::

    # synthetic sanity run
    python scripts/train_fewshot.py --config configs/smoke.yaml

    # 16-shot run on an image folder laid out as root/train/<class>/<img>.jpg
    python scripts/train_fewshot.py --config configs/fewshot_16shot.yaml \\
        --set data.root=/datasets/imagenet data.dataset=imagefolder

    # the 1/2/4/8/16-shot curve of Fig. 2
    python scripts/train_fewshot.py --config configs/smoke.yaml --shot-curve
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apl.cli import banner, base_parser, build_datasets, load_bank, load_config, save_json
from apl.eval import format_table
from apl.model import APLModel
from apl.train import Trainer, average_over_seeds, set_seed


def main() -> None:
    parser = base_parser("Few-shot prompt learning with APL")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--shot-curve", action="store_true",
                        help="sweep 1/2/4/8/16 shots per class")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="repeat the run over several seeds (paper: 1 2 3)")
    args = parser.parse_args()

    cfg = load_config(args)
    if args.shots is not None:
        cfg.data.num_shots = args.shots

    shots = [1, 2, 4, 8, 16] if args.shot_curve else [cfg.data.num_shots]
    seeds = args.seeds or [cfg.optim.seed]
    rows, payload = [], []

    for shot in shots:
        accuracies = []
        for seed in seeds:
            run_cfg = cfg.clone()
            run_cfg.data.num_shots = shot
            run_cfg.optim.seed = seed
            set_seed(seed)

            bank = load_bank(run_cfg) if run_cfg.data.dataset == "synthetic" else None
            train_set, test_set, categories, collate_fn = build_datasets(run_cfg, bank)
            print(banner(f"{shot} shot(s) / seed {seed} / {len(categories)} classes"))

            model = APLModel(
                run_cfg,
                categories=None if bank is not None else categories,
                bank=bank,
            )
            trainer = Trainer(model, run_cfg, collate_fn=collate_fn)
            trainer.fit(train_set, test_set, epochs=args.epochs)

            accuracy = trainer.evaluate(test_set)
            accuracies.append(accuracy)
            print(f"top-1 accuracy: {accuracy:.2f}%")
            print("per-branch:", {
                k: round(v, 2) for k, v in trainer.branch_accuracies(test_set).items()
            })
            trainer.save(Path(run_cfg.output_dir) / f"apl_{shot}shot_seed{seed}.pt")

        stats = average_over_seeds(accuracies)
        rows.append({
            "Shots": shot,
            "Top-1 (%)": round(stats["mean"], 2),
            "Std": round(stats["std"], 2),
            "Seeds": len(seeds),
        })
        payload.append({"shots": shot, "accuracies": accuracies, **stats})

    print(banner("Few-shot summary"))
    print(format_table(rows))
    print("\nsaved:", save_json(Path(cfg.output_dir) / "fewshot.json", payload))


if __name__ == "__main__":
    main()
