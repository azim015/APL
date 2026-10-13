#!/usr/bin/env python3
"""Base-to-new generalisation protocol (Table III).

Trains on the base half of the classes and evaluates on both halves, reporting
``Base``, ``New`` and their harmonic mean ``H``::

    python scripts/eval_base_to_new.py --config configs/smoke.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apl.cli import banner, base_parser, build_datasets, load_bank, load_config, save_json
from apl.data.fewshot import FewShotImageFolder
from apl.eval import format_table
from apl.train import average_over_seeds, run_base_to_new


def split_synthetic(train_set, test_set, categories):
    """Split the synthetic dataset's label space into base / new halves."""
    from apl.data.synthetic import SyntheticTrafficDataset

    half = (len(categories) + 1) // 2
    base_names, new_names = categories[:half], categories[half:]

    def rebuild(source, names, seed_shift):
        return SyntheticTrafficDataset(
            names, len(source), source.image_size,
            seed=source.seed + seed_shift, bank=source.bank,
        )

    return (
        rebuild(train_set, base_names, 0),
        rebuild(test_set, base_names, 0),
        rebuild(test_set, new_names, 13),
        base_names,
        new_names,
    )


def main() -> None:
    parser = base_parser("Base-to-new generalisation with APL")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args)
    seeds = args.seeds or [cfg.optim.seed]
    if args.epochs:
        cfg.optim.epochs = args.epochs

    results = []
    for seed in seeds:
        run_cfg = cfg.clone()
        run_cfg.optim.seed = seed

        if run_cfg.data.dataset == "synthetic":
            bank = load_bank(run_cfg)
            train_set, test_set, categories, collate_fn = build_datasets(run_cfg, bank)
            base_train, base_test, new_test, base_names, new_names = split_synthetic(
                train_set, test_set, categories
            )
        else:
            root = Path(run_cfg.data.root)
            collate_fn = None
            base_train = FewShotImageFolder(
                root / "train", run_cfg.data.num_shots, run_cfg.data.image_size,
                "base", seed,
            )
            test_root = root / "test" if (root / "test").exists() else root / "train"
            base_test = FewShotImageFolder(
                test_root, 0, run_cfg.data.image_size, "base", seed
            )
            new_test = FewShotImageFolder(
                test_root, 0, run_cfg.data.image_size, "new", seed
            )
            base_names = base_train.category_names()
            new_names = new_test.category_names()

        print(banner(
            f"seed {seed}: {len(base_names)} base / {len(new_names)} new classes"
        ))
        result = run_base_to_new(
            run_cfg, base_train, base_test, new_test, base_names, new_names,
            collate_fn=collate_fn,
        )
        print(result.as_row())
        results.append(result)

    rows = [{"Seed": r.seed, **r.as_row()} for r in results]
    rows.append({
        "Seed": "mean",
        "Base": round(average_over_seeds([r.base for r in results])["mean"], 2),
        "New": round(average_over_seeds([r.new for r in results])["mean"], 2),
        "H": round(average_over_seeds([r.harmonic for r in results])["mean"], 2),
    })
    print(banner("Base-to-new summary"))
    print(format_table(rows))
    print("\nsaved:", save_json(Path(cfg.output_dir) / "base_to_new.json", rows))


if __name__ == "__main__":
    main()
