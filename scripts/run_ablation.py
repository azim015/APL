#!/usr/bin/env python3
"""Component ablation study (Table IX).

    python scripts/run_ablation.py --config configs/smoke.yaml --epochs 10

Reports scene-level violation accuracy, Privacy Score and Semantic Utility for
the full model, for each removed component and for the single-branch variants.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apl.cli import banner, base_parser, build_datasets, load_bank, load_config, save_json
from apl.eval import VARIANTS, format_table, run_ablation
from apl.logic.rule_engine import NO_VIOLATION, VIOLATION


def main() -> None:
    parser = base_parser("APL component ablation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-eval", type=int, default=24)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="subset of variant names to run")
    args = parser.parse_args()

    cfg = load_config(args)
    bank = load_bank(cfg)
    train_set, test_set, _, collate_fn = build_datasets(cfg, bank)

    # scene-level ground truth in the V / N vocabulary
    samples = []
    for i in range(min(args.num_eval, len(test_set))):
        sample = test_set[i]
        labels = sample.get("scenario_labels")
        sample["scene_label_symbol"] = (
            VIOLATION if labels is not None and float(labels.sum()) > 0
            else NO_VIOLATION
        )
        samples.append(sample)

    print(banner("Ablation"))
    print(f"variants: {len(args.variants or VARIANTS)}  eval scenes: {len(samples)}")
    rows = run_ablation(
        cfg, train_set, samples, bank, collate_fn=collate_fn,
        epochs=args.epochs, variants=args.variants,
    )

    print(banner("Table IX - APL component ablation"))
    table = [r.as_row() for r in rows]
    print(format_table(table))
    print("\nsaved:", save_json(Path(cfg.output_dir) / "ablation.json", table))


if __name__ == "__main__":
    main()
