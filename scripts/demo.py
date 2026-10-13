#!/usr/bin/env python3
"""End-to-end APL demo: train briefly, then run the full inference pipeline.

Runs on CPU in well under a minute with ``configs/smoke.yaml`` and needs no
dataset download, so it doubles as an installation check::

    python scripts/demo.py --config configs/smoke.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from apl.cli import banner, base_parser, build_datasets, load_bank, load_config, save_json
from apl.eval import format_table, privacy_score, semantic_utility, text_quality
from apl.model import APLModel
from apl.train import Trainer, set_seed


def main() -> None:
    parser = base_parser("APL end-to-end demo on synthetic traffic scenes")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-scenes", type=int, default=6,
                        help="scenes to describe after training")
    args = parser.parse_args()

    cfg = load_config(args)
    cfg.data.dataset = "synthetic"
    set_seed(cfg.optim.seed)

    bank = load_bank(cfg)
    train_set, test_set, categories, collate_fn = build_datasets(cfg, bank)

    print(banner("1. Model"))
    model = APLModel(cfg, bank=bank)
    print(f"backbone provider : {model.backbone.provider} ({cfg.encoder.backbone})")
    print(f"scenarios M       : {len(bank)}")
    print(f"predicates / C    : {bank.num_predicates}")
    for key, value in model.parameter_report().items():
        print(f"  {key:20s} {value:,}")

    print(banner("2. Prompt learning"))
    trainer = Trainer(model, cfg, collate_fn=collate_fn)
    trainer.fit(train_set, test_set, epochs=args.epochs)
    print(f"test top-1 accuracy: {trainer.evaluate(test_set):.2f}%")
    print("per-branch accuracy:", {
        k: round(v, 2) for k, v in trainer.branch_accuracies(test_set).items()
    })

    print(banner("3. Iterative evidence extraction and rule-based decision"))
    descriptions, references, sensitive, relevant = [], [], [], []
    positive = []
    for i in range(min(args.num_scenes, len(test_set))):
        sample = test_set[i]
        out = model.infer(sample["image"].unsqueeze(0))
        descriptions.append(out.description.text)
        positive.append(out.description.positive_text)
        references.append(sample["reference"])
        sensitive.append(sample["sensitive_attributes"])
        relevant.append(sample["relevant_attributes"])

        print(f"\n--- scene {i} " + "-" * 50)
        print(f"description : {out.description.text}")
        print(f"stopped at  : t={out.iterations} ({out.stop_reason})")
        if out.refinement_prompts:
            print("refinement  :")
            for prompt in out.refinement_prompts[:3]:
                print(f"   - {prompt}")
        top = sorted(out.engine.decisions, key=lambda d: -d.gamma)[:3]
        print("top scenarios:")
        for decision in top:
            print("   " + decision.explain().splitlines()[0])

    print(banner("4. Privacy and utility of the generated descriptions"))
    quality = text_quality(descriptions, references)
    rows = [{
        "Privacy Score (%)": round(privacy_score(descriptions, sensitive), 1),
        "Semantic Utility (%)": round(semantic_utility(positive, relevant), 1),
        **quality.as_row(),
    }]
    print(format_table(rows))

    out_path = save_json(
        Path(cfg.output_dir) / "demo.json",
        {"descriptions": descriptions, "metrics": rows[0]},
    )
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
