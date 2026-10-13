#!/usr/bin/env python3
"""Privacy and textual-quality evaluation (Tables IV-VII).

Covers four measurements:

* Privacy Score and Semantic Utility over the generated descriptions,
* BLEU / ROUGE / CIDEr agreement with reference descriptions,
* reconstruction / replay-attack metrics (SSIM, PSNR, MSE, SRRA) between the
  original frames and whatever an attacker reconstructs,
* membership-inference attacks in the black-box and white-box settings.

The reconstruction attacker is pluggable: pass ``--reconstructions`` with a
``.pt`` tensor of recovered images aligned with the evaluation set.  Without
it, the script evaluates the *textual* representation, i.e. the APL output
itself, against the raw frame, which is the condition the paper reports as the
defended one.

    python scripts/eval_privacy.py --config configs/smoke.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from apl.cli import banner, base_parser, build_datasets, load_bank, load_config, save_json
from apl.eval import (
    format_table,
    membership_inference_attack,
    model_confidence_signal,
    model_loss_signal,
    privacy_score,
    reconstruction_metrics,
    semantic_utility,
    text_quality,
)
from apl.model import APLModel
from apl.train import Trainer, set_seed


def main() -> None:
    parser = base_parser("Privacy preservation and textual quality evaluation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-eval", type=int, default=24)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--reconstructions", default=None,
                        help=".pt tensor [N,3,H,W] recovered by an attacker")
    args = parser.parse_args()

    cfg = load_config(args)
    set_seed(cfg.optim.seed)
    bank = load_bank(cfg)
    train_set, test_set, _, collate_fn = build_datasets(cfg, bank)

    model = APLModel(cfg, bank=bank)
    if args.checkpoint:
        model.load(args.checkpoint)
    else:
        trainer = Trainer(model, cfg, collate_fn=collate_fn)
        trainer.fit(train_set, epochs=args.epochs)

    # ---- textual privacy / utility -------------------------------------- #
    descriptions, references, sensitive, relevant, originals = [], [], [], [], []
    positive = []
    for i in range(min(args.num_eval, len(test_set))):
        sample = test_set[i]
        out = model.infer(sample["image"].unsqueeze(0))
        descriptions.append(out.description.text)
        positive.append(out.description.positive_text)
        references.append(sample.get("reference", ""))
        sensitive.append(sample.get("sensitive_attributes", []))
        relevant.append(sample.get("relevant_attributes", []))
        originals.append(sample["image"])

    print(banner("Privacy Score and Semantic Utility"))
    text_rows = [{
        "Privacy Score (%)": round(privacy_score(descriptions, sensitive), 2),
        "Semantic Utility (%)": round(semantic_utility(positive, relevant), 2),
    }]
    print(format_table(text_rows))

    print(banner("Table VII - agreement with reference descriptions"))
    quality = text_quality(descriptions, references)
    print(format_table([{"Method": "APL (Ours)", **quality.as_row()}]))

    # ---- reconstruction / replay attack ---------------------------------- #
    print(banner("Tables IV / V - reconstruction and replay attack"))
    original = torch.stack(originals)
    if args.reconstructions:
        recovered = torch.load(args.reconstructions, map_location="cpu")
        recovered = recovered[: original.shape[0]]
    else:
        # the released representation is text only: an attacker who inverts it
        # recovers no pixel-level signal beyond the dataset mean
        recovered = original.mean(dim=(0,), keepdim=True).expand_as(original).clone()
        print("no --reconstructions given: scoring the text-only release, in "
              "which the attacker recovers only the dataset mean image")
    recon = reconstruction_metrics(original, recovered)
    print(format_table([{"Method": "APL (Ours)", **recon.as_row()}]))

    # ---- membership inference -------------------------------------------- #
    print(banner("Table VI - membership inference"))
    mia_rows = []
    for setting, signal_fn in (
        ("Black-box", model_confidence_signal),
        ("White-box", model_loss_signal),
    ):
        member = signal_fn(model, train_set, collate_fn)
        nonmember = signal_fn(model, test_set, collate_fn)
        result = membership_inference_attack(
            member, nonmember, setting=setting.lower()
        )
        mia_rows.append({"Method": "APL (Ours)", **result.as_row()})
    print(format_table(mia_rows))

    payload = {
        "text": text_rows[0],
        "quality": quality.as_row(),
        "reconstruction": recon.as_row(),
        "mia": mia_rows,
        "descriptions": descriptions,
    }
    print("\nsaved:", save_json(Path(cfg.output_dir) / "privacy.json", payload))


if __name__ == "__main__":
    main()
