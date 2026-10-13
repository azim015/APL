"""Deterministic synthetic traffic scenes.

The published experiments use BDD100K, the LISA traffic-light dataset, a
red-light-violation set, COCO and ADE20K, none of which can be redistributed
here.  This module generates a small, reproducible stand-in so that training,
inference, the refinement loop and every metric in :mod:`apl.eval` can be run
end-to-end without any download.

Each sample carries what the evaluation protocol needs:

* ``image``                 - a tensor scene with class-correlated structure,
* ``label``                 - the semantic/predicate class index,
* ``scenario_labels``       - binary violation labels over the scenario bank,
* ``sensitive_attributes``  - the PII present in the raw frame (Privacy Score),
* ``relevant_attributes``   - the task-relevant evidence (Semantic Utility),
* ``reference``             - a human-style reference caption (BLEU/ROUGE/CIDEr).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

SENSITIVE_POOL = [
    "driver_face",
    "passenger_face",
    "license_plate",
    "pedestrian_face",
    "street_address",
    "vehicle_vin",
]


@dataclass
class SyntheticSample:
    image: torch.Tensor
    label: int
    scenario_labels: torch.Tensor
    sensitive_attributes: List[str]
    relevant_attributes: List[str]
    reference: str
    index: int


class SyntheticTrafficDataset(Dataset):
    def __init__(
        self,
        categories: Sequence[str],
        num_samples: int = 64,
        image_size: int = 224,
        seed: int = 0,
        bank=None,
        noise: float = 0.35,
    ):
        self.categories = list(categories)
        self.num_classes = len(self.categories)
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.bank = bank
        self.noise = noise
        self._prototypes = self._build_prototypes()

    # ------------------------------------------------------------------ #
    def _build_prototypes(self) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed)
        size = self.image_size
        protos = torch.zeros(self.num_classes, 3, size, size)
        grid = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        for c in range(self.num_classes):
            freq = 1.0 + 1.5 * c
            phase = 2.0 * math.pi * c / max(1, self.num_classes)
            base = torch.sin(freq * math.pi * xx + phase) * torch.cos(
                freq * math.pi * yy - phase
            )
            colour = torch.rand(3, 1, 1, generator=g) * 0.6 + 0.2
            protos[c] = base.unsqueeze(0) * colour
        return protos

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        label = int(torch.randint(self.num_classes, (1,), generator=g))
        image = self._prototypes[label] + self.noise * torch.randn(
            3, self.image_size, self.image_size, generator=g
        )

        num_scenarios = len(self.bank) if self.bank is not None else 12
        scenario_labels = (
            torch.rand(num_scenarios, generator=g) < 0.12
        ).float()

        n_sensitive = int(torch.randint(1, 4, (1,), generator=g))
        perm = torch.randperm(len(SENSITIVE_POOL), generator=g)[:n_sensitive]
        sensitive = [SENSITIVE_POOL[i] for i in perm.tolist()]

        relevant = [self.categories[label]]
        if self.bank is not None:
            extra = torch.randperm(len(self.bank.predicates), generator=g)[:2]
            relevant += [self.bank.predicates[i].text for i in extra.tolist()]
        relevant = list(dict.fromkeys(relevant))

        reference = (
            f"The scene shows that {self.categories[label]}. "
            "No identifying details of road users are reported."
        )

        return {
            "image": image,
            "label": label,
            "scenario_labels": scenario_labels,
            "sensitive_attributes": sensitive,
            "relevant_attributes": relevant,
            "reference": reference,
            "index": idx,
        }


def collate(batch: Sequence[Dict]) -> Dict:
    """Collate that keeps the variable-length attribute lists intact."""
    out: Dict = {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }
    if "scenario_labels" in batch[0]:
        out["scenario_labels"] = torch.stack([b["scenario_labels"] for b in batch])
    for key in ("sensitive_attributes", "relevant_attributes", "reference"):
        if key in batch[0]:
            out[key] = [b[key] for b in batch]
    return out


def make_splits(
    categories: Sequence[str],
    num_train: int = 64,
    num_test: int = 32,
    image_size: int = 224,
    seed: int = 0,
    bank=None,
):
    train = SyntheticTrafficDataset(
        categories, num_train, image_size, seed=seed, bank=bank
    )
    test = SyntheticTrafficDataset(
        categories, num_test, image_size, seed=seed + 977, bank=bank
    )
    return train, test
