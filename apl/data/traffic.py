"""Traffic-scene dataset with scenario-level annotations.

The dataset is described by a JSON manifest so that any of the public sources
used in the paper (BDD100K, LISA, a red-light-violation set, COCO, ADE20K) can
be plugged in without changing the code:

.. code-block:: json

    {
      "classes": ["the traffic signal facing the vehicle shows a steady red light", "..."],
      "samples": [
        {
          "image": "images/0001.jpg",
          "label": 0,
          "scene_label": "violation",
          "violations": ["red_light_running"],
          "predicates": {"signal_state_red": 1, "vehicle_in_motion": 1},
          "sensitive_attributes": ["license_plate", "driver_face"],
          "relevant_attributes": ["signal_state_red", "vehicle_crosses_stop_line"],
          "reference": "A vehicle crosses the stop line while the signal is red."
        }
      ]
    }

``scene_label`` uses the paper's three-way vocabulary (``violation``,
``non-violation``, ``ambiguous``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .fewshot import load_image

SCENE_LABELS = {"violation": 0, "non-violation": 1, "ambiguous": 2}


class TrafficSceneDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 224,
        bank=None,
        root: Optional[str | Path] = None,
    ):
        self.manifest_path = Path(manifest)
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.root = Path(root) if root else self.manifest_path.parent
        self.classes: List[str] = raw.get("classes", [])
        self.samples: List[Dict] = raw["samples"]
        self.image_size = image_size
        self.bank = bank

    # ------------------------------------------------------------------ #
    def category_names(self) -> List[str]:
        if self.classes:
            return self.classes
        if self.bank is not None:
            return self.bank.semantic_categories()
        raise ValueError("the manifest defines no classes and no bank was given")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        path = self.root / item["image"]
        out: Dict = {
            "image": load_image(path, self.image_size),
            "label": int(item.get("label", 0)),
            "index": idx,
            "path": str(path),
            "scene_label": item.get("scene_label", "non-violation"),
            "violations": item.get("violations", []),
            "sensitive_attributes": item.get("sensitive_attributes", []),
            "relevant_attributes": item.get("relevant_attributes", []),
            "reference": item.get("reference", ""),
        }
        if self.bank is not None:
            labels = torch.zeros(len(self.bank))
            for j, scenario in enumerate(self.bank.scenarios):
                if scenario.violation_type in out["violations"]:
                    labels[j] = 1.0
            out["scenario_labels"] = labels
            predicates = item.get("predicates", {})
            if predicates:
                vector = torch.zeros(self.bank.num_predicates)
                for pid, value in predicates.items():
                    if pid in self.bank.predicate_index:
                        vector[self.bank.predicate_index[pid]] = float(value)
                out["predicate_labels"] = vector
        return out


def write_manifest(path: str | Path, classes: Sequence[str],
                   samples: Sequence[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"classes": list(classes), "samples": list(samples)}, fh, indent=2)
