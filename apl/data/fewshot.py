"""Few-shot image folders and the base-to-new split protocol.

Table II trains with 16 labelled examples per class; Table III splits the
classes into a *base* half (seen during training) and a *new* half (held out)
and reports the harmonic mean of the two accuracies.

``FewShotImageFolder`` reads the usual ``root/<class_name>/<image>`` layout
with Pillow only, so no torchvision dependency is required.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# CLIP's preprocessing constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_image(path: str | Path, image_size: int = 224) -> torch.Tensor:
    """Resize-crop-normalise an image exactly as CLIP expects it."""
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = image_size / min(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.BICUBIC)
        w, h = img.size
        left, top = (w - image_size) // 2, (h - image_size) // 2
        img = img.crop((left, top, left + image_size, top + image_size))
        tensor = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    tensor = tensor.view(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (tensor - mean) / std


class FewShotImageFolder(Dataset):
    def __init__(
        self,
        root: str | Path,
        num_shots: int = 16,
        image_size: int = 224,
        subsample: str = "all",
        seed: int = 1,
        class_names: Optional[Sequence[str]] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.num_shots = num_shots
        self.seed = seed
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")

        classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        ) if class_names is None else list(class_names)
        classes = self.subsample_classes(classes, subsample)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        rng = random.Random(seed)
        self.samples: List[Tuple[Path, int]] = []
        for name in classes:
            files = sorted(
                p for p in (self.root / name).iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if num_shots > 0 and len(files) > num_shots:
                files = rng.sample(files, num_shots)
            self.samples.extend((p, self.class_to_idx[name]) for p in files)

    # ------------------------------------------------------------------ #
    @staticmethod
    def subsample_classes(classes: Sequence[str], subsample: str) -> List[str]:
        """``base`` keeps the first half of the classes, ``new`` the rest."""
        classes = list(classes)
        if subsample == "all":
            return classes
        half = (len(classes) + 1) // 2
        if subsample == "base":
            return classes[:half]
        if subsample == "new":
            return classes[half:]
        raise ValueError(f"unknown subsample mode: {subsample}")

    # ------------------------------------------------------------------ #
    def category_names(self) -> List[str]:
        return [c.replace("_", " ") for c in self.classes]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, label = self.samples[idx]
        return {
            "image": load_image(path, self.image_size),
            "label": label,
            "index": idx,
            "path": str(path),
        }
