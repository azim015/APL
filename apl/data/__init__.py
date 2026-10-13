"""Dataset interfaces: few-shot folders, annotated traffic scenes, synthetic data."""

from .fewshot import FewShotImageFolder, load_image
from .synthetic import SyntheticTrafficDataset, collate, make_splits
from .traffic import TrafficSceneDataset, write_manifest

__all__ = [
    "FewShotImageFolder",
    "load_image",
    "SyntheticTrafficDataset",
    "make_splits",
    "collate",
    "TrafficSceneDataset",
    "write_manifest",
]
