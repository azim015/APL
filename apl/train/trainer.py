"""Training loop for the APL prompt-learning objective (Table I).

SGD, learning rate ``2e-3``, momentum ``0.9``, weight decay ``1e-4``,
batch size ``32``, ``50`` epochs, a five-epoch linear warm-up from ``1e-5``
followed by cosine decay to ``1e-6``.  Only the prompts, the feature
generator, the conditioning banks and the predicate thresholds are optimised -
the CLIP encoders stay frozen.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..model import APLModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: APLModel, cfg) -> torch.optim.Optimizer:
    params = model.trainable_parameters()
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unsupported optimizer: {cfg.optimizer}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    """Five-epoch linear warm-up, then cosine decay."""

    def lr_lambda(epoch: int) -> float:
        if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
            start = cfg.warmup_lr / cfg.lr
            return start + (1.0 - start) * (epoch / max(1, cfg.warmup_epochs))
        progress = (epoch - cfg.warmup_epochs) / max(
            1, cfg.epochs - cfg.warmup_epochs
        )
        progress = min(1.0, max(0.0, progress))
        floor = cfg.min_lr / cfg.lr
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    accuracy: float
    lr: float
    seconds: float
    terms: Dict[str, float] = field(default_factory=dict)


class Trainer:
    def __init__(self, model: APLModel, cfg, collate_fn: Optional[Callable] = None,
                 log_fn: Optional[Callable[[str], None]] = print):
        self.model = model
        self.cfg = cfg
        self.collate_fn = collate_fn
        self.log_fn = log_fn or (lambda _msg: None)
        self.optimizer = build_optimizer(model, cfg.optim)
        self.scheduler = build_scheduler(self.optimizer, cfg.optim)
        self.history: List[EpochRecord] = []

    # ------------------------------------------------------------------ #
    def log(self, message: str) -> None:
        self.log_fn(message)

    def loader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.optim.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.data.num_workers,
            collate_fn=self.collate_fn,
            drop_last=False,
        )

    # ------------------------------------------------------------------ #
    def fit(self, train_set: Dataset, val_set: Optional[Dataset] = None,
            epochs: Optional[int] = None) -> List[EpochRecord]:
        epochs = epochs or self.cfg.optim.epochs
        loader = self.loader(train_set, shuffle=True)
        self.model.train()

        for epoch in range(epochs):
            start = time.time()
            total_loss, correct, seen = 0.0, 0, 0
            term_sums: Dict[str, float] = {}

            for batch in loader:
                images = batch["image"].to(self.model.device)
                labels = batch["label"].to(self.model.device)
                out = self.model(images, labels)

                self.optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                if self.cfg.optim.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.trainable_parameters(), self.cfg.optim.grad_clip
                    )
                self.optimizer.step()

                batch_size = images.shape[0]
                total_loss += float(out.loss.detach()) * batch_size
                correct += int((out.probs.argmax(dim=-1) == labels).sum())
                seen += batch_size
                for key, value in out.loss_terms.items():
                    term_sums[key] = term_sums.get(key, 0.0) + value * batch_size

            self.scheduler.step()
            record = EpochRecord(
                epoch=epoch,
                loss=total_loss / max(1, seen),
                accuracy=100.0 * correct / max(1, seen),
                lr=self.optimizer.param_groups[0]["lr"],
                seconds=time.time() - start,
                terms={k: v / max(1, seen) for k, v in term_sums.items()},
            )
            self.history.append(record)
            message = (
                f"epoch {epoch + 1:3d}/{epochs} | loss {record.loss:.4f} | "
                f"train acc {record.accuracy:5.2f}% | lr {record.lr:.2e} | "
                f"{record.seconds:.1f}s"
            )
            if val_set is not None and (epoch + 1) % max(1, epochs // 5) == 0:
                message += f" | val acc {self.evaluate(val_set):5.2f}%"
                self.model.train()
            self.log(message)

        return self.history

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, dataset: Dataset) -> float:
        self.model.eval()
        loader = self.loader(dataset, shuffle=False)
        correct, seen = 0, 0
        for batch in loader:
            images = batch["image"].to(self.model.device)
            labels = batch["label"].to(self.model.device)
            probs = self.model(images, compute_loss=False).probs
            correct += int((probs.argmax(dim=-1) == labels).sum())
            seen += images.shape[0]
        return 100.0 * correct / max(1, seen)

    @torch.no_grad()
    def branch_accuracies(self, dataset: Dataset) -> Dict[str, float]:
        """Per-branch top-1 accuracy, used by the ablation study."""
        self.model.eval()
        loader = self.loader(dataset, shuffle=False)
        hits = {"gi": 0, "id": 0, "fg": 0, "ensemble": 0}
        seen = 0
        for batch in loader:
            images = batch["image"].to(self.model.device)
            labels = batch["label"].to(self.model.device)
            out = self.model(images, compute_loss=False)
            if out.logits_gi is not None:
                hits["gi"] += int((out.logits_gi.argmax(-1) == labels).sum())
            if out.logits_id is not None:
                hits["id"] += int((out.logits_id.argmax(-1) == labels).sum())
            if out.logits_fg is not None:
                fg = out.logits_fg.softmax(-1).mean(dim=1)
                hits["fg"] += int((fg.argmax(-1) == labels).sum())
            hits["ensemble"] += int((out.probs.argmax(-1) == labels).sum())
            seen += images.shape[0]
        return {k: 100.0 * v / max(1, seen) for k, v in hits.items()}

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
