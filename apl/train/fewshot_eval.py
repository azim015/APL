"""Few-shot and base-to-new evaluation protocols (Tables II and III).

``run_fewshot``      trains with ``K`` shots per class and reports top-1 accuracy.
``run_base_to_new``  trains on the base classes, then evaluates on the base and
                     on the held-out new classes and reports the harmonic mean
                     ``H = 2 * Base * New / (Base + New)``.
``shot_curve``       repeats the protocol for 1/2/4/8/16 shots (Fig. 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from torch.utils.data import Dataset

from ..model import APLModel
from .trainer import Trainer, set_seed


def harmonic_mean(base: float, new: float) -> float:
    if base + new == 0:
        return 0.0
    return 2.0 * base * new / (base + new)


@dataclass
class FewShotResult:
    shots: int
    accuracy: float
    seed: int
    history: List = field(default_factory=list)


@dataclass
class BaseToNewResult:
    base: float
    new: float
    harmonic: float
    seed: int

    def as_row(self) -> Dict[str, float]:
        return {
            "Base": round(self.base, 2),
            "New": round(self.new, 2),
            "H": round(self.harmonic, 2),
        }


def run_fewshot(
    cfg,
    train_set: Dataset,
    test_set: Dataset,
    categories: Sequence[str],
    collate_fn: Optional[Callable] = None,
    log_fn: Optional[Callable[[str], None]] = print,
) -> FewShotResult:
    set_seed(cfg.optim.seed)
    model = APLModel(cfg, categories=list(categories))
    trainer = Trainer(model, cfg, collate_fn=collate_fn, log_fn=log_fn)
    trainer.fit(train_set, test_set)
    accuracy = trainer.evaluate(test_set)
    return FewShotResult(
        shots=cfg.data.num_shots, accuracy=accuracy, seed=cfg.optim.seed,
        history=trainer.history,
    )


def run_base_to_new(
    cfg,
    base_train: Dataset,
    base_test: Dataset,
    new_test: Dataset,
    base_categories: Sequence[str],
    new_categories: Sequence[str],
    collate_fn: Optional[Callable] = None,
    log_fn: Optional[Callable[[str], None]] = print,
) -> BaseToNewResult:
    set_seed(cfg.optim.seed)
    model = APLModel(cfg, categories=list(base_categories))
    trainer = Trainer(model, cfg, collate_fn=collate_fn, log_fn=log_fn)
    trainer.fit(base_train, base_test)

    base_acc = trainer.evaluate(base_test)
    # the prompts and the feature generator transfer; only the label set changes
    model.set_categories(list(new_categories))
    new_acc = trainer.evaluate(new_test)
    model.set_categories(list(base_categories))

    return BaseToNewResult(
        base=base_acc, new=new_acc, harmonic=harmonic_mean(base_acc, new_acc),
        seed=cfg.optim.seed,
    )


def shot_curve(
    cfg,
    dataset_factory: Callable[[int], tuple],
    categories: Sequence[str],
    shots: Sequence[int] = (1, 2, 4, 8, 16),
    collate_fn: Optional[Callable] = None,
    log_fn: Optional[Callable[[str], None]] = print,
) -> List[FewShotResult]:
    """Fig. 2: accuracy as a function of the number of labelled examples."""
    results = []
    for k in shots:
        run_cfg = cfg.clone()
        run_cfg.data.num_shots = k
        train_set, test_set = dataset_factory(k)
        results.append(
            run_fewshot(run_cfg, train_set, test_set, categories, collate_fn, log_fn)
        )
    return results


def average_over_seeds(results: Sequence[float]) -> Dict[str, float]:
    values = list(results)
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "std": var ** 0.5}
