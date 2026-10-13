"""Configuration objects for APL.

The defaults reproduce Table I of the paper ("APL architecture, prompt-learning
configuration, and traffic-scene inference settings").  Every field can be
overridden from a YAML file or from the command line.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - PyYAML is a hard requirement but keep the import soft
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class EncoderConfig:
    """Frozen CLIP ViT-B/16 vision-language backbone (Sec. V-A)."""

    backbone: str = "ViT-B/16"
    #: `clip` (OpenAI package), `open_clip`, or `fallback` (bundled re-implementation).
    provider: str = "auto"
    pretrained: str = "openai"
    embed_dim: int = 512          # d
    vision_width: int = 768
    vision_layers: int = 12
    vision_heads: int = 12
    patch_size: int = 16
    image_resolution: int = 224
    text_width: int = 512
    text_layers: int = 12
    text_heads: int = 8
    vocab_size: int = 49408
    context_length: int = 77
    num_local_features: int = 196  # N = (224/16)^2
    freeze: bool = True            # E_t and E_v remain frozen
    device: str = "cpu"
    checkpoint: Optional[str] = None


@dataclass
class PromptConfig:
    """Learnable prompt tokens inserted at the encoder inputs (Table I)."""

    base_prompt: str = (
        "Describe the traffic scene including vehicles, signals, road markings"
    )
    class_template: str = "{base}: {category}"
    n_text_tokens: int = 4
    n_visual_tokens: int = 4
    init_std: float = 0.02
    insertion: str = "input"  # prompts are inserted at the encoder inputs only


@dataclass
class FeatureGeneratorConfig:
    """Cross-attention Feature Generator theta_FG (Sec. V-B2)."""

    dim: int = 512          # d
    key_dim: int = 512      # d_k
    mid_dim: int = 256      # d_mid = d / 2
    num_heads: int = 1      # one FG attention head
    dropout: float = 0.0
    residual: bool = True


@dataclass
class BranchConfig:
    """Three complementary feature branches (Sec. V-B3..V-B5)."""

    use_global_invariant: bool = True
    use_instance_dependent: bool = True
    use_fine_grained: bool = True
    num_fg_tokens: int = 4        # Z
    eta: float = 1.0              # multi-scale factor
    #: `proportion` -> k_z = ceil(z/Z * eta * N); `count` -> k_z = ceil(z * eta).
    fg_topk_mode: str = "proportion"
    fg_init_std: float = 0.02


@dataclass
class LossConfig:
    """Loss weights (Table I)."""

    lambda_gi: float = 1.0
    lambda_id: float = 1.0
    lambda_fg: float = 1.0
    lambda1: float = 0.1   # ||w^GI - w^zs||
    lambda2: float = 0.1   # ||f_hat - f_hat^zs||
    temperature: float = 0.07  # tau


@dataclass
class OptimConfig:
    """SGD schedule from Table I."""

    optimizer: str = "sgd"
    lr: float = 2e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    warmup_epochs: int = 5
    warmup_lr: float = 1e-5
    min_lr: float = 1e-6
    grad_clip: float = 0.0
    seed: int = 1


@dataclass
class LogicConfig:
    """Differentiable soft-logic interface and rule engine (Sec. V-E, V-G)."""

    scenario_bank: str = "data/scenarios/dmv_scenario_bank.json"
    tau_init: float = 0.25      # predicate threshold initialisation
    kappa: float = 10.0         # sigmoid temperature
    gamma_loose: float = 0.40   # Gamma_loose
    gamma_strict: float = 0.80  # Gamma_strict
    #: predicate is considered "unresolved" (a gap) below this satisfaction value
    gap_threshold: float = 0.50
    learnable_tau: bool = True
    #: FG rows are scored against local features instead of the global embedding
    fg_local_evidence: bool = True


@dataclass
class RefinementConfig:
    """Gap-conditioned iterative prompt refinement (Sec. V-F)."""

    t_max: int = 5
    epsilon: float = 1e-3
    max_gap_prompts: int = 8
    gap_token_length: int = 8  # L rows of alpha^gap
    enabled: bool = True


@dataclass
class DataConfig:
    root: str = "data"
    dataset: str = "synthetic"
    num_shots: int = 16
    subsample: str = "all"  # all | base | new
    image_size: int = 224
    num_workers: int = 0


@dataclass
class APLConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    feature_generator: FeatureGeneratorConfig = field(
        default_factory=FeatureGeneratorConfig
    )
    branches: BranchConfig = field(default_factory=BranchConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "runs/apl"

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to save configurations")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APLConfig":
        cfg = cls()
        _merge_into_dataclass(cfg, data or {})
        return cfg

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "APLConfig":
        data: Dict[str, Any] = {}
        if path is not None:
            if yaml is None:  # pragma: no cover
                raise RuntimeError("PyYAML is required to load configurations")
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(data)
        if overrides:
            cfg.update(overrides)
        return cfg

    # ------------------------------------------------------------------ #
    def update(self, overrides: Dict[str, Any]) -> "APLConfig":
        """Apply dotted-key overrides, e.g. ``{"optim.lr": 0.01}``."""
        for key, value in overrides.items():
            if value is None:
                continue
            target: Any = self
            parts = key.split(".")
            for part in parts[:-1]:
                if not hasattr(target, part):
                    raise KeyError(f"unknown configuration section: {key}")
                target = getattr(target, part)
            leaf = parts[-1]
            if not hasattr(target, leaf):
                raise KeyError(f"unknown configuration key: {key}")
            current = getattr(target, leaf)
            setattr(target, leaf, _coerce(current, value))
        return self

    def clone(self) -> "APLConfig":
        return copy.deepcopy(self)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _coerce(current: Any, value: Any) -> Any:
    if current is None or isinstance(value, type(current)):
        return value
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return str(value)
    return value


def _merge_into_dataclass(obj: Any, data: Dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown configuration key: {key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(obj, key, _coerce(current, value))


def parse_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """Parse ``key=value`` CLI overrides into a dict."""
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override must look like key=value, got: {pair}")
        key, value = pair.split("=", 1)
        out[key.strip()] = _literal(value.strip())
    return out


def _literal(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
