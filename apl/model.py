"""End-to-end Adaptive Prompt Learning model (Sec. V).

The five modules of the framework are wired here:

1. image VLM encoding              -> :class:`apl.encoders.CLIPBackbone`
2. adaptive prompt learning        -> :class:`apl.modules.PromptLearner` + branches
3. continuous evidence extraction  -> :class:`apl.logic.EvidenceExtractor`
4. scenario matching & rule engine -> :class:`apl.logic.RuleEngine`
5. gap-conditioned refinement      -> :class:`apl.refinement.PromptRefiner`

``APLModel.forward`` covers the prompt-learning (few-shot classification) path;
``APLModel.infer`` runs the full feedback-driven ITS pipeline of Algorithm 1 and
returns a privacy-preserving textual description plus one rule-based decision
per scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .config import APLConfig
from .describe import DescriptionComposer, SceneDescription
from .encoders import CLIPBackbone
from .logic import (
    EvidenceExtractor,
    RuleEngine,
    ScenarioBank,
    StructuredDescription,
    TextFeatureSet,
)
from .modules import (
    FeatureGenerator,
    FineGrainedBranch,
    GlobalInvariantBranch,
    InstanceDependentBranch,
    PromptLearner,
    apl_loss,
    ensemble_probability,
)
from .refinement import PromptRefiner, run_refinement_loop


@dataclass
class APLOutput:
    logits_gi: Optional[torch.Tensor] = None
    logits_id: Optional[torch.Tensor] = None
    logits_fg: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    loss_terms: Dict[str, float] = field(default_factory=dict)
    features: Optional[TextFeatureSet] = None
    f_hat: Optional[torch.Tensor] = None
    local: Optional[torch.Tensor] = None


@dataclass
class InferenceOutput:
    description: SceneDescription
    evidence: StructuredDescription
    engine: "object"
    gamma: torch.Tensor
    iterations: int
    stop_reason: str
    trace: List = field(default_factory=list)
    refinement_prompts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        out = self.engine.to_dict()
        out.update(
            {
                "description": self.description.text,
                "observed": self.description.observed,
                "uncertain": self.description.uncertain,
                "refinement_prompts": self.refinement_prompts,
                "trace": [t.to_dict() for t in self.trace],
            }
        )
        return out


class APLModel(nn.Module):
    def __init__(
        self,
        cfg: Optional[APLConfig] = None,
        categories: Optional[Sequence[str]] = None,
        bank: Optional[ScenarioBank] = None,
        backbone: Optional[CLIPBackbone] = None,
    ):
        super().__init__()
        self.cfg = cfg or APLConfig()
        self.backbone = backbone or CLIPBackbone(self.cfg.encoder)

        self.bank = bank
        if self.bank is None and categories is None:
            self.bank = ScenarioBank.load(self.cfg.logic.scenario_bank)
        if categories is None:
            categories = self.bank.semantic_categories()
        self.categories: List[str] = list(categories)

        self.prompt_learner = PromptLearner(
            self.backbone, self.categories, self.cfg.prompt
        )
        self.feature_generator = FeatureGenerator(self.cfg.feature_generator)

        b = self.cfg.branches
        self.gi_branch = (
            GlobalInvariantBranch(
                self.feature_generator, self.cfg.feature_generator.dim,
                init_std=b.fg_init_std,
            )
            if b.use_global_invariant else None
        )
        self.id_branch = (
            InstanceDependentBranch(self.feature_generator)
            if b.use_instance_dependent else None
        )
        self.fg_branch = (
            FineGrainedBranch(
                self.feature_generator, self.cfg.feature_generator.dim,
                num_tokens=b.num_fg_tokens, eta=b.eta,
                topk_mode=b.fg_topk_mode, init_std=b.fg_init_std,
            )
            if b.use_fine_grained else None
        )
        if self.gi_branch is None and self.id_branch is None and self.fg_branch is None:
            raise ValueError("at least one APL branch must be enabled")

        # ---- discrete symbolic layer (only when the rows of w are predicates)
        self.logic_enabled = (
            self.bank is not None
            and self.categories == self.bank.semantic_categories()
        )
        if self.logic_enabled:
            self.extractor = EvidenceExtractor(
                self.bank, self.cfg.logic, self.fg_branch
            )
            self.refiner = PromptRefiner(
                self.bank, self.prompt_learner, self.feature_generator,
                self.cfg.refinement, self.cfg.logic.gap_threshold,
            )
            self.rule_engine = RuleEngine(self.bank, self.cfg.logic)
            self.composer = DescriptionComposer(self.bank)
        else:
            self.extractor = None
            self.refiner = None
            self.rule_engine = None
            self.composer = None

        self.to(self.cfg.encoder.device)

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_image(self, images: torch.Tensor):
        """``f_hat, f = E_v(I)`` with the learnable visual prompt tokens."""
        images = images.to(self.device)
        return self.backbone.encode_image(images, self.prompt_learner.visual_ctx)

    def base_text_feature(self) -> torch.Tensor:
        """``w = E_t(p_0) in R^{C x d}``."""
        return self.prompt_learner()

    # ------------------------------------------------------------------ #
    def text_features(self, f_hat: torch.Tensor,
                      w: Optional[torch.Tensor] = None) -> TextFeatureSet:
        """``T_0 = {w^GI, h^ID, h^FG_z}``."""
        if w is None:
            w = self.base_text_feature()
        w_gi = self.gi_branch(w) if self.gi_branch is not None else None
        h_id = (
            self.id_branch(w, f_hat) if self.id_branch is not None else None
        )
        h_fg = self.fg_branch(w) if self.fg_branch is not None else None
        return TextFeatureSet(w_gi=w_gi, h_id=h_id, h_fg=h_fg, iteration=0)

    # ------------------------------------------------------------------ #
    # prompt-learning path
    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
    ) -> APLOutput:
        f_hat, local = self.encode_image(images)
        w = self.base_text_feature()
        features = self.text_features(f_hat, w)
        tau = self.cfg.loss.temperature

        logits_gi = logits_id = logits_fg = None
        if features.w_gi is not None:
            logits_gi = GlobalInvariantBranch.logits(features.w_gi, f_hat, tau)
        if features.h_id is not None:
            logits_id = InstanceDependentBranch.logits(features.h_id, f_hat, tau)
        if features.h_fg is not None:
            sim = self.fg_branch.similarity(features.h_fg, local)
            logits_fg = FineGrainedBranch.logits(sim, tau)

        out = APLOutput(
            logits_gi=logits_gi,
            logits_id=logits_id,
            logits_fg=logits_fg,
            probs=ensemble_probability(logits_gi, logits_id, logits_fg),
            features=features,
            f_hat=f_hat,
            local=local,
        )

        if labels is not None and compute_loss:
            labels = labels.to(f_hat.device)
            w_zs = f_hat_zs = None
            if self.gi_branch is not None:
                w_zs = self.prompt_learner.zero_shot_features()
                f_hat_zs = self.backbone.zero_shot_image(images.to(self.device))
            loss = apl_loss(
                logits_gi, logits_id, logits_fg, labels, self.cfg.loss,
                w_gi=features.w_gi, w_zs=w_zs, f_hat=f_hat, f_hat_zs=f_hat_zs,
            )
            out.loss = loss.total
            out.loss_terms = loss.item_dict()
        return out

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Ensemble class probabilities ``p(m|I)`` (Sec. V-B6)."""
        self.eval()
        return self.forward(images, compute_loss=False).probs

    # ------------------------------------------------------------------ #
    # full ITS pipeline (Algorithm 1)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer(self, images: torch.Tensor, index: int = 0) -> InferenceOutput:
        if not self.logic_enabled:
            raise RuntimeError(
                "the discrete layer is disabled: build the model with the "
                "scenario bank categories to run rule-based inference"
            )
        self.eval()
        f_hat, local = self.encode_image(images)
        features = self.text_features(f_hat)

        result = run_refinement_loop(
            self.extractor, self.refiner, features, f_hat, local,
            self.cfg.refinement, index=index,
        )
        engine = self.rule_engine.evaluate(
            result.description, result.gamma, index=index,
            iterations=result.iterations, stop_reason=result.stop_reason,
        )
        description = self.composer.compose(result.description, engine, index)
        return InferenceOutput(
            description=description,
            evidence=result.description,
            engine=engine,
            gamma=result.gamma[index].detach(),
            iterations=result.iterations,
            stop_reason=result.stop_reason,
            trace=result.trace,
            refinement_prompts=result.prompts,
        )

    @torch.no_grad()
    def infer_batch(self, images: torch.Tensor) -> List[InferenceOutput]:
        return [self.infer(images, index=i) for i in range(images.shape[0])]

    # ------------------------------------------------------------------ #
    def set_categories(self, categories: Sequence[str]) -> None:
        """Re-target the model at a different label set (base-to-new transfer)."""
        self.categories = list(categories)
        self.prompt_learner.set_categories(self.categories)
        self.logic_enabled = (
            self.bank is not None
            and self.categories == self.bank.semantic_categories()
        )

    # ------------------------------------------------------------------ #
    # parameters
    # ------------------------------------------------------------------ #
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def parameter_report(self) -> Dict[str, int]:
        """Trainable parameter budget, grouped by component (Table I)."""
        seen: set = set()
        report: Dict[str, int] = {}

        def count(name: str, params) -> None:
            total = 0
            for param in params:
                if param is None or not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))
                total += param.numel()
            report[name] = total

        count("prompt_tokens", self.prompt_learner.parameters())
        count("feature_generator", self.feature_generator.parameters())
        count("gi_alpha", [] if self.gi_branch is None else [self.gi_branch.alpha])
        count("fg_alpha", [] if self.fg_branch is None else [self.fg_branch.alpha])
        count("gap_tokens", [] if self.refiner is None else [self.refiner.gap_tokens])
        count(
            "predicate_thresholds",
            [] if self.extractor is None
            else [p for p in self.extractor.gate.parameters()],
        )
        report["total_trainable"] = self.num_trainable()
        report["backbone_frozen"] = sum(
            p.numel() for p in self.backbone.parameters() if not p.requires_grad
        )
        return report

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        state = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("backbone.")
        }
        torch.save({"state_dict": state, "config": self.cfg.to_dict(),
                    "categories": self.categories}, path)

    def load(self, path: str, strict: bool = False) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(payload["state_dict"], strict=strict)
