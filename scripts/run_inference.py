"""
APL Framework — Inference Entry Point (Algorithm 1)

Implements the full iterative violation evidence extraction pipeline
described in Algorithm 1 of the paper:

    Input:  image I, scenario bank S
    Output: violation decision y ∈ {violation, no_violation, insufficient_evidence}

Usage:
    from scripts.run_inference import APLPipeline, run_inference
    from PIL import Image

    image = Image.open("traffic_scene.jpg")
    result = run_inference(image)
    print(result)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from models import (
    VisionLanguageEncoder,
    AdaptivePromptLearning,
    EvidenceExtractor,
    ScenarioBankMatcher,
    DMVScenario,
    RuleEngine,
)
from models.rule_engine import DecisionResult, ViolationDecision
from models.scenario_bank import build_default_scenario_bank


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the APL inference pipeline."""
    feature_dim: int = 512
    num_semantic_classes: int = 11
    num_fine_grained: int = 4
    num_local_features: int = 196
    eta: float = 0.5
    temperature: float = 0.07
    max_iterations: int = 3
    convergence_eps: float = 0.01
    gamma_strict: float = 0.75
    gamma_loose: float = 0.40
    image_size: int = 224
    use_mock_encoder: bool = True


# ---------------------------------------------------------------------------
# Full APL Pipeline
# ---------------------------------------------------------------------------

class APLPipeline:
    """
    End-to-end Adaptive Prompt Learning pipeline.

    Wraps all five modules and runs Algorithm 1:
        1. Vision-Language Encoding
        2. Adaptive Prompt Learning (initial feature generation)
        3. Evidence Extraction
        4. Scenario Bank Matching
        5. Iterative Refinement until convergence or T_max
        6. Rule-Based Decision

    Args:
        config:    PipelineConfig (uses defaults if not provided)
        scenarios: Optional list of DMVScenario objects
        device:    torch.device (defaults to CPU)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        scenarios: Optional[List[DMVScenario]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or PipelineConfig()
        self.device = device or torch.device("cpu")

        cfg = self.config

        # Module 1: Vision-Language Encoder
        self.encoder = VisionLanguageEncoder(
            feature_dim=cfg.feature_dim,
            num_local_features=cfg.num_local_features,
            num_semantic_classes=cfg.num_semantic_classes,
            use_mock=cfg.use_mock_encoder,
        ).to(self.device)

        # Module 2: Adaptive Prompt Learning
        self.apl = AdaptivePromptLearning(
            feature_dim=cfg.feature_dim,
            num_semantic_classes=cfg.num_semantic_classes,
            num_fine_grained=cfg.num_fine_grained,
            eta=cfg.eta,
            temperature=cfg.temperature,
            num_local_features=cfg.num_local_features,
        ).to(self.device)

        # Module 3: Evidence Extractor
        self.extractor = EvidenceExtractor(
            feature_dim=cfg.feature_dim,
            num_semantic_classes=cfg.num_semantic_classes,
            num_fine_grained=cfg.num_fine_grained,
            temperature=cfg.temperature,
        ).to(self.device)

        # Module 4: Scenario Bank Matcher
        self.matcher = ScenarioBankMatcher(
            scenarios=scenarios or build_default_scenario_bank(),
            num_classes=cfg.num_semantic_classes,
        ).to(self.device)

        # Module 5: Rule Engine
        self.rule_engine = RuleEngine(
            gamma_strict=cfg.gamma_strict,
            gamma_loose=cfg.gamma_loose,
        ).to(self.device)

        # Set all modules to eval mode for inference
        self._eval_mode()

    def _eval_mode(self):
        for m in [self.encoder, self.apl, self.extractor, self.matcher, self.rule_engine]:
            m.eval()

    # ------------------------------------------------------------------

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a PIL image into a normalised tensor.

        Returns: 1 x 3 x H x W
        """
        import torchvision.transforms as T  # lazy import
        transform = T.Compose([
            T.Resize((self.config.image_size, self.config.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def preprocess_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Accept a raw tensor (B x 3 x H x W) and move to device."""
        return x.to(self.device)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        image_input,   # PIL.Image or torch.Tensor (B x 3 x H x W)
    ) -> Dict:
        """
        Run Algorithm 1: Iterative Violation Evidence Extraction.

        Args:
            image_input: PIL image or pre-processed tensor

        Returns:
            Dict with:
                "decisions":     List[DecisionResult]
                "gamma_history": List[float] per iteration
                "scenario":      Matched DMVScenario
                "description":   Final evidence description dict
                "num_iters":     Number of iterations run
        """
        cfg = self.config

        # --- Step 1: Image encoding ---
        if isinstance(image_input, Image.Image):
            x = self.preprocess_image(image_input)
        else:
            x = self.preprocess_tensor(image_input)

        f_hat, f, w = self.encoder(x)          # (B,d), (B,N,d), (M,d)

        # --- Step 2: Initial APL feature generation ---
        apl_out = self.apl(w, f_hat, f)        # dict with w_gi, h_id, h_fg, log_probs
        w_gi  = apl_out["w_gi"]                # M x d
        h_id  = apl_out["h_id"]                # B x M x d
        h_fg  = apl_out["h_fg"]                # Z x M x d
        w_current = w.clone()                  # M x d  — evolves during refinement

        # --- Step 3: Initial evidence extraction ---
        description = self.extractor(f_hat, f, w_gi, h_id, h_fg)

        # --- Step 4: Scenario bank matching ---
        gamma, best_idx, gap_predicates = self.matcher(description["probs_mean"])
        gamma_val = gamma[0].item()
        gamma_history = [gamma_val]
        scenario = self.matcher.scenarios[best_idx]

        # --- Step 5: Iterative refinement loop ---
        num_iters = 1
        for t in range(1, cfg.max_iterations):
            if not gap_predicates:
                break   # all evidence satisfied

            # Generate gap-conditioned signal α_gap
            gap_signal = self.extractor.build_gap_signal(
                gap_predicates, w_current, self.device
            )                                          # L x d

            # Refine text features: T_{t+1} = θ_SFG(T_t, α_gap)
            w_refined = self.apl.refine_with_gaps(w_current, gap_signal)  # M x d
            w_current = w_refined

            # Re-generate all three branches with refined prompts
            w_gi_new = self.apl.compute_gi_features()
            h_id_new = self.apl.compute_id_features(w_current, f_hat)
            h_fg_new = self.apl.compute_fg_features(w_current)

            # Re-extract evidence
            description = self.extractor(f_hat, f, w_gi_new, h_id_new, h_fg_new)

            # Re-match
            gamma_new, best_idx, gap_predicates = self.matcher(
                description["probs_mean"]
            )
            gamma_new_val = gamma_new[0].item()
            gamma_history.append(gamma_new_val)
            scenario = self.matcher.scenarios[best_idx]
            num_iters += 1

            # Convergence check: Δγ < ε
            if self.rule_engine.convergence_check(gamma_val, gamma_new_val, cfg.convergence_eps):
                gamma_val = gamma_new_val
                break
            gamma_val = gamma_new_val

        # --- Step 6: Rule-based decision ---
        final_gamma = torch.full(
            (x.size(0),), gamma_val, device=self.device
        )
        decisions = self.rule_engine(
            final_gamma,
            scenario_type=scenario.violation_type,
            scenario_idx=best_idx,
            num_iterations=num_iters,
            gamma_history=gamma_history,
        )

        return {
            "decisions": decisions,
            "gamma_history": gamma_history,
            "scenario": scenario,
            "description": description,
            "num_iters": num_iters,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_inference(
    image_input,
    config: Optional[PipelineConfig] = None,
) -> Dict:
    """
    Run the full APL pipeline on a single image.

    Args:
        image_input: PIL.Image or torch.Tensor (1 x 3 x H x W)
        config:      Optional PipelineConfig

    Returns:
        Dict with decision results and metadata
    """
    pipeline = APLPipeline(config=config)
    result = pipeline.run(image_input)
    decision = result["decisions"][0]
    return {
        "decision": decision.decision.value,
        "confidence": round(decision.confidence, 4),
        "scenario_type": decision.scenario_type,
        "num_iterations": decision.num_iterations,
        "gamma_history": result["gamma_history"],
        "description": {
            k: v.shape if isinstance(v, torch.Tensor) else v
            for k, v in result["description"].items()
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="APL Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--gamma-strict", type=float, default=0.75)
    parser.add_argument("--gamma-loose", type=float, default=0.40)
    args = parser.parse_args()

    cfg = PipelineConfig(
        max_iterations=args.max_iter,
        gamma_strict=args.gamma_strict,
        gamma_loose=args.gamma_loose,
    )

    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        # Use a random synthetic image for demonstration
        print("[INFO] No image provided — using synthetic random tensor.")
        img = torch.randn(1, 3, 224, 224)

    result = run_inference(img, config=cfg)
    print("\n=== APL Inference Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
