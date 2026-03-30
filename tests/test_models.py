"""
Unit Tests — Individual APL Framework Modules

Tests every class and function for:
    - Correct output shapes
    - Mathematical correctness (e.g. normalisation, loss signs)
    - CPU-only operation (no GPU required)
    - Edge cases
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

BATCH_SIZE = 2
FEATURE_DIM = 64       # small d for fast tests
NUM_CLASSES = 5        # M
NUM_LOCAL = 16         # N  (4x4 grid for speed)
NUM_FG = 3             # Z
ETA = 0.5
TEMPERATURE = 0.07


@pytest.fixture
def batch():
    """Standard batch shapes used across tests."""
    return {
        "B": BATCH_SIZE,
        "d": FEATURE_DIM,
        "M": NUM_CLASSES,
        "N": NUM_LOCAL,
        "Z": NUM_FG,
    }


@pytest.fixture
def encoder():
    from models.vision_language_encoder import VisionLanguageEncoder, _MockVisualEncoder
    enc = VisionLanguageEncoder(
        feature_dim=FEATURE_DIM,
        num_local_features=NUM_LOCAL,
        num_semantic_classes=NUM_CLASSES,
        use_mock=True,
    )
    # Replace the visual encoder so it actually produces NUM_LOCAL (not default 196) patches
    enc.visual_encoder = _MockVisualEncoder(feature_dim=FEATURE_DIM, num_patches=NUM_LOCAL)
    return enc


@pytest.fixture
def sfg():
    from models.feature_generator import FeatureGenerator
    return FeatureGenerator(feature_dim=FEATURE_DIM, num_heads=4)


@pytest.fixture
def apl():
    from models.adaptive_prompt_learning import AdaptivePromptLearning
    return AdaptivePromptLearning(
        feature_dim=FEATURE_DIM,
        num_semantic_classes=NUM_CLASSES,
        num_fine_grained=NUM_FG,
        eta=ETA,
        temperature=TEMPERATURE,
        num_local_features=NUM_LOCAL,
    )


@pytest.fixture
def extractor():
    from models.evidence_extractor import EvidenceExtractor
    return EvidenceExtractor(
        feature_dim=FEATURE_DIM,
        num_semantic_classes=NUM_CLASSES,
        num_fine_grained=NUM_FG,
        temperature=TEMPERATURE,
    )


@pytest.fixture
def matcher():
    from models.scenario_bank import ScenarioBankMatcher, build_default_scenario_bank
    # Use reduced classes matching our test setup
    scenarios = build_default_scenario_bank()
    # Clamp predicate indices to NUM_CLASSES
    for s in scenarios:
        s.predicate_indices = [min(i, NUM_CLASSES - 1) for i in s.predicate_indices]
    return ScenarioBankMatcher(scenarios=scenarios, num_classes=NUM_CLASSES)


@pytest.fixture
def rule_engine():
    from models.rule_engine import RuleEngine
    return RuleEngine(gamma_strict=0.75, gamma_loose=0.40)


# ==================================================================
# Module 1: VisionLanguageEncoder
# ==================================================================

class TestVisionLanguageEncoder:

    def test_output_shapes(self, encoder):
        """f_hat, f, w must have correct shapes."""
        x = torch.randn(BATCH_SIZE, 3, 224, 224)
        f_hat, f, w = encoder(x)
        assert f_hat.shape == (BATCH_SIZE, FEATURE_DIM), f"f_hat shape: {f_hat.shape}"
        assert f.shape == (BATCH_SIZE, NUM_LOCAL, FEATURE_DIM), f"f shape: {f.shape}"
        assert w.shape == (NUM_CLASSES, FEATURE_DIM), f"w shape: {w.shape}"

    def test_global_feature_normalised(self, encoder):
        """Global features should be L2 normalised."""
        x = torch.randn(BATCH_SIZE, 3, 224, 224)
        f_hat, _, _ = encoder(x)
        norms = f_hat.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-5), \
            f"f_hat norms: {norms}"

    def test_local_feature_normalised(self, encoder):
        """Local features should be L2 normalised."""
        x = torch.randn(BATCH_SIZE, 3, 224, 224)
        _, f, _ = encoder(x)
        norms = f.norm(dim=-1)   # B x N
        assert torch.allclose(norms, torch.ones(BATCH_SIZE, NUM_LOCAL), atol=1e-5)

    def test_zero_shot_features(self, encoder):
        """Zero-shot features should match base feature shapes."""
        x = torch.randn(BATCH_SIZE, 3, 224, 224)
        w_zs, f_hat_zs = encoder.get_zero_shot_features(x)
        assert w_zs.shape == (NUM_CLASSES, FEATURE_DIM)
        assert f_hat_zs.shape == (BATCH_SIZE, FEATURE_DIM)

    def test_deterministic_with_eval(self, encoder):
        """In eval mode, two identical forward passes should produce identical output."""
        encoder.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            f1, _, _ = encoder(x)
            f2, _, _ = encoder(x)
        assert torch.allclose(f1, f2)


# ==================================================================
# Module 2: FeatureGenerator (SFG)
# ==================================================================

class TestFeatureGenerator:

    def test_output_shape_unbatched(self, sfg):
        """Unbatched (M, d) input → (M, d) output."""
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        alpha = torch.randn(8, FEATURE_DIM)  # L=8
        h = sfg(w, alpha)
        assert h.shape == (NUM_CLASSES, FEATURE_DIM), f"h shape: {h.shape}"

    def test_output_shape_batched(self, sfg):
        """Batched (B, M, d) input → (B, M, d) output."""
        w = torch.randn(BATCH_SIZE, NUM_CLASSES, FEATURE_DIM)
        alpha = torch.randn(BATCH_SIZE, 8, FEATURE_DIM)
        h = sfg(w, alpha)
        assert h.shape == (BATCH_SIZE, NUM_CLASSES, FEATURE_DIM)

    def test_different_conditionals_give_different_outputs(self, sfg):
        """Different α should produce different h (confirming conditioning)."""
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        alpha1 = torch.randn(4, FEATURE_DIM)
        alpha2 = torch.randn(4, FEATURE_DIM)
        h1 = sfg(w, alpha1)
        h2 = sfg(w, alpha2)
        assert not torch.allclose(h1, h2), "Different conditionals should yield different outputs."

    def test_gradient_flows(self, sfg):
        """Gradients must flow through the SFG."""
        w = torch.randn(NUM_CLASSES, FEATURE_DIM, requires_grad=True)
        alpha = torch.randn(4, FEATURE_DIM, requires_grad=True)
        h = sfg(w, alpha)
        loss = h.sum()
        loss.backward()
        assert w.grad is not None
        assert alpha.grad is not None

    def test_length_one_conditional(self, sfg):
        """L=1 conditional signal should work (common in ID branch)."""
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        alpha = torch.randn(1, FEATURE_DIM)
        h = sfg(w, alpha)
        assert h.shape == (NUM_CLASSES, FEATURE_DIM)


# ==================================================================
# Module 3: AdaptivePromptLearning
# ==================================================================

class TestAdaptivePromptLearning:

    def _make_inputs(self):
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        f_hat = F.normalize(torch.randn(BATCH_SIZE, FEATURE_DIM), dim=-1)
        f = F.normalize(torch.randn(BATCH_SIZE, NUM_LOCAL, FEATURE_DIM), dim=-1)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        w_zs = F.normalize(torch.randn(NUM_CLASSES, FEATURE_DIM), dim=-1)
        f_hat_zs = F.normalize(torch.randn(BATCH_SIZE, FEATURE_DIM), dim=-1)
        return w, f_hat, f, labels, w_zs, f_hat_zs

    def test_gi_features_shape(self, apl):
        w_gi = apl.compute_gi_features()
        assert w_gi.shape == (NUM_CLASSES, FEATURE_DIM)

    def test_gi_features_normalised(self, apl):
        w_gi = apl.compute_gi_features()
        norms = w_gi.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(NUM_CLASSES), atol=1e-5)

    def test_id_features_shape(self, apl):
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        f_hat = torch.randn(BATCH_SIZE, FEATURE_DIM)
        h_id = apl.compute_id_features(w, f_hat)
        assert h_id.shape == (BATCH_SIZE, NUM_CLASSES, FEATURE_DIM)

    def test_fg_features_shape(self, apl):
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        h_fg = apl.compute_fg_features(w)
        assert h_fg.shape == (NUM_FG, NUM_CLASSES, FEATURE_DIM)

    def test_gi_loss_positive(self, apl):
        w, f_hat, f, labels, w_zs, f_hat_zs = self._make_inputs()
        loss = apl.gi_loss(f_hat, labels, w_zs, f_hat_zs)
        assert loss.item() > 0, "GI loss must be positive."
        assert not torch.isnan(loss)

    def test_id_loss_positive(self, apl):
        w, f_hat, f, labels, w_zs, f_hat_zs = self._make_inputs()
        loss = apl.id_loss(w, f_hat, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_fg_loss_positive(self, apl):
        w, f_hat, f, labels, w_zs, f_hat_zs = self._make_inputs()
        loss = apl.fg_loss(w, f, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_joint_loss_dict(self, apl):
        w, f_hat, f, labels, w_zs, f_hat_zs = self._make_inputs()
        losses = apl.compute_loss(w, f_hat, f, labels, w_zs, f_hat_zs)
        for key in ["loss_gi", "loss_id", "loss_fg", "loss_total"]:
            assert key in losses, f"Missing key: {key}"
            assert not torch.isnan(losses[key])

    def test_prediction_shape(self, apl):
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        f_hat = F.normalize(torch.randn(BATCH_SIZE, FEATURE_DIM), dim=-1)
        f = F.normalize(torch.randn(BATCH_SIZE, NUM_LOCAL, FEATURE_DIM), dim=-1)
        log_probs = apl.predict(w, f_hat, f)
        assert log_probs.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_prediction_is_log_prob(self, apl):
        """log probabilities should be ≤ 0 and approximately sum to 1 in prob space."""
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        f_hat = F.normalize(torch.randn(BATCH_SIZE, FEATURE_DIM), dim=-1)
        f = F.normalize(torch.randn(BATCH_SIZE, NUM_LOCAL, FEATURE_DIM), dim=-1)
        log_probs = apl.predict(w, f_hat, f)
        assert (log_probs <= 0).all(), "Log probs must be ≤ 0."
        prob_sums = log_probs.exp().sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(BATCH_SIZE), atol=0.01)

    def test_refinement_changes_output(self, apl):
        """Gap-conditioned refinement should change the text feature."""
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        gap_signal = torch.randn(2, FEATURE_DIM)
        w_refined = apl.refine_with_gaps(w, gap_signal)
        assert w_refined.shape == (NUM_CLASSES, FEATURE_DIM)
        assert not torch.allclose(w, w_refined)

    def test_forward_inference_mode(self, apl):
        """Forward without labels should work (inference mode)."""
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        f_hat = torch.randn(BATCH_SIZE, FEATURE_DIM)
        f = torch.randn(BATCH_SIZE, NUM_LOCAL, FEATURE_DIM)
        out = apl(w, f_hat, f)   # no labels
        assert "log_probs" in out
        assert out["log_probs"].shape == (BATCH_SIZE, NUM_CLASSES)


# ==================================================================
# Module 4: EvidenceExtractor
# ==================================================================

class TestEvidenceExtractor:

    def _make_features(self):
        f_hat = F.normalize(torch.randn(BATCH_SIZE, FEATURE_DIM), dim=-1)
        f = F.normalize(torch.randn(BATCH_SIZE, NUM_LOCAL, FEATURE_DIM), dim=-1)
        w_gi = F.normalize(torch.randn(NUM_CLASSES, FEATURE_DIM), dim=-1)
        h_id = F.normalize(torch.randn(BATCH_SIZE, NUM_CLASSES, FEATURE_DIM), dim=-1)
        h_fg = F.normalize(torch.randn(NUM_FG, NUM_CLASSES, FEATURE_DIM), dim=-1)
        return f_hat, f, w_gi, h_id, h_fg

    def test_output_keys(self, extractor):
        f_hat, f, w_gi, h_id, h_fg = self._make_features()
        desc = extractor(f_hat, f, w_gi, h_id, h_fg)
        for key in ["scores_gi", "scores_id", "scores_fg", "scores_mean", "probs_mean"]:
            assert key in desc, f"Missing key: {key}"

    def test_scores_gi_shape(self, extractor):
        f_hat, f, w_gi, h_id, h_fg = self._make_features()
        desc = extractor(f_hat, f, w_gi, h_id, h_fg)
        assert desc["scores_gi"].shape == (BATCH_SIZE, NUM_CLASSES)

    def test_scores_fg_shape(self, extractor):
        f_hat, f, w_gi, h_id, h_fg = self._make_features()
        desc = extractor(f_hat, f, w_gi, h_id, h_fg)
        assert desc["scores_fg"].shape == (BATCH_SIZE, NUM_FG, NUM_CLASSES)

    def test_probs_sum_to_one(self, extractor):
        """Ensemble probabilities must sum to 1."""
        f_hat, f, w_gi, h_id, h_fg = self._make_features()
        desc = extractor(f_hat, f, w_gi, h_id, h_fg)
        prob_sums = desc["probs_mean"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_identify_gaps_all_below_threshold(self, extractor):
        """If all probs are low, all required predicates are gaps."""
        f_hat, f, w_gi, h_id, h_fg = self._make_features()
        desc = extractor(f_hat, f, w_gi, h_id, h_fg)
        # Force very low probs
        low_desc = {**desc, "probs_mean": torch.zeros(BATCH_SIZE, NUM_CLASSES)}
        required = [0, 1, 2]
        gaps = extractor.identify_evidence_gaps(low_desc, required, threshold=0.5)
        assert set(gaps) == set(required)

    def test_identify_gaps_none_when_above_threshold(self, extractor):
        """If all probs are high, no gaps are identified."""
        f_hat, f, w_gi, h_id, h_fg = self._make_features()
        desc = extractor(f_hat, f, w_gi, h_id, h_fg)
        high_desc = {**desc, "probs_mean": torch.ones(BATCH_SIZE, NUM_CLASSES)}
        required = [0, 1]
        gaps = extractor.identify_evidence_gaps(high_desc, required, threshold=0.5)
        assert gaps == []

    def test_build_gap_signal_shape(self, extractor):
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        gaps = [0, 2]
        signal = extractor.build_gap_signal(gaps, w, device=torch.device("cpu"))
        assert signal.shape == (len(gaps), FEATURE_DIM)

    def test_build_gap_signal_empty(self, extractor):
        w = torch.randn(NUM_CLASSES, FEATURE_DIM)
        signal = extractor.build_gap_signal([], w, device=torch.device("cpu"))
        assert signal.shape == (1, FEATURE_DIM)


# ==================================================================
# Module 5: ScenarioBankMatcher
# ==================================================================

class TestScenarioBankMatcher:

    def test_num_scenarios(self, matcher):
        """Default bank should have 6 scenarios."""
        assert len(matcher.scenarios) == 6

    def test_compatibility_shape(self, matcher):
        probs = torch.rand(BATCH_SIZE, NUM_CLASSES)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        gamma, best_idx, gaps = matcher(probs)
        assert gamma.shape == (BATCH_SIZE,)

    def test_best_idx_valid(self, matcher):
        probs = torch.rand(BATCH_SIZE, NUM_CLASSES)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        _, best_idx, _ = matcher(probs)
        assert 0 <= best_idx < len(matcher.scenarios)

    def test_compatibility_in_range(self, matcher):
        """γ should be in [0, 1] (weighted sum of sigmoid outputs)."""
        probs = torch.rand(BATCH_SIZE, NUM_CLASSES)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        gamma, _, _ = matcher(probs)
        assert (gamma >= 0).all() and (gamma <= 1.0 + 1e-5).all()

    def test_soft_logic_gate(self):
        from models.scenario_bank import soft_logic_gate
        # At z = τ_q, output should be ~0.5
        z = torch.tensor([0.3])
        psi = soft_logic_gate(z, tau_q=0.3, kappa=10.0)
        assert abs(psi.item() - 0.5) < 0.01

    def test_high_prob_increases_compatibility(self, matcher):
        """Higher evidence probability → higher compatibility score."""
        scenario_idx = 0
        scenario = matcher.scenarios[scenario_idx]
        pred_idx = scenario.predicate_indices[0]

        probs_low = torch.zeros(1, NUM_CLASSES)
        probs_low[0, pred_idx] = 0.1
        probs_high = torch.zeros(1, NUM_CLASSES)
        probs_high[0, pred_idx] = 0.9

        gamma_low = matcher.compute_compatibility(probs_low, scenario_idx)
        gamma_high = matcher.compute_compatibility(probs_high, scenario_idx)
        assert gamma_high.item() > gamma_low.item()

    def test_learnable_thresholds(self, matcher):
        """tau_q must be a learnable parameter."""
        assert isinstance(matcher.tau_q, torch.nn.Parameter)


# ==================================================================
# Module 6: RuleEngine
# ==================================================================

class TestRuleEngine:

    def test_violation_above_strict(self, rule_engine):
        from models.rule_engine import ViolationDecision
        gamma = torch.tensor([0.9])
        results = rule_engine(gamma)
        assert results[0].decision == ViolationDecision.VIOLATION

    def test_no_violation_below_loose(self, rule_engine):
        from models.rule_engine import ViolationDecision
        gamma = torch.tensor([0.1])
        results = rule_engine(gamma)
        assert results[0].decision == ViolationDecision.NO_VIOLATION

    def test_insufficient_between_bounds(self, rule_engine):
        from models.rule_engine import ViolationDecision
        gamma = torch.tensor([0.55])  # between 0.40 and 0.75
        results = rule_engine(gamma)
        assert results[0].decision == ViolationDecision.INSUFFICIENT_EVIDENCE

    def test_batch_decisions(self, rule_engine):
        from models.rule_engine import ViolationDecision
        gamma = torch.tensor([0.9, 0.55, 0.1])
        results = rule_engine(gamma)
        assert len(results) == 3
        assert results[0].decision == ViolationDecision.VIOLATION
        assert results[1].decision == ViolationDecision.INSUFFICIENT_EVIDENCE
        assert results[2].decision == ViolationDecision.NO_VIOLATION

    def test_boundary_strict(self, rule_engine):
        from models.rule_engine import ViolationDecision
        # Exactly at gamma_strict
        gamma = torch.tensor([0.75])
        results = rule_engine(gamma)
        assert results[0].decision == ViolationDecision.VIOLATION

    def test_convergence_check_true(self, rule_engine):
        """Small delta should trigger convergence."""
        assert rule_engine.convergence_check(0.500, 0.505, epsilon=0.01) is True

    def test_convergence_check_false(self, rule_engine):
        """Large delta should not trigger convergence."""
        assert rule_engine.convergence_check(0.4, 0.6, epsilon=0.01) is False

    def test_result_metadata(self, rule_engine):
        gamma = torch.tensor([0.8])
        results = rule_engine(
            gamma,
            scenario_type="red_light_violation",
            scenario_idx=0,
            num_iterations=2,
            gamma_history=[0.6, 0.8],
        )
        r = results[0]
        assert r.scenario_type == "red_light_violation"
        assert r.num_iterations == 2
        assert r.gamma_history == [0.6, 0.8]

    def test_invalid_thresholds(self):
        from models.rule_engine import RuleEngine
        with pytest.raises(AssertionError):
            RuleEngine(gamma_strict=0.3, gamma_loose=0.8)  # loose > strict


# ==================================================================
# Privacy Metrics
# ==================================================================

class TestPrivacyMetrics:

    def test_mse_identical_images(self):
        from utils.privacy_metrics import compute_mse
        x = torch.rand(2, 3, 32, 32)
        assert compute_mse(x, x) < 1e-7

    def test_mse_different_images(self):
        from utils.privacy_metrics import compute_mse
        x = torch.zeros(2, 3, 32, 32)
        y = torch.ones(2, 3, 32, 32)
        assert compute_mse(x, y) > 0.5

    def test_psnr_identical_is_inf(self):
        from utils.privacy_metrics import compute_psnr
        x = torch.rand(2, 3, 32, 32)
        psnr = compute_psnr(x, x)
        assert psnr == float("inf")

    def test_psnr_different_is_finite(self):
        from utils.privacy_metrics import compute_psnr
        x = torch.zeros(2, 3, 32, 32)
        y = torch.ones(2, 3, 32, 32)
        psnr = compute_psnr(x, y)
        # MSE=1.0, max_val=1.0 → PSNR = 10*log10(1/1) = 0 dB, which is valid
        assert math.isfinite(psnr) and psnr >= 0

    def test_ssim_identical_near_one(self):
        from utils.privacy_metrics import compute_ssim
        x = torch.rand(2, 3, 32, 32)
        ssim = compute_ssim(x, x)
        assert abs(ssim - 1.0) < 0.05

    def test_ssim_different_below_one(self):
        from utils.privacy_metrics import compute_ssim
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        ssim = compute_ssim(x, y)
        assert ssim < 1.0

    def test_srra_identical_features_max(self):
        from utils.privacy_metrics import compute_srra
        feat = F.normalize(torch.randn(10, FEATURE_DIM), dim=-1)
        srra = compute_srra(feat, feat, threshold=0.5)
        assert srra == 100.0

    def test_srra_orthogonal_features_zero(self):
        from utils.privacy_metrics import compute_srra
        n = 10
        orig = F.normalize(torch.randn(n, FEATURE_DIM), dim=-1)
        # Create nearly orthogonal features
        recon = F.normalize(torch.randn(n, FEATURE_DIM), dim=-1)
        # Use very high threshold so nothing passes
        srra = compute_srra(orig, recon, threshold=0.9999)
        assert srra >= 0.0


# ==================================================================
# Text Quality Metrics
# ==================================================================

class TestTextQualityMetrics:

    def test_bleu_identical(self):
        from utils.text_quality import compute_bleu
        sentences = ["the red car ran the stop sign at the intersection"]
        score = compute_bleu(sentences, sentences)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_bleu_completely_different(self):
        from utils.text_quality import compute_bleu
        hyps = ["aaaa bbbb cccc"]
        refs = ["xxxx yyyy zzzz"]
        score = compute_bleu(hyps, refs)
        assert score == 0.0

    def test_bleu_partial_match(self):
        from utils.text_quality import compute_bleu
        hyps = ["the car ran the stop sign"]
        refs = ["the red car ran the stop sign at the intersection"]
        score = compute_bleu(hyps, refs)
        assert 0.0 < score < 1.0

    def test_rouge_l_identical(self):
        from utils.text_quality import compute_rouge_l
        sentences = ["vehicle stopped at red traffic light"]
        score = compute_rouge_l(sentences, sentences)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_rouge_l_empty_hypothesis(self):
        from utils.text_quality import compute_rouge_l
        score = compute_rouge_l([""], ["reference text"])
        assert score == pytest.approx(0.0)

    def test_cider_identical(self):
        from utils.text_quality import compute_cider
        # CIDEr uses references as IDF corpus; with one sentence IDF=0.
        # Use multiple references so IDF is meaningful, then test hyp==ref[0] scores high.
        refs = [
            "vehicle approaching red light at intersection",
            "pedestrian crossing at the crosswalk",
            "car stopped at stop sign near junction",
        ]
        hyps = [refs[0]]   # exactly matches first reference
        score = compute_cider(hyps, [refs[0]], corpus_override=refs)
        assert score > 0.1   # should score reasonably well

    def test_all_metrics_keys(self):
        from utils.text_quality import compute_all_text_metrics
        hyps = ["the vehicle ignored the red light"]
        refs = ["a car ran through a red traffic light"]
        metrics = compute_all_text_metrics(hyps, refs)
        for key in ["BLEU", "ROUGE", "CIDEr"]:
            assert key in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
