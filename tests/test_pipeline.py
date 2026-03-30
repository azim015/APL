"""
End-to-End Pipeline Tests

Tests the full APL inference pipeline (Algorithm 1):
    - Correct decision types returned
    - Iterative refinement loop convergence
    - Gap detection and refinement
    - Batch processing
    - CPU-only operation
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from scripts.run_inference import APLPipeline, PipelineConfig, run_inference
from models.rule_engine import ViolationDecision
from models.scenario_bank import build_default_scenario_bank, DMVScenario


# ---------------------------------------------------------------------------
# Small config for fast CPU tests
# ---------------------------------------------------------------------------

@pytest.fixture
def fast_config():
    return PipelineConfig(
        feature_dim=64,
        num_semantic_classes=5,
        num_fine_grained=2,
        num_local_features=16,
        eta=0.5,
        temperature=0.07,
        max_iterations=2,
        convergence_eps=0.01,
        gamma_strict=0.75,
        gamma_loose=0.40,
        image_size=32,   # small images for speed
        use_mock_encoder=True,
    )


@pytest.fixture
def pipeline(fast_config):
    scenarios = build_default_scenario_bank()
    # Clamp predicate indices to match our small class space
    for s in scenarios:
        s.predicate_indices = [min(i, fast_config.num_semantic_classes - 1)
                               for i in s.predicate_indices]
    return APLPipeline(config=fast_config, scenarios=scenarios)


# ==================================================================
# Core pipeline tests
# ==================================================================

class TestAPLPipeline:

    def test_run_with_tensor_returns_dict(self, pipeline, fast_config):
        """Pipeline should return a dict with all expected keys."""
        x = torch.randn(1, 3, fast_config.image_size, fast_config.image_size)
        result = pipeline.run(x)
        for key in ["decisions", "gamma_history", "scenario", "description", "num_iters"]:
            assert key in result, f"Missing key: {key}"

    def test_decisions_is_list(self, pipeline, fast_config):
        x = torch.randn(1, 3, fast_config.image_size, fast_config.image_size)
        result = pipeline.run(x)
        assert isinstance(result["decisions"], list)
        assert len(result["decisions"]) == 1

    def test_decision_is_valid_enum(self, pipeline, fast_config):
        x = torch.randn(1, 3, fast_config.image_size, fast_config.image_size)
        result = pipeline.run(x)
        decision = result["decisions"][0].decision
        assert decision in list(ViolationDecision), f"Invalid decision: {decision}"

    def test_gamma_history_non_empty(self, pipeline, fast_config):
        x = torch.randn(1, 3, fast_config.image_size, fast_config.image_size)
        result = pipeline.run(x)
        assert len(result["gamma_history"]) >= 1

    def test_num_iters_within_max(self, pipeline, fast_config):
        x = torch.randn(1, 3, fast_config.image_size, fast_config.image_size)
        result = pipeline.run(x)
        assert 1 <= result["num_iters"] <= fast_config.max_iterations

    def test_scenario_is_dmv_scenario(self, pipeline, fast_config):
        x = torch.randn(1, 3, fast_config.image_size, fast_config.image_size)
        result = pipeline.run(x)
        assert isinstance(result["scenario"], DMVScenario)

    def test_description_contains_probs(self, pipeline, fast_config):
        x = torch.randn(1, 3, fast_config.image_size, fast_config.image_size)
        result = pipeline.run(x)
        assert "probs_mean" in result["description"]

    def test_deterministic_in_eval_mode(self, pipeline, fast_config):
        """Two identical forward passes should return the same decision."""
        x = torch.randn(1, 3, fast_config.image_size, fast_config.image_size)
        result1 = pipeline.run(x)
        result2 = pipeline.run(x)
        assert result1["decisions"][0].decision == result2["decisions"][0].decision

    def test_batch_of_two(self, pipeline, fast_config):
        """Pipeline should handle batch size 2."""
        x = torch.randn(2, 3, fast_config.image_size, fast_config.image_size)
        result = pipeline.run(x)
        assert len(result["decisions"]) == 2

    def test_confidence_in_valid_range(self, pipeline, fast_config):
        """Confidence (γ) should be in [0, 1]."""
        x = torch.randn(1, 3, fast_config.image_size, fast_config.image_size)
        result = pipeline.run(x)
        conf = result["decisions"][0].confidence
        assert 0.0 <= conf <= 1.0 + 1e-6, f"Confidence out of range: {conf}"


# ==================================================================
# run_inference convenience function
# ==================================================================

class TestRunInferenceFunction:

    def test_returns_dict(self, fast_config):
        x = torch.randn(1, 3, 32, 32)
        result = run_inference(x, config=fast_config)
        assert isinstance(result, dict)

    def test_required_keys(self, fast_config):
        x = torch.randn(1, 3, 32, 32)
        result = run_inference(x, config=fast_config)
        for key in ["decision", "confidence", "scenario_type", "num_iterations", "gamma_history"]:
            assert key in result, f"Missing key: {key}"

    def test_decision_is_string(self, fast_config):
        x = torch.randn(1, 3, 32, 32)
        result = run_inference(x, config=fast_config)
        assert isinstance(result["decision"], str)
        assert result["decision"] in [v.value for v in ViolationDecision]

    def test_confidence_type(self, fast_config):
        x = torch.randn(1, 3, 32, 32)
        result = run_inference(x, config=fast_config)
        assert isinstance(result["confidence"], float)

    def test_gamma_history_is_list(self, fast_config):
        x = torch.randn(1, 3, 32, 32)
        result = run_inference(x, config=fast_config)
        assert isinstance(result["gamma_history"], list)


# ==================================================================
# Rule logic integration tests
# ==================================================================

class TestRuleLogicIntegration:

    def test_high_gamma_gives_violation(self, fast_config):
        """Manually construct pipeline state with very high γ."""
        from models.rule_engine import RuleEngine
        rule = RuleEngine(gamma_strict=0.75, gamma_loose=0.40)
        gamma = torch.tensor([0.95])
        results = rule(gamma, scenario_type="red_light_violation")
        assert results[0].decision == ViolationDecision.VIOLATION

    def test_low_gamma_gives_no_violation(self, fast_config):
        from models.rule_engine import RuleEngine
        rule = RuleEngine(gamma_strict=0.75, gamma_loose=0.40)
        gamma = torch.tensor([0.05])
        results = rule(gamma)
        assert results[0].decision == ViolationDecision.NO_VIOLATION

    def test_medium_gamma_gives_insufficient(self, fast_config):
        from models.rule_engine import RuleEngine
        rule = RuleEngine(gamma_strict=0.75, gamma_loose=0.40)
        gamma = torch.tensor([0.57])
        results = rule(gamma)
        assert results[0].decision == ViolationDecision.INSUFFICIENT_EVIDENCE


# ==================================================================
# Scenario bank integration
# ==================================================================

class TestScenarioBankIntegration:

    def test_all_scenarios_have_valid_weights(self):
        """All predicate weights in a scenario must sum to approximately 1.0."""
        scenarios = build_default_scenario_bank()
        for s in scenarios:
            total = sum(s.predicate_weights)
            assert abs(total - 1.0) < 1e-5, \
                f"Scenario '{s.violation_type}' weights sum to {total}"

    def test_custom_scenario_can_be_added(self, fast_config):
        """Custom scenarios should integrate without errors."""
        custom = DMVScenario(
            violation_type="school_zone_speeding",
            predicate_names=["vehicle", "road_sign"],
            predicate_indices=[0, min(5, fast_config.num_semantic_classes - 1)],
            predicate_weights=[0.5, 0.5],
            description="Vehicle exceeds speed in school zone.",
        )
        all_scenarios = build_default_scenario_bank() + [custom]
        # Clamp indices
        for s in all_scenarios:
            s.predicate_indices = [
                min(i, fast_config.num_semantic_classes - 1) for i in s.predicate_indices
            ]
        pipeline = APLPipeline(config=fast_config, scenarios=all_scenarios)
        x = torch.randn(1, 3, 32, 32)
        result = pipeline.run(x)
        assert result["decisions"][0].decision in list(ViolationDecision)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
