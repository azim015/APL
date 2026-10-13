"""Tests for the discrete layer: scenario bank, soft logic, rule engine."""

from __future__ import annotations

import unittest

import torch

from helpers import bank as load_bank, tiny_config, tiny_model
from apl.logic import (
    INSUFFICIENT,
    NO_VIOLATION,
    VIOLATION,
    RuleEngine,
    ScenarioBank,
    SoftLogicGate,
    compatibility_scores,
    gap_mask,
)
from apl.logic.scenario_bank import Predicate, Rule, Scenario


class TestScenarioBank(unittest.TestCase):
    def setUp(self):
        self.bank = load_bank()

    def test_bank_size_matches_table_i(self):
        self.assertEqual(len(self.bank), 12)

    def test_every_scenario_is_a_triple(self):
        for scenario in self.bank:
            self.assertTrue(scenario.violation_type)        # nu_j
            self.assertGreaterEqual(scenario.num_predicates, 3)  # E_j
            self.assertTrue(scenario.rule.required)         # r_j

    def test_predicate_weights_are_uniform_and_normalised(self):
        for scenario in self.bank:
            weights = scenario.weights
            self.assertAlmostEqual(sum(weights), 1.0, places=6)
            self.assertAlmostEqual(weights[0], 1.0 / scenario.num_predicates)

    def test_weight_matrix_rows_sum_to_one(self):
        matrix = self.bank.scenario_predicate_matrix()
        self.assertEqual(matrix.shape[0], len(self.bank))
        self.assertTrue(torch.allclose(matrix.sum(dim=1), torch.ones(len(self.bank)),
                                       atol=1e-6))

    def test_predicates_are_shared_across_scenarios(self):
        total = sum(s.num_predicates for s in self.bank)
        self.assertLess(self.bank.num_predicates, total)

    def test_round_trip_serialisation(self):
        restored = ScenarioBank.from_dict(self.bank.to_dict())
        self.assertEqual(len(restored), len(self.bank))
        self.assertEqual(restored.num_predicates, self.bank.num_predicates)

    def test_rule_evaluation(self):
        rule = Rule(type="all", required=["a", "b"])
        self.assertTrue(rule.evaluate({"a": True, "b": True}))
        self.assertFalse(rule.evaluate({"a": True, "b": False}))
        any_rule = Rule(type="any", required=["a", "b"])
        self.assertTrue(any_rule.evaluate({"a": False, "b": True}))

    def test_predicates_carry_no_identity_attributes(self):
        import re

        banned = ("name", "face", "plate", "gender", "race", "ethnicity")
        for predicate in self.bank.predicates:
            for word in banned:
                self.assertIsNone(
                    re.search(rf"\b{word}s?\b", predicate.text.lower()),
                    f"{predicate.id} mentions '{word}'",
                )


class TestSoftLogic(unittest.TestCase):
    def test_gate_is_calibrated_sigmoid(self):
        gate = SoftLogicGate(3, tau_init=0.25, kappa=10.0)
        z = torch.tensor([[0.25, 0.9, -0.5]])
        psi = gate(z).detach()
        self.assertAlmostEqual(float(psi[0, 0]), 0.5, places=5)
        self.assertGreater(float(psi[0, 1]), 0.99)
        self.assertLess(float(psi[0, 2]), 0.01)

    def test_tau_is_learnable(self):
        gate = SoftLogicGate(3)
        self.assertTrue(gate.tau.requires_grad)
        gate(torch.zeros(1, 3)).sum().backward()
        self.assertIsNotNone(gate.tau.grad)

    def test_calibration_moves_thresholds_towards_the_labels(self):
        gate = SoftLogicGate(2, tau_init=0.25, kappa=10.0)
        scores = torch.tensor([[0.8, 0.1], [0.75, 0.05], [0.9, 0.2]])
        labels = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        before = gate.tau.clone()
        gate.calibrate(scores, labels, steps=100)
        self.assertFalse(torch.allclose(before, gate.tau))

    def test_compatibility_score_is_a_weighted_average(self):
        psi = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        weights = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
        gamma = compatibility_scores(psi, weights)
        self.assertAlmostEqual(float(gamma[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(gamma[0, 1]), 0.0, places=6)

    def test_gap_mask_only_flags_required_predicates(self):
        psi = torch.tensor([[0.9, 0.1]])
        membership = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        mask = gap_mask(psi, membership, threshold=0.5)
        self.assertFalse(bool(mask[0, 0, 0]))   # satisfied
        self.assertFalse(bool(mask[0, 0, 1]))   # not required by scenario 0
        self.assertTrue(bool(mask[0, 1, 1]))    # required and unresolved


class TestRuleEngine(unittest.TestCase):
    def setUp(self):
        self.bank = load_bank()
        self.cfg = tiny_config().logic
        self.engine = RuleEngine(self.bank, self.cfg)

    def test_three_way_thresholds(self):
        self.assertEqual(self.engine.decide_score(0.95), VIOLATION)
        self.assertEqual(self.engine.decide_score(0.80), VIOLATION)
        self.assertEqual(self.engine.decide_score(0.60), INSUFFICIENT)
        self.assertEqual(self.engine.decide_score(0.40), INSUFFICIENT)
        self.assertEqual(self.engine.decide_score(0.39), NO_VIOLATION)

    def test_one_decision_per_scenario(self):
        from apl.logic.evidence import StructuredDescription

        psi = torch.full((1, self.bank.num_predicates), 0.95)
        description = StructuredDescription(
            predicate_ids=[p.id for p in self.bank.predicates],
            scores=psi.clone(), satisfaction=psi,
        )
        gamma = compatibility_scores(psi, self.bank.scenario_predicate_matrix())
        out = self.engine.evaluate(description, gamma)
        self.assertEqual(len(out.decisions), len(self.bank))
        self.assertEqual(out.scene_label, VIOLATION)
        # the loop does not stop at the first positive violation
        self.assertEqual(len(out.violations), len(self.bank))

    def test_insufficient_evidence_is_distinct_from_no_violation(self):
        from apl.logic.evidence import StructuredDescription

        psi = torch.full((1, self.bank.num_predicates), 0.55)
        description = StructuredDescription(
            predicate_ids=[p.id for p in self.bank.predicates],
            scores=psi.clone(), satisfaction=psi,
        )
        gamma = compatibility_scores(psi, self.bank.scenario_predicate_matrix())
        out = self.engine.evaluate(description, gamma)
        self.assertEqual(out.scene_label, INSUFFICIENT)
        self.assertEqual(out.violations, [])

    def test_report_mentions_the_legal_reference(self):
        model = tiny_model()
        out = model.infer(torch.randn(1, 3, 32, 32))
        self.assertIn("reference:", out.engine.report())


if __name__ == "__main__":
    unittest.main()
