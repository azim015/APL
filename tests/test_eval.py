"""Tests for the evaluation suite and the privacy filter."""

from __future__ import annotations

import unittest

import torch

from helpers import tiny_config
from apl.describe import detect_pii, sanitize
from apl.eval import (
    bleu,
    cider,
    membership_inference_attack,
    mse,
    privacy_score,
    psnr,
    replay_attack_success_rate,
    rouge_l,
    scenario_prf,
    semantic_utility,
    ssim,
    text_quality,
    violation_accuracy,
)
from apl.eval.utility import format_table


class TestPrivacyFilter(unittest.TestCase):
    def test_license_plate_is_redacted(self):
        report = sanitize("The car with license plate ABC1234 runs the red light.")
        self.assertNotIn("ABC1234", report.text)
        self.assertFalse(report.clean)

    def test_face_and_identity_are_redacted(self):
        report = sanitize("The driver's face is visible; his name is on the door.")
        self.assertNotIn("face", report.text.lower())
        self.assertGreater(report.num_removed, 0)

    def test_contact_details_are_redacted(self):
        report = sanitize("Call 555-123-4567 or driver@example.com for details.")
        self.assertNotIn("555-123-4567", report.text)
        self.assertNotIn("driver@example.com", report.text)

    def test_clean_description_is_untouched(self):
        text = ("The scene shows that the traffic signal facing the vehicle "
                "shows a steady red light.")
        report = sanitize(text)
        self.assertEqual(report.text, text)
        self.assertTrue(report.clean)

    def test_detect_pii_reports_categories(self):
        found = detect_pii("The elderly man's face is clearly visible.")
        self.assertTrue(found)


class TestPrivacyMetrics(unittest.TestCase):
    def test_privacy_score_bounds(self):
        clean = ["A vehicle crosses the stop line while the signal is red."]
        leaky = ["The driver's face and the license plate ABC1234 are visible."]
        attributes = [["driver_face", "license_plate"]]
        self.assertEqual(privacy_score(clean, attributes), 100.0)
        self.assertEqual(privacy_score(leaky, attributes), 0.0)

    def test_privacy_score_partial(self):
        texts = ["The license plate is readable but nobody is identifiable."]
        score = privacy_score(texts, [["driver_face", "license_plate"]])
        self.assertAlmostEqual(score, 50.0, places=5)

    def test_reconstruction_metrics_on_identical_images(self):
        x = torch.rand(2, 3, 16, 16)
        self.assertAlmostEqual(mse(x, x), 0.0, places=5)
        self.assertGreater(psnr(x, x), 80.0)
        self.assertAlmostEqual(ssim(x, x), 1.0, places=3)

    def test_reconstruction_metrics_degrade_with_noise(self):
        x = torch.rand(2, 3, 16, 16)
        noisy = (x + 0.5 * torch.randn_like(x)).clamp(0, 1)
        self.assertGreater(mse(x, noisy), 0.0)
        self.assertLess(ssim(x, noisy), 1.0)
        self.assertLess(psnr(x, noisy), 80.0)

    def test_replay_attack_success_rate(self):
        gallery = torch.eye(4)
        self.assertAlmostEqual(
            replay_attack_success_rate(gallery, gallery, 0.5), 100.0, places=4
        )
        shuffled = gallery[[1, 0, 3, 2]]
        self.assertAlmostEqual(
            replay_attack_success_rate(shuffled, gallery, 0.5), 0.0, places=4
        )

    def test_membership_inference_separates_members(self):
        member = torch.full((50,), 0.95)
        nonmember = torch.full((50,), 0.05)
        result = membership_inference_attack(member, nonmember)
        self.assertGreater(result.accuracy, 90.0)
        self.assertGreater(result.avg_risk, 0.5)

    def test_membership_inference_on_indistinguishable_signals(self):
        signal = torch.rand(100)
        result = membership_inference_attack(signal, signal.clone())
        self.assertLess(abs(result.accuracy - 50.0), 15.0)


class TestUtilityMetrics(unittest.TestCase):
    def test_semantic_utility(self):
        texts = ["the traffic signal shows a steady red light"]
        self.assertEqual(
            semantic_utility(texts, [["the traffic signal shows a steady red light"]]),
            100.0,
        )
        self.assertEqual(semantic_utility(texts, [["a pedestrian crosses"]]), 0.0)

    def test_violation_accuracy(self):
        self.assertEqual(violation_accuracy(["V", "N", "U"], ["V", "N", "N"]),
                         200.0 / 3)

    def test_scenario_prf(self):
        report = scenario_prf([["a", "b"], ["c"]], [["a"], ["c", "d"]])
        self.assertAlmostEqual(report["micro_precision"], 100.0 * 2 / 3, places=4)
        self.assertAlmostEqual(report["micro_recall"], 100.0 * 2 / 3, places=4)
        self.assertAlmostEqual(report["exact_set_accuracy"], 0.0, places=4)

    def test_format_table(self):
        table = format_table([{"a": 1, "b": "x"}, {"a": 22, "b": "yy"}])
        self.assertIn("a", table)
        self.assertEqual(len(table.splitlines()), 4)


class TestTextQuality(unittest.TestCase):
    def test_identical_text_scores_perfectly(self):
        text = ["a vehicle crosses the stop line while the signal is red"]
        self.assertAlmostEqual(bleu(text, [text]), 1.0, places=5)
        self.assertAlmostEqual(rouge_l(text, [text]), 1.0, places=5)

    def test_disjoint_text_scores_zero(self):
        self.assertEqual(
            bleu(["alpha beta gamma delta"], [["one two three four"]]), 0.0
        )
        self.assertEqual(
            rouge_l(["alpha beta gamma delta"], [["one two three four"]]), 0.0
        )

    def test_partial_overlap_is_between(self):
        candidate = ["a vehicle crosses the stop line"]
        reference = [["a vehicle crosses the painted stop line slowly"]]
        score = rouge_l(candidate, reference)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_cider_rewards_rare_shared_ngrams(self):
        candidates = ["a gate arm is lowered", "a vehicle stops"]
        references = [["a gate arm is lowered"], ["a cyclist waits"]]
        self.assertGreater(cider(candidates, references), 0.0)

    def test_report_contains_all_three_metrics(self):
        row = text_quality(["a red light"], ["a red light"]).as_row()
        self.assertEqual(set(row), {"BLEU", "ROUGE", "CIDEr"})


if __name__ == "__main__":
    unittest.main()
