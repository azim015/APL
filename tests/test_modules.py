"""Unit tests for the neural components (Sec. V-B)."""

from __future__ import annotations

import unittest

import torch

from helpers import tiny_config, tiny_model
from apl.config import FeatureGeneratorConfig
from apl.modules import (
    FeatureGenerator,
    FineGrainedBranch,
    GlobalInvariantBranch,
    InstanceDependentBranch,
    apl_loss,
    ensemble_probability,
)


class TestFeatureGenerator(unittest.TestCase):
    def setUp(self):
        self.cfg = FeatureGeneratorConfig(dim=16, key_dim=16, mid_dim=8)
        self.fg = FeatureGenerator(self.cfg)

    def test_shapes_unbatched(self):
        w = torch.randn(5, 16)
        alpha = torch.randn(3, 16)
        self.assertEqual(self.fg(w, alpha).shape, (5, 16))

    def test_shapes_batched(self):
        w = torch.randn(5, 16)
        alpha = torch.randn(4, 1, 16)        # one conditioning row per image
        self.assertEqual(self.fg(w, alpha).shape, (4, 5, 16))

    def test_identity_at_initialisation(self):
        """Zero-initialised output projections make theta_FG start as identity."""
        w = torch.randn(5, 16)
        alpha = torch.randn(2, 16)
        self.assertTrue(torch.allclose(self.fg(w, alpha), w, atol=1e-6))

    def test_conditioning_changes_output(self):
        torch.nn.init.normal_(self.fg.theta_out.weight, std=0.3)
        w = torch.randn(5, 16)
        a1, a2 = torch.randn(2, 16), torch.randn(2, 16)
        self.assertFalse(torch.allclose(self.fg(w, a1), self.fg(w, a2)))

    def test_gradients_flow(self):
        w = torch.randn(5, 16, requires_grad=True)
        alpha = torch.randn(2, 16)
        self.fg(w, alpha).sum().backward()
        self.assertIsNotNone(self.fg.theta_in1.weight.grad)

    def test_multi_head(self):
        cfg = FeatureGeneratorConfig(dim=16, key_dim=16, mid_dim=8, num_heads=4)
        fg = FeatureGenerator(cfg)
        self.assertEqual(fg(torch.randn(3, 16), torch.randn(2, 16)).shape, (3, 16))

    def test_batch_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.fg(torch.randn(2, 5, 16), torch.randn(3, 2, 16))


class TestBranches(unittest.TestCase):
    def setUp(self):
        self.fg = FeatureGenerator(
            FeatureGeneratorConfig(dim=16, key_dim=16, mid_dim=8)
        )
        self.w = torch.randn(6, 16)
        self.f_hat = torch.randn(4, 16)
        self.local = torch.randn(4, 20, 16)

    def test_global_invariant(self):
        branch = GlobalInvariantBranch(self.fg, 16)
        w_gi = branch(self.w)
        self.assertEqual(w_gi.shape, (6, 16))
        logits = GlobalInvariantBranch.logits(w_gi, self.f_hat, 0.07)
        self.assertEqual(logits.shape, (4, 6))

    def test_instance_dependent_is_image_specific(self):
        branch = InstanceDependentBranch(self.fg)
        torch.nn.init.normal_(self.fg.theta_out.weight, std=0.3)
        h_id = branch(self.w, self.f_hat)
        self.assertEqual(h_id.shape, (4, 6, 16))
        self.assertFalse(torch.allclose(h_id[0], h_id[1]))

    def test_fine_grained_shapes_and_topk(self):
        branch = FineGrainedBranch(self.fg, 16, num_tokens=4, eta=1.0)
        h_fg = branch(self.w)
        self.assertEqual(h_fg.shape, (4, 6, 16))
        sim = branch.similarity(h_fg, self.local)
        self.assertEqual(sim.shape, (4, 4, 6))
        self.assertTrue(bool((sim.abs() <= 1.0 + 1e-5).all()))

    def test_topk_ladder_is_monotonic(self):
        branch = FineGrainedBranch(self.fg, 16, num_tokens=4, eta=1.0)
        ks = [branch.top_k(z, 196) for z in range(4)]
        self.assertEqual(ks, sorted(ks))
        self.assertEqual(ks[-1], 196)
        counts = FineGrainedBranch(self.fg, 16, num_tokens=4, eta=1.0,
                                   topk_mode="count")
        self.assertEqual([counts.top_k(z, 196) for z in range(4)], [1, 2, 3, 4])

    def test_eta_controls_scale(self):
        narrow = FineGrainedBranch(self.fg, 16, num_tokens=4, eta=0.25)
        wide = FineGrainedBranch(self.fg, 16, num_tokens=4, eta=1.0)
        self.assertLess(narrow.top_k(3, 196), wide.top_k(3, 196))


class TestLosses(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_config().loss
        self.targets = torch.tensor([0, 2])
        self.gi = torch.randn(2, 5)
        self.id_ = torch.randn(2, 5)
        self.fg = torch.randn(2, 4, 5)

    def test_total_is_weighted_sum(self):
        out = apl_loss(self.gi, self.id_, self.fg, self.targets, self.cfg)
        expected = sum(out.terms[k] for k in ("gi", "id", "fg"))
        self.assertAlmostEqual(float(out.total), float(expected), places=5)

    def test_regularisers_increase_the_gi_term(self):
        w_gi = torch.randn(5, 8)
        w_zs = torch.randn(5, 8)
        f_hat = torch.randn(2, 8)
        f_zs = torch.randn(2, 8)
        plain = apl_loss(self.gi, None, None, self.targets, self.cfg)
        regularised = apl_loss(
            self.gi, None, None, self.targets, self.cfg,
            w_gi=w_gi, w_zs=w_zs, f_hat=f_hat, f_hat_zs=f_zs,
        )
        self.assertGreater(float(regularised.total), float(plain.total))

    def test_ensemble_probability_is_a_distribution(self):
        probs = ensemble_probability(self.gi, self.id_, self.fg)
        self.assertEqual(probs.shape, (2, 5))
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-5))

    def test_single_branch_ensemble(self):
        probs = ensemble_probability(None, self.id_, None)
        self.assertTrue(torch.allclose(probs, self.id_.softmax(-1), atol=1e-6))

    def test_no_branch_raises(self):
        with self.assertRaises(ValueError):
            apl_loss(None, None, None, self.targets, self.cfg)


class TestPromptLearner(unittest.TestCase):
    def test_base_feature_shape_and_retargeting(self):
        model = tiny_model()
        w = model.base_text_feature()
        self.assertEqual(w.shape, (len(model.categories), 64))
        model.set_categories(["red light", "stop sign"])
        self.assertEqual(model.base_text_feature().shape, (2, 64))
        self.assertFalse(model.logic_enabled)

    def test_prompt_tokens_are_trainable_and_encoders_frozen(self):
        model = tiny_model()
        self.assertTrue(model.prompt_learner.text_ctx.requires_grad)
        self.assertTrue(model.prompt_learner.visual_ctx.requires_grad)
        self.assertTrue(
            all(not p.requires_grad for p in model.backbone.parameters())
        )


if __name__ == "__main__":
    unittest.main()
