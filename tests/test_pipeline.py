"""Evidence extraction, Algorithm 1 and the end-to-end pipeline."""

from __future__ import annotations

import unittest
import warnings

import torch

from helpers import bank as load_bank, images, tiny_config, tiny_model
from apl.logic import TextFeatureSet
from apl.refinement import run_refinement_loop


class TestEvidenceExtraction(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.images = images(2)

    def test_text_feature_set_shapes(self):
        f_hat, local = self.model.encode_image(self.images)
        features = self.model.text_features(f_hat)
        c = len(self.model.categories)
        self.assertEqual(features.w_gi.shape, (c, 64))
        self.assertEqual(features.h_id.shape, (2, c, 64))
        self.assertEqual(features.h_fg.shape, (4, c, 64))
        rows = features.stacked_rows()
        self.assertEqual(rows.shape[1], 64)
        self.assertEqual(rows.shape[0], c + 2 * c + 4 * c)

    def test_evidence_rows_align_with_the_predicate_inventory(self):
        f_hat, local = self.model.encode_image(self.images)
        features = self.model.text_features(f_hat)
        description = self.model.extractor.extract(features, f_hat, local)
        self.assertEqual(
            description.predicate_ids,
            [p.id for p in self.model.bank.predicates],
        )
        self.assertEqual(
            description.scores.shape, (2, self.model.bank.num_predicates)
        )
        self.assertTrue(bool(((description.satisfaction >= 0) &
                              (description.satisfaction <= 1)).all()))

    def test_compatibility_and_gaps(self):
        f_hat, local = self.model.encode_image(self.images)
        features = self.model.text_features(f_hat)
        description = self.model.extractor.extract(features, f_hat, local)
        gamma = self.model.extractor.compatibility(description)
        self.assertEqual(gamma.shape, (2, len(self.model.bank)))
        self.assertTrue(bool(((gamma >= 0) & (gamma <= 1)).all()))
        gaps = self.model.extractor.gaps(description)
        self.assertEqual(
            gaps.shape, (2, len(self.model.bank), self.model.bank.num_predicates)
        )
        # a gap can only be flagged for a predicate the scenario requires
        membership = self.model.extractor.membership.bool()
        self.assertTrue(bool((gaps <= membership.unsqueeze(0)).all()))

    def test_gap_set_pairs_scenarios_with_predicates(self):
        f_hat, local = self.model.encode_image(self.images)
        features = self.model.text_features(f_hat)
        description = self.model.extractor.extract(features, f_hat, local)
        pairs = self.model.extractor.gap_set(description, index=0)
        self.assertTrue(pairs)
        scenario_ids = {s.id for s in self.model.bank}
        predicate_ids = {p.id for p in self.model.bank.predicates}
        for scenario_id, predicate_id in pairs:
            self.assertIn(scenario_id, scenario_ids)
            self.assertIn(predicate_id, predicate_ids)


class TestRefinementLoop(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.images = images(1, seed=3)

    def _run(self, cfg=None):
        f_hat, local = self.model.encode_image(self.images)
        features = self.model.text_features(f_hat)
        return run_refinement_loop(
            self.model.extractor, self.model.refiner, features, f_hat, local,
            cfg or self.model.cfg.refinement,
        )

    def test_loop_respects_t_max(self):
        cfg = tiny_config().refinement
        cfg.epsilon = 0.0           # never converge early
        cfg.t_max = 3
        result = self._run(cfg)
        self.assertLessEqual(result.iterations, 3)
        self.assertLessEqual(len(result.trace), 4)

    def test_disabled_refinement_runs_a_single_pass(self):
        cfg = tiny_config().refinement
        cfg.enabled = False
        result = self._run(cfg)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.prompts, [])

    def test_stop_reason_is_reported(self):
        result = self._run()
        self.assertIn(result.stop_reason,
                      {"no_gaps", "converged", "max_iterations"})

    def test_generated_prompts_target_missing_predicates(self):
        f_hat, local = self.model.encode_image(self.images)
        features = self.model.text_features(f_hat)
        description = self.model.extractor.extract(features, f_hat, local)
        gaps = self.model.extractor.gaps(description)
        ranked = self.model.refiner.rank_gaps(description, gaps)
        self.assertTrue(ranked)
        self.assertLessEqual(len(ranked), self.model.cfg.refinement.max_gap_prompts)
        # ranked by how many scenarios each missing predicate blocks
        counts = [len(g.scenario_ids) for g in ranked]
        self.assertEqual(counts, sorted(counts, reverse=True))
        prompts = self.model.refiner.generate_prompts(description, gaps)
        self.assertTrue(all(isinstance(p, str) and p for p in prompts))

    def test_gap_conditioning_changes_the_text_features(self):
        f_hat, local = self.model.encode_image(self.images)
        features = self.model.text_features(f_hat)
        # a non-degenerate generator, so refinement actually moves the features
        torch.nn.init.normal_(self.model.feature_generator.theta_out.weight, std=0.3)
        alpha = self.model.refiner.gap_conditioning(["check the signal state"])
        refined = self.model.refiner.refine_features(features, alpha)
        self.assertEqual(refined.iteration, 1)
        self.assertFalse(torch.allclose(refined.w_gi, features.w_gi))
        self.assertEqual(refined.h_id.shape, features.h_id.shape)
        self.assertEqual(refined.h_fg.shape, features.h_fg.shape)

    def test_empty_gap_set_returns_the_learnable_tokens(self):
        alpha = self.model.refiner.gap_conditioning([])
        self.assertTrue(torch.equal(alpha, self.model.refiner.gap_tokens))


class TestEndToEnd(unittest.TestCase):
    def test_forward_and_loss(self):
        model = tiny_model()
        out = model(images(3), torch.tensor([0, 1, 2]))
        c = len(model.categories)
        self.assertEqual(out.logits_gi.shape, (3, c))
        self.assertEqual(out.logits_id.shape, (3, c))
        self.assertEqual(out.logits_fg.shape, (3, 4, c))
        self.assertEqual(out.probs.shape, (3, c))
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertIsNotNone(model.prompt_learner.text_ctx.grad)

    def test_backbone_receives_no_gradient(self):
        model = tiny_model()
        out = model(images(2), torch.tensor([0, 1]))
        out.loss.backward()
        for param in model.backbone.parameters():
            self.assertIsNone(param.grad)

    def test_infer_returns_description_and_decisions(self):
        model = tiny_model()
        out = model.infer(images(1))
        self.assertTrue(out.description.text)
        self.assertEqual(len(out.engine.decisions), len(model.bank))
        self.assertIn(out.engine.scene_label, {"V", "U", "N"})
        payload = out.to_dict()
        self.assertIn("description", payload)
        self.assertIn("scenarios", payload)

    def test_single_branch_models_run(self):
        for key in ("use_global_invariant", "use_instance_dependent",
                    "use_fine_grained"):
            overrides = {
                f"branches.{k}": (k == key)
                for k in ("use_global_invariant", "use_instance_dependent",
                          "use_fine_grained")
            }
            model = tiny_model(**overrides)
            out = model(images(2), torch.tensor([0, 1]))
            self.assertTrue(torch.isfinite(out.loss))
            self.assertTrue(model.infer(images(1)).description.text)

    def test_no_branch_raises(self):
        with self.assertRaises(ValueError):
            tiny_model(**{
                "branches.use_global_invariant": False,
                "branches.use_instance_dependent": False,
                "branches.use_fine_grained": False,
            })

    def test_checkpoint_round_trip(self):
        import tempfile
        from pathlib import Path

        model = tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "apl.pt")
            model.save(path)
            restored = tiny_model(cfg=model.cfg)
            restored.load(path)
            before = model.prompt_learner.text_ctx
            after = restored.prompt_learner.text_ctx
            self.assertTrue(torch.allclose(before, after))

    def test_checkpoint_excludes_the_frozen_backbone(self):
        model = tiny_model()
        state = {k: v for k, v in model.state_dict().items()
                 if not k.startswith("backbone.")}
        self.assertTrue(all(not k.startswith("backbone.") for k in state))
        self.assertLess(len(state), len(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
