"""Configuration handling, data interfaces and the training loop."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from helpers import bank as load_bank, tiny_config, tiny_model
from apl.config import APLConfig, parse_overrides
from apl.data import SyntheticTrafficDataset
from apl.data.synthetic import collate
from apl.eval import run_ablation
from apl.train import Trainer, build_scheduler, harmonic_mean, set_seed


class TestConfig(unittest.TestCase):
    def test_table_i_defaults(self):
        cfg = APLConfig()
        self.assertEqual(cfg.encoder.backbone, "ViT-B/16")
        self.assertEqual(cfg.encoder.embed_dim, 512)
        self.assertEqual(cfg.encoder.num_local_features, 196)
        self.assertEqual(cfg.feature_generator.mid_dim,
                         cfg.feature_generator.dim // 2)
        self.assertEqual(cfg.branches.num_fg_tokens, 4)
        self.assertEqual(cfg.prompt.n_text_tokens, 4)
        self.assertEqual(cfg.prompt.n_visual_tokens, 4)
        self.assertAlmostEqual(cfg.loss.temperature, 0.07)
        self.assertAlmostEqual(cfg.loss.lambda1, 0.1)
        self.assertAlmostEqual(cfg.logic.tau_init, 0.25)
        self.assertAlmostEqual(cfg.logic.kappa, 10.0)
        self.assertAlmostEqual(cfg.logic.gamma_loose, 0.40)
        self.assertAlmostEqual(cfg.logic.gamma_strict, 0.80)
        self.assertEqual(cfg.refinement.t_max, 5)
        self.assertAlmostEqual(cfg.refinement.epsilon, 1e-3)
        self.assertAlmostEqual(cfg.optim.lr, 2e-3)
        self.assertEqual(cfg.optim.epochs, 50)
        self.assertEqual(cfg.optim.batch_size, 32)
        self.assertEqual(cfg.optim.warmup_epochs, 5)

    def test_dotted_overrides(self):
        cfg = APLConfig()
        cfg.update({"optim.lr": 0.5, "branches.use_fine_grained": False})
        self.assertAlmostEqual(cfg.optim.lr, 0.5)
        self.assertFalse(cfg.branches.use_fine_grained)
        with self.assertRaises(KeyError):
            cfg.update({"optim.does_not_exist": 1})

    def test_cli_override_parsing(self):
        parsed = parse_overrides(["optim.lr=0.01", "branches.eta=0.5",
                                  "refinement.enabled=false"])
        self.assertEqual(parsed["optim.lr"], 0.01)
        self.assertEqual(parsed["branches.eta"], 0.5)
        self.assertIs(parsed["refinement.enabled"], False)

    def test_yaml_round_trip(self):
        cfg = tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.yaml"
            cfg.save(path)
            restored = APLConfig.load(path)
        self.assertEqual(restored.encoder.embed_dim, cfg.encoder.embed_dim)
        self.assertEqual(restored.feature_generator.dim, cfg.feature_generator.dim)

    def test_shipped_configs_load(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        for path in sorted(root.glob("*.yaml")):
            self.assertIsInstance(APLConfig.load(path), APLConfig)

    def test_clone_is_deep(self):
        cfg = APLConfig()
        clone = cfg.clone()
        clone.optim.lr = 1.0
        self.assertNotAlmostEqual(cfg.optim.lr, 1.0)


class TestData(unittest.TestCase):
    def test_synthetic_sample_fields(self):
        bank = load_bank()
        dataset = SyntheticTrafficDataset(
            bank.semantic_categories(), 8, 32, seed=0, bank=bank
        )
        sample = dataset[0]
        self.assertEqual(sample["image"].shape, (3, 32, 32))
        self.assertIn("sensitive_attributes", sample)
        self.assertIn("relevant_attributes", sample)
        self.assertEqual(sample["scenario_labels"].shape, (len(bank),))

    def test_synthetic_dataset_is_deterministic(self):
        bank = load_bank()
        categories = bank.semantic_categories()
        a = SyntheticTrafficDataset(categories, 4, 32, seed=7, bank=bank)[2]
        b = SyntheticTrafficDataset(categories, 4, 32, seed=7, bank=bank)[2]
        self.assertTrue(torch.equal(a["image"], b["image"]))
        self.assertEqual(a["label"], b["label"])

    def test_collate_keeps_variable_length_fields(self):
        bank = load_bank()
        dataset = SyntheticTrafficDataset(
            bank.semantic_categories(), 4, 32, seed=0, bank=bank
        )
        batch = collate([dataset[i] for i in range(4)])
        self.assertEqual(batch["image"].shape, (4, 3, 32, 32))
        self.assertEqual(batch["label"].shape, (4,))
        self.assertEqual(len(batch["sensitive_attributes"]), 4)

    def test_base_new_class_split(self):
        from apl.data.fewshot import FewShotImageFolder

        classes = [f"c{i}" for i in range(10)]
        base = FewShotImageFolder.subsample_classes(classes, "base")
        new = FewShotImageFolder.subsample_classes(classes, "new")
        self.assertEqual(len(base), 5)
        self.assertEqual(len(new), 5)
        self.assertEqual(set(base) & set(new), set())


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.cfg = tiny_config()
        self.bank = load_bank()
        self.categories = self.bank.semantic_categories()
        self.train = SyntheticTrafficDataset(
            self.categories, 16, 32, seed=0, bank=self.bank
        )
        self.test = SyntheticTrafficDataset(
            self.categories, 8, 32, seed=11, bank=self.bank
        )

    def test_training_reduces_the_loss(self):
        set_seed(1)
        model = tiny_model(cfg=self.cfg)
        trainer = Trainer(model, self.cfg, collate_fn=collate, log_fn=lambda _m: None)
        history = trainer.fit(self.train, epochs=4)
        self.assertEqual(len(history), 4)
        self.assertLess(history[-1].loss, history[0].loss)

    def test_only_trainable_parameters_change(self):
        set_seed(1)
        model = tiny_model(cfg=self.cfg)
        before = model.prompt_learner.text_ctx.detach().clone()
        backbone_before = next(iter(model.backbone.parameters())).detach().clone()
        Trainer(model, self.cfg, collate_fn=collate,
                log_fn=lambda _m: None).fit(self.train, epochs=2)
        self.assertFalse(torch.allclose(before, model.prompt_learner.text_ctx))
        self.assertTrue(torch.allclose(
            backbone_before, next(iter(model.backbone.parameters()))
        ))

    def test_evaluate_returns_a_percentage(self):
        model = tiny_model(cfg=self.cfg)
        trainer = Trainer(model, self.cfg, collate_fn=collate,
                          log_fn=lambda _m: None)
        accuracy = trainer.evaluate(self.test)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)
        branches = trainer.branch_accuracies(self.test)
        self.assertEqual(set(branches), {"gi", "id", "fg", "ensemble"})

    def test_warmup_then_cosine_schedule(self):
        cfg = tiny_config()
        cfg.optim.epochs = 20
        cfg.optim.warmup_epochs = 5
        model = tiny_model(cfg=cfg)
        optimiser = torch.optim.SGD(model.trainable_parameters(), lr=cfg.optim.lr)
        scheduler = build_scheduler(optimiser, cfg.optim)
        lrs = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(cfg.optim.epochs):
                lrs.append(optimiser.param_groups[0]["lr"])
                scheduler.step()
        self.assertLess(lrs[0], lrs[4])               # warm-up rises
        self.assertAlmostEqual(lrs[5], cfg.optim.lr, places=6)
        self.assertLess(lrs[-1], lrs[5])              # cosine decays
        self.assertGreaterEqual(lrs[-1], cfg.optim.min_lr * 0.9)

    def test_harmonic_mean(self):
        self.assertAlmostEqual(harmonic_mean(80.0, 70.0), 74.666666, places=4)
        self.assertEqual(harmonic_mean(0.0, 0.0), 0.0)

    def test_ablation_runs_every_variant(self):
        cfg = tiny_config()
        cfg.optim.epochs = 1
        samples = [dict(self.test[i], scene_label_symbol="N") for i in range(2)]
        rows = run_ablation(
            cfg, self.train, samples, self.bank, collate_fn=collate, epochs=1,
            log_fn=None,
        )
        self.assertEqual(len(rows), 9)
        for row in rows:
            self.assertGreaterEqual(row.privacy_score, 0.0)
            self.assertLessEqual(row.privacy_score, 100.0)


if __name__ == "__main__":
    unittest.main()
