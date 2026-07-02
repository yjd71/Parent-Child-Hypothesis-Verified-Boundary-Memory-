import unittest
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import numpy as np
import torch

from config.mkcfg import Config
from SAM.SAM_refinement.sam_refine_visualizer import SamRefineVisualizer
from SAM.protoSAMprompt.train_pseudo_refiner import (
    _BaseSamPseudoLabelRefiner,
    Sam1PseudoLabelRefiner,
    Sam2PseudoLabelRefiner,
    build_sam_pseudo_label_refiner,
)

_data_module = types.ModuleType("data")
_data_module.prepare_dataloader = Mock()
_data_module.prepare_labeled_memory_dataloader = Mock()
with patch.dict(sys.modules, {"data": _data_module}):
    import engine.solver as solver_module

SemiSupervisedTrainer = solver_module.SemiSupervisedTrainer


class SAMRefineModeConfigTests(unittest.TestCase):
    @staticmethod
    def _config(mode):
        cfg = Config.__new__(Config)
        cfg.sam_refine_mode = mode
        cfg.svb_ablation_mode = "full"
        cfg.use_sv_ume = True
        cfg.use_ume_evidence_loss = True
        cfg.use_source_consistency_loss = True
        cfg.use_svb_weighted_unsup_loss = True
        cfg.others = {"use_svb_plr": True, "use_sv_ume": True}
        return cfg

    def test_off_mode_disables_all_sam_refinement(self):
        cfg = self._config(" OFF ")
        cfg._normalize_sam_refine_mode()
        self.assertEqual(cfg.sam_refine_mode, "off")
        self.assertFalse(cfg.use_svb_plr)
        self.assertFalse(cfg.use_sam_refine_unlabeled)
        self.assertFalse(cfg.use_sam_pseudo_refine)
        self.assertFalse(cfg.use_sv_ume)

    def test_legacy_mode_is_standalone_and_unweighted(self):
        cfg = self._config("legacy_auto")
        cfg._normalize_sam_refine_mode()
        self.assertFalse(cfg.use_svb_plr)
        self.assertTrue(cfg.use_sam_refine_unlabeled)
        self.assertTrue(cfg.use_sam_pseudo_refine)
        self.assertFalse(cfg.use_sv_ume)
        self.assertFalse(cfg.use_ume_evidence_loss)
        self.assertFalse(cfg.use_source_consistency_loss)
        self.assertFalse(cfg.use_svb_weighted_unsup_loss)
        self.assertEqual(cfg.others["sam_refine_mode"], "legacy_auto")
        self.assertFalse(cfg.others["use_svb_plr"])
        self.assertFalse(cfg.others["use_sv_ume"])

    def test_svb_mode_preserves_explicit_sv_ume_settings(self):
        cfg = self._config("svb")
        cfg._normalize_sam_refine_mode()
        self.assertTrue(cfg.use_svb_plr)
        self.assertTrue(cfg.use_sam_refine_unlabeled)
        self.assertFalse(cfg.use_sam_pseudo_refine)
        self.assertTrue(cfg.use_sv_ume)
        self.assertTrue(cfg.use_svb_weighted_unsup_loss)

    def test_invalid_mode_fails_fast(self):
        cfg = self._config("unknown")
        with self.assertRaises(ValueError):
            cfg._normalize_sam_refine_mode()


class LegacySAMBuilderTests(unittest.TestCase):
    def test_builder_selects_sam1_without_hidden_enable_switch(self):
        cfg = SimpleNamespace(sam_pseudo_backend="sam1", use_sam_pseudo_refine=False)
        with patch(
            "SAM.protoSAMprompt.train_pseudo_refiner.Sam1PseudoLabelRefiner",
            return_value=sentinel.sam1,
        ):
            wrapper = build_sam_pseudo_label_refiner(cfg, "cpu")
        self.assertIs(wrapper.refiner, sentinel.sam1)

    def test_builder_selects_sam2_without_hidden_enable_switch(self):
        cfg = SimpleNamespace(sam_pseudo_backend="sam2", use_sam_pseudo_refine=False)
        with patch(
            "SAM.protoSAMprompt.train_pseudo_refiner.Sam2PseudoLabelRefiner",
            return_value=sentinel.sam2,
        ):
            wrapper = build_sam_pseudo_label_refiner(cfg, "cpu")
        self.assertIs(wrapper.refiner, sentinel.sam2)


class LegacySAMSolverTests(unittest.TestCase):
    @staticmethod
    def _trainer(start=3, interval=2):
        trainer = SemiSupervisedTrainer.__new__(SemiSupervisedTrainer)
        trainer.config = SimpleNamespace(
            sam_refine_mode="legacy_auto",
            sam_pseudo_backend="sam2",
            sam_start_epoch=start,
            sam_refine_interval=interval,
        )
        trainer.device = torch.device("cpu")
        trainer.logger = None
        trainer.svb_plr = None
        trainer.legacy_sam_refiner = object()
        trainer._log_module_info = Mock()
        return trainer

    def test_legacy_schedule_reuses_start_and_interval(self):
        trainer = self._trainer(start=3, interval=2)
        self.assertFalse(trainer._legacy_sam_enabled_for_epoch(2))
        self.assertFalse(trainer._legacy_sam_enabled_for_epoch(3))
        self.assertTrue(trainer._legacy_sam_enabled_for_epoch(4))
        self.assertTrue(trainer._legacy_sam_enabled_for_epoch(6))

    def test_legacy_alignment_detaches_same_view_probability(self):
        trainer = self._trainer()
        pseudo = torch.rand(1, 1, 4, 4, requires_grad=True)
        aligned = trainer._align_legacy_pseudo_to_strong(pseudo, None)
        self.assertEqual(tuple(aligned.shape), (1, 1, 4, 4))
        self.assertFalse(aligned.requires_grad)

    def test_legacy_alignment_applies_weak_to_strong_geometry(self):
        trainer = self._trainer()
        pseudo = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        aligned = trainer._align_legacy_pseudo_to_strong(pseudo, {"hflip": True})
        expected = torch.tensor([[[[1.0, 0.0], [3.0, 2.0]]]])
        self.assertTrue(torch.equal(aligned, expected))

    def test_legacy_refiner_receives_weak_images_and_teacher_probabilities(self):
        trainer = self._trainer()
        fake_refiner = Mock(side_effect=lambda images, probs, **kwargs: probs + 0.1)
        trainer.legacy_sam_refiner = fake_refiner
        images = torch.rand(2, 3, 8, 8)
        teacher_prob = torch.rand(2, 1, 4, 4)
        refined = trainer._run_legacy_sam_refiner(images, teacher_prob, epoch=4, step=7)
        args, kwargs = fake_refiner.call_args
        self.assertEqual(tuple(args[0].shape), (2, 3, 8, 8))
        self.assertEqual(tuple(args[1].shape), (2, 1, 4, 4))
        self.assertEqual(kwargs, {"epoch": 4, "step": 7, "image_ids": None})
        self.assertTrue(torch.allclose(refined, teacher_prob + 0.1))

    def test_legacy_initialization_freezes_backend_and_excludes_svb(self):
        trainer = self._trainer()
        trainer.legacy_sam_refiner = None
        model = torch.nn.Linear(2, 2)
        inner = SimpleNamespace(sam=None, sam2_refiner=SimpleNamespace(model=model))
        wrapper = SimpleNamespace(refiner=inner)
        with patch.object(solver_module, "build_sam_pseudo_label_refiner", return_value=wrapper):
            trainer._init_legacy_sam_refiner()
        self.assertIs(trainer.legacy_sam_refiner, wrapper)
        self.assertIsNone(trainer.svb_plr)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        init_message = trainer._log_module_info.call_args.args[0]
        self.assertIn("[SAM-Legacy]", init_message)
        self.assertNotIn("[SVB-PLR]", init_message)

    def test_legacy_initialization_reuses_existing_frozen_refiner(self):
        trainer = self._trainer()
        existing = trainer.legacy_sam_refiner
        with patch.object(solver_module, "build_sam_pseudo_label_refiner") as builder:
            trainer._init_legacy_sam_refiner()
        builder.assert_not_called()
        self.assertIs(trainer.legacy_sam_refiner, existing)


class LegacySAMVisualizationTests(unittest.TestCase):
    @staticmethod
    def _backend_attrs(refiner):
        refiner.use_point = True
        refiner.use_box = True
        refiner.use_mask = True
        refiner.add_neg = True
        refiner.iters = 1
        refiner.margin = 0.0
        refiner.gamma = 4.0
        refiner.strength = 30
        refiner.threshold = 0.5

    def test_backend_refine_one_keeps_mask_only_compatibility(self):
        mask = np.ones((1, 4, 4), dtype=np.uint8)
        prompt_debug = {
            "boxes": torch.tensor([[0.0, 0.0, 3.0, 3.0]]),
            "point_coords": torch.tensor([[[1.0, 1.0]]]),
            "point_labels": torch.tensor([[1]]),
        }
        sam1 = Sam1PseudoLabelRefiner.__new__(Sam1PseudoLabelRefiner)
        self._backend_attrs(sam1)
        sam1.sam = sentinel.sam
        sam1.embedding_cache = None
        sam1.sam_refiner = Mock(return_value=(mask, None, prompt_debug))
        self.assertTrue(torch.equal(sam1._refine_one(None, mask[0]), torch.ones(4, 4)))

        sam2 = Sam2PseudoLabelRefiner.__new__(Sam2PseudoLabelRefiner)
        self._backend_attrs(sam2)
        sam2.sam2_refiner = Mock(return_value=(mask, None, [prompt_debug]))
        self.assertTrue(torch.equal(sam2._refine_one(None, mask[0]), torch.ones(4, 4)))

    def test_legacy_call_emits_visualization_with_actual_prompts(self):
        class DummyLegacyRefiner(_BaseSamPseudoLabelRefiner):
            def _refine_one(self, image_np, coarse_mask):
                return torch.ones_like(torch.as_tensor(coarse_mask)).float()

            def _refine_one_with_debug(self, image_np, coarse_mask):
                return self._refine_one(image_np, coarse_mask), {
                    "boxes": torch.tensor([[0.0, 0.0, 3.0, 3.0]]),
                    "point_coords": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
                    "point_labels": torch.tensor([1, 0]),
                }

        cfg = SimpleNamespace(
            sam_refine_mode="legacy_auto",
            vis_sam_refinement=False,
            sam_pseudo_threshold=0.5,
            sam_pseudo_fusion_alpha=0.5,
            sam_pseudo_iters=1,
            sam_pseudo_use_point=True,
            sam_pseudo_use_box=True,
            sam_pseudo_use_mask=True,
            sam_pseudo_add_neg=True,
            log_enable=False,
        )
        refiner = DummyLegacyRefiner(cfg, "cpu", None, "dummy", "dummy", "missing.pth")
        refiner.visualizer = Mock()
        output = refiner(
            torch.zeros(1, 3, 4, 4),
            torch.ones(1, 1, 4, 4),
            epoch=2,
            step=3,
            image_ids=["legacy-sample"],
        )
        self.assertEqual(tuple(output.shape), (1, 1, 4, 4))
        kwargs = refiner.visualizer.save.call_args.kwargs
        self.assertEqual(kwargs["image_ids"], ["legacy-sample"])
        prompt_pack = kwargs["sam_aux"]["prompt_pack"]
        self.assertEqual(tuple(prompt_pack["boxes"][0].shape), (1, 4))
        self.assertEqual(tuple(prompt_pack["pos_points"][0].shape), (1, 2))
        self.assertEqual(tuple(prompt_pack["neg_points"][0].shape), (1, 2))

    def test_prompt_panel_draws_points_and_boxes_and_has_clear_name(self):
        self.assertIn("prompt_points_boxes", SamRefineVisualizer.PANEL_NAMES)
        self.assertNotIn("points", SamRefineVisualizer.PANEL_NAMES)
        visualizer = SamRefineVisualizer(SimpleNamespace(vis_sam_refinement=True))
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        prompt_pack = {
            "boxes": [torch.tensor([[2.0, 2.0, 17.0, 17.0]])],
            "pos_points": [torch.tensor([[5.0, 5.0]])],
            "neg_points": [torch.empty(0, 2)],
            "boundary_points": [torch.empty(0, 2)],
        }
        panel = visualizer._prompt_points_boxes_panel(
            image,
            prompt_pack,
            idx=0,
            ref=torch.zeros(1, 1, 20, 20),
        )
        self.assertTrue(np.array_equal(panel[2, 2], np.array([0, 200, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(panel[5, 3], np.array([0, 255, 0], dtype=np.uint8)))

    def test_visualizer_writes_legacy_compatible_grid(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            visualizer = SamRefineVisualizer(
                SimpleNamespace(
                    vis_sam_refinement=True,
                    vis_sam_refine_interval=1,
                    vis_sam_refine_max_samples=1,
                    sam_refine_vis_dir=temp_dir,
                )
            )
            ref = torch.ones(1, 1, 8, 8)
            visualizer.save(
                images=torch.zeros(1, 3, 8, 8),
                teacher_prob=ref,
                sam_mask=ref,
                p_ref=ref,
                conf_ref=None,
                sam_aux={
                    "prompt_pack": {
                        "boxes": [torch.tensor([[1.0, 1.0, 6.0, 6.0]])],
                        "pos_points": [torch.tensor([[2.0, 2.0]])],
                        "neg_points": [torch.tensor([[5.0, 5.0]])],
                        "boundary_points": [torch.empty(0, 2)],
                    }
                },
                image_ids=["legacy-sample"],
                epoch=1,
                step=0,
            )
            output = Path(temp_dir) / "epoch_001" / "iter_000000" / "legacy-sample.png"
            self.assertTrue(output.is_file())

    def test_legacy_prompt_debug_scales_and_splits_point_labels(self):
        debug = {
            "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            "point_coords": torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
            "point_labels": torch.tensor([1, 0]),
        }
        normalized = _BaseSamPseudoLabelRefiner._normalize_prompt_debug(
            debug,
            source_hw=(100, 200),
            target_hw=(50, 100),
        )
        self.assertTrue(torch.equal(normalized["pos_points"], torch.tensor([[5.0, 10.0]])))
        self.assertTrue(torch.equal(normalized["neg_points"], torch.tensor([[15.0, 20.0]])))
        self.assertTrue(torch.equal(normalized["boxes"], torch.tensor([[5.0, 10.0, 15.0, 20.0]])))

    def test_legacy_visualization_receives_prompt_pack_and_image_ids(self):
        refiner = _BaseSamPseudoLabelRefiner.__new__(_BaseSamPseudoLabelRefiner)
        refiner.visualizer = Mock()
        refiner._save_visualization(
            images=sentinel.images,
            teacher_prob=sentinel.teacher,
            sam_mask=sentinel.sam_mask,
            p_ref=sentinel.p_ref,
            prompt_debug=[
                {
                    "pos_points": torch.tensor([[1.0, 2.0]]),
                    "neg_points": torch.tensor([[3.0, 4.0]]),
                    "boxes": torch.tensor([[0.0, 0.0, 5.0, 6.0]]),
                }
            ],
            image_ids=["sample-a"],
            epoch=4,
            step=8,
        )
        kwargs = refiner.visualizer.save.call_args.kwargs
        self.assertEqual(kwargs["image_ids"], ["sample-a"])
        self.assertTrue(torch.equal(kwargs["sam_aux"]["prompt_pack"]["boxes"][0], torch.tensor([[0.0, 0.0, 5.0, 6.0]])))


if __name__ == "__main__":
    unittest.main()
