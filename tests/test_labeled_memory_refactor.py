import os
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from CBM.config.labeled_memory import resolve_labeled_memory_profile
from CBM.diagnostics.visualization import save_memory_selection_visualizations
from CBM.memory.bank import DenseBoundaryMemory
from CBM.memory.builder import LabeledMemoryBuilder


REGIONS = ("fg_core", "fg_boundary", "bg_near", "bg_far")


def make_config(split=0.05, overrides=None, profile="auto"):
    return SimpleNamespace(
        cbm_labeled_memory_profile=profile,
        cbm_labeled_memory_profile_overrides=overrides or {},
        cbm_labeled_split=split,
        cbm_memory_vis_enable=False,
        cbm_memory_vis_max_images=2,
        cbm_memory_vis_seed=7,
        ckpt_dir=".",
    )


def compact_profile(split=0.05, cap=8, sample=32, **changes):
    profile = resolve_labeled_memory_profile(make_config(split))
    values = {
        "max_sizes": {region: cap for region in REGIONS},
        "sample_per_image": {region: sample for region in REGIONS},
    }
    values.update(changes)
    return replace(profile, **values)


class FakeTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def extract_cbm_memory_features(self, inputs, ema=True):
        del ema
        p3 = torch.cat([inputs, inputs[:, :1]], dim=1) * self.scale
        return {"x3": p3, "p3": p3}


class LabeledMemoryProfileTests(unittest.TestCase):
    def test_all_profiles_match_expected_values(self):
        expected = {
            0.01: ((256, 640, 640, 256), 8, 0.98),
            0.05: ((128, 384, 384, 128), 32, 0.98),
            0.10: ((128, 384, 384, 128), 32, 0.985),
            0.20: ((96, 256, 256, 96), 32, 0.99),
        }
        for split, (samples, top_k, feature_sim) in expected.items():
            profile = resolve_labeled_memory_profile(make_config(split))
            self.assertEqual(tuple(profile.sample_per_image[name] for name in REGIONS), samples)
            self.assertEqual(profile.top_img_k, top_k)
            self.assertAlmostEqual(profile.max_feature_sim, feature_sim)
            self.assertEqual(
                profile.max_sizes,
                {"fg_core": 8192, "fg_boundary": 16384, "bg_near": 16384, "bg_far": 8192},
            )

    def test_20p_performance_profile(self):
        profile = resolve_labeled_memory_profile(make_config(0.20, profile="20p_performance"))
        self.assertEqual(profile.top_img_k, 48)
        self.assertEqual(profile.max_sizes["fg_boundary"], 24576)
        self.assertEqual(profile.sample_per_image["bg_near"], 384)

    def test_auto_rejects_unsupported_split(self):
        with self.assertRaises(ValueError):
            resolve_labeled_memory_profile(make_config(0.15))


class DenseBoundaryMemorySelectionTests(unittest.TestCase):
    @staticmethod
    def _sample(image_index):
        generator = torch.Generator().manual_seed(100 + image_index)
        x3 = torch.randn(1, 4, 8, 8, generator=generator)
        p3 = torch.randn(1, 4, 8, 8, generator=generator)
        gt = torch.zeros(1, 1, 8, 8)
        gt[:, :, 1:4, 1:4] = 1
        gt[:, :, 5:7, 5:7] = 1
        return x3, p3, gt

    def _build(self, order):
        profile = compact_profile(
            cap=8,
            sample=32,
            use_feature_diversity=False,
            use_spatial_diversity=False,
            use_grid_quota=False,
        )
        memory = DenseBoundaryMemory(mem_dim=4, selection_config=profile)
        for image_index in order:
            x3, p3, gt = self._sample(image_index)
            memory.append_batch(x3, p3, gt, [f"image_{image_index}"])
        memory.finalize()
        return memory

    def test_image_balancing_is_order_independent(self):
        first = self._build([0, 1, 2, 3])
        second = self._build([3, 1, 0, 2])
        for region in REGIONS:
            first_ids = [item["uid"] for item in first.meta[region]]
            second_ids = [item["uid"] for item in second.meta[region]]
            self.assertEqual(first_ids, second_ids)
            self.assertLessEqual(len(first_ids), 8)
            self.assertEqual(len(first_ids), len(set(first_ids)))
            self.assertEqual(first.keys[region].size(0), first.values[region].size(0))
            self.assertEqual(first.keys[region].size(0), len(first.meta[region]))
        self.assertEqual(first.image_ids, sorted(first.image_ids))

    def test_components_and_grids_are_recorded(self):
        profile = compact_profile(cap=16, sample=64, use_feature_diversity=False)
        memory = DenseBoundaryMemory(mem_dim=4, selection_config=profile)
        x3, p3, gt = self._sample(0)
        memory.append_batch(x3, p3, gt, ["two_components"])
        memory.finalize()
        fg_meta = memory.meta["fg_core"] + memory.meta["fg_boundary"]
        self.assertTrue(fg_meta)
        self.assertTrue(all("component_id" in item and "grid_id" in item for item in fg_meta))
        self.assertGreaterEqual(len({item["component_id"] for item in fg_meta}), 2)
        self.assertGreaterEqual(len({item["grid_id"] for item in fg_meta}), 2)

    def test_checkpoint_roundtrip_and_legacy_meta(self):
        memory = self._build([0, 1])
        state = memory.to_state_dict()
        self.assertEqual(state["format_version"], 2)
        restored = DenseBoundaryMemory(mem_dim=4, selection_config=memory.selection_config)
        restored.load_state_dict(state)
        self.assertTrue(restored.is_ready())
        self.assertEqual(restored.image_ids, memory.image_ids)
        self.assertEqual(restored.meta["fg_boundary"], memory.meta["fg_boundary"])

        legacy = dict(state)
        legacy.pop("format_version")
        legacy.pop("build_info")
        legacy["meta"] = {
            region: [{"image_id": item["image_id"]} for item in state["meta"][region]]
            for region in REGIONS
        }
        restored.load_state_dict(legacy)
        keys, values, _ = restored.get_sub_memory([restored.image_ids[0]])
        self.assertEqual(keys.size(0), values.size(0))

    def test_logs_include_required_fields(self):
        memory = self._build([0, 1, 2])
        distribution = "\n".join(memory.distribution_log_lines())
        diversity = "\n".join(memory.diversity_log_lines())
        for field in ("total=", "unique_images=", "min=", "max=", "mean=", "top10_img_token_counts="):
            self.assertIn(field, distribution)
        for field in ("avg_used_components=", "avg_used_grids=", "avg_max_grid_ratio=", "avg_pairwise_feat_sim="):
            self.assertIn(field, diversity)

    def test_feature_nms_can_underfill_without_duplicates(self):
        profile = compact_profile(
            cap=64,
            sample=64,
            use_spatial_diversity=False,
            max_feature_sim=0.98,
            relaxed_max_feature_sim=0.995,
            allow_underfill=True,
        )
        memory = DenseBoundaryMemory(mem_dim=4, selection_config=profile)
        x3 = torch.ones(1, 4, 8, 8)
        p3 = torch.ones(1, 4, 8, 8)
        gt = torch.zeros(1, 1, 8, 8)
        gt[:, :, 2:6, 2:6] = 1
        memory.append_batch(x3, p3, gt, ["redundant"])
        candidate_count = sum(chunk.size for chunk in memory.candidate_pool["bg_far"])
        memory.finalize()
        selected = memory.meta["bg_far"]
        self.assertLess(len(selected), candidate_count)
        self.assertEqual(len(selected), len({item["uid"] for item in selected}))


class LabeledMemoryBuilderTests(unittest.TestCase):
    def test_builder_restores_training_state_and_visits_all_images(self):
        profile = compact_profile(cap=16, sample=32, use_feature_diversity=False)
        memory = DenseBoundaryMemory(mem_dim=4, selection_config=profile)
        config = make_config(0.05)
        model = FakeTeacher().train()
        inputs = torch.randn(3, 3, 8, 8)
        gt = torch.zeros(3, 1, 8, 8)
        gt[:, :, 2:6, 2:6] = 1
        loader = [(inputs, gt, ["a", "b", "c"], torch.arange(3))]
        builder = LabeledMemoryBuilder(memory, config=config)
        builder.prepare_epoch(model, loader, epoch=2)
        self.assertTrue(model.training)
        self.assertTrue(memory.is_ready())
        self.assertEqual(memory.image_ids, ["a", "b", "c"])

    def test_visualization_creates_seven_panel_image(self):
        memory = DenseBoundaryMemory(mem_dim=4, selection_config=compact_profile(cap=8, sample=16))
        x3 = torch.randn(1, 4, 8, 8)
        p3 = torch.randn(1, 4, 8, 8)
        gt = torch.zeros(1, 1, 8, 8)
        gt[:, :, 2:6, 2:6] = 1
        memory.append_batch(x3, p3, gt, ["visual"])
        memory.finalize()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(0.05)
            config.cbm_memory_vis_dir = temp_dir
            snapshots = {"visual": (torch.zeros(3, 8, 8), gt[0])}
            paths = save_memory_selection_visualizations(memory, snapshots, 1, 0.05, config)
            self.assertEqual(len(paths), 1)
            self.assertTrue(os.path.isfile(paths[0]))
            from PIL import Image

            with Image.open(paths[0]) as image:
                self.assertEqual(image.width, 8 * 7)

    def test_non_main_process_is_detected(self):
        builder = LabeledMemoryBuilder(None, config=make_config())
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.get_rank", return_value=1):
            self.assertFalse(builder._is_main_process())


if __name__ == "__main__":
    unittest.main()
