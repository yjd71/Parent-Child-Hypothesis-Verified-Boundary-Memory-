import math
import runpy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from CBM.memory.labels import REGION_NAMES
from CBM.sv_ume.quality_adaptive_fusion import QualityAdaptiveSourceFusion
from CBM.sv_ume.sam_refined_candidate_builder import SAMRefinedCandidateBuilder
from CBM.sv_ume.schedules import (
    can_use_lagged_memory,
    expected_unlabeled_source_epoch,
    should_build_after_epoch,
)
from CBM.sv_ume.sv_ume_manager import SVUMEManager
from CBM.sv_ume.ume_reliability import (
    TOKEN_FACTOR_NAMES,
    TOKEN_WEIGHTED_SUM_WEIGHTS,
    combine_token_reliability,
    compute_token_reliability,
)


def _safe_config(**overrides):
    values = dict(
        use_sv_ume=True,
        sv_ume_require_svb_plr=True,
        use_svb_plr=True,
        use_lagged_unlabeled_memory=True,
        build_unlabeled_memory_after_epoch=True,
        use_unlabeled_memory_during_current_epoch=False,
        rebuild_labeled_memory_each_epoch=True,
        do_not_update_labeled_memory_with_unlabeled=True,
        unlabeled_memory_source="sam_refined_pseudo_label",
        unlabeled_memory_feature_source="teacher_p3",
        use_sam_embedding_as_memory_key=False,
        retrieve_labeled_and_unlabeled_separately=True,
        use_aux_evidence_fusion=True,
        use_aux_feature_fusion=False,
        aux_fusion_mode="quality_adaptive_symmetric",
        gamma_max_final=0.25,
        use_aux_source_penalty=True,
        aux_source_penalty_value=0.25,
        allow_aux_dominate=False,
        use_fixed_matched_novel_ratio=False,
        sv_ume_start_epoch=29,
        unlabeled_to_labeled_ratio=1.0,
        region_capacity_ratio={region: 1.0 for region in REGION_NAMES},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _CandidateDependency:
    def build_batch(self, **kwargs):
        return {region: [] for region in REGION_NAMES}


class _MemoryDependency:
    def build_memory(self, **kwargs):
        return None

    def freeze_memory(self, memory):
        return memory

    def memory_state_dict(self, memory):
        return {}

    def load_memory_state_dict(self, state):
        return None


class _ReadyMemory:
    def is_ready(self):
        return True


class TokenReliabilityTests(unittest.TestCase):
    def test_three_score_modes_and_invalid_mode(self):
        values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        components = {
            name: torch.tensor([value], dtype=torch.float32)
            for name, value in zip(TOKEN_FACTOR_NAMES, values)
        }
        product = combine_token_reliability(components, "product")
        geometric = combine_token_reliability(components, "geometric_mean")
        weighted = combine_token_reliability(components, "weighted_sum")
        self.assertAlmostEqual(product.item(), math.prod(values), places=6)
        self.assertAlmostEqual(
            geometric.item(), math.prod(values) ** (1.0 / len(values)), places=6
        )
        self.assertAlmostEqual(
            weighted.item(),
            sum(weight * value for weight, value in zip(TOKEN_WEIGHTED_SUM_WEIGHTS, values)),
            places=6,
        )
        self.assertGreater(weighted.item(), product.item())
        with self.assertRaises(ValueError):
            combine_token_reliability(components, "unknown")

    def test_soft_boundary_context_survives_invalid_cbm_pixel(self):
        one = torch.ones(1, 1, 2, 2)
        regions = {region: torch.zeros_like(one) for region in REGION_NAMES}
        regions["fg_core"][0, 0, 0, 0] = 1.0
        regions["fg_boundary"][0, 0, 0, 1] = 1.0
        regions["bg_near"][0, 0, 1, 0] = 1.0
        regions["bg_far"][0, 0, 1, 1] = 1.0
        region_pack = {
            "p_ref3": one * 0.9,
            "conf_ref3": one,
            "valid": one,
            "regions": regions,
        }
        valid_map = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        retrieval = {
            "Y_ctx": torch.full((1, 4, 2, 2), 0.25),
            "U_map": torch.zeros_like(one),
            "cons_map": torch.zeros_like(one),
            "B3": torch.zeros_like(one),
            "gate3": one,
            "valid_map": valid_map,
            "prob3": one * 0.9,
        }
        memory = SimpleNamespace(
            keys={region: torch.ones(1, 2) for region in REGION_NAMES}
        )
        result = compute_token_reliability(
            one * 0.9,
            region_pack,
            torch.ones(1, 2, 2, 2),
            retrieval,
            memory,
            thresholds={region: 0.0 for region in REGION_NAMES},
            score_mode="weighted_sum",
        )
        boundary = (0, 0, 0, 1)
        self.assertAlmostEqual(result["components"]["r_context"][boundary].item(), 0.3, places=6)
        self.assertGreater(result["score"][boundary].item(), 0.0)
        self.assertFalse(result["cbm_valid"][boundary].item())


class CandidateDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _mock_outputs(cbm_valid=True, score=0.8, empty_boundary=False):
        shape = (1, 1, 2, 2)
        regions = {region: torch.zeros(shape) for region in REGION_NAMES}
        if not empty_boundary:
            regions["fg_boundary"][0, 0, 0, :] = 1.0
        else:
            regions["fg_core"][0, 0, 0, :] = 1.0
        regions["bg_near"][0, 0, 1, :] = 1.0
        pack = {
            "p_ref3": torch.full(shape, 0.7),
            "conf_ref3": torch.ones(shape),
            "valid": torch.ones(shape),
            "regions": regions,
            "sdf": torch.zeros(shape),
        }
        image_result = {
            "score": torch.tensor([0.9]),
            "evidence_valid": torch.tensor([True]),
            "allow_image": torch.tensor([True]),
            "global_metadata": {
                "global_type": ["matched"],
                "nearest_labeled_id": ["l0"],
                "sim_max": torch.tensor([0.9]),
            },
        }
        region_result = {
            "score": {region: torch.tensor([0.9]) for region in REGION_NAMES},
            "allow": {region: torch.tensor([True]) for region in REGION_NAMES},
            "components": {
                "region_diversity": {
                    region: torch.tensor([0.2]) for region in REGION_NAMES
                }
            },
        }
        component_maps = {
            name: torch.full(shape, 0.8) for name in TOKEN_FACTOR_NAMES
        }
        token_result = {
            "score": torch.full(shape, score),
            "components": component_maps,
            "cbm_valid": torch.full(shape, cbm_valid, dtype=torch.bool),
            "batch_valid_map": torch.ones(shape, dtype=torch.bool),
            "thresholds": {region: 0.2 for region in REGION_NAMES},
            "score_mode": "weighted_sum",
        }
        return pack, image_result, region_result, token_result

    def _build(self, **scenario):
        cfg = SimpleNamespace(
            sv_ume_regions=["fg_boundary", "bg_near"],
            sv_ume_token_score_mode="weighted_sum",
            sv_ume_diagnostics_interval=20,
            cbm_memory_dim=2,
            cbm_value_dim=8,
            tau_image=0.5,
            tau_region={region: 0.5 for region in REGION_NAMES},
            tau_token={region: 0.2 for region in REGION_NAMES},
        )
        builder = SAMRefinedCandidateBuilder(cfg)
        pack, image_result, region_result, token_result = self._mock_outputs(**scenario)
        inputs = dict(
            img=torch.ones(1, 3, 4, 4),
            img_id=["u0"],
            x3=torch.ones(1, 2, 2, 2),
            p3=torch.ones(1, 2, 2, 2),
            p_raw=torch.ones(1, 1, 2, 2) * 0.8,
            p_ref=torch.ones(1, 1, 2, 2) * 0.7,
            conf_ref=torch.ones(1, 1, 2, 2),
            sam_aux={"used_sam": True, "sam_mask": torch.ones(1, 1, 2, 2)},
            retrieval_aux={},
            labeled_memory=SimpleNamespace(mem_dim=2, value_dim=8),
            epoch=29,
            step=1,
        )
        module = "CBM.sv_ume.sam_refined_candidate_builder"
        with patch(f"{module}.build_sam_refined_regions", return_value=pack), patch(
            f"{module}.compute_image_consistency", return_value=image_result
        ), patch(
            f"{module}.compute_region_consistency", return_value=region_result
        ), patch(f"{module}.compute_token_reliability", return_value=token_result):
            return builder.build(**inputs)

    def test_boundary_only_builds_two_regions_without_disabled_rejections(self):
        result = self._build()
        self.assertEqual(len(result["candidate_pools"]["fg_boundary"]), 2)
        self.assertEqual(len(result["candidate_pools"]["bg_near"]), 2)
        self.assertEqual(len(result["candidate_pools"]["fg_core"]), 0)
        self.assertEqual(len(result["candidate_pools"]["bg_far"]), 0)
        self.assertEqual(sum(result["rejected"].values()), 0)
        self.assertEqual(result["stats"]["disabled_regions"], ["fg_core", "bg_far"])

    def test_cbm_threshold_and_empty_region_rejections_are_distinct(self):
        cbm = self._build(cbm_valid=False)
        threshold = self._build(score=0.1)
        empty = self._build(empty_boundary=True)
        self.assertGreater(cbm["rejected"]["token_cbm_invalid"], 0)
        self.assertGreater(threshold["rejected"]["token_below_threshold"], 0)
        self.assertEqual(threshold["rejected"]["token_cbm_invalid"], 0)
        self.assertGreater(empty["rejected"]["region_empty"], 0)


class FusionScheduleManagerTests(unittest.TestCase):
    @staticmethod
    def _retrieval(y_value, r_value):
        return {
            "Y_map": torch.full((1, 4, 2, 2), y_value),
            "R_map": torch.full((1, 2, 2, 2), r_value),
            "sim_mean": torch.ones(1, 1, 2, 2),
            "topk_consistency": torch.ones(1, 1, 2, 2),
            "memory_reliability": torch.ones(1, 1, 2, 2),
            "U_map": torch.zeros(1, 1, 2, 2),
            "valid_map": torch.ones(1, 1, 2, 2),
        }

    def test_safe_fusion_caps_aux_and_preserves_labeled_features(self):
        cfg = _safe_config()
        fusion = QualityAdaptiveSourceFusion(cfg)
        ret_l = self._retrieval(0.2, 3.0)
        ret_u = self._retrieval(0.8, 9.0)
        fused = fusion(ret_l, ret_u)
        self.assertLessEqual(fused["w_u_map"].max().item(), 0.25)
        self.assertTrue(torch.equal(fused["R_map"], ret_l["R_map"]))
        unpenalized = fusion.compute_score(ret_u, ret_l["Y_map"])
        self.assertTrue(torch.allclose(fused["score_u"], unpenalized - 0.25))

    def test_lagged_schedule_and_manager_accept_safe_config(self):
        cfg = _safe_config()
        manager = SVUMEManager(
            cfg,
            candidate_builder=_CandidateDependency(),
            memory_builder=_MemoryDependency(),
        )
        self.assertTrue(should_build_after_epoch(cfg, 29))
        self.assertIsNone(expected_unlabeled_source_epoch(cfg, 29))
        self.assertFalse(can_use_lagged_memory(cfg, 29, 28))
        memory = _ReadyMemory()
        manager.U_prev = memory
        manager._u_prev_epoch = 29
        self.assertIs(manager.get_unlabeled_memory_for_epoch(30), memory)
        self.assertEqual(manager.last_used_u_prev_epoch, 29)

    def test_manager_aggregates_batch_diagnostics(self):
        manager = SVUMEManager(
            _safe_config(),
            candidate_builder=_CandidateDependency(),
            memory_builder=_MemoryDependency(),
        )
        manager.epoch_stats = manager._new_epoch_stats(29, "collecting")
        summary = {"count": 4, "mean": 0.5, "p50": 0.5, "p90": 0.8, "p99": 0.9, "max": 1.0}
        manager._accumulate_candidate_diagnostics(
            {
                "rejected": {"token_below_threshold": 3},
                "stats": {
                    "batch_size": 2,
                    "image_score_mean": 0.7,
                    "image_score_min": 0.6,
                    "image_score_max": 0.8,
                    "region_pixel_counts": {"fg_boundary": 8},
                    "candidate_counts": {"fg_boundary": 4},
                    "global_type_counts": {"matched": 2},
                    "region_score_mean": {region: 0.5 for region in REGION_NAMES},
                    "token_score_quantiles": {"fg_boundary": summary},
                    "token_component_quantiles": {"r_teacher": summary},
                    "cbm_valid_ratio": {"fg_boundary": 0.75},
                },
            }
        )
        stats = manager.epoch_stats
        self.assertEqual(stats["rejected"]["token_below_threshold"], 3)
        self.assertEqual(stats["raw_candidate_counts"]["fg_boundary"], 4)
        self.assertAlmostEqual(stats["image_score"]["mean"], 0.7)
        self.assertAlmostEqual(stats["cbm_valid_ratio"]["fg_boundary"]["mean"], 0.75)

    def test_legacy_epoch_stats_receive_new_empty_defaults(self):
        restored = SVUMEManager._restore_epoch_stats(
            {"epoch": 28, "status": "legacy", "candidate_counts": {"fg_core": 2}}
        )
        self.assertEqual(restored["candidate_counts"]["fg_core"], 2)
        self.assertIn("token_score_quantiles", restored)
        self.assertEqual(restored["token_score_quantiles"], {})
        self.assertIn("cbm_valid_ratio", restored)


class SVUMEDebugConfigTests(unittest.TestCase):
    def test_debug_run_and_compatible_defaults(self):
        defaults = runpy.run_path("config/base/sam.py")
        self.assertEqual(defaults["sv_ume_token_score_mode"], "product")
        self.assertEqual(defaults["sv_ume_regions"], list(REGION_NAMES))
        self.assertEqual(defaults["aux_source_penalty_value"], 0.0)

        cfg = runpy.run_path("config/runs/sv_ume_svb_plr_full.py")
        self.assertEqual(cfg["tot_epochs"], 35)
        self.assertEqual(cfg["sv_ume_start_epoch"], 29)
        self.assertEqual(cfg["sv_ume_token_score_mode"], "weighted_sum")
        self.assertEqual(cfg["sv_ume_regions"], ["fg_boundary", "bg_near"])
        self.assertEqual(cfg["gamma_max_final"], 0.25)
        self.assertFalse(cfg["use_aux_feature_fusion"])
        self.assertFalse(cfg["sam_use_mask_prompt"])
        self.assertFalse(cfg["sam_pseudo_use_mask"])
        self.assertFalse(cfg["use_sam_embedding_cache"])
        self.assertFalse(cfg["sam_embedding_cache_disk"])
        self.assertFalse(cfg["sam_use_conformal"])
        self.assertTrue(str(cfg["sam2_checkpoint"]).startswith("/home/"))


if __name__ == "__main__":
    unittest.main()
