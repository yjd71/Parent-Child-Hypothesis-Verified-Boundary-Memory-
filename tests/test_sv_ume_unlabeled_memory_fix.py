import math
import io
import runpy
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import torch

from CBM.memory.labels import REGION_NAMES
from CBM.sv_ume.quality_adaptive_fusion import QualityAdaptiveSourceFusion
from CBM.sv_ume.sam_refined_candidate_builder import SAMRefinedCandidateBuilder
from CBM.sv_ume.ume_diversity_sampler import UMEDiversitySampler
from CBM.sv_ume.unlabeled_dense_memory import (
    UnlabeledDenseBoundaryMemory,
    UnlabeledMemoryToken,
)
from CBM.sv_ume.schedules import (
    can_use_lagged_memory,
    expected_unlabeled_source_epoch,
    should_build_after_epoch,
)
from CBM.sv_ume.sv_ume_manager import SVUMEManager, SVUMEZeroCandidatesError
from config import Config
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
        self.assertEqual(result["batch_valid_map"].shape, one.shape)
        self.assertTrue(bool(result["batch_valid_map"].all()))


class CandidateDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _mock_outputs(
        cbm_valid=True,
        score=0.8,
        empty_boundary=False,
        width=2,
        scalar_batch_valid=False,
        all_regions=False,
        region_scores=None,
        region_evidence_valid=True,
    ):
        shape = (1, 1, 4 if all_regions else 2, width)
        regions = {region: torch.zeros(shape) for region in REGION_NAMES}
        if all_regions:
            for row, region in enumerate(REGION_NAMES):
                regions[region][0, 0, row, :] = 1.0
        elif not empty_boundary:
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
            "threshold": 0.5,
            "components": {
                "global_teacher_sam_agreement": torch.tensor([0.9]),
                "cbm_supported_change_score": torch.tensor([0.8]),
                "sam_prompt_stability": torch.tensor([0.7]),
                "area_reasonable_score": torch.tensor([1.0]),
                "diversity_gain": torch.tensor([0.2]),
                "over_seg_penalty": torch.tensor([0.0]),
            },
            "global_metadata": {
                "global_type": ["matched"],
                "nearest_labeled_id": ["l0"],
                "sim_max": torch.tensor([0.9]),
            },
        }
        region_scores = region_scores or {region: 0.9 for region in REGION_NAMES}
        region_thresholds = {region: 0.5 for region in REGION_NAMES}
        region_result = {
            "score": {
                region: torch.tensor([region_scores[region]]) for region in REGION_NAMES
            },
            "allow": {
                region: torch.tensor(
                    [region_scores[region] > region_thresholds[region]]
                )
                for region in REGION_NAMES
            },
            "thresholds": region_thresholds,
            "evidence_valid": torch.tensor([region_evidence_valid]),
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
            "batch_valid_map": torch.ones(
                (1, 1, 1, 1) if scalar_batch_valid else shape,
                dtype=torch.bool,
            ),
            "thresholds": {region: 0.2 for region in REGION_NAMES},
            "score_mode": "weighted_sum",
        }
        return pack, image_result, region_result, token_result

    def _build(self, config_overrides=None, **scenario):
        config_values = dict(
            sv_ume_regions=["fg_boundary", "bg_near"],
            sv_ume_token_score_mode="weighted_sum",
            sv_ume_diagnostics_interval=20,
            sv_ume_region_gate_relaxation={
                region: 0.0 for region in REGION_NAMES
            },
            cbm_memory_dim=2,
            cbm_value_dim=8,
            tau_image=0.5,
            tau_region={region: 0.5 for region in REGION_NAMES},
            tau_token={region: 0.2 for region in REGION_NAMES},
        )
        config_values.update(config_overrides or {})
        cfg = SimpleNamespace(**config_values)
        builder = SAMRefinedCandidateBuilder(cfg)
        pack, image_result, region_result, token_result = self._mock_outputs(**scenario)
        height, width = pack["p_ref3"].shape[-2:]
        inputs = dict(
            img=torch.ones(1, 3, height * 2, width * 2),
            img_id=["u0"],
            x3=torch.ones(1, 2, height, width),
            p3=torch.ones(1, 2, height, width),
            p_raw=torch.ones(1, 1, height, width) * 0.8,
            p_ref=torch.ones(1, 1, height, width) * 0.7,
            conf_ref=torch.ones(1, 1, height, width),
            sam_aux={
                "used_sam": True,
                "sam_mask": torch.ones(1, 1, height, width),
            },
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

    def test_spatial_batch_valid_map_supports_column_four(self):
        result = self._build(width=5)
        boundary = result["candidate_pools"]["fg_boundary"]
        near = result["candidate_pools"]["bg_near"]
        self.assertEqual(len(boundary), 5)
        self.assertEqual(len(near), 5)
        self.assertIn((0, 4), [candidate.coord for candidate in boundary])
        self.assertIn((1, 4), [candidate.coord for candidate in near])

    def test_batch_valid_map_shape_error_is_explicit(self):
        with self.assertRaisesRegex(
            ValueError,
            r"batch_valid_map.*shape \(1, 1, 2, 5\).*got \(1, 1, 1, 1\)",
        ):
            self._build(width=5, scalar_batch_valid=True)

    def test_four_region_core_far_relaxed_gate_candidates_are_marked(self):
        relaxation = {
            "fg_core": 0.05,
            "fg_boundary": 0.0,
            "bg_near": 0.0,
            "bg_far": 0.05,
        }
        result = self._build(
            config_overrides={
                "sv_ume_regions": list(REGION_NAMES),
                "sv_ume_region_gate_relaxation": relaxation,
            },
            all_regions=True,
            region_scores={
                "fg_core": 0.47,
                "fg_boundary": 0.9,
                "bg_near": 0.9,
                "bg_far": 0.46,
            },
        )
        for region in REGION_NAMES:
            self.assertGreater(len(result["candidate_pools"][region]), 0)
        for region in ("fg_core", "bg_far"):
            self.assertTrue(
                all(
                    item.meta["region_gate_mode"] == "relaxed"
                    for item in result["candidate_pools"][region]
                )
            )
        for region in ("fg_boundary", "bg_near"):
            self.assertTrue(
                all(
                    item.meta["region_gate_mode"] == "strict"
                    for item in result["candidate_pools"][region]
                )
            )

    def test_log_distribution_passes_strict_half_threshold(self):
        scores = torch.tensor([0.47, 0.49, 0.56, 0.68])
        image_result = {
            "score": scores,
            "evidence_valid": torch.ones(4, dtype=torch.bool),
            "allow_image": scores > 0.5,
            "threshold": 0.5,
            "components": {"agreement": torch.ones(4)},
        }
        stats = SAMRefinedCandidateBuilder._summarize_image_admission(image_result)
        self.assertEqual(stats["image_evidence_valid_count"], 4)
        self.assertEqual(stats["image_above_threshold_count"], 2)
        self.assertEqual(stats["image_allowed_count"], 2)
        self.assertAlmostEqual(stats["image_score_quantiles"]["max"], 0.68, places=6)


class FourRegionDiversitySamplerTests(unittest.TestCase):
    @staticmethod
    def _cfg(**overrides):
        values = dict(
            use_diversity_selection=True,
            lambda_diversity=0.2,
            spatial_nms_distance=2,
            feature_dup_sim_threshold=0.95,
            sv_ume_target_fill_ratio=0.95,
            sv_ume_relaxed_fill=True,
            sv_ume_feature_nms_scope="same_image",
            sv_ume_relaxed_spatial_nms_distance=1,
            sv_ume_relaxed_feature_dup_sim_threshold=0.995,
            sample_per_image_unlabeled={region: 128 for region in REGION_NAMES},
            region_capacity_ratio={region: 1.0 for region in REGION_NAMES},
            cbm_memory_dim=2,
            cbm_value_dim=8,
            use_unlabeled_memory_ema_refresh=False,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _token(region, index, *, image_id=None, gate_mode="strict", reliability=0.9):
        image_id = image_id or f"u{index % 2}"
        image_number = int(image_id[1:]) if image_id[1:].isdigit() else 0
        value = torch.zeros(8)
        value[REGION_NAMES.index(region)] = 1.0
        value[5 if region.startswith("fg_") else 4] = 1.0
        value[7] = float(reliability)
        return UnlabeledMemoryToken(
            key=torch.tensor([1.0, float(index) * 1.0e-4]),
            value=value,
            global_key=torch.tensor([1.0, float(image_number) * 0.1]),
            meta={
                "image_id": image_id,
                "coord": (index // 8, index % 8),
                "region": region,
                "epoch_added": 29,
                "step_added": 0,
                "global_type": "matched",
                "region_gate_mode": gate_mode,
            },
            reliability=reliability,
        )

    @staticmethod
    def _labeled_memory(capacity):
        return SimpleNamespace(
            mem_dim=2,
            keys={region: torch.randn(capacity, 2) for region in REGION_NAMES},
            image_keys=torch.randn(2, 2),
        )

    def test_same_image_feature_nms_keeps_identical_cross_image_tokens(self):
        cfg = self._cfg(
            sv_ume_relaxed_fill=False,
            sample_per_image_unlabeled={region: 2 for region in REGION_NAMES},
        )
        pool = {region: [] for region in REGION_NAMES}
        pool["fg_core"] = [
            self._token("fg_core", 0, image_id="u0"),
            self._token("fg_core", 0, image_id="u1"),
        ]
        result = UMEDiversitySampler(cfg).select(
            candidate_pool=pool,
            labeled_memory=self._labeled_memory(2),
        )
        self.assertEqual(len(result["selected_tokens"]["fg_core"]), 2)

    def test_relaxed_fill_reaches_target_and_preserves_strict_priority(self):
        cfg = self._cfg()
        pool = {region: [] for region in REGION_NAMES}
        pool["fg_core"] = [
            self._token("fg_core", index, gate_mode="strict", reliability=0.8)
            for index in range(10)
        ] + [
            self._token("fg_core", index + 100, gate_mode="relaxed", reliability=0.99)
            for index in range(20)
        ]
        result = UMEDiversitySampler(cfg).select(
            candidate_pool=pool,
            labeled_memory=self._labeled_memory(20),
        )
        selected = result["selected_tokens"]["fg_core"]
        stats = result["stats"]["regions"]["fg_core"]
        self.assertEqual(len(selected), 19)
        self.assertEqual(stats["target_count"], 19)
        self.assertTrue(stats["target_reached"])
        self.assertGreaterEqual(stats["fill_ratio"], 0.95)
        self.assertEqual(
            sum(item.meta["region_gate_mode"] == "strict" for item in selected),
            10,
        )
        self.assertEqual(stats["selection_stage_counts"]["relaxed_gate_fill"], 9)

    def test_underfill_reason_reports_candidate_shortage(self):
        cfg = self._cfg()
        pool = {region: [] for region in REGION_NAMES}
        pool["fg_core"] = [self._token("fg_core", index) for index in range(5)]
        result = UMEDiversitySampler(cfg).select(
            candidate_pool=pool,
            labeled_memory=self._labeled_memory(20),
        )
        stats = result["stats"]["regions"]["fg_core"]
        self.assertEqual(stats["selected_tokens"], 5)
        self.assertEqual(stats["underfill_reason"], "candidate_shortage")

    def test_four_region_memory_builds_aligned_near_one_to_one_snapshot(self):
        cfg = self._cfg()
        pool = {
            region: [self._token(region, index) for index in range(24)]
            for region in REGION_NAMES
        }
        labeled = self._labeled_memory(20)
        selection = UMEDiversitySampler(cfg).select(
            candidate_pool=pool,
            labeled_memory=labeled,
        )
        memory = UnlabeledDenseBoundaryMemory(cfg).build_from_candidates(
            selection["selected_tokens"],
            labeled,
        )
        for region in REGION_NAMES:
            self.assertEqual(memory.keys[region].size(0), 19)
            self.assertEqual(memory.values[region].size(0), 19)
            self.assertEqual(len(memory.meta[region]), 19)
        self.assertTrue(memory.is_ready())


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

    def test_complete_collection_with_zero_candidates_fails_explicitly(self):
        class EmptyCandidateBuilder:
            def __init__(self):
                self.last_result = None

            def build_batch(self, **kwargs):
                self.last_result = {
                    "rejected": {"image_below_threshold": 1},
                    "stats": {
                        "batch_size": 1,
                        "image_score_mean": 0.49,
                        "image_score_min": 0.49,
                        "image_score_max": 0.49,
                        "image_threshold": 0.5,
                        "image_evidence_valid_count": 1,
                        "image_above_threshold_count": 0,
                        "image_allowed_count": 0,
                        "image_score_quantiles": {
                            "count": 1,
                            "mean": 0.49,
                            "p50": 0.49,
                            "p90": 0.49,
                            "p99": 0.49,
                            "max": 0.49,
                        },
                    },
                }
                return {region: [] for region in REGION_NAMES}

        manager = SVUMEManager(
            _safe_config(),
            candidate_builder=EmptyCandidateBuilder(),
            memory_builder=_MemoryDependency(),
        )
        with self.assertRaisesRegex(SVUMEZeroCandidatesError, "image_threshold=0.5"):
            manager.collect_candidates_after_epoch(
                teacher=object(),
                sam_refiner=object(),
                unlabeled_loader=[object()],
                labeled_memory=object(),
                memory_for_retrieval=object(),
                epoch=29,
                device=torch.device("cpu"),
            )
        self.assertEqual(manager.epoch_stats["status"], "zero_candidates")

    def test_candidate_build_promotion_and_next_epoch_use(self):
        candidate = SimpleNamespace(
            key=torch.tensor([1.0]),
            value=torch.tensor([1.0]),
            global_key=torch.tensor([1.0]),
            reliability=0.9,
            diversity=0.1,
            global_meta=None,
            meta={
                "image_id": "u0",
                "coord": (0, 0),
                "region": "fg_boundary",
                "epoch_added": 29,
                "step_added": 0,
                "global_type": "matched",
            },
        )

        class OneCandidateBuilder:
            last_result = None

            def build_batch(self, **kwargs):
                pools = {region: [] for region in REGION_NAMES}
                pools["fg_boundary"] = [candidate]
                self.last_result = {
                    "rejected": {},
                    "stats": {
                        "batch_size": 1,
                        "image_score_mean": 0.6,
                        "image_score_min": 0.6,
                        "image_score_max": 0.6,
                        "image_threshold": 0.5,
                        "image_evidence_valid_count": 1,
                        "image_above_threshold_count": 1,
                        "image_allowed_count": 1,
                        "candidate_counts": {
                            region: int(region == "fg_boundary") for region in REGION_NAMES
                        },
                    },
                }
                return pools

        class ReadyMemory:
            def __init__(self, pools):
                self.keys = {
                    region: torch.stack([item.key for item in pools[region]])
                    if pools[region]
                    else torch.empty(0, 1)
                    for region in REGION_NAMES
                }
                self.values = {
                    region: torch.stack([item.value for item in pools[region]])
                    if pools[region]
                    else torch.empty(0, 1)
                    for region in REGION_NAMES
                }
                self.meta = {
                    region: [dict(item.meta) for item in pools[region]]
                    for region in REGION_NAMES
                }
                self.global_meta = []

            def is_ready(self):
                return any(value.size(0) > 0 for value in self.keys.values())

        class ReadyMemoryBuilder(_MemoryDependency):
            def build_memory(self, **kwargs):
                return ReadyMemory(kwargs["candidate_pool"])

            def memory_state_dict(self, memory):
                return {"keys": memory.keys, "values": memory.values}

        manager = SVUMEManager(
            _safe_config(),
            candidate_builder=OneCandidateBuilder(),
            memory_builder=ReadyMemoryBuilder(),
        )
        labeled_memory = SimpleNamespace(
            keys={region: torch.ones(2, 1) for region in REGION_NAMES}
        )
        manager.collect_candidates_after_epoch(
            teacher=object(),
            sam_refiner=object(),
            unlabeled_loader=[object()],
            labeled_memory=labeled_memory,
            memory_for_retrieval=object(),
            epoch=29,
            device=torch.device("cpu"),
        )
        manager.build_next_memory(labeled_memory, epoch=29)
        promoted = manager.step_epoch()
        self.assertIsNotNone(promoted)
        self.assertIs(manager.get_unlabeled_memory_for_epoch(30), promoted)
        self.assertEqual(manager.last_used_u_prev_epoch, 29)


class SVUMEDebugConfigTests(unittest.TestCase):
    def test_debug_run_and_compatible_defaults(self):
        defaults = runpy.run_path("config/base/sam.py")
        self.assertEqual(defaults["sv_ume_token_score_mode"], "product")
        self.assertEqual(defaults["sv_ume_regions"], list(REGION_NAMES))
        self.assertEqual(defaults["aux_source_penalty_value"], 0.0)
        self.assertFalse(defaults["sv_ume_relaxed_fill"])
        self.assertEqual(defaults["sv_ume_feature_nms_scope"], "global")

        cfg = runpy.run_path("config/runs/sv_ume_svb_plr_full.py")
        self.assertEqual(cfg["tot_epochs"], 35)
        self.assertEqual(cfg["sv_ume_start_epoch"], 16)
        self.assertEqual(cfg["sv_ume_token_score_mode"], "weighted_sum")
        self.assertEqual(cfg["sv_ume_regions"], list(REGION_NAMES))
        self.assertTrue(cfg["sv_ume_relaxed_fill"])
        self.assertEqual(cfg["sv_ume_feature_nms_scope"], "same_image")
        self.assertEqual(cfg["sv_ume_region_gate_relaxation"]["fg_core"], 0.05)
        self.assertEqual(cfg["sv_ume_region_gate_relaxation"]["bg_far"], 0.05)
        self.assertTrue(str(cfg["sam2_checkpoint"]).startswith("/home/"))
        self.assertEqual(cfg["sv_ume_profile_name"], "four_region_near_1to1_v1")

    def test_config_identity_metadata_without_contract(self):
        cases = (
            ("config/runs/sv_ume_svb_plr_full.py", 16),
            ("config/runs/finetune_27_cbm_ume_plr_full.py", 6),
        )
        for path, start_epoch in cases:
            with self.subTest(path=path), redirect_stdout(io.StringIO()):
                cfg = Config(path)
            self.assertTrue(cfg.run_cfg_path.replace("\\", "/").endswith(path))
            self.assertEqual(len(cfg.run_cfg_sha256), 64)
            self.assertEqual(cfg.sv_ume_start_epoch, start_epoch)
            self.assertEqual(cfg.sv_ume_profile_name, "four_region_near_1to1_v1")
            self.assertFalse(hasattr(cfg, "sv_ume_profile_contract"))


if __name__ == "__main__":
    unittest.main()
