import unittest

import torch

from SAM.SAM_refinement.prompt_expert_selector import PromptExpertSelector
from SAM.SAM_refinement.sam_backend_adapter import ExistingSAMBackendAdapter
from SAM.SAM_refinement.sam_refine_visualizer import SamRefineVisualizer
from SAM.SAM_refinement.svb_utils import SAMInferenceError
from SAM.segment_anything.modeling.prompt_encoder import PromptEncoder
from utils.solver_logging import _sam_teacher_stats, _valid_candidate_ratio


class SAMCandidateValidityTests(unittest.TestCase):
    def test_padding_never_inserts_teacher_candidates(self):
        teacher = torch.ones((2, 1, 4, 4), dtype=torch.float32)
        sample_masks = [
            torch.full((1, 4, 4), 0.25),
            torch.stack(
                (
                    torch.full((4, 4), 0.5),
                    torch.full((4, 4), 0.75),
                )
            ),
        ]
        sample_scores = [torch.tensor([0.2]), torch.tensor([0.3, 0.4])]

        masks, scores, valid = ExistingSAMBackendAdapter._pad_candidates(
            sample_masks, sample_scores, teacher
        )

        self.assertEqual(tuple(masks.shape), (2, 2, 4, 4))
        self.assertEqual(valid.tolist(), [[True, False], [True, True]])
        self.assertTrue(torch.equal(masks[0, 1], torch.zeros((4, 4))))
        self.assertFalse(torch.equal(masks[0, 1], teacher[0, 0]))
        self.assertEqual(scores[0, 1].item(), 0.0)

    def test_selector_never_selects_invalid_teacher_like_candidate(self):
        teacher = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        real_sam = torch.zeros((4, 4), dtype=torch.float32)
        real_sam[1:3, 1:3] = 1.0
        candidates = torch.stack((teacher[0, 0], real_sam)).reshape(1, 2, 4, 4)

        selected, _, aux = PromptExpertSelector().select(
            [
                {
                    "expert": "test",
                    "masks": candidates,
                    "scores": torch.zeros((1, 2)),
                    "valid_candidates": torch.tensor([[False, True]]),
                    "backend_aux": {"fallback_samples": [(0, "synthetic failure")]},
                }
            ],
            teacher,
            {"evidence": {}, "refine_band": torch.zeros_like(teacher)},
        )

        self.assertTrue(torch.equal(selected[0, 0], real_sam))
        self.assertEqual(aux["best_candidate_index"].item(), 1)
        self.assertEqual(aux["valid_candidate_ratio"].item(), 0.5)

    def test_selector_raises_when_all_candidates_are_invalid(self):
        teacher = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        with self.assertRaisesRegex(SAMInferenceError, "all_sam_candidates_invalid"):
            PromptExpertSelector().select(
                [
                    {
                        "expert": "test",
                        "masks": torch.zeros((1, 1, 4, 4)),
                        "scores": torch.zeros((1, 1)),
                        "valid_candidates": torch.zeros((1, 1), dtype=torch.bool),
                        "backend_aux": {"fallback_samples": [(0, "synthetic failure")]},
                    }
                ],
                teacher,
                {"evidence": {}, "refine_band": torch.zeros_like(teacher)},
            )

    def test_sam1_prompt_batches_broadcast_singleton_inputs(self):
        boxes = torch.zeros((3, 4))
        points = torch.zeros((1, 5, 2))
        labels = torch.ones((1, 5), dtype=torch.long)
        masks = torch.zeros((1, 1, 256, 256))

        boxes, points, labels, masks = ExistingSAMBackendAdapter._broadcast_sam1_prompt_batches(
            boxes, points, labels, masks
        )

        self.assertEqual(tuple(boxes.shape), (3, 4))
        self.assertEqual(tuple(points.shape), (3, 5, 2))
        self.assertEqual(tuple(labels.shape), (3, 5))
        self.assertEqual(tuple(masks.shape), (3, 1, 256, 256))

        encoder = PromptEncoder(
            embed_dim=16,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        )
        sparse, dense = encoder(
            points=(points, labels),
            boxes=boxes,
            masks=masks,
        )
        self.assertEqual(sparse.size(0), 3)
        self.assertEqual(dense.size(0), 3)

    def test_sam1_logits_use_sigmoid_instead_of_clamp(self):
        logits = torch.tensor([[[-2.0]], [[0.0]], [[2.0]]])
        probabilities = ExistingSAMBackendAdapter._sam1_mask_logits_to_prob(logits)

        self.assertTrue(torch.allclose(probabilities, logits.sigmoid()))
        self.assertGreater(probabilities[0, 0, 0].item(), 0.0)
        self.assertLess(probabilities[2, 0, 0].item(), 1.0)

    def test_diagnostics_report_raw_sam_teacher_difference(self):
        teacher = torch.zeros((2, 1, 2, 2))
        sam = teacher.clone()
        sam[1] = 1.0

        mae, exact_ratio = _sam_teacher_stats(sam, teacher)
        valid_ratio = _valid_candidate_ratio(
            {"valid_candidates": torch.tensor([[True, False], [True, True]])}
        )

        self.assertEqual(mae, 0.5)
        self.assertEqual(exact_ratio, 0.5)
        self.assertEqual(valid_ratio, 0.75)

    def test_probability_visualization_uses_fixed_zero_one_scale(self):
        panel = SamRefineVisualizer._gray_prob_to_rgb(torch.tensor([[0.25, 0.75]]))

        self.assertGreater(int(panel[0, 0, 0]), 0)
        self.assertLess(int(panel[0, 1, 0]), 255)
        self.assertLess(int(panel[0, 0, 0]), int(panel[0, 1, 0]))


if __name__ == "__main__":
    unittest.main()
