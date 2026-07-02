import unittest

from utils.evaluation_paths import align_evaluation_paths


class EvaluatorPathAlignmentTests(unittest.TestCase):
    def test_aligns_by_stem_and_ignores_extra_predictions(self):
        gts = ["/gt/b.png", "/gt/a.png"]
        predictions = ["/pred/stale.png", "/pred/a.jpg", "/pred/b.png"]
        aligned_gts, aligned_predictions, extras = align_evaluation_paths(gts, predictions)
        self.assertEqual(aligned_gts, ["/gt/a.png", "/gt/b.png"])
        self.assertEqual(aligned_predictions, ["/pred/a.jpg", "/pred/b.png"])
        self.assertEqual(extras, ["stale"])

    def test_missing_prediction_reports_stems(self):
        with self.assertRaisesRegex(RuntimeError, "missing predictions.*b"):
            align_evaluation_paths(["/gt/a.png", "/gt/b.png"], ["/pred/a.png"])

    def test_duplicate_prediction_stems_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate prediction stems"):
            align_evaluation_paths(
                ["/gt/a.png"],
                ["/pred/a.png", "/pred/a.jpg"],
            )


if __name__ == "__main__":
    unittest.main()
