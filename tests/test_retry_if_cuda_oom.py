import unittest
from unittest.mock import call, patch

from SAM.SAM_refinement.svb_utils import SAMInferenceError
from utils.tools import retry_if_cuda_oom


class RetryIfCudaOOMTests(unittest.TestCase):
    @patch("utils.tools.logger.warn_info")
    def test_sam_inference_error_is_raised_without_retry(self, warn_info):
        calls = 0
        error = SAMInferenceError(
            "no valid SAM candidate",
            epoch=29,
            step=0,
            sample_indices=[4],
        )

        @retry_if_cuda_oom
        def fail():
            nonlocal calls
            calls += 1
            raise error

        with self.assertRaises(SAMInferenceError) as caught:
            fail()

        self.assertEqual(calls, 1)
        self.assertIs(caught.exception, error)
        warn_info.assert_not_called()

    @patch("utils.tools.gc.collect")
    @patch("utils.tools.torch.cuda.empty_cache")
    @patch("utils.tools.time.sleep")
    @patch("utils.tools.logger.warn_info")
    def test_persistent_cuda_oom_is_re_raised_after_final_attempt(
        self,
        warn_info,
        sleep,
        empty_cache,
        collect,
    ):
        calls = 0

        @retry_if_cuda_oom
        def fail():
            nonlocal calls
            calls += 1
            raise RuntimeError("CUDA out of memory while allocating tensor")

        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            fail()

        self.assertEqual(calls, 10)
        self.assertEqual(sleep.call_count, 9)
        self.assertEqual(empty_cache.call_count, 9)
        self.assertEqual(collect.call_count, 9)
        warn_info.assert_any_call("Reached maximum retry attempts for CUDA OOM.")

    @patch("utils.tools.gc.collect")
    @patch("utils.tools.torch.cuda.empty_cache")
    @patch("utils.tools.time.sleep")
    @patch("utils.tools.logger.warn_info")
    def test_cuda_oom_retry_can_recover(
        self,
        warn_info,
        sleep,
        empty_cache,
        collect,
    ):
        calls = 0

        @retry_if_cuda_oom
        def recover():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("CUDA out of memory")
            return "ok"

        self.assertEqual(recover(), "ok")
        self.assertEqual(calls, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(empty_cache.call_count, 2)
        self.assertEqual(collect.call_count, 2)
        self.assertNotIn(
            call("Reached maximum retry attempts for CUDA OOM."),
            warn_info.call_args_list,
        )


if __name__ == "__main__":
    unittest.main()
