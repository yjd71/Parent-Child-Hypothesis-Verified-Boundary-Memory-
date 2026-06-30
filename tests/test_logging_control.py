import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils.log_control import should_log
from utils.solver_logging import (
    log_training_progress,
    partition_training_metrics,
    should_log_training_progress,
)


class _ListLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class LoggingControlTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {
            "loss_pix": 1.25,
            "loss_gdt": 0.5,
            "loss_cbm_total": 0.2,
            "gate_mean": 0.3,
            "svb_sam_score_mean": 0.8,
            "conf_ref_mean": 0.7,
            "loss_sv_ume": 0.1,
        }

    def test_baseline_progress_uses_fixed_twenty_batch_interval(self):
        self.assertTrue(should_log_training_progress(0))
        self.assertFalse(should_log_training_progress(19))
        self.assertTrue(should_log_training_progress(20))
        self.assertFalse(should_log_training_progress(21))

    def test_partition_training_metrics_assigns_module_ownership(self):
        base, modules = partition_training_metrics(self.metrics)

        self.assertEqual(set(base), {"loss_pix", "loss_gdt"})
        self.assertEqual(set(modules["CBM"]), {"loss_cbm_total", "gate_mean"})
        self.assertEqual(
            set(modules["SVB-PLR"]),
            {"svb_sam_score_mean", "conf_ref_mean"},
        )
        self.assertEqual(set(modules["SV-UME"]), {"loss_sv_ume"})

    @patch("utils.solver_logging.wandb.log")
    def test_disabled_module_logging_does_not_disable_baseline_log(self, wandb_log):
        logger = _ListLogger()
        config = SimpleNamespace(log_enable=False, log_interval=20)
        log_base = should_log_training_progress(20)
        log_modules = should_log(config, 20)

        log_training_progress(
            logger=logger,
            loss_dict=self.metrics,
            title="Semi-Supervised Training Losses",
            wandb_prefix="Sup",
            epoch=19,
            total_epochs=30,
            batch_idx=20,
            num_batches=639,
            step=20,
            log_base=log_base,
            log_modules=log_modules,
        )

        self.assertEqual(len(logger.messages), 1)
        self.assertIn("loss_pix: 1.250", logger.messages[0])
        self.assertNotIn("[CBM]", logger.messages[0])
        payload = wandb_log.call_args.args[0]
        self.assertEqual(set(payload), {"Sup-loss_pix", "Sup-loss_gdt"})

    @patch("utils.solver_logging.wandb.log")
    def test_baseline_log_excludes_module_metrics(self, wandb_log):
        logger = _ListLogger()

        log_training_progress(
            logger=logger,
            loss_dict=self.metrics,
            title="Semi-Supervised Training Losses",
            wandb_prefix="Sup",
            epoch=19,
            total_epochs=30,
            batch_idx=20,
            num_batches=639,
            step=20,
            log_base=True,
            log_modules=False,
        )

        self.assertEqual(len(logger.messages), 1)
        self.assertIn("loss_pix: 1.250", logger.messages[0])
        self.assertNotIn("loss_cbm_total", logger.messages[0])
        self.assertNotIn("svb_sam_score_mean", logger.messages[0])
        payload = wandb_log.call_args.args[0]
        self.assertEqual(set(payload), {"Sup-loss_pix", "Sup-loss_gdt"})

    @patch("utils.solver_logging.wandb.log")
    def test_module_log_uses_independent_prefixed_lines(self, wandb_log):
        logger = _ListLogger()

        log_training_progress(
            logger=logger,
            loss_dict=self.metrics,
            title="Unsueprvised Training Losses",
            wandb_prefix="Unsup",
            epoch=19,
            total_epochs=30,
            batch_idx=30,
            num_batches=639,
            step=30,
            log_base=False,
            log_modules=True,
        )

        self.assertEqual(len(logger.messages), 3)
        self.assertTrue(logger.messages[0].startswith("[CBM] Unsup"))
        self.assertTrue(logger.messages[1].startswith("[SVB-PLR] Unsup"))
        self.assertTrue(logger.messages[2].startswith("[SV-UME] Unsup"))
        self.assertTrue(all("loss_pix" not in message for message in logger.messages))
        payload = wandb_log.call_args.args[0]
        self.assertNotIn("Unsup-loss_pix", payload)
        self.assertIn("Unsup-loss_cbm_total", payload)
        self.assertIn("Unsup-svb_sam_score_mean", payload)
        self.assertIn("Unsup-loss_sv_ume", payload)


if __name__ == "__main__":
    unittest.main()
