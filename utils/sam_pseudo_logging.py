class SamPseudoRefineLogger:
    def __init__(self, logger=None, enabled=True, interval=20):
        self.logger = logger
        self.enabled = bool(enabled)
        self.interval = max(1, int(interval))
        self._warned_reasons = set()

    def log(self, level, message):
        if self.logger is None:
            return
        method_names = {
            "key": ("key_info", "info"),
            "warn": ("warn_info", "info"),
            "info": ("info",),
        }.get(level, ("info",))
        for method_name in method_names:
            method = getattr(self.logger, method_name, None)
            if method is not None:
                method(message)
                return

    def warn_once(self, reason, message):
        if reason in self._warned_reasons:
            return
        self._warned_reasons.add(reason)
        self.log("warn", message)

    def should_log(self, step):
        if not self.enabled:
            return False
        if step is None:
            return True
        return int(step) % self.interval == 0

    def log_init(
        self,
        model_type,
        checkpoint,
        device,
        threshold,
        fusion_alpha,
        iters,
        use_point,
        use_box,
        use_mask,
        add_neg,
        margin,
        gamma,
        strength,
    ):
        self.log(
            "key",
            "[SAM_INIT] enabled=True model_type={} checkpoint={} checkpoint_exists={} "
            "device={} threshold={:.3f} fusion_alpha={:.3f} iters={} "
            "use_point={} use_box={} use_mask={} add_neg={} "
            "margin={:.3f} gamma={:.3f} strength={:.3f}".format(
                model_type,
                checkpoint,
                checkpoint.is_file(),
                device,
                threshold,
                fusion_alpha,
                iters,
                use_point,
                use_box,
                use_mask,
                add_neg,
                margin,
                gamma,
                strength,
            ),
        )

    @staticmethod
    def new_batch_stats(batch_size, image_hw, pseudo_hw):
        return {
            "batch_size": batch_size,
            "image_hw": tuple(image_hw),
            "pseudo_hw": tuple(pseudo_hw),
            "non_empty": 0,
            "refined": 0,
            "skipped_empty": 0,
            "skipped_error": 0,
        }

    @staticmethod
    def new_change_sums():
        return {
            "teacher_area": 0.0,
            "sam_area": 0.0,
            "fused_area": 0.0,
            "iou": 0.0,
            "changed": 0.0,
            "add": 0.0,
            "remove": 0.0,
            "delta": 0.0,
        }

    @staticmethod
    def mask_change_metrics(teacher_prob, sam_mask, fused, threshold):
        teacher_mask = teacher_prob > threshold
        sam_binary = sam_mask > 0.5
        fused_mask = fused > threshold

        teacher_float = teacher_mask.float()
        sam_float = sam_binary.float()
        intersection = (teacher_float * sam_float).sum()
        union = (teacher_mask | sam_binary).float().sum()
        iou = intersection / union.clamp_min(1.0)

        return {
            "teacher_area": teacher_float.mean().item(),
            "sam_area": sam_float.mean().item(),
            "fused_area": fused_mask.float().mean().item(),
            "iou": iou.item(),
            "changed": (teacher_mask != sam_binary).float().mean().item(),
            "add": (sam_binary & (~teacher_mask)).float().mean().item(),
            "remove": ((~sam_binary) & teacher_mask).float().mean().item(),
            "delta": (fused - teacher_prob).abs().mean().item(),
        }

    @staticmethod
    def add_change_metrics(change_sums, metrics):
        for key, value in metrics.items():
            change_sums[key] += value

    def log_batch(self, epoch, step, batch_stats, change_sums, total_refined, total_skipped):
        if not self.should_log(step):
            return

        self.log(
            "info",
            "[SAM_BATCH] epoch={} step={} batch={} image_hw={} pseudo_hw={} "
            "non_empty={} refined={} skipped_empty={} skipped_error={} "
            "total_refined={} total_skipped={}".format(
                epoch,
                step,
                batch_stats["batch_size"],
                batch_stats["image_hw"],
                batch_stats["pseudo_hw"],
                batch_stats["non_empty"],
                batch_stats["refined"],
                batch_stats["skipped_empty"],
                batch_stats["skipped_error"],
                total_refined,
                total_skipped,
            ),
        )

        refined = batch_stats["refined"]
        if refined == 0:
            return

        self.log(
            "info",
            "[SAM_CHANGE] epoch={} step={} mean_teacher_area={:.6f} mean_sam_area={:.6f} "
            "mean_fused_area={:.6f} mean_iou={:.6f} mean_changed={:.6f} "
            "mean_add={:.6f} mean_remove={:.6f} mean_delta={:.6f}".format(
                epoch,
                step,
                change_sums["teacher_area"] / refined,
                change_sums["sam_area"] / refined,
                change_sums["fused_area"] / refined,
                change_sums["iou"] / refined,
                change_sums["changed"] / refined,
                change_sums["add"] / refined,
                change_sums["remove"] / refined,
                change_sums["delta"] / refined,
            ),
        )
