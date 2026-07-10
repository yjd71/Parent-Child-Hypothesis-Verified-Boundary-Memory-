import os

import torch
import torch.distributed
import torch.nn as nn
import wandb
from torch.distributed import get_rank

from data import prepare_dataloader, prepare_labeled_memory_dataloader
from utils import AverageMeter, retry_if_cuda_oom
from utils.log_control import log_enabled, should_log
from utils.solver_logging import (
    log_info,
    log_training_progress,
    should_log_training_progress,
)
from PC_HBM.core.pc_config import pc_hbm_enabled, pc_hbm_unlabeled_enabled
from PC_HBM.training.pc_losses import compute_pc_hbm_unlabeled_loss, structure_aware_confidence
from .loss import PixLoss
from .evaluator import Evaluator


def training_epoch_range(epoch_st, total_epochs):
    """Return the strict 0-based, end-exclusive training epoch range."""
    return range(int(epoch_st), int(total_epochs))


class SemiSupervisedTrainer:
    def __init__(self, data_loaders, config, device, logger=None, writer=None):
        self.train_loader, self.test_loaders = data_loaders

        self.config = config
        self.writer = writer
        self.cnt = 0
        if self.config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        self.pix_loss = PixLoss(config)
        self.loss_log = AverageMeter()

        self.global_step = 0
        self.device = device
        self.logger = logger
        self.pc_hbm = None

    def _destory_model(self):
        try:
            del self.model, self.model_optimizer, self.model_lr_scheduler
        except Exception:
            pass

    def reset_trainer(self, model_lrsch, labeled_indices, split=None):
        self._destory_model()
        torch.cuda.empty_cache()
        self.model, self.model_optimizer, self.model_lr_scheduler, self.epoch_st = model_lrsch
        self.current_labeled_indices = labeled_indices
        self.pc_hbm = self._get_model_pc_hbm()

    @retry_if_cuda_oom
    def _train_batch(
        self,
        batch,
        gt_replace=None,
        loss_alpha=1.0,
        use_memory=False,
        enable_pc_hbm_loss=False,
        branch_name="Sup",
    ):
        inputs = batch[0].to(self.device)
        gts = batch[1].to(self.device) if gt_replace is None else gt_replace

        module_aux = None
        if self._pc_hbm_active():
            scaled_preds, module_aux = self.model.forward_pc_hbm(
                inputs,
                use_memory=use_memory,
                return_all_logits=True,
                epoch=getattr(self, "current_epoch", None),
            )
        else:
            scaled_preds = self.model(inputs)

        if self.config.out_ref and not self._pc_hbm_active():
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            loss_gdt = 0
            for idx, (gdt_pred, gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                gdt_pred = nn.functional.interpolate(
                    gdt_pred,
                    size=gdt_label.shape[2:],
                    mode='bilinear',
                    align_corners=True,
                ).sigmoid()
                gdt_label = gdt_label.sigmoid()
                cur = self.criterion_gdt(gdt_pred, gdt_label)
                loss_gdt = cur if idx == 0 else loss_gdt + cur

        if self._pc_hbm_active() and enable_pc_hbm_loss:
            loss_pix = self.pc_hbm.compute_losses(scaled_preds, module_aux, torch.clamp(gts, 0, 1)) * loss_alpha
            for loss_name, loss_value in self.pc_hbm.loss_dict.items():
                self.loss_dict[loss_name] = loss_value
        else:
            loss_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1)) * loss_alpha
        self.loss_dict['loss_pix'] = loss_pix.item()

        loss = loss_pix
        if self.config.out_ref and not self._pc_hbm_active():
            loss = loss + loss_gdt
            self.loss_dict['loss_gdt'] = loss_gdt.item() if hasattr(loss_gdt, "item") else float(loss_gdt)

        self.loss_log.update(loss.item(), inputs.size(0))
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    @retry_if_cuda_oom
    def train_epoch(self, epoch, total_epochs):
        self.logger.key_info("[+] Training epoch {} ...".format(epoch))
        self.current_epoch = int(epoch)
        self.model.train()
        self.loss_dict = {}
        self.pc_hbm = self._get_model_pc_hbm()
        if self._pc_hbm_active():
            self._prepare_pc_hbm_epoch(epoch)
        self.model.train()
        use_memory = self._pc_hbm_use_memory(epoch)
        enable_pc_hbm_loss = self._pc_hbm_active() and use_memory
        enable_unsup = self._pc_hbm_unlabeled_enabled(epoch) if self._pc_hbm_active() else self._unlabeled_enabled(epoch)

        iou_finetune_offset = int(self.config.IoU_finetune_last_epochs)
        if iou_finetune_offset < 0 and epoch >= total_epochs + iou_finetune_offset:
            self.pix_loss.lambdas_pix_last['bce'] *= 0
            self.pix_loss.lambdas_pix_last['ssim'] *= 1
            self.pix_loss.lambdas_pix_last['iou'] *= 0.5

        for batch_idx, (sup_batch, unsup_batch) in enumerate(zip(self.labeled_dataloader, self.unlabeled_dataloader)):
            log_base_progress = should_log_training_progress(batch_idx)
            log_module_progress = self._should_log_module()
            if self.writer and (not self.config.distributed_train or get_rank() == 0):
                if batch_idx < 25:
                    self.writer.add_image(
                        sup_batch[-2][0],
                        torch.cat((sup_batch[0][0], sup_batch[1][0].repeat(3, 1, 1)), dim=-1),
                        global_step=self.cnt,
                    )
                elif batch_idx == 25:
                    self.cnt += 1

            self._train_batch(
                sup_batch,
                use_memory=use_memory,
                enable_pc_hbm_loss=enable_pc_hbm_loss,
                branch_name="Sup",
            )
            if log_base_progress or log_module_progress:
                log_training_progress(
                    logger=self.logger,
                    loss_dict=self.loss_dict,
                    title='Semi-Supervised Training Losses',
                    wandb_prefix="Sup",
                    epoch=epoch,
                    total_epochs=total_epochs,
                    batch_idx=batch_idx,
                    num_batches=len(self.labeled_dataloader),
                    step=self.global_step,
                    distributed_train=self.config.distributed_train,
                    log_base=log_base_progress,
                    log_modules=log_module_progress,
                )

            if enable_unsup:
                if self._pc_hbm_active():
                    self._train_pc_hbm_unsup(unsup_batch, use_memory)
                    if log_base_progress or log_module_progress:
                        log_training_progress(
                            logger=self.logger,
                            loss_dict=self.loss_dict,
                            title='Unsueprvised Training Losses',
                            wandb_prefix="Unsup",
                            epoch=epoch,
                            total_epochs=total_epochs,
                            batch_idx=batch_idx,
                            num_batches=len(self.unlabeled_dataloader),
                            step=self.global_step,
                            distributed_train=self.config.distributed_train,
                            progress_label="Unsueprvised Training",
                            log_base=log_base_progress,
                            log_modules=log_module_progress,
                        )
                    self.global_step += 1
                    if self.config.distributed_train:
                        self.model.module.ema_update(self.global_step)
                    else:
                        self.model.ema_update(self.global_step)
                    continue
                if self.config.distributed_train:
                    self.model.module.teacher.eval()
                else:
                    self.model.teacher.eval()

                img_u_w, student_unsup_batch, geom, _ = self._extract_unsup_views(unsup_batch)
                img_u_w = img_u_w.to(self.device)
                with torch.no_grad():
                    teacher_preds = self.model(
                        img_u_w,
                        ema=True,
                    )
                    teacher_pseudo = teacher_preds[-1].sigmoid()
                pseudo_s = self._align_pseudo_to_strong(teacher_pseudo, geom)
                self._train_batch(
                    student_unsup_batch,
                    gt_replace=pseudo_s,
                    loss_alpha=float(getattr(self.config, "lambda_u", 1.0)),
                    use_memory=use_memory,
                    enable_pc_hbm_loss=False,
                    branch_name="Unsup",
                )

                if log_base_progress or log_module_progress:
                    log_training_progress(
                        logger=self.logger,
                        loss_dict=self.loss_dict,
                        title='Unsueprvised Training Losses',
                        wandb_prefix="Unsup",
                        epoch=epoch,
                        total_epochs=total_epochs,
                        batch_idx=batch_idx,
                        num_batches=len(self.unlabeled_dataloader),
                        step=self.global_step,
                        distributed_train=self.config.distributed_train,
                        progress_label="Unsueprvised Training",
                        log_base=log_base_progress,
                        log_modules=log_module_progress,
                    )

            self.global_step += 1

            if self._sync_teacher_this_epoch(epoch):
                if self.config.distributed_train:
                    self.model.module.ema_update(self.global_step, 0)
                else:
                    self.model.ema_update(self.global_step, 0)
            else:
                if self.config.distributed_train:
                    self.model.module.ema_update(self.global_step)
                else:
                    self.model.ema_update(self.global_step)

        return self.loss_log.avg

    def _prepare_pc_hbm_epoch(self, epoch):
        if self.pc_hbm is None:
            return
        if int(epoch) >= int(getattr(self.config, "parent_start_epoch", 6)):
            self.pc_hbm.prepare_epoch(self.model, self.memory_labeled_dataloader, epoch)
        ready = self.pc_hbm.memory.is_ready()
        error = getattr(self.pc_hbm, "memory_build_error", None)
        info = f"[PC-HBM] epoch={epoch}, memory_ready={ready}, memory_failed={getattr(self.pc_hbm, 'memory_build_failed', False)}"
        if error:
            info += f", fallback_reason={error}"
        self._log_module_info(info)

    def _train_pc_hbm_unsup(self, unsup_batch, use_memory):
        if self.config.distributed_train:
            self.model.module.teacher.eval()
        else:
            self.model.teacher.eval()
        img_u_w, student_unsup_batch, geom, _ = self._extract_unsup_views(unsup_batch)
        img_u_w = img_u_w.to(self.device)
        with torch.no_grad():
            _, teacher_aux = self.model.forward_pc_hbm(
                img_u_w,
                ema=True,
                use_memory=use_memory,
                return_all_logits=True,
                epoch=getattr(self, "current_epoch", None),
                forward_mode="teacher_pseudo",
                need_p1_pra=True,
                need_final_mixture=True,
                return_debug_aux=False,
                store_last_aux=False,
            )
            teacher_pseudo = teacher_aux.get("p_final", torch.sigmoid(teacher_aux["z_final"])).detach()
            confidence = structure_aware_confidence(teacher_aux).detach()
        pseudo_s = self._align_pseudo_to_strong(teacher_pseudo, geom)
        conf_s = self._align_pseudo_to_strong(confidence, geom)
        del teacher_aux, teacher_pseudo, confidence, img_u_w
        img_u_s = student_unsup_batch[0].to(self.device)
        student_core_only = bool(getattr(self.config, "pc_hbm_unsup_student_core_only", True))
        _, student_aux = self.model.forward_pc_hbm(
            img_u_s,
            use_memory=use_memory,
            return_all_logits=True,
            epoch=getattr(self, "current_epoch", None),
            forward_mode="student_core" if student_core_only else "full",
            need_p1_pra=False if student_core_only else getattr(self.config, "pc_hbm_unsup_student_need_p1_pra", None),
            need_final_mixture=False if student_core_only else getattr(self.config, "pc_hbm_unsup_student_need_final_mixture", None),
            return_debug_aux=False,
            store_last_aux=False,
        )
        loss_u, log = compute_pc_hbm_unlabeled_loss(
            student_aux,
            pseudo_s,
            conf_s,
            self.config,
            epoch=getattr(self, "current_epoch", None),
        )
        self.loss_dict["loss_pix"] = loss_u.item()
        for key, value in log.items():
            self.loss_dict[key] = float(value.detach().item())
        self.loss_log.update(loss_u.item(), img_u_s.size(0))
        self.model_optimizer.zero_grad()
        loss_u.backward()
        self.model_optimizer.step()
        del student_aux, pseudo_s, conf_s, img_u_s, loss_u

    def _extract_unsup_views(self, unsup_batch):
        if isinstance(unsup_batch, dict):
            img_u_w = self._first_existing(unsup_batch, ("img_u_w", "image_w", "weak", "image", "img"))
            img_u_s = self._first_existing(unsup_batch, ("img_u_s", "image_s", "strong"))
            if img_u_s is None:
                img_u_s = img_u_w
            geom = self._first_existing(unsup_batch, ("geom", "geometry", "transform"))
            image_ids = self._first_existing(unsup_batch, ("image_ids", "image_id", "ids", "id"))
            student_batch = [img_u_s, None, image_ids]
            return img_u_w, student_batch, geom, image_ids

        img_u_w = unsup_batch[0]
        geom = None
        image_ids = self._extract_image_ids_from_batch(unsup_batch)
        return img_u_w, unsup_batch, geom, image_ids

    @staticmethod
    def _first_existing(mapping, keys):
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None

    @staticmethod
    def _extract_image_ids_from_batch(batch):
        if isinstance(batch, (list, tuple)) and len(batch) > 2:
            return batch[2]
        return None

    def _align_pseudo_to_strong(self, pseudo_prob, geom):
        if geom is None:
            return pseudo_prob.detach()
        return self._apply_geom(pseudo_prob, geom).detach()

    @staticmethod
    def _apply_geom(tensor, geom):
        if callable(geom):
            return geom(tensor)
        for method_name in ("apply", "apply_mask", "apply_to_mask"):
            method = getattr(geom, method_name, None)
            if callable(method):
                return method(tensor)
        if isinstance(geom, dict):
            out = tensor
            if bool(geom.get("hflip", False) or geom.get("flip", False)):
                out = torch.flip(out, dims=(-1,))
            if bool(geom.get("vflip", False)):
                out = torch.flip(out, dims=(-2,))
            return out
        return tensor

    def _pc_hbm_use_memory(self, epoch):
        return bool(self.pc_hbm is not None and self.pc_hbm.enabled_for_epoch(epoch))

    def _unlabeled_enabled(self, epoch):
        return epoch >= self.config.sup_only_train_epoch

    def _pc_hbm_unlabeled_enabled(self, epoch):
        return bool(self.pc_hbm is not None and pc_hbm_unlabeled_enabled(self.config, epoch))

    def _sync_teacher_this_epoch(self, epoch):
        if self.pc_hbm is None:
            return epoch < self.config.sup_only_train_epoch
        return not pc_hbm_unlabeled_enabled(self.config, epoch)

    def _get_model_pc_hbm(self):
        model = self.model.module if hasattr(getattr(self, "model", None), "module") else getattr(self, "model", None)
        return getattr(model, "pc_hbm", None)

    def _pc_hbm_active(self):
        return bool(self.pc_hbm is not None and pc_hbm_enabled(self.config))

    def _log_module_info(self, message):
        if self._module_logging_enabled():
            log_info(self.logger, message)

    def _module_logging_enabled(self):
        return log_enabled(self.config)

    def _should_log_module(self, step=None):
        if step is None:
            step = getattr(self, "global_step", None)
        return should_log(self.config, step)

    def _should_evaluate_epoch(self, epoch):
        eval_start = int(getattr(self.config, "eval_epoch", 0))
        eval_step = int(getattr(self.config, "eval_step", 1))
        epoch = int(epoch)
        return eval_step > 0 and epoch >= eval_start and (epoch - eval_start) % eval_step == 0

    def _reset_optimizer_for_stage2(self, epoch):
        split_epoch = int(getattr(self.config, "sup_only_train_epoch", -1))
        if epoch != split_epoch or not bool(
            getattr(self.config, "reset_optimizer_at_stage2", False)
        ):
            return False

        from models.build_model import _build_lr_scheduler, _build_optimizer

        self.model_optimizer = _build_optimizer(
            self.config,
            self.model,
            lr=float(self.config.stage2_initial_lr),
        )
        self.model_lr_scheduler = _build_lr_scheduler(
            self.config,
            self.model_optimizer,
        )
        return True

    def _step_lr_before_epoch(self, epoch):
        previous_lr = float(self.model_optimizer.param_groups[0]["lr"])
        optimizer_was_reset = self._reset_optimizer_for_stage2(epoch)
        step_epoch = getattr(self.model_lr_scheduler, "step_epoch", None)
        if not callable(step_epoch):
            return

        step_epoch(epoch)
        current_lr = float(self.model_optimizer.param_groups[0]["lr"])
        current_stage = int(self.model_lr_scheduler.current_stage)
        if epoch == int(self.config.sup_only_train_epoch):
            self.logger.key_info(
                "[*] Two-stage cosine restart: "
                f"epoch={epoch}, previous_lr={previous_lr:.3e}, "
                f"restart_lr={current_lr:.3e}, "
                f"optimizer_state_preserved={not optimizer_was_reset}"
            )
        self.logger.key_info(
            f"[*] LR before epoch: epoch={epoch}, stage={current_stage}, lr={current_lr:.3e}"
        )

    def _should_save_checkpoint(self, epoch, total_epochs):
        save_step = int(getattr(self.config, "save_step", 0))
        save_last = int(getattr(self.config, "save_last", 0))
        regular_save = (
            save_step > 0
            and epoch >= total_epochs - save_last
            and epoch % save_step == 0
        )
        split_epoch = int(getattr(self.config, "sup_only_train_epoch", -1))
        boundary_save = bool(getattr(self.config, "save_stage_boundary", False)) and epoch in {
            split_epoch - 1,
            split_epoch,
        }
        return regular_save or boundary_save

    def _lr_schedule_meta(self, total_epochs):
        active_stage = getattr(self.model_lr_scheduler, "current_stage", 0)
        return {
            "scheduler_type": str(getattr(self.config, "scheduler_type", "multistep")),
            "tot_epochs": int(getattr(self.config, "tot_epochs", total_epochs)),
            "sup_only_train_epoch": int(
                getattr(self.config, "sup_only_train_epoch", -1)
            ),
            "active_stage": int(active_stage),
            "current_lr": [
                float(group["lr"]) for group in self.model_optimizer.param_groups
            ],
        }

    def launch_train(self, split, total_epochs: int):
        self.labeled_dataloader = prepare_dataloader(
            dataset=self.train_loader.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            to_be_distributed=self.config.distributed_train,
            is_train=True,
            labeled_indices=self.current_labeled_indices,
        )

        self.unlabeled_dataloader = prepare_dataloader(
            dataset=self.train_loader.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            to_be_distributed=self.config.distributed_train,
            is_train=True,
            labeled_indices=self.current_labeled_indices,
            is_unsup=True,
        )
        self.memory_labeled_dataloader = None
        if self.pc_hbm is not None:
            self.memory_labeled_dataloader = prepare_labeled_memory_dataloader(
                config=self.config,
                labeled_indices=self.current_labeled_indices,
            )
        assert len(self.labeled_dataloader) == len(self.unlabeled_dataloader), (
            "The lenth between labeled_dataloader and unlabeled_dataloader is not equal!"
        )

        for epoch in training_epoch_range(self.epoch_st, total_epochs):
            if self.config.distributed_train:
                self.unlabeled_dataloader.sampler.set_epoch(epoch)
                self.labeled_dataloader.sampler.set_epoch(epoch)
            self._step_lr_before_epoch(epoch)
            self.train_epoch(epoch, total_epochs)
            self.logger.success_info("[*] Epoch {} done.".format(epoch))
            self.logger.key_info("[*] Training Loss: {:.3f}".format(self.loss_log.avg))
            if not callable(getattr(self.model_lr_scheduler, "step_epoch", None)):
                self.model_lr_scheduler.step()
            current_lr = self.model_optimizer.param_groups[0]["lr"]
            self.logger.key_info("[*] Current LR: {:.3e}".format(current_lr))

            if (
                self._should_save_checkpoint(epoch, total_epochs)
                and ((not self.config.distributed_train) or torch.distributed.get_rank() == 0)
            ):
                model_dict = {
                    'model': self.model.module.state_dict() if self.config.distributed_train else self.model.state_dict(),
                    'optimizer': self.model_optimizer.state_dict(),
                    'lr_scheduler': self.model_lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'lr_schedule_meta': self._lr_schedule_meta(total_epochs),
                }
                pc_hbm = self._get_model_pc_hbm()
                if pc_hbm is not None and bool(getattr(self.config, "pc_hbm_checkpoint_memory", True)):
                    model_dict['pc_hbm_memory'] = pc_hbm.memory_state_dict()
                self.logger.freeze_info("[*] Saving model...")
                torch.save(model_dict, os.path.join(self.config.ckpt_dir, 'split{}_model_{}.pth'.format(split, epoch)))
                self.logger.success_info("[*] Model saved.")

            if self.config.distributed_train:
                torch.distributed.barrier()
            if self._should_evaluate_epoch(epoch):
                if (self.config.distributed_train and get_rank() == 0) or (not self.config.distributed_train):
                    self.evaluate_online(epoch, is_last=(epoch == total_epochs - 1))

    def evaluate_online(self, epoch, is_last=False):
        if self.config.distributed_train and get_rank() != 0:
            return
        self.logger.key_info("[+] Online evaluation created, model epoch: {}...".format(epoch))
        self.model.eval()
        evaluator = Evaluator.from_exists(
            config=self.config,
            logger=self.logger,
            device=self.device,
            model=self.model if not self.config.distributed_train else self.model.module,
        )
        for testset_name, testloader in self.test_loaders.items():
            evaluator.inference_on_dataset(
                testloader,
                testset_name,
                epoch=epoch,
            )
            result = evaluator.evaluate_inference_result(testloader, testset_name, epoch=epoch)
            wandb.log(
                {
                    'T-MAE': result['mae'],
                    'T-maxFm': result['f_max'],
                    'T-wFmeasure': result['f_wfm'],
                    'T-SMeasure': result['sm'],
                    'T-meanEm': result['e_mean'],
                    'T-meanFm': result['f_mean'],
                },
                step=wandb.run.step,
            )
            if is_last:
                wandb.log(
                    {
                        'F-MAE': result['mae'],
                        'F-maxFm': result['f_max'],
                        'F-wFmeasure': result['f_wfm'],
                        'F-SMeasure': result['sm'],
                        'F-meanEm': result['e_mean'],
                        'F-meanFm': result['f_mean'],
                    }
                )
        self.logger.key_info("[+] Online evaluation done...")
