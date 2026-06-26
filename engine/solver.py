import os

import torch
import torch.distributed
import torch.nn as nn
import wandb
from torch.distributed import get_rank

from data import prepare_dataloader, prepare_labeled_memory_dataloader
from utils import AverageMeter, retry_if_cuda_oom
from utils.solver_logging import (
    add_weighted_unsup_stats,
    log_info,
    log_svb_calibrator_state,
    log_training_progress,
    record_cbm_aux,
    record_svb_aux,
)
from CBM.config.schedule import cbm_should_rebuild_memory, cbm_stage_epoch, cbm_stage_id, cbm_stage_name, cbm_unlabeled_enabled
from CBM.diagnostics.visualization import save_pfi_binary_visualizations_v42
try:
    from .loss import PixLoss, weighted_seg_loss
except ImportError:
    from .loss import PixLoss
    weighted_seg_loss = None
try:
    from SAM.SAM_refinement.cbm_aux_adapter import build_retrieval_aux_from_cbm_aux
    from SAM.SAM_refinement.svb_plr import SAMVerifiedBoundaryPseudoLabelRefinement
    from SAM.SAM_refinement.svb_utils import binary_reliability
except ImportError:
    build_retrieval_aux_from_cbm_aux = None
    SAMVerifiedBoundaryPseudoLabelRefinement = None
    binary_reliability = None
from .evaluator import Evaluator


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
        self.cbm = None
        self.cbm_stage = 0
        self.svb_plr = None
        self._svb_plr_import_error_logged = False
        self._svb_same_view_warned = False
        self._svb_conformal_fitted = False
        self._svb_conformal_state_loaded = False
        self._svb_mode_logged = False
        self._init_svb_plr()

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
        self.cbm = self._get_model_cbm()
        self._init_svb_plr()

    def _init_svb_plr(self):
        svb_mode = str(getattr(self.config, "svb_ablation_mode", "full")).strip().lower()
        if not bool(getattr(self.config, "use_svb_plr", False)) or svb_mode == "off":
            self.svb_plr = None
            if bool(getattr(self.config, "use_svb_plr", False)) and not self._svb_mode_logged:
                self._log_info("[SVB-PLR] ablation_mode=off; using baseline pseudo labels.")
                self._svb_mode_logged = True
            return
        if SAMVerifiedBoundaryPseudoLabelRefinement is None:
            self.svb_plr = None
            if not self._svb_plr_import_error_logged:
                self._log_info("[SVB-PLR] modules unavailable; training will use baseline pseudo labels.")
                self._svb_plr_import_error_logged = True
            return
        if self.svb_plr is not None:
            return
        self.svb_plr = SAMVerifiedBoundaryPseudoLabelRefinement(
            self.config,
            device=self.device,
            logger=self.logger,
        )
        self.svb_plr.eval()
        for param in self.svb_plr.parameters():
            param.requires_grad = False
        self._restore_svb_conformal_state()
        if not self._svb_mode_logged:
            self._log_info("[SVB-PLR] ablation_mode={}".format(getattr(self.svb_plr, "ablation_mode", svb_mode)))
            self._svb_mode_logged = True

    def _restore_svb_conformal_state(self):
        if self._svb_conformal_state_loaded or self.svb_plr is None:
            return
        calibrator = getattr(self.svb_plr, "calibrator", None)
        if calibrator is None:
            self._svb_conformal_state_loaded = True
            return
        resume = getattr(self.config, "resume", None)
        if not resume:
            return
        if not os.path.isfile(resume):
            self._log_info("[SVB-PLR] conformal state restore skipped: resume checkpoint not found at '{}'.".format(resume))
            self._svb_conformal_state_loaded = True
            return
        try:
            checkpoint = torch.load(resume, map_location="cpu")
            state = checkpoint.get("svb_conformal_calibrator") if isinstance(checkpoint, dict) else None
            if not state:
                self._log_info("[SVB-PLR] conformal state restore skipped: checkpoint has no svb_conformal_calibrator.")
                self._svb_conformal_state_loaded = True
                return
            calibrator.load_calibrator_state(state, device=self.device)
            self._svb_conformal_fitted = bool(calibrator.is_fitted())
            self._svb_conformal_state_loaded = True
            log_svb_calibrator_state(self.logger, self.svb_plr, "[SVB-PLR] conformal state restored")
        except Exception as exc:
            self._svb_conformal_state_loaded = True
            self._log_info("[SVB-PLR] conformal state restore failed: {}".format(exc))

    def _svb_conformal_state_dict(self):
        if self.svb_plr is None:
            return None
        calibrator = getattr(self.svb_plr, "calibrator", None)
        to_state_dict = getattr(calibrator, "to_state_dict", None)
        if not callable(to_state_dict):
            return None
        try:
            return to_state_dict()
        except Exception as exc:
            self._log_info("[SVB-PLR] conformal state save skipped: {}".format(exc))
            return None

    @retry_if_cuda_oom
    def _train_batch(
        self,
        batch,
        gt_replace=None,
        gt_replace_conf=None,
        gt_replace_aux=None,
        loss_alpha=1.0,
        use_memory=False,
        enable_cbm_loss=False,
        branch_name="Sup",
    ):
        inputs = batch[0].to(self.device)
        gts = batch[1].to(self.device) if gt_replace is None else gt_replace

        cbm_aux = None
        if use_memory:
            scaled_preds, cbm_aux = self.model(inputs, use_memory=True, return_aux=True)
        else:
            scaled_preds = self.model(inputs)

        if self.config.out_ref:
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

        use_weighted_unsup = (
            branch_name == "Unsup"
            and gt_replace_conf is not None
            and bool(getattr(self.config, "use_svb_weighted_unsup_loss", False))
            and weighted_seg_loss is not None
        )
        if use_weighted_unsup:
            target = torch.clamp(gts.detach(), 0, 1)
            conf_ref = gt_replace_conf.detach().to(device=self.device, dtype=target.dtype).clamp(0, 1)
            loss_weight, refine_band, boost_map = self._svb_unsup_loss_weight(conf_ref, gt_replace_aux, target)
            loss_pix = self._weighted_unsup_pix_loss(scaled_preds, target, loss_weight) * loss_alpha
            add_weighted_unsup_stats(self.loss_dict, conf_ref, loss_weight, refine_band, boost_map)
        else:
            loss_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1)) * loss_alpha
            if branch_name == "Unsup":
                self.loss_dict['loss_weighted_unsup'] = 0.0
        self.loss_dict['loss_pix'] = loss_pix.item()

        loss = loss_pix
        if self.config.out_ref:
            loss = loss + loss_gdt
            self.loss_dict['loss_gdt'] = loss_gdt.item() if hasattr(loss_gdt, "item") else float(loss_gdt)

        if enable_cbm_loss and self.cbm is not None:
            loss_cbm = self.cbm.compute_losses(cbm_aux, torch.clamp(gts, 0, 1))
            loss = loss + loss_cbm
            for loss_name, loss_value in self.cbm.state.loss_dict.items():
                self.loss_dict[loss_name] = loss_value
        record_cbm_aux(self.loss_dict, self.cbm, self.cbm_stage, cbm_aux, branch_name, logger=self.logger)
        self._maybe_save_cbm_visualizations(cbm_aux, batch, branch_name)

        self.loss_log.update(loss.item(), inputs.size(0))
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def _weighted_unsup_pix_loss(self, scaled_preds, target, loss_weight):
        preds = scaled_preds if isinstance(scaled_preds, (list, tuple)) else [scaled_preds]
        loss = target.new_zeros(())
        for pred_lvl in preds:
            if pred_lvl.shape != target.shape:
                pred_lvl = nn.functional.interpolate(
                    pred_lvl,
                    size=target.shape[2:],
                    mode='bilinear',
                    align_corners=True,
                )
            loss = loss + weighted_seg_loss(pred_lvl, target, loss_weight)
        return loss

    def _svb_unsup_loss_weight(self, conf_ref, gt_replace_aux, target):
        refine_band = self._svb_refine_band_from_aux(gt_replace_aux, target)
        boost = max(0.0, float(getattr(self.config, "sam_boundary_loss_boost", 0.0)))
        boost_map = (1.0 + boost * refine_band).to(device=target.device, dtype=target.dtype)
        loss_weight = (conf_ref.detach() * boost_map.detach()).clamp_min(0.0)
        loss_weight = torch.nan_to_num(loss_weight, nan=0.0, posinf=1.0 + boost, neginf=0.0)
        return loss_weight.detach(), refine_band.detach(), boost_map.detach()

    def _svb_refine_band_from_aux(self, gt_replace_aux, target):
        refine_band = gt_replace_aux.get("refine_band") if isinstance(gt_replace_aux, dict) else None
        if not torch.is_tensor(refine_band):
            return target.new_zeros(target.shape)
        band = refine_band.detach().to(device=target.device, dtype=target.dtype)
        if band.dim() == 2:
            band = band.unsqueeze(0).unsqueeze(0)
        elif band.dim() == 3:
            band = band.unsqueeze(1)
        elif band.dim() == 4 and band.size(1) != 1:
            band = band[:, :1]
        elif band.dim() != 4:
            return target.new_zeros(target.shape)
        if band.size(0) != target.size(0):
            if band.size(0) == 1:
                band = band.expand(target.size(0), -1, -1, -1)
            else:
                return target.new_zeros(target.shape)
        if tuple(band.shape[-2:]) != tuple(target.shape[-2:]):
            band = nn.functional.interpolate(band, size=target.shape[-2:], mode='nearest')
        return band.clamp(0.0, 1.0).to(device=target.device, dtype=target.dtype)

    @retry_if_cuda_oom
    def train_epoch(self, epoch, total_epochs):
        self.logger.key_info("[+] Training epoch {} ...".format(epoch))
        self.current_epoch = int(epoch)
        self.model.train()
        self.loss_dict = {}
        self.cbm = self._get_model_cbm()
        self._prepare_cbm_epoch(epoch)
        self._prepare_svb_epoch(epoch)
        self.model.train()
        use_memory = self._cbm_use_memory(epoch)
        enable_labeled_cbm_loss = use_memory
        enable_unsup = self._unlabeled_enabled(epoch)

        if epoch > total_epochs + self.config.IoU_finetune_last_epochs:
            self.pix_loss.lambdas_pix_last['bce'] *= 0
            self.pix_loss.lambdas_pix_last['ssim'] *= 1
            self.pix_loss.lambdas_pix_last['iou'] *= 0.5

        for batch_idx, (sup_batch, unsup_batch) in enumerate(zip(self.labeled_dataloader, self.unlabeled_dataloader)):
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
                enable_cbm_loss=enable_labeled_cbm_loss,
                branch_name="Sup",
            )
            if batch_idx % 20 == 0:
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
                )

            if enable_unsup:
                if self.config.distributed_train:
                    self.model.module.teacher.eval()
                else:
                    self.model.teacher.eval()

                img_u_w, student_unsup_batch, geom, image_ids = self._extract_unsup_views(unsup_batch)
                img_u_w = img_u_w.to(self.device)
                if self.svb_plr is None:
                    with torch.no_grad():
                        teacher_preds = self.model(img_u_w, ema=True, use_memory=use_memory)
                        pseudo_s = teacher_preds[-1].sigmoid()
                    conf_s = None
                    sam_aux = {"used_sam": False}
                else:
                    with torch.no_grad():
                        teacher_preds, aux_t = self.model(
                            img_u_w,
                            ema=True,
                            use_memory=use_memory,
                            return_aux=True,
                        )
                        p_t = self._teacher_prob_from_aux(teacher_preds, aux_t)
                        retrieval_aux = (
                            build_retrieval_aux_from_cbm_aux(aux_t)
                            if build_retrieval_aux_from_cbm_aux is not None
                            else {}
                        )
                        p_ref, conf_ref, sam_aux = self.svb_plr.refine(
                            images=img_u_w,
                            teacher_prob=p_t,
                            retrieval_aux=retrieval_aux,
                            image_ids=image_ids,
                            epoch=epoch,
                            step=self.global_step,
                        )
                        pseudo_s, conf_s = self._align_weak_to_strong(p_ref, conf_ref, geom)
                self._train_batch(
                    student_unsup_batch,
                    gt_replace=pseudo_s,
                    gt_replace_conf=conf_s,
                    gt_replace_aux=sam_aux,
                    loss_alpha=float(getattr(self.config, "cbm_unsup_loss_alpha", 0.1)),
                    use_memory=use_memory,
                    enable_cbm_loss=False,
                    branch_name="Unsup",
                )
                if self.svb_plr is not None:
                    record_svb_aux(self.loss_dict, sam_aux, p_t, p_ref, conf_ref, logger=self.logger)

                if batch_idx % 20 == 0:
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
                        include_cbm_losses=False,
                        progress_label="Unsueprvised Training",
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

    def _prepare_cbm_epoch(self, epoch):
        if self.cbm is None:
            return
        self.cbm_stage = cbm_stage_id(self.config, epoch)
        stage_epoch = cbm_stage_epoch(self.config, epoch)
        stage_name = cbm_stage_name(self.config, epoch)
        if cbm_should_rebuild_memory(self.config, epoch):
            self.cbm.prepare_epoch(self.model, self.memory_labeled_dataloader, epoch)
        else:
            self.cbm.state.epoch = int(epoch)
            self.cbm.state.stage_epoch = stage_epoch
            self.cbm.state.stage_name = stage_name
            self.cbm.state.memory_ready = self.cbm.memory.is_ready()
            self._log_info(self.cbm.memory.diagnostic_string())
        ready = self.cbm.memory.is_ready()
        failed = getattr(self.cbm.state, "memory_build_failed", False)
        error = getattr(self.cbm.state, "memory_build_error", None)
        info = (
            f"[CBM] epoch={epoch}, stage_epoch={stage_epoch}, stage={self.cbm_stage}:{stage_name}, "
            f"memory_ready={ready}, memory_failed={failed}"
        )
        if error:
            info += f", fallback_reason={error}"
        self._log_info(info)

    def _prepare_svb_epoch(self, epoch):
        if self.svb_plr is None:
            return
        if not bool(getattr(self.config, "sam_use_conformal", False)):
            return
        if self._svb_conformal_fitted:
            return
        try:
            if int(epoch) != int(getattr(self.config, "sup_only_train_epoch", 0)):
                return
        except (TypeError, ValueError):
            return
        calibrator = getattr(self.svb_plr, "calibrator", None)
        sam_backend = getattr(self.svb_plr, "sam_backend", None)
        if calibrator is None or sam_backend is None:
            self._log_info("[SVB-PLR] conformal calibrator fit skipped: calibrator_or_backend_missing.")
            return
        labeled_loader = getattr(self, "memory_labeled_dataloader", None) or getattr(self, "labeled_dataloader", None)
        if labeled_loader is None:
            self._log_info("[SVB-PLR] conformal calibrator fit skipped: labeled_loader_missing.")
            return
        try:
            calibrator.fit(
                model=self.model,
                memory=self.cbm,
                labeled_loader=labeled_loader,
                sam_backend=sam_backend,
                device=self.device,
            )
            self._svb_conformal_fitted = True
            self._log_info("[SVB-PLR] conformal calibrator fitted at epoch {}.".format(epoch))
            log_svb_calibrator_state(self.logger, self.svb_plr, "[SVB-PLR] conformal calibrator state")
        except Exception as exc:
            self._log_info("[SVB-PLR] conformal calibrator fit skipped: {}".format(exc))

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

    def _teacher_prob_from_aux(self, teacher_preds, aux_t):
        aux_t = aux_t or {}
        p_t = aux_t.get("p_final") if isinstance(aux_t, dict) else None
        if p_t is not None:
            return p_t.detach().clamp(0, 1)
        preds = teacher_preds
        if isinstance(preds, tuple) and len(preds) == 2:
            preds = preds[1]
        return preds[-1].sigmoid().detach()

    def _align_weak_to_strong(self, p_ref, conf_ref, geom):
        if geom is None:
            if not self._svb_same_view_warned:
                self._log_info("[SVB-PLR] same-view pseudo labels are used; weak-to-strong geometry is not enabled.")
                self._svb_same_view_warned = True
            return p_ref.detach(), conf_ref.detach()
        try:
            pseudo_s = self._apply_geom(p_ref, geom).detach()
            conf_s = self._apply_geom(conf_ref, geom).detach()
            return pseudo_s, conf_s
        except Exception as exc:
            if not self._svb_same_view_warned:
                self._log_info("[SVB-PLR] apply_geom failed ({}); falling back to same-view pseudo labels.".format(exc))
                self._svb_same_view_warned = True
            return p_ref.detach(), conf_ref.detach()

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

    def _cbm_use_memory(self, epoch):
        return bool(self.cbm is not None and self.cbm.enabled_for_epoch(epoch))

    def _unlabeled_enabled(self, epoch):
        if self.cbm is None:
            return epoch >= self.config.sup_only_train_epoch
        return cbm_unlabeled_enabled(self.config, epoch)

    def _sync_teacher_this_epoch(self, epoch):
        if self.cbm is None:
            return epoch < self.config.sup_only_train_epoch
        return not cbm_unlabeled_enabled(self.config, epoch)

    def _get_model_cbm(self):
        model = self.model.module if hasattr(getattr(self, "model", None), "module") else getattr(self, "model", None)
        return getattr(model, "cbm", None)

    def _maybe_save_cbm_visualizations(self, aux, batch, branch_name):
        if not aux:
            return
        save_pfi_binary_visualizations_v42(
            aux=aux,
            batch=batch,
            epoch=int(getattr(self, "current_epoch", 0)),
            iteration=int(self.global_step),
            config=self.config,
            logger=self.logger,
            branch_name=branch_name,
        )

    def _log_info(self, message):
        log_info(self.logger, message)

    def _should_evaluate_epoch(self, epoch):
        eval_start = int(getattr(self.config, "eval_epoch", 0))
        eval_step = int(getattr(self.config, "eval_step", 1))
        epoch = int(epoch)
        return eval_step > 0 and epoch >= eval_start and (epoch - eval_start) % eval_step == 0

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
        if self.cbm is not None:
            self.memory_labeled_dataloader = prepare_labeled_memory_dataloader(
                config=self.config,
                labeled_indices=self.current_labeled_indices,
            )
        assert len(self.labeled_dataloader) == len(self.unlabeled_dataloader), (
            "The lenth between labeled_dataloader and unlabeled_dataloader is not equal!"
        )

        for epoch in range(self.epoch_st, total_epochs + 1):
            if self.config.distributed_train:
                self.unlabeled_dataloader.sampler.set_epoch(epoch)
                self.labeled_dataloader.sampler.set_epoch(epoch)
            self.train_epoch(epoch, total_epochs)
            self.logger.success_info("[*] Epoch {} done.".format(epoch))
            self.logger.key_info("[*] Training Loss: {:.3f}".format(self.loss_log.avg))

            if (
                epoch >= total_epochs - self.config.save_last
                and epoch % self.config.save_step == 0
                and ((not self.config.distributed_train) or torch.distributed.get_rank() == 0)
            ):
                model_dict = {
                    'model': self.model.module.state_dict() if self.config.distributed_train else self.model.state_dict(),
                    'optimizer': self.model_optimizer.state_dict(),
                    'lr_scheduler': self.model_lr_scheduler.state_dict(),
                    'epoch': epoch,
                }
                cbm = self._get_model_cbm()
                if cbm is not None and bool(getattr(self.config, "cbm_checkpoint_memory", True)):
                    model_dict['cbm_memory'] = cbm.memory_state_dict()
                svb_conformal_state = self._svb_conformal_state_dict()
                if svb_conformal_state is not None:
                    model_dict['svb_conformal_calibrator'] = svb_conformal_state
                self.logger.freeze_info("[*] Saving model...")
                torch.save(model_dict, os.path.join(self.config.ckpt_dir, 'split{}_model_{}.pth'.format(split, epoch)))
                self.logger.success_info("[*] Model saved.")

            if self.config.distributed_train:
                torch.distributed.barrier()
            if self._should_evaluate_epoch(epoch):
                if (self.config.distributed_train and get_rank() == 0) or (not self.config.distributed_train):
                    self.evaluate_online(epoch, is_last=(epoch == total_epochs))

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
            evaluator.inference_on_dataset(testloader, testset_name, epoch=epoch)
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
