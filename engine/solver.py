import os

import torch
import torch.distributed
import torch.nn as nn
import wandb
from torch.distributed import get_rank

from data import prepare_dataloader
from utils import AverageMeter, retry_if_cuda_oom
from .loss import PixLoss
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

    @retry_if_cuda_oom
    def _train_batch(self, batch, gt_replace=None, loss_alpha=1.0):
        inputs = batch[0].to(self.device)
        gts = batch[1].to(self.device) if gt_replace is None else gt_replace

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

        loss_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1)) * loss_alpha
        self.loss_dict['loss_pix'] = loss_pix.item()

        loss = loss_pix
        if self.config.out_ref:
            loss = loss + loss_gdt

        self.loss_log.update(loss.item(), inputs.size(0))
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    @retry_if_cuda_oom
    def train_epoch(self, epoch, total_epochs):
        self.logger.key_info("[+] Training epoch {} ...".format(epoch))
        self.model.train()
        self.loss_dict = {}

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

            self._train_batch(sup_batch)
            if batch_idx % 20 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(
                    epoch, total_epochs, batch_idx, len(self.labeled_dataloader)
                )
                info_loss = 'Semi-Supervised Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                if (not self.config.distributed_train) or (self.config.distributed_train and get_rank() == 0):
                    wandb.log({"Sup-" + k: v for k, v in self.loss_dict.items()}, step=self.global_step)
                self.logger.info(' '.join((info_progress, info_loss)))

            if epoch >= self.config.sup_only_train_epoch:
                if self.config.distributed_train:
                    self.model.module.teacher.eval()
                else:
                    self.model.teacher.eval()

                inputs = unsup_batch[0].to(self.device)
                with torch.no_grad():
                    p_labels = self.model(inputs, ema=True)[-1].sigmoid()
                self._train_batch(unsup_batch, gt_replace=p_labels, loss_alpha=0.1)

                if batch_idx % 20 == 0:
                    info_progress = 'Unsueprvised Training Epoch[{0}/{1}] Iter[{2}/{3}].'.format(
                        epoch, total_epochs, batch_idx, len(self.unlabeled_dataloader)
                    )
                    info_loss = 'Unsueprvised Training Losses'
                    for loss_name, loss_value in self.loss_dict.items():
                        info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                    self.logger.info(' '.join((info_progress, info_loss)))
                    if (not self.config.distributed_train) or (self.config.distributed_train and get_rank() == 0):
                        wandb.log({"Unsup-loss": self.loss_dict['loss_pix']}, step=self.global_step)

            self.global_step += 1

            if epoch < self.config.sup_only_train_epoch:
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
                self.logger.freeze_info("[*] Saving model...")
                torch.save(model_dict, os.path.join(self.config.ckpt_dir, 'split{}_model_{}.pth'.format(split, epoch)))
                self.logger.success_info("[*] Model saved.")

            if self.config.distributed_train:
                torch.distributed.barrier()
            if epoch % self.config.eval_step == 0:
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
            evaluator.inference_on_dataset(testloader, testset_name)
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
