import os
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import prettytable as pt

from tqdm import tqdm

from config import Config
from utils import Logger, save_tensor_img
from utils.evaluation_paths import align_evaluation_paths, list_image_paths
from models import build_model_eval
from PC_HBM import pc_hbm_enabled

from .metrics import calculate


class Evaluator:
    def __init__(self, config: Config, logger: Logger, resume: str, device: torch.device='cpu', model: nn.Module=None):
        self.config = config
        self.logger = logger
        self.device = device
        self.model = build_model_eval(config, logger, resume, device) if model is None else model
        self.logger.success_info("[o] Evaluator is ready!")

    @classmethod
    def from_exists(cls, config: Config, logger: Logger, resume: str=None, device: torch.device='cpu', model: nn.Module=None):
        logger.freeze_info("[+] Evaluator will be initialize with existing model...")
        return cls(config, logger, resume, device, model)
    
    def inference_on_dataset(
        self,
        dataloader: data.DataLoader,
        testset_name: str,
        ema=False,
        epoch: int=None,
        memory_t=None,
    ) -> None:
        current_save_dir = os.path.join(self.config.pred_save_root, testset_name)
        if epoch is not None:
            current_save_dir = os.path.join(current_save_dir, f'epoch_{epoch}')
        os.makedirs(current_save_dir, exist_ok=True)
        
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs = batch[0].to(self.device)
            label_paths = batch[-2]

            with torch.no_grad():
                if pc_hbm_enabled(self.config) and hasattr(self.model, "forward_pc_hbm"):
                    _, aux = self.model.forward_pc_hbm(
                        inputs,
                        ema=ema,
                        use_memory=True,
                        return_all_logits=True,
                        epoch=epoch,
                    )
                    scaled_preds = aux.get("p_final", torch.sigmoid(aux["z_final"]))
                elif memory_t is None:
                    scaled_preds = self.model(inputs, ema=ema)[-1].sigmoid()
                else:
                    predictions, aux = self.model(
                        inputs,
                        ema=ema,
                        use_memory=True,
                        memory_t=memory_t,
                        return_aux=True,
                    )
                    p_final = aux.get("p_final") if isinstance(aux, dict) else None
                    if torch.is_tensor(p_final):
                        scaled_preds = p_final
                    else:
                        outputs = predictions
                        if isinstance(outputs, tuple) and len(outputs) == 2:
                            outputs = outputs[1]
                        scaled_preds = outputs[-1].sigmoid()

            for idx_sample in range(scaled_preds.shape[0]):
                res = nn.functional.interpolate(
                    scaled_preds[idx_sample].unsqueeze(0),
                    size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                    mode='bilinear',
                    align_corners=True
                )
                save_imgfile_name = label_paths[idx_sample].replace('\\', '/').split('/')[-1]
                save_tensor_img(res, os.path.join(current_save_dir, save_imgfile_name))
                
    def evaluate_inference_result(self, dataloader: data.DataLoader, testset_name: str, save_dir_replace: str=None, epoch: int=None) -> dict:
        results_dir = os.path.join(self.config.pred_save_root, 'results')
        os.makedirs(results_dir, exist_ok=True)

        default_log_savedir = os.path.join(results_dir, '{}.log'.format(testset_name))
        log_savedir = default_log_savedir if save_dir_replace is None else save_dir_replace
        log_save_parent = os.path.dirname(log_savedir)
        if log_save_parent:
            os.makedirs(log_save_parent, exist_ok=True)

        current_result_dir = os.path.join(self.config.pred_save_root, testset_name)
        if epoch is not None:
            current_result_dir = os.path.join(current_result_dir, f'epoch_{epoch}')
        gt_path = os.path.join(self.config.data_root_dir, self.config.task, testset_name)
        assert os.path.isdir(current_result_dir), f"[x] {current_result_dir} not exists!"
        
        gt_paths = list_image_paths(os.path.join(gt_path, 'gt'))
        pred_paths = list_image_paths(current_result_dir)
        gt_paths, pred_paths, extra_prediction_stems = align_evaluation_paths(
            gt_paths,
            pred_paths,
        )
        if extra_prediction_stems:
            message = (
                f"[Evaluator] ignoring {len(extra_prediction_stems)} stale prediction(s) "
                f"not present in GT; examples={extra_prediction_stems[:20]}"
            )
            log_fn = (
                getattr(self.logger, "warn_info", None)
                or getattr(self.logger, "warning", None)
                or getattr(self.logger, "info", None)
            )
            if callable(log_fn):
                log_fn(message)
        
        with open(log_savedir, 'a+', encoding='utf-8') as fw:
            tb = pt.PrettyTable()
            tb.field_names = [
                "Dataset", "Task", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "maxEm", "meanFm",
                "adpEm", "adpFm",
            ]
            self.logger.key_info("[+] Starting calculate the evaluation result...")
            em, sm, fm, mae, wfm = calculate(gt_paths, pred_paths, metrics=['S', 'MAE', 'E', 'F', 'WF'], verbose=True)
            
            e_max, e_mean, e_adp = em['curve'].max(), em['curve'].mean(), em['adp'].mean()
            f_max, f_mean, f_wfm, f_adp = fm['curve'].max(), fm['curve'].mean(), wfm, fm['adp']
            
            tb.add_row([
                testset_name, self.config.task,
                f_max.round(3), f_wfm.round(3), mae.round(3), sm.round(3), e_mean.round(3), e_max.round(3), f_mean.round(3),
                e_adp.round(3), f_adp.round(3)
            ])
            
            self.logger.success_info('\n'+str(tb)+'\n')
            if epoch is not None:
                fw.write('Epoch {}:\n'.format(epoch))
            fw.write(str(tb).replace('+', '|') + '\n\n')
        
        return dict(e_max=e_max, e_mean=e_mean, e_adp=e_adp, f_max=f_max, f_mean=f_mean, f_wfm=f_wfm, f_adp=f_adp, mae=mae, sm=sm)
    
            
