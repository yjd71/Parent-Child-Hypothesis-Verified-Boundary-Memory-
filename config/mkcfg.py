import hashlib
import os

COMMON_CONFIG_DIR='config/base/common.py'
MODEL_CONFIG_DIR='config/base/model.py'
CBM_CONFIG_DIR='config/base/cbm.py'
SAM_CONFIG_DIR='config/base/sam.py'

from utils import Logger
logger = Logger(name='Config', path='/tmp/talnet_preload.log')

class Config:
    def __init__(self, run_cfg: str):
        logger.key_info("Initialize config...")
        run_cfg_path = os.path.abspath(os.path.normpath(run_cfg))
        self.merge_from_file(COMMON_CONFIG_DIR)
        self.merge_from_file(MODEL_CONFIG_DIR)
        self.merge_from_file(CBM_CONFIG_DIR)
        self.merge_from_file(SAM_CONFIG_DIR)
        self.merge_from_file(run_cfg_path)
        self._normalize_sam_refine_mode()
        self.run_cfg_path = run_cfg_path
        with open(run_cfg_path, "rb") as config_file:
            self.run_cfg_sha256 = hashlib.sha256(config_file.read()).hexdigest()
        logger.success_info("Config merged from {}.".format(run_cfg_path))
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")
    
    def _normalize_sam_refine_mode(self) -> None:
        raw_mode = getattr(self, "sam_refine_mode", "off")
        if not isinstance(raw_mode, str):
            raise TypeError("sam_refine_mode must be a string")
        mode = raw_mode.strip().lower()
        valid_modes = {"off", "legacy_auto", "svb"}
        if mode not in valid_modes:
            raise ValueError(
                "sam_refine_mode must be one of {}, got {!r}".format(
                    sorted(valid_modes), raw_mode
                )
            )

        self.sam_refine_mode = mode
        self.use_svb_plr = mode == "svb"
        self.use_sam_refine_unlabeled = mode != "off"
        self.use_sam_pseudo_refine = mode == "legacy_auto"

        if mode != "svb":
            self.use_sv_ume = False
            self.use_ume_evidence_loss = False
            self.use_source_consistency_loss = False
            self.use_svb_weighted_unsup_loss = False
        elif str(getattr(self, "svb_ablation_mode", "full")).strip().lower() == "off":
            raise ValueError(
                "sam_refine_mode='svb' is incompatible with svb_ablation_mode='off'"
            )

        if isinstance(getattr(self, "others", None), dict):
            self.others.update(
                {
                    "sam_refine_mode": mode,
                    "use_svb_plr": self.use_svb_plr,
                    "use_sam_refine_unlabeled": self.use_sam_refine_unlabeled,
                    "use_sv_ume": bool(getattr(self, "use_sv_ume", False)),
                    "use_svb_weighted_unsup_loss": bool(
                        getattr(self, "use_svb_weighted_unsup_loss", False)
                    ),
                }
            )

    def merge_from_file(self, config_dir: str) -> None:
        assert os.path.isfile(config_dir), f'[-] {config_dir} not found!'
        with open(config_dir, 'r', encoding='utf-8') as f:
            local_vars = {}
            exec(f.read(), {}, local_vars)
            for key, value in local_vars.items():
                setattr(self, key, value)
