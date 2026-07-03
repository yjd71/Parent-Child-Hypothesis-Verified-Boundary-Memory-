import hashlib
import os

COMMON_CONFIG_DIR='config/base/common.py'
MODEL_CONFIG_DIR='config/base/model.py'
CBM_CONFIG_DIR='config/base/cbm.py'

from utils import Logger
logger = Logger(name='Config', path='/tmp/talnet_preload.log')

class Config:
    def __init__(self, run_cfg: str):
        logger.key_info("Initialize config...")
        run_cfg_path = os.path.abspath(os.path.normpath(run_cfg))
        self.merge_from_file(COMMON_CONFIG_DIR)
        self.merge_from_file(MODEL_CONFIG_DIR)
        self.merge_from_file(CBM_CONFIG_DIR)
        self.merge_from_file(run_cfg_path)
        self.run_cfg_path = run_cfg_path
        with open(run_cfg_path, "rb") as config_file:
            self.run_cfg_sha256 = hashlib.sha256(config_file.read()).hexdigest()
        logger.success_info("Config merged from {}.".format(run_cfg_path))
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")
    
    def merge_from_file(self, config_dir: str) -> None:
        assert os.path.isfile(config_dir), f'[-] {config_dir} not found!'
        with open(config_dir, 'r', encoding='utf-8') as f:
            local_vars = {}
            exec(f.read(), {}, local_vars)
            for key, value in local_vars.items():
                setattr(self, key, value)
