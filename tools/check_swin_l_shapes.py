import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import Config
from models.build_model import build_model
from models.talnet import ModelEMA


def _assert_shape(name, tensor, expected):
    actual = tuple(tensor.shape[1:])
    print(f"{name}: {tuple(tensor.shape)}")
    assert actual == expected, f"{name} expected [B,{expected}], got {actual}"


def _last_prediction_shape(scaled_preds):
    if isinstance(scaled_preds, tuple):
        scaled_preds = scaled_preds[-1]
    return tuple(scaled_preds[-1].shape)


def main():
    parser = argparse.ArgumentParser(description="Validate TalNet swin_v1_l baseline feature shapes.")
    parser.add_argument("--config", default="config/runs/run.py")
    parser.add_argument("--img-size", default=640, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    cfg = Config(args.config)
    cfg.backbone = "swin_v1_l"
    cfg.lateral_channels_in_collection = [3072, 1536, 768, 384]
    cfg.cxt_num = 3
    cfg.cxt = [384, 768, 1536]
    cfg.cbm_pfi_enable = False
    cfg.use_pc_hbm = False
    cfg.compile_model = False
    cfg.img_size = int(args.img_size)

    device = torch.device(args.device)
    if args.no_pretrained and cfg.model_name == "Default":
        model = ModelEMA(config=cfg, bb_pretrained=False)
    else:
        model = build_model(cfg)
    model = model.to(device)
    model.eval()
    tal = model.student if hasattr(model, "student") else model

    x = torch.randn(args.batch_size, 3, cfg.img_size, cfg.img_size, device=device)
    with torch.no_grad():
        x1, x2, x3, x4, features = tal._build_decoder_features(x)
        state, p3, m3 = tal.decoder.forward_to_p3(features)
        state2, p2, m2 = tal.decoder.forward_p2_from_p3(state, p3)
        scaled_preds, p1, z_main = tal.decoder.forward_p1_from_p2(state2, p2)
        exported = model.forward_return_pc_hbm_features(x)
        legacy = model.extract_cbm_memory_features(x)

    S = int(cfg.img_size)
    _assert_shape("x1", x1, (384, S // 4, S // 4))
    _assert_shape("x2", x2, (768, S // 8, S // 8))
    _assert_shape("x3", x3, (1536, S // 16, S // 16))
    _assert_shape("x4", x4, (3072, S // 32, S // 32))
    _assert_shape("p3", p3, (768, S // 16, S // 16))
    _assert_shape("p2", p2, (384, S // 8, S // 8))
    _assert_shape("p1", p1, (192, S // 4, S // 4))
    _assert_shape("z_main", z_main, (1, S, S))
    print("m3:", None if m3 is None else tuple(m3.shape))
    print("m2:", None if m2 is None else tuple(m2.shape))

    assert _last_prediction_shape(scaled_preds) == tuple(z_main.shape)
    for key in ("x1", "x2", "x3", "x4", "p3", "p2", "m3", "m2", "channel_spec", "input_size"):
        assert key in exported, f"forward_return_pc_hbm_features missing {key}"
    assert tuple(exported["p2"].shape) == tuple(p2.shape)
    assert tuple(legacy["x3"].shape) == tuple(x3.shape)
    assert tuple(legacy["p3"].shape) == tuple(p3.shape)
    print("Swin-L shape check passed.")


if __name__ == "__main__":
    main()
