from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class TalNetChannelSpec:
    backbone: str
    lateral_channels: Tuple[int, int, int, int]
    x1: int
    x2: int
    x3: int
    x4: int
    p1: int
    p2: int
    p3: int
    p4: int
    pc_dim: int
    value_dim: int
    feature_version: str

    def as_dict(self):
        data = asdict(self)
        data["lateral_channels"] = list(self.lateral_channels)
        return data


def build_talnet_channel_spec(config) -> TalNetChannelSpec:
    channels = tuple(int(channel) for channel in getattr(config, "lateral_channels_in_collection"))
    if len(channels) != 4:
        raise ValueError(
            "TalNet expects four lateral channels ordered [x4, x3, x2, x1], "
            f"got {channels}"
        )

    x4, x3, x2, x1 = channels
    feature_version = str(getattr(config, "cbm_memory_feature_version", "swin_l_pc_hbm_v1"))
    return TalNetChannelSpec(
        backbone=str(getattr(config, "backbone", "")),
        lateral_channels=channels,
        x1=x1,
        x2=x2,
        x3=x3,
        x4=x4,
        p1=x1 // 2,
        p2=x1,
        p3=x2,
        p4=x3,
        pc_dim=int(getattr(config, "cbm_memory_dim", 128)),
        value_dim=int(getattr(config, "cbm_value_dim", 8)),
        feature_version=feature_version,
    )
