REGION_NAMES = ("fg_core", "fg_boundary", "bg_near", "bg_far")
REGION_TO_ID = {name: idx for idx, name in enumerate(REGION_NAMES)}

DEFAULT_SAMPLE_PER_IMAGE = {
    "fg_core": 128,
    "fg_boundary": 384,
    "bg_near": 384,
    "bg_far": 128,
}

DEFAULT_MAX_SIZES = {
    "fg_core": 8192,
    "fg_boundary": 16384,
    "bg_near": 16384,
    "bg_far": 8192,
}

VALUE_LAYOUT = ("fg_core", "fg_boundary", "bg_near", "bg_far", "bg", "fg", "sdf", "reliability")
