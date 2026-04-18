import copy
import json
import os

import numpy as np
from PIL import Image


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path, defaults):
    with open(config_path, "r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    return deep_update(copy.deepcopy(defaults), user_config)


def ensure_prompt_and_models(config, phase_name):
    if not config.get("prompt"):
        raise RuntimeError("`prompt` is required for {}.".format(phase_name))
    if not config["models"].get("base_model"):
        raise RuntimeError("`models.base_model` is required for {}.".format(phase_name))
    if not config["models"].get("inpaint_model"):
        raise RuntimeError("`models.inpaint_model` is required for {}.".format(phase_name))


def load_source_image(path, expected_size):
    image = Image.open(path).convert("RGB")
    if image.size != expected_size:
        image = image.resize(expected_size, resample=Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)


def resolve_run_dir(path):
    return os.path.abspath(path)

