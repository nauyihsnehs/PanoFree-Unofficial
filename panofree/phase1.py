import copy
import json
import os

import numpy as np
from PIL import Image

from .debug import save_phase1_debug_artifacts
from .pipeline import generate_initial_view, run_inpaint
from .warp import build_view_homography, warp_image_and_mask


def deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path):
    defaults = {
        "prompt": "",
        "seed": 1234,
        "models": {
            "base_model": "",
            "inpaint_model": "",
        },
        "input": {
            "source_image": "",
        },
        "source_view": {
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "fov_deg": 80.0,
            "width": 512,
            "height": 512,
        },
        "target_view": {
            "yaw_deg": 40.0,
            "pitch_deg": 0.0,
            "fov_deg": 80.0,
            "width": 512,
            "height": 512,
        },
        "generation": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
        },
        "inpaint": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
        },
        "output": {
            "run_dir": "outputs/phase1",
            "pano_width": 4096,
            "pano_height": 2048,
        },
    }

    with open(config_path, "r", encoding="utf-8") as handle:
        user_config = json.load(handle)

    config = deep_update(copy.deepcopy(defaults), user_config)
    validate_config(config)
    return config


def validate_config(config):
    if not config.get("prompt"):
        raise RuntimeError("`prompt` is required for Phase 1.")

    if not config["models"].get("base_model"):
        raise RuntimeError("`models.base_model` is required for Phase 1.")

    if not config["models"].get("inpaint_model"):
        raise RuntimeError("`models.inpaint_model` is required for Phase 1.")


def load_source_image(path, expected_size):
    image = Image.open(path).convert("RGB")
    if image.size != expected_size:
        image = image.resize(expected_size, resample=Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)


def build_inpaint_input(warped, missing_mask):
    result = warped.copy()
    result[missing_mask > 0] = 0
    return result


def run_phase1(config_path):
    config = load_config(config_path)

    source_image_path = config["input"].get("source_image")
    source_size = (
        config["source_view"]["width"],
        config["source_view"]["height"],
    )

    if source_image_path:
        initial_view = load_source_image(source_image_path, source_size)
    else:
        initial_view = generate_initial_view(config)

    homography = build_view_homography(config["source_view"], config["target_view"])
    warped, known_mask, missing_mask = warp_image_and_mask(
        initial_view,
        homography,
        (
            config["target_view"]["width"],
            config["target_view"]["height"],
        ),
    )
    inpaint_input = build_inpaint_input(warped, missing_mask)
    inpaint_output = run_inpaint(
        config["prompt"],
        inpaint_input,
        missing_mask,
        config,
    )

    run_dir = os.path.abspath(config["output"]["run_dir"])
    save_phase1_debug_artifacts(
        run_dir,
        {
            "config": config,
            "prompt": config["prompt"],
            "initial_view": initial_view,
            "homography": homography,
            "warped": warped,
            "known_mask": known_mask,
            "missing_mask": missing_mask,
            "inpaint_input": inpaint_input,
            "inpaint_output": inpaint_output,
            "source_view": config["source_view"],
            "target_view": config["target_view"],
            "pano_width": config["output"]["pano_width"],
            "pano_height": config["output"]["pano_height"],
        },
    )

    return {
        "run_dir": run_dir,
        "source_image_used": bool(source_image_path),
    }
