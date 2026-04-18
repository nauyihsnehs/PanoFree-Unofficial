import os

from .common import ensure_prompt_and_models, load_config as load_json_config, load_source_image, resolve_run_dir
from .debug import save_phase1_debug_artifacts
from .pipeline import generate_initial_view, run_inpaint
from .warp import build_view_homography, warp_image_and_mask


def load_phase1_config(config_path):
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

    config = load_json_config(config_path, defaults)
    validate_config(config)
    return config


def validate_config(config):
    ensure_prompt_and_models(config, "Phase 1")


def build_inpaint_input(warped, missing_mask):
    result = warped.copy()
    result[missing_mask > 0] = 0
    return result


def run_phase1(config_path):
    config = load_phase1_config(config_path)

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
    inpaint_input, inpaint_output = run_inpaint(
        config["prompt"],
        inpaint_input,
        missing_mask,
        config,
    )

    run_dir = resolve_run_dir(config["output"]["run_dir"])
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
