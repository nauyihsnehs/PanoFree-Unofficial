from copy import deepcopy

import numpy as np

from .common import ensure_prompt_and_models, load_config, load_source_image, resolve_run_dir
from .debug import save_phase2_debug_artifacts
from .phase1 import build_inpaint_input
from .pipeline import create_generator, generate_initial_view, run_guided_inpaint
from .stitch import stitch_equirectangular_views
from .warp import build_view_homography, warp_image_and_mask


def load_phase2_config(config_path):
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
        "generation": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
        },
        "inpaint": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "strength": 0.98,
        },
        "central_band": {
            "fov_deg": 80.0,
            "pitch_deg": 0.0,
            "yaw_stride_deg": 40.0,
            "steps_each_direction": 3,
            "merge_yaw_deg": 180.0,
            "sdedit_t0": 0.98,
            "stitch_pitch_min_deg": -40.0,
            "stitch_pitch_max_deg": 40.0,
        },
        "output": {
            "run_dir": "outputs/phase2/example_run",
            "pano_width": 4096,
            "pano_height": 2048,
        },
    }
    config = load_config(config_path, defaults)
    validate_phase2_config(config)
    return config


def validate_phase2_config(config):
    ensure_prompt_and_models(config, "Phase 2")
    if config["central_band"]["steps_each_direction"] != 3:
        raise RuntimeError("Phase 2 requires `central_band.steps_each_direction` to be 3.")


def build_view(base_view, yaw_deg, pitch_deg, fov_deg):
    view = deepcopy(base_view)
    view["yaw_deg"] = yaw_deg
    view["pitch_deg"] = pitch_deg
    view["fov_deg"] = fov_deg
    return view


def generate_or_load_initial_view(config, generator):
    source_image_path = config["input"].get("source_image")
    source_size = (
        config["source_view"]["width"],
        config["source_view"]["height"],
    )
    if source_image_path:
        return load_source_image(source_image_path, source_size), True
    return generate_initial_view(config, generator=generator), False


def create_initial_record(config, initial_view):
    return {
        "name": "x0",
        "kind": "initial",
        "view": deepcopy(config["source_view"]),
        "image": initial_view,
    }


def build_schedule(config):
    stride = config["central_band"]["yaw_stride_deg"]
    return [
        {"name": "x1", "yaw_deg": stride, "direction": 1, "step_index": 1},
        {"name": "x-1", "yaw_deg": -stride, "direction": -1, "step_index": 1},
        {"name": "x2", "yaw_deg": stride * 2.0, "direction": 1, "step_index": 2},
        {"name": "x-2", "yaw_deg": -stride * 2.0, "direction": -1, "step_index": 2},
        {"name": "x3", "yaw_deg": stride * 3.0, "direction": 1, "step_index": 3},
        {"name": "x-3", "yaw_deg": -stride * 3.0, "direction": -1, "step_index": 3},
    ]


def find_record(view_records, name):
    for record in view_records:
        if record["name"] == name:
            return record
    raise RuntimeError("Missing view record: {}".format(name))


def get_source_name(direction, step_index):
    if step_index == 1:
        return "x0"
    prefix = "x" if direction > 0 else "x-"
    return "{}{}".format(prefix, step_index - 1)


def get_guidance_name(direction, step_index):
    if step_index == 1:
        return "x0"
    if direction > 0:
        return "x-{}".format(step_index - 1)
    return "x{}".format(step_index - 1)


def run_step(config, step_spec, view_records, generator):
    source_name = get_source_name(step_spec["direction"], step_spec["step_index"])
    guidance_name = get_guidance_name(step_spec["direction"], step_spec["step_index"])
    source_record = find_record(view_records, source_name)
    guidance_record = find_record(view_records, guidance_name)

    target_view = build_view(
        config["source_view"],
        step_spec["yaw_deg"],
        config["central_band"]["pitch_deg"],
        config["central_band"]["fov_deg"],
    )
    homography = build_view_homography(source_record["view"], target_view)
    warped, known_mask, missing_mask = warp_image_and_mask(
        source_record["image"],
        homography,
        (target_view["width"], target_view["height"]),
    )
    inpaint_input = build_inpaint_input(warped, missing_mask)
    guided_input, inpaint_output = run_guided_inpaint(
        config["prompt"],
        inpaint_input,
        missing_mask,
        guidance_record["image"],
        config,
        generator=generator,
        strength=config["central_band"]["sdedit_t0"],
    )
    return {
        "name": step_spec["name"],
        "kind": "step",
        "view": target_view,
        "image": inpaint_output,
        "source_name": source_name,
        "guidance_name": guidance_name,
        "guidance_image": guidance_record["image"],
        "guided_input": guided_input,
        "warped": warped,
        "known_mask": known_mask,
        "missing_mask": missing_mask,
    }


def build_merge_composite(left_warped, right_warped, left_known_mask, right_known_mask):
    left_valid = left_known_mask > 0
    right_valid = right_known_mask > 0
    composite = np.zeros_like(left_warped)
    left_only = left_valid & ~right_valid
    right_only = right_valid & ~left_valid
    both = left_valid & right_valid
    composite[left_only] = left_warped[left_only]
    composite[right_only] = right_warped[right_only]
    composite[both] = (
        (left_warped[both].astype(np.float32) + right_warped[both].astype(np.float32)) * 0.5
    ).astype(np.uint8)
    missing_mask = np.where(left_valid | right_valid, 0, 255).astype(np.uint8)
    return composite, missing_mask


def run_merge_step(config, view_records, generator):
    left_record = find_record(view_records, "x3")
    right_record = find_record(view_records, "x-3")
    target_view = build_view(
        config["source_view"],
        config["central_band"]["merge_yaw_deg"],
        config["central_band"]["pitch_deg"],
        config["central_band"]["fov_deg"],
    )

    left_h = build_view_homography(left_record["view"], target_view)
    right_h = build_view_homography(right_record["view"], target_view)
    left_warped, left_known_mask, _ = warp_image_and_mask(
        left_record["image"],
        left_h,
        (target_view["width"], target_view["height"]),
    )
    right_warped, right_known_mask, _ = warp_image_and_mask(
        right_record["image"],
        right_h,
        (target_view["width"], target_view["height"]),
    )
    merge_composite, missing_mask = build_merge_composite(
        left_warped,
        right_warped,
        left_known_mask,
        right_known_mask,
    )
    guided_input, inpaint_output = run_guided_inpaint(
        config["prompt"],
        merge_composite,
        missing_mask,
        None,
        config,
        generator=generator,
        strength=config["central_band"]["sdedit_t0"],
    )
    return {
        "name": "x_merge",
        "kind": "merge",
        "view": target_view,
        "image": inpaint_output,
        "source_name": "x3+x-3",
        "guidance_name": "",
        "guidance_image": merge_composite,
        "guided_input": guided_input,
        "left_warped": left_warped,
        "right_warped": right_warped,
        "left_known_mask": left_known_mask,
        "right_known_mask": right_known_mask,
        "merge_composite": merge_composite,
        "missing_mask": missing_mask,
    }


def build_steps_manifest(view_records):
    manifest = []
    for record in view_records:
        if record["kind"] == "initial":
            continue
        manifest.append(
            {
                "name": record["name"],
                "kind": record["kind"],
                "yaw_deg": record["view"]["yaw_deg"],
                "pitch_deg": record["view"]["pitch_deg"],
                "source_name": record.get("source_name", ""),
                "guidance_name": record.get("guidance_name", ""),
            }
        )
    return manifest


def crop_band(canvas, pitch_min_deg, pitch_max_deg):
    pano_height = canvas.shape[0]
    start = int(round((0.5 - pitch_max_deg / 180.0) * (pano_height - 1)))
    end = int(round((0.5 - pitch_min_deg / 180.0) * (pano_height - 1)))
    start = max(start, 0)
    end = min(end + 1, pano_height)
    return canvas[start:end].copy()


def run_phase2(config_path):
    config = load_phase2_config(config_path)
    generator = create_generator(config.get("seed"))
    initial_view, source_image_used = generate_or_load_initial_view(config, generator)

    view_records = [create_initial_record(config, initial_view)]
    for step_spec in build_schedule(config):
        view_records.append(run_step(config, step_spec, view_records, generator))
    view_records.append(run_merge_step(config, view_records, generator))

    stitched_panorama, stitched_coverage = stitch_equirectangular_views(
        view_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        config["central_band"]["stitch_pitch_min_deg"],
        config["central_band"]["stitch_pitch_max_deg"],
    )
    band_crop = crop_band(
        stitched_panorama,
        config["central_band"]["stitch_pitch_min_deg"],
        config["central_band"]["stitch_pitch_max_deg"],
    )

    run_dir = resolve_run_dir(config["output"]["run_dir"])
    save_phase2_debug_artifacts(
        run_dir,
        {
            "config": config,
            "prompt": config["prompt"],
            "initial_view": initial_view,
            "steps_manifest": build_steps_manifest(view_records),
            "view_records": view_records[1:],
            "stitched_panorama": stitched_panorama,
            "stitched_coverage": stitched_coverage,
            "band_crop": band_crop,
            "pano_width": config["output"]["pano_width"],
            "pano_height": config["output"]["pano_height"],
        },
    )
    return {
        "run_dir": run_dir,
        "step_names": [record["name"] for record in view_records[1:]],
        "stitched_panorama_path": run_dir + "/04_central_360_equirect.png",
        "source_image_used": source_image_used,
    }
