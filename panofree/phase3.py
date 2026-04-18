from copy import deepcopy

import numpy as np

from .camera import view_center_on_equirectangular
from .common import ensure_prompt_and_models, load_config, load_source_image, resolve_run_dir
from .debug import save_phase3_debug_artifacts
from .phase1 import build_inpaint_input
from .phase2 import build_merge_composite, build_schedule, build_steps_manifest, build_view, crop_band, create_initial_record, find_record, generate_or_load_initial_view, get_guidance_name, get_source_name
from .pipeline import create_generator, run_guided_inpaint
from .risk import compute_view_risk_maps, merge_warped_risks, select_risky_known_pixels, smooth_remask, warp_risk_map
from .stitch import stitch_equirectangular_views
from .warp import build_view_homography, warp_image_and_mask


def load_phase3_config(config_path):
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
            "risk_weights": [0.8, 0.2, 0.0, 0.0],
            "erase_ratio": 0.05,
            "risk_gaussian_kernel": 9,
            "risk_gaussian_sigma": 2.0,
            "mask_median_kernel": 5,
            "mask_dilate_kernel": 3,
            "mask_dilate_iterations": 1,
            "risk_fallback_threshold": 0.5,
        },
        "output": {
            "run_dir": "outputs/phase3/example_run",
            "pano_width": 4096,
            "pano_height": 2048,
        },
    }
    config = load_config(config_path, defaults)
    validate_phase3_config(config)
    return config


def validate_phase3_config(config):
    ensure_prompt_and_models(config, "Phase 3")
    if config["central_band"]["steps_each_direction"] != 3:
        raise RuntimeError("Phase 3 requires `central_band.steps_each_direction` to be 3.")
    if len(config["central_band"]["risk_weights"]) != 4:
        raise RuntimeError("Phase 3 requires four central-band risk weights.")


def compute_initial_risks(config, initial_record):
    initial_center = view_center_on_equirectangular(
        config["source_view"],
        config["output"]["pano_width"],
        config["output"]["pano_height"],
    )
    initial_record["risk_maps"] = compute_view_risk_maps(
        initial_record,
        [initial_record],
        config,
        initial_center,
    )
    initial_record["guidance_image"] = initial_record["image"]
    initial_record["guided_input"] = initial_record["image"]
    return initial_center


def build_phase3_masks(base_missing_mask, known_mask, warped_combined_risk, config):
    selected_mask = select_risky_known_pixels(
        warped_combined_risk,
        known_mask,
        config["central_band"]["erase_ratio"],
        config["central_band"]["risk_fallback_threshold"],
    )
    remasked = np.where((base_missing_mask > 0) | (selected_mask > 0), 255, 0).astype(np.uint8)
    smoothed = smooth_remask(
        remasked,
        config["central_band"]["risk_gaussian_kernel"],
        config["central_band"]["risk_gaussian_sigma"],
        config["central_band"]["mask_median_kernel"],
        config["central_band"]["mask_dilate_kernel"],
        config["central_band"]["mask_dilate_iterations"],
    )
    return selected_mask, remasked, smoothed


def run_risky_step(config, step_spec, view_records, generator, initial_center):
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
    warped, known_mask, base_missing_mask = warp_image_and_mask(
        source_record["image"],
        homography,
        (target_view["width"], target_view["height"]),
    )
    warped_combined_risk = warp_risk_map(
        source_record["risk_maps"]["combined"],
        homography,
        (target_view["width"], target_view["height"]),
    )
    selected_mask, remasked_mask, smoothed_mask = build_phase3_masks(
        base_missing_mask,
        known_mask,
        warped_combined_risk,
        config,
    )
    inpaint_input = build_inpaint_input(warped, smoothed_mask)
    guided_input, inpaint_output = run_guided_inpaint(
        config["prompt"],
        inpaint_input,
        smoothed_mask,
        guidance_record["image"],
        config,
        generator=generator,
        strength=config["central_band"]["sdedit_t0"],
    )
    record = {
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
        "base_missing_mask": base_missing_mask,
        "risk_selected_mask": selected_mask,
        "remasked_mask": remasked_mask,
        "smoothed_mask": smoothed_mask,
        "missing_mask": smoothed_mask,
        "warped_combined_risk": warped_combined_risk,
    }
    record["risk_maps"] = compute_view_risk_maps(
        record,
        view_records + [record],
        config,
        initial_center,
    )
    return record


def run_risky_merge_step(config, view_records, generator, initial_center):
    left_record = find_record(view_records, "x3")
    right_record = find_record(view_records, "x-3")
    guidance_record = find_record(view_records, "x0")
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
    merge_composite, base_missing_mask = build_merge_composite(
        left_warped,
        right_warped,
        left_known_mask,
        right_known_mask,
    )
    left_warped_risk = warp_risk_map(
        left_record["risk_maps"]["combined"],
        left_h,
        (target_view["width"], target_view["height"]),
    )
    right_warped_risk = warp_risk_map(
        right_record["risk_maps"]["combined"],
        right_h,
        (target_view["width"], target_view["height"]),
    )
    warped_combined_risk = merge_warped_risks(
        left_warped_risk,
        right_warped_risk,
        left_known_mask,
        right_known_mask,
    )
    known_mask = np.where((left_known_mask > 0) | (right_known_mask > 0), 255, 0).astype(np.uint8)
    selected_mask, remasked_mask, smoothed_mask = build_phase3_masks(
        base_missing_mask,
        known_mask,
        warped_combined_risk,
        config,
    )
    inpaint_input = build_inpaint_input(merge_composite, smoothed_mask)
    guided_input, inpaint_output = run_guided_inpaint(
        config["prompt"],
        inpaint_input,
        smoothed_mask,
        guidance_record["image"],
        config,
        generator=generator,
        strength=config["central_band"]["sdedit_t0"],
    )
    record = {
        "name": "x_merge",
        "kind": "merge",
        "view": target_view,
        "image": inpaint_output,
        "source_name": "x3+x-3",
        "guidance_name": "x0",
        "guidance_image": guidance_record["image"],
        "guided_input": guided_input,
        "left_warped": left_warped,
        "right_warped": right_warped,
        "left_known_mask": left_known_mask,
        "right_known_mask": right_known_mask,
        "left_warped_risk": left_warped_risk,
        "right_warped_risk": right_warped_risk,
        "warped_combined_risk": warped_combined_risk,
        "merge_composite": merge_composite,
        "base_missing_mask": base_missing_mask,
        "risk_selected_mask": selected_mask,
        "remasked_mask": remasked_mask,
        "smoothed_mask": smoothed_mask,
        "missing_mask": smoothed_mask,
    }
    record["risk_maps"] = compute_view_risk_maps(
        record,
        view_records + [record],
        config,
        initial_center,
    )
    return record


def run_phase3(config_path):
    config = load_phase3_config(config_path)
    generator = create_generator(config.get("seed"))
    initial_view, source_image_used = generate_or_load_initial_view(config, generator)

    initial_record = create_initial_record(config, initial_view)
    initial_center = compute_initial_risks(config, initial_record)
    view_records = [initial_record]

    for step_spec in build_schedule(config):
        view_records.append(run_risky_step(config, step_spec, view_records, generator, initial_center))
    view_records.append(run_risky_merge_step(config, view_records, generator, initial_center))

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
    save_phase3_debug_artifacts(
        run_dir,
        {
            "config": config,
            "prompt": config["prompt"],
            "initial_view": initial_view,
            "steps_manifest": build_steps_manifest(view_records),
            "view_records": view_records,
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
