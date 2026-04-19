from copy import deepcopy

import numpy as np

from .camera import build_view_rotation
from .common import resolve_run_dir
from .debug import save_phase5_debug_artifacts
from .phase1 import build_inpaint_input
from .phase2 import build_steps_manifest, build_view, create_initial_record, generate_or_load_initial_view
from .phase3 import compute_initial_risks, run_risky_merge_step, run_risky_step
from .phase4 import build_phase4_masks, build_prior_crop, build_target_from_overlaps, create_expansion_record, load_phase4_config, run_expansion_ring
from .pipeline import create_generator, run_guided_inpaint
from .risk import compute_view_risk_maps
from .stitch import stitch_equirectangular_views


def apply_section_defaults(config, section_name, defaults):
    if section_name not in config:
        config[section_name] = deepcopy(defaults)
        return
    for key, value in defaults.items():
        if key not in config[section_name]:
            config[section_name][key] = deepcopy(value)


def load_phase5_config(config_path):
    config = load_phase4_config(config_path)
    apply_section_defaults(
        config,
        "pole_closure",
        {
            "fov_deg": 90.0,
            "sdedit_t0": 0.90,
            "guidance_scale": 1.0,
            "noise_variance_multiplier": 1.10,
            "risk_weights": [0.6, 0.2, 0.1, 0.1],
            "erase_ratio": 0.20,
            "prior_crop_ratio": 0.3333333333,
            "risk_gaussian_kernel": 9,
            "risk_gaussian_sigma": 2.0,
            "mask_median_kernel": 5,
            "mask_dilate_kernel": 3,
            "mask_dilate_iterations": 1,
            "risk_fallback_threshold": 0.5,
        },
    )
    if config["output"].get("run_dir") == "outputs/phase4/example_run":
        config["output"]["run_dir"] = "outputs/phase5/example_run"
    validate_phase5_config(config)
    return config


def validate_phase5_config(config):
    if len(config["pole_closure"]["risk_weights"]) != 4:
        raise RuntimeError("Phase 5 requires four pole-closure risk weights.")


def build_pole_view(config, pitch_deg):
    return build_view(
        config["source_view"],
        0.0,
        pitch_deg,
        config["pole_closure"]["fov_deg"],
    )


def build_view_direction(view):
    rotation = build_view_rotation(view)
    center_ray = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    direction = rotation @ center_ray
    norm = np.linalg.norm(direction)
    if norm <= 1e-8:
        return center_ray
    return direction / norm


def wrap_yaw_delta(yaw_deg, target_yaw_deg):
    delta = float(yaw_deg) - float(target_yaw_deg)
    while delta <= -180.0:
        delta += 360.0
    while delta > 180.0:
        delta -= 360.0
    return delta


def find_nearest_pole_record(records, pole_yaw_deg, pole_pitch_deg):
    if pole_pitch_deg > 0.0:
        pole_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        pole_direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    best_record = None
    best_alignment = -1e9
    best_yaw_distance = 1e9
    for record in records:
        direction = build_view_direction(record["view"])
        alignment = float(direction @ pole_direction)
        yaw_distance = abs(wrap_yaw_delta(record["view"]["yaw_deg"], pole_yaw_deg))
        if best_record is None:
            best_record = record
            best_alignment = alignment
            best_yaw_distance = yaw_distance
            continue
        if alignment > best_alignment + 1e-8:
            best_record = record
            best_alignment = alignment
            best_yaw_distance = yaw_distance
            continue
        if abs(alignment - best_alignment) <= 1e-8 and yaw_distance < best_yaw_distance:
            best_record = record
            best_alignment = alignment
            best_yaw_distance = yaw_distance
    if best_record is None:
        raise RuntimeError("Missing pole guidance record.")
    return best_record


def run_pole_target(config, name, pitch_deg, completed_records, risk_context_records, guidance_records, generator, initial_center, use_top):
    target_view = build_pole_view(config, pitch_deg)
    composite_bundle = build_target_from_overlaps(target_view, completed_records)
    guidance_record = find_nearest_pole_record(guidance_records, 0.0, pitch_deg)
    guidance_image = build_prior_crop(
        guidance_record["image"],
        target_view["width"],
        target_view["height"],
        use_top,
        config["pole_closure"]["prior_crop_ratio"],
    )
    selected_mask, remasked_mask, smoothed_mask = build_phase4_masks(
        composite_bundle["base_missing_mask"],
        composite_bundle["known_mask"],
        composite_bundle["warped_combined_risk"],
        config["pole_closure"],
    )
    inpaint_input = build_inpaint_input(composite_bundle["composite"], smoothed_mask)
    guided_input, output_image = run_guided_inpaint(
        config["prompt"],
        inpaint_input,
        smoothed_mask,
        guidance_image,
        config,
        generator=generator,
        strength=config["pole_closure"]["sdedit_t0"],
        guidance_scale=config["pole_closure"]["guidance_scale"],
        noise_variance_multiplier=config["pole_closure"]["noise_variance_multiplier"],
    )
    record = create_expansion_record(
        name,
        "pole",
        target_view,
        output_image,
        guidance_record["name"],
        guidance_image,
        guided_input,
        composite_bundle,
        selected_mask,
        remasked_mask,
        smoothed_mask,
    )
    record["risk_maps"] = compute_view_risk_maps(
        record,
        risk_context_records + [record],
        config,
        initial_center,
        config["pole_closure"],
    )
    return record


def build_phase5_note(expected_result_reached, uncovered_pixels):
    if expected_result_reached:
        return "Phase 5 reached its expected result: final full spherical panorama was produced end-to-end with top and bottom poles closed."
    return "Phase 5 did not reach its expected result: final full-sphere coverage still has {} uncovered pixels.".format(
        int(uncovered_pixels)
    )


def run_phase5(config_path):
    config = load_phase5_config(config_path)
    generator = create_generator(config.get("seed"))
    initial_view, source_image_used = generate_or_load_initial_view(config, generator)

    initial_record = create_initial_record(config, initial_view)
    initial_center = compute_initial_risks(config, initial_record)
    central_records = [initial_record]
    for step_spec in [
        {"name": "x1", "yaw_deg": config["central_band"]["yaw_stride_deg"], "direction": 1, "step_index": 1},
        {"name": "x-1", "yaw_deg": -config["central_band"]["yaw_stride_deg"], "direction": -1, "step_index": 1},
        {"name": "x2", "yaw_deg": config["central_band"]["yaw_stride_deg"] * 2.0, "direction": 1, "step_index": 2},
        {"name": "x-2", "yaw_deg": -config["central_band"]["yaw_stride_deg"] * 2.0, "direction": -1, "step_index": 2},
        {"name": "x3", "yaw_deg": config["central_band"]["yaw_stride_deg"] * 3.0, "direction": 1, "step_index": 3},
        {"name": "x-3", "yaw_deg": -config["central_band"]["yaw_stride_deg"] * 3.0, "direction": -1, "step_index": 3},
    ]:
        central_records.append(run_risky_step(config, step_spec, central_records, generator, initial_center))
    central_records.append(run_risky_merge_step(config, central_records, generator, initial_center))

    central_panorama, _ = stitch_equirectangular_views(
        central_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        config["central_band"]["stitch_pitch_min_deg"],
        config["central_band"]["stitch_pitch_max_deg"],
    )
    upward_records = run_expansion_ring(
        config,
        "u",
        config["expansion"]["pitch_offset_deg"],
        central_records,
        [],
        [],
        generator,
        initial_center,
        True,
    )
    downward_records = run_expansion_ring(
        config,
        "d",
        -config["expansion"]["pitch_offset_deg"],
        central_records,
        upward_records,
        [],
        generator,
        initial_center,
        False,
    )

    upward_panorama, _ = stitch_equirectangular_views(
        upward_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )
    downward_panorama, _ = stitch_equirectangular_views(
        downward_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )
    pre_pole_panorama, _ = stitch_equirectangular_views(
        central_records + upward_records + downward_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )

    pre_pole_records = central_records + upward_records + downward_records
    top_pole_record = run_pole_target(
        config,
        "top_pole",
        90.0,
        pre_pole_records,
        upward_records,
        upward_records,
        generator,
        initial_center,
        True,
    )
    bottom_pole_record = run_pole_target(
        config,
        "bottom_pole",
        -90.0,
        pre_pole_records,
        downward_records,
        downward_records,
        generator,
        initial_center,
        False,
    )
    pole_records = [top_pole_record, bottom_pole_record]

    top_pole_panorama, _ = stitch_equirectangular_views(
        [top_pole_record],
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )
    bottom_pole_panorama, _ = stitch_equirectangular_views(
        [bottom_pole_record],
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )
    full_panorama, full_coverage = stitch_equirectangular_views(
        pre_pole_records + pole_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )

    uncovered_pixels = int((full_coverage <= 0).sum())
    expected_result_reached = uncovered_pixels == 0
    phase_note = build_phase5_note(expected_result_reached, uncovered_pixels)
    run_dir = resolve_run_dir(config["output"]["run_dir"])
    save_phase5_debug_artifacts(
        run_dir,
        {
            "config": config,
            "prompt": config["prompt"],
            "central_manifest": build_steps_manifest(central_records),
            "upward_manifest": build_steps_manifest(upward_records),
            "downward_manifest": build_steps_manifest(downward_records),
            "pole_manifest": build_steps_manifest(pole_records),
            "central_records": central_records,
            "upward_records": upward_records,
            "downward_records": downward_records,
            "pole_records": pole_records,
            "central_panorama": central_panorama,
            "upward_panorama": upward_panorama,
            "downward_panorama": downward_panorama,
            "pre_pole_panorama": pre_pole_panorama,
            "top_pole_panorama": top_pole_panorama,
            "bottom_pole_panorama": bottom_pole_panorama,
            "full_panorama": full_panorama,
            "full_coverage": full_coverage,
            "phase_note": phase_note,
            "pano_width": config["output"]["pano_width"],
            "pano_height": config["output"]["pano_height"],
        },
    )
    return {
        "run_dir": run_dir,
        "pole_step_names": [record["name"] for record in pole_records],
        "stitched_panorama_path": run_dir + "/12_full_sphere_equirect.png",
        "source_image_used": source_image_used,
        "expected_result_reached": expected_result_reached,
    }
