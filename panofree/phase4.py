import numpy as np
from PIL import Image

from .common import ensure_prompt_and_models, load_config, resolve_run_dir
from .debug import save_phase4_debug_artifacts
from .phase1 import build_inpaint_input
from .phase2 import build_steps_manifest, build_view, create_initial_record, find_record, generate_or_load_initial_view
from .phase3 import compute_initial_risks, run_risky_merge_step, run_risky_step
from .pipeline import create_generator, run_guided_inpaint
from .risk import compute_view_risk_maps, select_risky_known_pixels, smooth_remask
from .stitch import build_boundary_weight_map, stitch_equirectangular_views
from .warp import build_view_to_view_remap, remap_with_visibility, require_cv2


def load_phase4_config(config_path):
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
        "expansion": {
            "pitch_offset_deg": 25.0,
            "fov_deg": 110.0,
            "yaw_stride_deg": 80.0,
            "steps_per_direction": 3,
            "sdedit_t0": 0.90,
            "guidance_scale": 2.0,
            "noise_variance_multiplier": 1.05,
            "risk_weights": [0.6, 0.2, 0.1, 0.1],
            "erase_ratio": 0.10,
            "prior_crop_ratio": 0.3333333333,
            "risk_gaussian_kernel": 9,
            "risk_gaussian_sigma": 2.0,
            "mask_median_kernel": 5,
            "mask_dilate_kernel": 3,
            "mask_dilate_iterations": 1,
            "risk_fallback_threshold": 0.5,
        },
        "output": {
            "run_dir": "outputs/phase4/example_run",
            "pano_width": 4096,
            "pano_height": 2048,
        },
    }
    config = load_config(config_path, defaults)
    validate_phase4_config(config)
    return config


def validate_phase4_config(config):
    ensure_prompt_and_models(config, "Phase 4")
    if config["central_band"]["steps_each_direction"] != 3:
        raise RuntimeError("Phase 4 requires `central_band.steps_each_direction` to be 3.")
    if config["expansion"]["steps_per_direction"] != 3:
        raise RuntimeError("Phase 4 requires `expansion.steps_per_direction` to be 3.")
    if len(config["expansion"]["risk_weights"]) != 4:
        raise RuntimeError("Phase 4 requires four expansion risk weights.")


def build_prior_crop(image, width, height, use_top, crop_ratio):
    image_height = image.shape[0]
    crop_height = max(int(round(image_height * float(crop_ratio))), 1)
    if use_top:
        cropped = image[:crop_height, :, :]
    else:
        cropped = image[image_height - crop_height:, :, :]
    resized = Image.fromarray(cropped.astype(np.uint8), mode="RGB").resize(
        (width, height),
        resample=Image.Resampling.LANCZOS,
    )
    return np.array(resized, dtype=np.uint8)


def build_phase4_masks(base_missing_mask, known_mask, warped_combined_risk, risk_config):
    selected_mask = select_risky_known_pixels(
        warped_combined_risk,
        known_mask,
        risk_config["erase_ratio"],
        risk_config["risk_fallback_threshold"],
    )
    remasked = np.where((base_missing_mask > 0) | (selected_mask > 0), 255, 0).astype(np.uint8)
    smoothed = smooth_remask(
        remasked,
        risk_config["risk_gaussian_kernel"],
        risk_config["risk_gaussian_sigma"],
        risk_config["mask_median_kernel"],
        risk_config["mask_dilate_kernel"],
        risk_config["mask_dilate_iterations"],
    )
    return selected_mask, remasked, smoothed


def build_expansion_view(config, yaw_deg, pitch_deg):
    return build_view(
        config["source_view"],
        yaw_deg,
        pitch_deg,
        config["expansion"]["fov_deg"],
    )


def get_expansion_guidance_name(name):
    mapping = {
        "u0": "x0",
        "u1": "x0",
        "u-1": "u1",
        "u_merge": "x0",
        "d0": "x0",
        "d1": "x0",
        "d-1": "d1",
        "d_merge": "x0",
    }
    return mapping[name]


def get_expansion_reference_pool(name, central_records, upward_records, downward_records):
    if name.startswith("u"):
        return central_records + upward_records + downward_records
    return central_records + upward_records + downward_records


def build_overlap_source(record, target_view):
    cv2 = require_cv2()
    map_x, map_y, valid = build_view_to_view_remap(record["view"], target_view)
    if int(valid.sum()) <= 0:
        return None

    warped = remap_with_visibility(
        record["image"],
        map_x,
        map_y,
        valid,
        cv2.INTER_LINEAR,
    ).astype(np.uint8)
    candidate_known_mask = np.where(valid, 255, 0).astype(np.uint8)
    source_weight = build_boundary_weight_map(
        record["view"]["width"],
        record["view"]["height"],
    ).astype(np.float32)
    candidate_weight = remap_with_visibility(
        source_weight,
        map_x,
        map_y,
        valid,
        cv2.INTER_LINEAR,
    ).astype(np.float32)
    candidate_weight = np.where(valid, np.maximum(candidate_weight, 1e-6), 0.0).astype(np.float32)
    candidate_risk = remap_with_visibility(
        record["risk_maps"]["combined"].astype(np.float32),
        map_x,
        map_y,
        valid,
        cv2.INTER_LINEAR,
    ).astype(np.float32)
    return {
        "name": record["name"],
        "warped": warped,
        "known_mask": candidate_known_mask,
        "weight": candidate_weight,
        "risk": candidate_risk,
    }


def build_target_from_overlaps(target_view, completed_records):
    composite_sum = np.zeros((target_view["height"], target_view["width"], 3), dtype=np.float32)
    weight_sum = np.zeros((target_view["height"], target_view["width"]), dtype=np.float32)
    known_mask = np.zeros((target_view["height"], target_view["width"]), dtype=np.uint8)
    warped_combined_risk = np.zeros((target_view["height"], target_view["width"]), dtype=np.float32)
    overlap_sources = []

    for record in completed_records:
        overlap_source = build_overlap_source(record, target_view)
        if overlap_source is None:
            continue

        valid = overlap_source["known_mask"] > 0
        composite_sum += overlap_source["warped"].astype(np.float32) * overlap_source["weight"][..., None]
        weight_sum += overlap_source["weight"]
        known_mask = np.where(valid, 255, known_mask).astype(np.uint8)
        warped_combined_risk = np.where(
            valid,
            np.maximum(warped_combined_risk, overlap_source["risk"]),
            warped_combined_risk,
        ).astype(np.float32)
        overlap_sources.append(
            {
                "name": overlap_source["name"],
                "warped": overlap_source["warped"],
                "known_mask": overlap_source["known_mask"],
                "weight": overlap_source["weight"],
            }
        )

    composite = np.zeros((target_view["height"], target_view["width"], 3), dtype=np.uint8)
    nonzero = weight_sum > 0.0
    composite[nonzero] = np.clip(
        composite_sum[nonzero] / weight_sum[nonzero, None],
        0.0,
        255.0,
    ).astype(np.uint8)
    base_missing_mask = np.where(known_mask > 0, 0, 255).astype(np.uint8)
    return {
        "composite": composite,
        "known_mask": known_mask,
        "base_missing_mask": base_missing_mask,
        "warped_combined_risk": warped_combined_risk,
        "overlap_sources": overlap_sources,
    }


def create_expansion_record(name, kind, target_view, output_image, guidance_name, guidance_image, guided_input, composite_bundle, selected_mask, remasked_mask, smoothed_mask):
    return {
        "name": name,
        "kind": kind,
        "view": target_view,
        "image": output_image,
        "source_name": ",".join([item["name"] for item in composite_bundle["overlap_sources"]]),
        "guidance_name": guidance_name,
        "guidance_image": guidance_image,
        "guided_input": guided_input,
        "warped": composite_bundle["composite"],
        "known_mask": composite_bundle["known_mask"],
        "base_missing_mask": composite_bundle["base_missing_mask"],
        "risk_selected_mask": selected_mask,
        "remasked_mask": remasked_mask,
        "smoothed_mask": smoothed_mask,
        "missing_mask": smoothed_mask,
        "warped_combined_risk": composite_bundle["warped_combined_risk"],
        "overlap_source_names": [item["name"] for item in composite_bundle["overlap_sources"]],
        "overlap_sources": composite_bundle["overlap_sources"],
    }


def run_expansion_target(config, name, kind, yaw_deg, pitch_deg, completed_records, risk_context_records, generator, initial_center, use_top):
    target_view = build_expansion_view(config, yaw_deg, pitch_deg)
    composite_bundle = build_target_from_overlaps(target_view, completed_records)
    guidance_name = get_expansion_guidance_name(name)
    guidance_record = find_record(completed_records, guidance_name)
    guidance_image = build_prior_crop(
        guidance_record["image"],
        target_view["width"],
        target_view["height"],
        use_top,
        config["expansion"]["prior_crop_ratio"],
    )
    selected_mask, remasked_mask, smoothed_mask = build_phase4_masks(
        composite_bundle["base_missing_mask"],
        composite_bundle["known_mask"],
        composite_bundle["warped_combined_risk"],
        config["expansion"],
    )
    inpaint_input = build_inpaint_input(composite_bundle["composite"], smoothed_mask)
    guided_input, output_image = run_guided_inpaint(
        config["prompt"],
        inpaint_input,
        smoothed_mask,
        guidance_image,
        config,
        generator=generator,
        strength=config["expansion"]["sdedit_t0"],
        guidance_scale=config["expansion"]["guidance_scale"],
        noise_variance_multiplier=config["expansion"]["noise_variance_multiplier"],
    )
    record = create_expansion_record(
        name,
        kind,
        target_view,
        output_image,
        guidance_name,
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
        config["expansion"],
    )
    return record


def run_expansion_ring(config, prefix, pitch_deg, central_records, upward_records, downward_records, generator, initial_center, use_top):
    if prefix == "u":
        ring_records = upward_records
    else:
        ring_records = downward_records

    schedule = [
        {"name": prefix + "0", "kind": "seed", "yaw_deg": 0.0},
        {"name": prefix + "1", "kind": "step", "yaw_deg": config["expansion"]["yaw_stride_deg"]},
        {"name": prefix + "-1", "kind": "step", "yaw_deg": -config["expansion"]["yaw_stride_deg"]},
        {"name": prefix + "_merge", "kind": "merge", "yaw_deg": 180.0},
    ]

    for step_spec in schedule:
        completed_records = get_expansion_reference_pool(
            step_spec["name"],
            central_records,
            upward_records,
            downward_records,
        )
        record = run_expansion_target(
            config,
            step_spec["name"],
            step_spec["kind"],
            step_spec["yaw_deg"],
            pitch_deg,
            completed_records,
            ring_records,
            generator,
            initial_center,
            use_top,
        )
        ring_records.append(record)
    return ring_records


def run_phase4(config_path):
    config = load_phase4_config(config_path)
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
    full_panorama, full_coverage = stitch_equirectangular_views(
        central_records + upward_records + downward_records,
        config["output"]["pano_width"],
        config["output"]["pano_height"],
        None,
        None,
    )

    run_dir = resolve_run_dir(config["output"]["run_dir"])
    save_phase4_debug_artifacts(
        run_dir,
        {
            "config": config,
            "prompt": config["prompt"],
            "central_manifest": build_steps_manifest(central_records),
            "upward_manifest": build_steps_manifest(upward_records),
            "downward_manifest": build_steps_manifest(downward_records),
            "central_records": central_records,
            "upward_records": upward_records,
            "downward_records": downward_records,
            "central_panorama": central_panorama,
            "upward_panorama": upward_panorama,
            "downward_panorama": downward_panorama,
            "full_panorama": full_panorama,
            "full_coverage": full_coverage,
            "pano_width": config["output"]["pano_width"],
            "pano_height": config["output"]["pano_height"],
        },
    )
    return {
        "run_dir": run_dir,
        "upward_step_names": [record["name"] for record in upward_records],
        "downward_step_names": [record["name"] for record in downward_records],
        "stitched_panorama_path": run_dir + "/08_full_sphere_without_poles_equirect.png",
        "source_image_used": source_image_used,
    }
